"""M2：A/B/C 共同参考、简单基线、多种子训练与历史复评。"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from svh.mapping_contract import mapping_contract_payload
from utils.config import load_config

from .baselines import predict_hold_last, predict_linear
from .experiment_runner import _unique_run_dir
from .keypoint_data import (build_keypoint_windows, create_keypoint_smoke_dataset, fit_normalization,
                            task_contract, validate_manifest, validate_task)
from .keypoint_metrics import compute_keypoint_metrics
from .representation_data import build_representation_batch
from .representation_metrics import continuation_summary, control_metrics
from .training import predict_neural_checkpoint, train_neural_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def baseline_predictions(batch, name: str, fit_frames: int = 2):
    pose = name.startswith("pose_")
    x = batch.points.x if pose else batch.history9
    horizons = batch.points.horizon_ms if pose else batch.query_ms
    steps = np.asarray(horizons) / batch.points.horizon_ms[0]
    if "linear" in name:
        pred = predict_linear(x, steps, fit_frames=fit_frames, feature_count=x.shape[-1],
                              output_bounds=None if pose else (0.0, 1.0))
    else:
        pred = predict_hold_last(x, steps, feature_count=x.shape[-1])
    if pose:
        controls, valid = batch.map_predictions(pred)
        return controls, valid, pred
    return pred, np.ones(pred.shape[:2], dtype=bool), None


def evaluate(batch, controls, valid, poses=None, *, motion_thresholds=None):
    strata = None
    if motion_thresholds is not None:
        low, high = motion_thresholds
        # 运动分层只由历史定义；手势转换是参考规则的分析标签，不输入预测器。
        complete = batch.target_valid[:, np.flatnonzero(np.isclose(batch.query_ms, 150))[0]]
        strata = {"low_motion": batch.motion_score <= low, "high_motion": batch.motion_score >= high,
                  "reference_gesture_transition": complete & batch.reference_transition,
                  "reference_gesture_no_transition": complete & ~batch.reference_transition}
    result = control_metrics(batch, controls, valid, strata=strata)
    if poses is not None:
        result["pose_metrics"] = compute_keypoint_metrics(batch.points, poses)
    return result


def write_reports(run_dir: Path, report: dict):
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = []
    for split in ("validation", "historical_reevaluation"):
        for method, result in report["baselines"].items():
            for horizon, metrics in result[split]["per_horizon_ms"].items():
                rows.append({"split": split, "method": method, "seed": "", "horizon_ms": horizon,
                             "sequence_equal_rmse": metrics["sequence_equal_rmse"],
                             "p95_frame_rmse": metrics["per_frame_rmse_p95"],
                             "reference_coverage": metrics["reference_coverage"],
                             "fallback_frames": metrics["fallback_frames_with_reference"]})
        for run in report["runs"]:
            for horizon, metrics in run[split]["per_horizon_ms"].items():
                rows.append({"split": split, "method": run["route"], "seed": run["seed"], "horizon_ms": horizon,
                             "sequence_equal_rmse": metrics["sequence_equal_rmse"],
                             "p95_frame_rmse": metrics["per_frame_rmse_p95"],
                             "reference_coverage": metrics["reference_coverage"],
                             "fallback_frames": metrics["fallback_frames_with_reference"]})
    with (run_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# M2 A/B/C 对照", "", report["evidence_scope"], "",
             f"验证集选择的简单基线：`{report['selection']['simple_baseline']}`。",
             f"验证集按种子均值选择的神经路线：`{report['selection']['neural_route']}`。", "",
             "| 方法 | 50ms RMSE | 100ms RMSE | 150ms RMSE |", "|---|---:|---:|---:|"]
    for method, result in report["baselines"].items():
        values = [result["historical_reevaluation"]["per_horizon_ms"][str(h)]["sequence_equal_rmse"] for h in (50, 100, 150)]
        lines.append(f"| {method} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    for route in ("A", "B", "C"):
        selected = [r for r in report["runs"] if r["route"] == route]
        values = [np.mean([r["historical_reevaluation"]["per_horizon_ms"][str(h)]["sequence_equal_rmse"] for r in selected]) for h in (50, 100, 150)]
        lines.append(f"| {route}（种子均值） | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", "RMSE 是 [0,1] 预览通道单位，按序列等权；不是物理关节误差。",
              "原始 take 配对、种子差异、预测失败回退和姿态误差见 JSON。",
              "subject4 为历史复评；原生 H2O 结果不证明摄像头泛化或设备效果。"]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_representation_experiment(*, config_path: Path, data_root: Path | None,
                                  output_root: Path, synthetic_smoke: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task = json.loads((config_path.parent / config["task_spec"]).read_text(encoding="utf-8"))
    validate_task(task)
    cfg = load_config(str(PROJECT_ROOT / config["mapping_config"]))
    if cfg.get("svh_preview_layout") != "svh_9ch" or cfg.get("svh_preview_channel_count") != 9:
        raise ValueError("M2 需要相同的 9 通道映射")
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("训练种子须非空且不重复")
    candidates = config["linear_fit_frames_candidates"]
    if not candidates or any(n < 2 or n > task["time"]["history_frames"] for n in candidates):
        raise ValueError("速度拟合长度不合法")
    run_dir = _unique_run_dir(output_root)
    if synthetic_smoke:
        data_root = run_dir / "synthetic_input"
        create_keypoint_smoke_dataset(data_root, task, articulated=True)
        config["seeds"] = seeds[:1]
        config["training"]["epochs"] = min(2, config["training"]["epochs"])
    if data_root is None:
        raise ValueError("真实 M2 实验需要 --data-root")
    manifest = validate_manifest(data_root, task)
    normalization = fit_normalization(data_root, manifest, task)
    batches = {}
    sample_seed = config["data_sample_seed"]

    def build(split, offset):
        points = build_keypoint_windows(data_root, manifest, task, normalization, split=split,
            seed=sample_seed + offset, **config["sampling"])
        batch = build_representation_batch(data_root, manifest, task, points, cfg)
        print(f"M2 {split}：{len(batch.points.x)} 个共同窗口，{len(np.unique(batch.points.sequence_ids))} 段", flush=True)
        return batch

    batches["train"], batches["val"] = build("train", 0), build("val", 1)
    motion_thresholds = np.quantile(batches["train"].motion_score, [0.25, 0.75]).tolist()

    def measure(batch, controls, valid, poses=None):
        return evaluate(batch, controls, valid, poses, motion_thresholds=motion_thresholds)

    val = batches["val"]
    baselines, tuning = {}, {}
    fits = {}
    for name in ("hold9", "linear9", "pose_hold", "pose_linear"):
        tested = candidates if "linear" in name else [2]
        metrics = {n: measure(val, *baseline_predictions(val, name, n)) for n in tested}
        best = min(tested, key=lambda n: metrics[n]["selection_score"])
        fits[name] = best
        tuning[name] = {str(n): m["selection_score"] for n, m in metrics.items()}
        baselines[name] = {"fit_frames": best if "linear" in name else None, "validation": metrics[best]}
    simple = min(baselines, key=lambda name: baselines[name]["validation"]["selection_score"])
    runs = []
    for seed in config["seeds"]:
        for route in ("A", "B", "C"):
            print(f"M2 开始 {route}，seed={seed}", flush=True)
            train_view = batches["train"].training_view(route)
            val_view = val.training_view(route)
            options = {"input_size": train_view.x.shape[-1], "output_size": train_view.y.shape[-1],
                       "output_bounds": None if route == "C" else (0.0, 1.0)}
            if route == "B":
                options["external_residual"] = True
            contract = {"task_id": train_view.task_id, "pose_task": task_contract(task, normalization["s_min_m"]),
                        "reference": config["reference"], "mapping": mapping_contract_payload(cfg),
                        "gesture_confirm_frames": cfg.get("stable_gesture_min_consecutive", 2),
                        "gesture_unknown_confirm_frames": cfg.get("stable_unknown_consecutive", 1),
                        "query_ms": list(val.query_ms), "input_size": options["input_size"], "output_size": options["output_size"]}
            def decode(batch, prediction):
                if route == "C":
                    controls, valid = batch.map_predictions(prediction)
                    return measure(batch, controls, valid, prediction)
                return measure(batch, prediction, np.ones(prediction.shape[:2], dtype=bool))
            checkpoint = run_dir / "checkpoints" / f"{route}_seed{seed}.pt"
            prediction, details = train_neural_model("residual_gru", train=train_view, val=val_view, test=val_view,
                architecture=config["architecture"], training_config={**config["training"], "run_label": f"M2_{route}_{seed}"},
                checkpoint_path=checkpoint, seed=seed, checkpoint_metadata=contract, model_options=options,
                validation_score=lambda prediction: decode(val, prediction)["selection_score"])
            reloaded, _ = predict_neural_checkpoint(checkpoint, split=val_view, device=details["device"],
                                                   batch_size=config["training"]["batch_size"])
            delta = float(np.max(np.abs(reloaded - prediction)))
            if delta > 1e-5:
                raise RuntimeError("M2 重载预测不一致")
            runs.append({"route": route, "seed": seed, "checkpoint": str(checkpoint),
                         "reload_max_abs_delta": delta, "training": details, "validation": decode(val, prediction)})
    # 所有路线、种子、速度窗口都在 validation 完成后，才构建历史复评。
    selected_route = min(("A", "B", "C"), key=lambda route: np.mean([r["validation"]["selection_score"] for r in runs if r["route"] == route]))
    batches["test"] = build("test", 2)
    test = batches["test"]
    for name in baselines:
        baselines[name]["historical_reevaluation"] = measure(test, *baseline_predictions(test, name, fits[name]))
    for run in runs:
        route = run["route"]
        pred, _ = predict_neural_checkpoint(Path(run["checkpoint"]), split=test.training_view(route),
                                            device=run["training"]["device"], batch_size=config["training"]["batch_size"])
        started = time.perf_counter()
        if route == "C":
            controls, valid = test.map_predictions(pred)
            run["historical_reevaluation"] = measure(test, controls, valid, pred)
        else:
            run["historical_reevaluation"] = measure(test, pred, np.ones(pred.shape[:2], dtype=bool))
        run["test_decode_and_metrics_seconds"] = time.perf_counter() - started
    continuation = {route: continuation_summary(baselines[simple]["historical_reevaluation"],
        [run["historical_reevaluation"] for run in runs if run["route"] == route], seed=sample_seed)
        for route in ("A", "B", "C")}
    report = {"schema_version": "representation-m2-report-v1", "status": "completed",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "data_root": str(data_root.resolve()),
              "synthetic": bool(manifest.get("synthetic")),
              "evidence_scope": "合成流程自检" if manifest.get("synthetic") else "H2O 共同代理参考的 A/B/C 对照，subject4 历史复评",
              "effective_config": config, "task": task, "normalization": normalization,
              "mapping": {"parameters": mapping_contract_payload(cfg), "reference": config["reference"],
                          "gesture_confirm_frames": cfg.get("stable_gesture_min_consecutive", 2),
                          "gesture_unknown_confirm_frames": cfg.get("stable_unknown_consecutive", 1)},
              "stratification": {"motion_score": "mean_last_8_frame_joint_displacement_in_anchor_palm_lengths_per_second",
                                 "training_quantiles": [0.25, 0.75], "motion_thresholds": motion_thresholds,
                                 "transition": "reference_stable_rule_changes_between_anchor_and_150ms_not_human_gesture_truth",
                                 "occlusion": "unavailable_no_per_joint_visibility_or_RGB_annotations"},
              "query_ms": list(val.query_ms), "sampling": {split: {"eligible_common_windows": len(batch.points.x),
                  "excluded_history_mapping": batch.excluded_history_mapping, "keypoint_sampling": batch.points.stats}
                  for split, batch in batches.items()},
              "selection": {"simple_baseline": simple, "neural_route": selected_route,
                            "linear_fit_validation_scores": tuning},
              "baselines": baselines, "runs": runs, "continuation": continuation,
              "notes": ["A/B 直接拟合七个共同查询目标，C 在五个规则姿态上插值后映射；不使用旧通道插值标签。",
                        "B 的 9 通道残差参考独立传入，编码器只读与 C 相同的 63 维姿态历史。",
                        "映射失败回退当前 9 通道并计数；评价分母不随预测失败缩小。",
                        "外部设备、摄像头泛化及计算就绪时间不属于本轮结论。"]}
    np.savez_compressed(run_dir / "common_target_sample.npz", history21=test.points.x[:1], history9=test.history9[:1],
                         targets9=test.targets9[:1], target_valid=test.target_valid[:1], query_ms=np.asarray(test.query_ms),
                         anchor_frame_id=test.points.anchor_frame_ids[:1], sequence_id=test.points.sequence_ids[:1])
    write_reports(run_dir, report)
    return run_dir / "report.json"
