"""M1 关键点实验：复用模型工厂、训练循环及运行目录管理。"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np

from .baselines import predict_hold_last, predict_linear
from .experiment_runner import _unique_run_dir
from .keypoint_data import (build_keypoint_windows, create_keypoint_smoke_dataset, denormalize,
                            fit_normalization, read_sequence, task_contract, validate_manifest, validate_task)
from .keypoint_metrics import compute_keypoint_metrics, query_trajectory
from .training import train_neural_model


def _classical(split, name: str, *, fit_frames: int = 2) -> np.ndarray:
    # 历史是规则 30Hz 网格，预测偏移与网格同单位。
    fps = 1000 / split.horizon_ms[0]
    steps = np.asarray(split.horizon_ms) * fps / 1000
    if name == "hold_last":
        return predict_hold_last(split.x, steps, feature_count=63)
    return predict_linear(split.x, steps, fit_frames=fit_frames, feature_count=63, output_bounds=None)


def _required_score(metrics: dict) -> float:
    value = metrics["selection_score"]
    if value is None or not np.isfinite(value):
        raise ValueError("验证集缺少 50/100ms 的有效目标，不能选型")
    return float(value)


def _write_reports(run_dir: Path, report: dict) -> None:
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = []
    lines = ["# M1 关键点实验", "", f"任务：`{report['task_id']}`；{report['evidence_scope']}。", "",
             "| 切分 | 方法 | 时距 ms | 归一化 MPJPE（序列等权） | MPJPE mm（序列等权） | 完整目标覆盖率 |",
             "|---|---|---:|---:|---:|---:|"]
    for name, result in report["results"].items():
        for split in ("validation", "historical_reevaluation"):
            for h, metrics in result[split]["per_horizon_ms"].items():
                row = {"split": split, "model": name, "horizon_ms": h,
                       "normalized_mpjpe": metrics["normalized_mpjpe"]["sequence_equal_mean"],
                       "mpjpe_mm": metrics["mpjpe_mm"]["sequence_equal_mean"],
                       "complete_target_fraction": metrics["coverage"]["complete_target_fraction"]}
                rows.append(row)
                def display(value):
                    return "无有效标签" if value is None else f"{value:.6f}"
                lines.append(f"| {split} | {name} | {h} | {display(row['normalized_mpjpe'])} | "
                             f"{display(row['mpjpe_mm'])} | {row['complete_target_fraction']:.2%} |")
    lines += ["", f"验证集选择：`{report['selection']['selected_label']}`；常速度窗口：{report['selection']['linear_fit_frames']} 帧。",
              "", "本轮只完成 M1 接线和小样本实测；单种子结果不替代 M2 正式收益对照。",
              "`model.json` 始终指向 residual GRU 实验候选，不自动启用到实时链路。",
              "网络耗时仅含设备驻留输入的 forward；不包含检测、归一化、传输或输出处理。"]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (run_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_keypoint_experiment(*, config_path: Path, data_root: Path | None, output_root: Path,
                            synthetic_smoke: bool = False) -> Path:
    from prediction.keypoint_model_loader import load_keypoint_prediction_model
    import torch

    config = json.loads(config_path.read_text(encoding="utf-8"))
    task = json.loads((config_path.parent / config["task_spec"]).read_text(encoding="utf-8"))
    validate_task(task)
    task["status"] = "M1_experiment"
    task["training_runner_support"] = "run_second_round_with_keypoint_run_v1"
    sampling = config["sampling"]
    if any(value is not None and (not isinstance(value, int) or value < 1) for value in sampling.values()):
        raise ValueError("抽样数量必须为正整数或 null（全部）")
    candidates = config["linear_fit_frames_candidates"]
    if not candidates or any(not isinstance(n, int) or n < 2 or n > task["time"]["history_frames"] for n in candidates):
        raise ValueError("常速度拟合长度须在 [2,history_frames] 内")
    run_dir = _unique_run_dir(output_root)
    if synthetic_smoke:
        data_root = run_dir / "synthetic_input"
        create_keypoint_smoke_dataset(data_root, task)
        config["training"]["epochs"] = min(2, config["training"]["epochs"])
        config["training"]["patience"] = min(2, config["training"]["patience"])
    if data_root is None:
        raise ValueError("真实实验需要 data_root")
    data_root = data_root.resolve()
    manifest = validate_manifest(data_root, task)
    normalization = fit_normalization(data_root, manifest, task)
    seed = int(config["seed"])
    splits = {}
    for index, split in enumerate(("train", "val")):
        splits[split] = build_keypoint_windows(data_root, manifest, task, normalization, split=split,
                                              seed=seed + index, **sampling)
        print(f"{split}：{len(splits[split].x)} 个关键点窗口", flush=True)
    train, val = splits["train"], splits["val"]
    hold_val = compute_keypoint_metrics(val, _classical(val, "hold_last"))
    linear_metrics = {n: compute_keypoint_metrics(val, _classical(val, "linear", fit_frames=n)) for n in candidates}
    linear_n = min(candidates, key=lambda n: _required_score(linear_metrics[n]))
    contract = task_contract(task, normalization["s_min_m"])
    prediction, details = train_neural_model("residual_gru", train=train, val=val, test=val,
        architecture=config["architecture"], training_config=config["training"],
        checkpoint_path=run_dir / "checkpoints/residual_gru.pt", seed=seed,
        checkpoint_metadata=contract, model_options={"input_size": 63, "output_size": 63, "output_bounds": None},
        validation_score=lambda pred: _required_score(compute_keypoint_metrics(val, pred)))
    neural_val = compute_keypoint_metrics(val, prediction)
    validation = {"hold_last": hold_val, "linear": linear_metrics[linear_n], "residual_gru": neural_val}
    selected = min(validation, key=lambda name: _required_score(validation[name]))
    model_spec = {"schema_version": "handai-keypoint-model-v1", "model_name": "residual_gru",
                  "checkpoint": "checkpoints/residual_gru.pt", "task": task,
                  "s_min_m": normalization["s_min_m"], "architecture": config["architecture"],
                  "role": "M1_candidate_not_enabled_in_live_pipeline"}
    model_path = run_dir / "model.json"
    model_path.write_text(json.dumps(model_spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    loaded = load_keypoint_prediction_model(model_path, expected_task_id=task["task_id"], device=details["device"])
    reload_val = np.concatenate([loaded.predict_normalized(val.x[i:i + 128]) for i in range(0, len(val.x), 128)])
    reload_delta = float(np.max(np.abs(prediction - reload_val)))
    if reload_delta > 1e-5:
        raise RuntimeError("模型重载后的归一化预测与训练评测不一致")
    # 历史复评在全部 validation 选型完成后才构建和推理。
    test = build_keypoint_windows(data_root, manifest, task, normalization, split="test", seed=seed + 2, **sampling)
    splits["test"] = test
    predictions = {"hold_last": _classical(test, "hold_last"), "linear": _classical(test, "linear", fit_frames=linear_n),
                   "residual_gru": np.concatenate([loaded.predict_normalized(test.x[i:i + 128]) for i in range(0, len(test.x), 128)])}
    results = {name: {"validation": validation[name], "historical_reevaluation": compute_keypoint_metrics(test, pred)}
               for name, pred in predictions.items()}
    entry = next(e for e in manifest["sequences"] if e["sequence_id"] == test.sequence_ids[0])
    seq = read_sequence(data_root, entry, task)
    anchor = int(np.flatnonzero(seq.frame_ids == test.anchor_frame_ids[0])[0])
    output = loaded.predict(seq.inputs[:anchor + 1], seq.timestamps[:anchor + 1], input_task=task,
                            frame_ids=seq.frame_ids[:anchor + 1], mask=seq.input_mask[:anchor + 1],
                            segment_ids=seq.segments[:anchor + 1])
    expected = denormalize(query_trajectory(predictions["residual_gru"][:1], test.horizon_ms, test.eval_horizon_ms),
                           test.wrists[:1], test.scales[:1])[0]
    api_delta = float(np.max(np.abs(expected - output["evaluation_landmarks_3d_m"])))
    if api_delta > 1e-5:
        raise RuntimeError("原始坐标加载接口与离线预测不一致")
    np.savez_compressed(run_dir / "window_sample.npz", x=test.x[:1], y=test.y[:1], target_mask=test.target_mask[:1],
        input_mask=test.input_mask[:1], eval_y=test.eval_y[:1], eval_mask=test.eval_mask[:1],
        history_times=test.history_times[:1], history_brackets=test.history_brackets[:1],
        target_times=test.target_times[:1], target_brackets=test.target_brackets[:1],
        eval_target_times=test.eval_target_times[:1], eval_brackets=test.eval_brackets[:1],
        anchor_times=test.anchor_times[:1], wrists=test.wrists[:1], scales=test.scales[:1],
        prediction=predictions["residual_gru"][:1])
    report = {"schema_version": "keypoint-m1-report-v1", "status": "completed", "task_id": task["task_id"],
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "data_root": str(data_root),
              "synthetic": bool(manifest.get("synthetic", False)),
              "evidence_scope": "非空合成数据链路自检" if manifest.get("synthetic") else "H2O 单种子小样本开发实验，subject4 为历史复评",
              "effective_config": config, "task": task, "normalization": normalization,
              "windows": {name: {"sampled": len(value.x), **value.stats} for name, value in splits.items()},
              "selection": {"split": "val", "selected_label": selected, "linear_fit_frames": linear_n,
                            "linear_validation_scores": {str(n): _required_score(value) for n, value in linear_metrics.items()}},
              "results": results, "training": details,
              "load_verification": {"normalized_reload_max_abs_delta": reload_delta, "raw_api_max_abs_delta_m": api_delta},
              "acceptance": {"m1_functional": True, "algorithm_acceptance": "not_assessed_in_M1"},
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "numpy": np.__version__, "device": details["device"],
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              "model_config": str(model_path), "latency_scope": "forward_only_device_resident_input"}
    _write_reports(run_dir, report)
    return run_dir / "report.json"
