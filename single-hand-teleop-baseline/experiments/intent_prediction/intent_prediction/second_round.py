from __future__ import annotations

"""第二轮：只在 validation 选残差模型与门控，冻结后只评估一次 test。"""

import csv
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import predict_hold_last
from .gating import apply_motion_gate, fit_motion_gate
from .metrics import compute_forecast_metrics, compute_motion_stratified_metrics, observed_motion_score
from .models import TORCH_AVAILABLE
from .sequence_data import WindowSplit, build_window_split, create_synthetic_smoke_dataset, load_manifest
from .training import predict_neural_checkpoint, train_neural_model


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = output_root.resolve() / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _improvement_percent(reference: float, value: float) -> float:
    if reference <= 0.0:
        return 0.0 if value <= 0.0 else float("-inf")
    return float(100.0 * (reference - value) / reference)


def _build_split(
    data_root: Path,
    *,
    split: str,
    history_frames: int,
    horizons: tuple[int, ...],
    stride: int,
    max_windows: dict[str, Any],
    seed: int,
) -> WindowSplit:
    split_seed_offset = {"train": 0, "val": 1, "test": 2}[split]
    return build_window_split(
        data_root,
        split=split,
        history_frames=history_frames,
        horizon_ms=horizons,
        stride=stride,
        max_windows=max_windows.get(split),
        seed=seed + split_seed_offset,
    )


def _validate_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = config.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("second-round candidates 必须是非空列表")
    labels: set[str] = set()
    offsets: set[int] = set()
    validated = []
    for raw in candidates:
        candidate = dict(raw)
        label = str(candidate.get("label", ""))
        model = str(candidate.get("model", ""))
        offset = int(candidate.get("seed_offset"))
        if not re.fullmatch(r"[a-z0-9_]+", label):
            raise ValueError(f"候选 label 只能包含小写字母、数字和下划线：{label!r}")
        if label in labels:
            raise ValueError(f"候选 label 重复：{label}")
        if offset in offsets:
            raise ValueError(f"候选 seed_offset 重复：{offset}")
        if model not in {"gru", "residual_gru"}:
            raise ValueError(f"第二轮只允许 gru / residual_gru，实际为：{model}")
        candidate["label"] = label
        candidate["model"] = model
        candidate["seed_offset"] = offset
        candidate["loss"] = dict(candidate.get("loss", {}))
        labels.add(label)
        offsets.add(offset)
        validated.append(candidate)
    return validated


def _validation_summary(record: dict[str, Any]) -> dict[str, Any]:
    best = record["gate_fit"]["best"]
    return {
        "label": record["label"],
        "model": record["model"],
        "seed": record["seed"],
        "checkpoint": record["training"]["checkpoint"],
        "checkpoint_sha256": record["training"]["checkpoint_sha256"],
        "raw_metrics": record["gate_fit"]["raw_model_metrics"],
        "gated_metrics": best["full_metrics"],
        "gate_objective": best["objective"],
        "gate_parameters": {
            "recent_frames": record["gate_fit"]["recent_frames"],
            "threshold_percentile": best["threshold_percentile"],
            "threshold": best["threshold"],
            "temperature_fraction": best["temperature_fraction"],
            "temperature": best["temperature"],
            "alpha_by_horizon": best["alpha_by_horizon"],
        },
    }


def _choose_candidate(
    validation_records: list[dict[str, Any]],
    *,
    minimum_objective_improvement: float,
) -> dict[str, Any]:
    ranked = sorted(
        (_validation_summary(record) for record in validation_records),
        key=lambda item: (item["gate_objective"], sum(item["gate_parameters"]["alpha_by_horizon"]), item["label"]),
    )
    best = ranked[0]
    required_objective = 1.0 - float(minimum_objective_improvement)
    if best["gate_objective"] < required_objective - 1e-12:
        return {
            "selected_label": best["label"],
            "selected_model": best["model"],
            "selected_validation_objective": best["gate_objective"],
            "selection_reason": "best_validation_gate_objective_below_hold_last",
            "gate_parameters": best["gate_parameters"],
            "checkpoint": best["checkpoint"],
            "checkpoint_sha256": best["checkpoint_sha256"],
            "ranked_validation_summaries": ranked,
        }
    return {
        "selected_label": "hold_last",
        "selected_model": "hold_last",
        "selected_validation_objective": 1.0,
        "selection_reason": "no_validation_candidate_beats_exact_hold_last",
        "gate_parameters": None,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "ranked_validation_summaries": ranked,
    }


def _acceptance_result(
    *,
    hold_metrics: dict[str, Any],
    selected_metrics: dict[str, Any],
    hold_strata: dict[str, Any],
    selected_strata: dict[str, Any],
    latency_p95_ms: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    q90_hold = hold_strata["strata"]["q90"]["metrics"]
    q90_selected = selected_strata["strata"]["q90"]["metrics"]
    if q90_hold is None or q90_selected is None:
        raise RuntimeError("q90 动态分层为空，无法执行第二轮验收")
    observed = {
        "overall_rmse_improvement_percent": _improvement_percent(hold_metrics["rmse"], selected_metrics["rmse"]),
        "q90_rmse_improvement_percent": _improvement_percent(q90_hold["rmse"], q90_selected["rmse"]),
        "q90_p95_improvement_percent": _improvement_percent(
            q90_hold["p95_abs_error"], q90_selected["p95_abs_error"]
        ),
        "overall_mae_regression_percent": -_improvement_percent(hold_metrics["mae"], selected_metrics["mae"]),
        "range_violation_rate": float(selected_metrics["range_violation_rate"]),
        "single_window_p95_ms": float(latency_p95_ms),
    }
    criteria = {
        "overall_rmse_improvement": observed["overall_rmse_improvement_percent"]
        >= float(config.get("minimum_overall_rmse_improvement_percent", 3.0)),
        "q90_rmse_improvement": observed["q90_rmse_improvement_percent"]
        >= float(config.get("minimum_q90_rmse_improvement_percent", 5.0)),
        "q90_p95_improvement": observed["q90_p95_improvement_percent"]
        >= float(config.get("minimum_q90_p95_improvement_percent", 0.0)),
        "overall_mae_non_regression": observed["overall_mae_regression_percent"]
        <= float(config.get("maximum_overall_mae_regression_percent", 0.5)),
        "range_safety": observed["range_violation_rate"]
        <= float(config.get("maximum_range_violation_rate", 0.0)),
        "latency_budget": observed["single_window_p95_ms"]
        <= float(config.get("maximum_single_window_p95_ms", 10.0)),
    }
    return {
        "criteria_config": dict(config),
        "observed": observed,
        "criteria_pass": criteria,
        "offline_gate_passed": bool(all(criteria.values())),
        "meaning": "仅表示可进入后续离线回放/影子模式，不表示 Unity、摄像头或真机链路已验收。",
    }


def _write_validation_csv(path: Path, validation_records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "label",
            "model",
            "seed",
            "raw_mae",
            "raw_rmse",
            "gated_mae",
            "gated_rmse",
            "gated_p95_abs_error",
            "gate_objective",
            "threshold_percentile",
            "temperature_fraction",
            "alpha_by_horizon",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in validation_records:
            summary = _validation_summary(record)
            gate = summary["gate_parameters"]
            writer.writerow(
                {
                    "label": summary["label"],
                    "model": summary["model"],
                    "seed": summary["seed"],
                    "raw_mae": summary["raw_metrics"]["mae"],
                    "raw_rmse": summary["raw_metrics"]["rmse"],
                    "gated_mae": summary["gated_metrics"]["mae"],
                    "gated_rmse": summary["gated_metrics"]["rmse"],
                    "gated_p95_abs_error": summary["gated_metrics"]["p95_abs_error"],
                    "gate_objective": summary["gate_objective"],
                    "threshold_percentile": gate["threshold_percentile"],
                    "temperature_fraction": gate["temperature_fraction"],
                    "alpha_by_horizon": json.dumps(gate["alpha_by_horizon"]),
                }
            )


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    selection = report["selection"]
    hold = report["test_results"]["hold_last"]["metrics"]
    selected = report["test_results"]["selected_gated"]["metrics"]
    acceptance = report["acceptance"]
    lines = [
        "# 单右手控制意图预测：第二轮离线报告",
        "",
        f"- 选中方案：`{selection['selected_label']}`（`{selection['selected_model']}`）",
        f"- 离线门槛：**{'通过' if acceptance['offline_gate_passed'] else '未通过'}**",
        f"- validation 选型先冻结：`selection.json` SHA-256 `{report['selection_sha256']}`",
        "- test 在选择文件落盘后才加载；本轮未用 test 重选模型或门控。",
        "- subject4 曾用于第一轮评测，因此它不是整个项目历史上的全新盲测集。",
        "",
        "## Validation 候选",
        "",
        "| 候选 | 模型 | raw MAE | gated MAE | gated RMSE | objective | alpha(50/100/150ms) |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in selection["ranked_validation_summaries"]:
        lines.append(
            "| {label} | {model} | {raw_mae:.6f} | {gated_mae:.6f} | {gated_rmse:.6f} | "
            "{objective:.6f} | {alpha} |".format(
                label=item["label"],
                model=item["model"],
                raw_mae=item["raw_metrics"]["mae"],
                gated_mae=item["gated_metrics"]["mae"],
                gated_rmse=item["gated_metrics"]["rmse"],
                objective=item["gate_objective"],
                alpha=" / ".join(f"{value:.2f}" for value in item["gate_parameters"]["alpha_by_horizon"]),
            )
        )
    lines.extend(
        [
            "",
            "## Subject4 测试（选择后仅评估选中方案）",
            "",
            "| 方法 | MAE | RMSE | P95 绝对误差 | 越界率 |",
            "|---|---:|---:|---:|---:|",
            f"| hold-last | {hold['mae']:.8f} | {hold['rmse']:.8f} | {hold['p95_abs_error']:.8f} | {hold['range_violation_rate']:.3g} |",
            f"| selected gated | {selected['mae']:.8f} | {selected['rmse']:.8f} | {selected['p95_abs_error']:.8f} | {selected['range_violation_rate']:.3g} |",
            "",
            "## 预注册离线门槛",
            "",
        ]
    )
    for name, passed in acceptance["criteria_pass"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            acceptance["meaning"],
            "H2O 是公开姿态序列离线证据；没有摄像头、Unity Play Mode、UDP 实时抖动或真机安全证据。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_second_round(
    *,
    config_path: Path,
    data_root: Path | None,
    output_root: Path,
    synthetic_smoke: bool = False,
) -> Path:
    if not TORCH_AVAILABLE:
        raise RuntimeError("第二轮需要 PyTorch；请使用 handai-intent-prediction 独立环境")
    config_path = config_path.resolve()
    source_config_sha256 = _hash_file(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = _unique_run_dir(output_root)
    protocol_events: list[dict[str, Any]] = []

    if synthetic_smoke:
        data_root = run_dir / "synthetic_input"
        create_synthetic_smoke_dataset(data_root, seed=int(config.get("seed", 20260823)))
        config = json.loads(json.dumps(config))
        config["training"]["epochs"] = min(2, int(config["training"].get("epochs", 2)))
        config["training"]["patience"] = min(2, int(config["training"].get("patience", 2)))
        config["max_windows"] = {"train": 700, "val": 250, "test": 250}
        config["research_claims_allowed"] = False
    if data_root is None:
        raise ValueError("真实第二轮必须传入 data_root；代码自检可使用 synthetic_smoke")
    data_root = data_root.resolve()
    manifest_path = data_root / "manifest.json"
    data_manifest = load_manifest(data_root)
    data_contract = {
        "manifest_sha256": _hash_file(manifest_path),
        "mapping_config_sha256": data_manifest.get("mapping_config_sha256"),
        "mapping_contract_version": data_manifest.get("mapping_contract_version"),
        "mapping_contract_sha256": data_manifest.get("mapping_contract_sha256"),
        "dataset_fps": data_manifest.get("fps"),
    }

    seed = int(config.get("seed", 20260823))
    history_frames = int(config.get("history_frames", 30))
    horizons = tuple(int(value) for value in config.get("horizon_ms", [50, 100, 150]))
    stride = int(config.get("stride", 1))
    max_windows = dict(config.get("max_windows", {}))
    candidates = _validate_candidates(config)
    gate_config = dict(config.get("gate", {}))

    train = _build_split(
        data_root,
        split="train",
        history_frames=history_frames,
        horizons=horizons,
        stride=stride,
        max_windows=max_windows,
        seed=seed,
    )
    val = _build_split(
        data_root,
        split="val",
        history_frames=history_frames,
        horizons=horizons,
        stride=stride,
        max_windows=max_windows,
        seed=seed,
    )
    protocol_events.append({"order": 1, "event": "train_and_validation_loaded", "at_utc": _utc_now()})
    train_ids = set(train.sequence_ids.tolist())
    val_ids = set(val.sequence_ids.tolist())
    if train_ids & val_ids:
        raise RuntimeError("train 与 validation 发现重复 sequence_id，已拒绝运行")

    validation_hold = predict_hold_last(val.x, val.horizon_steps)
    validation_hold_metrics = compute_forecast_metrics(val.y, validation_hold, horizon_ms=horizons)
    validation_records: list[dict[str, Any]] = []
    for candidate in candidates:
        training_config = json.loads(json.dumps(config.get("training", {})))
        training_config["run_label"] = candidate["label"]
        training_config["loss"] = candidate["loss"]
        candidate_seed = seed + int(candidate["seed_offset"])
        checkpoint_path = run_dir / "checkpoints" / f"{candidate['label']}.pt"
        validation_prediction, training_details = train_neural_model(
            candidate["model"],
            train=train,
            val=val,
            test=val,
            architecture=dict(config.get("architecture", {})),
            training_config=training_config,
            checkpoint_path=checkpoint_path,
            seed=candidate_seed,
            checkpoint_metadata=data_contract,
        )
        gate_fit = fit_motion_gate(
            val.x,
            val.y,
            validation_prediction,
            horizon_ms=horizons,
            config=gate_config,
        )
        validation_records.append(
            {
                "label": candidate["label"],
                "model": candidate["model"],
                "seed": candidate_seed,
                "loss": candidate["loss"],
                "training": training_details,
                "evaluation_split": "validation",
                "gate_fit": gate_fit,
            }
        )
    protocol_events.append({"order": 2, "event": "all_candidates_evaluated_on_validation", "at_utc": _utc_now()})

    selection = _choose_candidate(
        validation_records,
        minimum_objective_improvement=float(
            config.get("selection", {}).get("minimum_objective_improvement", 0.0)
        ),
    )
    selection_document = {
        "schema_version": "intent-second-round-selection-v1",
        "created_at_utc": _utc_now(),
        "selection_fit_split": "validation",
        "test_loaded": False,
        "test_metrics_available": False,
        "effective_config_sha256": _hash_json(config),
        "data_contract": data_contract,
        "validation_hold_metrics": validation_hold_metrics,
        **selection,
    }
    selection_path = run_dir / "selection.json"
    selection_path.write_text(json.dumps(selection_document, ensure_ascii=False, indent=2), encoding="utf-8")
    selection_sha256 = _hash_file(selection_path)
    protocol_events.append(
        {
            "order": 3,
            "event": "selection_frozen_to_disk_before_test_load",
            "at_utc": _utc_now(),
            "selection_sha256": selection_sha256,
        }
    )

    # 关键顺序约束：到此处 selection.json 已经落盘并哈希，才允许构造 test 窗口。
    test = _build_split(
        data_root,
        split="test",
        history_frames=history_frames,
        horizons=horizons,
        stride=stride,
        max_windows=max_windows,
        seed=seed,
    )
    protocol_events.append({"order": 4, "event": "test_loaded_after_selection_freeze", "at_utc": _utc_now()})
    test_ids = set(test.sequence_ids.tolist())
    if train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("test 与 train/validation 发现重复 sequence_id，已拒绝运行")

    hold_prediction = predict_hold_last(test.x, test.horizon_steps)
    hold_metrics = compute_forecast_metrics(test.y, hold_prediction, horizon_ms=horizons)
    selected_label = selection["selected_label"]
    if selected_label == "hold_last":
        raw_prediction = hold_prediction
        selected_prediction = hold_prediction
        inference_details = {
            "model_name": "hold_last",
            "device": "numpy_cpu",
            "parameter_count": 0,
            "latency_single_window": {"samples": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0},
        }
        test_gate_summary = None
    else:
        raw_prediction, inference_details = predict_neural_checkpoint(
            Path(str(selection["checkpoint"])),
            split=test,
            batch_size=int(config.get("training", {}).get("batch_size", 256)),
            device=str(config.get("training", {}).get("device", "auto")),
        )
        gate_parameters = selection["gate_parameters"]
        selected_prediction, test_gate_summary = apply_motion_gate(
            test.x,
            raw_prediction,
            threshold=float(gate_parameters["threshold"]),
            temperature=float(gate_parameters["temperature"]),
            alpha_by_horizon=list(gate_parameters["alpha_by_horizon"]),
            recent_frames=int(gate_parameters["recent_frames"]),
        )
    protocol_events.append({"order": 5, "event": "selected_candidate_evaluated_once_on_test", "at_utc": _utc_now()})

    raw_metrics = compute_forecast_metrics(test.y, raw_prediction, horizon_ms=horizons)
    selected_metrics = compute_forecast_metrics(test.y, selected_prediction, horizon_ms=horizons)
    recent_frames = int(gate_config.get("recent_frames", 8))
    validation_motion_scores = observed_motion_score(val.x, recent_frames=recent_frames)
    motion_thresholds = {
        "q50": float(np.percentile(validation_motion_scores, 50)),
        "q75": float(np.percentile(validation_motion_scores, 75)),
        "q90": float(np.percentile(validation_motion_scores, 90)),
    }
    hold_strata = compute_motion_stratified_metrics(
        test.y,
        hold_prediction,
        history=test.x,
        horizon_ms=horizons,
        thresholds=motion_thresholds,
        recent_frames=recent_frames,
    )
    selected_strata = compute_motion_stratified_metrics(
        test.y,
        selected_prediction,
        history=test.x,
        horizon_ms=horizons,
        thresholds=motion_thresholds,
        recent_frames=recent_frames,
    )
    acceptance = _acceptance_result(
        hold_metrics=hold_metrics,
        selected_metrics=selected_metrics,
        hold_strata=hold_strata,
        selected_strata=selected_strata,
        latency_p95_ms=float(inference_details["latency_single_window"]["p95_ms"]),
        config=dict(config.get("acceptance", {})),
    )

    configuration_claims_allowed = bool(config.get("research_claims_allowed", True))
    research_claims_allowed = bool(
        data_manifest.get("research_claims_allowed", False)
        and configuration_claims_allowed
        and not synthetic_smoke
    )
    warning = None
    if synthetic_smoke:
        warning = "synthetic smoke 只证明代码链路可执行，不得作为模型效果证据。"
    elif not research_claims_allowed:
        warning = "配置或数据 manifest 禁止将本次运行作为研究效果证据。"

    report = {
        "schema_version": "intent-second-round-report-v1",
        "status": "completed",
        "created_at_utc": _utc_now(),
        "run_purpose": str(config.get("run_purpose", "unspecified")),
        "synthetic": bool(data_manifest.get("synthetic", False)),
        "configuration_claims_allowed": configuration_claims_allowed,
        "research_claims_allowed": research_claims_allowed,
        "warning": warning,
        "confirmatory_status": (
            "internal_re_evaluation_not_pristine_holdout"
            if not synthetic_smoke
            else "synthetic_smoke_non_claimable"
        ),
        "historical_test_reuse_disclosure": (
            "subject4 已用于第一轮结果诊断；第二轮虽在本次运行内先冻结 validation 选择再加载 test，"
            "但 subject4 不是整个项目历史上的全新盲测集。"
            if not synthetic_smoke
            else None
        ),
        "config_path": str(config_path),
        "source_config_sha256": source_config_sha256,
        "effective_config_sha256": _hash_json(config),
        "data_root": str(data_root),
        "dataset": data_manifest.get("dataset"),
        "manifest_sha256": _hash_file(manifest_path),
        "data_contract": data_contract,
        "history_frames": history_frames,
        "horizon_ms": list(horizons),
        "effective_horizon_frames": test.horizon_steps.tolist(),
        "window_counts": {"train": len(train.x), "val": len(val.x), "test": len(test.x)},
        "sequence_counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "sequence_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
        "protocol": {
            "selection_fit_split": "validation",
            "test_loaded_after_selection_freeze": True,
            "test_based_reselection_performed": False,
            "test_models_evaluated": (
                ["hold_last"] if selected_label == "hold_last" else ["hold_last", selected_label]
            ),
            "events": protocol_events,
        },
        "selection_path": str(selection_path),
        "selection_sha256": selection_sha256,
        "selection": selection,
        "validation_hold_metrics": validation_hold_metrics,
        "validation_candidates": validation_records,
        "motion_strata_thresholds_from_validation": motion_thresholds,
        "test_results": {
            "hold_last": {"metrics": hold_metrics, "motion_strata": hold_strata},
            "selected_raw_diagnostic": {
                "label": selected_label,
                "metrics": raw_metrics,
                "inference": inference_details,
            },
            "selected_gated": {
                "label": selected_label,
                "metrics": selected_metrics,
                "motion_strata": selected_strata,
                "gate_summary": test_gate_summary,
            },
        },
        "acceptance": acceptance,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch_available": TORCH_AVAILABLE,
        },
        "scope_boundary": {
            "single_right_hand": True,
            "input": "past svh_preview.target_positions only",
            "unity_modified": False,
            "camera_required": False,
            "hardware_required": False,
            "proves_unity_play_mode": False,
            "proves_real_time_udp": False,
            "proves_real_hardware_safety": False,
        },
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_validation_csv(run_dir / "validation_candidates.csv", validation_records)
    _write_markdown(run_dir / "report.md", report)
    return report_path
