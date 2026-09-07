from __future__ import annotations

"""第一轮 Linear/Kalman/GRU/TCN/Transformer 统一实验入口。"""

import csv
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import predict_hold_last, predict_kalman_cv, predict_linear
from .metrics import compute_forecast_metrics
from .models import TORCH_AVAILABLE
from .sequence_data import build_window_split, create_synthetic_smoke_dataset, load_manifest
from .training import train_neural_model


CLASSICAL_MODELS = {"hold_last", "linear", "kalman"}
NEURAL_MODELS = {"gru", "tcn", "transformer"}
NEURAL_MODEL_SEED_OFFSETS = {"gru": 103, "tcn": 104, "transformer": 105}


def _unique_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = output_root.resolve() / timestamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def _classical_prediction(name: str, x: np.ndarray, horizons: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if name == "hold_last":
        return predict_hold_last(x, horizons)
    if name == "linear":
        return predict_linear(x, horizons, fit_frames=int(config.get("linear_fit_frames", 8)))
    if name == "kalman":
        return predict_kalman_cv(
            x,
            horizons,
            process_variance=float(config.get("kalman_process_variance", 2e-3)),
            measurement_variance=float(config.get("kalman_measurement_variance", 8e-3)),
        )
    raise ValueError(f"未知传统模型：{name}")


def _write_metrics_csv(path: Path, results: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model",
                "status",
                "mae",
                "rmse",
                "p95_abs_error",
                "improvement_vs_hold_percent",
                "parameter_count",
                "device",
            ],
        )
        writer.writeheader()
        for name, result in results.items():
            metrics = result.get("metrics", {})
            details = result.get("details", {})
            writer.writerow(
                {
                    "model": name,
                    "status": result.get("status"),
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "p95_abs_error": metrics.get("p95_abs_error"),
                    "improvement_vs_hold_percent": result.get("improvement_vs_hold_percent"),
                    "parameter_count": details.get("parameter_count", 0),
                    "device": details.get("device", "numpy"),
                }
            )


def run_first_round(
    *,
    config_path: Path,
    data_root: Path | None,
    output_root: Path,
    synthetic_smoke: bool = False,
    model_names: list[str] | None = None,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = _unique_run_dir(output_root)
    if synthetic_smoke:
        data_root = run_dir / "synthetic_input"
        create_synthetic_smoke_dataset(data_root, seed=int(config.get("seed", 20260823)))
        config = json.loads(json.dumps(config))
        config["training"]["epochs"] = min(2, int(config["training"].get("epochs", 2)))
        config["training"]["patience"] = min(2, int(config["training"].get("patience", 2)))
        config["max_windows"] = {"train": 700, "val": 250, "test": 250}
    if data_root is None:
        raise ValueError("真实实验必须传入 data_root；仅代码自检请使用 synthetic_smoke")
    data_root = data_root.resolve()
    data_manifest = load_manifest(data_root)

    history_frames = int(config.get("history_frames", 30))
    horizons = tuple(int(value) for value in config.get("horizon_ms", [50, 100, 150]))
    stride = int(config.get("stride", 1))
    max_windows_config = config.get("max_windows", {})
    seed = int(config.get("seed", 20260823))
    splits = {
        split: build_window_split(
            data_root,
            split=split,
            history_frames=history_frames,
            horizon_ms=horizons,
            stride=stride,
            max_windows=max_windows_config.get(split),
            seed=seed + index,
        )
        for index, split in enumerate(("train", "val", "test"))
    }
    train_ids = set(splits["train"].sequence_ids.tolist())
    val_ids = set(splits["val"].sequence_ids.tolist())
    test_ids = set(splits["test"].sequence_ids.tolist())
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("训练、验证、测试之间发现重复 sequence_id，已拒绝运行")

    requested = [name.lower() for name in (model_names or config.get("models", []))]
    unknown = set(requested) - CLASSICAL_MODELS - NEURAL_MODELS
    if unknown:
        raise ValueError(f"未知模型：{sorted(unknown)}")

    results: dict[str, Any] = {}
    for name in requested:
        if name not in CLASSICAL_MODELS:
            continue
        started = time.perf_counter()
        prediction = _classical_prediction(
            name,
            splits["test"].x,
            splits["test"].horizon_steps,
            config.get("classical", {}),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        results[name] = {
            "status": "completed",
            "metrics": compute_forecast_metrics(splits["test"].y, prediction, horizon_ms=horizons),
            "details": {
                "device": "numpy_cpu",
                "parameter_count": 0,
                "batch_inference_ms": elapsed_ms,
                "approx_ms_per_window": elapsed_ms / len(prediction),
            },
        }

    for name in requested:
        if name not in NEURAL_MODELS:
            continue
        if not TORCH_AVAILABLE:
            results[name] = {
                "status": "skipped",
                "reason": "当前解释器未安装 PyTorch；请使用 intent prediction 独立环境",
            }
            continue
        prediction, details = train_neural_model(
            name,
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            architecture=config.get("architecture", {}),
            training_config=config.get("training", {}),
            checkpoint_path=run_dir / "checkpoints" / f"{name}.pt",
            seed=seed + NEURAL_MODEL_SEED_OFFSETS[name],
        )
        results[name] = {
            "status": "completed",
            "metrics": compute_forecast_metrics(splits["test"].y, prediction, horizon_ms=horizons),
            "details": details,
        }

    hold_mae = results.get("hold_last", {}).get("metrics", {}).get("mae")
    if hold_mae is not None and hold_mae > 0:
        for result in results.values():
            model_mae = result.get("metrics", {}).get("mae")
            if model_mae is not None:
                result["improvement_vs_hold_percent"] = float(100.0 * (hold_mae - model_mae) / hold_mae)

    completed = {
        name: result
        for name, result in results.items()
        if result.get("status") == "completed" and result.get("metrics", {}).get("mae") is not None
    }
    best_model = min(completed, key=lambda name: completed[name]["metrics"]["mae"]) if completed else None
    configuration_claims_allowed = bool(config.get("research_claims_allowed", True))
    research_claims_allowed = (
        bool(data_manifest.get("research_claims_allowed", False))
        and configuration_claims_allowed
        and not synthetic_smoke
    )
    if synthetic_smoke:
        warning = "该运行只能证明代码链路可执行，不能作为模型有效性的论文证据。"
    elif not configuration_claims_allowed:
        warning = "该运行使用开发/快速迭代配置，不能与正式配置混用或作为论文效果结论。"
    elif not data_manifest.get("research_claims_allowed", False):
        warning = "数据 manifest 禁止将该运行作为研究效果结论。"
    else:
        warning = None
    report = {
        "schema_version": "intent-first-round-report-v1",
        "status": "completed" if all(result.get("status") == "completed" for result in results.values()) else "completed_with_skips",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "dataset": data_manifest.get("dataset"),
        "synthetic": bool(data_manifest.get("synthetic", False)),
        "run_purpose": str(config.get("run_purpose", "unspecified")),
        "configuration_claims_allowed": configuration_claims_allowed,
        "research_claims_allowed": research_claims_allowed,
        "warning": warning,
        "config_path": str(config_path.resolve()),
        "effective_config": config,
        "history_frames": history_frames,
        "horizon_ms": list(horizons),
        "effective_horizon_frames": splits["test"].horizon_steps.tolist(),
        "window_counts": {split: int(value.x.shape[0]) for split, value in splits.items()},
        "sequence_counts": {split: len(set(value.sequence_ids.tolist())) for split, value in splits.items()},
        "sequence_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
        "models_requested": requested,
        "best_model_by_mae": best_model,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch_available": TORCH_AVAILABLE,
        },
        "results": results,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_metrics_csv(run_dir / "metrics.csv", results)
    return report_path
