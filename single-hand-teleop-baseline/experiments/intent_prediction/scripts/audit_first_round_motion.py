from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from intent_prediction.baselines import predict_hold_last, predict_kalman_cv, predict_linear  # noqa: E402
from intent_prediction.metrics import (  # noqa: E402
    compute_forecast_metrics,
    compute_motion_stratified_metrics,
    observed_motion_score,
)
from intent_prediction.sequence_data import build_window_split  # noqa: E402
from intent_prediction.training import predict_neural_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计第一轮模型在不同历史运动强度下的表现")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, required=True, help="含 GRU/TCN 与传统基线的完整报告")
    parser.add_argument("--strict-transformer-report", type=Path, required=True, help="严格确定性 Transformer 报告")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--recent-frames", type=int, default=8)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    comparison = _load(args.comparison_report)
    strict_transformer = _load(args.strict_transformer_report)
    config = _load(Path(comparison["config_path"]))
    horizons = tuple(int(value) for value in config["horizon_ms"])
    common = {
        "history_frames": int(config["history_frames"]),
        "horizon_ms": horizons,
        "stride": int(config["stride"]),
    }
    data_root = args.data_root.resolve()
    val = build_window_split(
        data_root,
        split="val",
        max_windows=config["max_windows"].get("val"),
        seed=int(config["seed"]) + 1,
        **common,
    )
    test = build_window_split(
        data_root,
        split="test",
        max_windows=config["max_windows"].get("test"),
        seed=int(config["seed"]) + 2,
        **common,
    )
    validation_scores = observed_motion_score(val.x, recent_frames=args.recent_frames)
    thresholds = {
        "motion_ge_validation_q50": float(np.percentile(validation_scores, 50)),
        "motion_ge_validation_q75": float(np.percentile(validation_scores, 75)),
        "motion_ge_validation_q90": float(np.percentile(validation_scores, 90)),
    }

    classical_cfg = config["classical"]
    predictions = {
        "hold_last": (predict_hold_last(test.x, test.horizon_steps), {"source": "recomputed"}),
        "linear": (
            predict_linear(test.x, test.horizon_steps, fit_frames=int(classical_cfg["linear_fit_frames"])),
            {"source": "recomputed"},
        ),
        "kalman": (
            predict_kalman_cv(
                test.x,
                test.horizon_steps,
                process_variance=float(classical_cfg["kalman_process_variance"]),
                measurement_variance=float(classical_cfg["kalman_measurement_variance"]),
            ),
            {"source": "recomputed"},
        ),
    }
    for name in ("gru", "tcn"):
        checkpoint = Path(comparison["results"][name]["details"]["checkpoint"])
        predictions[name] = predict_neural_checkpoint(
            checkpoint,
            split=test,
            batch_size=int(config["training"]["batch_size"]),
        )
    transformer_checkpoint = Path(strict_transformer["results"]["transformer"]["details"]["checkpoint"])
    predictions["transformer_strict"] = predict_neural_checkpoint(
        transformer_checkpoint,
        split=test,
        batch_size=int(config["training"]["batch_size"]),
    )

    source_metrics = {
        "hold_last": comparison["results"]["hold_last"]["metrics"],
        "linear": comparison["results"]["linear"]["metrics"],
        "kalman": comparison["results"]["kalman"]["metrics"],
        "gru": comparison["results"]["gru"]["metrics"],
        "tcn": comparison["results"]["tcn"]["metrics"],
        "transformer_strict": strict_transformer["results"]["transformer"]["metrics"],
    }
    source_details = {
        "hold_last": comparison["results"]["hold_last"]["details"],
        "linear": comparison["results"]["linear"]["details"],
        "kalman": comparison["results"]["kalman"]["details"],
        "gru": comparison["results"]["gru"]["details"],
        "tcn": comparison["results"]["tcn"]["details"],
        "transformer_strict": strict_transformer["results"]["transformer"]["details"],
    }
    results = {}
    for name, (prediction, details) in predictions.items():
        overall = compute_forecast_metrics(test.y, prediction, horizon_ms=horizons)
        expected_mae = float(source_metrics[name]["mae"])
        results[name] = {
            "overall": overall,
            "motion_stratified": compute_motion_stratified_metrics(
                test.y,
                prediction,
                history=test.x,
                horizon_ms=horizons,
                thresholds=thresholds,
                recent_frames=args.recent_frames,
            ),
            "prediction_source": details,
            "complexity": {
                key: source_details[name].get(key)
                for key in (
                    "device",
                    "parameter_count",
                    "epochs_completed",
                    "training_seconds",
                    "latency_single_window",
                )
            },
            "source_report_mae": expected_mae,
            "source_report_mae_delta": float(overall["mae"] - expected_mae),
            "overall_mae_matches_source": bool(abs(overall["mae"] - expected_mae) <= 1e-7),
        }

    hold = results["hold_last"]
    for result in results.values():
        hold_q90 = hold["motion_stratified"]["strata"]["motion_ge_validation_q90"]["metrics"]
        result_q90 = result["motion_stratified"]["strata"]["motion_ge_validation_q90"]["metrics"]
        result["improvement_vs_hold_percent"] = {
            "overall_mae": float(
                100.0 * (hold["overall"]["mae"] - result["overall"]["mae"]) / hold["overall"]["mae"]
            ),
            "overall_rmse": float(
                100.0 * (hold["overall"]["rmse"] - result["overall"]["rmse"]) / hold["overall"]["rmse"]
            ),
            "motion_q90_mae": float(100.0 * (hold_q90["mae"] - result_q90["mae"]) / hold_q90["mae"]),
            "motion_q90_rmse": float(100.0 * (hold_q90["rmse"] - result_q90["rmse"]) / hold_q90["rmse"]),
        }

    report = {
        "schema_version": "intent-motion-audit-v1",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_claims_allowed": bool(
            comparison.get("research_claims_allowed") and strict_transformer.get("research_claims_allowed")
        ),
        "data_root": str(data_root),
        "comparison_report": str(args.comparison_report.resolve()),
        "strict_transformer_report": str(args.strict_transformer_report.resolve()),
        "window_counts": {"val": len(val.x), "test": len(test.x)},
        "motion_definition": {
            "uses_future_target": False,
            "description": "最近若干已观测帧相邻差分绝对值在时间和9通道上的均值",
            "recent_frames": args.recent_frames,
            "threshold_source": "validation split only",
            "validation_thresholds": thresholds,
            "validation_score_percentiles": {
                str(p): float(np.percentile(validation_scores, p)) for p in (0, 25, 50, 75, 90, 100)
            },
        },
        "results": results,
    }
    output = args.output or args.comparison_report.resolve().parent / "motion_audit_strict_transformer.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output.resolve()), "status": report["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
