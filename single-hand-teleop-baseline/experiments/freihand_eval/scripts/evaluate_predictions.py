from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from freihand.io import (
    limit_split,
    load_config,
    load_predictions,
    load_split_annotations,
    resolve_path,
    write_json,
    write_text,
)
from freihand.metrics import evaluate_predictions
from freihand.projection import numpy_to_jsonable_points, project_xyz_to_uv
from freihand.report import render_eval_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction JSON against FreiHAND annotations.")
    parser.add_argument("--config", default="../configs/freihand_eval.yaml", help="Path to freihand_eval.yaml")
    parser.add_argument("--split", default=None, help="Override split name, e.g. training or evaluation")
    parser.add_argument("--predictions", default=None, help="Override predictions.json path")
    return parser.parse_args()


def resolve_config_arg(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (THIS_DIR / path).resolve()


def resolve_cli_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _project_gt_2d(config, split):
    if split.xyz is None or split.K is None:
        return {}
    uv = project_xyz_to_uv(
        split.xyz,
        split.K,
        z_epsilon=float(config.data.get("projection", {}).get("z_epsilon", 1.0e-8)),
    )
    return {sid: numpy_to_jsonable_points(uv[i]) for i, sid in enumerate(split.sample_ids)}


def main() -> None:
    args = parse_args()
    config = load_config(resolve_config_arg(args.config))
    split_name = args.split or str(config.data.get("project", {}).get("default_split", "evaluation"))
    max_samples = config.data.get("runtime", {}).get("max_samples")
    split = limit_split(load_split_annotations(config, split_name, require_xyz=True, require_K=False), max_samples)

    prediction_path = (
        resolve_cli_path(args.predictions)
        if args.predictions
        else resolve_path(config, config.data["paths"]["prediction_json"])
    )
    predictions = load_predictions(prediction_path)
    gt_2d_by_id = _project_gt_2d(config, split)

    metric_cfg = config.data.get("metrics", {})
    metrics = evaluate_predictions(
        predictions,
        gt_2d_by_id,
        joint_count=int(metric_cfg.get("joint_count", 21)),
        pck_2d_thresholds_px=[float(v) for v in metric_cfg.get("pck_2d_thresholds_px", [5, 10, 20, 30])],
    )

    metrics_path = resolve_path(config, config.data["paths"]["eval_metrics_json"])
    report_path = resolve_path(config, config.data["paths"]["eval_report_md"])
    write_json(metrics_path, metrics)
    write_text(report_path, render_eval_report(metrics, split=split_name, prediction_path=str(prediction_path)))
    print(f"metrics saved to {metrics_path}")
    print(f"report saved to {report_path}")


if __name__ == "__main__":
    main()
