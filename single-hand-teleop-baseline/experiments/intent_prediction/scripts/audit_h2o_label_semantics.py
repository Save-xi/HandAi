from __future__ import annotations

"""量化 H2O v2 代理标签中的手势上下文和无内参投影偏差。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT / "src", EXPERIMENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from control.control_representation import build_control_representation  # noqa: E402
from features.hand_features import extract_hand_features  # noqa: E402
from gesture.rule_based_gesture import GestureStabilizer, infer_gesture_raw  # noqa: E402
from intent_prediction.h2o_adapter import (  # noqa: E402
    _iter_numeric_pose_files,
    canonicalize_h2o_camera_xy,
    canonicalize_h2o_normalized_perspective_xy,
    discover_h2o_takes,
    h2o_frame_to_svh9,
    read_h2o_right_hand_pose,
)
from svh.svh_adapter import build_svh_command_preview  # noqa: E402
from utils.config import load_config  # noqa: E402


def _new_stats() -> dict[str, Any]:
    return {
        "takes": 0,
        "valid_frames": 0,
        "invalid_frames": 0,
        "rejected_frames": 0,
        "gesture_raw_stable_different_frames": 0,
        "gesture_label_different_frames": 0,
        "gesture_abs_error_sum": 0.0,
        "gesture_max_channel_abs_error": 0.0,
        "projection_label_different_frames": 0,
        "projection_abs_error_sum": 0.0,
        "projection_max_channel_abs_error": 0.0,
    }


def _preview_positions(
    base_features: dict[str, Any],
    *,
    raw_gesture: str,
    stable_gesture: str,
    cfg: dict[str, Any],
) -> np.ndarray:
    features = deepcopy(base_features)
    features["gesture_raw"] = raw_gesture
    features["gesture_stable"] = stable_gesture
    control = build_control_representation(features, cfg)
    features["control_representation"] = control
    preview = build_svh_command_preview(features, cfg)
    positions = np.asarray(preview.get("target_positions", []), dtype=np.float64)
    if not preview.get("valid", False) or positions.shape != (9,):
        raise ValueError("审计帧未生成有效 svh_9ch")
    return positions


def _record_difference(
    stats: dict[str, Any],
    delta: np.ndarray,
    *,
    count_key: str,
    sum_key: str,
    max_key: str,
) -> None:
    if np.any(delta > 1e-12):
        stats[count_key] += 1
        stats[sum_key] += float(np.sum(delta))
        stats[max_key] = max(float(stats[max_key]), float(np.max(delta)))


def _finalize(stats: dict[str, Any]) -> dict[str, Any]:
    result = dict(stats)
    frames = max(1, int(stats["valid_frames"]))
    channels = frames * 9
    result["gesture_raw_stable_different_fraction"] = (
        float(stats["gesture_raw_stable_different_frames"]) / frames
    )
    result["gesture_label_different_fraction"] = float(stats["gesture_label_different_frames"]) / frames
    result["gesture_mean_channel_abs_error_over_all_frames"] = float(stats["gesture_abs_error_sum"]) / channels
    result["projection_label_different_fraction"] = float(stats["projection_label_different_frames"]) / frames
    result["projection_mean_channel_abs_error_over_all_frames"] = (
        float(stats["projection_abs_error_sum"]) / channels
    )
    return result


def run_audit(
    *,
    h2o_root: Path,
    mapping_config_path: Path,
    split_policy: str,
    limit_takes: int | None,
) -> dict[str, Any]:
    root = h2o_root.resolve()
    config_path = mapping_config_path.resolve()
    cfg = load_config(str(config_path))
    takes = discover_h2o_takes(root, split_policy=split_policy)
    if limit_takes is not None:
        takes = takes[: max(0, int(limit_takes))]
    if not takes:
        raise FileNotFoundError("没有找到可审计的 H2O cam4/hand_pose take")

    grouped: dict[str, dict[str, Any]] = {"all": _new_stats()}
    for take in takes:
        grouped.setdefault(take.subject, _new_stats())
        keys = ("all", take.subject)
        for key in keys:
            grouped[key]["takes"] += 1
        stabilizer = GestureStabilizer(
            confirm_frames=int(cfg.get("stable_gesture_min_consecutive", 2)),
            unknown_confirm_frames=int(cfg.get("stable_unknown_consecutive", 1)),
        )
        previous_frame: int | None = None
        for frame_id, pose_path in _iter_numeric_pose_files(take.cam_dir / "hand_pose"):
            if previous_frame is not None and frame_id != previous_frame + 1:
                stabilizer = GestureStabilizer(
                    confirm_frames=int(cfg.get("stable_gesture_min_consecutive", 2)),
                    unknown_confirm_frames=int(cfg.get("stable_unknown_consecutive", 1)),
                )
            previous_frame = frame_id
            try:
                valid, points_xyz = read_h2o_right_hand_pose(pose_path)
                if not valid:
                    for key in keys:
                        grouped[key]["invalid_frames"] += 1
                    stabilizer = GestureStabilizer(
                        confirm_frames=int(cfg.get("stable_gesture_min_consecutive", 2)),
                        unknown_confirm_frames=int(cfg.get("stable_unknown_consecutive", 1)),
                    )
                    continue

                legacy_xy = canonicalize_h2o_camera_xy(points_xyz)
                perspective_xy = canonicalize_h2o_normalized_perspective_xy(points_xyz)
                base = extract_hand_features(
                    [(float(x), float(y)) for x, y in legacy_xy],
                    handedness="Right",
                    confidence=1.0,
                    timestamp=frame_id / 30.0,
                    landmarks_xyz=[tuple(float(value) for value in point) for point in points_xyz],
                )
                raw_gesture = infer_gesture_raw(base, cfg)
                stable_gesture = stabilizer.update(raw_gesture)
                raw_positions = _preview_positions(
                    base,
                    raw_gesture=raw_gesture,
                    stable_gesture=raw_gesture,
                    cfg=cfg,
                )
                stable_positions = _preview_positions(
                    base,
                    raw_gesture=raw_gesture,
                    stable_gesture=stable_gesture,
                    cfg=cfg,
                )
                perspective_positions, _ = h2o_frame_to_svh9(
                    points_xyz,
                    perspective_xy,
                    timestamp_s=frame_id / 30.0,
                    mapping_cfg=cfg,
                )
            except (OSError, ValueError, FloatingPointError):
                for key in keys:
                    grouped[key]["rejected_frames"] += 1
                stabilizer = GestureStabilizer(
                    confirm_frames=int(cfg.get("stable_gesture_min_consecutive", 2)),
                    unknown_confirm_frames=int(cfg.get("stable_unknown_consecutive", 1)),
                )
                continue

            gesture_delta = np.abs(raw_positions - stable_positions)
            projection_delta = np.abs(raw_positions - perspective_positions)
            for key in keys:
                stats = grouped[key]
                stats["valid_frames"] += 1
                stats["gesture_raw_stable_different_frames"] += int(raw_gesture != stable_gesture)
                _record_difference(
                    stats,
                    gesture_delta,
                    count_key="gesture_label_different_frames",
                    sum_key="gesture_abs_error_sum",
                    max_key="gesture_max_channel_abs_error",
                )
                _record_difference(
                    stats,
                    projection_delta,
                    count_key="projection_label_different_frames",
                    sum_key="projection_abs_error_sum",
                    max_key="projection_max_channel_abs_error",
                )

    return {
        "schema_version": "h2o-label-semantics-audit-v1",
        "scope": "single_right_hand_pose_only_proxy_labels",
        "h2o_root": str(root),
        "mapping_config": str(config_path),
        "split_policy": split_policy,
        "limit_takes": limit_takes,
        "definitions": {
            "gesture_comparison": "v2 raw_gesture_as_stable_proxy versus runtime consecutive stabilizer",
            "projection_comparison": "v2 camera x/y fallback versus normalized x/z,y/z perspective fallback",
            "truth_boundary": "both sides are deterministic proxy mappings, not physical SVH joint truth",
        },
        "groups": {key: _finalize(value) for key, value in sorted(grouped.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 H2O v2 标签的手势上下文与 pose-only 投影偏差")
    parser.add_argument("--h2o-root", type=Path, required=True)
    parser.add_argument(
        "--mapping-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "svh_9ch_preview.yaml",
    )
    parser.add_argument("--split-policy", choices=("cross_subject", "official"), default="cross_subject")
    parser.add_argument("--limit-takes", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="可选 JSON 输出；父目录会自动创建")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_audit(
        h2o_root=args.h2o_root,
        mapping_config_path=args.mapping_config,
        split_policy=args.split_policy,
        limit_takes=args.limit_takes,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(output)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
