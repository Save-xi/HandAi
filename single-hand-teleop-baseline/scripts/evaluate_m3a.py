"""M3-A 开发诊断：固定释放事件 + 同一检测结果的三路映射。

仅用于开发片段与工程边界，不提供手势准确率、三维真值或预测收益。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/intent_prediction"))

from control.control_representation import build_control_representation  # noqa: E402
from control.open_release import OpenReleaseState  # noqa: E402
from features.hand_features import empty_features  # noqa: E402
from gesture.rule_based_gesture import GestureStabilizer, infer_gesture_raw  # noqa: E402
from intent_prediction.camera_domain_eval import resolve_media_timestamp_ms  # noqa: E402
from perception.mediapipe_hand import MediaPipeHandDetector  # noqa: E402
from pipeline import HandPipeline  # noqa: E402
from svh.mapping_contract import mapping_contract_payload  # noqa: E402
from svh.svh_adapter import build_svh_command_preview  # noqa: E402
from svh.svh_layout import SVH_9CH_NAMES  # noqa: E402
from utils.config import load_config  # noqa: E402


def jump_statistics(positions: list) -> dict:
    # 不把失效占位 [] 当成零通道，也不跨失效帧算跳变。
    differences = [np.abs(np.array(b) - a) for a, b in zip(positions, positions[1:]) if a and b]
    array = np.asarray(differences)
    return {
        "consecutive_valid_pairs": len(differences),
        "per_channel_p95": np.percentile(array, 95, axis=0).tolist() if len(array) else None,
        "per_channel_max": array.max(axis=0).tolist() if len(array) else None,
        "max_jump": float(array.max()) if len(array) else None,
    }


def feature_trace(samples: list, cfg: dict, fps: float = 30) -> list:
    stabilizer = GestureStabilizer(int(cfg.get("stable_gesture_min_consecutive", 2)),
                                  int(cfg.get("stable_unknown_consecutive", 1)))
    release = OpenReleaseState()
    trace = []
    for i, sample in enumerate(samples):
        payload = empty_features(i / fps)
        if sample is not None:
            curl, ratio, pinch = sample
            payload.update(detected=True, hand_open_ratio=ratio, pinch_distance_norm=pinch,
                           finger_curl={name: curl for name in ("thumb", "index", "middle", "ring", "little")})
            payload["gesture_raw"] = infer_gesture_raw(payload, cfg)
            payload["gesture_stable"] = stabilizer.update(payload["gesture_raw"])
        else:
            stabilizer.reset()
            release.reset()
        payload["control_representation"] = build_control_representation(payload, cfg, release_state=release)
        preview = build_svh_command_preview(payload, cfg)
        trace.append({"frame_index": i, "source_time_ms": i * 1000 / fps,
                      "gesture_raw": payload["gesture_raw"], "gesture_stable": payload["gesture_stable"],
                      "valid": preview["valid"], "positions": preview["target_positions"],
                      "release_weight": release.weight if cfg.get("control_open_release_mode") == "continuous_v1" else None})
    return trace


def settling_index(trace: list, event_start: int) -> int:
    """首次达到并保持最终输出的 90% 转换量；按每路自身终点计算。"""
    initial = np.asarray(trace[event_start - 1]["positions"])
    final = np.asarray(trace[-1]["positions"])
    tolerance = 0.10 * np.abs(final - initial) + 1e-8
    for i in range(event_start, len(trace)):
        if all(row["valid"] and np.all(np.abs(np.asarray(row["positions"]) - final) <= tolerance) for row in trace[i:]):
            return i
    raise ValueError("事件没有稳定终点")


def release_report(fps: float = 30) -> dict:
    old, new = load_config("configs/ai.yaml"), load_config("configs/ai_m3a.yaml")
    micro = [(c, 0.96, 1.0) for c in (0.4499, 0.4499, 0.4501)]
    cases = {"curl_boundary": micro}
    open_pose, unknown, pinch, fist = (0.1, 0.96, 1.0), (0.6, 0.96, 1.0), (0.3, 0.96, 0.12), (0.7, 0.5, 0.8)
    for name, a, b in (("open_to_unknown", open_pose, unknown), ("unknown_to_open", unknown, open_pose),
                       ("open_to_pinch", open_pose, pinch), ("pinch_to_open", pinch, open_pose),
                       ("normal_opening", fist, open_pose), ("normal_closing", open_pose, fist)):
        cases[name] = [a] * 3 + [tuple((1 - t) * np.array(a) + t * np.array(b)) for t in np.linspace(0, 1, 16)[1:]] + [b] * 3
    cases["loss_recovery"] = [open_pose] * 3 + [None, None] + [open_pose] * 3
    report = {"fps": fps, "source_period_ms": 1000 / fps, "channel_order": list(SVH_9CH_NAMES),
              "transition_definition": "first persistent 90% endpoint response after frame 3; each arm uses its own endpoint",
              "cases": {}}
    for name, samples in cases.items():
        case = {"input_features": samples}
        for arm, cfg in (("legacy_release", old), ("continuous_release", new)):
            trace = feature_trace(samples, cfg, fps)
            case[arm] = {**jump_statistics([row["positions"] for row in trace]), "trace": trace}
            if name not in {"curl_boundary", "loss_recovery"}:
                index = settling_index(trace, 3)
                case[arm].update(settled_frame=index, transition_lag_ms=(index - 3) * 1000 / fps)
        if "settled_frame" in case["legacy_release"]:
            case["added_lag_ms"] = (case["continuous_release"]["settled_frame"] - case["legacy_release"]["settled_frame"]) * 1000 / fps
        report["cases"][name] = case
    boundary = report["cases"]["curl_boundary"]
    extra = max(case.get("added_lag_ms", 0) for case in report["cases"].values())
    report["gates"] = {"microjump_at_most_0_05": boundary["continuous_release"]["max_jump"] <= 0.05,
                       "extra_lag_at_most_one_source_period": extra <= 1000 / fps + 1e-8,
                       "maximum_added_lag_ms": extra,
                       "loss_immediately_invalid": all(not row["valid"] for row in report["cases"]["loss_recovery"]["continuous_release"]["trace"][3:5])}
    return report


def video_report(video: Path, limit: int, run_dir: Path) -> dict:
    old = load_config("configs/ai.yaml")
    new = load_config("configs/ai_m3a.yaml")
    configs = {"legacy_geometry_release": old,
               "corrected_geometry_legacy_release": {**new, "control_open_release_mode": "legacy"},
               "corrected_geometry_continuous_release": new}
    pipelines = {name: HandPipeline(cfg) for name, cfg in configs.items()}
    positions, reasons, gestures, resets = ({name: [] for name in configs}, {name: Counter() for name in configs},
                                          {name: Counter() for name in configs}, Counter())
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("开发诊断要求视频提供有效 FPS")
    detector = MediaPipeHandDetector(max_num_hands=int(old.get("max_num_hands", 2)), input_mirrored=bool(old.get("input_mirrored", False)))
    previous, timestamp_sources, count = None, Counter(), 0
    dimensions = set()
    trace_path = run_dir / f"{video.stem}.frames.jsonl"
    disagreements = Counter()
    try:
        with trace_path.open("w", encoding="utf-8") as handle:
            for i in range(limit):
                available, frame = capture.read()
                if not available:
                    break
                decision = resolve_media_timestamp_ms(i, raw_pts_ms=float(capture.get(cv2.CAP_PROP_POS_MSEC)),
                    nominal_fps=fps, previous_timestamp_ms=previous)
                previous = decision.timestamp_ms
                timestamp_sources[decision.source] += 1
                dimensions.add((int(frame.shape[1]), int(frame.shape[0])))
                detections = detector.detect(frame)  # 每个源帧只检测一次，三路共用。
                outputs = {}
                for name, pipeline in pipelines.items():
                    output = pipeline.process_detections(detections, frame_index=i, timestamp=previous / 1000, fps=fps)
                    outputs[name] = output
                    positions[name].append(output["svh_preview"]["target_positions"])
                    reasons[name][output["input_diagnostics"]["reason"]] += 1
                    gestures[name][output["gesture_stable"]] += 1
                    resets[name] += int(output["input_diagnostics"]["state_reset"])
                baseline = outputs["legacy_geometry_release"]
                for name, output in outputs.items():
                    if baseline["control_ready"] and output["control_ready"] and baseline["gesture_raw"] != output["gesture_raw"]:
                        disagreements[name] += 1
                handle.write(json.dumps({"source_video": video.name, "timestamp_source": decision.source, "arms": outputs}, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
    finally:
        detector.close()
        capture.release()
    if count == 0:
        raise ValueError("视频没有可读取的开发帧")
    return {"video": video.name, "frames": count, "nominal_fps": fps, "image_sizes": sorted(dimensions),
            "timestamp_sources": dict(timestamp_sources), "trace_file": trace_path.name,
            "scope": "historical development clip; person/session unknown; no manual labels",
            "arms": {name: {"valid_frames": sum(bool(p) for p in positions[name]), "reasons": dict(reasons[name]),
                             "stable_gestures": dict(gestures[name]), "state_reset_frames": resets[name],
                             "raw_gesture_disagreements_with_legacy_on_joint_valid_frames": disagreements[name],
                             **jump_statistics(positions[name]), "mapping": mapping_contract_payload(configs[name])} for name in configs}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, action="append", default=[], help="开发视频，可重复传入")
    parser.add_argument("--max-frames", type=int, default=180, help="每段只读取前 N 帧")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/m3a", help="诊断输出根目录")
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames 必须大于 0")
    if len({video.stem for video in args.video}) != len(args.video):
        parser.error("开发视频文件名必须唯一，避免覆盖逐帧输出")
    run_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"scope": "M3-A development diagnostics, not accuracy or device acceptance",
              "release": release_report(), "videos": []}
    for video in args.video:
        result = video_report(video, args.max_frames, run_dir)
        report["videos"].append(result)
        print(f"{video.name}: {result['frames']} 帧完成", flush=True)
    path = run_dir / "summary.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"报告：{path}")
    print(json.dumps(report["release"]["gates"], ensure_ascii=False))
    if not all(report["release"]["gates"][key] for key in ("microjump_at_most_0_05", "extra_lag_at_most_one_source_period", "loss_immediately_invalid")):
        raise SystemExit("释放研发门槛未通过，保留诊断并停止验收")


if __name__ == "__main__":
    main()
