from __future__ import annotations

"""不用摄像头生成确定性的 SVH 9 通道 Unity 预览轨迹。"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from output.frame_payload_contract import prepare_frame_payload
from output.json_exporter import JsonExporter


POSES = (
    ("open", [0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.80]),
    ("pinch", [0.72, 0.88, 0.68, 0.58, 0.16, 0.16, 0.12, 0.12, 0.18]),
    ("open", [0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.80]),
    ("fist", [0.90, 0.80, 0.92, 0.92, 0.92, 0.92, 0.90, 0.90, 0.05]),
    ("open", [0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.80]),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无摄像头的 Unity SVH 9 通道预览演示")
    parser.add_argument("--host", default="127.0.0.1", help="Unity UDP 地址，默认仅本机")
    parser.add_argument("--port", default=18080, type=int, help="Unity UDP 端口")
    parser.add_argument("--duration", default=12.0, type=float, help="完整演示时长（秒）")
    parser.add_argument("--fps", default=30.0, type=float, help="发送帧率")
    parser.add_argument("--max-frames", default=None, type=int, help="限制帧数，适合 smoke test")
    parser.add_argument("--send", action="store_true", help="实际发送 UDP；默认只离线校验 payload")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="允许发送到非 loopback 地址；本阶段通常不应使用",
    )
    return parser.parse_args()


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _interpolate_pose(progress: float) -> tuple[str, list[float]]:
    progress = min(1.0, max(0.0, progress))
    segment_count = len(POSES) - 1
    position = progress * segment_count
    segment = min(segment_count - 1, int(position))
    local = _smoothstep(position - segment)
    start_name, start = POSES[segment]
    end_name, end = POSES[segment + 1]
    values = [
        float(start_value + (end_value - start_value) * local)
        for start_value, end_value in zip(start, end)
    ]
    gesture = end_name if local >= 0.5 else start_name
    return gesture, values


def _synthetic_landmarks() -> list[list[float]]:
    return [
        [0.50, 0.88],
        [0.43, 0.80],
        [0.38, 0.71],
        [0.34, 0.62],
        [0.30, 0.54],
        [0.44, 0.68],
        [0.43, 0.54],
        [0.42, 0.42],
        [0.41, 0.31],
        [0.50, 0.66],
        [0.50, 0.50],
        [0.50, 0.37],
        [0.50, 0.25],
        [0.56, 0.68],
        [0.57, 0.53],
        [0.58, 0.41],
        [0.59, 0.31],
        [0.62, 0.72],
        [0.65, 0.59],
        [0.67, 0.49],
        [0.69, 0.40],
    ]


def build_synthetic_payload(frame_index: int, progress: float) -> dict[str, Any]:
    gesture, positions = _interpolate_pose(progress)
    now_ms = time.time() * 1000.0
    landmarks_2d = _synthetic_landmarks()
    finger_flex = {
        "thumb": positions[0],
        "index": positions[3],
        "middle": positions[5],
        "ring": positions[6],
        "little": positions[7],
    }
    pinch_strength = max(positions[1], positions[2])
    payload = {
        "timestamp": now_ms / 1000.0,
        "frame_index": frame_index,
        "detected": True,
        "handedness": "Right",
        "confidence": 1.0,
        "control_ready": True,
        "gesture_raw": gesture,
        "gesture_stable": gesture,
        "pinch_distance_norm": max(0.05, 1.0 - pinch_strength),
        "hand_open_ratio": max(0.0, 1.0 - sum(positions[:8]) / 8.0),
        "finger_curl": finger_flex,
        "landmarks_2d": landmarks_2d,
        "landmarks_3d": [[x, y, 0.0] for x, y in landmarks_2d],
        "control_representation": {
            "valid": True,
            "features_valid": True,
            "command_ready": True,
            "source": "synthetic_preview_demo",
            "gesture_context": gesture,
            "preferred_mapping": "pinch" if gesture == "pinch" else "grasp",
            "grasp_close": sum(positions[:8]) / 8.0,
            "thumb_index_proximity": pinch_strength,
            "effective_pinch_strength": pinch_strength,
            "pinch_strength": pinch_strength,
            "support_flex": (positions[5] + positions[6] + positions[7]) / 3.0,
            "finger_flex": finger_flex,
        },
        "svh_preview": {
            "enabled": True,
            "mode": "preview",
            "valid": True,
            "command_source": "control_representation",
            "target_channels": list(range(9)),
            "target_positions": positions,
            "target_ticks_preview": [],
            "protocol_hint": {
                "set_control_state_addr": "0x09",
                "set_all_channels_addr": "0x03",
                "transport": "mock",
                "channel_layout": "svh_9ch",
                "channel_order": "thumb_flexion,thumb_opposition,index_finger_distal,index_finger_proximal,middle_finger_distal,middle_finger_proximal,ring_finger,pinky,finger_spread",
                "position_units": "normalized_preview",
                "target_tick_units": "none",
            },
        },
        "fps": 0.0,
        "latency_ms": 0.0,
        "timing": {
            "schema_version": 1,
            "clock": "unix_epoch_ms",
            "source_read_start_unix_ms": now_ms,
            "source_read_end_unix_ms": now_ms,
            "detection_end_unix_ms": now_ms,
            "baseline_end_unix_ms": now_ms,
            "preview_end_unix_ms": now_ms,
            "payload_ready_unix_ms": now_ms,
            "udp_send_attempt_unix_ms": None,
        },
    }
    return prepare_frame_payload(payload, include_deprecated_aliases=False)


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration 必须大于 0")
    if args.fps <= 0.0:
        raise ValueError("--fps 必须大于 0")
    if args.send and args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote:
        raise ValueError("默认拒绝向非 loopback 地址发送；确认后显式添加 --allow-remote")

    frame_count = max(1, int(math.ceil(args.duration * args.fps)))
    if args.max_frames is not None:
        frame_count = min(frame_count, max(0, int(args.max_frames)))
    exporter = JsonExporter(
        output_path=str(PROJECT_ROOT / "outputs" / "synthetic_preview_last.json"),
        save_last_json=False,
        unity_udp_enabled=bool(args.send),
        unity_udp_host=args.host,
        unity_udp_port=args.port,
    )
    started = time.perf_counter()
    try:
        for frame_index in range(frame_count):
            progress = 1.0 if frame_count <= 1 else frame_index / (frame_count - 1)
            payload = build_synthetic_payload(frame_index, progress)
            if args.send:
                payload["timing"]["udp_send_attempt_unix_ms"] = time.time() * 1000.0
                exporter.send_prepared_frame(payload)
                target_elapsed = (frame_index + 1) / args.fps
                remaining = target_elapsed - (time.perf_counter() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        exporter.close()

    mode = "已发送到 Unity" if args.send else "仅完成离线 contract 校验，未发送 UDP"
    print(f"{mode}；帧数={frame_count}，目标={args.host}:{args.port}")
    if not args.send:
        print("实际预览请先确认 Unity 的 Apply Baseline Preview To Hardware=false，再添加 --send。")


if __name__ == "__main__":
    main()
