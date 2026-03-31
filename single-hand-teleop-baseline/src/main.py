from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from capture.webcam import WebcamSource
from features.hand_features import empty_features, extract_hand_features, invalidate_control_features
from gesture.rule_based_gesture import GestureStabilizer, infer_gesture_raw
from output.json_exporter import JsonExporter
from perception.hand_filter import select_right_hand
from perception.landmark_quality import assess_control_readiness
from perception.mediapipe_hand import MediaPipeHandDetector
from utils.config import load_config
from utils.logger import get_logger
from utils.recent_frames import RecentFrameBuffer
from utils.timer import FrameTimer, now_ts
from visualize.overlay_2d import compose_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single right hand teleop baseline demo")
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    parser.add_argument("--camera-index", default=None, type=int, help="Override camera index")
    parser.add_argument("--print-json", action="store_true", help="Print frame JSON to console")
    parser.add_argument("--save-jsonl", action="store_true", help="Enable per-frame JSONL logging for this session")
    return parser.parse_args()


def _build_jsonl_session_path(cfg: dict) -> str:
    output_dir = Path(str(cfg.get("jsonl_output_dir", "outputs")))
    return str(output_dir / f"session_{datetime.now():%Y%m%d_%H%M%S}.jsonl")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger()

    if args.camera_index is not None:
        cfg["camera_index"] = args.camera_index
    if args.save_jsonl:
        cfg["save_jsonl"] = True

    source = WebcamSource(
        camera_index=int(cfg["camera_index"]),
        width=int(cfg["display_width"]),
        height=int(cfg["display_height"]),
    )
    detector = MediaPipeHandDetector(
        max_num_hands=int(cfg.get("max_num_hands", 2)),
        min_detection_confidence=float(cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(cfg.get("min_tracking_confidence", 0.5)),
        input_mirrored=bool(cfg.get("input_mirrored", False)),
    )
    exporter = JsonExporter(
        output_path=str(cfg.get("output_json_path", "examples/sample_output.json")),
        save_last_json=bool(cfg.get("save_last_json", True)),
        jsonl_path=_build_jsonl_session_path(cfg) if bool(cfg.get("save_jsonl", False)) else None,
        logger=logger,
    )
    history = RecentFrameBuffer(maxlen=int(cfg.get("recent_frames_buffer_size", 10)))
    stabilizer = GestureStabilizer(
        confirm_frames=int(cfg.get("stable_gesture_min_consecutive", 2)),
        unknown_confirm_frames=int(cfg.get("stable_unknown_consecutive", 1)),
    )
    print_every_n_frames = max(1, int(cfg.get("console_print_every_n_frames", 5)))
    landmarks_preview_count = max(0, int(cfg.get("console_landmarks_preview_count", 3)))

    timer = FrameTimer()
    frame_index = 0
    logger.info("Press q to quit.")

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                logger.warning("Cannot read frame from input source.")
                break

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            right = select_right_hand(detections)
            ts = now_ts()

            if right is None:
                payload = empty_features(ts)
            else:
                payload = extract_hand_features(
                    right.landmarks_2d,
                    right.handedness,
                    right.confidence,
                    ts,
                    landmarks_xyz=right.landmarks_xyz,
                )
                quality = assess_control_readiness(right.landmarks_2d, cfg)
                if not bool(quality["control_ready"]):
                    payload = invalidate_control_features(payload)
                if bool(cfg.get("draw_landmarks", True)):
                    detector.draw_landmarks(frame, right.landmarks_2d)

            payload["gesture_raw"] = infer_gesture_raw(payload, cfg)
            payload["gesture"] = stabilizer.update(payload["gesture_raw"])
            history.append(payload)
            dt = timer.tick()
            payload["fps"] = 1.0 / dt if dt > 1e-6 else 0.0
            payload["latency_ms"] = (time.perf_counter() - t0) * 1000.0

            if args.print_json and frame_index % print_every_n_frames == 0:
                exporter.print_console(payload, landmarks_preview_count=landmarks_preview_count)
            exporter.save_last_frame(payload)
            exporter.append_jsonl(payload)
            exporter.send(payload)

            view = compose_view(frame, payload)
            cv2.imshow("single-right-hand-teleop-baseline", view)
            frame_index += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
