from __future__ import annotations

import argparse
import time

import cv2

from capture.webcam import WebcamSource
from features.hand_features import empty_features, extract_hand_features
from gesture.rule_based_gesture import infer_gesture
from output.json_exporter import JsonExporter
from perception.hand_filter import select_right_hand
from perception.mediapipe_hand import MediaPipeHandDetector
from utils.config import load_config
from utils.logger import get_logger
from utils.timer import FrameTimer, now_ts
from visualize.overlay_2d import compose_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single right hand teleop baseline demo")
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    parser.add_argument("--camera-index", default=None, type=int, help="Override camera index")
    parser.add_argument("--print-json", action="store_true", help="Print frame JSON to console")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger()

    if args.camera_index is not None:
        cfg["camera_index"] = args.camera_index

    source = WebcamSource(
        camera_index=int(cfg["camera_index"]),
        width=int(cfg["display_width"]),
        height=int(cfg["display_height"]),
    )
    detector = MediaPipeHandDetector(
        max_num_hands=int(cfg.get("max_num_hands", 2)),
        min_detection_confidence=float(cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(cfg.get("min_tracking_confidence", 0.5)),
    )
    exporter = JsonExporter(
        output_path=str(cfg.get("output_json_path", "examples/sample_output.json")),
        save_last_json=bool(cfg.get("save_last_json", True)),
    )

    timer = FrameTimer()
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
                payload = extract_hand_features(right.landmarks_2d, right.handedness, right.confidence, ts)
                if bool(cfg.get("draw_landmarks", True)):
                    detector.draw_landmarks(frame, right.landmarks_2d)

            payload["gesture"] = infer_gesture(payload, cfg)
            dt = timer.tick()
            payload["fps"] = 1.0 / dt if dt > 1e-6 else 0.0
            payload["latency_ms"] = (time.perf_counter() - t0) * 1000.0

            if args.print_json:
                exporter.print_console(payload)
            exporter.save_last_frame(payload)
            exporter.send(payload)

            view = compose_view(frame, payload)
            cv2.imshow("single-right-hand-teleop-baseline", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
