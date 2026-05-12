from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
PROJECT_ROOT = MODULE_ROOT.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in [MODULE_ROOT, SRC_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from freihand.io import load_config, resolve_path, split_config_path, write_json
from perception.hand_filter import select_right_hand
from perception.mediapipe_hand import MediaPipeHandDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the current MediaPipe baseline on FreiHAND images.")
    parser.add_argument("--config", default="../configs/freihand_eval.yaml", help="Path to freihand_eval.yaml")
    parser.add_argument("--split", default=None, help="Override split name, e.g. training or evaluation")
    parser.add_argument("--max-samples", default=None, type=int, help="Limit the number of images for a smoke test")
    parser.add_argument("--output", default=None, help="Override output predictions JSON path")
    parser.add_argument("--input-mirrored", action="store_true", help="Treat images as mirrored/selfie view")
    parser.add_argument("--prefer-any-hand", action="store_true", help="Use the highest-confidence hand if no Right hand is selected")
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


def _pixel_points(normalized_points, width: int, height: int) -> list[list[float]]:
    return [[float(x) * width, float(y) * height] for x, y in normalized_points]


def _choose_detection(detections, *, prefer_any_hand: bool):
    selected = select_right_hand(detections)
    if selected is not None or not prefer_any_hand or not detections:
        return selected
    return sorted(detections, key=lambda item: item.confidence, reverse=True)[0]


def main() -> None:
    args = parse_args()
    config = load_config(resolve_config_arg(args.config))
    split_name = args.split or str(config.data.get("project", {}).get("default_split", "evaluation"))
    pipeline_cfg = config.data.get("current_pipeline", {})
    image_root = split_config_path(config, split_name, "image_root")
    if image_root is None or not image_root.exists():
        raise FileNotFoundError(f"missing image_root for split '{split_name}': {image_root}")

    extension = str(pipeline_cfg.get("image_extension", ".png"))
    image_paths = sorted(image_root.glob(f"*{extension}"))
    max_samples = args.max_samples if args.max_samples is not None else config.data.get("runtime", {}).get("max_samples")
    if max_samples is not None:
        image_paths = image_paths[: max(0, int(max_samples))]

    detector = MediaPipeHandDetector(
        max_num_hands=int(pipeline_cfg.get("max_num_hands", 2)),
        min_detection_confidence=float(pipeline_cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(pipeline_cfg.get("min_tracking_confidence", 0.5)),
        input_mirrored=bool(args.input_mirrored or pipeline_cfg.get("input_mirrored", False)),
        static_image_mode=bool(pipeline_cfg.get("static_image_mode", True)),
    )
    predictions = {}
    detected_count = 0
    try:
        for index, image_path in enumerate(image_paths):
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            start = time.perf_counter()
            detections = detector.detect(frame)
            selected = _choose_detection(detections, prefer_any_hand=bool(args.prefer_any_hand))
            latency_ms = (time.perf_counter() - start) * 1000.0
            sample_id = image_path.stem
            if selected is None:
                predictions[sample_id] = {
                    "keypoints_2d": None,
                    "latency_ms": latency_ms,
                    "detected": False,
                    "handedness": None,
                    "confidence": None,
                }
                continue

            height, width = frame.shape[:2]
            detected_count += 1
            predictions[sample_id] = {
                "keypoints_2d": _pixel_points(selected.landmarks_2d, width, height),
                "latency_ms": latency_ms,
                "detected": True,
                "handedness": selected.handedness,
                "confidence": selected.confidence,
            }
            if index > 0 and index % 500 == 0:
                print(f"processed {index}/{len(image_paths)} images; detected {detected_count}")
    finally:
        detector.close()

    output_path = (
        resolve_cli_path(args.output)
        if args.output
        else resolve_path(config, config.data["paths"]["current_pipeline_prediction_json"])
    )
    write_json(output_path, predictions)
    print(f"saved {len(predictions)} predictions to {output_path}")
    print(f"detected complete 2D hands: {detected_count}/{len(image_paths)}")


if __name__ == "__main__":
    main()
