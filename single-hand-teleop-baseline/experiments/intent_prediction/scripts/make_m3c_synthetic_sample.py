"""生成仅供工具冒烟的编号空画面；全部身份与标签均为合成。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(EXPERIMENT)]
from intent_prediction.camera_m3c import import_clip, load_timeline, save_annotations  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs/datasets/camera_m3c_synthetic_demo")
    args = parser.parse_args()
    args.dataset.mkdir(parents=True, exist_ok=True)
    video = args.dataset / "synthetic_numbered.avi"
    if video.exists():
        raise FileExistsError(f"演示目录已有视频：{video}")
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 360))
    if not writer.isOpened():
        raise RuntimeError("无法写入合成视频")
    for i in range(12):
        frame = np.full((360, 640, 3), 245, dtype=np.uint8)
        cv2.putText(frame, f"SYNTHETIC FRAME {i:02d}", (35, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (25, 25, 25), 3)
        writer.write(frame)
    writer.release()
    clip = import_clip(args.dataset, video=video, person_id="synthetic-person",
                       session_id="synthetic-session", clip_id="synthetic-numbered",
                       scenario="synthetic-empty-frame", role="development",
                       split_policy="independent_person", synthetic=True)
    timeline = load_timeline(clip)
    save_annotations(clip, {
        "schema_version": "camera-m3c-annotation-v1", "clip_id": "synthetic-numbered",
        "keypoint_frames": [{"frame_id": i, "presence": "absent", "sampling_purpose": "uniform",
                              "points": []} for i in (0, 5, 11)],
        "gesture_intervals": [{"start_ms": timeline[0]["pts_ms"],
                               "end_ms": timeline[-1]["pts_ms"] + 50, "label": "absent"}],
        "transition_events": [],
    })
    print(f"合成样例已生成：{clip}")


if __name__ == "__main__":
    main()
