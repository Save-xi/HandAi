from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "intent_prediction"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from intent_prediction import camera_domain_eval as camera_eval  # noqa: E402


def test_media_timestamp_prefers_pts_and_falls_back_monotonically():
    first = camera_eval.resolve_media_timestamp_ms(
        0,
        raw_pts_ms=0.0,
        nominal_fps=30.0,
        previous_timestamp_ms=None,
    )
    assert first.timestamp_ms == 0.0
    assert first.source == "container_pts_ms"

    repeated = camera_eval.resolve_media_timestamp_ms(
        1,
        raw_pts_ms=0.0,
        nominal_fps=30.0,
        previous_timestamp_ms=first.timestamp_ms,
    )
    assert repeated.timestamp_ms == pytest.approx(1000.0 / 30.0)
    assert repeated.source == "frame_index_over_nominal_fps"

    valid = camera_eval.resolve_media_timestamp_ms(
        2,
        raw_pts_ms=66.666,
        nominal_fps=30.0,
        previous_timestamp_ms=repeated.timestamp_ms,
    )
    assert valid.timestamp_ms == pytest.approx(66.666)
    assert valid.source == "container_pts_ms"


def test_media_timestamp_continuity_fallback_handles_shifted_first_pts():
    first = camera_eval.resolve_media_timestamp_ms(
        0,
        raw_pts_ms=1000.0,
        nominal_fps=25.0,
        previous_timestamp_ms=None,
    )
    second = camera_eval.resolve_media_timestamp_ms(
        1,
        raw_pts_ms=float("nan"),
        nominal_fps=25.0,
        previous_timestamp_ms=first.timestamp_ms,
    )
    assert second.timestamp_ms == pytest.approx(1040.0)
    assert second.source == "continuity_fallback_period"


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan")])
def test_media_timestamp_rejects_invalid_nominal_fps(fps):
    with pytest.raises(ValueError, match="nominal_fps"):
        camera_eval.resolve_media_timestamp_ms(
            0,
            raw_pts_ms=0.0,
            nominal_fps=fps,
            previous_timestamp_ms=None,
        )


def test_parse_video_specs_requires_unique_existing_ids(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    specs = camera_eval.parse_video_specs([f"V1={first}", f"V2={second}"])
    assert [spec.video_id for spec in specs] == ["V1", "V2"]
    with pytest.raises(ValueError, match="重复"):
        camera_eval.parse_video_specs([f"V1={first}", f"V1={second}"])
    with pytest.raises(ValueError, match="同一视频"):
        camera_eval.parse_video_specs([f"V1={first}", f"V2={first}"])


class _FakeCapture:
    def __init__(self) -> None:
        self.frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(3)]
        self.pts = [0.0, 0.0, 66.666]
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def get(self, prop):
        if prop == camera_eval.cv2.CAP_PROP_FPS:
            return 30.0
        if prop == camera_eval.cv2.CAP_PROP_FRAME_COUNT:
            return 3.0
        if prop == camera_eval.cv2.CAP_PROP_FRAME_WIDTH:
            return 3.0
        if prop == camera_eval.cv2.CAP_PROP_FRAME_HEIGHT:
            return 2.0
        if prop == camera_eval.cv2.CAP_PROP_POS_MSEC:
            return self.pts[max(0, self.index - 1)]
        return 0.0

    def release(self) -> None:
        self.released = True


def test_video_processing_uses_media_timeline_not_wall_clock(tmp_path, monkeypatch):
    capture = _FakeCapture()
    observed_timestamps: list[float] = []
    closed: list[bool] = []

    def fake_builder(cfg, *, input_mirrored, logger):
        del cfg, input_mirrored, logger

        def process(frame, frame_index, timestamp_s, source_fps):
            del frame, source_fps
            observed_timestamps.append(timestamp_s)
            return (
                {
                    "timestamp": timestamp_s,
                    "frame_index": frame_index,
                    "detected": True,
                    "control_ready": True,
                    "svh_preview": {"valid": True},
                },
                1.0,
            )

        return process, lambda: closed.append(True)

    monkeypatch.setattr(camera_eval, "_build_video_payload_processor", fake_builder)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    output = tmp_path / "baseline.jsonl"
    summary = camera_eval.process_video_to_baseline_jsonl(
        camera_eval.VideoSpec("V1", video),
        output_path=output,
        cfg={},
        timeline_config={
            "synthetic_epoch_seconds": 1700000000.0,
            "minimum_nominal_fps": 1.0,
            "maximum_nominal_fps": 240.0,
        },
        input_mirrored=False,
        logger=camera_eval.logging.getLogger("test"),
        capture_factory=lambda _: capture,
    )
    assert capture.released is True
    assert closed == [True]
    assert observed_timestamps == pytest.approx(
        [1700000000.0, 1700000000.0 + 1.0 / 30.0, 1700000000.0 + 0.066666]
    )
    assert summary["source_timeline"]["effective_fps"] == pytest.approx(30.0, rel=2e-4)
    assert summary["source_timeline"]["median_interval_fps"] == pytest.approx(30.0, rel=2e-4)
    assert summary["source_timeline"]["duration_based_fps"] == pytest.approx(30.0003, rel=2e-4)
    assert summary["source_timeline"]["timestamp_source_counts"] == {
        "container_pts_ms": 2,
        "frame_index_over_nominal_fps": 1,
    }
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["frame_index"] for row in rows] == [0, 1, 2]


def test_camera_domain_module_does_not_create_udp_exporter():
    source = Path(camera_eval.__file__).read_text(encoding="utf-8")
    assert "JsonExporter" not in source
