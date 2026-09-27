import argparse
from types import SimpleNamespace

import numpy as np
import pytest

import main as runtime
from output.frame_payload_contract import validate_frame_payload
from utils.config import load_config


@pytest.mark.parametrize("mode", ["video_file", "webcam"])
def test_main_exposes_real_source_time_separately_from_legacy_unix_timestamp(monkeypatch, mode):
    cfg = load_config("configs/ai_m3a.yaml")
    cfg.update(input_source_type=mode, headless=True, gui_enabled=False, save_jsonl=False, prediction_shadow_enabled=False)
    args = argparse.Namespace(config="ignored", max_frames=1, print_json=False)
    output = []
    source = SimpleNamespace(read=lambda: (True, np.zeros((10, 10, 3), dtype=np.uint8)),
        release=lambda: None, nominal_fps=30.0, last_pts_ms=0.0)
    monkeypatch.setattr(runtime, "parse_args", lambda: args)
    monkeypatch.setattr(runtime, "load_config", lambda _: cfg)
    monkeypatch.setattr(runtime, "_apply_cli_overrides", lambda config, _: config)
    monkeypatch.setattr(runtime, "_build_input_source", lambda *_: source)
    monkeypatch.setattr(runtime, "_build_detector", lambda *_: SimpleNamespace(detect=lambda _: [], close=lambda: None))
    monkeypatch.setattr(runtime, "_build_exporter", lambda *_, **__: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(runtime, "_send_prepared_udp", lambda *_, **__: None)
    monkeypatch.setattr(runtime, "_emit_prepared_debug_outputs", lambda payload, **_: output.append(payload))
    runtime.main()
    payload = output[0]
    source_time = payload["source_timing"]
    assert validate_frame_payload(payload) == []
    assert payload["timestamp"] > 1e9  # Unix 兼容字段；不能拿来当媒体源时刻。
    assert payload["timing"]["detection_end_unix_ms"] >= payload["timing"]["source_read_end_unix_ms"]
    if mode == "video_file":
        assert source_time["timebase"] == "media_pts_ms" and source_time["source_time_ms"] == 0
        assert source_time["timestamp_source"] == "container_pts_ms"
    else:
        assert source_time["timebase"] == "monotonic_ms"
        assert source_time["source_time_ms"] == source_time["read_return_monotonic_ms"]
        assert source_time["timestamp_source"] == "source_read_return"
