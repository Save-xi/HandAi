from __future__ import annotations

import importlib.util
from pathlib import Path

from output.frame_payload_contract import validate_frame_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_demo_module():
    path = PROJECT_ROOT / "scripts" / "send_unity_preview_demo.py"
    spec = importlib.util.spec_from_file_location("synthetic_unity_preview_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_preview_demo_generates_valid_open_pinch_and_fist_frames():
    demo = _load_demo_module()

    payloads = [
        demo.build_synthetic_payload(0, 0.0),
        demo.build_synthetic_payload(1, 0.25),
        demo.build_synthetic_payload(2, 0.75),
        demo.build_synthetic_payload(3, 1.0),
    ]

    for payload in payloads:
        assert validate_frame_payload(payload) == []
        assert payload["detected"] is True
        assert len(payload["landmarks_2d"]) == 21
        assert len(payload["svh_preview"]["target_positions"]) == 9
        assert all(0.0 <= value <= 1.0 for value in payload["svh_preview"]["target_positions"])

    assert payloads[0]["gesture_stable"] == "open"
    assert any(payload["gesture_stable"] == "pinch" for payload in payloads)
    assert any(payload["gesture_stable"] == "fist" for payload in payloads)
