import argparse
import logging
from pathlib import Path

from features.hand_features import extract_hand_features
from main import _apply_cli_overrides, _apply_extension_chain, _build_runtime_mode


def _args(**overrides):
    base = {
        "config": "configs/default.yaml",
        "camera_index": None,
        "video_file": None,
        "input_mirrored": False,
        "enable_control": False,
        "preview_svh": False,
        "no_gui": False,
        "headless": False,
        "max_frames": None,
        "print_json": False,
        "save_jsonl": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_runtime_mode_defaults_to_baseline_only():
    runtime = _build_runtime_mode(
        {
            "input_source_type": "webcam",
            "input_mirrored": False,
            "gui_enabled": True,
            "headless": False,
            "enable_control_extension": False,
            "svh_enable_preview": False,
        }
    )

    assert runtime.input_source_type == "webcam"
    assert runtime.gui_enabled is True
    assert runtime.headless is False
    assert runtime.control_extension_enabled is False
    assert runtime.svh_preview_enabled is False


def test_cli_overrides_enable_preview_headless_and_video_file(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    updated = _apply_cli_overrides(
        {
            "input_source_type": "webcam",
            "input_mirrored": False,
            "gui_enabled": True,
            "headless": False,
            "enable_control_extension": False,
            "svh_enable_preview": False,
        },
        _args(
            video_file="fixtures/demo.mp4",
            input_mirrored=True,
            preview_svh=True,
            headless=True,
            save_jsonl=True,
        ),
    )

    runtime = _build_runtime_mode(updated)
    assert updated["input_source_type"] == "video_file"
    assert Path(updated["video_file_path"]).name == "demo.mp4"
    assert updated["save_jsonl"] is True
    assert runtime.input_mirrored is True
    assert runtime.gui_enabled is False
    assert runtime.headless is True
    assert runtime.control_extension_enabled is True
    assert runtime.svh_preview_enabled is True


def test_extension_chain_disabled_degrades_to_stable_placeholders(control_cfg, svh_cfg):
    cfg = {
        **control_cfg,
        **svh_cfg,
        "enable_control_extension": False,
        "svh_enable_preview": False,
    }
    runtime = _build_runtime_mode(cfg)
    payload = {
        "detected": True,
        "gesture_stable": "open",
        "finger_curl": {"thumb": 0.02, "index": 0.01, "middle": 0.01, "ring": 0.01, "little": 0.01},
        "hand_open_ratio": 0.95,
        "pinch_distance_norm": 0.90,
    }

    _apply_extension_chain(
        payload,
        cfg,
        runtime,
        svh_transport=None,
        logger=logging.getLogger("test-runtime"),
    )

    assert payload["control_representation"]["valid"] is False
    assert payload["control_ready"] is False
    assert payload["svh_preview"]["enabled"] is False
    assert payload["svh_preview"]["valid"] is False


def test_preview_svh_mode_auto_enables_control_extension(control_cfg, svh_cfg, synthetic_hand_pose):
    cfg = {
        **control_cfg,
        **svh_cfg,
        "enable_control_extension": False,
        "svh_enable_preview": True,
    }
    runtime = _build_runtime_mode(cfg)
    xy, xyz = synthetic_hand_pose("open")
    payload = extract_hand_features(xy, handedness="Right", confidence=0.98, timestamp=1.23, landmarks_xyz=xyz)
    payload["gesture_raw"] = "open"
    payload["gesture_stable"] = "open"

    _apply_extension_chain(
        payload,
        cfg,
        runtime,
        svh_transport=None,
        logger=logging.getLogger("test-runtime"),
    )

    assert runtime.control_extension_enabled is True
    assert payload["control_representation"]["features_valid"] is True
    assert payload["svh_preview"]["enabled"] is True
