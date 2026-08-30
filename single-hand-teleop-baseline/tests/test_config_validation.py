from __future__ import annotations

from copy import deepcopy

import pytest

from utils.config import load_config, validate_config


def test_shipped_configs_pass_startup_validation():
    for path in (
        "configs/default.yaml",
        "configs/svh_9ch_preview.yaml",
        "configs/unity_udp_preview.yaml",
    ):
        assert load_config(path)


def test_unity_udp_rejects_non_loopback_destination():
    cfg = load_config("configs/unity_udp_preview.yaml")
    cfg["unity_udp_host"] = "192.168.1.20"

    with pytest.raises(ValueError, match="loopback"):
        validate_config(cfg)


def test_mapping_layout_count_and_open_release_thresholds_are_consistent():
    cfg = load_config("configs/svh_9ch_preview.yaml")
    bad_layout = deepcopy(cfg)
    bad_layout["svh_preview_channel_count"] = 5
    with pytest.raises(ValueError, match="svh_preview_channel_count"):
        validate_config(bad_layout)

    bad_release = deepcopy(cfg)
    bad_release["control_open_release_start_ratio"] = 0.96
    bad_release["control_open_release_full_ratio"] = 0.95
    with pytest.raises(ValueError, match="start_ratio"):
        validate_config(bad_release)


def test_runtime_output_cannot_overwrite_frozen_examples():
    cfg = load_config("configs/default.yaml")
    cfg["output_json_path"] = str(
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "examples"
        / "sample_output.json"
    )

    with pytest.raises(ValueError, match="examples"):
        validate_config(cfg)
