from __future__ import annotations

from copy import deepcopy

import pytest

from utils.config import load_config, validate_config


def test_shipped_configs_pass_startup_validation():
    for path in (
        "configs/ai.yaml",
        "configs/default.yaml",
        "configs/svh_9ch_preview.yaml",
        "configs/unity_udp_preview.yaml",
    ):
        assert load_config(path)


def test_config_inheritance_rejects_cycles(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="循环"):
        load_config(str(tmp_path / "a.yaml"))


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


def test_open_release_ratios_accept_hand_open_ratio_above_one():
    # hand_open_ratio 不是概率：V1 有效帧 40% 大于 1.0（p90 约 1.14），M3-C 需能把 full 调到 1 以上。
    cfg = load_config("configs/ai_m3a.yaml")
    cfg["control_open_release_full_ratio"] = 1.05
    validate_config(cfg)
    cfg["control_open_release_start_ratio"] = 1.02
    cfg["control_open_release_full_ratio"] = 1.14
    validate_config(cfg)

    cfg["control_open_release_start_ratio"] = 1.10
    cfg["control_open_release_full_ratio"] = 1.05
    with pytest.raises(ValueError, match="start_ratio 必须小于 full_ratio"):
        validate_config(cfg)


@pytest.mark.parametrize("key", ["control_open_release_start_ratio", "control_open_release_full_ratio"])
@pytest.mark.parametrize("value", [0, -0.1, float("nan"), float("inf"), True, "1.05"])
def test_open_release_ratios_must_be_finite_positive_numbers(key, value):
    cfg = load_config("configs/ai_m3a.yaml")
    cfg[key] = value
    with pytest.raises(ValueError, match=f"{key} 必须是正数"):
        validate_config(cfg)


def test_runtime_output_cannot_overwrite_frozen_examples():
    cfg = load_config("configs/default.yaml")
    cfg["output_json_path"] = str(
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "examples"
        / "sample_output.json"
    )

    with pytest.raises(ValueError, match="examples"):
        validate_config(cfg)
