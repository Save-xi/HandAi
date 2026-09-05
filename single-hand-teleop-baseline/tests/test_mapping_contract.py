from copy import deepcopy

import pytest

from svh.mapping_contract import assert_mapping_compatible, legacy_v2_mapping_contract
from utils.config import load_config


def test_existing_model_accepts_ai_and_preview_configs():
    for path in ("configs/ai.yaml", "configs/svh_9ch_preview.yaml", "configs/unity_udp_preview.yaml"):
        cfg = load_config(path)
        assert_mapping_compatible(cfg, legacy_v2_mapping_contract())


def test_output_settings_do_not_change_model_inputs():
    cfg = load_config("configs/ai.yaml")
    cfg.update(output_json_path="outputs/another.json", unity_udp_port=19000)
    assert_mapping_compatible(cfg, legacy_v2_mapping_contract())


def test_mapping_change_reports_the_actual_parameter():
    cfg = load_config("configs/ai.yaml")
    cfg["control_grasp_closed_ref"] = 0.7
    with pytest.raises(ValueError, match="control_grasp_closed_ref"):
        assert_mapping_compatible(cfg, legacy_v2_mapping_contract())


def test_channel_permutation_is_rejected():
    expected = deepcopy(legacy_v2_mapping_contract())
    expected["channel_order"].reverse()
    with pytest.raises(ValueError, match="channel_order"):
        assert_mapping_compatible(load_config("configs/ai.yaml"), expected)
