from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from svh.mapping_contract import (
    FROZEN_MAPPING_IMPLEMENTATION_SHA256_BY_VERSION,
    MAPPING_CONTRACT_VERSION,
    assert_mapping_implementation_compatible,
    mapping_algorithm_sha256,
    mapping_contract_sha256,
    mapping_implementation_sha256,
)
from utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "svh_9ch_preview.yaml"


def test_v2_parameter_contract_stays_compatible_with_frozen_checkpoint():
    cfg = load_config(str(CONFIG_PATH))
    assert mapping_contract_sha256(cfg) == "453c18c3ae851c325cd3b882409800c074f97f7f7ad898865d1d4895b2e318a0"


def test_current_mapping_implementation_matches_frozen_version():
    cfg = load_config(str(CONFIG_PATH))
    frozen = FROZEN_MAPPING_IMPLEMENTATION_SHA256_BY_VERSION[MAPPING_CONTRACT_VERSION]
    assert frozen == "9373dc0857df09609a7a06179afa7e933f918d276384211646cf152c07d690ce"
    assert mapping_implementation_sha256(cfg) == frozen
    assert assert_mapping_implementation_compatible(cfg, {}) == frozen


def test_runtime_gesture_context_drift_is_rejected():
    cfg = load_config(str(CONFIG_PATH))
    cfg["stable_gesture_min_consecutive"] = 3
    with pytest.raises(ValueError, match="implementation SHA-256 不匹配"):
        assert_mapping_implementation_compatible(cfg, {})


def test_algorithm_ast_digest_changes_when_mapping_formula_changes(tmp_path):
    for relative in (
        "src/features/geometry_utils.py",
        "src/features/hand_features.py",
        "src/gesture/rule_based_gesture.py",
        "src/control/control_representation.py",
        "src/svh/svh_adapter.py",
        "experiments/intent_prediction/intent_prediction/h2o_adapter.py",
    ):
        source = PROJECT_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    original = mapping_algorithm_sha256(project_root=tmp_path)
    control_path = tmp_path / "src/control/control_representation.py"
    content = control_path.read_text(encoding="utf-8")
    assert "0.60 * grasp_from_flex" in content
    control_path.write_text(content.replace("0.60 * grasp_from_flex", "0.61 * grasp_from_flex", 1), encoding="utf-8")
    assert mapping_algorithm_sha256(project_root=tmp_path) != original
