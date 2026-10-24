"""映射修订不能静默改变 H2O v2 或摄像头 v4 的逐帧语义。

样本在 Windows 上生成；Linux 的数学库可能让浮点末位不同，因此浮点只容许 1e-12 的舍入差异，
结构、键和其余字段仍逐项严格相等。
"""
import json
from pathlib import Path

import pytest

from perception.base import HandDetection
from pipeline import HandPipeline
from svh.mapping_contract import mapping_contract_payload
from utils.config import load_config

CASES = json.loads((Path(__file__).parent / "fixtures/mapping_legacy_reference.json").read_text(encoding="utf-8"))


def _assert_same(actual, expected, path="payload"):
    if isinstance(expected, float) or isinstance(actual, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12), path
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key, value in expected.items():
            _assert_same(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, value in enumerate(expected):
            _assert_same(actual[index], value, f"{path}[{index}]")
    else:
        assert actual == expected, path


@pytest.mark.parametrize("case", CASES, ids=["h2o-v2", "camera-v4"])
def test_original_mapping_payloads_remain_frame_exact(case):
    cfg = load_config(case["config"])
    assert mapping_contract_payload(cfg) == case["mapping"]
    pipe = HandPipeline(cfg)
    for i, event in enumerate(case["events"]):
        ds = [HandDetection(**event["detection"])] if event["detection"] else []
        actual = pipe.process_detections(ds, frame_index=i, timestamp=event["timestamp"])
        actual.pop("latency_ms")
        _assert_same(actual, event["expected"], f"events[{i}]")
