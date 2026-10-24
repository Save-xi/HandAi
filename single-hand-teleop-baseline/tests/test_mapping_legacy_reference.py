"""映射修订不能静默改变 H2O v2 或摄像头 v4 的逐帧语义。"""
import json
from pathlib import Path

import pytest

from perception.base import HandDetection
from pipeline import HandPipeline
from svh.mapping_contract import mapping_contract_payload
from utils.config import load_config

CASES = json.loads((Path(__file__).parent / "fixtures/mapping_legacy_reference.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=["h2o-v2", "camera-v4"])
def test_original_mapping_payloads_remain_frame_exact(case):
    cfg = load_config(case["config"])
    assert mapping_contract_payload(cfg) == case["mapping"]
    pipe = HandPipeline(cfg)
    for i, event in enumerate(case["events"]):
        ds = [HandDetection(**event["detection"])] if event["detection"] else []
        actual = pipe.process_detections(ds, frame_index=i, timestamp=event["timestamp"])
        actual.pop("latency_ms")
        assert actual == event["expected"]
