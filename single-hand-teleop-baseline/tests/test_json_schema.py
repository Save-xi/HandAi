from output.json_exporter import JsonExporter


def test_json_export_fields(tmp_path):
    obj = {
        "timestamp": 1.0,
        "detected": True,
        "handedness": "Right",
        "confidence": 0.95,
        "gesture_raw": "open",
        "gesture": "open",
        "pinch_distance_norm": 0.1,
        "hand_open_ratio": 0.7,
        "finger_curl": {"thumb": 0.1, "index": 0.1, "middle": 0.1, "ring": 0.1, "little": 0.1},
        "landmarks_2d": [],
        "fps": 30.0,
        "latency_ms": 10.0,
    }
    p = tmp_path / "last.json"
    jsonl_path = tmp_path / "session.jsonl"
    ex = JsonExporter(str(p), save_last_json=True, jsonl_path=str(jsonl_path))
    ex.save_last_frame(obj)
    ex.append_jsonl(obj)
    content = ex.to_json_str(obj)
    console_obj = ex.to_console_obj(obj, landmarks_preview_count=2)

    for key in [
        "timestamp",
        "detected",
        "handedness",
        "confidence",
        "gesture_raw",
        "gesture",
        "pinch_distance_norm",
        "hand_open_ratio",
        "finger_curl",
        "landmarks_2d",
        "fps",
        "latency_ms",
    ]:
        assert key in content
    assert p.exists()
    assert jsonl_path.exists()
    assert console_obj["landmarks_count"] == 0
    assert console_obj["landmarks_2d_preview"] == []
