from output.json_exporter import JsonExporter


def test_json_export_fields(tmp_path):
    obj = {
        "timestamp": 1.0,
        "detected": True,
        "handedness": "Right",
        "gesture": "open",
        "pinch_distance": 0.1,
        "hand_open_ratio": 0.7,
        "finger_curl": {"thumb": 0.1, "index": 0.1, "middle": 0.1, "ring": 0.1, "little": 0.1},
        "landmarks_2d": [],
        "fps": 30.0,
        "latency_ms": 10.0,
    }
    p = tmp_path / "last.json"
    ex = JsonExporter(str(p), save_last_json=True)
    ex.save_last_frame(obj)
    content = ex.to_json_str(obj)

    for key in [
        "timestamp",
        "detected",
        "handedness",
        "gesture",
        "pinch_distance",
        "hand_open_ratio",
        "finger_curl",
        "landmarks_2d",
        "fps",
        "latency_ms",
    ]:
        assert key in content
    assert p.exists()
