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
        "control_representation": {
            "valid": True,
            "features_valid": True,
            "command_ready": True,
            "source": "features",
            "gesture_context": "open",
            "preferred_mapping": "grasp",
            "grasp_close": 0.1,
            "thumb_index_proximity": 0.2,
            "effective_pinch_strength": 0.0,
            "pinch_strength": 0.0,
            "support_flex": 0.1,
            "finger_flex": {"thumb": 0.1, "index": 0.1, "middle": 0.1, "ring": 0.1, "little": 0.1},
        },
        "svh": {
            "enabled": True,
            "mode": "preview",
            "valid": True,
            "command_source": "control_representation",
            "target_channels": [0, 1, 2, 3, 4],
            "target_positions": [0.1, 0.2, 0.3, 0.4, 0.5],
            "target_ticks_preview": [],
            "protocol_hint": {
                "set_control_state_addr": "0x09",
                "set_all_channels_addr": "0x03",
                "transport": "mock",
                "channel_layout": "compact5",
                "channel_order": "thumb,index,middle,ring,little",
                "position_units": "normalized_preview",
                "target_tick_units": "none",
            },
        },
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
        "control_representation",
        "svh",
        "fps",
        "latency_ms",
    ]:
        assert key in content
    assert p.exists()
    assert jsonl_path.exists()
    assert console_obj["landmarks_count"] == 0
    assert console_obj["landmarks_2d_preview"] == []
    assert console_obj["control_representation"]["valid"] is True
    assert console_obj["control_representation"]["features_valid"] is True
    assert console_obj["control_representation"]["command_ready"] is True
    assert console_obj["svh"]["target_positions_count"] == 5
    assert console_obj["svh"]["target_positions_preview"] == [0.1, 0.2]
    assert console_obj["svh"]["target_ticks_count"] == 0
    assert console_obj["svh"]["target_ticks_preview_short"] == []
    assert console_obj["svh"]["protocol_hint"]["channel_layout"] == "compact5"
