import json
from pathlib import Path

from output.frame_payload_contract import normalize_frame_payload, validate_frame_payload
from output.json_exporter import JsonExporter
from utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sample_payload(frame_index: int) -> dict:
    return normalize_frame_payload(
        {
            "timestamp": 1000.0 + frame_index,
            "frame_index": frame_index,
            "detected": True,
            "handedness": "Right",
            "confidence": 0.95,
            "control_ready": True,
            "gesture_raw": "open",
            "gesture_stable": "open",
            "pinch_distance_norm": 0.1,
            "hand_open_ratio": 0.8,
            "finger_curl": {"thumb": 0.1, "index": 0.1, "middle": 0.1, "ring": 0.1, "little": 0.1},
            "landmarks_2d": [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            "landmarks_3d": [[0.1, 0.2, -0.01], [0.2, 0.3, -0.02], [0.3, 0.4, -0.03]],
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
            "svh_preview": {
                "enabled": False,
                "mode": "disabled",
                "valid": False,
                "command_source": None,
                "target_channels": [],
                "target_positions": [],
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
        },
        include_deprecated_aliases=False,
    )


def test_direct_exporter_methods_remain_immediately_readable(tmp_path):
    output_path = tmp_path / "last.json"
    jsonl_path = tmp_path / "session.jsonl"
    exporter = JsonExporter(
        str(output_path),
        save_last_json=True,
        jsonl_path=str(jsonl_path),
        export_last_every_n_frames=5,
        jsonl_flush_interval=10,
    )
    payload = _sample_payload(3)

    exporter.save_last_frame(payload)
    exporter.append_jsonl(payload)
    exporter.close()

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()

    assert validate_frame_payload(saved) == []
    assert len(lines) == 1
    assert validate_frame_payload(json.loads(lines[0])) == []


def test_export_prepared_frame_throttles_last_json_until_close(tmp_path):
    output_path = tmp_path / "last.json"
    exporter = JsonExporter(
        str(output_path),
        save_last_json=True,
        export_last_every_n_frames=5,
    )

    exporter.export_prepared_frame(_sample_payload(0), frame_index=0)
    first_saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert first_saved["frame_index"] == 0

    exporter.export_prepared_frame(_sample_payload(1), frame_index=1)
    still_saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert still_saved["frame_index"] == 0

    exporter.close()
    final_saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert final_saved["frame_index"] == 1
    assert validate_frame_payload(final_saved) == []


def test_export_prepared_frame_honors_jsonl_flush_interval(tmp_path):
    jsonl_path = tmp_path / "session.jsonl"
    exporter = JsonExporter(
        str(tmp_path / "last.json"),
        save_last_json=False,
        jsonl_path=str(jsonl_path),
        jsonl_flush_interval=3,
    )

    exporter.export_prepared_frame(_sample_payload(0), frame_index=0)
    exporter.export_prepared_frame(_sample_payload(1), frame_index=1)
    assert exporter._jsonl_pending_lines == 2
    assert exporter._jsonl_flush_count == 0

    exporter.export_prepared_frame(_sample_payload(2), frame_index=2)
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    exporter.close()

    assert exporter._jsonl_pending_lines == 0
    assert exporter._jsonl_flush_count == 1
    assert len(lines) == 3
    for line in lines:
        assert validate_frame_payload(json.loads(line)) == []


def test_default_config_exposes_exporter_tuning(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT.parent)

    cfg = load_config("configs/default.yaml")

    assert cfg["export_last_every_n_frames"] == 5
    assert cfg["jsonl_flush_interval"] == 10
