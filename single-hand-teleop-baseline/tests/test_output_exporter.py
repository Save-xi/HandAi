import json
import socket
from pathlib import Path
from types import SimpleNamespace

from output.frame_payload_contract import normalize_frame_payload, prepare_frame_payload, validate_frame_payload
from output.json_exporter import JsonExporter
from main import _drain_prediction_results, _emit_prepared_debug_outputs, _send_prepared_udp
from main import _build_jsonl_session_path
from prediction.shadow_predictor import PredictionShadow
from utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _landmarks_2d() -> list[list[float]]:
    return [[0.1 + index * 0.01, 0.2 + index * 0.01] for index in range(21)]


def _landmarks_3d() -> list[list[float]]:
    return [[x, y, -index * 0.001] for index, (x, y) in enumerate(_landmarks_2d())]


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
            "landmarks_2d": _landmarks_2d(),
            "landmarks_3d": _landmarks_3d(),
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
    assert list(tmp_path.glob("*.tmp")) == []


def test_session_paths_do_not_collide_within_same_second(tmp_path):
    cfg = {"jsonl_output_dir": str(tmp_path)}

    first = _build_jsonl_session_path(cfg)
    second = _build_jsonl_session_path(cfg)

    assert first != second
    assert Path(first).parent == tmp_path
    assert Path(second).parent == tmp_path


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
    assert cfg["unity_udp_enabled"] is False
    assert cfg["unity_udp_host"] == "127.0.0.1"
    assert cfg["unity_udp_port"] == 18080
    assert cfg["prediction_shadow_enabled"] is False
    assert cfg["prediction_shadow_horizon_ms"] == [50, 100, 150]


def test_exporter_constructor_keeps_unity_udp_disabled_by_default(tmp_path):
    exporter = JsonExporter(str(tmp_path / "last.json"), save_last_json=False)
    try:
        assert exporter.unity_udp_enabled is False
        assert exporter._unity_udp_socket is None
        exporter.send_prepared_frame(_sample_payload(0))
        assert exporter._unity_udp_socket is None
        assert exporter._unity_udp_send_count == 0
    finally:
        exporter.close()


def test_send_prepared_frame_can_broadcast_over_unity_udp(tmp_path):
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    _, port = receiver.getsockname()

    exporter = JsonExporter(
        str(tmp_path / "last.json"),
        save_last_json=False,
        unity_udp_enabled=True,
        unity_udp_host="127.0.0.1",
        unity_udp_port=port,
    )

    payload = _sample_payload(7)
    exporter.send_prepared_frame(payload)
    raw, _ = receiver.recvfrom(65535)
    exporter.close()
    receiver.close()

    received = json.loads(raw.decode("utf-8"))
    assert received["frame_index"] == 7
    assert validate_frame_payload(received) == []


def test_actual_udp_and_baseline_jsonl_stay_frozen_while_prediction_uses_separate_jsonl(tmp_path):
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    _, port = receiver.getsockname()
    baseline_jsonl_path = tmp_path / "baseline-session.jsonl"
    prediction_jsonl_path = tmp_path / "prediction-session.jsonl"
    baseline_exporter = JsonExporter(
        str(tmp_path / "last.json"),
        save_last_json=False,
        jsonl_path=str(baseline_jsonl_path),
        unity_udp_enabled=True,
        unity_udp_host="127.0.0.1",
        unity_udp_port=port,
    )
    prediction_exporter = JsonExporter(
        str(tmp_path / "latest-prediction.json"),
        save_last_json=False,
        jsonl_path=str(prediction_jsonl_path),
        unity_udp_enabled=False,
    )
    payload = _sample_payload(8)

    _send_prepared_udp(payload, exporter=baseline_exporter)
    _emit_prepared_debug_outputs(
        payload,
        frame_index=8,
        exporter=baseline_exporter,
        print_json=False,
        print_every_n_frames=1,
        landmarks_preview_count=2,
    )
    prediction_payload = dict(payload)
    prediction_payload["prediction_diagnostics"] = PredictionShadow.unavailable(
        "test-only unavailable model"
    ).observe(payload)
    prediction_payload = prepare_frame_payload(
        prediction_payload,
        include_deprecated_aliases=False,
    )
    worker = SimpleNamespace(drain_results=lambda: [prediction_payload])
    assert _drain_prediction_results(worker, prediction_exporter) == 1  # type: ignore[arg-type]
    raw, _ = receiver.recvfrom(65535)
    baseline_exporter.close()
    prediction_exporter.close()
    receiver.close()

    udp_payload = json.loads(raw.decode("utf-8"))
    baseline_jsonl_payload = json.loads(baseline_jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    prediction_jsonl_payload = json.loads(prediction_jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert "prediction_diagnostics" not in udp_payload
    assert "prediction_diagnostics" not in baseline_jsonl_payload
    assert prediction_jsonl_payload["prediction_diagnostics"]["status"] == "initialization_error"
    assert udp_payload["svh_preview"] == prediction_jsonl_payload["svh_preview"]
    assert validate_frame_payload(udp_payload) == []
    assert validate_frame_payload(baseline_jsonl_payload) == []
    assert validate_frame_payload(prediction_jsonl_payload) == []
