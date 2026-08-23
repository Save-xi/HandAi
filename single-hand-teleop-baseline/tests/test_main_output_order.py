from __future__ import annotations

from main import _emit_prepared_frame
from output.frame_payload_contract import normalize_frame_payload


def _payload() -> dict:
    landmarks_2d = [[0.1 + index * 0.01, 0.2 + index * 0.01] for index in range(21)]
    return normalize_frame_payload(
        {
            "timestamp": 1000.0,
            "frame_index": 0,
            "detected": True,
            "handedness": "Right",
            "confidence": 0.95,
            "control_ready": False,
            "gesture_raw": "unknown",
            "gesture_stable": "unknown",
            "pinch_distance_norm": None,
            "hand_open_ratio": None,
            "finger_curl": {
                "thumb": None,
                "index": None,
                "middle": None,
                "ring": None,
                "little": None,
            },
            "landmarks_2d": landmarks_2d,
            "landmarks_3d": [[x, y, 0.0] for x, y in landmarks_2d],
            "control_representation": {
                "valid": False,
                "features_valid": False,
                "command_ready": False,
                "source": None,
                "gesture_context": None,
                "preferred_mapping": None,
                "grasp_close": None,
                "thumb_index_proximity": None,
                "effective_pinch_strength": None,
                "pinch_strength": None,
                "support_flex": None,
                "finger_flex": {
                    "thumb": None,
                    "index": None,
                    "middle": None,
                    "ring": None,
                    "little": None,
                },
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
                    "channel_layout": "svh_9ch",
                    "channel_order": "thumb_flexion,thumb_opposition,index_finger_distal,index_finger_proximal,middle_finger_distal,middle_finger_proximal,ring_finger,pinky,finger_spread",
                    "position_units": "normalized_preview",
                    "target_tick_units": "none",
                },
            },
            "fps": 30.0,
            "latency_ms": 10.0,
            "timing": {
                "schema_version": 1,
                "clock": "unix_epoch_ms",
                "source_read_start_unix_ms": 1_000_000.0,
                "source_read_end_unix_ms": 1_000_001.0,
                "detection_end_unix_ms": 1_000_005.0,
                "baseline_end_unix_ms": 1_000_006.0,
                "preview_end_unix_ms": 1_000_007.0,
                "payload_ready_unix_ms": 1_000_008.0,
                "udp_send_attempt_unix_ms": None,
            },
        },
        include_deprecated_aliases=False,
    )


class _RecordingExporter:
    unity_udp_enabled = True

    def __init__(self) -> None:
        self.events: list[str] = []

    def send_prepared_frame(self, payload: dict) -> None:
        assert payload["timing"]["udp_send_attempt_unix_ms"] is not None
        self.events.append("udp")

    def print_console(self, payload: dict, landmarks_preview_count: int) -> None:
        assert landmarks_preview_count == 2
        self.events.append("console")

    def export_prepared_frame(self, payload: dict, *, frame_index: int) -> None:
        assert frame_index == 0
        self.events.append("disk")


def test_emit_frame_prioritizes_udp_over_console_and_disk():
    exporter = _RecordingExporter()
    payload = _payload()

    _emit_prepared_frame(
        payload,
        frame_index=0,
        exporter=exporter,  # type: ignore[arg-type]
        print_json=True,
        print_every_n_frames=1,
        landmarks_preview_count=2,
    )

    assert exporter.events == ["udp", "console", "disk"]
