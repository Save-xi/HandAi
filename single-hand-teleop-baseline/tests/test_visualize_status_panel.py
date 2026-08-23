import numpy as np

from output.frame_payload_contract import normalize_frame_payload
from visualize.status_panel import build_status_panel


def _landmarks_2d() -> list[list[float]]:
    return [[0.1 + index * 0.01, 0.2 + index * 0.01] for index in range(21)]


def _landmarks_3d() -> list[list[float]]:
    return [[x, y, -index * 0.001] for index, (x, y) in enumerate(_landmarks_2d())]


def _sample_payload() -> dict:
    return normalize_frame_payload(
        {
            "timestamp": 1.0,
            "frame_index": 0,
            "detected": True,
            "handedness": "Right",
            "confidence": 0.98,
            "control_ready": True,
            "gesture_raw": "pinch",
            "gesture_stable": "pinch",
            "pinch_distance_norm": 0.16,
            "hand_open_ratio": 0.81,
            "finger_curl": {"thumb": 0.03, "index": 0.22, "middle": 0.01, "ring": 0.01, "little": 0.02},
            "landmarks_2d": _landmarks_2d(),
            "landmarks_3d": _landmarks_3d(),
            "control_representation": {
                "valid": True,
                "features_valid": True,
                "command_ready": True,
                "source": "features",
                "gesture_context": "pinch",
                "preferred_mapping": "pinch",
                "grasp_close": 0.2,
                "thumb_index_proximity": 0.7,
                "effective_pinch_strength": 0.7,
                "pinch_strength": 0.7,
                "support_flex": 0.05,
                "finger_flex": {"thumb": 0.03, "index": 0.22, "middle": 0.01, "ring": 0.01, "little": 0.02},
            },
            "svh_preview": {
                "enabled": True,
                "mode": "preview",
                "valid": True,
                "command_source": "control_representation",
                "target_channels": list(range(9)),
                "target_positions": [0.6, 0.5, 0.52, 0.48, 0.03, 0.03, 0.02, 0.02, 0.08],
                "target_ticks_preview": [0] * 9,
                "protocol_hint": {
                    "set_control_state_addr": "0x09",
                    "set_all_channels_addr": "0x03",
                    "transport": "mock",
                    "channel_layout": "svh_9ch",
                    "channel_order": "thumb_flexion,thumb_opposition,index_finger_distal,index_finger_proximal,middle_finger_distal,middle_finger_proximal,ring_finger,pinky,finger_spread",
                    "position_units": "normalized_preview",
                    "target_tick_units": "encoder_ticks_preview",
                },
            },
            "fps": 30.0,
            "latency_ms": 12.0,
        },
        include_deprecated_aliases=False,
    )


def test_build_status_panel_renders_non_empty_panel():
    panel = build_status_panel(480, 420, _sample_payload())

    assert panel.shape == (480, 420, 3)
    assert panel.dtype == np.uint8
    assert int(panel.sum()) > 0
