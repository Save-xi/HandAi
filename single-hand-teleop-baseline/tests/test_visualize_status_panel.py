import numpy as np

from output.frame_payload_contract import normalize_frame_payload
from pipeline import HandPipeline
from utils.config import load_config
from visualize.status_panel import build_status_panel, build_status_rows


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


def test_status_rows_show_input_reason_and_release_weight_or_none():
    payload = _sample_payload()
    rows = build_status_rows(payload)
    assert "输入: 无" in rows and "释放权重: 无" in rows
    payload["input_diagnostics"] = {"reason": "out_of_bounds", "release_weight": 0.25}
    rows = build_status_rows(payload)
    assert "输入: out_of_bounds" in rows and "释放权重: 0.250" in rows


def test_pipeline_failure_reason_fits_on_720_high_panel():
    payload = HandPipeline(load_config("configs/ai_m3a.yaml")).process_detections([], frame_index=0, timestamp=1.0)
    rows = build_status_rows(payload)
    assert "输入: no_right_hand" in rows and "释放权重: 0.000" in rows
    # 首行 y=18、行距 30px；最后一行须整行落在 720 高画面内并实际绘出。
    assert 18 + len(rows) * 30 <= 720
    panel = build_status_panel(720, 426, payload)
    assert panel[18 + (len(rows) - 1) * 30:].any()
