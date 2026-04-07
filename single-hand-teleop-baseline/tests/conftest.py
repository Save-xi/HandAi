from __future__ import annotations

import pytest


@pytest.fixture
def gesture_cfg():
    return {
        "pinch_distance_norm_threshold": 0.45,
        "pinch_open_ratio_min": 0.75,
        "pinch_support_curl_max": 0.65,
        "open_ratio_threshold": 0.85,
        "open_mean_curl_max": 0.45,
        "fist_ratio_threshold": 0.85,
        "fist_mean_curl_min": 0.45,
        "fist_compact_ratio_threshold": 0.65,
        "stable_gesture_window": 5,
        "stable_gesture_min_consecutive": 2,
        "stable_unknown_consecutive": 1,
    }


@pytest.fixture
def control_cfg():
    return {
        "control_grasp_open_ref": 0.02,
        "control_grasp_closed_ref": 0.55,
        "control_pinch_open_ref": 0.45,
        "control_pinch_closed_ref": 0.08,
        "control_hand_open_ratio_open_ref": 0.95,
        "control_hand_open_ratio_closed_ref": 0.25,
        "control_pinch_index_open_ref": 0.05,
        "control_pinch_index_closed_ref": 0.35,
    }


@pytest.fixture
def svh_cfg():
    return {
        "svh_enable_preview": True,
        "svh_enable_gesture_fallback": False,
        "svh_preview_layout": "compact5",
        "svh_preview_channel_count": 5,
        "svh_preview_mode": "preview",
        "svh_transport": "mock",
        "svh_protocol_sync_bytes": [76, 170],
        "svh_grasp_open_ref": 0.02,
        "svh_grasp_closed_ref": 0.55,
        "svh_pinch_open_ref": 0.45,
        "svh_pinch_closed_ref": 0.08,
        "svh_hand_open_ratio_open_ref": 0.95,
        "svh_hand_open_ratio_closed_ref": 0.25,
        "svh_position_open_value": 0.0,
        "svh_position_closed_value": 1.0,
        "svh_thumb_grasp_scale": 0.85,
        "svh_thumb_opposition_scale": 0.75,
        "svh_pinch_support_scale": 0.20,
        "svh_open_spread_scale": 0.25,
        "svh_grasp_spread_scale": 0.05,
        "svh_pinch_spread_scale": 0.10,
        "svh_pinch_index_open_ref": 0.05,
        "svh_pinch_index_closed_ref": 0.35,
    }


@pytest.fixture
def synthetic_hand_pose():
    def _make(kind: str = "open"):
        xyz = [
            (0.0, 0.0, 0.0),  # 手腕
            (-1.0, 0.5, 0.0),  # 拇指 CMC
            (-1.4, 1.0, 0.0),  # 拇指 MCP
            (-1.8, 1.4, 0.0),  # 拇指 IP
            (-2.2, 1.8, 0.0),  # 拇指指尖
            (-0.8, 1.0, 0.0),  # 食指 MCP
            (-0.8, 2.0, 0.0),  # 食指 PIP
            (-0.8, 3.0, 0.0),  # 食指 DIP
            (-0.8, 4.0, 0.0),  # 食指指尖
            (0.0, 1.0, 0.0),  # 中指 MCP
            (0.0, 2.1, 0.0),  # 中指 PIP
            (0.0, 3.2, 0.0),  # 中指 DIP
            (0.0, 4.3, 0.0),  # 中指指尖
            (0.8, 1.0, 0.0),  # 无名指 MCP
            (0.8, 2.0, 0.0),  # 无名指 PIP
            (0.8, 3.0, 0.0),  # 无名指 DIP
            (0.8, 4.0, 0.0),  # 无名指指尖
            (1.6, 1.0, 0.0),  # 小指 MCP
            (1.6, 1.9, 0.0),  # 小指 PIP
            (1.6, 2.8, 0.0),  # 小指 DIP
            (1.6, 3.7, 0.0),  # 小指指尖
        ]

        if kind == "fist":
            xyz[2] = (-1.2, 0.8, 0.0)
            xyz[3] = (-0.7, 0.45, -0.20)
            xyz[4] = (-0.1, 0.25, -0.40)
            curled_coords = {
                6: (-0.8, 1.45, 0.0),
                7: (-0.15, 1.25, -0.25),
                8: (0.2, 0.72, -0.45),
                10: (0.0, 1.55, 0.0),
                11: (0.65, 1.35, -0.20),
                12: (0.95, 0.78, -0.45),
                14: (0.8, 1.48, 0.0),
                15: (1.25, 1.28, -0.18),
                16: (1.42, 0.80, -0.38),
                18: (1.55, 1.35, 0.0),
                19: (1.9, 1.12, -0.12),
                20: (2.0, 0.72, -0.28),
            }
            for idx, point in curled_coords.items():
                xyz[idx] = point
        elif kind == "pinch":
            xyz[2] = (-1.15, 0.95, 0.0)
            xyz[3] = (-0.75, 1.45, -0.08)
            xyz[4] = (-0.22, 1.95, -0.18)
            xyz[6] = (-0.72, 1.55, -0.02)
            xyz[7] = (-0.52, 1.82, -0.10)
            xyz[8] = (-0.18, 1.98, -0.18)
            xyz[10] = (0.0, 2.05, 0.0)
            xyz[11] = (0.0, 3.08, -0.02)
            xyz[12] = (0.0, 4.05, -0.02)
            xyz[14] = (0.82, 1.95, 0.0)
            xyz[15] = (0.82, 2.95, -0.02)
            xyz[16] = (0.84, 3.85, -0.02)
            xyz[18] = (1.62, 1.85, 0.0)
            xyz[19] = (1.65, 2.70, -0.02)
            xyz[20] = (1.70, 3.45, -0.02)

        xy = [(x, y) for x, y, _ in xyz]
        return xy, xyz

    return _make
