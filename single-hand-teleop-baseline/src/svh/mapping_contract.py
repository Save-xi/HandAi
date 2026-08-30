from __future__ import annotations

"""H2O 标签、训练 checkpoint 与实时 SVH 9 通道映射的语义指纹。"""

import hashlib
import json
from typing import Any, Dict

from svh.svh_layout import SVH_9CH_NAMES


MAPPING_CONTRACT_VERSION = "svh9-label-v2-open-release"

# 只纳入会改变 9 通道 normalized target 的参数；输出路径、UDP 端口、日志和
# 推理开关不属于标签语义，修改它们不应让模型失效。
_MAPPING_DEFAULTS: Dict[str, Any] = {
    "pinch_distance_norm_threshold": 0.45,
    "pinch_open_ratio_min": 0.75,
    "pinch_support_curl_max": 0.65,
    "open_ratio_threshold": 0.85,
    "open_mean_curl_max": 0.45,
    "fist_ratio_threshold": 0.85,
    "fist_mean_curl_min": 0.45,
    "fist_compact_ratio_threshold": 0.65,
    "control_grasp_open_ref": 0.02,
    "control_grasp_closed_ref": 0.55,
    "control_pinch_open_ref": 0.45,
    "control_pinch_closed_ref": 0.08,
    "control_hand_open_ratio_open_ref": 0.95,
    "control_hand_open_ratio_closed_ref": 0.25,
    "control_pinch_index_open_ref": 0.05,
    "control_pinch_index_closed_ref": 0.35,
    "control_open_release_enabled": False,
    "control_open_release_start_ratio": 0.85,
    "control_open_release_full_ratio": 0.95,
    "svh_enable_gesture_fallback": False,
    "svh_preview_layout": "svh_9ch",
    "svh_preview_channel_count": 9,
    "svh_position_open_value": 0.0,
    "svh_position_closed_value": 1.0,
    "svh_thumb_grasp_scale": 0.85,
    "svh_thumb_opposition_scale": 0.75,
    "svh_pinch_support_scale": 0.20,
    "svh_open_spread_scale": 0.25,
    "svh_grasp_spread_scale": 0.05,
    "svh_pinch_spread_scale": 0.10,
}


def mapping_contract_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """返回可序列化的有效映射语义，不包含运行期无关配置。"""

    return {
        "version": MAPPING_CONTRACT_VERSION,
        "single_right_hand": True,
        "channel_order": list(SVH_9CH_NAMES),
        "parameters": {
            key: cfg.get(key, default)
            for key, default in sorted(_MAPPING_DEFAULTS.items())
        },
    }


def mapping_contract_sha256(cfg: Dict[str, Any]) -> str:
    encoded = json.dumps(
        mapping_contract_payload(cfg),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
