from __future__ import annotations

"""配置加载与路径解析。

配置文件里的相对路径统一解析到项目根目录，避免从仓库根目录或子项目目录
运行时得到不同输出位置。
"""

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""single-hand-teleop-baseline 子项目根目录。"""


def _resolve_config_path(path: str) -> Path:
    """解析配置文件路径。

    优先使用调用者当前目录下的路径；找不到时再回到项目根目录下查找。
    这样既支持在子项目目录运行，也支持从仓库根目录运行。
    """

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    return PROJECT_ROOT / candidate


def _require_positive_int(cfg: Dict[str, Any], key: str, errors: list[str]) -> None:
    if key not in cfg:
        return
    value = cfg[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{key} 必须是正整数")


def _require_positive_number(cfg: Dict[str, Any], key: str, errors: list[str]) -> None:
    if key not in cfg:
        return
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
        errors.append(f"{key} 必须是正数")


def _require_probability(cfg: Dict[str, Any], key: str, errors: list[str]) -> None:
    if key not in cfg:
        return
    value = cfg[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= 1.0
    ):
        errors.append(f"{key} 必须位于 [0, 1]")


def validate_config(cfg: Dict[str, Any]) -> None:
    """对会影响运行安全和 contract 的配置做启动前硬校验。"""

    errors: list[str] = []
    for key in (
        "display_width",
        "display_height",
        "max_num_hands",
        "recent_frames_buffer_size",
        "console_print_every_n_frames",
        "export_last_every_n_frames",
        "jsonl_flush_interval",
        "svh_mock_history_size",
    ):
        _require_positive_int(cfg, key, errors)
    for key in (
        "prediction_shadow_target_fps",
        "prediction_shadow_max_frame_gap_ms",
    ):
        _require_positive_number(cfg, key, errors)
    for key in (
        "min_detection_confidence",
        "min_tracking_confidence",
        "control_ready_min_in_bounds_ratio",
        "control_open_release_start_ratio",
        "control_open_release_full_ratio",
    ):
        _require_probability(cfg, key, errors)

    source_type = str(cfg.get("input_source_type", "webcam")).strip().lower()
    if source_type not in {"webcam", "video_file"}:
        errors.append("input_source_type 只能是 webcam 或 video_file")

    if bool(cfg.get("unity_udp_enabled", False)):
        host = str(cfg.get("unity_udp_host", "127.0.0.1")).strip().lower()
        if host != "localhost" and not host.startswith("127."):
            errors.append("Unity UDP 预览只允许 loopback 地址（127.0.0.1/localhost）")
    if "unity_udp_port" in cfg:
        port = cfg["unity_udp_port"]
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            errors.append("unity_udp_port 必须是 1..65535 的整数")

    if str(cfg.get("svh_transport", "mock")).strip().lower() != "mock":
        errors.append("当前已验证的 svh_transport 只能是 mock；真机传输不属于 Phase 1/1.5")

    layout = str(cfg.get("svh_preview_layout", "compact5"))
    expected_count = {"compact5": 5, "svh_9ch": 9}.get(layout)
    if expected_count is None:
        errors.append("svh_preview_layout 只能是 compact5 或 svh_9ch")
    elif "svh_preview_channel_count" in cfg:
        count = cfg["svh_preview_channel_count"]
        if count != expected_count:
            errors.append(
                f"svh_preview_channel_count 必须与 {layout} 对齐为 {expected_count}"
            )

    if bool(cfg.get("control_open_release_enabled", False)):
        start = cfg.get("control_open_release_start_ratio")
        full = cfg.get("control_open_release_full_ratio")
        if not isinstance(start, (int, float)) or isinstance(start, bool):
            errors.append("启用 open-release 时必须配置 control_open_release_start_ratio")
        elif not isinstance(full, (int, float)) or isinstance(full, bool):
            errors.append("启用 open-release 时必须配置 control_open_release_full_ratio")
        elif float(start) >= float(full):
            errors.append("control_open_release_start_ratio 必须小于 full_ratio")

    for open_key, closed_key in (
        ("control_grasp_open_ref", "control_grasp_closed_ref"),
        ("control_pinch_open_ref", "control_pinch_closed_ref"),
        ("control_hand_open_ratio_open_ref", "control_hand_open_ratio_closed_ref"),
        ("svh_position_open_value", "svh_position_closed_value"),
    ):
        if open_key in cfg and closed_key in cfg and cfg[open_key] == cfg[closed_key]:
            errors.append(f"{open_key} 与 {closed_key} 不能相等")

    for key in ("svh_9ch_open_ticks", "svh_9ch_closed_ticks"):
        if key in cfg:
            value = cfg[key]
            if not isinstance(value, list) or len(value) != 9 or any(
                not isinstance(item, int) or isinstance(item, bool) for item in value
            ):
                errors.append(f"{key} 必须恰好包含 9 个整数")

    output_path = cfg.get("output_json_path")
    if bool(cfg.get("save_last_json", True)) and isinstance(output_path, str) and output_path:
        resolved_output = Path(output_path).resolve()
        examples_dir = (PROJECT_ROOT / "examples").resolve()
        try:
            resolved_output.relative_to(examples_dir)
        except ValueError:
            pass
        else:
            errors.append("output_json_path 不能写入受版本控制的 examples/；请使用 outputs/")

    if errors:
        raise ValueError("配置校验失败：" + "; ".join(errors))


def load_config(path: str) -> Dict[str, Any]:
    """读取 yaml 配置，并规范化常用输出路径。"""

    config_path = _resolve_config_path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件根节点必须是 YAML 对象：{config_path}")

    for key in (
        "video_file_path",
        "output_json_path",
        "jsonl_output_dir",
        "prediction_shadow_output_json_path",
        "prediction_shadow_jsonl_output_dir",
        "prediction_shadow_selection_path",
        "prediction_shadow_checkpoint_path",
        "prediction_shadow_report_path",
    ):
        value = cfg.get(key)
        if isinstance(value, str) and value:
            path_value = Path(value)
            if not path_value.is_absolute():
                # 这些路径是运行产物位置，固定到项目根目录能减少“从哪里启动就写到哪里”的混乱。
                cfg[key] = str(PROJECT_ROOT / path_value)

    validate_config(cfg)

    return cfg
