from __future__ import annotations

"""配置加载与路径解析。

配置文件里的相对路径统一解析到项目根目录，避免从仓库根目录或子项目目录
运行时得到不同输出位置。
"""

import math
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
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or float(value) <= 0.0:
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


def _validate_frame_gap(cfg: Dict[str, Any], errors: list[str]) -> None:
    """断流阈值 = min(上限, max(下限, 倍数×近期中位周期))；上限不得压过下限。"""
    _require_positive_int(cfg, "mapping_frame_gap_period_window", errors)
    _require_positive_number(cfg, "mapping_frame_gap_ceiling_ms", errors)
    if "mapping_frame_gap_period_factor" in cfg:
        factor = cfg["mapping_frame_gap_period_factor"]
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor < 1:
            errors.append("mapping_frame_gap_period_factor 必须是不小于 1 的有限数")
    floor = cfg.get("mapping_max_frame_gap_ms", 100.0)
    ceiling = cfg.get("mapping_frame_gap_ceiling_ms", 1000.0)
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (floor, ceiling))
    if numbers and ceiling < floor:
        errors.append("mapping_frame_gap_ceiling_ms 不能小于 mapping_max_frame_gap_ms")


def validate_config(cfg: Dict[str, Any]) -> None:
    """对会影响运行安全和 contract 的配置做启动前硬校验。"""

    errors: list[str] = []
    stage = cfg.get("camera_mapping_stage", 4)
    if type(stage) is not int or stage not in (4, 5, 6, 7, 8, 9, 10):
        errors.append("camera_mapping_stage 当前只支持 4/5/6/7/8/9/10")
    elif stage >= 5:
        if cfg.get("geometry_mode") != "image_width_xyz_v1" or cfg.get("svh_preview_layout") != "svh_9ch":
            errors.append("新映射只接受 M3-A 几何与 svh_9ch 布局")
        if cfg.get("feature_curl_mode") != "per_finger_tip_v1":
            errors.append("新映射必须显式选择 feature_curl_mode=per_finger_tip_v1")
        fingers = {"thumb", "index", "middle", "ring", "little"}
        opened, closed = cfg.get("finger_tip_open_refs"), cfg.get("finger_tip_closed_refs")
        if not all(isinstance(values, dict) and set(values) == fingers for values in (opened, closed)):
            errors.append("逐指参考必须明确 finger_tip_open_refs/finger_tip_closed_refs 的五根手指")
        else:
            for finger in fingers:
                a, b = opened[finger], closed[finger]
                if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (a, b)) or not b < a:
                    errors.append(f"{finger} 的逐指参考须满足有限正数 closed < open")
    elif cfg.get("feature_curl_mode", "legacy") != "legacy":
        errors.append("feature_curl_mode 的新语义必须选择新的 camera_mapping_stage")
    if type(stage) is int and stage >= 6:
        if cfg.get("control_open_release_mode") != "none" or cfg.get("control_open_release_enabled", False):
            errors.append("M2 新版本须显式关闭全局释放并选择 control_open_release_mode=none")
    elif cfg.get("control_open_release_mode") == "none":
        errors.append("关闭全局释放的新语义须选择 camera_mapping_stage>=6")
    if type(stage) is int and stage >= 7:
        if cfg.get("pinch_distance_mode") != "max_2d_geometry3d_v1":
            errors.append("M7 新版本必须显式选择 max_2d_geometry3d_v1 捏合距离")
    elif cfg.get("pinch_distance_mode", "image_2d") != "image_2d":
        errors.append("三维捏合排除规则必须选择 camera_mapping_stage>=7")
    if type(stage) is int and stage >= 8:
        if cfg.get("svh_mapping_transition_mode") != "continuous_blend_v1":
            errors.append("M6 新版本必须显式选择 continuous_blend_v1")
        start, full = cfg.get("svh_pinch_blend_start"), cfg.get("svh_pinch_blend_full")
        if (not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in (start, full))
                or full - start < 0.4 - 1e-9):
            errors.append("捏合混合区间须在 [0,1] 内且宽度至少 0.4，以限制扫描单步变化")
    elif cfg.get("svh_mapping_transition_mode", "discrete") != "discrete":
        errors.append("连续混合规则必须选择 camera_mapping_stage>=8")
    if type(stage) is int and stage >= 9:
        if cfg.get("control_finger_range_mode") != "long_finger_v1":
            errors.append("M8 新版本必须显式选择 long_finger_v1")
        weight = cfg.get("control_grasp_flex_weight")
        if type(weight) not in (int, float) or not math.isfinite(weight) or not 0 <= weight <= 1:
            errors.append("control_grasp_flex_weight 必须位于 [0,1]")
        fingers = {"index", "middle", "ring", "little"}
        opened, closed = cfg.get("control_finger_open_refs"), cfg.get("control_finger_closed_refs")
        if not all(isinstance(v, dict) and set(v) == fingers for v in (opened, closed)):
            errors.append("四指控制端点必须明确 control_finger_open_refs/control_finger_closed_refs")
        else:
            for name in fingers:
                a, b = opened[name], closed[name]
                if not all(type(v) in (int, float) and math.isfinite(v) for v in (a, b)) or not 0 <= a < b <= 1:
                    errors.append(f"{name} 的控制端点须满足 0 <= open < closed <= 1")
    elif cfg.get("control_finger_range_mode", "legacy") != "legacy":
        errors.append("四指新端点必须选择 camera_mapping_stage>=9")
    elif "control_grasp_flex_weight" in cfg:
        errors.append("新握紧度配比必须选择 camera_mapping_stage>=9")
    filter_mode = cfg.get("control_flex_filter_mode", "none")
    if filter_mode not in {"none", "one_euro_v1"}:
        errors.append("control_flex_filter_mode 只能是 none 或 one_euro_v1")
    if filter_mode == "one_euro_v1":
        if type(stage) is not int or stage < 9:
            errors.append("自适应 flex 滤波须显式选择 camera_mapping_stage>=9")
    if type(stage) is int and stage >= 9:
        if cfg.get("control_flex_filter_mode") not in {"none", "one_euro_v1"}:
            errors.append("M8 必须明确 control_flex_filter_mode，不能依赖隐式缺省")
        for key in ("control_flex_filter_min_cutoff_hz", "control_flex_filter_derivative_cutoff_hz"):
            value = cfg.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                errors.append(f"{key} 必须是有限正数")
        beta = cfg.get("control_flex_filter_beta")
        if type(beta) not in (int, float) or not math.isfinite(beta) or beta < 0:
            errors.append("control_flex_filter_beta 必须是有限非负数")
    thumb_keys = ("thumb_tip_proximity_weight", "svh_thumb_control_mode", "svh_thumb_grasp_open_ref",
                  "svh_thumb_grasp_closed_ref", "svh_thumb_grasp_guide_weight")
    if type(stage) is int and stage >= 10:
        if cfg.get("svh_thumb_control_mode") != "calibrated_grasp_v1":
            errors.append("v10 必须显式选择 svh_thumb_control_mode=calibrated_grasp_v1")
        for key in ("thumb_tip_proximity_weight", "svh_thumb_grasp_guide_weight"):
            value = cfg.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                errors.append(f"{key} 必须位于 [0,1]")
        a, b = cfg.get("svh_thumb_grasp_open_ref"), cfg.get("svh_thumb_grasp_closed_ref")
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (a, b)) or not 0 <= a < b <= 1:
            errors.append("拇指抓握引导端点须满足 0 <= open < closed <= 1")
    elif any(key in cfg for key in thumb_keys):
        errors.append("拇指新参考/配比语义必须选择 camera_mapping_stage>=10")
    for key in (
        "display_width",
        "display_height",
        "max_num_hands",
        "console_print_every_n_frames",
        "export_last_every_n_frames",
        "jsonl_flush_interval",
    ):
        _require_positive_int(cfg, key, errors)
    for key in (
        "prediction_shadow_target_fps",
        "prediction_shadow_max_frame_gap_ms",
        "geometry_min_palm_size",
        "geometry_min_bone_palm_ratio",
        "control_release_curl_half_width",
        "control_release_pinch_band",
        "mapping_max_frame_gap_ms",
        # hand_open_ratio 不是概率，真实张开手常大于 1；释放阈值只要求有限正数。
        "control_open_release_start_ratio",
        "control_open_release_full_ratio",
    ):
        _require_positive_number(cfg, key, errors)
    for key in (
        "min_detection_confidence",
        "min_tracking_confidence",
        "control_ready_min_in_bounds_ratio",
        "control_release_hysteresis",
    ):
        _require_probability(cfg, key, errors)
    _validate_frame_gap(cfg, errors)

    source_type = str(cfg.get("input_source_type", "webcam")).strip().lower()
    if cfg.get("geometry_mode", "legacy_image_xyz") not in {"legacy_image_xyz", "image_width_xyz_v1"}:
        errors.append("geometry_mode 必须是 legacy_image_xyz 或 image_width_xyz_v1")
    if cfg.get("control_open_release_mode", "legacy") not in {"legacy", "continuous_v1", "none"}:
        errors.append("control_open_release_mode 必须是 legacy、continuous_v1 或 none")
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


def _load_yaml(config_path: Path, seen: set[Path]) -> Dict[str, Any]:
    config_path = config_path.resolve()
    if config_path in seen:
        raise ValueError(f"配置继承形成循环：{config_path}")
    seen.add(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件根节点必须是 YAML 对象：{config_path}")
    parent = cfg.pop("extends", None)
    if parent is not None:
        if not isinstance(parent, str) or not parent.strip():
            raise ValueError("extends 必须是基础配置文件路径")
        cfg = {**_load_yaml(config_path.parent / parent, seen), **cfg}
    return cfg


def load_config(path: str) -> Dict[str, Any]:
    """读取配置；extends 相对当前 YAML，其余文件路径相对子项目根目录。"""

    cfg = _load_yaml(_resolve_config_path(path), set())

    for key in (
        "video_file_path",
        "output_json_path",
        "jsonl_output_dir",
        "prediction_shadow_model_path",
        "prediction_shadow_output_json_path",
        "prediction_shadow_jsonl_output_dir",
    ):
        value = cfg.get(key)
        if isinstance(value, str) and value:
            path_value = Path(value)
            if not path_value.is_absolute():
                # 这些路径是运行产物位置，固定到项目根目录能减少“从哪里启动就写到哪里”的混乱。
                cfg[key] = str(PROJECT_ROOT / path_value)

    validate_config(cfg)

    return cfg
