"""摄像头坐标声明与最低可用性检查；不把单目相对深度当作度量真值。"""
from __future__ import annotations

import numpy as np

from perception.base import HandDetection
from perception.landmark_quality import assess_control_readiness

GEOMETRY_TASK = "camera_mediapipe_geometry_xyz_v1"
LEGACY_TASK = "legacy_mediapipe_image_xyz"
FINGER_BONES = [(i, i + 1) for start in (1, 5, 9, 13, 17) for i in range(start, start + 3)]


def prepare_geometry(detection: HandDetection | None, cfg: dict, *, frame=None) -> dict:
    """先检查结构与有限值，再转换/检查尺度，最后检查原始图像边界。

    外部输入错误以 reason 返回，调用方不能先提特征再做这一步。
    geometry_landmarks 只包含验证过的有限数组；失效帧不能消费控制量。
    """
    corrected = cfg.get("geometry_mode", "legacy_image_xyz") == "image_width_xyz_v1"
    result = {
        "task_id": GEOMETRY_TASK if corrected else LEGACY_TASK,
        "image_width": None, "image_height": None,
        "input_valid": False, "reason": "no_right_hand", "geometry_landmarks": [],
    }
    if detection is None:
        return result

    def invalid(reason):
        result["reason"] = reason
        return result

    try:
        xy = np.asarray(detection.landmarks_2d, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return invalid("invalid_xy_shape")
    if xy.shape != (21, 2):
        return invalid("invalid_xy_shape")
    try:
        xyz = np.asarray(detection.landmarks_xyz, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return invalid("invalid_xyz_shape")
    if xyz.shape != (21, 3):
        return invalid("invalid_xyz_shape")
    # 不把字符串/对象数组默认为数值关键点，避免后续几何收到混合类型。
    if any(np.asarray(value).dtype.kind not in "iuf" for value in (detection.landmarks_2d, detection.landmarks_xyz)):
        return invalid("nonnumeric_landmarks")
    if not np.isfinite(xy).all() or not np.isfinite(xyz).all():
        return invalid("nonfinite_landmarks")
    if corrected and detection.coordinate_space is None:
        return invalid("missing_coordinate_space")
    if detection.coordinate_space not in (None, "mediapipe_image_xyz"):
        return invalid("unsupported_coordinate_space")
    if not np.allclose(xy, xyz[:, :2], rtol=0, atol=1e-7):
        return invalid("inconsistent_xy_xyz")
    try:
        confidence = float(detection.confidence)
    except (TypeError, ValueError, OverflowError):
        return invalid("invalid_confidence")
    if isinstance(detection.confidence, bool) or not np.isfinite(confidence) or not 0 <= confidence <= 1:
        return invalid("invalid_confidence")

    width, height = detection.image_width, detection.image_height
    if corrected and frame is not None:
        actual_height, actual_width = frame.shape[:2]
        if (width is not None and width != actual_width) or (height is not None and height != actual_height):
            return invalid("image_size_mismatch")
        width, height = int(actual_width), int(actual_height)
    if width is not None or height is not None or corrected:
        if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (width, height)):
            return invalid("missing_or_invalid_image_size")
        result.update(image_width=width, image_height=height)
    geometry = xyz.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        if corrected:
            geometry[:, 1] *= height / width
        primary = float(np.linalg.norm(geometry[0, :2] - geometry[9, :2]))
        palm_size = primary if primary > 1e-6 else float(np.linalg.norm(geometry[5, :2] - geometry[17, :2]))
        lengths = np.array([np.linalg.norm(geometry[a] - geometry[b]) for a, b in FINGER_BONES])
    if not np.isfinite(geometry).all() or not np.isfinite(palm_size) or not np.isfinite(lengths).all():
        return invalid("nonfinite_geometry")
    if palm_size < float(cfg.get("geometry_min_palm_size", 1e-4)):
        return invalid("degenerate_palm")
    minimum_bone = max(1e-6, palm_size * float(cfg.get("geometry_min_bone_palm_ratio", 1e-3)))
    if np.any(lengths <= minimum_bone):
        return invalid("degenerate_bone")
    result["geometry_landmarks"] = geometry.tolist()
    if not assess_control_readiness(xy, cfg)["control_ready"]:
        return invalid("out_of_bounds")
    result.update(input_valid=True, reason="ok")
    return result
