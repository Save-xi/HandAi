from __future__ import annotations

"""把 H2O pose-only 右手序列转换为当前项目的 9 通道预览序列。"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from control.control_representation import build_control_representation
from features.hand_features import extract_hand_features
from gesture.rule_based_gesture import infer_gesture_raw
from svh.svh_adapter import build_svh_command_preview
from svh.mapping_contract import (
    H2O_LABEL_GESTURE_CONTEXT_POLICY,
    MAPPING_CONTRACT_VERSION,
    RUNTIME_GESTURE_CONTEXT_POLICY,
    assert_mapping_implementation_compatible,
    mapping_contract_sha256,
)
from utils.config import load_config


H2O_POSE_VALUE_COUNT = 128
H2O_RIGHT_VALID_INDEX = 64
H2O_RIGHT_POINTS_START = 65
HAND_LANDMARK_COUNT = 21
SVH_CHANNEL_COUNT = 9
H2O_DEFAULT_FPS = 30.0
POSE_ONLY_PROJECTION_LEGACY = "legacy_camera_xy_wrist_origin_palm_scale"
POSE_ONLY_PROJECTION_NORMALIZED_PERSPECTIVE = "normalized_perspective_wrist_origin_palm_scale"


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: float
    height: float


@dataclass(frozen=True)
class H2OTake:
    subject: str
    action: str
    take: str
    cam_dir: Path
    split: str

    @property
    def identifier(self) -> str:
        return f"{self.subject}_{self.action}_{self.take}_cam4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_h2o_right_hand_pose(path: Path) -> tuple[bool, np.ndarray]:
    """读取 H2O hand_pose 文本中的右手有效位和 21x3 相机坐标。"""

    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ", dtype=np.float64)
    if values.size != H2O_POSE_VALUE_COUNT:
        raise ValueError(f"{path} 应包含 {H2O_POSE_VALUE_COUNT} 个数，实际为 {values.size}")

    valid_value = float(values[H2O_RIGHT_VALID_INDEX])
    if not math.isfinite(valid_value):
        raise ValueError(f"{path} 的右手有效位不是有限数")

    points = values[H2O_RIGHT_POINTS_START:].reshape(HAND_LANDMARK_COUNT, 3)
    if not np.isfinite(points).all():
        raise ValueError(f"{path} 的右手坐标包含 NaN 或无穷值")
    return bool(round(valid_value)), points.astype(np.float32)


def read_h2o_intrinsics(path: Path) -> CameraIntrinsics:
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ", dtype=np.float64)
    if values.size != 6:
        raise ValueError(f"{path} 应包含 fx fy cx cy width height 六个数")
    fx, fy, cx, cy, width, height = (float(value) for value in values)
    if fx <= 0 or fy <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"{path} 的焦距和图像尺寸必须为正数")
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def project_h2o_points_normalized(points_xyz: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """按 H2O 的针孔相机内参，把 3D 点投影为 MediaPipe 风格归一化 2D 点。"""

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.shape != (HAND_LANDMARK_COUNT, 3):
        raise ValueError(f"points_xyz 形状必须是 (21, 3)，实际为 {points.shape}")
    z = points[:, 2]
    if not np.isfinite(points).all() or np.any(z <= 1e-8):
        raise ValueError("H2O 右手点必须有限且位于相机前方")

    u = intrinsics.fx * points[:, 0] / z + intrinsics.cx
    v = intrinsics.fy * points[:, 1] / z + intrinsics.cy
    normalized = np.stack((u / intrinsics.width, v / intrinsics.height), axis=1)
    return normalized.astype(np.float32)


def canonicalize_h2o_camera_xy(points_xyz: np.ndarray) -> np.ndarray:
    """旧 v2 兼容路径：直接用相机坐标 x/y 构造尺度无关的 2D 几何。

    这不是针孔相机投影，只为精确复现已经冻结的 v2 标签。新实验应显式选择
    ``normalized_perspective_wrist_origin_palm_scale``。
    """

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.shape != (HAND_LANDMARK_COUNT, 3) or not np.isfinite(points).all():
        raise ValueError(f"points_xyz 形状必须是有限的 (21, 3)，实际为 {points.shape}")
    xy = points[:, :2]
    origin = xy[0]
    scale = float(np.linalg.norm(xy[9] - origin))
    if scale <= 1e-8:
        scale = float(np.linalg.norm(xy[5] - xy[17]))
    if scale <= 1e-8:
        raise ValueError("H2O 掌长和掌宽同时退化，无法构造规范化 2D 几何")
    return ((xy - origin) / scale).astype(np.float32)


def canonicalize_h2o_normalized_perspective_xy(points_xyz: np.ndarray) -> np.ndarray:
    """无内参时先做 x/z、y/z 归一化透视投影，再消除平移与统一尺度。

    当前特征只使用 2D 距离比和角度，因此针孔模型中的主点平移与共同焦距会在
    后续规范化中消去；这比直接使用相机坐标 x/y 更接近 MediaPipe 图像域。
    """

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.shape != (HAND_LANDMARK_COUNT, 3) or not np.isfinite(points).all():
        raise ValueError(f"points_xyz 形状必须是有限的 (21, 3)，实际为 {points.shape}")
    depth = points[:, 2]
    if np.any(np.abs(depth) <= 1e-8):
        raise ValueError("H2O 点包含接近零的相机深度，无法做归一化透视投影")
    xy = points[:, :2] / depth[:, None]
    origin = xy[0]
    scale = float(np.linalg.norm(xy[9] - origin))
    if scale <= 1e-8:
        scale = float(np.linalg.norm(xy[5] - xy[17]))
    if scale <= 1e-8:
        raise ValueError("H2O 透视投影后的掌长和掌宽同时退化")
    return ((xy - origin) / scale).astype(np.float32)


def h2o_frame_to_svh9(
    points_xyz: np.ndarray,
    points_xy: np.ndarray,
    *,
    timestamp_s: float,
    mapping_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    """复用现有单右手特征与 SVH adapter，生成一帧 9 通道监督信号。"""

    features = extract_hand_features(
        [(float(x), float(y)) for x, y in points_xy],
        handedness="Right",
        confidence=1.0,
        timestamp=float(timestamp_s),
        landmarks_xyz=[(float(x), float(y), float(z)) for x, y, z in points_xyz],
    )
    if not features.get("detected", False):
        raise ValueError("完整的 H2O 21 点未能生成有效特征")

    raw_gesture = infer_gesture_raw(features, mapping_cfg)
    features["gesture_raw"] = raw_gesture
    features["gesture_stable"] = raw_gesture
    control = build_control_representation(features, mapping_cfg)
    features["control_representation"] = control
    preview = build_svh_command_preview(features, mapping_cfg)
    positions = np.asarray(preview.get("target_positions", []), dtype=np.float32)
    if not preview.get("valid", False) or positions.shape != (SVH_CHANNEL_COUNT,):
        raise ValueError("当前 H2O 帧未生成有效的 svh_9ch 预览标签")

    summary = {
        "pinch_distance_norm": float(features["pinch_distance_norm"]),
        "hand_open_ratio": float(features["hand_open_ratio"]),
        "grasp_close": float(control["grasp_close"]),
        "effective_pinch_strength": float(control["effective_pinch_strength"]),
    }
    return positions, summary


def _split_for(subject: str, action: str, policy: str) -> str:
    if policy == "cross_subject":
        return {"subject1": "train", "subject2": "train", "subject3": "val", "subject4": "test"}.get(
            subject, "ignore"
        )
    if policy == "official":
        if subject == "subject4":
            return "test"
        if subject == "subject3" and action in {"k2", "o1", "o2"}:
            return "val"
        if subject in {"subject1", "subject2", "subject3"}:
            return "train"
        return "ignore"
    raise ValueError(f"未知 H2O 切分策略：{policy}")


def discover_h2o_takes(root: Path, *, split_policy: str = "cross_subject") -> list[H2OTake]:
    root = root.resolve()
    takes: list[H2OTake] = []
    for hand_pose_dir in sorted(root.glob("subject*/*/*/cam4/hand_pose")):
        cam_dir = hand_pose_dir.parent
        relative = cam_dir.relative_to(root)
        if len(relative.parts) != 4:
            continue
        subject, action, take, camera = relative.parts
        if camera != "cam4":
            continue
        split = _split_for(subject, action, split_policy)
        if split != "ignore":
            takes.append(H2OTake(subject=subject, action=action, take=take, cam_dir=cam_dir, split=split))
    return takes


def _read_optional_action_label(cam_dir: Path, frame_id: int) -> int:
    path = cam_dir / "action_label" / f"{frame_id:06d}.txt"
    if not path.exists():
        return -1
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ", dtype=np.float64)
    return int(round(float(values[0]))) if values.size else -1


def _flush_segment(
    *,
    output_root: Path,
    take: H2OTake,
    segment_index: int,
    fps: float,
    frame_ids: list[int],
    landmarks_3d: list[np.ndarray],
    landmarks_2d: list[np.ndarray],
    controls_9ch: list[np.ndarray],
    feature_summary: list[list[float]],
    action_labels: list[int],
    min_segment_frames: int,
) -> dict[str, Any] | None:
    if len(frame_ids) < min_segment_frames:
        return None

    sequence_id = f"{take.identifier}_segment{segment_index:03d}"
    relative_path = Path("sequences") / take.split / f"{sequence_id}.npz"
    output_path = output_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray("intent-sequence-v1"),
        source=np.asarray("h2o"),
        sequence_id=np.asarray(sequence_id),
        split=np.asarray(take.split),
        subject=np.asarray(take.subject),
        fps=np.asarray(float(fps), dtype=np.float32),
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        timestamps_s=np.asarray(frame_ids, dtype=np.float64) / float(fps),
        landmarks_3d=np.asarray(landmarks_3d, dtype=np.float32),
        landmarks_2d=np.asarray(landmarks_2d, dtype=np.float32),
        controls_9ch=np.asarray(controls_9ch, dtype=np.float32),
        feature_summary=np.asarray(feature_summary, dtype=np.float32),
        action_labels=np.asarray(action_labels, dtype=np.int16),
    )
    return {
        "path": relative_path.as_posix(),
        "sequence_id": sequence_id,
        "subject": take.subject,
        "action": take.action,
        "take": take.take,
        "split": take.split,
        "frames": len(frame_ids),
        "first_frame": int(frame_ids[0]),
        "last_frame": int(frame_ids[-1]),
        "sha256": _sha256(output_path),
    }


def _iter_numeric_pose_files(hand_pose_dir: Path) -> Iterable[tuple[int, Path]]:
    indexed: list[tuple[int, Path]] = []
    for path in hand_pose_dir.glob("*.txt"):
        try:
            indexed.append((int(path.stem), path))
        except ValueError:
            continue
    return iter(sorted(indexed, key=lambda item: item[0]))


def preprocess_h2o_pose_dataset(
    *,
    h2o_root: Path,
    output_root: Path,
    mapping_config_path: Path,
    fps: float = H2O_DEFAULT_FPS,
    split_policy: str = "cross_subject",
    min_segment_frames: int = 40,
    limit_takes: int | None = None,
    pose_only_projection_policy: str = POSE_ONLY_PROJECTION_LEGACY,
) -> Path:
    """把已解压的 H2O pose-only 数据转成逐序列 NPZ 与 manifest。"""

    if fps <= 0:
        raise ValueError("fps 必须大于 0")
    if pose_only_projection_policy not in {
        POSE_ONLY_PROJECTION_LEGACY,
        POSE_ONLY_PROJECTION_NORMALIZED_PERSPECTIVE,
    }:
        raise ValueError(f"未知 pose-only 投影策略：{pose_only_projection_policy}")
    h2o_root = h2o_root.resolve()
    output_root = output_root.resolve()
    if not h2o_root.exists():
        raise FileNotFoundError(f"H2O 根目录不存在：{h2o_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"输出目录必须为空，避免覆盖既有实验数据：{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    mapping_config_path = mapping_config_path.resolve()
    mapping_cfg = load_config(str(mapping_config_path))
    if mapping_cfg.get("svh_preview_layout") != "svh_9ch" or int(mapping_cfg.get("svh_preview_channel_count", 0)) != 9:
        raise ValueError("mapping config 必须启用 svh_9ch 且通道数为 9")
    mapping_implementation_sha256 = assert_mapping_implementation_compatible(mapping_cfg)

    takes = discover_h2o_takes(h2o_root, split_policy=split_policy)
    if limit_takes is not None:
        takes = takes[: max(0, int(limit_takes))]
    if not takes:
        raise FileNotFoundError(
            "没有找到 subject*/动作/序号/cam4/hand_pose/*.txt；请确认4个 pose 压缩包已统一解压到同一根目录"
        )

    sequence_entries: list[dict[str, Any]] = []
    diagnostics = {
        "takes": len(takes),
        "pose_files": 0,
        "invalid_frames": 0,
        "rejected_frames": 0,
        "short_segments": 0,
        "takes_with_intrinsics": 0,
        "takes_with_canonical_xy": 0,
        "takes_with_normalized_perspective": 0,
    }

    for take in takes:
        intrinsics_path = take.cam_dir / "cam_intrinsics.txt"
        intrinsics = read_h2o_intrinsics(intrinsics_path) if intrinsics_path.exists() else None
        if intrinsics is None:
            if pose_only_projection_policy == POSE_ONLY_PROJECTION_LEGACY:
                diagnostics["takes_with_canonical_xy"] += 1
            else:
                diagnostics["takes_with_normalized_perspective"] += 1
        else:
            diagnostics["takes_with_intrinsics"] += 1
        segment_index = 0
        frame_ids: list[int] = []
        landmarks_3d: list[np.ndarray] = []
        landmarks_2d: list[np.ndarray] = []
        controls_9ch: list[np.ndarray] = []
        feature_summary: list[list[float]] = []
        action_labels: list[int] = []
        previous_frame: int | None = None

        def flush() -> None:
            nonlocal segment_index, frame_ids, landmarks_3d, landmarks_2d, controls_9ch, feature_summary, action_labels
            entry = _flush_segment(
                output_root=output_root,
                take=take,
                segment_index=segment_index,
                fps=fps,
                frame_ids=frame_ids,
                landmarks_3d=landmarks_3d,
                landmarks_2d=landmarks_2d,
                controls_9ch=controls_9ch,
                feature_summary=feature_summary,
                action_labels=action_labels,
                min_segment_frames=min_segment_frames,
            )
            if entry is None and frame_ids:
                diagnostics["short_segments"] += 1
            elif entry is not None:
                sequence_entries.append(entry)
                segment_index += 1
            frame_ids = []
            landmarks_3d = []
            landmarks_2d = []
            controls_9ch = []
            feature_summary = []
            action_labels = []

        for frame_id, pose_path in _iter_numeric_pose_files(take.cam_dir / "hand_pose"):
            diagnostics["pose_files"] += 1
            if previous_frame is not None and frame_id != previous_frame + 1:
                flush()
            previous_frame = frame_id
            try:
                valid, points_xyz = read_h2o_right_hand_pose(pose_path)
                if not valid:
                    diagnostics["invalid_frames"] += 1
                    flush()
                    continue
                points_xy = (
                    project_h2o_points_normalized(points_xyz, intrinsics)
                    if intrinsics is not None
                    else (
                        canonicalize_h2o_camera_xy(points_xyz)
                        if pose_only_projection_policy == POSE_ONLY_PROJECTION_LEGACY
                        else canonicalize_h2o_normalized_perspective_xy(points_xyz)
                    )
                )
                positions, summary = h2o_frame_to_svh9(
                    points_xyz,
                    points_xy,
                    timestamp_s=frame_id / fps,
                    mapping_cfg=mapping_cfg,
                )
            except (ValueError, OSError):
                diagnostics["rejected_frames"] += 1
                flush()
                continue

            frame_ids.append(frame_id)
            landmarks_3d.append(points_xyz)
            landmarks_2d.append(points_xy)
            controls_9ch.append(positions)
            feature_summary.append(
                [
                    summary["pinch_distance_norm"],
                    summary["hand_open_ratio"],
                    summary["grasp_close"],
                    summary["effective_pinch_strength"],
                ]
            )
            action_labels.append(_read_optional_action_label(take.cam_dir, frame_id))
        flush()

    if not sequence_entries:
        raise RuntimeError("找到 H2O 文件，但没有形成足够长的有效右手序列")

    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        entries = [entry for entry in sequence_entries if entry["split"] == split]
        split_counts[split] = {"sequences": len(entries), "frames": sum(int(entry["frames"]) for entry in entries)}

    manifest_projection_policy = (
        "perspective_if_intrinsics_else_camera_xy_wrist_origin_palm_scale"
        if pose_only_projection_policy == POSE_ONLY_PROJECTION_LEGACY
        else f"perspective_if_intrinsics_else_{pose_only_projection_policy}"
    )
    manifest = {
        "schema_version": "intent-dataset-manifest-v1",
        "dataset": "h2o_pose_only_right_hand",
        "synthetic": False,
        "research_claims_allowed": True,
        "source_root": str(h2o_root),
        "fps": float(fps),
        "split_policy": split_policy,
        "mapping_config": str(mapping_config_path),
        "mapping_config_sha256": _sha256(mapping_config_path),
        "mapping_contract_version": MAPPING_CONTRACT_VERSION,
        "mapping_contract_sha256": mapping_contract_sha256(mapping_cfg),
        "mapping_implementation_sha256": mapping_implementation_sha256,
        "h2o_label_gesture_context_policy": H2O_LABEL_GESTURE_CONTEXT_POLICY,
        "runtime_gesture_context_policy": RUNTIME_GESTURE_CONTEXT_POLICY,
        "projection_policy": manifest_projection_policy,
        "joint_order": "MediaPipe-compatible: wrist, thumb[1:5], index[5:9], middle[9:13], ring[13:17], little[17:21]",
        "feature_summary_fields": [
            "pinch_distance_norm",
            "hand_open_ratio",
            "grasp_close",
            "effective_pinch_strength",
        ],
        "split_counts": split_counts,
        "diagnostics": diagnostics,
        "sequences": sequence_entries,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
