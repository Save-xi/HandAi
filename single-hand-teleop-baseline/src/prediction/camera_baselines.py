"""M3-B：摄像头坐标的保持/常速度预测与完整映射状态推进。"""
from __future__ import annotations

import time
import numpy as np

from perception.base import HandDetection
from perception.geometry import GEOMETRY_TASK
from pipeline import HandPipeline

HORIZONS_MS = (50.0, 100.0, 150.0)


def geometry_detection(points, width: int, height: int) -> HandDetection:
    xyz = np.asarray(points, dtype=float).copy()
    if xyz.shape != (21, 3) or not np.isfinite(xyz).all() or width <= 0 or height <= 0:
        raise ValueError("需要有限的摄像头几何 21 点与有效宽高")
    xyz[:, 1] *= width / height
    return HandDetection(xyz[:, :2].tolist(), xyz.tolist(), "Right", 1.0, width, height, "mediapipe_image_xyz")


def observe_frame(pipeline: HandPipeline, frame: dict, previous: dict | None = None) -> dict:
    """只推进实际可见源帧；缺帧先断开状态，即使时间间隔尚小于阈值。"""
    if previous is not None and (frame["frame_id"] != previous["frame_id"] + 1
                                or frame["segment_id"] != previous["segment_id"]):
        pipeline.reset()
    detections = ([geometry_detection(frame["geometry_landmarks"], frame["image_width"], frame["image_height"])]
                  if frame["input_valid"] else [])
    return pipeline.process_detections(detections, frame_index=frame["frame_id"], timestamp=frame["source_time_ms"] / 1000)


def sample_pose(frames: list[dict], target_ms: float, *, segment_id: int, max_gap_ms=100.0):
    """标签/历史共同插值函数；不跨坏帧、缺帧、换尺寸或过大间隔。"""
    if not frames:
        return None
    times = np.array([f["source_time_ms"] for f in frames])
    index = int(np.searchsorted(times, target_ms))
    for exact in (index, index - 1):
        if 0 <= exact < len(times) and abs(times[exact] - target_ms) < 1e-6:
            row = frames[exact]
            return np.asarray(row["geometry_landmarks"], dtype=float) if row["input_valid"] and row["segment_id"] == segment_id else None
    if index == 0 or index == len(times):
        return None
    a, b = frames[index - 1], frames[index]
    if (not a["input_valid"] or not b["input_valid"] or a["segment_id"] != segment_id or b["segment_id"] != segment_id
        or b["frame_id"] != a["frame_id"] + 1 or not 0 < times[index] - times[index - 1] <= max_gap_ms):
        return None
    weight = (target_ms - times[index - 1]) / (times[index] - times[index - 1])
    return (1 - weight) * np.asarray(a["geometry_landmarks"]) + weight * np.asarray(b["geometry_landmarks"])


def future_mapping(anchor: HandPipeline, frame: dict, dense_points: list, query_points: list,
                   *, fps=30.0, horizons=HORIZONS_MS) -> dict:
    """规则点推进手势与释放，分数查询读取前一网格完整状态，不重复更新。"""
    grid = np.arange(1, len(dense_points) + 1) * 1000 / fps
    state = anchor.fork_mapping()
    states, grid_payloads = [state.fork_mapping()], []
    alive = True
    for step, (points, offset) in enumerate(zip(dense_points, grid), 1):
        payload = None
        if alive and points is not None:
            try:
                detection = geometry_detection(points, frame["image_width"], frame["image_height"])
                payload = state.process_detections([detection], frame_index=frame["frame_id"] + step,
                                                   timestamp=(frame["source_time_ms"] + offset) / 1000)
                alive = bool(payload["control_ready"] and payload["svh_preview"]["valid"])
            except ValueError:
                alive = False
        else:
            alive = False
        states.append(state.fork_mapping() if alive else None)
        grid_payloads.append(payload if alive else None)
    positions, gestures, reasons = [], [], []
    for points, horizon in zip(query_points, horizons):
        step = int(np.count_nonzero(grid <= horizon + 1e-6))
        local = states[step]
        payload = None
        if points is not None and local is not None:
            exact = step > 0 and abs(grid[step - 1] - horizon) < 1e-6
            if exact:
                payload = grid_payloads[step - 1]
            else:
                try:
                    d = geometry_detection(points, frame["image_width"], frame["image_height"])
                    payload = local.query_detections([d], frame_index=frame["frame_id"] + step,
                        timestamp=(frame["source_time_ms"] + horizon) / 1000)
                except ValueError:
                    pass
        valid = payload is not None and payload["control_ready"] and payload["svh_preview"]["valid"]
        positions.append(payload["svh_preview"]["target_positions"] if valid else None)
        gestures.append(payload["gesture_stable"] if valid else None)
        reasons.append(None if valid else "invalid_future_geometry_or_state")
    return {"positions": positions, "gestures": gestures, "reasons": reasons}


def predict_camera(history: list[dict], anchor: HandPipeline, current: dict, *, method: str, window=4, fps=30.0) -> dict:
    """仅接收锚点及之前的数据；计时覆盖拟合、未来姿态与全部后处理。"""
    started = time.perf_counter()
    frame = history[-1]
    if anchor.cfg.get("geometry_mode") != "image_width_xyz_v1":
        raise ValueError(f"M3-B 仅接受 {GEOMETRY_TASK}")
    if method not in {"hold_channels", "hold_pose", "linear_pose"}:
        raise ValueError("未知摄像头基线")
    result = {"positions": [None] * 3, "gestures": [None] * 3, "reasons": ["invalid_current"] * 3}
    model_end = started
    if current["control_ready"] and current["svh_preview"]["valid"]:
        if method == "hold_channels":
            result = {"positions": [list(current["svh_preview"]["target_positions"]) for _ in HORIZONS_MS],
                      "gestures": [current["gesture_stable"]] * 3, "reasons": [None] * 3}
            model_end = time.perf_counter()
        else:
            pose = np.asarray(frame["geometry_landmarks"], dtype=float)
            velocity = np.zeros_like(pose)
            warmup = False
            if method == "linear_pose":
                if not isinstance(window, int) or isinstance(window, bool) or window < 2:
                    raise ValueError("速度窗口必须为至少 2 个采样点")
                times = np.arange(-(window - 1), 1) * 1000 / fps
                past = [sample_pose(history, frame["source_time_ms"] + offset, segment_id=frame["segment_id"])
                        for offset in times]
                warmup = any(points is None for points in past)
                if not warmup:
                    wrist = pose[0]
                    primary = float(np.linalg.norm(pose[0, :2] - pose[9, :2]))
                    scale = primary if primary > 1e-6 else float(np.linalg.norm(pose[5, :2] - pose[17, :2]))
                    normalized = (np.array(past) - wrist) / scale
                    centered = times - times.mean()
                    velocity = np.tensordot(centered, normalized, axes=1) / (centered @ centered) * scale
            if warmup:
                result["reasons"] = ["history_warmup"] * 3
                model_end = time.perf_counter()
            else:
                grid = np.arange(1, int(np.ceil(max(HORIZONS_MS) * fps / 1000)) + 1) * 1000 / fps
                dense = pose[None] + grid[:, None, None] * velocity
                # 常速度和保持都是线性轨迹，此式等同先插值姿态再映射。
                queries = pose[None] + np.asarray(HORIZONS_MS)[:, None, None] * velocity
                model_end = time.perf_counter()
                result = future_mapping(anchor, frame, dense, queries, fps=fps)
    ended = time.perf_counter()
    result.update(model_ms=(model_end - started) * 1000, postprocess_ms=(ended - model_end) * 1000,
                  total_ms=(ended - started) * 1000)
    return result


def reference_camera(frames: list[dict], index: int, anchor: HandPipeline, *, fps=30.0) -> dict:
    """共同参考可读取真实未来；预测函数不接收这个参数或结果。"""
    row = frames[index]
    if not row["input_valid"]:
        return {"positions": [None] * 3, "gestures": [None] * 3, "reasons": ["invalid_anchor"] * 3}
    grid = np.arange(1, int(np.ceil(max(HORIZONS_MS) * fps / 1000)) + 1) * 1000 / fps
    # 最多需要锚点后 166.7ms；显式保留原始段号和缺帧限制。
    future = frames[index:index + max(64, int(fps))]
    dense = [sample_pose(future, row["source_time_ms"] + offset, segment_id=row["segment_id"]) for offset in grid]
    queries = [sample_pose(future, row["source_time_ms"] + offset, segment_id=row["segment_id"]) for offset in HORIZONS_MS]
    return future_mapping(anchor, row, dense, queries, fps=fps)
