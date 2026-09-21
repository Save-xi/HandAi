"""M2 共用的姿态后映射参考；手势状态只沿规则时间网格推进。"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from control.control_representation import build_control_representation
from features.hand_features import extract_hand_features
from gesture.rule_based_gesture import GestureStabilizer, infer_gesture_raw
from svh.svh_adapter import build_svh_command_preview

from .h2o_adapter import canonicalize_h2o_normalized_perspective_xy
from .keypoint_data import KeypointWindows, denormalize, read_sequence, sample_points
from .keypoint_metrics import query_trajectory


def new_gesture_state(cfg: dict) -> GestureStabilizer:
    return GestureStabilizer(int(cfg.get("stable_gesture_min_consecutive", 2)),
                            int(cfg.get("stable_unknown_consecutive", 1)))


def map_pose(points: np.ndarray, timestamp_s: float, state: GestureStabilizer, cfg: dict,
             *, advance: bool) -> np.ndarray:
    """M2 无内参替代坐标；查询帧不额外累计一次手势确认。"""
    xyz = np.asarray(points, dtype=float)
    if xyz.shape != (21, 3) or not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 1e-8):
        raise ValueError("映射需要有限且位于相机前方的 21 点")
    xy = canonicalize_h2o_normalized_perspective_xy(xyz)
    features = extract_hand_features(xy.tolist(), landmarks_xyz=xyz.tolist(), handedness="Right",
                                     confidence=1.0, timestamp=float(timestamp_s))
    if not features.get("detected"):
        raise ValueError("姿态未生成有效特征")
    raw = infer_gesture_raw(features, cfg)
    stable = state.update(raw) if advance else state.stable_gesture
    features.update(gesture_raw=raw, gesture_stable=stable)
    features["control_representation"] = build_control_representation(features, cfg)
    preview = build_svh_command_preview(features, cfg)
    values = np.asarray(preview.get("target_positions", []), dtype=np.float32)
    if not preview.get("valid") or values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("姿态映射未生成有效 9 通道")
    return values


def future_channels(dense_points: np.ndarray, query_points: np.ndarray, *, anchor_time: float,
                    anchor_state: GestureStabilizer, grid_ms: tuple[float, ...], query_ms: tuple[float, ...],
                    cfg: dict, fallback: np.ndarray, dense_valid: np.ndarray | None = None,
                    query_valid: np.ndarray | None = None, mapper=map_pose,
                    state_trace: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """标签与预测调用同一规则；两条支路各自复制锚点状态，互不修改。"""
    dense_valid = np.ones(len(grid_ms), dtype=bool) if dense_valid is None else np.asarray(dense_valid, dtype=bool)
    query_valid = np.ones(len(query_ms), dtype=bool) if query_valid is None else np.asarray(query_valid, dtype=bool)
    state = copy(anchor_state)
    states = [copy(state)]
    alive = True
    for points, offset, available in zip(dense_points, grid_ms, dense_valid):
        alive = alive and bool(available)
        if alive:
            try:
                mapper(points, anchor_time + offset / 1000, state, cfg, advance=True)
            except ValueError:
                alive = False
        states.append(copy(state) if alive else None)
        if state_trace is not None:
            state_trace.append(state.stable_gesture if alive else None)
    values = np.repeat(np.asarray(fallback, dtype=np.float32)[None], len(query_ms), axis=0)
    valid = np.zeros(len(query_ms), dtype=bool)
    grid = np.asarray(grid_ms)
    for index, (points, offset, available) in enumerate(zip(query_points, query_ms, query_valid)):
        # 对 50ms 只推进 33.3ms 的网格状态；100ms 包含该时刻的第3次更新。
        step = int(np.count_nonzero(grid <= offset + 1e-7))
        query_state = states[step]
        if not available or query_state is None:
            continue
        try:
            values[index] = mapper(points, anchor_time + offset / 1000, copy(query_state), cfg, advance=False)
            valid[index] = True
        except ValueError:
            pass
    return values, valid


@dataclass
class RepresentationBatch:
    points: KeypointWindows
    history9: np.ndarray
    targets9: np.ndarray
    target_valid: np.ndarray
    states: list[GestureStabilizer]
    query_ms: tuple[float, ...]
    mapping_cfg: dict
    excluded_history_mapping: int
    motion_score: np.ndarray
    reference_transition: np.ndarray

    def training_view(self, route: str):
        if route not in {"A", "B", "C"}:
            raise ValueError("路线必须为 A/B/C")
        return SimpleNamespace(
            x=self.history9 if route == "A" else self.points.x,
            y=self.points.y if route == "C" else self.targets9,
            target_mask=self.points.target_mask if route == "C" else np.repeat(self.target_valid[..., None], 9, axis=-1),
            residual_base=self.history9[:, -1].copy() if route == "B" else None,
            horizon_ms=self.points.horizon_ms if route == "C" else self.query_ms,
            task_id=f"h2o_m2_{route}",
        )

    def map_predictions(self, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """只使用预测轨迹、锚点尺度和过去状态；不读未来标签/掩码。"""
        dense = denormalize(prediction.reshape(len(prediction), -1, 21, 3), self.points.wrists, self.points.scales)
        queries = denormalize(query_trajectory(prediction, self.points.horizon_ms, self.query_ms),
                              self.points.wrists, self.points.scales)
        result, validity = [], []
        for n in range(len(prediction)):
            values, valid = future_channels(dense[n], queries[n], anchor_time=float(self.points.anchor_times[n]),
                anchor_state=self.states[n], grid_ms=self.points.horizon_ms, query_ms=self.query_ms,
                cfg=self.mapping_cfg, fallback=self.history9[n, -1])
            result.append(values)
            validity.append(valid)
        return np.asarray(result, dtype=np.float32), np.asarray(validity)


def build_representation_batch(root: Path, manifest: dict, task: dict, points: KeypointWindows,
                               mapping_cfg: dict) -> RepresentationBatch:
    # 浮点计算得到的 100ms 可能略有偏差，不能重复生成同一个预测头。
    query_ms = tuple(np.unique(np.round(np.r_[points.horizon_ms, points.eval_horizon_ms], 7)).tolist())
    grid_ms = points.horizon_ms
    entries = {entry["sequence_id"]: entry for entry in manifest["sequences"]}
    cache = {}
    for identity in np.unique(points.sequence_ids):
        seq = read_sequence(root, entries[str(identity)], task)
        if not np.allclose(np.diff(seq.timestamps), 1 / task["time"]["fps"], rtol=0, atol=1e-8):
            raise ValueError("M2 首轮参考只支持等间隔 H2O；不规则摄像头重采样留给 M3")
        controls = np.zeros((len(seq.timestamps), 9), dtype=np.float32)
        valid = np.zeros(len(seq.timestamps), dtype=bool)
        states = []
        state = new_gesture_state(mapping_cfg)
        for index, (pose, timestamp) in enumerate(zip(seq.inputs, seq.timestamps)):
            if index and seq.segments[index] != seq.segments[index - 1]:
                state = new_gesture_state(mapping_cfg)
            try:
                if not seq.input_mask[index].all():
                    raise ValueError("历史姿态缺点")
                controls[index] = map_pose(pose, timestamp, state, mapping_cfg, advance=True)
                valid[index] = True
            except ValueError:
                state = new_gesture_state(mapping_cfg)
            states.append(copy(state))
        cache[str(identity)] = (seq, controls, valid, states)
    keep, histories, targets, validity, anchor_states, transitions = [], [], [], [], [], []
    for n, identity in enumerate(points.sequence_ids):
        seq, controls, valid, states = cache[str(identity)]
        history_indices = np.searchsorted(seq.timestamps, points.history_times[n] - 1e-8)
        if not np.allclose(seq.timestamps[history_indices], points.history_times[n], rtol=0, atol=1e-8):
            raise ValueError("M2 历史与参考状态网格未对齐")
        if not valid[history_indices].all():
            continue
        anchor = int(history_indices[-1])
        now = float(points.anchor_times[n])
        dense, dense_mask, _ = sample_points(seq, now + np.asarray(grid_ms) / 1000,
            anchor=anchor, max_gap_s=task["time"]["max_label_gap_s"])
        queries, query_mask, _ = sample_points(seq, now + np.asarray(query_ms) / 1000,
            anchor=anchor, max_gap_s=task["time"]["max_label_gap_s"])
        trace = []
        values, mask = future_channels(dense, queries, anchor_time=now, anchor_state=states[anchor],
            grid_ms=grid_ms, query_ms=query_ms, cfg=mapping_cfg, fallback=controls[anchor],
            dense_valid=dense_mask.all(axis=1), query_valid=query_mask.all(axis=1), state_trace=trace)
        keep.append(n)
        histories.append(controls[history_indices])
        targets.append(np.where(mask[:, None], values, 0.0))
        validity.append(mask)
        anchor_states.append(copy(states[anchor]))
        transitions.append(any(gesture != states[anchor].stable_gesture
                               for gesture, offset in zip(trace, grid_ms) if gesture is not None and offset <= 150 + 1e-7))
    if not keep:
        raise ValueError("没有可供 A/B/C 共同使用的完整历史映射")
    selected = replace(points, **{field.name: getattr(points, field.name)[keep]
        for field in fields(points) if isinstance(getattr(points, field.name), np.ndarray)})
    recent = selected.x[:, -min(8, selected.x.shape[1]):].reshape(len(keep), -1, 21, 3)
    motion = np.linalg.norm(np.diff(recent, axis=1), axis=-1).mean(axis=(1, 2)) * task["time"]["fps"]
    return RepresentationBatch(selected, np.asarray(histories), np.asarray(targets), np.asarray(validity),
                                anchor_states, query_ms, mapping_cfg, len(points.x) - len(keep),
                                motion, np.asarray(transitions))
