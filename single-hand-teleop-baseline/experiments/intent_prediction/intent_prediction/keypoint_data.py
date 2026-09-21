"""21 点任务定义、因果时间采样、锚点归一化及带掩码的窗口。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .sequence_data import load_manifest


JOINT_NAMES = ("wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
               "index_mcp", "index_pip", "index_dip", "index_tip",
               "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
               "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
               "little_mcp", "little_pip", "little_dip", "little_tip")
TASK_ID = "h2o_camera3d_right21"


def validate_task(task: dict) -> None:
    """校验影响数值含义的字段；来源链接等说明不参与兼容判断。"""
    coords, timing, norm = task["coordinates"], task["time"], task["normalization"]
    if task["task_id"] != TASK_ID or coords["frame"] != "cam4_camera" or coords["raw_unit"] != "m":
        raise ValueError("M1 仅支持 H2O 右手相机 3D 米坐标，不接受 MediaPipe 图像相对坐标")
    if coords["hand"] != "right" or tuple(coords["joint_order"]) != JOINT_NAMES:
        raise ValueError("右手或 21 点顺序不匹配")
    if coords["axes"] != ["x_camera_right", "y_camera_down", "z_camera_forward"]:
        raise ValueError("相机坐标轴定义不匹配")
    if not np.isfinite(timing["fps"]) or timing["fps"] <= 0 or timing["history_frames"] < 2:
        raise ValueError("fps 必须有限且为正，历史至少 2 帧")
    if timing["forecast_steps"] != list(range(1, max(timing["forecast_steps"]) + 1)):
        raise ValueError("forecast_steps 必须从 1 开始连续，供未来路径查询")
    horizons = np.asarray(timing["evaluation_horizons_ms"], dtype=float) / 1000
    if not len(horizons) or not np.isfinite(horizons).all() or np.any(np.diff(horizons) <= 0):
        raise ValueError("评测时距必须有限且严格递增")
    if horizons[0] < 1 / timing["fps"] - 1e-9 or horizons[-1] > max(timing["forecast_steps"]) / timing["fps"] + 1e-9:
        raise ValueError("评测时距必须在预测轨迹内，不允许外推冒充标签")
    if timing["anchor_stride"] < 1 or not np.isfinite(timing["max_label_gap_s"]) or timing["max_label_gap_s"] <= 0:
        raise ValueError("步长和最大标注间隔必须为正")
    if timing["cross_explicit_invalid_segment"] or norm["future_wrist_or_scale_used"] or norm["clip_to_0_1"]:
        raise ValueError("不允许跨无效段、未来归一化或关键点值域截断")
    if (norm["wrist_index"], norm["palm_length_indices"], norm["fallback_width_indices"]) != (0, [0, 9], [5, 17]):
        raise ValueError("锚点腕点/掌长/掌宽定义不匹配")
    if norm["method"] != "anchor_wrist_and_scale_shared_by_history_and_future" or norm["rotation_alignment"]:
        raise ValueError("M1 使用同一锚点变换，不做旋转对齐")
    fraction = norm["s_min_fraction_of_positive_palm_median"]
    if norm["s_min_fit_split"] != "train" or not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("尺度下限须由训练集正掌长中位数的有效比例拟合")
    if task["data"]["input_key"] != "landmarks_3d" or task["data"]["target_key"] != "landmarks_3d":
        raise ValueError("M1 的输入和目标均为 landmarks_3d；其他表示留到 M2")


def task_contract(task: dict, s_min_m: float) -> dict:
    """checkpoint 与输入共享的坐标/时间定义，不包含路径或文件身份。"""
    validate_task(task)
    if not np.isfinite(s_min_m) or s_min_m <= 0:
        raise ValueError("s_min_m 必须有限且为正")
    return {"task_id": task["task_id"], "coordinates": {key: task["coordinates"][key]
            for key in ("frame", "axes", "raw_unit", "hand", "joint_order")},
            "time": {key: task["time"][key] for key in ("fps", "history_frames", "forecast_steps",
                      "evaluation_horizons_ms", "max_label_gap_s", "timestamp_origin")},
            "input_key": task["data"]["input_key"], "target_key": task["data"]["target_key"],
            "normalization": "anchor_wrist_and_palm_with_width_fallback",
            "s_min_m": float(s_min_m), "input_missing_policy": "reject_incomplete_history",
            "input_size": 63, "output_size": 63}


@dataclass
class KeypointSequence:
    sequence_id: str
    subject: str
    timestamps: np.ndarray
    frame_ids: np.ndarray
    inputs: np.ndarray
    targets: np.ndarray
    input_mask: np.ndarray
    target_mask: np.ndarray
    segments: np.ndarray


def make_sequence(sequence_id: str, subject: str, timestamps: np.ndarray, points: np.ndarray,
                  *, max_gap_s: float, frame_ids: np.ndarray | None = None,
                  mask: np.ndarray | None = None, frame_valid: np.ndarray | None = None,
                  segment_ids: np.ndarray | None = None, targets: np.ndarray | None = None,
                  target_mask: np.ndarray | None = None) -> KeypointSequence:
    times = np.asarray(timestamps, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    n = len(times)
    if times.shape != (n,) or n < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("时间必须是一维有限严格递增数组，至少 2 帧")
    if points.shape != (n, 21, 3):
        raise ValueError("关键点必须为非空 [T,21,3]")
    ids = np.arange(n, dtype=np.int64) if frame_ids is None else np.asarray(frame_ids)
    if ids.shape != (n,) or ids.dtype.kind not in "iu" or np.any(np.diff(ids) <= 0):
        raise ValueError("frame_ids 必须为严格递增整数")

    def boolean_mask(value, shape):
        if value is None:
            return np.ones(shape, dtype=bool)
        value = np.asarray(value)
        if value.shape != shape or not np.isin(value, [0, 1]).all():
            raise ValueError(f"有效掩码应为 0/1 数组 {shape}")
        return value.astype(bool)

    valid = boolean_mask(frame_valid, (n,))
    in_mask = boolean_mask(mask, (n, 21)) & np.isfinite(points).all(axis=-1) & valid[:, None]
    target_points = points.copy() if targets is None else np.asarray(targets, dtype=np.float64)
    if target_points.shape != points.shape:
        raise ValueError("监督关键点形状不匹配")
    out_mask = (in_mask.copy() if target_mask is None and targets is None else boolean_mask(target_mask, (n, 21)))
    out_mask &= np.isfinite(target_points).all(axis=-1) & valid[:, None]
    frame_present = valid & in_mask.any(axis=1) & out_mask.any(axis=1)
    breaks = (np.diff(ids) != 1) | (np.diff(times) > max_gap_s + 1e-9)
    breaks |= ~frame_present[:-1] | ~frame_present[1:]
    if segment_ids is not None:
        segment_ids = np.asarray(segment_ids)
        if segment_ids.shape != (n,):
            raise ValueError("segment_ids 长度不匹配")
        breaks |= segment_ids[1:] != segment_ids[:-1]
    segments = np.r_[0, np.cumsum(breaks)]
    return KeypointSequence(sequence_id, subject, times, ids,
                            np.where(in_mask[..., None], points, 0.0),
                            np.where(out_mask[..., None], target_points, 0.0), in_mask, out_mask, segments)


def read_sequence(root: Path, entry: dict, task: dict) -> KeypointSequence:
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("序列路径越出数据目录")
    with np.load(path, allow_pickle=False) as data:
        if "source" in data and str(data["source"].item()) not in ("h2o", "synthetic_keypoint"):
            raise ValueError("序列来源不是当前 H2O 关键点任务")
        if "task_id" in data and str(data["task_id"].item()) != task["task_id"]:
            raise ValueError("序列任务与配置不一致")
        for key in ("sequence_id", "subject", "split"):
            if str(data[key].item()) != str(entry[key]):
                raise ValueError(f"{path.name}: {key} 与 manifest 不一致")
        if not np.isclose(float(data["fps"]), task["time"]["fps"], rtol=0, atol=1e-6):
            raise ValueError("序列名义 fps 与任务不一致")
        xyz = data[task["data"]["input_key"]]
        return make_sequence(entry["sequence_id"], entry["subject"], data["timestamps_s"], xyz,
                             frame_ids=data["frame_ids"], max_gap_s=task["time"]["max_label_gap_s"],
                             mask=data["landmark_mask"] if "landmark_mask" in data else None,
                             frame_valid=data["frame_valid"] if "frame_valid" in data else None,
                             segment_ids=data["segment_ids"] if "segment_ids" in data else None)


def validate_manifest(root: Path, task: dict) -> dict:
    manifest = load_manifest(root)
    known_native = manifest.get("dataset") == "h2o_pose_only_right_hand" and not manifest.get("synthetic")
    known_synthetic = (manifest.get("dataset") == "synthetic_keypoint_21" and manifest.get("synthetic")
                       and manifest.get("task_id") == task["task_id"])
    if not (known_native or known_synthetic) or manifest.get("task_id", task["task_id"]) != task["task_id"]:
        raise ValueError("数据任务不兼容；不能仅凭 [T,21,3] 形状混用坐标域")
    subjects = {person: split for split, people in task["splits"].items() for person in people}
    if len(subjects) != sum(map(len, task["splits"].values())):
        raise ValueError("人员在多个切分重复")
    seen_ids, seen_paths, takes = set(), set(), {}
    for entry in manifest["sequences"]:
        identity, path = entry["sequence_id"], str((root / entry["path"]).resolve())
        if identity in seen_ids or path in seen_paths or subjects.get(entry["subject"]) != entry["split"]:
            raise ValueError("数据有重复序列/路径或人员跨切分")
        seen_ids.add(identity)
        seen_paths.add(path)
        take = (entry["subject"], entry.get("action"), entry.get("take"))
        if "first_frame" in entry and "last_frame" in entry:
            first, last = int(entry["first_frame"]), int(entry["last_frame"])
            if any(first <= right and left <= last for left, right in takes.get(take, [])):
                raise ValueError("同一原始 take 的保留帧重复")
            takes.setdefault(take, []).append((first, last))
    if not seen_ids:
        raise ValueError("数据清单为空")
    return manifest


def fit_normalization(root: Path, manifest: dict, task: dict) -> dict:
    lengths = []
    for entry in manifest["sequences"]:
        if entry["split"] != "train":
            continue
        seq = read_sequence(root, entry, task)
        values = np.linalg.norm(seq.inputs[:, 9] - seq.inputs[:, 0], axis=-1)
        valid = seq.input_mask[:, 0] & seq.input_mask[:, 9] & (values > 0)
        lengths.extend(values[valid].tolist())
    if not lengths:
        raise ValueError("训练集无可用正掌长")
    median = float(np.median(lengths))
    return {"fit_split": "train", "fit_scope": "all_training_sequences_before_pilot_subsampling",
            "positive_frames": len(lengths), "palm_median_m": median,
            "s_min_m": median * task["normalization"]["s_min_fraction_of_positive_palm_median"]}


def sample_points(seq: KeypointSequence, query: np.ndarray, *, anchor: int, max_gap_s: float,
                  history: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """插值仅取紧邻时间包围点；历史调用物理截断未来数组，禁止跨段。"""
    end = anchor + 1 if history else len(seq.timestamps)
    times = seq.timestamps[:end]
    query = np.asarray(query, dtype=np.float64)
    right = np.searchsorted(times, query, side="left")
    right = np.clip(right, 0, len(times) - 1)
    exact = np.isclose(times[right], query, rtol=0, atol=1e-9)
    # 浮点运算可能使精确网格点略落到前一个观测的右侧。
    previous = np.maximum(right - 1, 0)
    previous_exact = np.isclose(times[previous], query, rtol=0, atol=1e-9)
    right = np.where(~exact & previous_exact, previous, right)
    exact |= previous_exact
    left = np.where(exact, right, np.maximum(right - 1, 0))
    span = times[right] - times[left]
    valid = (query >= times[0] - 1e-9) & (query <= times[-1] + 1e-9)
    valid &= (span <= max_gap_s + 1e-9) & (exact | (span > 0))
    valid &= (seq.segments[left] == seq.segments[anchor]) & (seq.segments[right] == seq.segments[anchor])
    values, masks = (seq.inputs, seq.input_mask) if history else (seq.targets, seq.target_mask)
    mask = masks[left] & masks[right] & valid[:, None]
    alpha = np.divide(query - times[left], span, out=np.zeros_like(query), where=span > 0)
    points = values[left] * (1 - alpha[:, None, None]) + values[right] * alpha[:, None, None]
    brackets = np.stack([times[left], times[right]], axis=-1)
    brackets[~valid] = np.nan
    return np.where(mask[..., None], points, 0.0), mask, brackets


def prepare_history(seq: KeypointSequence, anchor: int, task: dict, s_min_m: float) -> tuple[dict | None, str]:
    timing = task["time"]
    query = seq.timestamps[anchor] + np.arange(1 - timing["history_frames"], 1) / timing["fps"]
    segment_start = np.flatnonzero(seq.segments == seq.segments[anchor])[0]
    if query[0] < seq.timestamps[segment_start] - 1e-9:
        return None, "warmup_or_segment_boundary"
    points, mask, brackets = sample_points(seq, query, anchor=anchor, max_gap_s=timing["max_label_gap_s"], history=True)
    if not mask.all():
        return None, "incomplete_history"
    wrist = seq.inputs[anchor, 0]
    scale = float(np.linalg.norm(seq.inputs[anchor, 9] - wrist))
    fallback = scale < s_min_m
    if fallback:
        scale = float(np.linalg.norm(seq.inputs[anchor, 5] - seq.inputs[anchor, 17]))
    if not np.isfinite(scale) or scale < s_min_m:
        return None, "invalid_scale"
    return {"x": ((points - wrist) / scale).reshape(len(query), 63).astype(np.float32),
            "input_mask": mask, "history_times": query, "history_brackets": brackets,
            "wrist": wrist, "scale": scale, "width_fallback": fallback}, "ok"


def denormalize(points: np.ndarray, wrists: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """points=[N,H,21,3]；只使用对应锚点的腕点与尺度。"""
    return np.asarray(points) * np.asarray(scales)[:, None, None, None] + np.asarray(wrists)[:, None, None, :]


@dataclass
class KeypointWindows:
    x: np.ndarray
    y: np.ndarray
    target_mask: np.ndarray
    eval_y: np.ndarray
    eval_mask: np.ndarray
    input_mask: np.ndarray
    wrists: np.ndarray
    scales: np.ndarray
    sequence_ids: np.ndarray
    subjects: np.ndarray
    anchor_frame_ids: np.ndarray
    anchor_times: np.ndarray
    history_times: np.ndarray
    history_brackets: np.ndarray
    target_times: np.ndarray
    target_brackets: np.ndarray
    eval_target_times: np.ndarray
    eval_brackets: np.ndarray
    horizon_ms: tuple[float, ...]
    eval_horizon_ms: tuple[int, ...]
    stats: dict
    task_id: str = TASK_ID


def build_keypoint_windows(root: Path, manifest: dict, task: dict, normalization: dict, *, split: str,
                           seed: int, sequences_per_subject: int | None = None,
                           windows_per_sequence: int | None = None) -> KeypointWindows:
    rng = np.random.default_rng(seed)
    chosen = []
    for subject in task["splits"][split]:
        entries = sorted((e for e in manifest["sequences"] if e["subject"] == subject), key=lambda e: e["sequence_id"])
        if sequences_per_subject is not None and len(entries) > sequences_per_subject:
            indices = sorted(rng.choice(len(entries), sequences_per_subject, replace=False).tolist())
            entries = [entries[index] for index in indices]
        chosen.extend(entries)
    timing = task["time"]
    offsets = np.asarray(timing["forecast_steps"]) / timing["fps"]
    eval_offsets = np.asarray(timing["evaluation_horizons_ms"]) / 1000
    samples, stats, sequence_stats = [], Counter(), []
    for entry in chosen:
        seq = read_sequence(root, entry, task)
        local, candidates = Counter(), []
        local["source_frames"] = len(seq.timestamps)
        local["input_missing_frames"] = int(np.sum(~seq.input_mask.all(axis=1)))
        for anchor in range(0, len(seq.timestamps), int(timing["anchor_stride"])):
            local["anchors_considered"] += 1
            history, reason = prepare_history(seq, anchor, task, normalization["s_min_m"])
            if history is None:
                local[reason] += 1
                continue
            local["history_ready"] += 1
            local["width_fallback"] += int(history["width_fallback"])
            targets, target_mask, target_brackets = sample_points(seq, seq.timestamps[anchor] + offsets,
                anchor=anchor, max_gap_s=timing["max_label_gap_s"])
            eval_points, eval_mask, eval_brackets = sample_points(seq, seq.timestamps[anchor] + eval_offsets,
                anchor=anchor, max_gap_s=timing["max_label_gap_s"])
            local["no_supervision"] += int(not target_mask.any())
            local["full_future_path"] += int(target_mask.all())
            for h, mask_at_h in zip(timing["evaluation_horizons_ms"], eval_mask):
                local[f"target_{h}ms_complete"] += int(mask_at_h.all())
                local[f"target_{h}ms_points"] += int(mask_at_h.sum())
            wrist, scale = history["wrist"], history["scale"]
            record = {key: history[key] for key in ("x", "input_mask", "history_times", "history_brackets")}
            record.update(y=np.where(target_mask[..., None], (targets - wrist) / scale, 0).reshape(len(offsets), 63),
                          target_mask=target_mask,
                          eval_y=np.where(eval_mask[..., None], (eval_points - wrist) / scale, 0), eval_mask=eval_mask,
                          wrists=wrist, scales=scale, sequence_ids=seq.sequence_id, subjects=seq.subject,
                          anchor_frame_ids=seq.frame_ids[anchor], anchor_times=seq.timestamps[anchor],
                          target_times=seq.timestamps[anchor] + offsets, target_brackets=target_brackets,
                          eval_target_times=seq.timestamps[anchor] + eval_offsets, eval_brackets=eval_brackets)
            candidates.append(record)
        if windows_per_sequence is not None and len(candidates) > windows_per_sequence:
            indices = sorted(rng.choice(len(candidates), windows_per_sequence, replace=False).tolist())
            candidates = [candidates[i] for i in indices]
        local["sampled_windows"] = len(candidates)
        stats.update(local)
        sequence_stats.append({"sequence_id": seq.sequence_id, "subject": seq.subject, **local})
        samples.extend(candidates)
    if not samples:
        raise ValueError(f"{split} 未形成完整历史窗口")
    arrays = {key: np.asarray([sample[key] for sample in samples]) for key in samples[0]}
    for key in ("x", "y", "eval_y"):
        arrays[key] = arrays[key].astype(np.float32)
    return KeypointWindows(**arrays, horizon_ms=tuple((offsets * 1000).tolist()),
                            eval_horizon_ms=tuple(timing["evaluation_horizons_ms"]),
                            stats={"split": split, "counts_before_window_subsampling": dict(stats),
                                   "sequences": sequence_stats})


def create_keypoint_smoke_dataset(root: Path, task: dict, *, frames: int = 100, articulated: bool = False) -> Path:
    """非空 21 点合成数据；用于链路验证，不代表真实效果。"""
    entries = []
    fps = task["time"]["fps"]
    for split, subjects in task["splits"].items():
        for number, subject in enumerate(subjects):
            t = np.arange(frames) / fps
            hand = np.zeros((21, 3))
            for finger in range(5):
                for joint in range(4):
                    hand[1 + 4 * finger + joint] = [(finger - 2) * 0.024, 0.04 + joint * 0.025, joint * 0.002]
            xyz = np.broadcast_to(hand, (frames, 21, 3)).copy()
            if articulated:
                # M2 需要指形变化，刚体平移可能被映射抵消，产生全零通道训练损失。
                for finger in range(5):
                    bend = 0.8 + 0.7 * np.sin(4 * t + number + 0.15 * finger)
                    for joint, length in enumerate((0.025, 0.025, 0.020), start=1):
                        index = 1 + 4 * finger + joint
                        xyz[:, index] = xyz[:, index - 1]
                        xyz[:, index, 1] += length * np.cos(bend * joint)
                        xyz[:, index, 2] -= length * np.sin(bend * joint)
            xyz[:, :, 0] += (0.02 * t + 0.008 * np.sin(t * 2 + number))[:, None]
            xyz[:, :, 1] += (-0.02 + 0.01 * np.cos(t))[:, None]
            xyz[:, :, 2] += (0.5 + 0.015 * t)[:, None]
            identity = f"synthetic_keypoint_{subject}"
            path = root / "sequences" / split / f"{identity}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, sequence_id=identity, subject=subject, split=split, fps=fps,
                                 frame_ids=np.arange(frames), timestamps_s=t, landmarks_3d=xyz.astype(np.float32))
            entries.append({"path": path.relative_to(root).as_posix(), "sequence_id": identity,
                            "subject": subject, "split": split, "action": "synthetic", "take": "0",
                            "first_frame": 0, "last_frame": frames - 1, "frames": frames})
    manifest = {"schema_version": "intent-dataset-manifest-v1", "dataset": "synthetic_keypoint_21",
                "synthetic": True, "articulated": articulated, "research_claims_allowed": False,
                "task_id": task["task_id"], "sequences": entries}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
