from __future__ import annotations

"""9 通道控制序列的存储、窗口化与确定性 smoke 数据。"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


SVH_CHANNEL_NAMES = (
    "thumb_flexion",
    "thumb_opposition",
    "index_finger_distal",
    "index_finger_proximal",
    "middle_finger_distal",
    "middle_finger_proximal",
    "ring_finger",
    "pinky",
    "finger_spread",
)


@dataclass(frozen=True)
class WindowSplit:
    x: np.ndarray
    y: np.ndarray
    horizon_ms: tuple[int, ...]
    horizon_steps: np.ndarray
    sequence_ids: np.ndarray
    anchor_frame_ids: np.ndarray

    def __post_init__(self) -> None:
        if self.x.ndim != 3 or self.x.shape[-1] != len(SVH_CHANNEL_NAMES):
            raise ValueError(f"x 应为 [样本, 历史帧, 9]，实际为 {self.x.shape}")
        if self.y.ndim != 3 or self.y.shape[1:] != (len(self.horizon_ms), len(SVH_CHANNEL_NAMES)):
            raise ValueError(f"y 应为 [样本, 预测距离, 9]，实际为 {self.y.shape}")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("x 与 y 样本数必须一致")


def load_manifest(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到预处理 manifest：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "intent-dataset-manifest-v1":
        raise ValueError("不支持的数据 manifest 版本")
    return manifest


def _interpolated_future(values: np.ndarray, anchor: int, offset_frames: float) -> np.ndarray:
    target_index = anchor + float(offset_frames)
    lower = int(math.floor(target_index))
    upper = int(math.ceil(target_index))
    if lower == upper:
        return values[lower]
    alpha = float(target_index - lower)
    return (1.0 - alpha) * values[lower] + alpha * values[upper]


def _iter_sequence_windows(
    sequence_path: Path,
    *,
    history_frames: int,
    horizon_ms: tuple[int, ...],
    stride: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, str, int, np.ndarray]]:
    with np.load(sequence_path, allow_pickle=False) as data:
        controls = np.asarray(data["controls_9ch"], dtype=np.float32)
        frame_ids = np.asarray(data["frame_ids"], dtype=np.int32)
        sequence_id = str(data["sequence_id"].item())
        fps = float(data["fps"].item())

    offsets = np.asarray(horizon_ms, dtype=np.float64) * fps / 1000.0
    max_offset = float(np.max(offsets))
    max_anchor = len(controls) - 1 - int(math.ceil(max_offset))
    for anchor in range(history_frames - 1, max_anchor + 1, stride):
        history = controls[anchor - history_frames + 1 : anchor + 1]
        future = np.stack([_interpolated_future(controls, anchor, offset) for offset in offsets], axis=0)
        yield history, future, sequence_id, int(frame_ids[anchor]), offsets.astype(np.float32)


def build_window_split(
    dataset_root: Path,
    *,
    split: str,
    history_frames: int,
    horizon_ms: tuple[int, ...],
    stride: int = 1,
    max_windows: int | None = None,
    seed: int = 20260823,
) -> WindowSplit:
    if history_frames < 2:
        raise ValueError("history_frames 至少为 2")
    if not horizon_ms or any(value <= 0 for value in horizon_ms):
        raise ValueError("horizon_ms 必须包含正整数")
    if stride < 1:
        raise ValueError("stride 至少为 1")

    dataset_root = dataset_root.resolve()
    manifest = load_manifest(dataset_root)
    entries = [entry for entry in manifest.get("sequences", []) if entry.get("split") == split]
    if not entries:
        raise ValueError(f"manifest 中没有 split={split} 的序列")

    rng = np.random.default_rng(seed)
    reservoir: list[tuple[np.ndarray, np.ndarray, str, int, np.ndarray]] = []
    seen = 0
    cap = None if max_windows is None else max(1, int(max_windows))
    for entry in entries:
        path = dataset_root / str(entry["path"])
        for sample in _iter_sequence_windows(
            path,
            history_frames=history_frames,
            horizon_ms=horizon_ms,
            stride=stride,
        ):
            seen += 1
            if cap is None or len(reservoir) < cap:
                reservoir.append(sample)
                continue
            replacement = int(rng.integers(0, seen))
            if replacement < cap:
                reservoir[replacement] = sample

    if not reservoir:
        raise ValueError(f"split={split} 没有形成任何可用窗口")
    rng.shuffle(reservoir)
    x = np.stack([item[0] for item in reservoir]).astype(np.float32)
    y = np.stack([item[1] for item in reservoir]).astype(np.float32)
    sequence_ids = np.asarray([item[2] for item in reservoir])
    anchor_frame_ids = np.asarray([item[3] for item in reservoir], dtype=np.int32)
    horizon_steps = np.asarray(reservoir[0][4], dtype=np.float32)
    return WindowSplit(
        x=x,
        y=y,
        horizon_ms=horizon_ms,
        horizon_steps=horizon_steps,
        sequence_ids=sequence_ids,
        anchor_frame_ids=anchor_frame_ids,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def create_synthetic_smoke_dataset(output_root: Path, *, seed: int = 20260823) -> Path:
    """创建只用于验证代码链路的确定性 9 通道序列，禁止作为研究结论。"""

    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"synthetic smoke 输出目录必须为空：{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    split_layout = {"train": 6, "val": 2, "test": 2}
    entries: list[dict[str, Any]] = []
    fps = 30.0
    length = 180

    for split, count in split_layout.items():
        for sequence_index in range(count):
            t = np.arange(length, dtype=np.float64) / fps
            phase = 0.37 * sequence_index + {"train": 0.0, "val": 0.2, "test": 0.4}[split]
            base = []
            for channel in range(len(SVH_CHANNEL_NAMES)):
                slow = 0.46 + 0.28 * np.sin(2.0 * np.pi * (0.22 + 0.025 * channel) * t + phase + channel * 0.13)
                fast = 0.08 * np.sin(2.0 * np.pi * (0.65 + 0.03 * channel) * t + channel * 0.21)
                transition = 0.12 * np.tanh(3.0 * np.sin(2.0 * np.pi * 0.09 * t + phase))
                base.append(slow + fast + transition)
            controls = np.stack(base, axis=1)
            controls += rng.normal(0.0, 0.004, size=controls.shape)
            controls = np.clip(controls, 0.0, 1.0).astype(np.float32)
            sequence_id = f"synthetic_{split}_{sequence_index:02d}"
            relative_path = Path("sequences") / split / f"{sequence_id}.npz"
            path = output_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                schema_version=np.asarray("intent-sequence-v1"),
                source=np.asarray("synthetic_smoke"),
                sequence_id=np.asarray(sequence_id),
                split=np.asarray(split),
                subject=np.asarray(sequence_id),
                fps=np.asarray(fps, dtype=np.float32),
                frame_ids=np.arange(length, dtype=np.int32),
                timestamps_s=t,
                landmarks_3d=np.empty((length, 0, 3), dtype=np.float32),
                landmarks_2d=np.empty((length, 0, 2), dtype=np.float32),
                controls_9ch=controls,
                feature_summary=np.empty((length, 0), dtype=np.float32),
                action_labels=np.full(length, -1, dtype=np.int16),
            )
            entries.append(
                {
                    "path": relative_path.as_posix(),
                    "sequence_id": sequence_id,
                    "subject": sequence_id,
                    "split": split,
                    "frames": length,
                    "sha256": _sha256(path),
                }
            )

    manifest = {
        "schema_version": "intent-dataset-manifest-v1",
        "dataset": "synthetic_smoke",
        "synthetic": True,
        "research_claims_allowed": False,
        "warning": "仅用于验证预处理、训练和评测代码可运行，不得用于论文效果结论。",
        "fps": fps,
        "split_policy": "disjoint_synthetic_sequences",
        "split_counts": {
            split: {
                "sequences": count,
                "frames": count * length,
            }
            for split, count in split_layout.items()
        },
        "sequences": entries,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
