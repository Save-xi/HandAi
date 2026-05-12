from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import yaml


DEFAULT_JOINT_COUNT = 21


@dataclass(frozen=True)
class LoadedConfig:
    """配置内容和配置文件路径打包保存，便于解析相对路径。"""

    data: Dict[str, Any]
    path: Path


@dataclass(frozen=True)
class SplitAnnotations:
    """一个 FreiHAND split 中与评估相关的 annotation。"""

    name: str
    sample_ids: list[str]
    K: np.ndarray | None
    xyz: np.ndarray | None
    scale: np.ndarray | None


def load_config(config_path: str | Path) -> LoadedConfig:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return LoadedConfig(data=data, path=path)


def resolve_path(config: LoadedConfig, value: str | Path) -> Path:
    """把 yaml 里的路径解析成绝对路径。

    相对路径按配置文件所在目录解析，这样脚本从任意工作目录启动都能找到文件。
    """

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config.path.parent / path).resolve()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    ensure_parent_dir(path)
    path.write_text(text, encoding="utf-8")


def sample_id(index: int) -> str:
    return f"{index:08d}"


def normalize_sample_mapping(raw: Any) -> Dict[str, Any]:
    """把 FreiHAND 常见的 list annotation 转成 00000000 风格的样本字典。"""

    if isinstance(raw, list):
        return {sample_id(i): value for i, value in enumerate(raw)}
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    raise TypeError(f"annotation root must be list or dict, got {type(raw).__name__}")


def _annotation_path(config: LoadedConfig, split: str, key: str) -> Path | None:
    value = config.data.get("annotations", {}).get(split, {}).get(key)
    if not value:
        return None
    return resolve_path(config, value)


def split_config_path(config: LoadedConfig, split: str, key: str) -> Path | None:
    """读取某个 split 下的路径配置，例如 image_root。"""

    return _annotation_path(config, split, key)


def _load_optional_array(config: LoadedConfig, split: str, key: str) -> tuple[list[str], np.ndarray | None]:
    path = _annotation_path(config, split, key)
    if path is None or not path.exists():
        return [], None
    mapping = normalize_sample_mapping(read_json(path))
    ids = list(mapping.keys())
    try:
        array = np.asarray([mapping[sid] for sid in ids], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{split}.{key} contains non-numeric values: {path}") from exc
    return ids, array


def _shared_order(*id_lists: Iterable[str]) -> list[str]:
    lists = [list(ids) for ids in id_lists if ids]
    if not lists:
        return []
    shared = set(lists[0])
    for ids in lists[1:]:
        shared &= set(ids)
    return [sid for sid in lists[0] if sid in shared]


def _reorder(array: np.ndarray | None, source_ids: list[str], target_ids: list[str]) -> np.ndarray | None:
    if array is None:
        return None
    index = {sid: i for i, sid in enumerate(source_ids)}
    return np.asarray([array[index[sid]] for sid in target_ids], dtype=float)


def load_split_annotations(
    config: LoadedConfig,
    split: str,
    *,
    require_xyz: bool = False,
    require_K: bool = False,
) -> SplitAnnotations:
    """读取一个 split 的 K / xyz / scale，并按共同样本 id 对齐。"""

    K_ids, K = _load_optional_array(config, split, "K")
    xyz_ids, xyz = _load_optional_array(config, split, "xyz")
    scale_ids, scale = _load_optional_array(config, split, "scale")

    if require_K and K is None:
        raise FileNotFoundError(f"missing K annotation for split '{split}'")
    if require_xyz and xyz is None:
        raise FileNotFoundError(f"missing xyz annotation for split '{split}'")

    ids = _shared_order(K_ids, xyz_ids, scale_ids)
    if not ids:
        ids = K_ids or xyz_ids or scale_ids

    return SplitAnnotations(
        name=split,
        sample_ids=ids,
        K=_reorder(K, K_ids, ids),
        xyz=_reorder(xyz, xyz_ids, ids),
        scale=_reorder(scale, scale_ids, ids),
    )


def contains_none(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, list):
        return any(contains_none(item) for item in value)
    if isinstance(value, dict):
        return any(contains_none(item) for item in value.values())
    return False


def _shape_of(value: Any) -> tuple[int, ...] | None:
    try:
        return tuple(np.asarray(value, dtype=float).shape)
    except (TypeError, ValueError):
        return None


def _has_nan(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    if array.dtype.kind not in {"f", "i", "u"}:
        return False
    return bool(np.isnan(array).any())


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def inspect_annotation_file(path: Path | None, expected_shape: tuple[int, ...] | None) -> Dict[str, Any]:
    if path is None:
        return {"exists": False, "path": None, "error": "path is not configured"}
    info: Dict[str, Any] = {"exists": path.exists(), "path": str(path)}
    if not path.exists():
        info["error"] = "file not found"
        return info

    try:
        mapping = normalize_sample_mapping(read_json(path))
    except Exception as exc:  # noqa: BLE001 - inspection should report bad files instead of crashing.
        info.update({"error": f"{type(exc).__name__}: {exc}", "count": 0})
        return info

    shape_counts: Dict[str, int] = {}
    invalid_shape_count = 0
    none_count = 0
    nan_count = 0
    numeric_count = 0
    for value in mapping.values():
        shape = _shape_of(value)
        key = "invalid" if shape is None else str(shape)
        shape_counts[key] = shape_counts.get(key, 0) + 1
        if expected_shape is not None and shape != expected_shape:
            invalid_shape_count += 1
        if contains_none(value):
            none_count += 1
        if _has_nan(value):
            nan_count += 1
        if shape is not None or _is_finite_number(value):
            numeric_count += 1

    info.update(
        {
            "count": len(mapping),
            "shape_counts": shape_counts,
            "expected_shape": str(expected_shape) if expected_shape is not None else None,
            "invalid_shape_count": invalid_shape_count,
            "contains_none_count": none_count,
            "contains_nan_count": nan_count,
            "numeric_sample_count": numeric_count,
        }
    )
    return info


def inspect_configured_annotations(config: LoadedConfig) -> Dict[str, Any]:
    joint_count = int(config.data.get("metrics", {}).get("joint_count", DEFAULT_JOINT_COUNT))
    result: Dict[str, Any] = {}
    for split in config.data.get("annotations", {}):
        K_path = _annotation_path(config, split, "K")
        xyz_path = _annotation_path(config, split, "xyz")
        scale_path = _annotation_path(config, split, "scale")
        K_info = inspect_annotation_file(K_path, (3, 3))
        xyz_info = inspect_annotation_file(xyz_path, (joint_count, 3))
        scale_info = inspect_annotation_file(scale_path, ())
        xyz_count = int(xyz_info.get("count", 0) or 0)
        result[split] = {
            "K": K_info,
            "xyz": xyz_info,
            "scale": scale_info,
            "sample_count": max(
                int(K_info.get("count", 0) or 0),
                int(xyz_info.get("count", 0) or 0),
                int(scale_info.get("count", 0) or 0),
            ),
            "all_xyz_samples_have_21_keypoints": bool(
                xyz_count > 0 and int(xyz_info.get("invalid_shape_count", 0) or 0) == 0
            ),
        }
    return result


def limit_split(split: SplitAnnotations, max_samples: int | None) -> SplitAnnotations:
    if max_samples is None:
        return split
    max_samples = max(0, int(max_samples))
    return SplitAnnotations(
        name=split.name,
        sample_ids=split.sample_ids[:max_samples],
        K=None if split.K is None else split.K[:max_samples],
        xyz=None if split.xyz is None else split.xyz[:max_samples],
        scale=None if split.scale is None else split.scale[:max_samples],
    )


def load_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, Mapping):
        raise TypeError("predictions.json root must be an object keyed by sample id")
    return {str(key): dict(value) for key, value in data.items()}
