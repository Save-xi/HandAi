"""M0：只读核对 H2O 21 点、原始帧、切分和时间；不训练，不修改数据。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def contiguous_runs(frame_ids: np.ndarray) -> list[np.ndarray]:
    """返回已排序帧号的连续段；显式无效帧应在调用前移除。"""
    return [part for part in np.split(frame_ids, np.flatnonzero(np.diff(frame_ids) != 1) + 1) if len(part)]


def palm_lengths(points: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.linalg.norm(points[:, indices[1]] - points[:, indices[0]], axis=-1)


def fit_scale_floor(train_points: np.ndarray, config: dict) -> float:
    values = palm_lengths(train_points, config["palm_length_indices"])
    positive = values[np.isfinite(values) & (values > 0)]
    if not len(positive):
        raise ValueError("训练集没有有限正掌长")
    return float(np.median(positive) * config["s_min_fraction_of_positive_palm_median"])


def distribution(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return dict(zip(("min", "p01", "median", "p99", "max"),
                    map(float, np.quantile(values, [0, 0.01, 0.5, 0.99, 1]))), count=len(values))


def audit_h2o(raw_root: Path, processed_root: Path, spec: dict) -> dict:
    manifest = json.loads((processed_root / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["sequences"]
    fps = float(spec["time"]["fps"])
    history = int(spec["time"]["history_frames"])
    min_segment = int(spec["data"]["legacy_min_segment_frames"])
    if fps <= 0 or history < 1 or min_segment < 1:
        raise ValueError("fps、历史长度和旧片段下限必须为正")
    if spec["time"]["anchor_stride"] != 1:
        raise ValueError("M0 潜在锚点清点采用 stride=1；抽样策略由 M1 实现")
    subjects = {subject: split for split, group in spec["splits"].items() for subject in group}
    if len(subjects) != sum(map(len, spec["splits"].values())):
        raise ValueError("同一人员出现在多个切分中")
    issues: dict[str, dict] = {}

    def issue(kind: str, example: str) -> None:
        item = issues.setdefault(kind, {"count": 0, "examples": []})
        item["count"] += 1
        if len(item["examples"]) < 5:
            item["examples"].append(example)

    groups = defaultdict(list)
    points_by_split = defaultdict(list)
    counts = {split: Counter() for split in spec["splits"]}
    rows = []
    seen_ids, seen_paths, seen_frames = set(), set(), set()
    max_time_error = max_legacy_xy_error = 0.0
    for entry in entries:
        sequence_id = entry["sequence_id"]
        path = (processed_root / entry["path"]).resolve()
        if not path.is_relative_to(processed_root.resolve()):
            raise ValueError(f"数据路径越出指定目录：{path}")
        if sequence_id in seen_ids or path in seen_paths:
            issue("duplicate_sequence", sequence_id)
        seen_ids.add(sequence_id)
        seen_paths.add(path)
        split, subject = entry["split"], entry["subject"]
        if subjects.get(subject) != split:
            issue("subject_split", sequence_id)
            continue
        with np.load(path, allow_pickle=False) as data:
            ids = data["frame_ids"]
            times = data["timestamps_s"]
            xyz = data["landmarks_3d"].astype(np.float64)
            xy = data["landmarks_2d"]
            controls = data["controls_9ch"]
            n = len(ids)
            if n < 1 or ids.shape != (n,) or times.shape != (n,) or xyz.shape != (n, 21, 3):
                issue("keypoint_shape", sequence_id)
                continue
            if xy.shape != (n, 21, 2) or controls.shape != (n, 9):
                issue("legacy_shape", sequence_id)
                continue
            if not all(np.isfinite(a).all() for a in (xyz, xy, controls, times)):
                issue("nonfinite_array", sequence_id)
                continue
            if ids.dtype.kind not in "iu" or np.any(np.diff(ids) != 1):
                issue("noncontiguous_frame_ids", sequence_id)
            time_error = float(np.max(np.abs(times - ids / fps)))
            max_time_error = max(max_time_error, time_error)
            if time_error > 1e-9 or np.any(np.diff(times) <= 0) or float(data["fps"]) != fps:
                issue("timestamp_or_fps", sequence_id)
            for key in ("sequence_id", "subject", "split"):
                if str(data[key].item()) != str(entry[key]):
                    issue("npz_metadata", f"{sequence_id}:{key}")
            if (n, int(ids[0]), int(ids[-1])) != (entry["frames"], entry["first_frame"], entry["last_frame"]):
                issue("manifest_frame_count", sequence_id)
            if np.any((controls < 0) | (controls > 1)):
                issue("legacy_control_range", sequence_id)
            # 独立复算旧 XY 公式；它不是像素坐标或针孔投影。
            old_scale = np.linalg.norm(xyz[:, 9, :2] - xyz[:, 0, :2], axis=1)
            width = np.linalg.norm(xyz[:, 5, :2] - xyz[:, 17, :2], axis=1)
            old_scale = np.where(old_scale > 1e-8, old_scale, width)
            if np.any(old_scale <= 1e-8):
                issue("legacy_degenerate_xy", sequence_id)
            else:
                expected_xy = ((xyz[:, :, :2] - xyz[:, :1, :2]) / old_scale[:, None, None]).astype(np.float32)
                error = float(np.max(np.abs(expected_xy - xy)))
                max_legacy_xy_error = max(max_legacy_xy_error, error)
                if error > 1e-5:
                    issue("legacy_xy_formula", sequence_id)
            take = f"{subject}/{entry['action']}/{entry['take']}/cam4"
            for frame_id in ids:
                key = (take, int(frame_id))
                if key in seen_frames:
                    issue("overlapping_raw_frame", f"{take}:{frame_id}")
                seen_frames.add(key)
            groups[take].append((ids.copy(), xyz.copy()))
            points_by_split[split].append(xyz.copy())
            row = {"sequence_id": sequence_id, "split": split, "subject": subject,
                   "take": take, "frames": n, "first_frame": int(ids[0]), "last_frame": int(ids[-1]),
                   "history_ready_anchors": max(0, n - history + 1), "targets_available": {}}
            # 此处只清点已有连续、有限标注的潜在锚点；不是 M1 建窗器。
            anchors = times[history - 1:]
            for horizon in spec["time"]["evaluation_horizons_ms"]:
                row["targets_available"][str(horizon)] = int(np.sum(anchors + horizon / 1000 <= times[-1] + 1e-9))
            row["full_future_path_anchors"] = int(np.sum(
                anchors + max(spec["time"]["forecast_steps"]) / fps <= times[-1] + 1e-9))
            rows.append(row)
            counts[split].update(sequences=1, frames=n, history_ready_anchors=row["history_ready_anchors"],
                                 full_future_path_anchors=row["full_future_path_anchors"])
            for horizon, value in row["targets_available"].items():
                counts[split][f"targets_{horizon}ms"] += value
    listed = seen_paths
    actual = {p.resolve() for p in (processed_root / "sequences").rglob("*.npz")}
    for path in actual - listed:
        issue("unlisted_npz", str(path.relative_to(processed_root)))
    raw_counts = Counter()
    raw_splits = {split: Counter() for split in spec["splits"]}
    short_runs, unexpected_exclusions = [], []
    max_raw_delta = 0.0
    take_dirs = sorted(raw_root.glob("subject*/*/*/cam4/hand_pose"))
    if not take_dirs or not entries:
        raise ValueError("原始 take 或预处理序列为空")
    remaining_takes = set(groups)
    for number, pose_dir in enumerate(take_dirs, 1):
        cam_dir = pose_dir.parent
        take = cam_dir.relative_to(raw_root).as_posix()
        subject = take.split("/")[0]
        if subject not in subjects:
            issue("unknown_raw_subject", take)
            continue
        raw_counts["takes"] += 1
        for name, key in (("cam_intrinsics.txt", "takes_with_intrinsics"), ("rgb", "takes_with_rgb")):
            raw_counts[key] += int((cam_dir / name).exists())
        valid_ids, valid_points = [], []
        per_take = Counter()
        files = sorted((p for p in pose_dir.glob("*.txt") if p.stem.isdigit()), key=lambda p: int(p.stem))
        for pose_file in files:
            per_take["pose_files"] += 1
            try:
                values = np.asarray(pose_file.read_text(encoding="utf-8").split(), dtype=np.float64)
                if values.shape != (128,) or values[64] not in (0, 1) or not np.isfinite(values[65:]).all():
                    raise ValueError("128 个数、0/1 有效位或有限右手点不满足")
            except (ValueError, OSError) as exc:
                per_take["malformed_frames"] += 1
                issue("malformed_raw_pose", f"{take}/{pose_file.name}: {exc}")
                continue
            if values[64] == 0:
                per_take["unannotated_right_frames"] += 1
                continue
            valid_ids.append(int(pose_file.stem))
            valid_points.append(values[65:].reshape(21, 3).astype(np.float32))
        ids = np.asarray(valid_ids, dtype=np.int64)
        per_take["annotated_right_frames"] = len(ids)
        runs = contiguous_runs(ids)
        expected_ids = set()
        for run in runs:
            if len(run) < min_segment:
                short_runs.append({"take": take, "frames": len(run), "first_frame": int(run[0]), "last_frame": int(run[-1])})
                per_take["short_segment_frames"] += len(run)
                per_take["short_segments"] += 1
            else:
                expected_ids.update(map(int, run))
        kept = groups.get(take, [])
        kept_ids = np.concatenate([item[0] for item in kept]) if kept else np.empty(0, dtype=np.int64)
        missing = expected_ids - set(map(int, kept_ids))
        extra = set(map(int, kept_ids)) - expected_ids
        if missing:
            unexpected_exclusions.append({"take": take, "frames": len(missing), "first_frame": min(missing)})
            issue("unexplained_exclusion", take)
        if extra:
            issue("unexpected_retained_frame", f"{take}: {sorted(extra)[:5]}")
        raw_lookup = {frame_id: point for frame_id, point in zip(valid_ids, valid_points)}
        for kept_frame_ids, kept_xyz in kept:
            for frame_id, point in zip(kept_frame_ids, kept_xyz):
                original = raw_lookup.get(int(frame_id))
                if original is None:
                    issue("missing_raw_correspondence", f"{take}:{frame_id}")
                else:
                    delta = float(np.max(np.abs(original - point)))
                    max_raw_delta = max(max_raw_delta, delta)
                    per_take["compared_frames"] += 1
                    if delta > 1e-7:
                        issue("raw_xyz_mismatch", f"{take}:{frame_id}")
        raw_counts.update(per_take)
        raw_splits[subjects[subject]].update(per_take)
        remaining_takes.discard(take)
        if number % 40 == 0:
            print(f"原始帧核对：{number}/{len(take_dirs)} 个 take", flush=True)
    for take in remaining_takes:
        issue("missing_raw_take", take)
    norm = spec["normalization"]
    if norm["s_min_fit_split"] != "train":
        raise ValueError("尺度下限只能使用训练集拟合")
    train = np.concatenate(points_by_split["train"])
    floor = fit_scale_floor(train, norm)
    scale_stats = {}
    for split, parts in points_by_split.items():
        xyz = np.concatenate(parts)
        palm = palm_lengths(xyz, norm["palm_length_indices"])
        width = palm_lengths(xyz, norm["fallback_width_indices"])
        length_ok, width_ok = np.isfinite(palm) & (palm >= floor), np.isfinite(width) & (width >= floor)
        scale_stats[split] = {
            "palm_length_m": distribution(palm), "palm_width_m": distribution(width),
            "width_fallback_frames": int(np.sum(~length_ok & width_ok)),
            "invalid_scale_frames": int(np.sum(~length_ok & ~width_ok)),
            "nonpositive_z_frames": int(np.sum(np.any(xyz[:, :, 2] <= 0, axis=1)))
        }
    return {
        "status": "passed" if not issues else "data_issues_found", "task_id": spec["task_id"],
        "raw_root": str(raw_root), "processed_root": str(processed_root),
        "issues": issues, "processed": {"sequences": len(rows), "splits": counts},
        "raw": {"totals": raw_counts, "splits": raw_splits, "short_runs": short_runs,
                "unexplained_exclusions": unexpected_exclusions},
        "numeric_comparison": {"raw_to_npz_max_abs_delta_m_after_float32": max_raw_delta,
                               "timestamp_vs_frame_id_div_fps_max_abs_delta_s": max_time_error,
                               "legacy_xy_formula_max_abs_delta": max_legacy_xy_error},
        "normalization": {"s_min_m": floor, "fit_split": "train", "splits": scale_stats},
        "coverage_note": f"连续标注潜在锚点，尚未建立训练窗口；history={history}，stride=1，"
                         f"完整未来路径={max(spec['time']['forecast_steps'])}步；逐帧不是独立样本。",
        "sequences": rows
    }


def video_inventory(root: Path) -> dict:
    import cv2

    rows = []
    for path in sorted(root.glob("*.mp4")):
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rows.append({"file": path.name, "opened": bool(cap.isOpened()), "nominal_fps": fps,
                     "container_frame_count": count, "nominal_duration_s": count / fps if fps > 0 else None,
                     "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     "role": "historical_development" if path.stem in [f"v{i}" for i in range(1, 8)] else "pilot_or_mirror_check"})
        cap.release()
    return {"root": str(root), "scope": "container_metadata_only_not_full_decode_or_PTS_validation",
            "participant_session_ids": "not_established_by_this_inventory", "files": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "experiments/intent_prediction/configs/keypoint_m0.json")
    parser.add_argument("--raw-root", type=Path, help="覆盖配置中的原始数据位置")
    parser.add_argument("--processed-root", type=Path, help="覆盖配置中的 NPZ 位置")
    parser.add_argument("--video-root", type=Path, help="可选：只读视频容器信息，不运行检测器")
    parser.add_argument("--output", type=Path, required=True, help="JSON 报告路径；可以正常重跑覆盖报告")
    args = parser.parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8"))
    raw = (args.raw_root or PROJECT_ROOT / spec["data"]["raw_root"]).resolve()
    processed = (args.processed_root or PROJECT_ROOT / spec["data"]["processed_root"]).resolve()
    report = audit_h2o(raw, processed, spec)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["specification"] = spec
    if args.video_root:
        report["camera_inventory"] = video_inventory(args.video_root)
    # 写入对象仅为报告；防止误把 --output 指向待核对的数据。
    output = args.output.resolve()
    if output.is_relative_to(raw) or output.is_relative_to(processed) or output == args.config.resolve():
        raise ValueError("报告路径不能覆盖输入配置或写入原始/预处理数据目录")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"M0 数据核对：{report['status']}；报告：{output}")
    return int(bool(report["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
