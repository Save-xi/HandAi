"""验证 M0 核对能识别数据损坏、跨段和切分泄漏。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/intent_prediction/scripts"))
from audit_keypoint_data import audit_h2o, contiguous_runs  # noqa: E402


@pytest.fixture
def tiny_dataset(tmp_path):
    spec = json.loads((ROOT / "experiments/intent_prediction/configs/keypoint_m0.json").read_text(encoding="utf-8"))
    spec["data"]["legacy_min_segment_frames"] = 4
    spec["time"]["history_frames"] = 3
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    entries = []
    for subject, split, n, scale in (("subject1", "train", 14, 0.1), ("subject3", "val", 6, 10.0), ("subject4", "test", 3, 20.0)):
        pose_dir = raw / subject / "h1/0/cam4/hand_pose"
        pose_dir.mkdir(parents=True)
        xyz = np.zeros((21, 3), dtype=np.float32)
        xyz[:, 2] = 1.0
        xyz[9, 1] = scale
        xyz[5, 0], xyz[17, 0] = -scale / 2, scale / 2
        for i in range(n):
            valid = not (subject == "subject1" and i == 5)
            values = np.r_[0.0, np.zeros(63), float(valid), xyz.ravel()]
            np.savetxt(pose_dir / f"{i:06d}.txt", values[None], fmt="%.17g")
        ranges = [(0, 5), (6, 14)] if subject == "subject1" else ([(0, n)] if n >= 4 else [])
        for segment, (start, stop) in enumerate(ranges):
            ids = np.arange(start, stop, dtype=np.int32)
            sequence_id = f"{subject}_h1_0_cam4_segment{segment:03d}"
            rel = f"sequences/{split}/{sequence_id}.npz"
            path = processed / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            points = np.repeat(xyz[None], len(ids), axis=0)
            np.savez(path, sequence_id=sequence_id, subject=subject, split=split, fps=30.0,
                     frame_ids=ids, timestamps_s=ids / 30.0, landmarks_3d=points,
                     landmarks_2d=points[:, :, :2] / np.float32(scale), controls_9ch=np.zeros((len(ids), 9)))
            entries.append(dict(path=rel, sequence_id=sequence_id, subject=subject, split=split,
                                action="h1", take="0", frames=len(ids), first_frame=start, last_frame=stop - 1))
    manifest = processed / "manifest.json"
    manifest.write_text(json.dumps({"sequences": entries}), encoding="utf-8")
    return raw, processed, spec, entries


def test_reconcile_exclusions_time_and_train_only_scale(tiny_dataset):
    raw, processed, spec, _ = tiny_dataset
    result = audit_h2o(raw, processed, spec)
    assert result["status"] == "passed"
    assert result["raw"]["totals"]["pose_files"] == 23
    assert result["raw"]["totals"]["unannotated_right_frames"] == 1
    assert result["raw"]["totals"]["short_segment_frames"] == 3
    assert result["raw"]["totals"]["compared_frames"] == 19
    assert result["normalization"]["s_min_m"] == pytest.approx(0.001)
    assert result["numeric_comparison"]["raw_to_npz_max_abs_delta_m_after_float32"] == 0
    # train 两段各自预热，不把无标注帧两边拼接；50ms 需要末尾两帧。
    assert result["processed"]["splits"]["train"]["history_ready_anchors"] == 9
    assert result["processed"]["splits"]["train"]["targets_50ms"] == 5


@pytest.mark.parametrize("field,kind", [("landmarks_3d", "raw_xyz_mismatch"), ("timestamps_s", "timestamp_or_fps")])
def test_detect_numerical_corruption(tiny_dataset, field, kind):
    raw, processed, spec, entries = tiny_dataset
    path = processed / entries[0]["path"]
    with np.load(path) as source:
        data = {key: source[key] for key in source.files}
    data[field] = data[field].copy()
    data[field].flat[-1] += 0.01
    np.savez(path, **data)
    result = audit_h2o(raw, processed, spec)
    assert result["status"] == "data_issues_found"
    assert kind in result["issues"]


def test_detect_overlapping_frames_and_subject_split(tiny_dataset):
    raw, processed, spec, entries = tiny_dataset
    manifest = processed / "manifest.json"
    manifest.write_text(json.dumps({"sequences": entries + [entries[0], dict(entries[0], split="test")]}), encoding="utf-8")
    issues = audit_h2o(raw, processed, spec)["issues"]
    assert "overlapping_raw_frame" in issues
    assert "subject_split" in issues


def test_missing_valid_segment_is_not_short_segment(tiny_dataset):
    raw, processed, spec, entries = tiny_dataset
    (processed / "manifest.json").write_text(json.dumps({"sequences": entries[1:]}), encoding="utf-8")
    result = audit_h2o(raw, processed, spec)
    assert result["raw"]["unexplained_exclusions"][0]["frames"] == 5
    assert "unlisted_npz" in result["issues"]
    assert "unexplained_exclusion" in result["issues"]


def test_invalid_frame_and_gap_break_segments():
    assert contiguous_runs(np.array([], dtype=int)) == []
    assert [part.tolist() for part in contiguous_runs(np.array([0, 1, 3, 6, 7]))] == [[0, 1], [3], [6, 7]]
