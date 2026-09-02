from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "intent_prediction"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from intent_prediction import camera_domain_blind_freeze as blind_freeze  # noqa: E402
from intent_prediction import camera_domain_eval as camera_eval  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_video(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def _locked_environment() -> dict:
    return {
        "python": "3.10.0",
        "python_executable": "C:/miniconda/envs/handai-intent-prediction/python.exe",
        "python_executable_sha256": "1" * 64,
        "python_prefix": "C:/miniconda/envs/handai-intent-prediction",
        "conda_prefix": "C:/miniconda/envs/handai-intent-prediction",
        "conda_default_env": "handai-intent-prediction",
        "conda_environment_name": "handai-intent-prediction",
        "conda_executable": "C:/miniconda/Scripts/conda.exe",
        "conda_explicit_spec_sha256": "2" * 64,
        "conda_explicit_spec_line_count": 42,
        "conda_explicit_spec_error": None,
        "pip_freeze_sha256": "3" * 64,
        "pip_freeze_line_count": 17,
        "pip_freeze_error": None,
        "cuda_available": True,
    }


def _video_summary(
    video_id: str,
    *,
    baseline_jsonl_path: Path,
    control_ready_fraction: float = 1.0,
    svh_valid_fraction: float = 1.0,
    longest_invalid_run_ms: float = 0.0,
    stable_gesture_counts: dict[str, int] | None = None,
) -> dict:
    return {
        "video_id": video_id,
        "video_path": f"D:/blind/{video_id}.mp4",
        "video_sha256": "a" * 64,
        "video_bytes": 1,
        "baseline_jsonl_path": str(baseline_jsonl_path),
        "baseline_jsonl_sha256": "b" * 64,
        "metadata": {
            "decoded_frame_count": 6,
            "metadata_frame_count": 6,
            "nominal_fps": 10.0,
            "width": 1280,
            "height": 720,
        },
        "source_timeline": {
            "duration_ms": 500.0,
            "duration_based_fps": 10.0,
            "timestamp_source_counts": {"container_pts_ms": 6},
        },
        "observation_counts": {
            "control_ready_fraction": control_ready_fraction,
            "svh_valid_fraction": svh_valid_fraction,
            "stable_gesture_counts": stable_gesture_counts or {},
        },
        "observation_continuity": {
            "longest_invalid_run_ms": longest_invalid_run_ms,
            "longest_ready_run_ms": 500.0,
            "starts_ready": True,
            "ends_ready": True,
            "recovery_count": 0,
            "runs": [],
        },
    }


def _common_video_rule(profile: str) -> dict:
    return {
        "profile": profile,
        "minimum_duration_ms": 400.0,
        "maximum_duration_ms": 600.0,
        "minimum_nominal_fps": 9.0,
        "maximum_nominal_fps": 11.0,
        "minimum_duration_based_fps": 9.0,
        "maximum_duration_based_fps": 11.0,
        "minimum_width": 640,
        "minimum_height": 480,
        "maximum_timestamp_fallback_fraction": 0.0,
        "minimum_control_ready_fraction": 0.0,
        "minimum_svh_valid_fraction": 0.0,
        "require_metadata_frame_count_match": True,
    }


def _clean_b1_gate() -> dict:
    rule = _common_video_rule("clean")
    rule.update(
        {
            "minimum_control_ready_fraction": 0.8,
            "minimum_svh_valid_fraction": 0.8,
            "maximum_invalid_run_duration_ms": 120.0,
            "minimum_stable_gesture_frames": {"open": 2, "fist": 1},
        }
    )
    return {"video_requirements": {"B1": rule}}


def _invalid_rule(windows: list[dict]) -> dict:
    rule = _common_video_rule("intentional_invalid")
    rule.update(
        {
            "maximum_control_ready_fraction": 1.0,
            "maximum_svh_valid_fraction": 1.0,
            "windows": windows,
        }
    )
    return rule


def _write_baseline_rows(path: Path, readiness: list[bool]) -> None:
    rows = [
        {
            "timestamp": 1_000.0 + index * 0.1,
            "control_ready": ready,
            "svh_preview": {"valid": ready},
        }
        for index, ready in enumerate(readiness)
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def _write_task_rows(
    path: Path,
    *,
    gestures: list[str],
    grasp_close: list[float] | None = None,
) -> None:
    rows = []
    for index, gesture in enumerate(gestures):
        control = {
            "grasp_close": (
                grasp_close[index] if grasp_close is not None else 0.5
            )
        }
        rows.append(
            {
                "timestamp": 1_000.0 + index * 0.1,
                "control_ready": True,
                "gesture_stable": gesture,
                "control_representation": control,
                "svh_preview": {"valid": True},
            }
        )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _decision_gate() -> dict:
    return {
        "primary_delays_ms": [50.0, 100.0],
        "minimum_primary_gated_rmse_improvement_percent": 3.0,
        "minimum_primary_dynamic_gated_rmse_improvement_percent": 4.0,
        "minimum_primary_gated_p95_improvement_percent": 2.0,
        "minimum_each_video_gated_rmse_improvement_percent": 1.0,
        "minimum_each_video_gated_p95_improvement_percent": 1.0,
        "minimum_each_video_evaluable_rows": 5,
        "minimum_primary_conditional_prediction_available_fraction": 0.9,
        "minimum_primary_end_to_end_prediction_coverage_fraction": 0.85,
        "maximum_primary_range_violation_rate": 0.0,
        "require_live_runtime": False,
    }


def _algorithm() -> dict:
    def per_video_summary() -> dict:
        return {
            "source": {"evaluable_rows": 8},
            "primary_delay_summary": {
                "improvement_percent_vs_hold": {"gated_rmse": 2.0, "gated_p95": 2.0}
            },
        }

    return {
        "status": "evaluated",
        "source_errors": [],
        "aggregate": {
            "primary_delays_ms": [50.0, 100.0],
            "primary_delay_summary": {
                "improvement_percent_vs_hold": {"gated_rmse": 4.0, "gated_p95": 3.0},
                "dynamic_q90": {
                    "improvement_percent_vs_hold": {"gated_rmse": 5.0}
                },
                "conditional_prediction_available_fraction": 0.95,
                "end_to_end_prediction_coverage_fraction": 0.9,
                "methods": {"gated": {"range_violation_rate": 0.0}},
            }
        },
        "per_video": {"B1": per_video_summary(), "B2": per_video_summary()},
    }


def test_blind_inventory_rejects_duplicate_b_content_and_development_sha(tmp_path):
    b1 = _write_video(tmp_path / "B1.mp4", b"same-blind-content")
    b2 = _write_video(tmp_path / "B2.mp4", b"same-blind-content")
    forbidden = _sha256(b1)

    with pytest.raises(ValueError, match="开发集"):
        camera_eval.verify_blind_video_inventory(
            [camera_eval.VideoSpec("B1", b1)],
            blind_policy={"forbidden_video_sha256": [forbidden]},
        )

    with pytest.raises(ValueError, match="SHA 重复"):
        camera_eval.verify_blind_video_inventory(
            [camera_eval.VideoSpec("B1", b1), camera_eval.VideoSpec("B2", b2)],
            blind_policy={"forbidden_video_sha256": ["0" * 64]},
        )


def test_b1_clean_gate_requires_continuity_and_declared_gestures(tmp_path):
    baseline_path = tmp_path / "B1.jsonl"
    summary = _video_summary(
        "B1",
        baseline_jsonl_path=baseline_path,
        control_ready_fraction=1.0,
        svh_valid_fraction=1.0,
        stable_gesture_counts={"open": 3, "fist": 2},
    )
    passed = camera_eval.evaluate_blind_input_gate([summary], gate=_clean_b1_gate())
    assert passed["passed"] is True, json.dumps(passed, ensure_ascii=False, sort_keys=True)
    assert passed["criteria"]["B1_invalid_run_duration"] is True
    assert passed["criteria"]["B1_gesture_open"] is True

    broken = deepcopy(summary)
    broken["observation_continuity"]["longest_invalid_run_ms"] = 121.0
    broken["observation_counts"]["stable_gesture_counts"]["fist"] = 0
    rejected = camera_eval.evaluate_blind_input_gate([broken], gate=_clean_b1_gate())
    assert rejected["passed"] is False
    assert rejected["criteria"]["B1_invalid_run_duration"] is False
    assert rejected["criteria"]["B1_gesture_fist"] is False


def test_clean_task_gate_checks_order_speed_and_continuous_grasp(tmp_path):
    b1_path = tmp_path / "B1.jsonl"
    _write_task_rows(
        b1_path,
        gestures=["open", "fist", "open", "fist", "open"],
    )
    b1 = _video_summary(
        "B1",
        baseline_jsonl_path=b1_path,
        stable_gesture_counts={"open": 3, "fist": 2},
    )
    b1_rule = _common_video_rule("clean")
    b1_rule.update(
        {
            "maximum_invalid_run_duration_ms": 100.0,
            "minimum_longest_ready_run_ms": 400.0,
            "minimum_stable_gesture_frames": {"open": 2, "fist": 2},
            "task_evidence": {
                "gesture_sequence": ["open", "fist", "open"],
                "minimum_gesture_sequence_occurrences": 2,
                "minimum_gesture_transition_count": 4,
                "maximum_median_gesture_transition_ms": 101.0,
            },
        }
    )
    passed = camera_eval.evaluate_blind_input_gate(
        [b1], gate={"video_requirements": {"B1": b1_rule}}
    )
    assert passed["passed"] is True, json.dumps(passed, ensure_ascii=False)
    assert passed["criteria"]["B1_task_gesture_sequence"] is True

    _write_task_rows(
        b1_path,
        gestures=["open", "fist", "fist", "fist", "open"],
    )
    rejected = camera_eval.evaluate_blind_input_gate(
        [b1], gate={"video_requirements": {"B1": b1_rule}}
    )
    assert rejected["passed"] is False
    assert rejected["criteria"]["B1_task_gesture_sequence"] is False

    b3_path = tmp_path / "B3.jsonl"
    _write_task_rows(
        b3_path,
        gestures=["fist"] * 5,
        grasp_close=[0.7, 0.95, 0.7, 0.96, 0.7],
    )
    b3 = _video_summary(
        "B3",
        baseline_jsonl_path=b3_path,
        stable_gesture_counts={"fist": 5},
    )
    b3_rule = _common_video_rule("clean")
    b3_rule.update(
        {
            "maximum_invalid_run_duration_ms": 100.0,
            "minimum_longest_ready_run_ms": 400.0,
            "minimum_stable_gesture_frames": {"fist": 5},
            "task_evidence": {
                "grasp_close_sequence": ["low", "high", "low"],
                "grasp_close_low_maximum": 0.75,
                "grasp_close_high_minimum": 0.9,
                "minimum_grasp_close_range": 0.2,
                "minimum_grasp_sequence_occurrences": 2,
            },
        }
    )
    grasp_passed = camera_eval.evaluate_blind_input_gate(
        [b3], gate={"video_requirements": {"B3": b3_rule}}
    )
    assert grasp_passed["passed"] is True
    assert grasp_passed["criteria"]["B3_task_grasp_close_sequence"] is True


def test_b5_b6_task_aware_invalid_and_recovery_windows_are_enforced(tmp_path):
    b5_path = tmp_path / "B5.jsonl"
    b6_path = tmp_path / "B6.jsonl"
    # B5: 正常控制 -> 人为遮挡/invalid -> 恢复；B6: 两次出画/回画 episode。
    _write_baseline_rows(b5_path, [True, True, False, False, True, True])
    _write_baseline_rows(b6_path, [False, True, False, True, False, True])
    b5 = _video_summary("B5", baseline_jsonl_path=b5_path)
    b6 = _video_summary("B6", baseline_jsonl_path=b6_path)
    gate = {
        "video_requirements": {
            "B5": _invalid_rule(
                [
                    {"name": "before_occlusion", "expected": "ready", "start_ms": 0, "end_ms": 200, "minimum_matching_fraction": 1.0},
                    {"name": "occlusion_safe_open", "expected": "invalid", "start_ms": 200, "end_ms": 400, "minimum_matching_fraction": 1.0},
                    {"name": "after_occlusion_recovery", "expected": "ready", "start_ms": 400, "end_ms": 600, "minimum_matching_fraction": 1.0},
                ]
            ),
            "B6": _invalid_rule(
                [
                    {"name": "exit_1", "expected": "invalid", "start_ms": 0, "end_ms": 100, "minimum_matching_fraction": 1.0},
                    {"name": "return_1", "expected": "ready", "start_ms": 100, "end_ms": 200, "minimum_matching_fraction": 1.0},
                    {"name": "exit_2", "expected": "invalid", "start_ms": 200, "end_ms": 300, "minimum_matching_fraction": 1.0},
                    {"name": "return_2", "expected": "ready", "start_ms": 300, "end_ms": 400, "minimum_matching_fraction": 1.0},
                    {"name": "exit_3", "expected": "invalid", "start_ms": 400, "end_ms": 500, "minimum_matching_fraction": 1.0},
                    {"name": "return_3", "expected": "ready", "start_ms": 500, "end_ms": 600, "minimum_matching_fraction": 1.0},
                ]
            ),
        }
    }

    passed = camera_eval.evaluate_blind_input_gate([b5, b6], gate=gate)
    assert passed["passed"] is True, json.dumps(passed, ensure_ascii=False, sort_keys=True)
    assert passed["details"]["B5"]["windows"]["occlusion_safe_open"]["passed"] is True
    assert passed["details"]["B6"]["windows"]["return_3"]["passed"] is True

    _write_baseline_rows(b5_path, [True, True, True, False, True, True])
    rejected = camera_eval.evaluate_blind_input_gate([b5, b6], gate=gate)
    assert rejected["passed"] is False
    assert rejected["criteria"]["B5_window_occlusion_safe_open"] is False


def test_intentional_invalid_gate_enforces_episode_count_and_recovery(tmp_path):
    path = tmp_path / "B5.jsonl"
    _write_baseline_rows(path, [True, True, False, False, True, True])
    summary = _video_summary(
        "B5",
        baseline_jsonl_path=path,
        control_ready_fraction=2 / 3,
        svh_valid_fraction=2 / 3,
    )
    summary["observation_continuity"].update(
        {
            "starts_ready": True,
            "ends_ready": True,
            "recovery_count": 1,
            "runs": [
                {"ready": True, "duration_ms": 200.0},
                {"ready": False, "duration_ms": 200.0},
                {"ready": True, "duration_ms": 200.0},
            ],
        }
    )
    rule = _invalid_rule(
        [
            {"name": "before", "expected": "ready", "start_ms": 0, "end_ms": 200, "minimum_matching_fraction": 1.0},
            {"name": "invalid", "expected": "invalid", "start_ms": 200, "end_ms": 400, "minimum_matching_fraction": 1.0},
            {"name": "after", "expected": "ready", "start_ms": 400, "end_ms": 600, "minimum_matching_fraction": 1.0},
        ]
    )
    rule.update(
        {
            "required_invalid_episode_count": 1,
            "minimum_recovery_count": 1,
            "minimum_invalid_run_duration_ms": 150.0,
            "maximum_invalid_run_duration_ms": 250.0,
            "require_starts_ready": True,
            "require_ends_ready": True,
        }
    )
    passed = camera_eval.evaluate_blind_input_gate(
        [summary], gate={"video_requirements": {"B5": rule}}
    )
    assert passed["passed"] is True

    broken = deepcopy(summary)
    broken["observation_continuity"]["runs"].append(
        {"ready": False, "duration_ms": 180.0}
    )
    rejected = camera_eval.evaluate_blind_input_gate(
        [broken], gate={"video_requirements": {"B5": rule}}
    )
    assert rejected["passed"] is False
    assert rejected["criteria"]["B5_invalid_episode_count"] is False


@pytest.mark.parametrize(
    ("mutation", "criterion"),
    [
        (
            lambda value: value["aggregate"]["primary_delay_summary"][
                "improvement_percent_vs_hold"
            ].update({"gated_p95": 1.0}),
            "aggregate_gated_p95",
        ),
        (
            lambda value: value["per_video"]["B2"]["primary_delay_summary"][
                "improvement_percent_vs_hold"
            ].update({"gated_p95": 0.5}),
            "each_video_gated_p95",
        ),
        (
            lambda value: value["per_video"]["B2"]["primary_delay_summary"][
                "improvement_percent_vs_hold"
            ].update({"gated_rmse": 0.5}),
            "each_video_gated_rmse",
        ),
        (
            lambda value: value["aggregate"]["primary_delay_summary"]["methods"][
                "gated"
            ].update({"range_violation_rate": 0.01}),
            "range_violation",
        ),
    ],
)
def test_algorithm_gate_rejects_p95_worst_video_and_range_regressions(mutation, criterion):
    videos = [{"video_id": "B1"}, {"video_id": "B2"}]
    input_gate = {"criteria": {"B1_input": True, "B2_input": True}, "details": {}}
    accepted = camera_eval.evaluate_blind_decision(
        videos, _algorithm(), None, gate=_decision_gate(), input_gate=input_gate
    )
    assert accepted["status"] == "blind_gate_passed"
    assert accepted["branch"] == "keep_v2_shadow"

    altered = _algorithm()
    mutation(altered)
    rejected = camera_eval.evaluate_blind_decision(
        videos, altered, None, gate=_decision_gate(), input_gate=input_gate
    )
    assert rejected["status"] == "blind_gate_failed"
    assert rejected["branch"] == "v3_pre_registered_candidate"
    assert rejected["criteria"]["algorithm"][criterion] is False


def test_attempt_receipt_is_atomic_one_time_and_uses_default_path(
    tmp_path, monkeypatch
):
    state_root = tmp_path / "blind-state"
    monkeypatch.setattr(blind_freeze, "BLIND_FREEZE_STATE_ROOT", state_root)
    token = "a" * 64
    receipt_path = blind_freeze.default_attempt_receipt_path(token)
    assert receipt_path == (state_root / "attempt_receipts" / f"{token}.json").resolve()
    # manifest 放到哪里都不能改变同一确定性身份的 receipt。
    assert receipt_path == blind_freeze.default_attempt_receipt_path(token)
    video_path = _write_video(tmp_path / "B1.mp4", b"video")
    baseline_path = tmp_path / "B1.jsonl"
    baseline_path.write_text("{}\n", encoding="utf-8")
    freeze_state = {
        "path": str(tmp_path / "copied-anywhere.json"),
        "sha256": "c" * 64,
        "attempt_token": token,
        "attempt_identity_sha256": token,
        "git": {"revision": "d" * 40},
        "blind_inputs": {
            "videos": {
                "B1": {
                    "path": str(video_path),
                    "path_scope": "absolute",
                    "bytes": video_path.stat().st_size,
                    "sha256": _sha256(video_path),
                }
            }
        },
    }
    video_summaries = [
        {
            "video_id": "B1",
            "video_path": str(video_path),
            "video_sha256": _sha256(video_path),
            "video_bytes": video_path.stat().st_size,
            "baseline_jsonl_path": str(baseline_path),
            "baseline_jsonl_sha256": _sha256(baseline_path),
        }
    ]
    reserved = blind_freeze.reserve_blind_attempt(
        receipt_path=receipt_path,
        freeze_state=freeze_state,
        video_summaries=video_summaries,
    )
    assert reserved["status"] == "reserved"
    original_bytes = receipt_path.read_bytes()
    with pytest.raises(RuntimeError, match="已存在"):
        blind_freeze.reserve_blind_attempt(
            receipt_path=receipt_path,
            freeze_state=freeze_state,
            video_summaries=video_summaries,
        )
    assert receipt_path.read_bytes() == original_bytes
    with pytest.raises(ValueError, match="路径不可覆盖"):
        blind_freeze.reserve_blind_attempt(
            receipt_path=tmp_path / "alternate-receipt.json",
            freeze_state=freeze_state,
            video_summaries=video_summaries,
        )

    blind_freeze.finalize_blind_attempt(receipt_path, status="completed")
    finalized = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert finalized["status"] == "completed"
    assert finalized["completed_at_utc"]
    with pytest.raises(ValueError, match="已经冻结"):
        blind_freeze.finalize_blind_attempt(receipt_path, status="failed")
    assert not list(receipt_path.parent.glob(".*.tmp"))


def test_formal_cli_exposes_no_manifest_or_receipt_path_override():
    run_script = (
        EXPERIMENT_ROOT / "scripts" / "run_camera_domain_eval.py"
    ).read_text(encoding="utf-8")
    freeze_script = (
        EXPERIMENT_ROOT / "scripts" / "freeze_camera_domain_blind.py"
    ).read_text(encoding="utf-8")
    assert "--blind-attempt-receipt" not in run_script
    assert '"--output"' not in freeze_script


def test_freeze_manifest_binds_git_artifacts_environment_and_b_videos(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "blind-config.json"
    protocol_path = tmp_path / "protocol.md"
    runtime_path = tmp_path / "runtime.yaml"
    delay_path = tmp_path / "delay.json"
    selection_path = tmp_path / "selection.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    report_path = tmp_path / "second-round-report.json"
    state_root = tmp_path / "blind-state"
    monkeypatch.setattr(blind_freeze, "BLIND_FREEZE_STATE_ROOT", state_root)
    video_ids = [f"B{index}" for index in range(1, 8)]
    config = {
        "schema_version": "camera-domain-eval-config-v1",
        "protocol_stage": "blind_frozen",
        "required_conda_environment": "handai-intent-prediction",
        "input_mirrored": False,
        "video_sets": {"blind": video_ids},
        "blind_policy": {
            "enabled": True,
            "gate": {"frozen": True},
            "forbidden_video_sha256": ["f" * 64],
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    protocol_path.write_text("frozen protocol", encoding="utf-8")
    runtime_path.write_text("runtime", encoding="utf-8")
    delay_path.write_text("{}", encoding="utf-8")
    selection_path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"checkpoint")
    report_path.write_text(
        json.dumps(
            {
                "motion_strata_thresholds_from_validation": {"q90": 0.25},
                "acceptance": {"offline_gate_passed": False},
            }
        ),
        encoding="utf-8",
    )
    video_paths = {
        video_id: _write_video(tmp_path / f"{video_id}.mp4", video_id.encode())
        for video_id in video_ids
    }
    runtime_cfg = {
        "input_mirrored": False,
        "prediction_shadow_selection_path": str(selection_path),
        "prediction_shadow_checkpoint_path": str(checkpoint_path),
        "prediction_shadow_report_path": str(report_path),
    }
    git = {
        "available": True,
        "revision": "1" * 40,
        "tree": "2" * 40,
        "origin_main": "1" * 40,
        "branch": "main",
        "tracked_worktree_clean": True,
        "worktree_clean": True,
        "tracked_status_lines": [],
        "untracked_status_lines": [],
    }
    predictor = SimpleNamespace(
        initialization_error=None,
        model_label="residual_motion4",
        device="cuda",
        history_frames=30,
        horizon_ms=[50, 100, 150],
        selection_sha256=_sha256(selection_path),
        checkpoint_sha256=_sha256(checkpoint_path),
    )
    monkeypatch.setattr(blind_freeze, "git_snapshot", lambda: deepcopy(git))
    monkeypatch.setattr(
        blind_freeze,
        "prepare_effective_runtime_config",
        lambda _: deepcopy(runtime_cfg),
    )
    monkeypatch.setattr(
        blind_freeze,
        "build_prediction_shadow",
        lambda *args, **kwargs: predictor,
    )
    monkeypatch.setattr(
        blind_freeze,
        "assert_mapping_implementation_compatible",
        lambda _: "implementation-sha",
    )
    monkeypatch.setattr(
        blind_freeze,
        "mapping_contract_sha256",
        lambda _: "contract-sha",
    )
    current_environment = _locked_environment()
    monkeypatch.setattr(
        blind_freeze,
        "environment_identity",
        lambda: deepcopy(current_environment),
    )

    created = blind_freeze.create_blind_freeze_manifest(
        evaluation_config_path=config_path,
        protocol_path=protocol_path,
        runtime_config_path=runtime_path,
        delay_config_path=delay_path,
        blind_video_paths=video_paths,
        logger=SimpleNamespace(),
    )
    manifest_path = Path(created["path"])
    attempt_token = created["manifest"]["attempt_token"]
    assert attempt_token == created["manifest"]["attempt_identity_sha256"]
    assert manifest_path == blind_freeze.default_blind_freeze_manifest_path(
        attempt_token
    )
    assert created["attempt_receipt_path"] == str(
        blind_freeze.default_attempt_receipt_path(attempt_token)
    )
    assert created["manifest"]["environment"] == current_environment
    verified = blind_freeze.verify_blind_freeze_manifest(
        manifest_path=manifest_path,
        expected_manifest_sha256=created["sha256"],
        evaluation_config_path=config_path,
        protocol_path=protocol_path,
        runtime_config_path=runtime_path,
        delay_config_path=delay_path,
        evaluation_config=config,
        input_mirrored=False,
        blind_video_paths=video_paths,
    )
    assert verified["git"]["revision"] == "1" * 40
    assert set(verified["blind_inputs"]["videos"]) == set(video_ids)
    with pytest.raises(RuntimeError, match="身份已登记"):
        blind_freeze.create_blind_freeze_manifest(
            evaluation_config_path=config_path,
            protocol_path=protocol_path,
            runtime_config_path=runtime_path,
            delay_config_path=delay_path,
            blind_video_paths=video_paths,
            logger=SimpleNamespace(),
        )

    with pytest.raises(ValueError, match="路径不可覆盖"):
        blind_freeze.create_blind_freeze_manifest(
            evaluation_config_path=config_path,
            protocol_path=protocol_path,
            runtime_config_path=runtime_path,
            delay_config_path=delay_path,
            blind_video_paths=video_paths,
            output_path=tmp_path / "alternate-freeze.json",
            logger=SimpleNamespace(),
        )

    identity_variant = deepcopy(created["manifest"])
    identity_variant["created_at_utc"] = "2099-01-01T00:00:00Z"
    identity_variant["blind_inputs"]["sealed_at_utc"] = "2099-01-01T00:00:00Z"
    for record in identity_variant["artifacts"].values():
        record["path"] = f"renamed/{record['sha256']}"
        record["path_scope"] = "project_relative"
    assert blind_freeze.attempt_identity_sha256(identity_variant) == attempt_token

    current_environment["conda_explicit_spec_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="环境"):
        blind_freeze.verify_blind_freeze_manifest(
            manifest_path=manifest_path,
            expected_manifest_sha256=created["sha256"],
            evaluation_config_path=config_path,
            protocol_path=protocol_path,
            runtime_config_path=runtime_path,
            delay_config_path=delay_path,
            evaluation_config=config,
            input_mirrored=False,
            blind_video_paths=video_paths,
        )
    current_environment["conda_explicit_spec_sha256"] = "2" * 64

    video_paths["B1"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="B1 SHA-256 漂移"):
        blind_freeze.verify_blind_freeze_manifest(
            manifest_path=manifest_path,
            expected_manifest_sha256=created["sha256"],
            evaluation_config_path=config_path,
            protocol_path=protocol_path,
            runtime_config_path=runtime_path,
            delay_config_path=delay_path,
            evaluation_config=config,
            input_mirrored=False,
            blind_video_paths=video_paths,
        )

    video_paths["B1"].write_bytes(b"B1")
    tampered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered_manifest["attempt_token"] = "f" * 64
    manifest_path.write_text(
        json.dumps(tampered_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="deterministic|确定性"):
        blind_freeze.verify_blind_freeze_manifest(
            manifest_path=manifest_path,
            expected_manifest_sha256=_sha256(manifest_path),
            evaluation_config_path=config_path,
            protocol_path=protocol_path,
            runtime_config_path=runtime_path,
            delay_config_path=delay_path,
            evaluation_config=config,
            input_mirrored=False,
            blind_video_paths=video_paths,
        )


def test_freeze_creation_rejects_nonclean_worktree_before_model_load(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
                {
                    "protocol_stage": "blind_frozen",
                    "required_conda_environment": "handai-intent-prediction",
                    "input_mirrored": False,
                    "video_sets": {"blind": ["B1"]},
                "blind_policy": {
                    "enabled": True,
                    "gate": {"frozen": True},
                    "forbidden_video_sha256": ["f" * 64],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        blind_freeze,
        "git_snapshot",
        lambda: {"available": True, "worktree_clean": False},
    )

    def model_must_not_load(*args, **kwargs):
        raise AssertionError("dirty worktree must fail before model load")

    monkeypatch.setattr(
        blind_freeze,
        "build_prediction_shadow",
        model_must_not_load,
    )
    with pytest.raises(RuntimeError, match="工作树"):
        blind_freeze.create_blind_freeze_manifest(
            evaluation_config_path=config_path,
            protocol_path=tmp_path / "protocol.md",
            runtime_config_path=tmp_path / "runtime.yaml",
            delay_config_path=tmp_path / "delay.json",
            blind_video_paths={"B1": tmp_path / "B1.mp4"},
            output_path=tmp_path / "freeze.json",
            logger=SimpleNamespace(),
        )


@pytest.mark.parametrize(
    "environment_patch",
    [
        {"conda_environment_name": "single-right-hand-baseline"},
        {"conda_explicit_spec_sha256": None},
        {"pip_freeze_sha256": None},
    ],
)
def test_freeze_rejects_wrong_or_unlocked_conda_before_model_load(
    tmp_path, monkeypatch, environment_patch
):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "protocol_stage": "blind_frozen",
                "required_conda_environment": "handai-intent-prediction",
                "input_mirrored": False,
                "video_sets": {"blind": ["B1"]},
                "blind_policy": {
                    "enabled": True,
                    "gate": {"frozen": True},
                    "forbidden_video_sha256": ["f" * 64],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        blind_freeze,
        "git_snapshot",
        lambda: {
            "available": True,
            "revision": "1" * 40,
            "origin_main": "1" * 40,
            "worktree_clean": True,
        },
    )
    environment = _locked_environment()
    environment.update(environment_patch)
    monkeypatch.setattr(
        blind_freeze, "environment_identity", lambda: deepcopy(environment)
    )

    def model_must_not_load(*args, **kwargs):
        raise AssertionError("invalid environment must fail before model load")

    monkeypatch.setattr(
        blind_freeze, "build_prediction_shadow", model_must_not_load
    )
    with pytest.raises(RuntimeError, match="Conda 环境|环境未形成"):
        blind_freeze.create_blind_freeze_manifest(
            evaluation_config_path=config_path,
            protocol_path=tmp_path / "protocol.md",
            runtime_config_path=tmp_path / "runtime.yaml",
            delay_config_path=tmp_path / "delay.json",
            blind_video_paths={"B1": tmp_path / "B1.mp4"},
            logger=SimpleNamespace(),
        )


def test_stage_blind_video_is_content_addressed_read_only_and_detects_drift(
    tmp_path, monkeypatch
):
    state_root = tmp_path / "blind-state"
    monkeypatch.setattr(blind_freeze, "BLIND_FREEZE_STATE_ROOT", state_root)
    source = _write_video(tmp_path / "B1.mp4", b"frozen-video-bytes")
    token = "a" * 64
    freeze_state = {
        "attempt_token": token,
        "blind_inputs": {
            "videos": {
                "B1": {
                    "path": str(source),
                    "path_scope": "absolute",
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            }
        },
    }
    staged = blind_freeze.stage_blind_video_inputs({"B1": source}, freeze_state)[
        "B1"
    ]
    try:
        assert staged != source.resolve()
        assert staged.parent == (state_root / "sealed_inputs" / token).resolve()
        assert token in str(staged)
        assert _sha256(staged) == freeze_state["blind_inputs"]["videos"]["B1"][
            "sha256"
        ]
        assert staged.stat().st_mode & stat.S_IWRITE == 0
        source.write_bytes(b"mutated-after-staging")
        assert staged.read_bytes() == b"frozen-video-bytes"
        with pytest.raises(ValueError, match="B1 SHA-256 漂移"):
            blind_freeze.stage_blind_video_inputs({"B1": source}, freeze_state)
    finally:
        staged.chmod(stat.S_IREAD | stat.S_IWRITE)


def _patch_blind_run_preflight(tmp_path: Path, monkeypatch) -> tuple[dict, Path, Path]:
    monkeypatch.setattr(
        blind_freeze, "BLIND_FREEZE_STATE_ROOT", tmp_path / "blind-state"
    )
    config_path = tmp_path / "config.json"
    protocol_path = tmp_path / "protocol.md"
    runtime_path = tmp_path / "runtime.yaml"
    delay_path = tmp_path / "delay.json"
    manifest_path = tmp_path / "freeze.json"
    for path, content in (
        (config_path, "{}"),
        (protocol_path, "protocol"),
        (runtime_path, "runtime"),
        (delay_path, "{}"),
        (manifest_path, "{}"),
    ):
        path.write_text(content, encoding="utf-8")
    config = {
        "video_sets": {"blind": ["B1"]},
        "runtime_config_path": str(runtime_path),
        "protocol_document_path": str(protocol_path),
        "delay_config_path": str(delay_path),
        "timeline": {},
        "blind_policy": {"enabled": True, "gate": _decision_gate()},
        "claim_status": "camera_domain_pseudo_ground_truth_shadow_only",
    }
    token = "1" * 64
    freeze_state = {
        "path": str(manifest_path),
        "sha256": "a" * 64,
        "attempt_token": token,
        "attempt_identity_sha256": token,
        "git": {"revision": "b" * 40},
        "model_identity": {},
    }
    summary = {
        "video_id": "B1",
        "video_path": str(tmp_path / "B1.mp4"),
        "video_sha256": "c" * 64,
        "video_bytes": 1,
        "baseline_jsonl_path": str(tmp_path / "B1.jsonl"),
        "baseline_jsonl_sha256": "d" * 64,
    }
    monkeypatch.setattr(camera_eval, "load_evaluation_config", lambda _: config)
    monkeypatch.setattr(
        camera_eval,
        "verify_protocol_state",
        lambda **kwargs: {"role": "blind", "protocol_stage": "blind_frozen"},
    )
    monkeypatch.setattr(
        camera_eval,
        "prepare_effective_runtime_config",
        lambda _: {"input_mirrored": False},
    )
    monkeypatch.setattr(camera_eval, "_read_json", lambda _: {})
    monkeypatch.setattr(
        camera_eval,
        "verify_blind_freeze_manifest",
        lambda **kwargs: deepcopy(freeze_state),
    )
    monkeypatch.setattr(
        camera_eval,
        "verify_blind_video_inventory",
        lambda *args, **kwargs: {"B1": summary["video_sha256"]},
    )
    monkeypatch.setattr(
        camera_eval,
        "stage_blind_video_inputs",
        lambda paths, state: {
            video_id: path.resolve() for video_id, path in paths.items()
        },
    )
    monkeypatch.setattr(
        camera_eval,
        "verify_processed_video_identities",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        blind_freeze,
        "verify_processed_video_identities",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        camera_eval,
        "process_video_to_baseline_jsonl",
        lambda *args, **kwargs: deepcopy(summary),
    )
    monkeypatch.setattr(camera_eval, "_write_markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(camera_eval, "_write_video_csv", lambda *args, **kwargs: None)
    return config, config_path, manifest_path


def test_blind_input_failure_skips_algorithm_and_does_not_consume_receipt(
    tmp_path, monkeypatch
):
    _, config_path, manifest_path = _patch_blind_run_preflight(tmp_path, monkeypatch)
    monkeypatch.setattr(
        camera_eval,
        "evaluate_blind_input_gate",
        lambda *args, **kwargs: {
            "passed": False,
            "criteria": {"B1_task": False},
            "details": {"B1": {"reason": "input"}},
        },
    )

    def algorithm_must_not_run(*args, **kwargs):
        raise AssertionError("input gate failure must skip algorithm metrics")

    monkeypatch.setattr(
        camera_eval,
        "evaluate_camera_algorithm_utility",
        algorithm_must_not_run,
    )
    receipt_path = blind_freeze.default_attempt_receipt_path("1" * 64)
    video_path = _write_video(tmp_path / "B1.mp4", b"B1")
    report_path = camera_eval.run_camera_domain_evaluation(
        video_specs=[camera_eval.VideoSpec("B1", video_path)],
        evaluation_config_path=config_path,
        output_root=tmp_path / "outputs",
        role="blind",
        expected_config_sha256="e" * 64,
        expected_protocol_sha256="f" * 64,
        blind_freeze_manifest_path=manifest_path,
        expected_blind_freeze_manifest_sha256="a" * 64,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["algorithm_utility"]["status"] == "not_evaluated_input_gate_failed"
    assert report["decision"]["branch"] == "input_or_clock_repair"
    assert not receipt_path.exists()


def test_blind_runner_decodes_sealed_copy_not_original(tmp_path, monkeypatch):
    _, config_path, manifest_path = _patch_blind_run_preflight(tmp_path, monkeypatch)
    observed: dict[str, object] = {}
    sealed_path = tmp_path / "sealed" / "B1-frozen.mp4"

    def stage_then_mutate(paths, freeze_state):
        del freeze_state
        source = paths["B1"]
        sealed_path.parent.mkdir(parents=True)
        sealed_path.write_bytes(source.read_bytes())
        source.write_bytes(b"mutated-original")
        return {"B1": sealed_path}

    def process_sealed(spec, **kwargs):
        observed["path"] = spec.path
        observed["bytes"] = spec.path.read_bytes()
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}\n", encoding="utf-8")
        return {
            "video_id": "B1",
            "video_path": str(spec.path),
            "video_sha256": _sha256(spec.path),
            "video_bytes": spec.path.stat().st_size,
            "baseline_jsonl_path": str(output_path),
            "baseline_jsonl_sha256": _sha256(output_path),
        }

    monkeypatch.setattr(camera_eval, "stage_blind_video_inputs", stage_then_mutate)
    monkeypatch.setattr(
        camera_eval, "process_video_to_baseline_jsonl", process_sealed
    )
    monkeypatch.setattr(
        camera_eval,
        "evaluate_blind_input_gate",
        lambda *args, **kwargs: {
            "passed": False,
            "criteria": {"B1_task": False},
            "details": {"B1": {}},
        },
    )
    video_path = _write_video(tmp_path / "B1.mp4", b"frozen-original")
    camera_eval.run_camera_domain_evaluation(
        video_specs=[camera_eval.VideoSpec("B1", video_path)],
        evaluation_config_path=config_path,
        output_root=tmp_path / "outputs",
        role="blind",
        expected_config_sha256="e" * 64,
        expected_protocol_sha256="f" * 64,
        blind_freeze_manifest_path=manifest_path,
        expected_blind_freeze_manifest_sha256="a" * 64,
    )
    assert observed == {"path": sealed_path, "bytes": b"frozen-original"}


@pytest.mark.parametrize(
    ("raised", "expected_type"),
    [
        (RuntimeError("model exploded"), RuntimeError),
        (KeyboardInterrupt(), KeyboardInterrupt),
        (SystemExit(42), SystemExit),
    ],
)
def test_blind_algorithm_baseexception_consumes_attempt_as_failed_receipt(
    tmp_path, monkeypatch, raised, expected_type
):
    _, config_path, manifest_path = _patch_blind_run_preflight(tmp_path, monkeypatch)
    monkeypatch.setattr(
        camera_eval,
        "evaluate_blind_input_gate",
        lambda *args, **kwargs: {
            "passed": True,
            "criteria": {"B1_task": True},
            "details": {"B1": {}},
        },
    )

    def algorithm_fails(*args, **kwargs):
        raise raised

    monkeypatch.setattr(
        camera_eval,
        "evaluate_camera_algorithm_utility",
        algorithm_fails,
    )
    receipt_path = blind_freeze.default_attempt_receipt_path("1" * 64)
    video_path = _write_video(tmp_path / "B1.mp4", b"B1")
    with pytest.raises(expected_type):
        camera_eval.run_camera_domain_evaluation(
            video_specs=[camera_eval.VideoSpec("B1", video_path)],
            evaluation_config_path=config_path,
            output_root=tmp_path / "outputs",
            role="blind",
            expected_config_sha256="e" * 64,
            expected_protocol_sha256="f" * 64,
            blind_freeze_manifest_path=manifest_path,
            expected_blind_freeze_manifest_sha256="a" * 64,
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert expected_type.__name__ in receipt["error"]


def test_blind_report_phase_baseexception_finalizes_failed_receipt(
    tmp_path, monkeypatch
):
    _, config_path, manifest_path = _patch_blind_run_preflight(tmp_path, monkeypatch)
    monkeypatch.setattr(
        camera_eval,
        "evaluate_blind_input_gate",
        lambda *args, **kwargs: {
            "passed": True,
            "criteria": {"B1_task": True},
            "details": {"B1": {}},
        },
    )
    monkeypatch.setattr(
        camera_eval,
        "evaluate_camera_algorithm_utility",
        lambda *args, **kwargs: ({"status": "evaluated"}, [], []),
    )
    monkeypatch.setattr(
        camera_eval, "verify_algorithm_identity", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        camera_eval,
        "evaluate_blind_decision",
        lambda *args, **kwargs: {
            "status": "blind_gate_passed",
            "branch": "keep_v2_shadow",
            "reason": "test",
        },
    )

    def interrupt_report(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(camera_eval, "_write_markdown", interrupt_report)
    video_path = _write_video(tmp_path / "B1.mp4", b"B1")
    receipt_path = blind_freeze.default_attempt_receipt_path("1" * 64)
    with pytest.raises(KeyboardInterrupt):
        camera_eval.run_camera_domain_evaluation(
            video_specs=[camera_eval.VideoSpec("B1", video_path)],
            evaluation_config_path=config_path,
            output_root=tmp_path / "outputs",
            role="blind",
            expected_config_sha256="e" * 64,
            expected_protocol_sha256="f" * 64,
            blind_freeze_manifest_path=manifest_path,
            expected_blind_freeze_manifest_sha256="a" * 64,
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert "KeyboardInterrupt" in receipt["error"]


def test_receipt_finalize_failure_does_not_mask_original_baseexception(
    tmp_path, monkeypatch
):
    _, config_path, manifest_path = _patch_blind_run_preflight(tmp_path, monkeypatch)
    monkeypatch.setattr(
        camera_eval,
        "evaluate_blind_input_gate",
        lambda *args, **kwargs: {
            "passed": True,
            "criteria": {"B1_task": True},
            "details": {"B1": {}},
        },
    )

    def interrupt_algorithm(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        camera_eval, "evaluate_camera_algorithm_utility", interrupt_algorithm
    )

    def fail_finalize(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(camera_eval, "finalize_blind_attempt", fail_finalize)
    video_path = _write_video(tmp_path / "B1.mp4", b"B1")
    receipt_path = blind_freeze.default_attempt_receipt_path("1" * 64)
    with pytest.raises(KeyboardInterrupt):
        camera_eval.run_camera_domain_evaluation(
            video_specs=[camera_eval.VideoSpec("B1", video_path)],
            evaluation_config_path=config_path,
            output_root=tmp_path / "outputs",
            role="blind",
            expected_config_sha256="e" * 64,
            expected_protocol_sha256="f" * 64,
            blind_freeze_manifest_path=manifest_path,
            expected_blind_freeze_manifest_sha256="a" * 64,
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "reserved"


@pytest.mark.parametrize(
    ("manifest_value", "expected_sha"),
    [(None, "a" * 64), ("manifest.json", None)],
)
def test_blind_missing_manifest_or_sha_fails_before_output_creation(
    tmp_path, monkeypatch, manifest_value, expected_sha
):
    """正式 B1-B7 在解码和 _unique_run_dir 前必须拿到 freeze manifest 与显式 SHA。"""

    config_path = tmp_path / "blind-config.json"
    protocol_path = tmp_path / "protocol.md"
    runtime_path = tmp_path / "runtime.yaml"
    delay_path = tmp_path / "delay.json"
    config_path.write_text("{}", encoding="utf-8")
    protocol_path.write_text("frozen protocol", encoding="utf-8")
    runtime_path.write_text("{}", encoding="utf-8")
    delay_path.write_text("{}", encoding="utf-8")
    videos = [
        camera_eval.VideoSpec(f"B{index}", _write_video(tmp_path / f"B{index}.mp4", bytes([index])))
        for index in range(1, 8)
    ]
    config = {
        "protocol_stage": "blind_frozen",
        "blind_policy": {"enabled": True, "gate": {}},
        "video_sets": {"blind": [f"B{index}" for index in range(1, 8)]},
        "timeline": {
            "synthetic_epoch_seconds": 1_700_000_000.0,
            "minimum_nominal_fps": 1.0,
            "maximum_nominal_fps": 240.0,
        },
        "runtime_config_path": str(runtime_path),
        "protocol_document_path": str(protocol_path),
        "delay_config_path": str(delay_path),
    }
    monkeypatch.setattr(camera_eval, "load_evaluation_config", lambda _: config)

    def decoder_must_not_run(*args, **kwargs):
        raise AssertionError("missing manifest/SHA must fail before video decoding")

    monkeypatch.setattr(camera_eval, "process_video_to_baseline_jsonl", decoder_must_not_run)
    output_root = tmp_path / "output"
    manifest_path = None if manifest_value is None else tmp_path / manifest_value
    with pytest.raises((ValueError, FileNotFoundError), match="manifest|冻结"):
        camera_eval.run_camera_domain_evaluation(
            video_specs=videos,
            evaluation_config_path=config_path,
            output_root=output_root,
            role="blind",
            expected_config_sha256=_sha256(config_path),
            expected_protocol_sha256=_sha256(protocol_path),
            blind_freeze_manifest_path=manifest_path,
            expected_blind_freeze_manifest_sha256=expected_sha,
        )
    assert not output_root.exists()
