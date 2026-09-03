from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "intent_prediction"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from intent_prediction import camera_domain_eval as camera_eval  # noqa: E402


def test_media_timestamp_prefers_pts_and_falls_back_monotonically():
    first = camera_eval.resolve_media_timestamp_ms(
        0,
        raw_pts_ms=0.0,
        nominal_fps=30.0,
        previous_timestamp_ms=None,
    )
    assert first.timestamp_ms == 0.0
    assert first.source == "container_pts_ms"

    repeated = camera_eval.resolve_media_timestamp_ms(
        1,
        raw_pts_ms=0.0,
        nominal_fps=30.0,
        previous_timestamp_ms=first.timestamp_ms,
    )
    assert repeated.timestamp_ms == pytest.approx(1000.0 / 30.0)
    assert repeated.source == "frame_index_over_nominal_fps"

    valid = camera_eval.resolve_media_timestamp_ms(
        2,
        raw_pts_ms=66.666,
        nominal_fps=30.0,
        previous_timestamp_ms=repeated.timestamp_ms,
    )
    assert valid.timestamp_ms == pytest.approx(66.666)
    assert valid.source == "container_pts_ms"


def test_media_timestamp_continuity_fallback_handles_shifted_first_pts():
    first = camera_eval.resolve_media_timestamp_ms(
        0,
        raw_pts_ms=1000.0,
        nominal_fps=25.0,
        previous_timestamp_ms=None,
    )
    second = camera_eval.resolve_media_timestamp_ms(
        1,
        raw_pts_ms=float("nan"),
        nominal_fps=25.0,
        previous_timestamp_ms=first.timestamp_ms,
    )
    assert second.timestamp_ms == pytest.approx(1040.0)
    assert second.source == "continuity_fallback_period"


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan")])
def test_media_timestamp_rejects_invalid_nominal_fps(fps):
    with pytest.raises(ValueError, match="nominal_fps"):
        camera_eval.resolve_media_timestamp_ms(
            0,
            raw_pts_ms=0.0,
            nominal_fps=fps,
            previous_timestamp_ms=None,
        )


def test_parse_video_specs_requires_unique_existing_ids(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    specs = camera_eval.parse_video_specs([f"V1={first}", f"V2={second}"])
    assert [spec.video_id for spec in specs] == ["V1", "V2"]
    with pytest.raises(ValueError, match="重复"):
        camera_eval.parse_video_specs([f"V1={first}", f"V1={second}"])
    with pytest.raises(ValueError, match="同一视频"):
        camera_eval.parse_video_specs([f"V1={first}", f"V2={first}"])


def test_development_protocol_can_be_recorded_but_blind_is_rejected(tmp_path):
    config_path = tmp_path / "config.json"
    protocol_path = tmp_path / "protocol.md"
    config_path.write_text("{}", encoding="utf-8")
    protocol_path.write_text("draft", encoding="utf-8")
    config = {
        "protocol_stage": "development",
        "blind_policy": {"enabled": False, "gate": None},
    }
    state = camera_eval.verify_protocol_state(
        role="development",
        evaluation_config=config,
        evaluation_config_path=config_path,
        protocol_path=protocol_path,
        expected_config_sha256=None,
        expected_protocol_sha256=None,
    )
    assert state["frozen_for_this_run"] is False
    assert state["protocol_sha256"] == hashlib.sha256(b"draft").hexdigest()
    with pytest.raises(ValueError, match="development"):
        camera_eval.verify_protocol_state(
            role="blind",
            evaluation_config=config,
            evaluation_config_path=config_path,
            protocol_path=protocol_path,
            expected_config_sha256=state["evaluation_config_sha256"],
            expected_protocol_sha256=state["protocol_sha256"],
        )


class _FakeCapture:
    def __init__(self) -> None:
        self.frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(3)]
        self.pts = [0.0, 0.0, 66.666]
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def get(self, prop):
        if prop == camera_eval.cv2.CAP_PROP_FPS:
            return 30.0
        if prop == camera_eval.cv2.CAP_PROP_FRAME_COUNT:
            return 3.0
        if prop == camera_eval.cv2.CAP_PROP_FRAME_WIDTH:
            return 3.0
        if prop == camera_eval.cv2.CAP_PROP_FRAME_HEIGHT:
            return 2.0
        if prop == camera_eval.cv2.CAP_PROP_POS_MSEC:
            return self.pts[max(0, self.index - 1)]
        return 0.0

    def release(self) -> None:
        self.released = True


def test_video_processing_uses_media_timeline_not_wall_clock(tmp_path, monkeypatch):
    capture = _FakeCapture()
    observed_timestamps: list[float] = []
    closed: list[bool] = []

    def fake_builder(cfg, *, input_mirrored, logger):
        del cfg, input_mirrored, logger

        def process(frame, frame_index, timestamp_s, source_fps):
            del frame, source_fps
            observed_timestamps.append(timestamp_s)
            return (
                {
                    "timestamp": timestamp_s,
                    "frame_index": frame_index,
                    "detected": True,
                    "control_ready": True,
                    "svh_preview": {"valid": True},
                },
                1.0,
            )

        return process, lambda: closed.append(True)

    monkeypatch.setattr(camera_eval, "_build_video_payload_processor", fake_builder)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    output = tmp_path / "baseline.jsonl"
    summary = camera_eval.process_video_to_baseline_jsonl(
        camera_eval.VideoSpec("V1", video),
        output_path=output,
        cfg={},
        timeline_config={
            "synthetic_epoch_seconds": 1700000000.0,
            "minimum_nominal_fps": 1.0,
            "maximum_nominal_fps": 240.0,
        },
        input_mirrored=False,
        logger=camera_eval.logging.getLogger("test"),
        capture_factory=lambda _: capture,
    )
    assert capture.released is True
    assert closed == [True]
    assert observed_timestamps == pytest.approx(
        [1700000000.0, 1700000000.0 + 1.0 / 30.0, 1700000000.0 + 0.066666]
    )
    assert summary["source_timeline"]["effective_fps"] == pytest.approx(30.0, rel=2e-4)
    assert summary["source_timeline"]["median_interval_fps"] == pytest.approx(30.0, rel=2e-4)
    assert summary["source_timeline"]["duration_based_fps"] == pytest.approx(30.0003, rel=2e-4)
    assert summary["source_timeline"]["timestamp_source_counts"] == {
        "container_pts_ms": 2,
        "frame_index_over_nominal_fps": 1,
    }
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["frame_index"] for row in rows] == [0, 1, 2]


def _write_live_runtime_fixture(tmp_path):
    run_id = "20260901T120000000000Z-1234-deadbeef"
    baseline_path = tmp_path / f"session_{run_id}.jsonl"
    prediction_path = tmp_path / f"prediction_session_{run_id}.jsonl"
    manifest_path = tmp_path / f"runtime_session_{run_id}.json"
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("prediction_shadow_enabled: true\n", encoding="utf-8")
    identity = {
        "model_label": "test-shadow",
        "device": "cpu",
        "history_frames": 30,
        "horizon_ms": [50, 100, 150],
        "selection_sha256": "selection-sha",
        "checkpoint_sha256": "checkpoint-sha",
    }
    baseline_rows = [
        {
            "frame_index": index,
            "timestamp": 1000.0 + index * 0.04,
            "latency_ms": 10.0 + index,
            "control_ready": index > 0,
            "svh_preview": {"valid": index > 0},
            "timing": {
                "schema_version": 1,
                "source_read_start_unix_ms": 2000.0 + index * 40.0,
                "source_read_end_unix_ms": 2001.0 + index * 40.0,
                "udp_send_attempt_unix_ms": 2011.0 + index * 40.0,
            },
        }
        for index in range(4)
    ]
    prediction_rows = [
        {
            "frame_index": source_index,
            "timestamp": baseline_rows[source_index]["timestamp"],
            "prediction_diagnostics": {
                "source_frame_index": source_index,
                "source_timestamp_unix_ms": baseline_rows[source_index]["timestamp"] * 1000.0,
                "status": status,
                "observed_fps": 25.0,
                "inference_ms": 2.0 if status == "predicted" else None,
                "model_label": identity["model_label"],
                "device": identity["device"],
                "history_frames_required": identity["history_frames"],
                "horizon_ms": identity["horizon_ms"],
                "selection_sha256": identity["selection_sha256"],
                "checkpoint_sha256": identity["checkpoint_sha256"],
            },
        }
        for source_index, status in ((1, "warming_up"), (2, "predicted"))
    ]
    baseline_path.write_text(
        "\n".join(json.dumps(row) for row in baseline_rows) + "\n",
        encoding="utf-8",
    )
    prediction_path.write_text(
        "\n".join(json.dumps(row) for row in prediction_rows) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "handai-runtime-session-v1",
        "run_id": run_id,
        "status": "completed",
        "config": {
            "path": str(config_path.resolve()),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
        "runtime": {"prediction_shadow_requested": True},
        "outputs": {
            "baseline_jsonl": {
                "path": str(baseline_path.resolve()),
                "exists": True,
                "bytes": baseline_path.stat().st_size,
                "sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                "rows": len(baseline_rows),
            },
            "prediction_jsonl": {
                "path": str(prediction_path.resolve()),
                "exists": True,
                "bytes": prediction_path.stat().st_size,
                "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                "rows": len(prediction_rows),
            },
        },
        "frames": {
            "processed": len(baseline_rows),
            "first_frame_index": 0,
            "last_frame_index": 3,
            "first_timestamp_unix_s": baseline_rows[0]["timestamp"],
            "last_timestamp_unix_s": baseline_rows[-1]["timestamp"],
        },
        "prediction": {
            "enabled": True,
            **identity,
            "initialization_error": None,
            "worker": {
                "submitted": 4,
                "completed": 2,
                "dropped_input": 2,
                "dropped_result": 0,
                "stopped": True,
            },
        },
        "safety": {"prediction_modifies_unity_udp": False},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return baseline_path, prediction_path, manifest_path, baseline_rows


def _timing_metric(sample_count: int) -> dict:
    return {
        "total_sample_count": sample_count,
        "retained_sample_count": sample_count,
        "capacity": 4096,
        "p50_ms": 5.0,
        "p95_ms": 8.0,
        "max_ms": 10.0,
    }


def test_live_runtime_analysis_is_separate_from_algorithm_replay(tmp_path):
    baseline_path, prediction_path, _, baseline_rows = _write_live_runtime_fixture(tmp_path)
    unity_path = tmp_path / "unity_timing_test.json"
    unity_path.write_text(
        json.dumps(
            {
                "schema_version": "handai-unity-timing-summary-v1",
                "accepted_packet_count": 2,
                "source_session": {
                    "first_frame_index": 1,
                    "last_frame_index": 2,
                    "first_source_timestamp_unix_ms": baseline_rows[1]["timestamp"] * 1000.0,
                    "last_source_timestamp_unix_ms": baseline_rows[2]["timestamp"] * 1000.0,
                },
                "metrics": {
                    "python_post_capture_to_udp_ms": _timing_metric(2),
                    "udp_delivery_ms": _timing_metric(2),
                    "unity_main_thread_queue_ms": _timing_metric(2),
                    "source_read_end_to_target_apply_ms": _timing_metric(2),
                },
                "counters": {
                    "overwritten_packet_count": 1,
                    "frame_gap_count": 1,
                    "rejected_packet_count": 0,
                    "stale_packet_count": 0,
                    "watchdog_open_count": 1,
                },
                "safety": {
                    "baseline_udp_hardware_forwarding_compiled": False,
                    "apply_baseline_preview_to_hardware": False,
                    "real_svh_in_scope": False,
                },
            }
        ),
        encoding="utf-8",
    )
    report = camera_eval.analyze_live_runtime_logs(
        baseline_path,
        prediction_path,
        unity_timing_path=unity_path,
    )
    assert report["worker_result_coverage_all_frames"] == 0.5
    assert report["predicted_coverage_all_frames"] == 0.25
    assert report["predicted_coverage_valid_frames"] == pytest.approx(1.0 / 3.0)
    assert report["source_duration_based_fps"] == pytest.approx(25.0)
    assert report["source_median_interval_fps"] == pytest.approx(25.0)
    assert report["python_post_capture_to_udp_ms"]["p50"] == 10.0
    assert report["unity_timing"]["summary"]["accepted_packet_count"] == 2

    markdown_path = tmp_path / "live_runtime.md"
    camera_eval._write_markdown(
        markdown_path,
        {
            "protocol": {"role": "development", "protocol_stage": "development"},
            "git": {},
            "decision": {"status": "development_only_no_release_decision"},
            "videos": [],
            "algorithm_utility": {"status": "not_evaluated", "source_errors": []},
            "live_runtime": report,
        },
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "accepted=2" in markdown
    assert "overwritten=1" in markdown
    assert "frame_gap=1" in markdown
    assert "watchdog_open=1" in markdown
    assert "real_svh_in_scope=False" in markdown


def test_live_runtime_analysis_rejects_incomplete_unity_safety_evidence(tmp_path):
    baseline_path, prediction_path, _, baseline_rows = _write_live_runtime_fixture(tmp_path)
    unity_path = tmp_path / "unity_timing_invalid.json"
    summary = {
        "schema_version": "handai-unity-timing-summary-v1",
        "accepted_packet_count": 2,
        "source_session": {
            "first_frame_index": 1,
            "last_frame_index": 2,
            "first_source_timestamp_unix_ms": baseline_rows[1]["timestamp"] * 1000.0,
            "last_source_timestamp_unix_ms": baseline_rows[2]["timestamp"] * 1000.0,
        },
        "metrics": {
            "python_post_capture_to_udp_ms": _timing_metric(2),
            "udp_delivery_ms": _timing_metric(2),
            "unity_main_thread_queue_ms": _timing_metric(2),
            "source_read_end_to_target_apply_ms": _timing_metric(2),
        },
        "counters": {
            "overwritten_packet_count": 0,
            "frame_gap_count": 0,
            "rejected_packet_count": 0,
            "stale_packet_count": 0,
            "watchdog_open_count": 0,
        },
        "safety": {
            "baseline_udp_hardware_forwarding_compiled": False,
            "apply_baseline_preview_to_hardware": True,
            "real_svh_in_scope": False,
        },
    }
    unity_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="apply_baseline_preview_to_hardware"):
        camera_eval.analyze_live_runtime_logs(
            baseline_path,
            prediction_path,
            unity_timing_path=unity_path,
        )


def test_live_runtime_analysis_rejects_duplicate_baseline_frame_index(tmp_path):
    baseline_path, prediction_path, _, _ = _write_live_runtime_fixture(tmp_path)
    first_line = baseline_path.read_text(encoding="utf-8").splitlines()[0]
    with baseline_path.open("a", encoding="utf-8") as handle:
        handle.write(first_line + "\n")
    with pytest.raises(ValueError, match="重复 frame_index"):
        camera_eval.analyze_live_runtime_logs(baseline_path, prediction_path)


def test_live_runtime_analysis_rejects_cross_session_pair_before_loose_intersection(tmp_path):
    baseline_path = tmp_path / "session_run-a.jsonl"
    prediction_path = tmp_path / "prediction_session_run-b.jsonl"
    baseline_path.write_text("{}\n", encoding="utf-8")
    prediction_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="run_id 不一致"):
        camera_eval.analyze_live_runtime_logs(baseline_path, prediction_path)


def test_live_runtime_analysis_rejects_prediction_identity_drift(tmp_path):
    baseline_path, prediction_path, manifest_path, _ = _write_live_runtime_fixture(tmp_path)
    prediction_rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    prediction_rows[0]["prediction_diagnostics"]["selection_sha256"] = "wrong-selection"
    prediction_path.write_text(
        "\n".join(json.dumps(row) for row in prediction_rows) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["prediction_jsonl"].update(
        {
            "bytes": prediction_path.stat().st_size,
            "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="selection_sha256"):
        camera_eval.analyze_live_runtime_logs(baseline_path, prediction_path)


def test_camera_domain_module_does_not_create_udp_exporter():
    source = Path(camera_eval.__file__).read_text(encoding="utf-8")
    assert "JsonExporter" not in source
    assert '"udp_created": False' in source


def _blind_video(ready_fraction: float = 0.9):
    return {
        "video_id": "B1",
        "metadata": {
            "decoded_frame_count": 100,
            "metadata_frame_count": 100,
            "nominal_fps": 30.0,
            "width": 1280,
            "height": 720,
        },
        "source_timeline": {
            "timestamp_source_counts": {"container_pts_ms": 100},
            "duration_ms": 3_300.0,
            "duration_based_fps": 30.0,
        },
        "observation_counts": {
            "control_ready_fraction": ready_fraction,
            "svh_valid_fraction": ready_fraction,
            "stable_gesture_counts": {"open": 2},
        },
        "observation_continuity": {"longest_invalid_run_ms": 20.0},
    }


def _blind_algorithm(gated_improvement: float = 4.0):
    return {
        "status": "evaluated",
        "source_errors": [],
        "aggregate": {
            "primary_delays_ms": [50.0, 100.0],
            "primary_delay_summary": {
                "improvement_percent_vs_hold": {
                    "gated_rmse": gated_improvement,
                    "gated_p95": 4.0,
                },
                "dynamic_q90": {"improvement_percent_vs_hold": {"gated_rmse": 6.0}},
                "conditional_prediction_available_fraction": 0.95,
                "end_to_end_prediction_coverage_fraction": 0.9,
                "methods": {"gated": {"range_violation_rate": 0.0}},
            }
        },
        "per_video": {
            "B1": {
                "source": {"evaluable_rows": 20},
                "primary_delay_summary": {
                    "improvement_percent_vs_hold": {
                        "gated_rmse": gated_improvement,
                        "gated_p95": 4.0,
                    }
                },
            }
        },
    }


def _blind_gate():
    b1_rule = {
        "profile": "clean",
        "minimum_duration_ms": 3_000.0,
        "maximum_duration_ms": 4_000.0,
        "minimum_nominal_fps": 20.0,
        "maximum_nominal_fps": 60.0,
        "minimum_duration_based_fps": 20.0,
        "maximum_duration_based_fps": 60.0,
        "minimum_width": 640,
        "minimum_height": 480,
        "maximum_timestamp_fallback_fraction": 0.1,
        "minimum_control_ready_fraction": 0.8,
        "minimum_svh_valid_fraction": 0.8,
        "require_metadata_frame_count_match": True,
        "maximum_invalid_run_duration_ms": 100.0,
        "minimum_stable_gesture_frames": {"open": 1},
    }
    return {
        "primary_delays_ms": [50.0, 100.0],
        "minimum_primary_gated_rmse_improvement_percent": 3.0,
        "minimum_primary_dynamic_gated_rmse_improvement_percent": 5.0,
        "minimum_primary_gated_p95_improvement_percent": 3.0,
        "minimum_each_video_gated_rmse_improvement_percent": 3.0,
        "minimum_each_video_gated_p95_improvement_percent": 3.0,
        "minimum_each_video_evaluable_rows": 10,
        "minimum_primary_conditional_prediction_available_fraction": 0.9,
        "minimum_primary_end_to_end_prediction_coverage_fraction": 0.85,
        "maximum_primary_range_violation_rate": 0.0,
        "require_live_runtime": False,
        "video_requirements": {"B1": b1_rule},
    }


def test_blind_decision_uses_three_distinct_branches():
    keep = camera_eval.evaluate_blind_decision(
        [_blind_video()],
        _blind_algorithm(),
        None,
        gate=_blind_gate(),
    )
    assert keep["status"] == "blind_gate_passed"
    assert keep["branch"] == "keep_v2_shadow"

    v3 = camera_eval.evaluate_blind_decision(
        [_blind_video()],
        _blind_algorithm(gated_improvement=0.0),
        None,
        gate=_blind_gate(),
    )
    assert v3["status"] == "blind_gate_failed"
    assert v3["branch"] == "v3_pre_registered_candidate"

    input_repair = camera_eval.evaluate_blind_decision(
        [_blind_video(ready_fraction=0.5)],
        _blind_algorithm(gated_improvement=0.0),
        None,
        gate=_blind_gate(),
    )
    assert input_repair["status"] == "blind_gate_failed"
    assert input_repair["branch"] == "input_or_clock_repair"


def test_checked_in_camera_domain_config_is_valid():
    config = camera_eval.load_evaluation_config(camera_eval.DEFAULT_CONFIG_PATH)
    assert config["video_sets"]["development"] == ["V1", "V2", "V3", "V4", "V5", "V6", "V7"]
    assert config["video_sets"]["blind"] == ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
    assert config["blind_policy"]["enabled"] is False

    blind = camera_eval.load_evaluation_config(camera_eval.DEFAULT_BLIND_CONFIG_PATH)
    assert blind["protocol_stage"] == "blind_frozen"
    assert blind["required_conda_environment"] == "handai-intent-prediction"
    assert blind["input_mirrored"] is False
    assert blind["blind_policy"]["enabled"] is True
    assert set(blind["blind_policy"]["gate"]["video_requirements"]) == {
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
    }
    assert blind["blind_policy"]["gate"]["video_requirements"]["B6"][
        "required_invalid_episode_count"
    ] == 3


def test_checked_in_development_config_refuses_blind_run_before_video_decode(tmp_path):
    specs = [camera_eval.VideoSpec(f"B{index}", tmp_path / f"B{index}.mp4") for index in range(1, 8)]
    manifest_path = tmp_path / "placeholder-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="development"):
        camera_eval.run_camera_domain_evaluation(
            video_specs=specs,
            evaluation_config_path=camera_eval.DEFAULT_CONFIG_PATH,
            output_root=tmp_path / "outputs",
            role="blind",
            blind_freeze_manifest_path=manifest_path,
            expected_blind_freeze_manifest_sha256=hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
        )
    assert not (tmp_path / "outputs").exists()
