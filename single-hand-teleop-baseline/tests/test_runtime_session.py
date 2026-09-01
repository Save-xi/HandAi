from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from output.json_exporter import JsonExporter
from utils.runtime_session import (
    RUNTIME_SESSION_SCHEMA_VERSION,
    RuntimeSessionRecorder,
    create_runtime_session_artifacts,
)

from test_output_exporter import _sample_payload


def test_runtime_session_manifest_is_atomically_frozen_with_log_identity(tmp_path):
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("camera_index: 0\n", encoding="utf-8")
    cfg = {
        "jsonl_output_dir": str(tmp_path),
        "camera_index": 0,
        "save_jsonl": True,
        "prediction_shadow_enabled": False,
        "unity_udp_enabled": True,
        "unity_udp_host": "127.0.0.1",
        "unity_udp_port": 18080,
    }
    artifacts = create_runtime_session_artifacts(
        cfg,
        prediction_requested=False,
        run_id="20260901T120000000000Z-1234-deadbeef",
    )
    runtime = SimpleNamespace(
        input_source_type="webcam",
        video_file_path=None,
        input_mirrored=False,
        control_extension_enabled=True,
        svh_preview_enabled=True,
    )
    recorder = RuntimeSessionRecorder(
        artifacts,
        config_path=config_path,
        cfg=cfg,
        runtime=runtime,
    )
    exporter = JsonExporter(
        str(tmp_path / "latest.json"),
        save_last_json=False,
        jsonl_path=str(artifacts.baseline_jsonl_path),
    )
    for frame_index in range(2):
        payload = _sample_payload(frame_index)
        recorder.observe_baseline(payload)
        exporter.export_prepared_frame(payload, frame_index=frame_index)
    exporter.close()
    recorder.finalize(
        status="completed",
        error=None,
        baseline_exporter=exporter,
        prediction_exporter=None,
        prediction_worker=None,
        prediction_worker_stopped=None,
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    baseline_bytes = artifacts.baseline_jsonl_path.read_bytes()
    assert manifest["schema_version"] == RUNTIME_SESSION_SCHEMA_VERSION
    assert manifest["status"] == "completed"
    assert manifest["frames"] == {
        "processed": 2,
        "first_frame_index": 0,
        "last_frame_index": 1,
        "first_timestamp_unix_s": 1000.0,
        "last_timestamp_unix_s": 1001.0,
    }
    assert manifest["outputs"]["baseline_jsonl"]["rows"] == 2
    assert manifest["outputs"]["baseline_jsonl"]["sha256"] == hashlib.sha256(
        baseline_bytes
    ).hexdigest()
    assert manifest["outputs"]["prediction_jsonl"] is None
    assert list(tmp_path.glob("*.tmp")) == []
