from __future__ import annotations

"""实时摄像头运行的证据会话。

本模块只管理日志文件名、旁路 manifest 与文件哈希。run_id 不写入 canonical
逐帧 payload，因此不会改变 Unity UDP contract，也不会进入控制链路。
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Mapping
import uuid


RUNTIME_SESSION_SCHEMA_VERSION = "handai-runtime-session-v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_run_id() -> str:
    """生成可安全放入文件名、且能跨输出目录复用的一次运行标识。"""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def build_jsonl_session_path(
    cfg: Mapping[str, Any],
    *,
    output_dir_key: str = "jsonl_output_dir",
    prefix: str = "session",
    run_id: str | None = None,
) -> str:
    """为一次运行生成 JSONL 路径；显式 run_id 可绑定多个输出。"""

    output_dir = Path(str(cfg.get(output_dir_key, "outputs"))).resolve()
    identity = run_id or build_run_id()
    return str(output_dir / f"{prefix}_{identity}.jsonl")


@dataclass(frozen=True)
class RuntimeSessionArtifacts:
    run_id: str
    baseline_jsonl_path: Path
    prediction_jsonl_path: Path | None
    manifest_path: Path


def create_runtime_session_artifacts(
    cfg: Mapping[str, Any],
    *,
    prediction_requested: bool,
    run_id: str | None = None,
) -> RuntimeSessionArtifacts:
    identity = run_id or build_run_id()
    baseline_path = Path(
        build_jsonl_session_path(cfg, run_id=identity)
    )
    prediction_path = (
        Path(
            build_jsonl_session_path(
                cfg,
                output_dir_key="prediction_shadow_jsonl_output_dir",
                prefix="prediction_session",
                run_id=identity,
            )
        )
        if prediction_requested
        else None
    )
    manifest_path = baseline_path.parent / f"runtime_session_{identity}.json"
    return RuntimeSessionArtifacts(
        run_id=identity,
        baseline_jsonl_path=baseline_path,
        prediction_jsonl_path=prediction_path,
        manifest_path=manifest_path,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path | None, *, rows: int | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    exists = resolved.is_file()
    return {
        "path": str(resolved),
        "exists": exists,
        "bytes": resolved.stat().st_size if exists else None,
        "sha256": file_sha256(resolved) if exists else None,
        "rows": int(rows) if rows is not None else None,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class RuntimeSessionRecorder:
    """增量记录一次实时运行，结束时冻结日志路径、哈希和 worker 统计。"""

    def __init__(
        self,
        artifacts: RuntimeSessionArtifacts,
        *,
        config_path: Path,
        cfg: Mapping[str, Any],
        runtime: Any,
    ) -> None:
        self.artifacts = artifacts
        self._baseline_rows = 0
        self._first_frame_index: int | None = None
        self._last_frame_index: int | None = None
        self._first_timestamp: float | None = None
        self._last_timestamp: float | None = None
        resolved_config = config_path.resolve()
        self._manifest: dict[str, Any] = {
            "schema_version": RUNTIME_SESSION_SCHEMA_VERSION,
            "run_id": artifacts.run_id,
            "status": "running",
            "claim_status": "single_right_hand_unity_preview_runtime_evidence",
            "created_at_utc": utc_now_iso(),
            "completed_at_utc": None,
            "error": None,
            "config": {
                "path": str(resolved_config),
                "sha256": file_sha256(resolved_config),
                "resolved": dict(cfg),
            },
            "runtime": {
                "input_source_type": str(runtime.input_source_type),
                "camera_index": int(cfg.get("camera_index", 0)),
                "video_file_path": runtime.video_file_path,
                "input_mirrored": bool(runtime.input_mirrored),
                "control_extension_enabled": bool(runtime.control_extension_enabled),
                "svh_preview_enabled": bool(runtime.svh_preview_enabled),
                "unity_udp_enabled": bool(cfg.get("unity_udp_enabled", False)),
                "unity_udp_host": str(cfg.get("unity_udp_host", "127.0.0.1")),
                "unity_udp_port": int(cfg.get("unity_udp_port", 18080)),
                "prediction_shadow_requested": bool(cfg.get("prediction_shadow_enabled", False)),
                "save_jsonl": bool(cfg.get("save_jsonl", False)),
            },
            "outputs": {
                "baseline_jsonl": _file_record(artifacts.baseline_jsonl_path, rows=0),
                "prediction_jsonl": _file_record(artifacts.prediction_jsonl_path, rows=0),
            },
            "frames": {
                "processed": 0,
                "first_frame_index": None,
                "last_frame_index": None,
                "first_timestamp_unix_s": None,
                "last_timestamp_unix_s": None,
            },
            "prediction": {
                "enabled": False,
                "model_label": None,
                "device": None,
                "history_frames": None,
                "horizon_ms": [],
                "selection_sha256": None,
                "checkpoint_sha256": None,
                "initialization_error": None,
                "worker": None,
            },
            "safety": {
                "single_right_hand_only": True,
                "prediction_shadow_only": True,
                "prediction_modifies_unity_udp": False,
                "real_svh_in_scope": False,
            },
        }
        self.write_snapshot()

    @property
    def manifest_path(self) -> Path:
        return self.artifacts.manifest_path

    def write_snapshot(self) -> None:
        _atomic_write_json(self.artifacts.manifest_path, self._manifest)

    def record_prediction_identity(self, shadow: Any | None) -> None:
        if shadow is None:
            return
        self._manifest["prediction"] = {
            "enabled": True,
            "model_label": shadow.model_label,
            "device": shadow.device,
            "history_frames": int(shadow.history_frames),
            "horizon_ms": [int(value) for value in shadow.horizon_ms],
            "selection_sha256": shadow.selection_sha256,
            "checkpoint_sha256": shadow.checkpoint_sha256,
            "initialization_error": shadow.initialization_error,
            "worker": None,
        }
        self.write_snapshot()

    def observe_baseline(self, payload: Mapping[str, Any]) -> None:
        frame_index = int(payload["frame_index"])
        timestamp = float(payload["timestamp"])
        if self._first_frame_index is None:
            self._first_frame_index = frame_index
            self._first_timestamp = timestamp
        self._last_frame_index = frame_index
        self._last_timestamp = timestamp
        self._baseline_rows += 1

    def finalize(
        self,
        *,
        status: str,
        error: BaseException | None,
        baseline_exporter: Any | None,
        prediction_exporter: Any | None,
        prediction_worker: Any | None,
        prediction_worker_stopped: bool | None,
    ) -> None:
        if status not in {"completed", "interrupted", "failed"}:
            raise ValueError(f"不支持的 runtime session 状态：{status}")
        baseline_stats = baseline_exporter.diagnostic_snapshot() if baseline_exporter is not None else {}
        prediction_stats = (
            prediction_exporter.diagnostic_snapshot()
            if prediction_exporter is not None
            else {}
        )
        baseline_rows = int(baseline_stats.get("jsonl_write_count", self._baseline_rows))
        prediction_rows = (
            int(prediction_stats.get("jsonl_write_count", 0))
            if self.artifacts.prediction_jsonl_path is not None
            else None
        )
        self._manifest["status"] = status
        self._manifest["completed_at_utc"] = utc_now_iso()
        self._manifest["error"] = (
            f"{type(error).__name__}: {error}" if error is not None else None
        )
        self._manifest["outputs"] = {
            "baseline_jsonl": _file_record(
                self.artifacts.baseline_jsonl_path,
                rows=baseline_rows,
            ),
            "prediction_jsonl": _file_record(
                self.artifacts.prediction_jsonl_path,
                rows=prediction_rows,
            ),
        }
        self._manifest["frames"] = {
            "processed": self._baseline_rows,
            "first_frame_index": self._first_frame_index,
            "last_frame_index": self._last_frame_index,
            "first_timestamp_unix_s": self._first_timestamp,
            "last_timestamp_unix_s": self._last_timestamp,
        }
        if prediction_worker is not None:
            self._manifest["prediction"]["worker"] = {
                "submitted": int(prediction_worker.submitted_count),
                "completed": int(prediction_worker.completed_count),
                "dropped_input": int(prediction_worker.dropped_input_count),
                "dropped_result": int(prediction_worker.dropped_result_count),
                "stopped": prediction_worker_stopped,
            }
        self.write_snapshot()
