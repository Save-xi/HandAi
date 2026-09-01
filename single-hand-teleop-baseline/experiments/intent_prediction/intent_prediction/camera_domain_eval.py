from __future__ import annotations

"""真实摄像头视频域的确定性评测工具。

算法评测和实时能力评测在这里被刻意拆开：

- 固定视频按容器 PTS（失败时按 frame_index / nominal_fps）构造源时间轴；
- MediaPipe、control_representation 与 svh_preview 仍复用现役单右手代码；
- 预测模型在生成 baseline JSONL 后同步重放，不经过 latest-only worker；
- 真实摄像头运行产生的 baseline/prediction JSONL 只用于评估吞吐与异步覆盖。

本模块不创建 UDP exporter，也不接触 Unity 或真实 SVH。
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import logging
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from gesture.rule_based_gesture import GestureStabilizer
from main import RuntimeMode, _apply_extension_chain, _build_baseline_payload, _build_detector
from output.frame_payload_contract import prepare_frame_payload
from prediction.shadow_predictor import build_prediction_shadow
from utils.config import load_config
from utils.runtime_session import RUNTIME_SESSION_SCHEMA_VERSION

from intent_prediction.delay_injection import (
    RuntimeTraceGroup,
    _concat_scenario_arrays,
    _summarize_arrays,
    _write_scenario_csv,
    _write_sequence_csv,
    build_runtime_jsonl_forecast_traces,
    evaluate_network_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "camera_domain_eval_v1.json"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs" / "camera_domain_eval_v1"
REPORT_SCHEMA_VERSION = "camera-domain-eval-report-v1"
UNITY_TIMING_SUMMARY_SCHEMA_VERSION = "handai-unity-timing-summary-v1"


@dataclass(frozen=True)
class VideoSpec:
    """命令行给出的一段固定视频。"""

    video_id: str
    path: Path


@dataclass(frozen=True)
class TimestampDecision:
    """一帧最终采用的媒体时间戳及来源。"""

    timestamp_ms: float
    source: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_run_dir(output_root: Path) -> Path:
    run_dir = output_root.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _resolve_relative(base_file: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_file.resolve().parent / candidate).resolve()


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _percentiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"p50": None, "p95": None, "max": None}
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _nonnegative_int_property(value: Any) -> int:
    if not _finite_number(value) or float(value) < 0.0:
        return 0
    return int(float(value))


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def resolve_media_timestamp_ms(
    frame_index: int,
    *,
    raw_pts_ms: float | None,
    nominal_fps: float,
    previous_timestamp_ms: float | None,
) -> TimestampDecision:
    """选择严格递增的媒体时间戳，绝不使用处理 wall-clock 代替源时间轴。"""

    if frame_index < 0:
        raise ValueError("frame_index 必须是非负整数")
    if not math.isfinite(float(nominal_fps)) or float(nominal_fps) <= 0.0:
        raise ValueError("nominal_fps 必须是有限正数")
    period_ms = 1000.0 / float(nominal_fps)
    raw_valid = raw_pts_ms is not None and math.isfinite(float(raw_pts_ms)) and float(raw_pts_ms) >= 0.0
    if raw_valid and (
        previous_timestamp_ms is None or float(raw_pts_ms) > previous_timestamp_ms + 1e-6
    ):
        return TimestampDecision(float(raw_pts_ms), "container_pts_ms")

    fallback_ms = float(frame_index) * period_ms
    if previous_timestamp_ms is not None and fallback_ms <= previous_timestamp_ms + 1e-6:
        fallback_ms = previous_timestamp_ms + period_ms
        source = "continuity_fallback_period"
    else:
        source = "frame_index_over_nominal_fps"
    return TimestampDecision(float(fallback_ms), source)


def parse_video_specs(values: Iterable[str]) -> list[VideoSpec]:
    """解析 ``V1=D:\\path\\clip.mp4`` 形式的视频参数。"""

    specs: list[VideoSpec] = []
    seen: set[str] = set()
    seen_paths: set[Path] = set()
    for raw in values:
        video_id, separator, raw_path = str(raw).partition("=")
        video_id = video_id.strip()
        if not separator or not video_id or not raw_path.strip():
            raise ValueError(f"视频参数必须是 ID=PATH：{raw}")
        if not all(character.isalnum() or character in {"-", "_"} for character in video_id):
            raise ValueError(f"视频 ID 只能包含字母、数字、-、_：{video_id}")
        if video_id in seen:
            raise ValueError(f"视频 ID 重复：{video_id}")
        path = Path(raw_path.strip()).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"视频文件不存在：{path}")
        if path in seen_paths:
            raise ValueError(f"同一视频文件不能冒充多个任务：{path}")
        seen.add(video_id)
        seen_paths.add(path)
        specs.append(VideoSpec(video_id=video_id, path=path))
    if not specs:
        raise ValueError("至少需要一个 --video ID=PATH")
    return specs


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根节点必须是对象：{path}")
    return value


def load_evaluation_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    config = _read_json(path)
    if config.get("schema_version") != "camera-domain-eval-config-v1":
        raise ValueError("camera-domain 配置 schema_version 不受支持")
    video_sets = config.get("video_sets")
    if not isinstance(video_sets, dict):
        raise ValueError("video_sets 配置缺失")
    for role in ("development", "blind"):
        required_ids = video_sets.get(role)
        if not isinstance(required_ids, list) or not required_ids or len(set(required_ids)) != len(required_ids):
            raise ValueError(f"video_sets.{role} 必须是非空且不重复的列表")
    timeline = config.get("timeline")
    if not isinstance(timeline, dict):
        raise ValueError("timeline 配置缺失")
    for key in ("synthetic_epoch_seconds", "minimum_nominal_fps", "maximum_nominal_fps"):
        if not _finite_number(timeline.get(key)):
            raise ValueError(f"timeline.{key} 必须是有限数字")
    if float(timeline["minimum_nominal_fps"]) <= 0.0:
        raise ValueError("minimum_nominal_fps 必须大于 0")
    if float(timeline["maximum_nominal_fps"]) <= float(timeline["minimum_nominal_fps"]):
        raise ValueError("maximum_nominal_fps 必须大于 minimum_nominal_fps")
    return config


def verify_protocol_state(
    *,
    role: str,
    evaluation_config: dict[str, Any],
    evaluation_config_path: Path,
    protocol_path: Path,
    expected_config_sha256: str | None,
    expected_protocol_sha256: str | None,
) -> dict[str, Any]:
    """开发集允许草稿；盲测必须同时冻结机器配置和人类可读协议。"""

    config_sha256 = _hash_file(evaluation_config_path)
    protocol_sha256 = _hash_file(protocol_path)
    if expected_config_sha256 and expected_config_sha256.lower() != config_sha256:
        raise ValueError("evaluation config SHA-256 与预期不一致")
    if expected_protocol_sha256 and expected_protocol_sha256.lower() != protocol_sha256:
        raise ValueError("protocol SHA-256 与预期不一致")
    if role == "blind":
        blind = evaluation_config.get("blind_policy")
        if evaluation_config.get("protocol_stage") != "blind_frozen":
            raise ValueError("当前配置仍是 development，拒绝运行盲测准入")
        if not isinstance(blind, dict) or blind.get("enabled") is not True or not isinstance(blind.get("gate"), dict):
            raise ValueError("blind_policy 尚未冻结")
        if not expected_config_sha256 or not expected_protocol_sha256:
            raise ValueError("盲测必须显式提供配置和协议的预期 SHA-256")
    return {
        "role": role,
        "protocol_stage": evaluation_config.get("protocol_stage"),
        "evaluation_config_path": str(evaluation_config_path),
        "evaluation_config_sha256": config_sha256,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "frozen_for_this_run": bool(expected_config_sha256 and expected_protocol_sha256),
    }


def _git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    try:
        revision = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current")
        status = run("status", "--short")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    status_lines = status.splitlines() if status else []
    tracked_changes = [line for line in status_lines if not line.startswith("??")]
    untracked_paths = [line[3:] for line in status_lines if line.startswith("??")]
    return {
        "available": True,
        "revision": revision,
        "branch": branch,
        "worktree_clean": not status_lines,
        "tracked_worktree_clean": not tracked_changes,
        "tracked_status_lines": tracked_changes,
        "untracked_paths": untracked_paths,
        "status_lines": status_lines,
    }


def _build_video_payload_processor(
    cfg: dict[str, Any],
    *,
    input_mirrored: bool,
    logger: logging.Logger,
) -> tuple[Callable[[Any, int, float, float], tuple[dict[str, Any], float]], Callable[[], None]]:
    runtime = RuntimeMode(
        gui_enabled=False,
        headless=True,
        input_source_type="video_file",
        input_mirrored=input_mirrored,
        control_extension_enabled=True,
        svh_preview_enabled=True,
        video_file_path=None,
    )
    detector = _build_detector(cfg, runtime)
    stabilizer = GestureStabilizer(
        confirm_frames=int(cfg.get("stable_gesture_min_consecutive", 2)),
        unknown_confirm_frames=int(cfg.get("stable_unknown_consecutive", 1)),
    )

    def process(frame: Any, frame_index: int, timestamp_s: float, source_fps: float) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        payload = _build_baseline_payload(
            frame,
            detector,
            cfg,
            stabilizer,
            draw_landmarks=False,
        )
        # _build_baseline_payload 内部 wall-clock 只用于实时运行；固定视频评测必须覆盖为媒体时间。
        payload["timestamp"] = float(timestamp_s)
        _apply_extension_chain(
            payload,
            cfg,
            runtime,
            svh_transport=None,
            logger=logger,
        )
        payload["frame_index"] = int(frame_index)
        payload["fps"] = float(source_fps)
        payload["latency_ms"] = (time.perf_counter() - started) * 1000.0
        prepared = prepare_frame_payload(payload, include_deprecated_aliases=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return prepared, float(elapsed_ms)

    return process, detector.close


def process_video_to_baseline_jsonl(
    spec: VideoSpec,
    *,
    output_path: Path,
    cfg: dict[str, Any],
    timeline_config: dict[str, Any],
    input_mirrored: bool,
    logger: logging.Logger,
    capture_factory: Callable[[str], Any] = cv2.VideoCapture,
) -> dict[str, Any]:
    """按媒体时间轴生成不含 prediction_diagnostics 的 canonical baseline JSONL。"""

    capture = capture_factory(str(spec.path))
    if capture is None or not capture.isOpened():
        if capture is not None:
            capture.release()
        raise RuntimeError(f"无法打开视频：{spec.path}")
    nominal_fps = float(capture.get(cv2.CAP_PROP_FPS))
    minimum_fps = float(timeline_config["minimum_nominal_fps"])
    maximum_fps = float(timeline_config["maximum_nominal_fps"])
    if not math.isfinite(nominal_fps) or not minimum_fps <= nominal_fps <= maximum_fps:
        capture.release()
        raise ValueError(f"视频 nominal FPS 无效：{nominal_fps}")
    metadata_frame_count = _nonnegative_int_property(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = _nonnegative_int_property(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = _nonnegative_int_property(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    epoch_seconds = float(timeline_config["synthetic_epoch_seconds"])
    try:
        processor, close_processor = _build_video_payload_processor(
            cfg,
            input_mirrored=input_mirrored,
            logger=logger,
        )
    except Exception:
        capture.release()
        raise
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    previous_timestamp_ms: float | None = None
    timestamps_ms: list[float] = []
    processing_ms: list[float] = []
    timestamp_source_counts: dict[str, int] = {}
    detected_frames = 0
    control_ready_frames = 0
    svh_valid_frames = 0
    wall_started = time.perf_counter()
    try:
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                raw_pts = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                decision = resolve_media_timestamp_ms(
                    frame_index,
                    raw_pts_ms=raw_pts,
                    nominal_fps=nominal_fps,
                    previous_timestamp_ms=previous_timestamp_ms,
                )
                previous_timestamp_ms = decision.timestamp_ms
                timestamps_ms.append(decision.timestamp_ms)
                timestamp_source_counts[decision.source] = timestamp_source_counts.get(decision.source, 0) + 1
                payload, elapsed_ms = processor(
                    frame,
                    frame_index,
                    epoch_seconds + decision.timestamp_ms / 1000.0,
                    nominal_fps,
                )
                processing_ms.append(elapsed_ms)
                detected_frames += int(payload.get("detected") is True)
                control_ready_frames += int(payload.get("control_ready") is True)
                preview = payload.get("svh_preview")
                svh_valid_frames += int(isinstance(preview, dict) and preview.get("valid") is True)
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                frame_index += 1
    finally:
        capture.release()
        close_processor()
    wall_seconds = time.perf_counter() - wall_started
    if frame_index == 0:
        raise RuntimeError(f"视频没有可读帧：{spec.path}")
    deltas_ms = np.diff(np.asarray(timestamps_ms, dtype=np.float64))
    if deltas_ms.size and np.any(deltas_ms <= 0.0):
        raise RuntimeError("内部错误：媒体时间轴不是严格递增")
    median_interval_fps = (
        float(1000.0 / np.median(deltas_ms)) if deltas_ms.size else nominal_fps
    )
    duration_ms = float(timestamps_ms[-1] - timestamps_ms[0]) if frame_index > 1 else 0.0
    duration_based_fps = (
        float((frame_index - 1) * 1000.0 / duration_ms)
        if frame_index > 1 and duration_ms > 0.0
        else nominal_fps
    )
    return {
        "video_id": spec.video_id,
        "video_path": str(spec.path),
        "video_sha256": _hash_file(spec.path),
        "video_bytes": spec.path.stat().st_size,
        "baseline_jsonl_path": str(output_path.resolve()),
        "baseline_jsonl_sha256": _hash_file(output_path),
        "metadata": {
            "nominal_fps": nominal_fps,
            "metadata_frame_count": metadata_frame_count,
            "decoded_frame_count": frame_index,
            "width": width,
            "height": height,
        },
        "source_timeline": {
            "clock": "synthetic_epoch_plus_media_time",
            "synthetic_epoch_seconds": epoch_seconds,
            "strictly_increasing": True,
            "first_media_timestamp_ms": timestamps_ms[0],
            "last_media_timestamp_ms": timestamps_ms[-1],
            "duration_ms": duration_ms,
            "median_interval_fps": median_interval_fps,
            "duration_based_fps": duration_based_fps,
            # 兼容 v1 报告读取方；新报告与 Markdown 不再把它笼统称为“源 FPS”。
            "effective_fps": median_interval_fps,
            "timestamp_source_counts": timestamp_source_counts,
        },
        "offline_processing_capacity": {
            "meaning": (
                "throughput 包含解码、检测、映射和 JSONL 写入；processing_ms 是单帧检测/映射/规范化。"
                "二者都不是视频源 FPS，也不是真实 worker 覆盖率。"
            ),
            "wall_seconds": float(wall_seconds),
            "throughput_fps": float(frame_index / wall_seconds) if wall_seconds > 0.0 else None,
            "processing_ms": _percentiles(processing_ms),
        },
        "observation_counts": {
            "frames": frame_index,
            "detected_frames": detected_frames,
            "control_ready_frames": control_ready_frames,
            "svh_valid_frames": svh_valid_frames,
            "detected_fraction": float(detected_frames / frame_index),
            "control_ready_fraction": float(control_ready_frames / frame_index),
            "svh_valid_fraction": float(svh_valid_frames / frame_index),
        },
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 根节点不是对象")
            rows.append(value)
    return rows


def _extract_session_run_id(path: Path, *, prefix: str) -> str:
    pattern = re.compile(rf"^{re.escape(prefix)}_(?P<run_id>[A-Za-z0-9][A-Za-z0-9.-]*)\.jsonl$")
    match = pattern.fullmatch(path.name)
    if match is None:
        raise ValueError(
            f"{path.name} 不符合严格会话命名 {prefix}_<run_id>.jsonl；"
            "请使用新版 --save-jsonl 产生的日志"
        )
    return match.group("run_id")


def _same_resolved_path(first: Path, second: Path) -> bool:
    return str(first.resolve()).casefold() == str(second.resolve()).casefold()


def _validate_manifest_output(
    record: Any,
    *,
    expected_path: Path,
    expected_rows: int,
    label: str,
) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"runtime manifest 缺少 {label} 输出记录")
    recorded_path = record.get("path")
    if not isinstance(recorded_path, str) or not _same_resolved_path(Path(recorded_path), expected_path):
        raise ValueError(f"runtime manifest 的 {label} 路径与传入日志不一致")
    if record.get("exists") is not True:
        raise ValueError(f"runtime manifest 标记 {label} 不存在")
    if record.get("rows") != expected_rows:
        raise ValueError(
            f"runtime manifest 的 {label} 行数不一致："
            f"manifest={record.get('rows')}, actual={expected_rows}"
        )
    if record.get("bytes") != expected_path.stat().st_size:
        raise ValueError(f"runtime manifest 的 {label} 字节数不一致")
    if record.get("sha256") != _hash_file(expected_path):
        raise ValueError(f"runtime manifest 的 {label} SHA-256 不一致")


def _validate_runtime_manifest(
    manifest_path: Path,
    *,
    run_id: str,
    baseline_path: Path,
    prediction_path: Path,
    baseline_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != RUNTIME_SESSION_SCHEMA_VERSION:
        raise ValueError("runtime manifest schema_version 不受支持")
    if manifest.get("run_id") != run_id:
        raise ValueError("runtime manifest run_id 与日志文件名不一致")
    if manifest.get("status") not in {"completed", "interrupted"}:
        raise ValueError(f"runtime manifest 尚不可用于评估：status={manifest.get('status')}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("runtime manifest 缺少 outputs")
    _validate_manifest_output(
        outputs.get("baseline_jsonl"),
        expected_path=baseline_path,
        expected_rows=len(baseline_rows),
        label="baseline_jsonl",
    )
    _validate_manifest_output(
        outputs.get("prediction_jsonl"),
        expected_path=prediction_path,
        expected_rows=len(prediction_rows),
        label="prediction_jsonl",
    )
    frames = manifest.get("frames")
    if not isinstance(frames, dict) or frames.get("processed") != len(baseline_rows):
        raise ValueError("runtime manifest 的 processed 帧数与 baseline JSONL 不一致")
    first = baseline_rows[0]
    last = baseline_rows[-1]
    expected_frame_fields = {
        "first_frame_index": first.get("frame_index"),
        "last_frame_index": last.get("frame_index"),
        "first_timestamp_unix_s": first.get("timestamp"),
        "last_timestamp_unix_s": last.get("timestamp"),
    }
    for key, expected in expected_frame_fields.items():
        actual = frames.get(key)
        if _finite_number(expected):
            if not _finite_number(actual) or not math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"runtime manifest 的 {key} 与 baseline JSONL 不一致")
        elif actual != expected:
            raise ValueError(f"runtime manifest 的 {key} 与 baseline JSONL 不一致")
    config = manifest.get("config")
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("path"), str)
        or not isinstance(config.get("sha256"), str)
        or len(config["sha256"]) != 64
    ):
        raise ValueError("runtime manifest 缺少配置文件身份")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("prediction_shadow_requested") is not True:
        raise ValueError("runtime manifest 未证明本次运行请求了 prediction shadow")
    safety = manifest.get("safety")
    if not isinstance(safety, dict) or safety.get("prediction_modifies_unity_udp") is not False:
        raise ValueError("runtime manifest 缺少 prediction 不改 UDP 的安全边界")
    return manifest


def _validate_unity_timing_summary(
    path: Path,
    *,
    baseline_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    resolved = path.resolve()
    summary = _read_json(resolved)
    if summary.get("schema_version") != UNITY_TIMING_SUMMARY_SCHEMA_VERSION:
        raise ValueError("Unity timing summary schema_version 不受支持")
    source = summary.get("source_session")
    if not isinstance(source, dict):
        raise ValueError("Unity timing summary 缺少 source_session")
    first_index = source.get("first_frame_index")
    last_index = source.get("last_frame_index")
    if (
        not isinstance(first_index, int)
        or isinstance(first_index, bool)
        or not isinstance(last_index, int)
        or isinstance(last_index, bool)
        or first_index > last_index
        or first_index not in baseline_by_index
        or last_index not in baseline_by_index
    ):
        raise ValueError("Unity timing summary 的首末 frame_index 无法与 baseline 配对")
    for index, key in (
        (first_index, "first_source_timestamp_unix_ms"),
        (last_index, "last_source_timestamp_unix_ms"),
    ):
        actual = source.get(key)
        expected = float(baseline_by_index[index]["timestamp"]) * 1000.0
        if not _finite_number(actual) or not math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=0.01
        ):
            raise ValueError(f"Unity timing summary 的 {key} 与 baseline 不一致")
    accepted = summary.get("accepted_packet_count")
    if (
        not isinstance(accepted, int)
        or isinstance(accepted, bool)
        or accepted <= 0
        or accepted > len(baseline_by_index)
    ):
        raise ValueError("Unity timing summary 的 accepted_packet_count 无效")
    metrics = summary.get("metrics")
    required_metrics = (
        "python_post_capture_to_udp_ms",
        "udp_delivery_ms",
        "unity_main_thread_queue_ms",
        "source_read_end_to_target_apply_ms",
    )
    if not isinstance(metrics, dict):
        raise ValueError("Unity timing summary 缺少 metrics")
    for name in required_metrics:
        metric = metrics.get(name)
        if not isinstance(metric, dict):
            raise ValueError(f"Unity timing summary 缺少指标 {name}")
        total = metric.get("total_sample_count")
        retained = metric.get("retained_sample_count")
        capacity = metric.get("capacity")
        if (
            not isinstance(total, int)
            or isinstance(total, bool)
            or not isinstance(retained, int)
            or isinstance(retained, bool)
            or not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or capacity <= 0
            or total != accepted
            or retained != min(total, capacity)
        ):
            raise ValueError(f"Unity timing summary 指标 {name} 的样本计数无效")
        p50 = metric.get("p50_ms")
        p95 = metric.get("p95_ms")
        maximum = metric.get("max_ms")
        if (
            not _finite_number(p50)
            or not _finite_number(p95)
            or not _finite_number(maximum)
            or float(p50) > float(p95)
            or float(p95) > float(maximum)
        ):
            raise ValueError(f"Unity timing summary 指标 {name} 的分位数无效")

    counters = summary.get("counters")
    required_counters = (
        "overwritten_packet_count",
        "frame_gap_count",
        "rejected_packet_count",
        "stale_packet_count",
        "watchdog_open_count",
    )
    if not isinstance(counters, dict):
        raise ValueError("Unity timing summary 缺少 counters")
    for name in required_counters:
        value = counters.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Unity timing summary 计数器 {name} 无效")

    safety = summary.get("safety")
    if not isinstance(safety, dict):
        raise ValueError("Unity timing summary 缺少 safety")
    required_false_safety = (
        "baseline_udp_hardware_forwarding_compiled",
        "apply_baseline_preview_to_hardware",
        "real_svh_in_scope",
    )
    for name in required_false_safety:
        if safety.get(name) is not False:
            raise ValueError(f"Unity timing summary 安全边界 {name} 未保持关闭")
    return {
        "path": str(resolved),
        "sha256": _hash_file(resolved),
        "pairing_policy": (
            "first_and_last_source_frame_plus_timestamp_with_timing_counters_and_safety"
        ),
        "summary": summary,
    }


def analyze_live_runtime_logs(
    baseline_path: Path,
    prediction_path: Path,
    *,
    manifest_path: Path | None = None,
    unity_timing_path: Path | None = None,
) -> dict[str, Any]:
    """严格配对并分析真实异步 worker 日志；不替代同步算法重放。"""

    baseline_path = baseline_path.resolve()
    prediction_path = prediction_path.resolve()
    baseline_run_id = _extract_session_run_id(baseline_path, prefix="session")
    prediction_run_id = _extract_session_run_id(
        prediction_path,
        prefix="prediction_session",
    )
    if baseline_run_id != prediction_run_id:
        raise ValueError("live baseline 与 prediction JSONL 的 run_id 不一致")
    baseline_rows = _load_jsonl(baseline_path)
    prediction_rows = _load_jsonl(prediction_path)
    if not baseline_rows:
        raise ValueError("live baseline JSONL 为空")

    baseline_by_index: dict[int, dict[str, Any]] = {}
    baseline_order: list[int] = []
    timestamps: list[float] = []
    for row_number, row in enumerate(baseline_rows, start=1):
        frame_index = row.get("frame_index")
        timestamp = row.get("timestamp")
        if not isinstance(frame_index, int) or isinstance(frame_index, bool):
            raise ValueError(f"live baseline 第 {row_number} 行 frame_index 非法")
        if frame_index in baseline_by_index:
            raise ValueError(f"live baseline 存在重复 frame_index={frame_index}")
        if not _finite_number(timestamp):
            raise ValueError(f"live baseline 第 {row_number} 行 timestamp 非法")
        baseline_by_index[frame_index] = row
        baseline_order.append(frame_index)
        timestamps.append(float(timestamp))
    if len(baseline_order) >= 2 and np.any(np.diff(np.asarray(baseline_order)) <= 0):
        raise ValueError("live baseline frame_index 未严格递增")
    timestamp_deltas = np.diff(np.asarray(timestamps, dtype=np.float64))
    if timestamp_deltas.size and np.any(timestamp_deltas <= 0.0):
        raise ValueError("live baseline timestamp 未严格递增")

    resolved_manifest = (
        manifest_path.resolve()
        if manifest_path is not None
        else baseline_path.parent / f"runtime_session_{baseline_run_id}.json"
    )
    manifest = _validate_runtime_manifest(
        resolved_manifest,
        run_id=baseline_run_id,
        baseline_path=baseline_path,
        prediction_path=prediction_path,
        baseline_rows=baseline_rows,
        prediction_rows=prediction_rows,
    )
    prediction_identity = manifest.get("prediction")
    if not isinstance(prediction_identity, dict) or prediction_identity.get("enabled") is not True:
        raise ValueError("runtime manifest 缺少已启用的 prediction 身份")

    valid_indexes = {
        index
        for index, row in baseline_by_index.items()
        if row.get("control_ready") is True
        and isinstance(row.get("svh_preview"), dict)
        and row["svh_preview"].get("valid") is True
    }
    prediction_by_index: dict[int, dict[str, Any]] = {}
    prediction_order: list[int] = []
    status_counts: dict[str, int] = {}
    inference_ms: list[float] = []
    observed_fps: list[float] = []
    identity_fields = {
        "model_label": "model_label",
        "device": "device",
        "history_frames_required": "history_frames",
        "horizon_ms": "horizon_ms",
        "selection_sha256": "selection_sha256",
        "checkpoint_sha256": "checkpoint_sha256",
    }
    for row_number, row in enumerate(prediction_rows, start=1):
        diagnostics = row.get("prediction_diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError(f"live prediction 第 {row_number} 行缺少 prediction_diagnostics")
        source_index = diagnostics.get("source_frame_index")
        if not isinstance(source_index, int) or isinstance(source_index, bool):
            raise ValueError(f"live prediction 第 {row_number} 行 source_frame_index 非法")
        if source_index in prediction_by_index:
            raise ValueError(f"live prediction 存在重复 source_frame_index={source_index}")
        if source_index not in baseline_by_index:
            raise ValueError(f"live prediction 引用了 baseline 中不存在的 frame_index={source_index}")
        if row.get("frame_index") != source_index:
            raise ValueError(f"live prediction 顶层 frame_index 与诊断源帧不一致：{source_index}")
        baseline_timestamp = float(baseline_by_index[source_index]["timestamp"])
        if not _finite_number(row.get("timestamp")) or not math.isclose(
            float(row["timestamp"]), baseline_timestamp, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"live prediction 与 baseline timestamp 不一致：frame_index={source_index}")
        source_timestamp_ms = diagnostics.get("source_timestamp_unix_ms")
        if not _finite_number(source_timestamp_ms) or not math.isclose(
            float(source_timestamp_ms), baseline_timestamp * 1000.0, rel_tol=0.0, abs_tol=0.01
        ):
            raise ValueError(
                f"live prediction 诊断源时间戳与 baseline 不一致：frame_index={source_index}"
            )
        for diagnostic_key, manifest_key in identity_fields.items():
            if diagnostics.get(diagnostic_key) != prediction_identity.get(manifest_key):
                raise ValueError(
                    f"live prediction 的 {diagnostic_key} 与 runtime manifest 不一致"
                )
        prediction_by_index[source_index] = diagnostics
        prediction_order.append(source_index)
        status = str(diagnostics.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if _finite_number(diagnostics.get("inference_ms")):
            inference_ms.append(float(diagnostics["inference_ms"]))
        if _finite_number(diagnostics.get("observed_fps")):
            observed_fps.append(float(diagnostics["observed_fps"]))
    if len(prediction_order) >= 2 and np.any(np.diff(np.asarray(prediction_order)) <= 0):
        raise ValueError("live prediction source_frame_index 未严格递增")

    result_indexes = set(prediction_by_index)
    predicted_indexes = {
        index
        for index, diagnostics in prediction_by_index.items()
        if diagnostics.get("status") == "predicted"
    }
    baseline_count = len(baseline_by_index)
    valid_count = len(valid_indexes)
    source_duration_fps = (
        float((len(timestamps) - 1) / (timestamps[-1] - timestamps[0]))
        if len(timestamps) >= 2
        else None
    )
    source_median_interval_fps = (
        float(1.0 / np.median(timestamp_deltas)) if timestamp_deltas.size else None
    )
    latency_values = [
        float(row["latency_ms"])
        for row in baseline_rows
        if _finite_number(row.get("latency_ms"))
    ]
    source_read_call_ms: list[float] = []
    python_post_capture_to_udp_ms: list[float] = []
    for row in baseline_rows:
        timing = row.get("timing")
        if not isinstance(timing, dict) or timing.get("schema_version") != 1:
            continue
        read_start = timing.get("source_read_start_unix_ms")
        read_end = timing.get("source_read_end_unix_ms")
        udp_send = timing.get("udp_send_attempt_unix_ms")
        if _finite_number(read_start) and _finite_number(read_end):
            source_read_call_ms.append(float(read_end) - float(read_start))
        if _finite_number(read_end) and _finite_number(udp_send):
            python_post_capture_to_udp_ms.append(float(udp_send) - float(read_end))
    unity_timing = (
        _validate_unity_timing_summary(
            unity_timing_path,
            baseline_by_index=baseline_by_index,
        )
        if unity_timing_path is not None
        else None
    )
    config_record = manifest["config"]
    current_config_path = Path(config_record["path"])
    config_current_match = (
        current_config_path.is_file()
        and config_record["sha256"] == _hash_file(current_config_path)
    )
    return {
        "claim_status": "real_time_worker_diagnostics",
        "meaning": "只衡量真实运行吞吐、队列结果覆盖与状态；不替代媒体时间轴上的算法效用评测。",
        "run_id": baseline_run_id,
        "pairing_policy": "shared_run_id_plus_manifest_paths_hashes_rows_and_per_frame_identity",
        "manifest_path": str(resolved_manifest),
        "manifest_sha256": _hash_file(resolved_manifest),
        "manifest_status": manifest["status"],
        "manifest_config_current_match": config_current_match,
        "baseline_path": str(baseline_path),
        "baseline_sha256": _hash_file(baseline_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": _hash_file(prediction_path),
        "baseline_frames": baseline_count,
        "valid_control_frames": valid_count,
        "prediction_result_frames": len(result_indexes),
        "predicted_frames": len(predicted_indexes),
        "worker_result_coverage_all_frames": float(len(result_indexes) / baseline_count),
        "predicted_coverage_all_frames": float(len(predicted_indexes) / baseline_count),
        "predicted_coverage_valid_frames": (
            float(len(predicted_indexes & valid_indexes) / valid_count) if valid_count else None
        ),
        "status_counts": status_counts,
        "source_timeline_strictly_increasing": True,
        "source_duration_based_fps": source_duration_fps,
        "source_median_interval_fps": source_median_interval_fps,
        "baseline_processing_ms": _percentiles(latency_values),
        "source_read_call_ms": _percentiles(source_read_call_ms),
        "python_post_capture_to_udp_ms": _percentiles(python_post_capture_to_udp_ms),
        "prediction_observed_fps": _percentiles(observed_fps),
        "inference_ms": _percentiles(inference_ms),
        "prediction_identity": prediction_identity,
        "worker_manifest": prediction_identity.get("worker"),
        "unity_timing": unity_timing,
        "timing_boundary": (
            "Python post-capture 从 source.read() 返回后开始；Unity source->target apply 从同一边界到主线程应用。"
            "不包含相机曝光、传感器读取前等待、显示刷新或人体/机械响应。"
        ),
    }


def _evaluate_trace_set(
    groups: list[RuntimeTraceGroup],
    *,
    predictor: Any,
    delay_config: dict[str, Any],
    dynamic_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    traces = [trace for group in groups for trace in group.traces]
    configured_horizons = tuple(int(value) for value in delay_config.get("horizon_ms", []))
    if configured_horizons != tuple(int(value) for value in predictor.horizon_ms):
        raise ValueError("delay config horizon_ms 与现役 predictor 不一致")
    if int(delay_config.get("history_frames", -1)) != int(predictor.history_frames):
        raise ValueError("delay config history_frames 与现役 predictor 不一致")
    network = dict(delay_config["network_matrix"])
    scenarios, sequence_rows, arrays_by_scenario = evaluate_network_matrix(
        traces,
        horizon_ms=tuple(int(value) for value in predictor.horizon_ms),
        delays_ms=tuple(float(value) for value in network["delays_ms"]),
        jitters_ms=tuple(float(value) for value in network["jitters_ms"]),
        loss_rates=tuple(float(value) for value in network["loss_rates"]),
        seed=int(delay_config["seed"]),
        dynamic_threshold=dynamic_threshold,
    )
    primary_delays = {
        float(value) for value in delay_config["retention_gate"]["primary_delays_ms"]
    }
    primary_arrays = _concat_scenario_arrays(
        arrays for key, arrays in arrays_by_scenario.items() if key[0] in primary_delays
    )
    no_impairment = arrays_by_scenario.get((0.0, 0.0, 0.0))
    summary = {
        "source_count": len(groups),
        "segment_count": len(traces),
        "source_rows": {
            "total_rows": sum(group.total_rows for group in groups),
            "valid_rows": sum(group.valid_rows for group in groups),
            "evaluable_rows": sum(group.evaluable_rows for group in groups),
            "discarded_short_segment_rows": sum(
                group.discarded_short_segment_rows for group in groups
            ),
            "discarded_short_segment_count": sum(
                group.discarded_short_segment_count for group in groups
            ),
            "predicted_rows": sum(group.predicted_rows for group in groups),
        },
        "primary_delays_ms": sorted(primary_delays),
        "primary_delay_summary": _summarize_arrays(primary_arrays),
        "no_impairment_summary": _summarize_arrays(no_impairment) if no_impairment is not None else None,
    }
    return summary, scenarios, sequence_rows


def evaluate_camera_algorithm_utility(
    video_summaries: list[dict[str, Any]],
    *,
    runtime_cfg: dict[str, Any],
    delay_config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """同步重放 baseline JSONL；不使用异步 worker，因此不会因离线快读丢帧。"""

    predictor = build_prediction_shadow(runtime_cfg, logger=logger)
    if predictor is None:
        raise RuntimeError("runtime config 未启用 prediction shadow")
    if predictor.initialization_error is not None:
        raise RuntimeError(f"prediction shadow 初始化失败：{predictor.initialization_error}")
    report_path = Path(str(runtime_cfg["prediction_shadow_report_path"]))
    second_round_report = _read_json(report_path)
    dynamic_threshold = float(second_round_report["motion_strata_thresholds_from_validation"]["q90"])
    groups: list[RuntimeTraceGroup] = []
    source_errors: list[dict[str, str]] = []
    group_by_id: dict[str, RuntimeTraceGroup] = {}
    for summary in video_summaries:
        predictor.reset_history()
        baseline_path = Path(summary["baseline_jsonl_path"])
        try:
            group = build_runtime_jsonl_forecast_traces(
                baseline_path,
                predictor=predictor,
                recent_frames=int(predictor.gate_recent_frames),
            )
        except Exception as exc:
            logger.warning("%s 未形成可评估预测 trace：%s", summary["video_id"], exc)
            source_errors.append(
                {
                    "video_id": str(summary["video_id"]),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        groups.append(group)
        group_by_id[str(summary["video_id"])] = group
    identity = {
        "model_label": predictor.model_label,
        "device": predictor.device,
        "horizon_ms": list(predictor.horizon_ms),
        "history_frames": predictor.history_frames,
        "selection_sha256": predictor.selection_sha256,
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "offline_gate_passed": bool(
            second_round_report.get("acceptance", {}).get("offline_gate_passed", False)
        ),
        "selection_path": str(runtime_cfg["prediction_shadow_selection_path"]),
        "checkpoint_path": str(runtime_cfg["prediction_shadow_checkpoint_path"]),
        "second_round_report_path": str(runtime_cfg["prediction_shadow_report_path"]),
    }
    if not groups:
        return (
            {
                "status": "no_evaluable_source",
                "claim_status": "camera_domain_pseudo_ground_truth_only",
                "model_identity": identity,
                "source_errors": source_errors,
                "per_video": {},
            },
            [],
            [],
        )
    aggregate, scenarios, sequence_rows = _evaluate_trace_set(
        groups,
        predictor=predictor,
        delay_config=delay_config,
        dynamic_threshold=dynamic_threshold,
    )
    per_video: dict[str, Any] = {}
    for video_id, group in group_by_id.items():
        per_video[video_id] = _evaluate_trace_set(
            [group],
            predictor=predictor,
            delay_config=delay_config,
            dynamic_threshold=dynamic_threshold,
        )[0]
        per_video[video_id]["source"] = {
            "total_rows": group.total_rows,
            "valid_rows": group.valid_rows,
            "evaluable_rows": group.evaluable_rows,
            "discarded_short_segment_rows": group.discarded_short_segment_rows,
            "discarded_short_segment_count": group.discarded_short_segment_count,
            "predicted_rows": group.predicted_rows,
            "status_counts": group.status_counts,
            "segment_count": len(group.traces),
        }
    return (
        {
            "status": "evaluated",
            "claim_status": "camera_domain_pseudo_ground_truth_only",
            "meaning": (
                "目标是后续视觉映射得到的 svh_preview，不是实体关节真值或用户意图真值；"
                "重叠接收 tick 也不能当作独立统计样本。"
            ),
            "timeline_policy": "deterministic_media_timeline_synchronous_replay",
            "model_identity": identity,
            "dynamic_threshold_source": {
                "path": str(report_path),
                "sha256": _hash_file(report_path),
                "validation_q90": dynamic_threshold,
            },
            "source_errors": source_errors,
            "aggregate": aggregate,
            "per_video": per_video,
        },
        scenarios,
        sequence_rows,
    )


def _write_video_csv(path: Path, videos: list[dict[str, Any]], algorithm: dict[str, Any]) -> None:
    fields = [
        "video_id",
        "decoded_frames",
        "nominal_fps",
        "source_median_interval_fps",
        "source_duration_based_fps",
        "effective_source_fps",
        "throughput_fps",
        "detected_fraction",
        "control_ready_fraction",
        "svh_valid_fraction",
        "primary_gated_rmse_improvement_percent",
        "conditional_prediction_available_fraction",
        "end_to_end_prediction_coverage_fraction",
    ]
    per_video = algorithm.get("per_video") if isinstance(algorithm.get("per_video"), dict) else {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for video in videos:
            model_summary = per_video.get(video["video_id"], {})
            primary = model_summary.get("primary_delay_summary") if isinstance(model_summary, dict) else None
            writer.writerow(
                {
                    "video_id": video["video_id"],
                    "decoded_frames": video["metadata"]["decoded_frame_count"],
                    "nominal_fps": video["metadata"]["nominal_fps"],
                    "source_median_interval_fps": video["source_timeline"]["median_interval_fps"],
                    "source_duration_based_fps": video["source_timeline"]["duration_based_fps"],
                    "effective_source_fps": video["source_timeline"]["effective_fps"],
                    "throughput_fps": video["offline_processing_capacity"]["throughput_fps"],
                    "detected_fraction": video["observation_counts"]["detected_fraction"],
                    "control_ready_fraction": video["observation_counts"]["control_ready_fraction"],
                    "svh_valid_fraction": video["observation_counts"]["svh_valid_fraction"],
                    "primary_gated_rmse_improvement_percent": (
                        primary["improvement_percent_vs_hold"]["gated_rmse"] if primary else None
                    ),
                    "conditional_prediction_available_fraction": (
                        primary["conditional_prediction_available_fraction"] if primary else None
                    ),
                    "end_to_end_prediction_coverage_fraction": (
                        primary["end_to_end_prediction_coverage_fraction"] if primary else None
                    ),
                }
            )


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 真实摄像头域评测报告",
        "",
        f"- 运行角色：`{report['protocol']['role']}`",
        f"- 协议阶段：`{report['protocol']['protocol_stage']}`",
        f"- Git revision：`{report['git'].get('revision', 'unknown')}`",
        f"- 自动决策：`{report['decision']['status']}`",
        "",
        "## 固定视频与时间轴",
        "",
        "| 视频 | 帧数 | nominal FPS | PTS 中位间隔 FPS | PTS 时长均值 FPS | 离线吞吐 FPS | control ready | PTS/回退 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for video in report["videos"]:
        counts = video["source_timeline"]["timestamp_source_counts"]
        source_text = ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))
        lines.append(
            "| {video_id} | {frames} | {nominal:.3f} | {median:.3f} | {duration:.3f} | {throughput:.3f} | {ready:.2%} | {counts} |".format(
                video_id=video["video_id"],
                frames=video["metadata"]["decoded_frame_count"],
                nominal=video["metadata"]["nominal_fps"],
                median=video["source_timeline"]["median_interval_fps"],
                duration=video["source_timeline"]["duration_based_fps"],
                throughput=video["offline_processing_capacity"]["throughput_fps"] or 0.0,
                ready=video["observation_counts"]["control_ready_fraction"],
                counts=source_text,
            )
        )
    algorithm = report["algorithm_utility"]
    lines.extend(["", "## 算法效用（媒体时间轴同步重放）", ""])
    if algorithm.get("status") == "evaluated":
        primary = algorithm["aggregate"]["primary_delay_summary"]
        dynamic = primary.get("dynamic_q90")
        lines.extend(
            [
                f"- 总体 gated RMSE 改善：{primary['improvement_percent_vs_hold']['gated_rmse']}",
                f"- 条件预测覆盖：{primary['conditional_prediction_available_fraction']:.6f}",
                f"- 端到端预测覆盖：{primary['end_to_end_prediction_coverage_fraction']:.6f}",
                f"- 动态 q90 gated RMSE 改善：{dynamic['improvement_percent_vs_hold']['gated_rmse'] if dynamic else None}",
                (
                    "- trace 行数口径："
                    f"valid={algorithm['aggregate']['source_rows']['valid_rows']}，"
                    f"evaluable={algorithm['aggregate']['source_rows']['evaluable_rows']}，"
                    "因不足 2 帧被丢弃="
                    f"{algorithm['aggregate']['source_rows']['discarded_short_segment_rows']}"
                ),
            ]
        )
    else:
        lines.append(f"- 状态：`{algorithm.get('status')}`")
        for error in algorithm.get("source_errors", []):
            lines.append(f"- {error['video_id']}：{error['error']}")
    if report.get("live_runtime") is not None:
        live = report["live_runtime"]
        lines.extend(
            [
                "",
                "## 真实异步运行能力",
                "",
                f"- 严格配对 run_id：`{live['run_id']}`",
                f"- manifest：`{live['manifest_path']}`",
                f"- worker 结果覆盖：{live['worker_result_coverage_all_frames']:.6f}",
                f"- 全帧 predicted 覆盖：{live['predicted_coverage_all_frames']:.6f}",
                f"- 有效控制帧 predicted 覆盖：{live['predicted_coverage_valid_frames']}",
                (
                    "- Python source.read 返回后至 UDP send attempt："
                    f"P50={live['python_post_capture_to_udp_ms']['p50']} ms，"
                    f"P95={live['python_post_capture_to_udp_ms']['p95']} ms"
                ),
            ]
        )
        unity_timing = live.get("unity_timing")
        if isinstance(unity_timing, dict):
            unity_summary = unity_timing["summary"]
            metrics = unity_summary["metrics"]
            source_to_target = metrics["source_read_end_to_target_apply_ms"]
            counters = unity_summary["counters"]
            safety = unity_summary["safety"]
            lines.extend(
                [
                    f"- Unity timing：`{unity_timing['path']}`",
                    (
                        "- source.read 返回后至 Unity 主线程应用："
                        f"P50={source_to_target['p50_ms']} ms，"
                        f"P95={source_to_target['p95_ms']} ms，"
                        f"max={source_to_target['max_ms']} ms"
                    ),
                    (
                        "- Unity 数据包/安全计数："
                        f"accepted={unity_summary['accepted_packet_count']}，"
                        f"overwritten={counters['overwritten_packet_count']}，"
                        f"frame_gap={counters['frame_gap_count']}，"
                        f"rejected={counters['rejected_packet_count']}，"
                        f"stale={counters['stale_packet_count']}，"
                        f"watchdog_open={counters['watchdog_open_count']}"
                    ),
                    (
                        "- Unity 硬件边界："
                        "hardware_forwarding_compiled="
                        f"{safety['baseline_udp_hardware_forwarding_compiled']}，"
                        "apply_preview_to_hardware="
                        f"{safety['apply_baseline_preview_to_hardware']}，"
                        f"real_svh_in_scope={safety['real_svh_in_scope']}"
                    ),
                ]
            )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- 固定视频指标使用视觉映射的未来 `svh_preview` 作为伪真值，不是实体手关节或真实意图真值。",
            "- 离线吞吐、源视频 FPS、真实异步 worker 覆盖是三个不同指标。",
            "- PTS 中位间隔 FPS 与首末时间戳时长均值 FPS 分列报告；前者不是完整视频时长均值。",
            "- timing 起点是 `source.read()` 返回后，不包含相机曝光、此前等待、显示刷新或人体/机械响应。",
            "- development 阶段不产生 release/上线结论；盲测前必须另行冻结门槛和双 SHA。",
            "- 本工具不创建 UDP exporter，预测结果仍不进入 Unity payload。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_blind_decision(
    videos: list[dict[str, Any]],
    algorithm: dict[str, Any],
    live_runtime: dict[str, Any] | None,
    *,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """按冻结 gate 生成三分支结论；gate 为空时绝不能调用。"""

    required_fields = (
        "minimum_each_control_ready_fraction",
        "maximum_each_timestamp_fallback_fraction",
        "minimum_primary_gated_rmse_improvement_percent",
        "minimum_primary_dynamic_gated_rmse_improvement_percent",
        "minimum_primary_conditional_prediction_available_fraction",
        "minimum_primary_end_to_end_prediction_coverage_fraction",
        "maximum_primary_range_violation_rate",
    )
    for field in required_fields:
        if not _finite_number(gate.get(field)):
            raise ValueError(f"blind gate 缺少有限数值字段：{field}")
    input_criteria: dict[str, bool] = {}
    for video in videos:
        video_id = str(video["video_id"])
        frame_count = int(video["metadata"]["decoded_frame_count"])
        source_counts = dict(video["source_timeline"]["timestamp_source_counts"])
        fallback_count = sum(
            int(value) for key, value in source_counts.items() if key != "container_pts_ms"
        )
        fallback_fraction = float(fallback_count / frame_count) if frame_count else 1.0
        input_criteria[f"{video_id}_control_ready"] = (
            float(video["observation_counts"]["control_ready_fraction"])
            >= float(gate["minimum_each_control_ready_fraction"])
        )
        input_criteria[f"{video_id}_timestamp_fallback"] = (
            fallback_fraction <= float(gate["maximum_each_timestamp_fallback_fraction"])
        )
    input_criteria["algorithm_trace_evaluable"] = algorithm.get("status") == "evaluated"
    input_criteria["all_video_sources_evaluable"] = not bool(algorithm.get("source_errors"))
    algorithm_criteria: dict[str, bool] = {}
    if algorithm.get("status") == "evaluated":
        primary = algorithm["aggregate"]["primary_delay_summary"]
        dynamic = primary.get("dynamic_q90")
        gated_improvement = primary["improvement_percent_vs_hold"]["gated_rmse"]
        dynamic_improvement = (
            dynamic["improvement_percent_vs_hold"]["gated_rmse"] if dynamic else None
        )
        algorithm_criteria = {
            "overall_gated_rmse": (
                gated_improvement is not None
                and float(gated_improvement)
                >= float(gate["minimum_primary_gated_rmse_improvement_percent"])
            ),
            "dynamic_gated_rmse": (
                dynamic_improvement is not None
                and float(dynamic_improvement)
                >= float(gate["minimum_primary_dynamic_gated_rmse_improvement_percent"])
            ),
            "conditional_prediction_coverage": (
                float(primary["conditional_prediction_available_fraction"])
                >= float(gate["minimum_primary_conditional_prediction_available_fraction"])
            ),
            "end_to_end_prediction_coverage": (
                float(primary["end_to_end_prediction_coverage_fraction"])
                >= float(gate["minimum_primary_end_to_end_prediction_coverage_fraction"])
            ),
            "range_violation": (
                float(primary["methods"]["gated"]["range_violation_rate"])
                <= float(gate["maximum_primary_range_violation_rate"])
            ),
        }
    else:
        algorithm_criteria = {"algorithm_evaluable": False}
    runtime_criteria: dict[str, bool] = {}
    require_live = bool(gate.get("require_live_runtime", False))
    if require_live:
        if not _finite_number(gate.get("minimum_live_predicted_coverage_valid_frames")):
            raise ValueError("require_live_runtime=true 时必须冻结 live predicted coverage 门槛")
        runtime_criteria["live_runtime_present"] = live_runtime is not None
        runtime_criteria["live_predicted_coverage_valid_frames"] = bool(
            live_runtime is not None
            and _finite_number(live_runtime.get("predicted_coverage_valid_frames"))
            and float(live_runtime["predicted_coverage_valid_frames"])
            >= float(gate["minimum_live_predicted_coverage_valid_frames"])
        )
    input_passed = bool(input_criteria) and all(input_criteria.values())
    algorithm_passed = bool(algorithm_criteria) and all(algorithm_criteria.values())
    runtime_passed = all(runtime_criteria.values()) if runtime_criteria else True
    if not input_passed or not runtime_passed:
        branch = "input_or_clock_repair"
        reason = "输入时间轴、检测连续性或真实异步能力未过冻结门槛；暂不评价模型版本。"
    elif not algorithm_passed:
        branch = "v3_pre_registered_candidate"
        reason = "输入健康但 v2 算法效用未过冻结门槛；允许开展一次独立 v3 修正实验。"
    else:
        branch = "keep_v2_shadow"
        reason = "v2 在冻结盲测上通过；继续 default-off shadow，v3 仅作为可选消融。"
    return {
        "status": "blind_gate_passed" if input_passed and algorithm_passed and runtime_passed else "blind_gate_failed",
        "branch": branch,
        "reason": reason,
        "criteria": {
            "input": input_criteria,
            "algorithm": algorithm_criteria,
            "runtime": runtime_criteria,
        },
    }


def run_camera_domain_evaluation(
    *,
    video_specs: list[VideoSpec],
    evaluation_config_path: Path = DEFAULT_CONFIG_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    role: str = "development",
    allow_partial: bool = False,
    runtime_config_override: Path | None = None,
    protocol_override: Path | None = None,
    input_mirrored_override: bool | None = None,
    expected_config_sha256: str | None = None,
    expected_protocol_sha256: str | None = None,
    live_baseline_jsonl: Path | None = None,
    live_prediction_jsonl: Path | None = None,
    live_session_manifest: Path | None = None,
    live_unity_timing_json: Path | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    if role not in {"development", "blind"}:
        raise ValueError("role 只能是 development 或 blind")
    logger = logger or logging.getLogger("camera-domain-eval")
    evaluation_config_path = evaluation_config_path.resolve()
    evaluation_config = load_evaluation_config(evaluation_config_path)
    required_ids = set(str(value) for value in evaluation_config["video_sets"][role])
    supplied_ids = {spec.video_id for spec in video_specs}
    resolved_video_paths = [spec.path.resolve() for spec in video_specs]
    if len(set(resolved_video_paths)) != len(resolved_video_paths):
        raise ValueError("同一视频文件不能用于多个任务 ID")
    missing_ids = sorted(required_ids - supplied_ids)
    unexpected_ids = sorted(supplied_ids - required_ids)
    if unexpected_ids:
        raise ValueError(f"出现协议外视频 ID：{unexpected_ids}")
    if missing_ids and not allow_partial:
        raise ValueError(f"缺少协议视频：{missing_ids}；开发冒烟可显式使用 --allow-partial")
    if role == "blind" and (missing_ids or allow_partial):
        raise ValueError("盲测禁止缺视频或 --allow-partial")
    runtime_config_path = (
        runtime_config_override.resolve()
        if runtime_config_override is not None
        else _resolve_relative(evaluation_config_path, evaluation_config["runtime_config_path"])
    )
    protocol_path = (
        protocol_override.resolve()
        if protocol_override is not None
        else _resolve_relative(evaluation_config_path, evaluation_config["protocol_document_path"])
    )
    if not protocol_path.is_file():
        raise FileNotFoundError(f"协议文档不存在：{protocol_path}")
    protocol = verify_protocol_state(
        role=role,
        evaluation_config=evaluation_config,
        evaluation_config_path=evaluation_config_path,
        protocol_path=protocol_path,
        expected_config_sha256=expected_config_sha256,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    runtime_cfg = load_config(str(runtime_config_path))
    runtime_cfg["unity_udp_enabled"] = False
    runtime_cfg["save_last_json"] = False
    runtime_cfg["save_jsonl"] = False
    runtime_cfg["prediction_shadow_enabled"] = True
    runtime_cfg["enable_control_extension"] = True
    runtime_cfg["svh_enable_preview"] = True
    input_mirrored = (
        bool(input_mirrored_override)
        if input_mirrored_override is not None
        else bool(runtime_cfg.get("input_mirrored", False))
    )
    delay_config_path = _resolve_relative(evaluation_config_path, evaluation_config["delay_config_path"])
    delay_config = _read_json(delay_config_path)
    run_dir = _unique_run_dir(output_root)
    baseline_dir = run_dir / "baseline_jsonl"
    videos: list[dict[str, Any]] = []
    for spec in video_specs:
        logger.info("按媒体时间轴处理 %s：%s", spec.video_id, spec.path)
        videos.append(
            process_video_to_baseline_jsonl(
                spec,
                output_path=baseline_dir / f"{spec.video_id}.jsonl",
                cfg=runtime_cfg,
                timeline_config=dict(evaluation_config["timeline"]),
                input_mirrored=input_mirrored,
                logger=logger,
            )
        )
    algorithm, scenarios, sequence_rows = evaluate_camera_algorithm_utility(
        videos,
        runtime_cfg=runtime_cfg,
        delay_config=delay_config,
        logger=logger,
    )
    live_runtime = None
    if (live_baseline_jsonl is None) != (live_prediction_jsonl is None):
        raise ValueError("--live-baseline-jsonl 与 --live-prediction-jsonl 必须同时提供")
    if live_baseline_jsonl is None and (
        live_session_manifest is not None or live_unity_timing_json is not None
    ):
        raise ValueError("live manifest/Unity timing 必须与两份 live JSONL 一起提供")
    if live_baseline_jsonl is not None and live_prediction_jsonl is not None:
        live_runtime = analyze_live_runtime_logs(
            live_baseline_jsonl,
            live_prediction_jsonl,
            manifest_path=live_session_manifest,
            unity_timing_path=live_unity_timing_json,
        )
    decision: dict[str, Any]
    if role == "development":
        decision = {
            "status": "development_only_no_release_decision",
            "branch": None,
            "reason": "开发集只用于校验时间轴、诊断输入并冻结盲测门槛，禁止据此宣称上线。",
        }
    else:
        decision = evaluate_blind_decision(
            videos,
            algorithm,
            live_runtime,
            gate=dict(evaluation_config["blind_policy"]["gate"]),
        )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "claim_status": "single_right_hand_unity_preview_camera_domain_diagnostic",
        "protocol": protocol,
        "git": _git_snapshot(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "mediapipe": _package_version("mediapipe"),
            "torch": _package_version("torch"),
        },
        "inputs": {
            "runtime_config_path": str(runtime_config_path),
            "runtime_config_sha256": _hash_file(runtime_config_path),
            "delay_config_path": str(delay_config_path),
            "delay_config_sha256": _hash_file(delay_config_path),
            "input_mirrored": input_mirrored,
            "allow_partial": allow_partial,
            "missing_video_ids": missing_ids,
        },
        "videos": videos,
        "algorithm_utility": algorithm,
        "live_runtime": live_runtime,
        "decision": decision,
        "safety": {
            "udp_created": False,
            "unity_payload_modified_by_prediction": False,
            "real_svh_in_scope": False,
        },
    }
    report_path = run_dir / "camera_domain_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(run_dir / "camera_domain_report.md", report)
    _write_video_csv(run_dir / "video_summary.csv", videos, algorithm)
    if scenarios:
        _write_scenario_csv(run_dir / "scenario_metrics.csv", scenarios)
    if sequence_rows:
        _write_sequence_csv(run_dir / "sequence_metrics.csv", sequence_rows)
    return report_path
