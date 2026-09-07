from __future__ import annotations

"""摄像头视频的 AI 评测：媒体时间轴、有效帧统计及预测误差比较。"""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from pipeline import HandPipeline
from perception.mediapipe_hand import MediaPipeHandDetector
from prediction.shadow_predictor import build_prediction_shadow
from utils.config import load_config
from intent_prediction.delay_injection import (
    RuntimeTraceGroup, _concat_scenario_arrays, _summarize_arrays,
    _write_scenario_csv, _write_sequence_csv,
    build_runtime_jsonl_forecast_traces, evaluate_network_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "camera_domain_eval_v1.json"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs" / "camera_domain_eval_v1"
REPORT_SCHEMA_VERSION = "camera-domain-eval-report-v2"

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


def _summarize_readiness_runs(
    readiness: list[bool],
    timestamps_ms: list[float],
    *,
    nominal_fps: float,
) -> dict[str, Any]:
    if len(readiness) != len(timestamps_ms) or not readiness:
        raise ValueError("readiness 与媒体时间戳必须等长且非空")
    period_ms = 1000.0 / float(nominal_fps)
    runs: list[dict[str, Any]] = []
    start = 0
    for end_exclusive in range(1, len(readiness) + 1):
        if end_exclusive < len(readiness) and readiness[end_exclusive] == readiness[start]:
            continue
        end = end_exclusive - 1
        duration_ms = float(timestamps_ms[end] - timestamps_ms[start] + period_ms)
        runs.append(
            {
                "ready": bool(readiness[start]),
                "start_frame_index": start,
                "end_frame_index": end,
                "frame_count": end_exclusive - start,
                "start_media_timestamp_ms": float(timestamps_ms[start]),
                "end_media_timestamp_ms": float(timestamps_ms[end]),
                "duration_ms": max(0.0, duration_ms),
            }
        )
        start = end_exclusive
    valid_runs = [run for run in runs if run["ready"]]
    invalid_runs = [run for run in runs if not run["ready"]]
    return {
        "starts_ready": bool(readiness[0]),
        "ends_ready": bool(readiness[-1]),
        "ready_run_count": len(valid_runs),
        "invalid_run_count": len(invalid_runs),
        "recovery_count": sum(
            1
            for previous, current in zip(runs, runs[1:])
            if previous["ready"] is False and current["ready"] is True
        ),
        "longest_ready_run_ms": max(
            (float(run["duration_ms"]) for run in valid_runs),
            default=0.0,
        ),
        "longest_invalid_run_ms": max(
            (float(run["duration_ms"]) for run in invalid_runs),
            default=0.0,
        ),
        "runs": runs,
    }


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
    readiness: list[bool] = []
    stable_gesture_counts: dict[str, int] = {}
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
                preview_valid = isinstance(preview, dict) and preview.get("valid") is True
                svh_valid_frames += int(preview_valid)
                readiness.append(payload.get("control_ready") is True and preview_valid)
                stable_gesture = str(payload.get("gesture_stable", "unknown"))
                stable_gesture_counts[stable_gesture] = (
                    stable_gesture_counts.get(stable_gesture, 0) + 1
                )
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
    readiness_summary = _summarize_readiness_runs(
        readiness,
        timestamps_ms,
        nominal_fps=nominal_fps,
    )
    return {
        "video_id": spec.video_id,
        "video_path": str(spec.path),
        "video_bytes": spec.path.stat().st_size,
        "baseline_jsonl_path": str(output_path.resolve()),
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
            "stable_gesture_counts": stable_gesture_counts,
        },
        "observation_continuity": readiness_summary,
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
    dynamic_threshold = predictor.dynamic_threshold
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
        "offline_gate_passed": bool(
            predictor.offline_gate_passed
        ),
        "model_config_path": predictor.model_path,
        "checkpoint_path": predictor.checkpoint_path,
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
                "model_config_path": predictor.model_path,
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


def load_evaluation_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("schema_version") != "camera-domain-eval-config-v2":
        raise ValueError("camera-domain 配置版本不受支持")
    timeline = config.get("timeline", {})
    for key in ("synthetic_epoch_seconds", "minimum_nominal_fps", "maximum_nominal_fps"):
        if not _finite_number(timeline.get(key)):
            raise ValueError(f"timeline.{key} 必须是有限数字")
    if not 0 < timeline["minimum_nominal_fps"] < timeline["maximum_nominal_fps"]:
        raise ValueError("媒体帧率范围无效")
    if timeline["synthetic_epoch_seconds"] < 0:
        raise ValueError("媒体时间原点不能为负数")
    return config


def _build_video_payload_processor(cfg: dict, *, input_mirrored: bool, logger):
    detector = MediaPipeHandDetector(
        max_num_hands=int(cfg.get("max_num_hands", 2)),
        min_detection_confidence=float(cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(cfg.get("min_tracking_confidence", 0.5)),
        input_mirrored=input_mirrored,
    )
    try:
        pipeline = HandPipeline(cfg, detector=detector, logger=logger)
    except Exception:
        detector.close()
        raise

    def process(frame, frame_index, timestamp_s, source_fps):
        started = time.perf_counter()
        payload = pipeline.process_frame(frame, frame_index=frame_index,
                                         timestamp=timestamp_s, fps=source_fps)
        return payload, (time.perf_counter() - started) * 1000.0

    return process, detector.close


def _write_markdown(path: Path, report: dict) -> None:
    lines = ["# 摄像头域 AI 评测", "", "按媒体时间重放；报告可重复生成。", "",
             "| 视频 | 帧数 | 源 FPS | 有效控制比例 | 单帧处理 P95/ms |",
             "|---|---:|---:|---:|---:|"]
    for video in report["videos"]:
        count = video["observation_counts"]
        timing = video["offline_processing_capacity"]["processing_ms"]
        lines.append(f"| {video['video_id']} | {count['frames']} | "
                     f"{video['source_timeline']['duration_based_fps']:.2f} | "
                     f"{count['control_ready_fraction']:.2%} | {timing['p95']:.2f} |")
    algorithm = report["algorithm_evaluation"]
    lines.extend(["", f"预测评测状态：`{algorithm['status']}`。"])
    if algorithm.get("aggregate"):
        lines.extend(["", "## 预测误差与覆盖率", "", "```json",
                      json.dumps(algorithm["aggregate"]["primary_delay_summary"], ensure_ascii=False, indent=2),
                      "```"])
    lines.extend(["", "目标为视觉映射生成的 9 通道代理标签；软件延迟注入用于算法比较。",
                  "姿态检测精度须另用带关键点真值的数据评测。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_camera_domain_evaluation(*, video_specs: list[VideoSpec],
                                 evaluation_config_path: Path = DEFAULT_CONFIG_PATH,
                                 output_root: Path = DEFAULT_OUTPUT_ROOT,
                                 evaluate_prediction: bool = True,
                                 input_mirrored: bool | None = None,
                                 logger=None) -> Path:
    """任意命名的视频均可评测；每次写入独立结果目录。"""
    if not video_specs:
        raise ValueError("至少需要一段视频")
    # API 调用同样检查重复输入，避免重复计数。
    video_specs = parse_video_specs(f"{spec.video_id}={spec.path}" for spec in video_specs)
    logger = logger or logging.getLogger("handai.camera-eval")
    evaluation_config_path = evaluation_config_path.resolve()
    config = load_evaluation_config(evaluation_config_path)
    runtime_path = _resolve_relative(evaluation_config_path, config["runtime_config_path"])
    cfg = load_config(str(runtime_path))
    cfg.update(gui_enabled=False, headless=True, unity_udp_enabled=False,
               enable_control_extension=True, svh_enable_preview=True,
               prediction_shadow_enabled=evaluate_prediction)
    mirrored = bool(cfg.get("input_mirrored", False)) if input_mirrored is None else input_mirrored
    cfg["input_mirrored"] = mirrored
    run_dir = _unique_run_dir(output_root)
    videos = []
    for spec in video_specs:
        logger.info("处理视频 %s：%s", spec.video_id, spec.path)
        videos.append(process_video_to_baseline_jsonl(
            spec, output_path=run_dir / f"{spec.video_id}_baseline.jsonl",
            cfg=cfg, timeline_config=config["timeline"], input_mirrored=mirrored, logger=logger,
        ))
    algorithm = {"status": "skipped"}
    if evaluate_prediction:
        delay_config = _read_json(_resolve_relative(evaluation_config_path, config["delay_config_path"]))
        algorithm, scenarios, sequences = evaluate_camera_algorithm_utility(
            videos, runtime_cfg=cfg, delay_config=delay_config, logger=logger,
        )
        _write_scenario_csv(run_dir / "scenario_metrics.csv", scenarios)
        _write_sequence_csv(run_dir / "sequence_metrics.csv", sequences)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION, "created_at_utc": _utc_now(),
        "evaluation_config": config, "runtime_config": cfg,
        "videos": videos, "algorithm_evaluation": algorithm,
        "claim_status": "camera_domain_pseudo_ground_truth_only",
    }
    report_path = run_dir / "camera_domain_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_video_csv(run_dir / "video_metrics.csv", videos, algorithm)
    _write_markdown(run_dir / "camera_domain_report.md", report)
    return report_path
