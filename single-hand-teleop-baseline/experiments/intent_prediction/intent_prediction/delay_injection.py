from __future__ import annotations

"""冻结网络扰动下的单右手 9 通道预测回放评估。"""

import csv
import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .gating import apply_motion_gate
from .metrics import observed_motion_score
from .sequence_data import SVH_CHANNEL_NAMES, WindowSplit, build_window_split, load_manifest
from .training import predict_neural_checkpoint


REPORT_SCHEMA_VERSION = "intent-delay-injection-report-v1"
CONFIG_SCHEMA_VERSION = "intent-delay-injection-config-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique_run_dir(output_root: Path) -> Path:
    run_dir = output_root.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _resolve_from_config(config_path: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def _improvement_percent(reference: float, value: float) -> float | None:
    if reference <= 1e-12:
        return None
    return float(100.0 * (reference - value) / reference)


def _command_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != len(SVH_CHANNEL_NAMES):
        raise ValueError(f"命令真值与预测必须同为 [N, 9]，实际为 {truth.shape} 和 {prediction.shape}")
    if len(truth) == 0:
        raise ValueError("命令指标至少需要一个样本")
    error = prediction - truth
    absolute = np.abs(error)
    violation = np.logical_or(prediction < 0.0, prediction > 1.0)
    return {
        "samples": int(len(truth)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "p95_abs_error": float(np.percentile(absolute, 95)),
        "range_violation_rate": float(np.mean(violation)),
    }


@dataclass(frozen=True)
class SequenceForecastTrace:
    """一个连续序列上每个可发送帧的当前值与冻结预测。"""

    sequence_id: str
    subject: str
    timestamps_ms: np.ndarray
    frame_ids: np.ndarray
    history: np.ndarray
    current: np.ndarray
    raw_forecast: np.ndarray
    gated_forecast: np.ndarray
    prediction_available: np.ndarray
    motion_score: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.timestamps_ms)
        if count < 1:
            raise ValueError("trace 至少需要一帧")
        if self.timestamps_ms.shape != (count,) or not np.all(np.isfinite(self.timestamps_ms)):
            raise ValueError("timestamps_ms 必须是一维有限数值")
        if np.any(np.diff(self.timestamps_ms) <= 0.0):
            raise ValueError("timestamps_ms 必须严格递增")
        if self.frame_ids.shape != (count,):
            raise ValueError("frame_ids 长度不匹配")
        if np.any(np.diff(self.frame_ids) <= 0):
            raise ValueError("frame_ids 必须严格递增")
        if self.history.ndim != 3 or self.history.shape[0] != count or self.history.shape[2] != 9:
            raise ValueError("history 必须为 [N, T, 9]")
        if self.current.shape != (count, 9):
            raise ValueError("current 必须为 [N, 9]")
        if self.raw_forecast.ndim != 3 or self.raw_forecast.shape[0] != count or self.raw_forecast.shape[2] != 9:
            raise ValueError("raw_forecast 必须为 [N, H, 9]")
        if self.gated_forecast.shape != self.raw_forecast.shape:
            raise ValueError("gated_forecast 与 raw_forecast 形状必须一致")
        if self.prediction_available.shape != (count,):
            raise ValueError("prediction_available 长度不匹配")
        if self.motion_score.shape != (count,):
            raise ValueError("motion_score 长度不匹配")
        for name, values in (
            ("history", self.history),
            ("current", self.current),
            ("raw_forecast", self.raw_forecast),
            ("gated_forecast", self.gated_forecast),
            ("motion_score", self.motion_score),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} 包含非有限值")


@dataclass(frozen=True)
class ScenarioArrays:
    truth: np.ndarray
    hold_last: np.ndarray
    raw: np.ndarray
    gated: np.ndarray
    dynamic_mask: np.ndarray
    prediction_available: np.ndarray
    ages_ms: np.ndarray
    horizon_clamped: np.ndarray
    sequence_ids: np.ndarray
    receiver_ticks: int


@dataclass(frozen=True)
class RuntimeTraceGroup:
    """一个真实 JSONL 输入及其分段后的可回放预测 trace。"""

    source_path: Path
    total_rows: int
    valid_rows: int
    evaluable_rows: int
    discarded_short_segment_rows: int
    discarded_short_segment_count: int
    predicted_rows: int
    status_counts: dict[str, int]
    traces: list[SequenceForecastTrace]


def build_runtime_jsonl_forecast_traces(
    jsonl_path: Path,
    *,
    predictor: Any,
    recent_frames: int,
) -> RuntimeTraceGroup:
    """用现役 PredictionShadow 顺序重放真实 JSONL，并按 invalid/gap 分段。"""

    path = jsonl_path.resolve()
    history_frames = int(predictor.history_frames)
    horizon_ms = tuple(int(value) for value in predictor.horizon_ms)
    max_gap_ms = float(predictor.max_frame_gap_ms)
    traces: list[SequenceForecastTrace] = []
    total_rows = 0
    valid_rows = 0
    predicted_rows = 0
    status_counts: dict[str, int] = {}
    segment_index = 0
    evaluable_rows = 0
    discarded_short_segment_rows = 0
    discarded_short_segment_count = 0

    segment_timestamps: list[float] = []
    segment_frame_ids: list[int] = []
    segment_history: list[np.ndarray] = []
    segment_current: list[np.ndarray] = []
    segment_raw: list[np.ndarray] = []
    segment_gated: list[np.ndarray] = []
    segment_available: list[bool] = []
    segment_motion: list[float] = []

    def flush_segment() -> None:
        nonlocal segment_index
        nonlocal evaluable_rows
        nonlocal discarded_short_segment_rows
        nonlocal discarded_short_segment_count
        if len(segment_timestamps) >= 2:
            sequence_id = f"{path.stem}_segment{segment_index:03d}"
            traces.append(
                SequenceForecastTrace(
                    sequence_id=sequence_id,
                    subject="runtime_camera",
                    timestamps_ms=np.asarray(segment_timestamps, dtype=np.float64),
                    frame_ids=np.asarray(segment_frame_ids, dtype=np.int64),
                    history=np.stack(segment_history).astype(np.float32),
                    current=np.stack(segment_current).astype(np.float32),
                    raw_forecast=np.stack(segment_raw).astype(np.float32),
                    gated_forecast=np.stack(segment_gated).astype(np.float32),
                    prediction_available=np.asarray(segment_available, dtype=bool),
                    motion_score=np.asarray(segment_motion, dtype=np.float64),
                )
            )
            evaluable_rows += len(segment_timestamps)
            segment_index += 1
        elif segment_timestamps:
            discarded_short_segment_rows += len(segment_timestamps)
            discarded_short_segment_count += 1
        segment_timestamps.clear()
        segment_frame_ids.clear()
        segment_history.clear()
        segment_current.clear()
        segment_raw.clear()
        segment_gated.clear()
        segment_available.clear()
        segment_motion.clear()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            diagnostic = predictor.observe(payload)
            status = str(diagnostic.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            preview = payload.get("svh_preview")
            positions = preview.get("target_positions") if isinstance(preview, dict) else None
            timestamp = payload.get("timestamp")
            frame_index = payload.get("frame_index")
            valid = bool(
                payload.get("detected") is True
                and payload.get("control_ready") is True
                and isinstance(preview, dict)
                and preview.get("enabled") is True
                and preview.get("valid") is True
                and isinstance(positions, list)
                and len(positions) == 9
                and isinstance(timestamp, (int, float))
                and np.isfinite(float(timestamp))
                and isinstance(frame_index, int)
            )
            if not valid:
                flush_segment()
                continue
            values = np.asarray(positions, dtype=np.float32)
            if not np.all(np.isfinite(values)):
                flush_segment()
                continue
            timestamp_ms = float(timestamp) * 1000.0
            if segment_timestamps and (
                timestamp_ms <= segment_timestamps[-1]
                or timestamp_ms - segment_timestamps[-1] > max_gap_ms
            ):
                flush_segment()
            valid_rows += 1
            segment_timestamps.append(timestamp_ms)
            segment_frame_ids.append(int(frame_index))
            segment_current.append(values)
            recent_values = segment_current[-history_frames:]
            history = np.stack(recent_values)
            if len(history) < history_frames:
                history = np.concatenate(
                    [np.repeat(history[:1], history_frames - len(history), axis=0), history],
                    axis=0,
                )
            segment_history.append(history.astype(np.float32))

            available = status == "predicted"
            if available:
                raw = np.asarray(diagnostic.get("raw_prediction"), dtype=np.float32)
                gated = np.asarray(diagnostic.get("gated_prediction"), dtype=np.float32)
                expected_shape = (len(horizon_ms), 9)
                if raw.shape != expected_shape or gated.shape != expected_shape:
                    raise ValueError(
                        f"{path}:{line_number} predicted 诊断形状错误：raw={raw.shape}, gated={gated.shape}"
                    )
                predicted_rows += 1
            else:
                raw = np.repeat(values[None, :], len(horizon_ms), axis=0)
                gated = raw.copy()
            segment_raw.append(raw)
            segment_gated.append(gated)
            segment_available.append(available)
            motion = diagnostic.get("motion_score")
            if not isinstance(motion, (int, float)) or not np.isfinite(float(motion)):
                count = min(max(2, int(recent_frames)), len(history))
                motion = float(np.mean(np.abs(np.diff(history[-count:], axis=0))))
            segment_motion.append(float(motion))
    flush_segment()
    if not traces:
        raise RuntimeError(f"真实 JSONL 没有形成至少 2 帧的连续有效片段：{path}")
    if predicted_rows == 0:
        raise RuntimeError(f"真实 JSONL 没有产生 predicted 帧，拒绝伪装成预测回放：{path}")
    return RuntimeTraceGroup(
        source_path=path,
        total_rows=total_rows,
        valid_rows=valid_rows,
        evaluable_rows=evaluable_rows,
        discarded_short_segment_rows=discarded_short_segment_rows,
        discarded_short_segment_count=discarded_short_segment_count,
        predicted_rows=predicted_rows,
        status_counts=status_counts,
        traces=traces,
    )


def build_h2o_forecast_traces(
    split: WindowSplit,
    *,
    raw_prediction: np.ndarray,
    gated_prediction: np.ndarray,
    fps: float,
    subject_by_sequence: dict[str, str],
    recent_frames: int,
) -> list[SequenceForecastTrace]:
    """把随机顺序的窗口恢复为逐序列、逐帧发送 trace。"""

    if fps <= 0.0:
        raise ValueError("fps 必须为正数")
    raw = np.asarray(raw_prediction, dtype=np.float32)
    gated = np.asarray(gated_prediction, dtype=np.float32)
    expected = split.y.shape
    if raw.shape != expected or gated.shape != expected:
        raise ValueError(f"预测形状必须与 y 一致：expected={expected}, raw={raw.shape}, gated={gated.shape}")
    scores = observed_motion_score(split.x, recent_frames=recent_frames)
    traces: list[SequenceForecastTrace] = []
    for sequence_id in sorted(set(split.sequence_ids.tolist())):
        indices = np.flatnonzero(split.sequence_ids == sequence_id)
        order = indices[np.argsort(split.anchor_frame_ids[indices], kind="stable")]
        frame_ids = split.anchor_frame_ids[order].astype(np.int64)
        timestamps_ms = frame_ids.astype(np.float64) * (1000.0 / float(fps))
        traces.append(
            SequenceForecastTrace(
                sequence_id=str(sequence_id),
                subject=str(subject_by_sequence.get(str(sequence_id), "unknown")),
                timestamps_ms=timestamps_ms,
                frame_ids=frame_ids,
                history=np.asarray(split.x[order], dtype=np.float32),
                current=np.asarray(split.x[order, -1, :], dtype=np.float32),
                raw_forecast=raw[order],
                gated_forecast=gated[order],
                prediction_available=np.ones(len(order), dtype=bool),
                motion_score=np.asarray(scores[order], dtype=np.float64),
            )
        )
    return traces


def _scenario_seed(
    seed: int,
    sequence_id: str,
    *,
    delay_ms: float,
    jitter_ms: float,
    loss_rate: float,
) -> int:
    material = f"{seed}|{sequence_id}|{delay_ms:.9f}|{jitter_ms:.9f}|{loss_rate:.9f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little", signed=False)


def _interpolate_forecast(
    current: np.ndarray,
    forecast: np.ndarray,
    *,
    age_ms: float,
    horizon_ms: tuple[int, ...],
) -> tuple[np.ndarray, bool]:
    """在 0ms hold 与冻结预测 horizon 之间线性插值；超上限时夹到最远 horizon。"""

    age = max(0.0, float(age_ms))
    if age <= 0.0:
        return np.asarray(current, dtype=np.float64).copy(), False
    axes = np.asarray((0.0, *[float(value) for value in horizon_ms]), dtype=np.float64)
    values = np.concatenate(
        [np.asarray(current, dtype=np.float64)[None, :], np.asarray(forecast, dtype=np.float64)],
        axis=0,
    )
    clamped = age > axes[-1]
    effective_age = min(age, float(axes[-1]))
    upper = int(np.searchsorted(axes, effective_age, side="right"))
    if upper >= len(axes):
        return values[-1].copy(), clamped
    lower = max(0, upper - 1)
    if lower == upper or axes[upper] <= axes[lower]:
        return values[lower].copy(), clamped
    weight = float((effective_age - axes[lower]) / (axes[upper] - axes[lower]))
    return (1.0 - weight) * values[lower] + weight * values[upper], clamped


def _evaluate_trace(
    trace: SequenceForecastTrace,
    *,
    delay_ms: float,
    jitter_ms: float,
    loss_rate: float,
    horizon_ms: tuple[int, ...],
    seed: int,
    dynamic_threshold: float,
) -> tuple[dict[str, Any], ScenarioArrays | None]:
    if not horizon_ms or tuple(sorted(set(horizon_ms))) != tuple(horizon_ms):
        raise ValueError("horizon_ms 必须为严格递增的唯一正整数")
    if any(value <= 0 for value in horizon_ms):
        raise ValueError("horizon_ms 必须为严格递增的唯一正整数")
    if trace.raw_forecast.shape[1] != len(horizon_ms):
        raise ValueError("forecast 的 horizon 数量与 horizon_ms 不一致")
    rng = np.random.default_rng(
        _scenario_seed(
            seed,
            trace.sequence_id,
            delay_ms=delay_ms,
            jitter_ms=jitter_ms,
            loss_rate=loss_rate,
        )
    )
    count = len(trace.timestamps_ms)
    jitter = rng.uniform(-jitter_ms, jitter_ms, size=count) if jitter_ms > 0.0 else np.zeros(count)
    effective_delay = np.maximum(0.0, delay_ms + jitter)
    lost = rng.random(count) < loss_rate if loss_rate > 0.0 else np.zeros(count, dtype=bool)
    arrival = trace.timestamps_ms + effective_delay
    delivered_indices = np.flatnonzero(~lost)
    event_order = delivered_indices[np.argsort(arrival[delivered_indices], kind="stable")]
    adjacent_reordered = int(np.count_nonzero(np.diff(arrival[delivered_indices]) < 0.0)) if len(delivered_indices) > 1 else 0

    truth_rows: list[np.ndarray] = []
    hold_rows: list[np.ndarray] = []
    raw_rows: list[np.ndarray] = []
    gated_rows: list[np.ndarray] = []
    dynamic_rows: list[bool] = []
    available_rows: list[bool] = []
    age_rows: list[float] = []
    clamped_rows: list[bool] = []

    event_cursor = 0
    latest_source = -1
    for receiver_index, receiver_time in enumerate(trace.timestamps_ms):
        while event_cursor < len(event_order) and arrival[event_order[event_cursor]] <= receiver_time + 1e-9:
            latest_source = max(latest_source, int(event_order[event_cursor]))
            event_cursor += 1
        if latest_source < 0:
            continue
        age = max(0.0, float(receiver_time - trace.timestamps_ms[latest_source]))
        hold = np.asarray(trace.current[latest_source], dtype=np.float64)
        available = bool(trace.prediction_available[latest_source])
        if available:
            raw_command, raw_clamped = _interpolate_forecast(
                hold,
                trace.raw_forecast[latest_source],
                age_ms=age,
                horizon_ms=horizon_ms,
            )
            gated_command, gated_clamped = _interpolate_forecast(
                hold,
                trace.gated_forecast[latest_source],
                age_ms=age,
                horizon_ms=horizon_ms,
            )
            clamped = raw_clamped or gated_clamped
        else:
            raw_command = hold.copy()
            gated_command = hold.copy()
            clamped = False
        truth_rows.append(np.asarray(trace.current[receiver_index], dtype=np.float64))
        hold_rows.append(hold)
        raw_rows.append(raw_command)
        gated_rows.append(gated_command)
        dynamic_rows.append(bool(trace.motion_score[latest_source] >= dynamic_threshold))
        available_rows.append(available)
        age_rows.append(age)
        clamped_rows.append(clamped)

    if not truth_rows:
        return {
            "sequence_id": trace.sequence_id,
            "subject": trace.subject,
            "source_packets": count,
            "lost_packets": int(np.count_nonzero(lost)),
            "adjacent_reordered_arrivals": adjacent_reordered,
            "receiver_ticks": count,
            "evaluated_ticks": 0,
            "no_packet_ticks": count,
            "prediction_available_ticks": 0,
            "receiver_coverage_fraction": 0.0,
            "prediction_available_fraction": None,
            "conditional_prediction_available_fraction": None,
            "end_to_end_prediction_coverage_fraction": 0.0,
            "horizon_clamped_fraction": None,
            "age_ms": None,
            "methods": None,
            "improvement_percent_vs_hold": None,
        }, None
    arrays = ScenarioArrays(
        truth=np.stack(truth_rows),
        hold_last=np.stack(hold_rows),
        raw=np.stack(raw_rows),
        gated=np.stack(gated_rows),
        dynamic_mask=np.asarray(dynamic_rows, dtype=bool),
        prediction_available=np.asarray(available_rows, dtype=bool),
        ages_ms=np.asarray(age_rows, dtype=np.float64),
        horizon_clamped=np.asarray(clamped_rows, dtype=bool),
        sequence_ids=np.full(len(truth_rows), trace.sequence_id),
        receiver_ticks=count,
    )
    hold_metrics = _command_metrics(arrays.truth, arrays.hold_last)
    raw_metrics = _command_metrics(arrays.truth, arrays.raw)
    gated_metrics = _command_metrics(arrays.truth, arrays.gated)
    prediction_available_ticks = int(np.count_nonzero(arrays.prediction_available))
    evaluated_ticks = int(len(arrays.truth))
    conditional_prediction_fraction = float(np.mean(arrays.prediction_available))
    summary = {
        "sequence_id": trace.sequence_id,
        "subject": trace.subject,
        "source_packets": count,
        "lost_packets": int(np.count_nonzero(lost)),
        "adjacent_reordered_arrivals": adjacent_reordered,
        "receiver_ticks": count,
        "evaluated_ticks": evaluated_ticks,
        "no_packet_ticks": int(count - evaluated_ticks),
        "prediction_available_ticks": prediction_available_ticks,
        "receiver_coverage_fraction": float(evaluated_ticks / count),
        # 旧字段保留 v1 口径：只在“至少已有一个包”的 tick 上计算。
        "prediction_available_fraction": conditional_prediction_fraction,
        "conditional_prediction_available_fraction": conditional_prediction_fraction,
        "end_to_end_prediction_coverage_fraction": float(prediction_available_ticks / count),
        "horizon_clamped_fraction": float(np.mean(arrays.horizon_clamped)),
        "age_ms": {
            "p50": float(np.percentile(arrays.ages_ms, 50)),
            "p95": float(np.percentile(arrays.ages_ms, 95)),
            "max": float(np.max(arrays.ages_ms)),
        },
        "methods": {"hold_last": hold_metrics, "raw": raw_metrics, "gated": gated_metrics},
        "improvement_percent_vs_hold": {
            "raw_rmse": _improvement_percent(hold_metrics["rmse"], raw_metrics["rmse"]),
            "gated_rmse": _improvement_percent(hold_metrics["rmse"], gated_metrics["rmse"]),
            "gated_p95": _improvement_percent(hold_metrics["p95_abs_error"], gated_metrics["p95_abs_error"]),
        },
    }
    return summary, arrays


def _concat_scenario_arrays(parts: Iterable[ScenarioArrays]) -> ScenarioArrays:
    values = list(parts)
    if not values:
        raise ValueError("至少需要一个 ScenarioArrays")
    return ScenarioArrays(
        truth=np.concatenate([item.truth for item in values]),
        hold_last=np.concatenate([item.hold_last for item in values]),
        raw=np.concatenate([item.raw for item in values]),
        gated=np.concatenate([item.gated for item in values]),
        dynamic_mask=np.concatenate([item.dynamic_mask for item in values]),
        prediction_available=np.concatenate([item.prediction_available for item in values]),
        ages_ms=np.concatenate([item.ages_ms for item in values]),
        horizon_clamped=np.concatenate([item.horizon_clamped for item in values]),
        sequence_ids=np.concatenate([item.sequence_ids for item in values]),
        receiver_ticks=sum(item.receiver_ticks for item in values),
    )


def _summarize_arrays(arrays: ScenarioArrays) -> dict[str, Any]:
    hold = _command_metrics(arrays.truth, arrays.hold_last)
    raw = _command_metrics(arrays.truth, arrays.raw)
    gated = _command_metrics(arrays.truth, arrays.gated)
    dynamic = None
    if np.any(arrays.dynamic_mask):
        dynamic_hold = _command_metrics(arrays.truth[arrays.dynamic_mask], arrays.hold_last[arrays.dynamic_mask])
        dynamic_raw = _command_metrics(arrays.truth[arrays.dynamic_mask], arrays.raw[arrays.dynamic_mask])
        dynamic_gated = _command_metrics(arrays.truth[arrays.dynamic_mask], arrays.gated[arrays.dynamic_mask])
        dynamic = {
            "samples": int(np.count_nonzero(arrays.dynamic_mask)),
            "methods": {"hold_last": dynamic_hold, "raw": dynamic_raw, "gated": dynamic_gated},
            "improvement_percent_vs_hold": {
                "raw_rmse": _improvement_percent(dynamic_hold["rmse"], dynamic_raw["rmse"]),
                "gated_rmse": _improvement_percent(dynamic_hold["rmse"], dynamic_gated["rmse"]),
                "gated_p95": _improvement_percent(
                    dynamic_hold["p95_abs_error"], dynamic_gated["p95_abs_error"]
                ),
            },
        }
    evaluated_ticks = int(len(arrays.truth))
    receiver_ticks = int(arrays.receiver_ticks)
    prediction_available_ticks = int(np.count_nonzero(arrays.prediction_available))
    conditional_prediction_fraction = float(np.mean(arrays.prediction_available))
    return {
        "samples": evaluated_ticks,
        "receiver_ticks": receiver_ticks,
        "evaluated_ticks": evaluated_ticks,
        "no_packet_ticks": int(receiver_ticks - evaluated_ticks),
        "prediction_available_ticks": prediction_available_ticks,
        "receiver_coverage_fraction": float(evaluated_ticks / receiver_ticks),
        "methods": {"hold_last": hold, "raw": raw, "gated": gated},
        "improvement_percent_vs_hold": {
            "raw_rmse": _improvement_percent(hold["rmse"], raw["rmse"]),
            "gated_rmse": _improvement_percent(hold["rmse"], gated["rmse"]),
            "gated_p95": _improvement_percent(hold["p95_abs_error"], gated["p95_abs_error"]),
        },
        "dynamic_q90": dynamic,
        # 兼容既有 report v1；新字段把条件覆盖与端到端覆盖明确拆开。
        "prediction_available_fraction": conditional_prediction_fraction,
        "conditional_prediction_available_fraction": conditional_prediction_fraction,
        "end_to_end_prediction_coverage_fraction": float(prediction_available_ticks / receiver_ticks),
        "horizon_clamped_fraction": float(np.mean(arrays.horizon_clamped)),
        "age_ms": {
            "p50": float(np.percentile(arrays.ages_ms, 50)),
            "p95": float(np.percentile(arrays.ages_ms, 95)),
            "max": float(np.max(arrays.ages_ms)),
        },
    }


def evaluate_network_matrix(
    traces: list[SequenceForecastTrace],
    *,
    horizon_ms: tuple[int, ...],
    delays_ms: tuple[float, ...],
    jitters_ms: tuple[float, ...],
    loss_rates: tuple[float, ...],
    seed: int,
    dynamic_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[float, float, float], ScenarioArrays]]:
    if not traces:
        raise ValueError("至少需要一个 forecast trace")
    if not horizon_ms or tuple(sorted(set(horizon_ms))) != tuple(horizon_ms) or any(
        value <= 0 for value in horizon_ms
    ):
        raise ValueError("horizon_ms 必须为严格递增的唯一正整数")
    scenario_reports: list[dict[str, Any]] = []
    sequence_reports: list[dict[str, Any]] = []
    arrays_by_scenario: dict[tuple[float, float, float], ScenarioArrays] = {}
    for delay_ms in delays_ms:
        for jitter_ms in jitters_ms:
            for loss_rate in loss_rates:
                if (
                    not np.isfinite(delay_ms)
                    or not np.isfinite(jitter_ms)
                    or not np.isfinite(loss_rate)
                    or delay_ms < 0.0
                    or jitter_ms < 0.0
                    or not 0.0 <= loss_rate < 1.0
                ):
                    raise ValueError("delay/jitter 必须非负，loss_rate 必须位于 [0, 1)")
                parts: list[ScenarioArrays] = []
                for trace in traces:
                    sequence_summary, arrays = _evaluate_trace(
                        trace,
                        delay_ms=delay_ms,
                        jitter_ms=jitter_ms,
                        loss_rate=loss_rate,
                        horizon_ms=horizon_ms,
                        seed=seed,
                        dynamic_threshold=dynamic_threshold,
                    )
                    sequence_reports.append(
                        {
                            "delay_ms": delay_ms,
                            "jitter_ms": jitter_ms,
                            "loss_rate": loss_rate,
                            **sequence_summary,
                        }
                    )
                    if arrays is not None:
                        parts.append(arrays)
                if not parts:
                    raise RuntimeError(
                        f"整个场景没有形成可评估样本：delay={delay_ms}, jitter={jitter_ms}, loss={loss_rate}"
                    )
                scenario_arrays = _concat_scenario_arrays(parts)
                key = (float(delay_ms), float(jitter_ms), float(loss_rate))
                arrays_by_scenario[key] = scenario_arrays
                summary = _summarize_arrays(scenario_arrays)
                scenario_reports.append(
                    {
                        "delay_ms": delay_ms,
                        "jitter_ms": jitter_ms,
                        "loss_rate": loss_rate,
                        "sequence_count": len(traces),
                        "evaluated_sequence_count": len(parts),
                        **summary,
                    }
                )
    return scenario_reports, sequence_reports, arrays_by_scenario


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"配置 schema_version 必须为 {CONFIG_SCHEMA_VERSION}")
    if config.get("protocol_frozen_before_run") is not True:
        raise ValueError("正式扰动实验必须在运行前冻结 protocol_frozen_before_run=true")
    network = config.get("network_matrix")
    if not isinstance(network, dict):
        raise ValueError("network_matrix 必须是对象")
    for key in ("delays_ms", "jitters_ms", "loss_rates"):
        values = network.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"network_matrix.{key} 必须是非空数组")
        try:
            numeric = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"network_matrix.{key} 必须只包含数值") from exc
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError(f"network_matrix.{key} 必须只包含有限数值")
        if key in {"delays_ms", "jitters_ms"} and any(value < 0.0 for value in numeric):
            raise ValueError(f"network_matrix.{key} 不能包含负数")
        if key == "loss_rates" and any(not 0.0 <= value < 1.0 for value in numeric):
            raise ValueError("network_matrix.loss_rates 必须位于 [0, 1)")
    horizons = config.get("horizon_ms")
    if (
        not isinstance(horizons, list)
        or not horizons
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in horizons)
    ):
        raise ValueError("horizon_ms 必须是正整数数组")
    if sorted(horizons) != horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizon_ms 必须严格递增且不能重复")
    gate = config.get("retention_gate")
    if not isinstance(gate, dict):
        raise ValueError("retention_gate 必须是对象")


def _retention_result(
    arrays_by_scenario: dict[tuple[float, float, float], ScenarioArrays],
    scenario_reports: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = dict(config["retention_gate"])
    primary_delays = {float(value) for value in gate["primary_delays_ms"]}
    selected_parts = [arrays for key, arrays in arrays_by_scenario.items() if key[0] in primary_delays]
    if not selected_parts:
        raise ValueError("retention_gate.primary_delays_ms 没有匹配任何场景")
    aggregate_arrays = _concat_scenario_arrays(selected_parts)
    aggregate = _summarize_arrays(aggregate_arrays)
    dynamic = aggregate["dynamic_q90"]
    if dynamic is None:
        raise RuntimeError("primary 场景没有 q90 动态样本")
    scenario_improvements = [
        float(item["improvement_percent_vs_hold"]["gated_rmse"])
        for item in scenario_reports
        if float(item["delay_ms"]) in primary_delays
        and item["improvement_percent_vs_hold"]["gated_rmse"] is not None
    ]
    worst_scenario_regression = max(0.0, -min(scenario_improvements)) if scenario_improvements else float("inf")
    observed = {
        "aggregate_rmse_improvement_percent": aggregate["improvement_percent_vs_hold"]["gated_rmse"],
        "aggregate_p95_improvement_percent": aggregate["improvement_percent_vs_hold"]["gated_p95"],
        "q90_rmse_improvement_percent": dynamic["improvement_percent_vs_hold"]["gated_rmse"],
        "worst_scenario_rmse_regression_percent": worst_scenario_regression,
        "range_violation_rate": aggregate["methods"]["gated"]["range_violation_rate"],
        "prediction_available_fraction": aggregate["prediction_available_fraction"],
        "conditional_prediction_available_fraction": aggregate[
            "conditional_prediction_available_fraction"
        ],
        "receiver_coverage_fraction": aggregate["receiver_coverage_fraction"],
        "end_to_end_prediction_coverage_fraction": aggregate[
            "end_to_end_prediction_coverage_fraction"
        ],
        "horizon_clamped_fraction": aggregate["horizon_clamped_fraction"],
    }
    criteria = {
        "aggregate_rmse_improvement": observed["aggregate_rmse_improvement_percent"]
        >= float(gate["minimum_aggregate_rmse_improvement_percent"]),
        "aggregate_p95_non_regression": observed["aggregate_p95_improvement_percent"]
        >= float(gate["minimum_aggregate_p95_improvement_percent"]),
        "q90_rmse_improvement": observed["q90_rmse_improvement_percent"]
        >= float(gate["minimum_q90_rmse_improvement_percent"]),
        "worst_scenario_bounded": observed["worst_scenario_rmse_regression_percent"]
        <= float(gate["maximum_worst_scenario_rmse_regression_percent"]),
        "range_safety": observed["range_violation_rate"] <= float(gate["maximum_range_violation_rate"]),
        "prediction_coverage": observed["prediction_available_fraction"]
        >= float(gate["minimum_prediction_available_fraction"]),
    }
    passed = bool(all(criteria.values()))
    return {
        "primary_delays_ms": sorted(primary_delays),
        "criteria_config": gate,
        "observed": observed,
        "criteria_pass": criteria,
        "prediction_coverage_metric": (
            "prediction_available_fraction conditional on a receiver tick having at least one arrived packet; "
            "kept unchanged because the v1 retention protocol was frozen before the formal run"
        ),
        "retention_gate_passed": passed,
        "decision": (
            "continue_shadow_research_only"
            if passed
            else "retain_hold_last_as_control_reference_and_keep_model_shadow_only"
        ),
        "scope": "只决定是否值得继续影子研究；无论结果如何都不授权模型进入 Unity UDP 或真机。",
        "aggregate_primary_metrics": aggregate,
    }


def _write_scenario_csv(path: Path, scenarios: list[dict[str, Any]]) -> None:
    fields = [
        "delay_ms",
        "jitter_ms",
        "loss_rate",
        "samples",
        "receiver_ticks",
        "evaluated_ticks",
        "no_packet_ticks",
        "receiver_coverage_fraction",
        "prediction_available_fraction",
        "conditional_prediction_available_fraction",
        "end_to_end_prediction_coverage_fraction",
        "horizon_clamped_fraction",
        "hold_rmse",
        "raw_rmse",
        "gated_rmse",
        "raw_rmse_improvement_percent",
        "gated_rmse_improvement_percent",
        "gated_p95_improvement_percent",
        "q90_gated_rmse_improvement_percent",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in scenarios:
            dynamic = item.get("dynamic_q90")
            writer.writerow(
                {
                    "delay_ms": item["delay_ms"],
                    "jitter_ms": item["jitter_ms"],
                    "loss_rate": item["loss_rate"],
                    "samples": item["samples"],
                    "receiver_ticks": item["receiver_ticks"],
                    "evaluated_ticks": item["evaluated_ticks"],
                    "no_packet_ticks": item["no_packet_ticks"],
                    "receiver_coverage_fraction": item["receiver_coverage_fraction"],
                    "prediction_available_fraction": item["prediction_available_fraction"],
                    "conditional_prediction_available_fraction": item[
                        "conditional_prediction_available_fraction"
                    ],
                    "end_to_end_prediction_coverage_fraction": item[
                        "end_to_end_prediction_coverage_fraction"
                    ],
                    "horizon_clamped_fraction": item["horizon_clamped_fraction"],
                    "hold_rmse": item["methods"]["hold_last"]["rmse"],
                    "raw_rmse": item["methods"]["raw"]["rmse"],
                    "gated_rmse": item["methods"]["gated"]["rmse"],
                    "raw_rmse_improvement_percent": item["improvement_percent_vs_hold"]["raw_rmse"],
                    "gated_rmse_improvement_percent": item["improvement_percent_vs_hold"]["gated_rmse"],
                    "gated_p95_improvement_percent": item["improvement_percent_vs_hold"]["gated_p95"],
                    "q90_gated_rmse_improvement_percent": (
                        dynamic["improvement_percent_vs_hold"]["gated_rmse"] if dynamic else None
                    ),
                }
            )


def _write_sequence_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "delay_ms",
        "jitter_ms",
        "loss_rate",
        "sequence_id",
        "subject",
        "source_packets",
        "lost_packets",
        "receiver_ticks",
        "evaluated_ticks",
        "no_packet_ticks",
        "prediction_available_ticks",
        "receiver_coverage_fraction",
        "prediction_available_fraction",
        "conditional_prediction_available_fraction",
        "end_to_end_prediction_coverage_fraction",
        "horizon_clamped_fraction",
        "hold_rmse",
        "raw_rmse",
        "gated_rmse",
        "gated_rmse_improvement_percent",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            methods = item.get("methods")
            improvements = item.get("improvement_percent_vs_hold")
            writer.writerow(
                {
                    "delay_ms": item["delay_ms"],
                    "jitter_ms": item["jitter_ms"],
                    "loss_rate": item["loss_rate"],
                    "sequence_id": item["sequence_id"],
                    "subject": item["subject"],
                    "source_packets": item["source_packets"],
                    "lost_packets": item["lost_packets"],
                    "receiver_ticks": item["receiver_ticks"],
                    "evaluated_ticks": item["evaluated_ticks"],
                    "no_packet_ticks": item["no_packet_ticks"],
                    "prediction_available_ticks": item["prediction_available_ticks"],
                    "receiver_coverage_fraction": item["receiver_coverage_fraction"],
                    "prediction_available_fraction": item["prediction_available_fraction"],
                    "conditional_prediction_available_fraction": item[
                        "conditional_prediction_available_fraction"
                    ],
                    "end_to_end_prediction_coverage_fraction": item[
                        "end_to_end_prediction_coverage_fraction"
                    ],
                    "horizon_clamped_fraction": item["horizon_clamped_fraction"],
                    "hold_rmse": methods["hold_last"]["rmse"] if methods else None,
                    "raw_rmse": methods["raw"]["rmse"] if methods else None,
                    "gated_rmse": methods["gated"]["rmse"] if methods else None,
                    "gated_rmse_improvement_percent": improvements["gated_rmse"] if improvements else None,
                }
            )


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    retention = report["retention"]
    lines = [
        "# 单右手意图预测：延迟/抖动/丢包冻结回放",
        "",
        f"- 状态：`{report['status']}`",
        f"- 场景数：{len(report['scenarios'])}",
        f"- 选中模型：`{report['model']['selected_label']}`",
        f"- retention gate：`{str(retention['retention_gate_passed']).lower()}`",
        f"- 决策：`{retention['decision']}`",
        "",
        "## 预注册 primary 汇总",
        "",
        "| 指标 | 观测值 |",
        "|---|---:|",
    ]
    for key, value in retention["observed"].items():
        lines.append(f"| `{key}` | {value:.6f} |")
    runtime = report.get("runtime_jsonl_replay")
    if runtime is not None:
        runtime_summary = runtime["primary_delay_summary"]
        lines.extend(
            [
                "",
                "## 真实摄像头 JSONL 探索性回放",
                "",
                f"- 总体 gated RMSE 改善：{runtime_summary['improvement_percent_vs_hold']['gated_rmse']:.6f}%",
                f"- q90 gated RMSE 改善：{runtime_summary['dynamic_q90']['improvement_percent_vs_hold']['gated_rmse']:.6f}%",
                "- prediction 条件覆盖率（已有包的 tick）："
                f"{runtime_summary['conditional_prediction_available_fraction']:.6f}",
                "- prediction 端到端覆盖率（全部接收 tick）："
                f"{runtime_summary['end_to_end_prediction_coverage_fraction']:.6f}",
                f"- 超 150 ms horizon 比例：{runtime_summary['horizon_clamped_fraction']:.6f}",
                "- 该结果以之后的视觉映射输出作伪真值，不参与 H2O retention gate。",
            ]
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "该实验只在公开 H2O 代理控制序列上模拟本机接收端可见的旧帧，",
            "不等于真实 UDP、Unity 帧调度、头显网络或实体灵巧手已经验收。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_delay_injection(
    *,
    config_path: Path,
    data_root: Path,
    output_root: Path,
    runtime_trace_groups: list[RuntimeTraceGroup] | None = None,
) -> Path:
    """按预先冻结的网络矩阵评估 H2O test，不重新选模型或拟合 gate。"""

    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    data_root = data_root.resolve()
    manifest = load_manifest(data_root)
    selection_path = _resolve_from_config(config_path, config["selection_path"])
    second_round_report_path = _resolve_from_config(config_path, config["second_round_report_path"])
    checkpoint_path = _resolve_from_config(config_path, config["checkpoint_path"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    second_round_report = json.loads(second_round_report_path.read_text(encoding="utf-8"))
    selected_label = str(selection.get("selected_label"))
    if selected_label == "hold_last":
        raise ValueError("selection 选择了 hold_last，没有模型可做扰动诊断")
    data_contract = dict(selection.get("data_contract") or {})
    if data_contract.get("mapping_contract_version") != manifest.get("mapping_contract_version"):
        raise ValueError("selection 与 H2O manifest 的 mapping contract 版本不一致")
    for contract_key in (
        "mapping_contract",
        "h2o_label_gesture_context_policy",
        "runtime_gesture_context_policy",
        "projection_policy",
        "joint_order",
    ):
        expected = data_contract.get(contract_key)
        if expected is not None and expected != manifest.get(contract_key):
            raise ValueError(f"selection 与 H2O manifest 的 {contract_key} 不一致")

    history_frames = int(config["history_frames"])
    horizon_ms = tuple(int(value) for value in config["horizon_ms"])
    split = build_window_split(
        data_root,
        split=str(config.get("split", "test")),
        history_frames=history_frames,
        horizon_ms=horizon_ms,
        stride=int(config.get("stride", 1)),
        max_windows=config.get("max_windows"),
        seed=int(config["seed"]),
    )
    raw_prediction, inference = predict_neural_checkpoint(
        checkpoint_path,
        split=split,
        batch_size=int(config.get("batch_size", 256)),
        device=str(config.get("device", "auto")),
    )
    gate = dict(selection["gate_parameters"])
    gated_prediction, gate_summary = apply_motion_gate(
        split.x,
        raw_prediction,
        threshold=float(gate["threshold"]),
        temperature=float(gate["temperature"]),
        alpha_by_horizon=list(gate["alpha_by_horizon"]),
        recent_frames=int(gate["recent_frames"]),
    )
    subject_by_sequence = {
        str(entry["sequence_id"]): str(entry.get("subject", "unknown"))
        for entry in manifest.get("sequences", [])
    }
    traces = build_h2o_forecast_traces(
        split,
        raw_prediction=raw_prediction,
        gated_prediction=gated_prediction,
        fps=float(manifest["fps"]),
        subject_by_sequence=subject_by_sequence,
        recent_frames=int(gate["recent_frames"]),
    )
    network = dict(config["network_matrix"])
    dynamic_threshold = float(second_round_report["motion_strata_thresholds_from_validation"]["q90"])
    scenarios, sequence_rows, arrays_by_scenario = evaluate_network_matrix(
        traces,
        horizon_ms=horizon_ms,
        delays_ms=tuple(float(value) for value in network["delays_ms"]),
        jitters_ms=tuple(float(value) for value in network["jitters_ms"]),
        loss_rates=tuple(float(value) for value in network["loss_rates"]),
        seed=int(config["seed"]),
        dynamic_threshold=dynamic_threshold,
    )
    retention = _retention_result(arrays_by_scenario, scenarios, config=config)
    runtime_report = None
    runtime_scenarios: list[dict[str, Any]] = []
    runtime_sequence_rows: list[dict[str, Any]] = []
    if runtime_trace_groups:
        runtime_traces = [trace for group in runtime_trace_groups for trace in group.traces]
        runtime_scenarios, runtime_sequence_rows, runtime_arrays = evaluate_network_matrix(
            runtime_traces,
            horizon_ms=horizon_ms,
            delays_ms=tuple(float(value) for value in network["delays_ms"]),
            jitters_ms=tuple(float(value) for value in network["jitters_ms"]),
            loss_rates=tuple(float(value) for value in network["loss_rates"]),
            seed=int(config["seed"]),
            dynamic_threshold=dynamic_threshold,
        )
        primary_delays = {float(value) for value in config["retention_gate"]["primary_delays_ms"]}
        runtime_primary = _concat_scenario_arrays(
            arrays for key, arrays in runtime_arrays.items() if key[0] in primary_delays
        )
        runtime_report = {
            "claim_status": "exploratory_camera_domain_pseudo_ground_truth",
            "meaning": (
                "目标是后续视觉映射输出，不是实体手关节真值；只用于检查趋势是否跨域，"
                "不参与 H2O retention gate。"
            ),
            "sources": [
                {
                    "path": str(group.source_path),
                    "total_rows": group.total_rows,
                    "valid_rows": group.valid_rows,
                    "evaluable_rows": group.evaluable_rows,
                    "discarded_short_segment_rows": group.discarded_short_segment_rows,
                    "discarded_short_segment_count": group.discarded_short_segment_count,
                    "predicted_rows": group.predicted_rows,
                    "status_counts": group.status_counts,
                    "segment_count": len(group.traces),
                }
                for group in runtime_trace_groups
            ],
            "primary_delay_summary": _summarize_arrays(runtime_primary),
            "scenarios": runtime_scenarios,
        }
    run_dir = _unique_run_dir(output_root)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "completed",
        "created_at_utc": _utc_now(),
        "run_purpose": str(config.get("run_purpose", "unspecified")),
        "source_config_path": str(config_path),
        "effective_config": config,
        "protocol": {
            "frozen_before_run": True,
            "test_based_reselection_performed": False,
            "gate_refit_performed": False,
            "network_jitter_distribution": "deterministic_uniform_symmetric_then_nonnegative_clip",
            "packet_receiver_policy": "latest_source_frame_among_packets_arrived_by_receiver_tick",
            "prediction_policy": "linear_interpolation_from_hold_at_0ms_to_frozen_50_100_150ms_forecasts",
            "age_above_max_horizon_policy": "clamp_to_farthest_forecast_and_report_fraction",
            "prediction_coverage_policy": (
                "report conditional_on_arrived_packet and end_to_end_over_all_receiver_ticks separately; "
                "formal_v1_gate_keeps_the_preregistered_conditional_metric"
            ),
        },
        "dataset": {
            "root": str(data_root),
            "name": manifest.get("dataset"),
            "split": str(config.get("split", "test")),
            "window_count": int(len(split.x)),
            "sequence_count": int(len(traces)),
            "fps": float(manifest["fps"]),
            "data_contract": data_contract,
        },
        "model": {
            "selected_label": selected_label,
            "selection_path": str(selection_path),
            "checkpoint_path": str(checkpoint_path),
            "horizon_ms": list(horizon_ms),
            "inference": inference,
            "gate_parameters": gate,
            "gate_summary_on_test_windows": gate_summary,
            "prior_offline_gate_passed": bool(second_round_report["acceptance"]["offline_gate_passed"]),
        },
        "network_matrix": network,
        "dynamic_q90_threshold_from_validation": dynamic_threshold,
        "retention": retention,
        "scenarios": scenarios,
        "runtime_jsonl_replay": runtime_report,
        "scope_boundary": {
            "single_right_hand": True,
            "public_pose_proxy_only": True,
            "camera_required": False,
            "unity_modified": False,
            "real_udp_measured": False,
            "real_hardware_authorized": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_scenario_csv(run_dir / "scenario_metrics.csv", scenarios)
    _write_sequence_csv(run_dir / "sequence_metrics.csv", sequence_rows)
    if runtime_report is not None:
        _write_scenario_csv(run_dir / "runtime_scenario_metrics.csv", runtime_scenarios)
        _write_sequence_csv(run_dir / "runtime_sequence_metrics.csv", runtime_sequence_rows)
    _write_markdown(run_dir / "report.md", report)
    return report_path
