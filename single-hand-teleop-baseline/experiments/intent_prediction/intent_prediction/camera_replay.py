"""M3-B 有界队列调度与共同参考评分；不把条件有效误差冒充全程覆盖。"""
from __future__ import annotations

from collections import Counter
import numpy as np

from pipeline import HandPipeline
from prediction.camera_baselines import HORIZONS_MS, observe_frame, predict_camera, reference_camera
from perception.geometry import GEOMETRY_TASK
from svh.mapping_contract import mapping_contract_payload
from svh.svh_layout import SVH_9CH_NAMES


def latest_schedule(arrivals: list, costs: list) -> tuple[list, list]:
    """单服务者 + 一个 latest-only 等待槽；完成事件先于同刻到达事件。"""
    if len(arrivals) != len(costs) or any(not np.isfinite(c) or c < 0 for c in costs):
        raise ValueError("调度成本必须有限非负并与输入等长")
    if any(a is not None and (not np.isfinite(a) or a < 0) for a in arrivals):
        raise ValueError("到达时间必须有限非负或缺失")
    scheduled, reasons = [None] * len(arrivals), ["source_drop" if a is None else None for a in arrivals]
    busy_until, pending = 0.0, None

    def start(index, at):
        scheduled[index] = {"start_ms": float(at), "end_ms": float(at + costs[index]),
                            "queue_ms": float(at - arrivals[index])}
        return at + costs[index]

    for at, index in sorted((a, i) for i, a in enumerate(arrivals) if a is not None):
        if pending is not None and busy_until <= at:
            busy_until = start(pending, max(busy_until, arrivals[pending]))
            pending = None
        if busy_until <= at:
            busy_until = start(index, at)
        else:
            if pending is not None:
                reasons[pending] = "queue_replaced"
            pending = index
    if pending is not None:
        start(pending, max(busy_until, arrivals[pending]))
    return scheduled, reasons


def common_references(frames: list, cfg: dict, fps: float) -> list:
    pipeline = HandPipeline(cfg)
    result, previous = [], None
    for index, frame in enumerate(frames):
        observe_frame(pipeline, frame, previous)
        result.append({**reference_camera(frames, index, pipeline, fps=fps), "frame_id": frame["frame_id"]})
        previous = frame
    return result


def run_scenario(frames: list, cfg: dict, *, methods: list[tuple[str, int]], fps: float,
                 scenario: dict, algorithm_only=False) -> dict[str, list]:
    """先共同感知，再按方法各自成本调度预测；锚点快照只含该源帧及过去。"""
    n = len(frames)
    if algorithm_only:
        arrivals = [f["source_time_ms"] for f in frames]
        baseline = [{"start_ms": t, "end_ms": t, "queue_ms": 0.0} for t in arrivals]
        base_reasons = [None] * n
    else:
        delays, every = scenario["arrival_delay_cycle_ms"], scenario["drop_every"]
        arrivals, previous_arrival = [], 0.0
        for i, frame in enumerate(frames):
            # FIFO 到达；较晚一帧不穿过前一帧，排队造成的附加延迟可直接读取。
            at = max(previous_arrival, frame["source_time_ms"] + delays[i % len(delays)])
            previous_arrival = at
            arrivals.append(None if every and (i + 1) % every == 0 else at)
        costs = [f["detection_ms"] + f["baseline_mapping_ms"] + scenario["detection_stall_ms"] for f in frames]
        baseline, base_reasons = latest_schedule(arrivals, costs)
    names = [method if method != "linear_pose" else f"linear_pose_w{window}" for method, window in methods]
    forecasts = {name: [None] * n for name in names}
    currents, epochs = [None] * n, [-1] * n
    pipeline, history, previous, epoch = HandPipeline(cfg), [], None, 0
    for i, frame in enumerate(frames):
        if baseline[i] is None:
            continue
        broken = previous is not None and (frame["frame_id"] != previous["frame_id"] + 1 or frame["segment_id"] != previous["segment_id"])
        current = observe_frame(pipeline, frame, previous)
        if broken or not current["control_ready"] or current["input_diagnostics"]["state_reset"]:
            history = []
            epoch += 1
        history.append(frame)
        # 只保留足够覆盖最长配置窗口的过去；边界检查仍由时间/段号负责。
        window_span = max(w for _, w in methods) * 1000 / fps + 100
        history = [row for row in history if row["source_time_ms"] >= frame["source_time_ms"] - window_span]
        currents[i], epochs[i] = current, epoch
        for name, (method, window) in zip(names, methods):
            forecasts[name][i] = predict_camera(history, pipeline, current, method=method, window=window, fps=fps)
        previous = frame
    observed = [i for i in range(n) if baseline[i] is not None]
    observed_ready = np.array([baseline[i]["end_ms"] for i in observed])
    records = {}
    for name in names:
        prediction_arrivals = [base["end_ms"] if base is not None and currents[i]["control_ready"] else None
                               for i, base in enumerate(baseline)]
        costs = [(forecasts[name][i]["total_ms"] if forecasts[name][i] is not None else 0.0)
                 + scenario["prediction_stall_ms"] for i in range(n)]
        prediction_schedule, prediction_reasons = latest_schedule(prediction_arrivals, costs)
        if algorithm_only:
            prediction_schedule = [{"start_ms": t, "end_ms": t, "queue_ms": 0.0} if t is not None else None
                                   for t in prediction_arrivals]
        rows = []
        for i, frame in enumerate(frames):
            base, pred, candidate, current = baseline[i], prediction_schedule[i], forecasts[name][i], currents[i]
            base_ready = None if base is None else base["end_ms"]
            for h, horizon in enumerate(HORIZONS_MS):
                target = frame["source_time_ms"] + horizon
                positions = None if candidate is None else candidate["positions"][h]
                model_ms = 0.0 if candidate is None else candidate["model_ms"]
                post_ms = 0.0 if candidate is None else candidate["postprocess_ms"]
                ready = frame["source_time_ms"] if algorithm_only else (None if pred is None else pred["end_ms"])
                reason = None
                if base is None:
                    reason = "source_drop" if base_reasons[i] == "source_drop" else "baseline_queue_replaced"
                elif not current["control_ready"]:
                    reason = frame["reason"] if not frame["input_valid"] else "invalid_current"
                elif not algorithm_only and pred is None:
                    reason = "prediction_queue_replaced" if prediction_reasons[i] == "queue_replaced" else "prediction_unavailable"
                elif positions is None:
                    reason = candidate["reasons"][h]
                elif not algorithm_only and observed:
                    last = int(np.searchsorted(observed_ready, ready + 1e-7, side="right") - 1)
                    if epochs[observed[last]] != epochs[i]:
                        reason = "history_reset_before_ready"
                if reason is None and ready > target + 1e-7:
                    reason = "deadline_miss"
                predicted = reason is None
                fallback = not predicted and current is not None and current["control_ready"] and base_ready <= target + 1e-7
                output = positions if predicted else (current["svh_preview"]["target_positions"] if fallback else None)
                gesture = candidate["gestures"][h] if predicted else (current["gesture_stable"] if fallback else None)
                rows.append({"schema_version": "camera-simple-forecast-v1", "video_id": frame["video_id"],
                    "person_id": frame.get("person_id"), "session_id": frame.get("session_id"),
                    "frame_id": frame["frame_id"], "source_time_ms": frame["source_time_ms"],
                    "timebase": "media_pts_schedule_ms", "timestamp_source": frame["timestamp_source"],
                    "scenario": "algorithm_only" if algorithm_only else scenario["name"], "method": name,
                    "horizon_ms": horizon, "target_time_ms": target, "ready_time_ms": ready,
                    "output_ready_time_ms": ready if predicted else (base_ready if fallback else None),
                    "task_id": GEOMETRY_TASK, "mapping_version": mapping_contract_payload(cfg)["version"],
                    "channel_order": list(SVH_9CH_NAMES),
                    "current_positions": None if current is None or not current["control_ready"] else current["svh_preview"]["target_positions"],
                    # 未被调度的工作没有运行时候选；离线成本采样不能冒充已计算输出。
                    "candidate_positions": positions if algorithm_only or pred is not None else None,
                    "output_positions": output, "gesture": gesture,
                    "prediction_valid": positions is not None and (algorithm_only or pred is not None), "prediction_on_time": predicted,
                    "valid": output is not None, "on_time": output is not None,
                    "fallback_reason": reason, "used_fallback": fallback,
                    "deadline_miss": bool(base is not None and current["control_ready"] and (
                        base_ready > target + 1e-7 or (positions is not None and ready is not None and ready > target + 1e-7))),
                    "remaining_lookahead_ms": None if ready is None else target - ready,
                    "timing": {"arrival_ms": arrivals[i], "detection_start_ms": None if base is None else base["start_ms"],
                        "detection_end_ms": None if base is None else (base["start_ms"] + (0 if algorithm_only else frame["detection_ms"] + scenario["detection_stall_ms"])),
                        "baseline_ready_ms": base_ready, "prediction_enqueued_ms": prediction_arrivals[i],
                        "prediction_start_ms": None if pred is None else pred["start_ms"],
                        "prediction_end_ms": None if pred is None else pred["start_ms"] + (0 if algorithm_only else model_ms),
                        "postprocess_end_ms": ready, "measured_model_ms": model_ms, "measured_postprocess_ms": post_ms,
                        "injected_prediction_stall_ms": 0 if algorithm_only else scenario["prediction_stall_ms"],
                        "baseline_queue_ms": None if base is None else base["queue_ms"],
                        "prediction_queue_ms": None if pred is None else pred["queue_ms"]}})
        records[name] = rows
    return records


def transitions(rows: list, gestures: list) -> list:
    return [(rows[i]["target_time_ms"], gestures[i - 1], gestures[i]) for i in range(1, len(rows))
            if gestures[i - 1] is not None and gestures[i] is not None and gestures[i] != gestures[i - 1]
            and rows[i]["frame_id"] == rows[i - 1]["frame_id"] + 1]


def summarize(rows: list, references: list) -> dict:
    result = {}
    reference_by_id = {reference["frame_id"]: reference for reference in references}
    for h, horizon in enumerate(HORIZONS_MS):
        selected = [row for row in rows if row["horizon_ms"] == horizon]
        errors, conditional, ref_gestures = [], [], []
        missing = 0
        for row in selected:
            reference = reference_by_id[row["frame_id"]]
            target = reference["positions"][h]
            ref_gestures.append(reference["gestures"][h])
            if target is None:
                continue
            if row["output_positions"] is None:
                errors.append(1.0)
                missing += 1
            else:
                mse = float(np.mean((np.asarray(row["output_positions"]) - target) ** 2))
                errors.append(mse)
                conditional.append(mse)
        jumps = [np.abs(np.asarray(b["output_positions"]) - a["output_positions"])
                 for a, b in zip(selected, selected[1:]) if a["valid"] and b["valid"]
                 and a["frame_id"] + 1 == b["frame_id"]]
        events = transitions(selected, ref_gestures)
        candidate_events = transitions(selected, [r["gesture"] for r in selected])
        used, lags = set(), []
        for at, a, b in events:
            options = [(abs(t - at), i, t - at) for i, (t, c, d) in enumerate(candidate_events)
                       if i not in used and (a, b) == (c, d) and abs(t - at) <= 250]
            if options:
                _, index, lag = min(options)
                used.add(index)
                lags.append(lag)
        size = len(selected)
        count = lambda key: sum(bool(r[key]) for r in selected)
        remaining = [r["remaining_lookahead_ms"] for r in selected if r["remaining_lookahead_ms"] is not None]
        latencies = [r["ready_time_ms"] - r["source_time_ms"] for r in selected if r["ready_time_ms"] is not None]
        queues = [r["timing"]["prediction_queue_ms"] for r in selected if r["timing"]["prediction_queue_ms"] is not None]
        result[str(int(horizon))] = {
            "source_count": size, "reference_count": len(errors), "reference_without_output": missing,
            "penalized_rmse": float(np.sqrt(np.mean(errors))) if errors else None,
            "penalized_p95_frame_rmse": float(np.percentile(np.sqrt(errors), 95)) if errors else None,
            "conditional_output_rmse": float(np.sqrt(np.mean(conditional))) if conditional else None,
            "output_count": count("valid"), "prediction_count": count("prediction_on_time"), "fallback_count": count("used_fallback"),
            "no_output_count": size - count("valid"), "deadline_miss_count": count("deadline_miss"),
            "output_coverage": count("valid") / size, "prediction_coverage": count("prediction_on_time") / size,
            "on_time_prediction_rate_all_sources": count("prediction_on_time") / size,
            "reasons": dict(Counter(r["fallback_reason"] for r in selected if r["fallback_reason"])),
            "remaining_lookahead_p05_ms": float(np.percentile(remaining, 5)) if remaining else None,
            "candidate_ready_latency_p50_ms": float(np.median(latencies)) if latencies else None,
            "candidate_ready_latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "candidate_ready_latency_max_ms": float(np.max(latencies)) if latencies else None,
            "prediction_queue_p95_ms": float(np.percentile(queues, 95)) if queues else None,
            "per_channel_jump_p95": np.percentile(jumps, 95, axis=0).tolist() if jumps else None,
            "per_channel_jump_max": np.max(jumps, axis=0).tolist() if jumps else None,
            "reference_transition_count": len(events), "matched_transitions": len(lags),
            "missed_transitions": len(events) - len(lags), "unmatched_prediction_transitions": len(candidate_events) - len(used),
            "transition_lag_median_ms": float(np.median(lags)) if lags else None,
            "transition_abs_lag_p95_ms": float(np.percentile(np.abs(lags), 95)) if lags else None}
    return result
