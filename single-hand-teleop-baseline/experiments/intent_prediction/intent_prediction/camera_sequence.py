"""摄像头开发序列：按事先声明的用途提取，保留全部源帧与失效原因。"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from capture.timeline import resolve_media_timestamp_ms
from perception.geometry import GEOMETRY_TASK
from perception.hand_filter import select_right_hand
from perception.mediapipe_hand import MediaPipeHandDetector
from pipeline import HandPipeline
from svh.mapping_contract import mapping_contract_payload
from utils.config import load_config


def read_config(path: Path) -> tuple[dict, dict]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "camera-m3b-development-v1":
        raise ValueError("不支持的摄像头开发配置")
    videos = config.get("videos", [])
    if not videos or len({v["video_id"] for v in videos}) != len(videos):
        raise ValueError("视频 ID 不能为空或重复")
    resolved_files = [(Path(config["video_root"]) / v["file"]).resolve() for v in videos]
    if len(set(resolved_files)) != len(videos):
        raise ValueError("同一源视频不能分配多个用途")
    sessions = {}
    for entry in videos:
        if entry["role"] not in {"development_validation", "historical_reevaluation"}:
            raise ValueError("本阶段只接受开发选型和历史复评")
        if entry.get("person_id") is None or entry.get("session_id") is None:
            if config.get("provenance_policy") != "unknown_sources_are_historical_development_only":
                raise ValueError("未知人员/会话只能用于明确声明的历史开发")
        else:
            session = (entry["person_id"], entry["session_id"])
            if session in sessions and sessions[session] != entry["role"]:
                raise ValueError("同一已知会话不能跨选型/复评分配")
            sessions[session] = entry["role"]
    if {v["role"] for v in videos} != {"development_validation", "historical_reevaluation"}:
        raise ValueError("必须事先声明开发选型与历史复评两组")
    if config.get("missing_output_error") != 1.0:
        raise ValueError("无输出惩罚固定为归一化通道最大绝对误差 1，不能通过修改惩罚选型")
    if config.get("selection_metric") != "nominal_50_100ms_sequence_equal_penalized_rmse":
        raise ValueError("选型指标与实现不一致")
    if not isinstance(config.get("max_frames_per_video"), int) or config["max_frames_per_video"] < 1:
        raise ValueError("max_frames_per_video 必须为正整数")
    fps = config.get("prediction_fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not math.isfinite(fps) or not 10 <= fps <= 120:
        raise ValueError("prediction_fps 必须在 10..120 之间")
    windows = config.get("velocity_windows")
    if not windows or any(not isinstance(w, int) or isinstance(w, bool) or not 2 <= w <= 32 for w in windows):
        raise ValueError("速度窗口必须为 2..32 的整数")
    scenarios = config.get("scenarios", [])
    if not scenarios or scenarios[0]["name"] != "nominal" or len({s["name"] for s in scenarios}) != len(scenarios):
        raise ValueError("第一个场景必须为 nominal，名称不能重复")
    for scenario in scenarios:
        values = scenario["arrival_delay_cycle_ms"] + [scenario["detection_stall_ms"], scenario["prediction_stall_ms"]]
        if not scenario["arrival_delay_cycle_ms"] or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values):
            raise ValueError("扰动时间必须为有限非负数")
        if not isinstance(scenario["drop_every"], int) or scenario["drop_every"] < 0:
            raise ValueError("drop_every 必须是非负整数")
    cfg = load_config(str((path.parent / config["runtime_config"]).resolve()))
    if cfg.get("geometry_mode") != "image_width_xyz_v1" or cfg.get("control_open_release_mode") != "continuous_v1":
        raise ValueError("M3-B 只使用 M3-A 新几何/连续释放配置")
    return config, cfg


def validate_sequence(frames: list[dict], cfg: dict) -> None:
    """缓存也保留必要的坐标、时序、数值检查；不检查文件身份。"""
    previous, seen_segments, active_segment = None, set(), None
    for index, row in enumerate(frames):
        if row["frame_id"] != index or row.get("task_id") != GEOMETRY_TASK or row.get("timebase") != "media_pts_ms":
            raise ValueError("观测必须保留全部连续源帧和摄像头坐标/时间声明")
        for key in ("source_time_ms", "read_ms", "detection_ms", "baseline_mapping_ms"):
            if not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f"观测 {key} 必须有限非负")
        if previous and row["source_time_ms"] <= previous["source_time_ms"]:
            raise ValueError("源时间必须严格递增")
        if not all(isinstance(row[key], int) and not isinstance(row[key], bool) and row[key] > 0 for key in ("image_width", "image_height")):
            raise ValueError("观测宽高必须为正整数")
        if not isinstance(row["input_valid"], bool):
            raise ValueError("观测有效位必须为布尔值")
        if row["input_valid"]:
            geometry = np.asarray(row["geometry_landmarks"], dtype=float)
            raw = np.asarray(row["raw_landmarks_xyz"], dtype=float)
            if geometry.shape != (21, 3) or raw.shape != (21, 3) or not np.isfinite(geometry).all() or not np.isfinite(raw).all():
                raise ValueError("有效观测必须含有限 21x3 原始/几何坐标")
            expected = raw.copy()
            expected[:, 1] *= row["image_height"] / row["image_width"]
            if not np.allclose(expected, geometry, atol=1e-7, rtol=0):
                raise ValueError("缓存几何与声明的宽高转换不一致")
            if not isinstance(row["segment_id"], int) or row["segment_id"] < 0:
                raise ValueError("有效段号必须为非负整数")
            if previous and (row["source_time_ms"] - previous["source_time_ms"] > cfg.get("mapping_max_frame_gap_ms", 100)
                or (row["image_width"], row["image_height"]) != (previous["image_width"], previous["image_height"])):
                active_segment = None
            if row["segment_id"] != active_segment:
                if row["segment_id"] in seen_segments:
                    raise ValueError("坏帧、间隔或换尺寸之后不能重用旧段号")
                seen_segments.add(row["segment_id"])
                active_segment = row["segment_id"]
        else:
            if row["segment_id"] != -1:
                raise ValueError("无效观测段号必须为 -1")
            active_segment = None
        previous = row


def extract_sequence(entry: dict, config: dict, cfg: dict, run_dir: Path) -> list[dict]:
    path = Path(config["video_root"]) / entry["file"]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or not 1 <= fps <= 240:
        cap.release()
        raise ValueError("视频须提供 1..240 的名义 FPS")
    detector = MediaPipeHandDetector(max_num_hands=int(cfg.get("max_num_hands", 2)),
        min_detection_confidence=float(cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(cfg.get("min_tracking_confidence", 0.5)), input_mirrored=bool(cfg.get("input_mirrored", False)))
    pipeline = HandPipeline(cfg)
    frames, previous, segment, previous_valid = [], None, -1, False
    try:
        with (run_dir / f"{entry['video_id']}.observations.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(config["max_frames_per_video"]):
                read_start = time.perf_counter()
                ok, frame = cap.read()
                read_end = time.perf_counter()
                if not ok:
                    break
                decision = resolve_media_timestamp_ms(index, raw_pts_ms=float(cap.get(cv2.CAP_PROP_POS_MSEC)),
                    nominal_fps=fps, previous_timestamp_ms=previous)
                detect_start = time.perf_counter()
                detections = detector.detect(frame)
                detect_end = time.perf_counter()
                payload = pipeline.process_detections(detections, frame_index=index, timestamp=decision.timestamp_ms / 1000)
                baseline_end = time.perf_counter()
                diag = payload["input_diagnostics"]
                valid = bool(payload["control_ready"] and payload["svh_preview"]["valid"])
                if valid and (not previous_valid or diag["state_reset"]):
                    segment += 1
                right = select_right_hand(detections)
                def raw_points(points):
                    # 原始坏点保留位置，非有限值用 JSON null 表示；不进入几何模型。
                    return [[float(v) if math.isfinite(float(v)) else None for v in p] for p in points]
                row = {"video_id": entry["video_id"], "person_id": entry.get("person_id"), "session_id": entry.get("session_id"),
                    "role": entry["role"], "frame_id": index, "source_time_ms": decision.timestamp_ms,
                    "timebase": "media_pts_ms", "timestamp_source": decision.source, "nominal_fps": fps,
                    "image_width": int(frame.shape[1]), "image_height": int(frame.shape[0]),
                    "raw_landmarks_2d": raw_points(right.landmarks_2d) if right else [],
                    "raw_landmarks_xyz": raw_points(right.landmarks_xyz) if right else [],
                    "geometry_landmarks": diag["geometry_landmarks"], "input_valid": valid, "reason": diag["reason"],
                    "segment_id": segment if valid else -1, "task_id": GEOMETRY_TASK,
                    "read_ms": (read_end - read_start) * 1000, "detection_ms": (detect_end - detect_start) * 1000,
                    "baseline_mapping_ms": (baseline_end - detect_end) * 1000,
                    "measured_detection_end_monotonic_ms": detect_end * 1000,
                    "measured_baseline_end_monotonic_ms": baseline_end * 1000}
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                frames.append(row)
                previous, previous_valid = decision.timestamp_ms, valid
    finally:
        detector.close()
        cap.release()
    if not frames:
        raise ValueError("没有可处理的源帧")
    return frames


def sequence_summary(frames: list[dict], cfg: dict) -> dict:
    first = frames[0]
    return {key: first[key] for key in ("video_id", "person_id", "session_id", "role", "nominal_fps")} | {
        "source_frames": len(frames), "valid_frames": sum(f["input_valid"] for f in frames),
        "valid_segments": len({f["segment_id"] for f in frames if f["input_valid"]}),
        "reasons": dict(Counter(f["reason"] for f in frames)),
        "timestamp_sources": dict(Counter(f["timestamp_source"] for f in frames)),
        "mapping": mapping_contract_payload(cfg)}
