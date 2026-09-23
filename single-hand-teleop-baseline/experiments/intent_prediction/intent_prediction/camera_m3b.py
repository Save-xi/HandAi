"""开发选型先于历史复评；保存可重放序列、逐源候选与完整分母。"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .camera_replay import common_references, run_scenario, summarize
from .camera_sequence import extract_sequence, read_config, sequence_summary, validate_sequence
from svh.mapping_contract import mapping_contract_payload


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def average_scores(videos: list[dict], scenario: str, methods: list[str]) -> dict:
    values = {}
    for method in methods:
        scores = []
        for video in videos:
            horizons = video["scenarios"][scenario][method]
            pair = [horizons[str(h)]["penalized_rmse"] for h in (50, 100)]
            if any(value is None for value in pair):
                raise ValueError("选型视频缺少可评分的 50/100ms 参考，不能静默排除")
            scores.append(float(np.mean(pair)))
        values[method] = float(np.mean(scores))
    return values


def evaluate_video(entry, frames, cfg, config, run_dir, methods):
    references = common_references(frames, cfg, config["prediction_fps"])
    with (run_dir / f"{entry['video_id']}.references.jsonl").open("w", encoding="utf-8") as handle:
        for reference in references:
            handle.write(json.dumps(reference, ensure_ascii=False, allow_nan=False) + "\n")
    summary = sequence_summary(frames, cfg)
    summary["scenarios"] = {}
    for scenario in [None] + config["scenarios"]:
        actual = config["scenarios"][0] if scenario is None else scenario
        records = run_scenario(frames, cfg, methods=methods, fps=config["prediction_fps"],
            scenario=actual, algorithm_only=scenario is None)
        name = "algorithm_only" if scenario is None else scenario["name"]
        summary["scenarios"][name] = {}
        for method, rows in records.items():
            summary["scenarios"][name][method] = summarize(rows, references)
            with (run_dir / f"{entry['video_id']}.{name}.{method}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"{entry['video_id']} {name} 完成，{len(frames)} 个源时刻", flush=True)
    return summary


def run(config_path: Path, output_root: Path, *, observations_root: Path | None = None) -> Path:
    config, cfg = read_config(config_path)
    run_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    settings = {"mapping": mapping_contract_payload(cfg), "quality": {key: cfg.get(key) for key in (
        "control_ready_min_in_bounds_ratio", "control_ready_palm_center_margin", "control_ready_palm_core_oob_tolerance")}}
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "observation_settings.json", settings)
    if observations_root is not None:
        saved = json.loads((observations_root / "observation_settings.json").read_text(encoding="utf-8"))
        if saved != settings:
            raise ValueError("缓存观测的几何/质量/映射设置不同，请重新提取")

    def load_frames(entry):
        if observations_root is None:
            frames = extract_sequence(entry, config, cfg, run_dir)
            validate_sequence(frames, cfg)
            return frames
        path = observations_root / f"{entry['video_id']}.observations.jsonl"
        frames = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        frames = frames[:config["max_frames_per_video"]]
        if not frames or any(f["video_id"] != entry["video_id"] for f in frames):
            raise ValueError("缓存序列为空或 video_id 不一致")
        for frame in frames:
            for key in ("role", "person_id", "session_id"):
                frame[key] = entry.get(key)
        validate_sequence(frames, cfg)
        # 复用观测时明确记录源目录，避免把缓存当作本次检测计时。
        with (run_dir / path.name).open("w", encoding="utf-8") as handle:
            for frame in frames:
                handle.write(json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n")
        return frames

    report = {"schema_version": "camera-m3b-report-v1", "config": config, "mapping": settings["mapping"],
        "observation_cache_source": str(observations_root) if observations_root is not None else None,
        "scope": "历史摄像头开发片段的检测轨迹代理评测；人员/会话未知时不支持独立泛化结论",
        "reference": "各源锚点的完整无扰动状态副本 + 同段真实未来姿态；所有方法及调度场景共享",
        "timing": "媒体 PTS 上的两阶段成本重放；感知成本来自实际测量，预测含当前方法全部状态复制/后处理；每阶段一个等待槽",
        "timing_limits": "从 source.read 返回计时，不含曝光、解码等待、文件导出、设备/网络实测；丢帧不重新运行 MediaPipe 跟踪器",
        "error_policy": "有参考却无输出时各通道平方误差按 1 惩罚；明确另列条件输出误差。预测失败优先保持当前；无有效当前或基础输出逾期则无输出。",
        "coverage_denominator": "全部实际源帧，包含预热、坏帧、源丢弃、队列替换、逾期和尾部无参考帧",
        "development_validation": [], "historical_reevaluation": []}
    candidates = [("hold_channels", 2), ("hold_pose", 2)] + [("linear_pose", w) for w in config["velocity_windows"]]
    for entry in config["videos"]:
        if entry["role"] == "development_validation":
            report["development_validation"].append(evaluate_video(entry, load_frames(entry), cfg, config, run_dir, candidates))
    names = [m if m != "linear_pose" else f"linear_pose_w{w}" for m, w in candidates]
    scores = average_scores(report["development_validation"], "nominal", names)
    linear = min((name for name in scores if name.startswith("linear_pose")), key=lambda name: (scores[name], name))
    selected_window = int(linear.rsplit("w", 1)[1])
    winner = min(("hold_channels", "hold_pose", linear), key=lambda name: (scores[name], name))
    report["selection"] = {"metric": config["selection_metric"], "scores": scores, "linear_window": selected_window,
        "winner": winner, "independent_validation": False,
        "note": "按预先分配文件做历史开发选型，人员/会话未知；不在复评后改名为独立测试"}
    # 先写选型结果，再提取/评分复评组；无需文件身份或一次性运行限制。
    write_json(run_dir / "selection.json", report["selection"])
    selected = [("hold_channels", 2), ("hold_pose", 2), ("linear_pose", selected_window)]
    for entry in config["videos"]:
        if entry["role"] == "historical_reevaluation":
            report["historical_reevaluation"].append(evaluate_video(entry, load_frames(entry), cfg, config, run_dir, selected))
    report["reevaluation_50_100_scores"] = {scenario: average_scores(report["historical_reevaluation"], scenario,
        ["hold_channels", "hold_pose", linear]) for scenario in ["algorithm_only"] + [s["name"] for s in config["scenarios"]]}
    write_json(run_dir / "report.json", report)
    print(f"M3-B 报告：{run_dir / 'report.json'}", flush=True)
    return run_dir
