"""逐帧日志的断流规则重放：比较计时重置对手势与 9 通道输出的影响。

读取主 CLI 的逐帧 JSONL（landmarks_3d 与源时间），只经 HandPipeline.process_detections 重放，
不重新检测图像。三路使用同一份关键点和时间：
- cadence：当前配置，断流阈值随该流近期周期放宽；
- fixed_floor：上限压到下限，等同 v3 的固定 100ms 阈值；
- no_timing_reset：从不因计时重置，只作差异参照，不是推荐规则。
坏帧、倒序和换尺寸三路相同处理；失效原因按重放重新判定。结果不是手势准确率。
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perception.base import HandDetection  # noqa: E402
from pipeline import HandPipeline  # noqa: E402
from utils.config import load_config  # noqa: E402

NEVER_MS = 1e9


def source_time_s(row: dict) -> float:
    """优先 M3-B 起的 source_timing；旧日志只有检测后生成的 timestamp。"""
    timing = row.get("source_timing")
    return float(timing["source_time_ms"]) / 1000.0 if isinstance(timing, dict) else float(row["timestamp"])


def detections_for(row: dict, width: int, height: int) -> list[HandDetection]:
    points = row.get("landmarks_3d") or []
    if not row.get("detected") or len(points) != 21:
        return []
    diagnostics = row.get("input_diagnostics") or {}
    size = (diagnostics.get("image_width") or width, diagnostics.get("image_height") or height)
    xyz = [[float(v) for v in p] for p in points]
    return [HandDetection([p[:2] for p in xyz], xyz, str(row.get("handedness") or "Right"),
                          float(row.get("confidence") or 0.0), int(size[0]), int(size[1]), "mediapipe_image_xyz")]


def replay(rows: list[dict], cfg: dict, width: int, height: int) -> list[dict]:
    pipeline = HandPipeline(cfg)
    return [pipeline.process_detections(detections_for(row, width, height), frame_index=i, timestamp=source_time_s(row))
            for i, row in enumerate(rows)]


def positions(output: dict):
    preview = output["svh_preview"]
    return np.asarray(preview["target_positions"], dtype=float) if preview["valid"] else None


def arm_summary(outputs: list[dict], reference: list[dict], times: list[float]) -> dict:
    diagnostics = [o["input_diagnostics"] for o in outputs]
    gaps = [(times[i] - times[i - 1]) * 1000.0 for i, d in enumerate(diagnostics) if d["reset_reason"] == "frame_gap"]
    changed = [i for i, (a, b) in enumerate(zip(outputs, reference)) if a["gesture_stable"] != b["gesture_stable"]]
    mapping = [i for i, (a, b) in enumerate(zip(outputs, reference))
               if a["control_representation"]["preferred_mapping"] != b["control_representation"]["preferred_mapping"]]
    deviation = max((float(np.max(np.abs(a - b))) for a, b in ((positions(x), positions(y)) for x, y in zip(outputs, reference))
                     if a is not None and b is not None), default=0.0)
    limits = [d["frame_gap_limit_ms"] for d in diagnostics if d["frame_gap_limit_ms"] is not None]
    return {
        "valid_frames": sum(d["input_valid"] for d in diagnostics),
        "state_reset_frames": sum(d["state_reset"] for d in diagnostics),
        "reset_reasons": dict(Counter(d["reset_reason"] for d in diagnostics if d["reset_reason"])),
        "frame_gap_resets": len(gaps),
        "frame_gap_reset_interval_ms": [round(min(gaps), 1), round(max(gaps), 1)] if gaps else None,
        "gesture_frames_differing_from_no_timing_reset": len(changed),
        "gesture_changes": dict(Counter(f"{reference[i]['gesture_stable']}->{outputs[i]['gesture_stable']}" for i in changed)),
        "mapping_switch_frames": len(mapping),
        "max_channel_deviation": round(deviation, 6),
        "frame_gap_limit_ms": ({"min": round(min(limits), 1), "median": round(float(np.median(limits)), 1),
                                "max": round(max(limits), 1)} if limits else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", type=Path, required=True, help="主 CLI 逐帧 JSONL")
    parser.add_argument("--config", default="configs/ai_m3a.yaml", help="新几何/连续释放配置")
    parser.add_argument("--width", type=int, default=1280, help="日志缺少图像宽度时使用")
    parser.add_argument("--height", type=int, default=720, help="日志缺少图像高度时使用")
    parser.add_argument("--out", type=Path, help="另存 JSON 结果")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.session.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("日志没有逐帧记录")
    cfg = load_config(args.config)
    if cfg.get("geometry_mode") != "image_width_xyz_v1":
        raise SystemExit("断流重放只适用于 M3-A 新几何配置")
    arms = {"cadence": cfg,
            "fixed_floor": {**cfg, "mapping_frame_gap_ceiling_ms": cfg.get("mapping_max_frame_gap_ms", 100.0)},
            "no_timing_reset": {**cfg, "mapping_max_frame_gap_ms": NEVER_MS, "mapping_frame_gap_ceiling_ms": NEVER_MS}}
    outputs = {name: replay(rows, arm, args.width, args.height) for name, arm in arms.items()}
    times = [source_time_s(row) for row in rows]
    intervals = np.diff(times) * 1000.0
    result = {"session": str(args.session), "config": args.config, "source_frames": len(rows),
              "time_source": "source_timing.source_time_ms" if "source_timing" in rows[0] else "timestamp",
              "mapping_version": outputs["cadence"][0]["input_diagnostics"]["mapping_version"],
              "interval_ms": ({"median": round(float(np.median(intervals)), 1), "p95": round(float(np.percentile(intervals, 95)), 1),
                               "max": round(float(np.max(intervals)), 1)} if len(intervals) else None),
              "arms": {name: arm_summary(values, outputs["no_timing_reset"], times) for name, values in outputs.items()}}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
