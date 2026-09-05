from __future__ import annotations

"""不需要摄像头的预测模型推理自检。"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prediction.shadow_predictor import build_prediction_shadow  # noqa: E402
from svh.svh_layout import SVH_9CH_NAMES  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logger import get_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无摄像头的单右手 9 通道预测影子模式 smoke")
    parser.add_argument("--config", default="configs/svh_9ch_preview.yaml")
    parser.add_argument("--model", default=None, help="可选模型 JSON 配置")
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda")
    parser.add_argument("--frames", type=int, default=36, help="合成连续帧数；默认覆盖 30 帧历史窗口")
    parser.add_argument("--output", type=str, default=None, help="可选 JSON 报告路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 30:
        raise SystemExit("--frames 必须至少为 30，才能覆盖冻结 checkpoint 的历史窗口")

    cfg = load_config(args.config)
    if args.model:
        cfg["prediction_shadow_model_path"] = str(Path(args.model).resolve())
    cfg["prediction_shadow_enabled"] = True
    cfg["prediction_shadow_device"] = args.device
    logger = get_logger()
    shadow = build_prediction_shadow(cfg, logger=logger)
    if shadow is None:
        raise SystemExit("影子模式没有启用")

    start = time.perf_counter()
    final_diagnostic = None
    preview_unchanged = True
    base_timestamp = time.time()
    for frame_index in range(args.frames):
        positions = [
            0.5 + 0.25 * math.sin(frame_index * 0.12 + channel * 0.18)
            for channel in range(9)
        ]
        payload = {
            "frame_index": frame_index,
            "timestamp": base_timestamp + frame_index / 30.0,
            "svh_preview": {
                "valid": True,
                "target_positions": positions,
                "protocol_hint": {
                    "channel_layout": "svh_9ch",
                    "channel_order": ",".join(SVH_9CH_NAMES),
                },
            },
        }
        preview_before = deepcopy(payload["svh_preview"])
        final_diagnostic = shadow.observe(payload)
        preview_unchanged = preview_unchanged and payload["svh_preview"] == preview_before

    report = {
        "schema_version": "prediction-shadow-smoke-v1",
        "camera_required": False,
        "frames": args.frames,
        "preview_unchanged": preview_unchanged,
        "elapsed_ms": (time.perf_counter() - start) * 1000.0,
        "final_diagnostic": final_diagnostic,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        logger.info("影子模式 smoke 报告已写入：%s", output_path)

    if final_diagnostic is None or final_diagnostic.get("status") != "predicted" or not preview_unchanged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
