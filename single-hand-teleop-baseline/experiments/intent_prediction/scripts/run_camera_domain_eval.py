from __future__ import annotations

"""直接评测本地视频的姿态稳定性和预测误差。"""

import argparse
import json
import logging
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
for folder in (EXPERIMENT_ROOT, PROJECT_ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from intent_prediction.camera_domain_eval import (  # noqa: E402
    DEFAULT_CONFIG_PATH, DEFAULT_OUTPUT_ROOT, parse_video_specs, run_camera_domain_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="摄像头视频 AI 评测")
    parser.add_argument("--video", action="append", required=True, help="ID=视频路径；可重复指定")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-prediction", action="store_true", help="只评测姿态处理，不加载预测模型")
    parser.add_argument("--input-mirrored", action="store_true", default=None, help="输入为自拍镜像")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_camera_domain_evaluation(
        video_specs=parse_video_specs(args.video), evaluation_config_path=args.config,
        output_root=args.output_root, evaluate_prediction=not args.skip_prediction,
        input_mirrored=args.input_mirrored,
    )
    print(json.dumps({"report": str(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
