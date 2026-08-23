from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT / "src", EXPERIMENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from intent_prediction.h2o_adapter import preprocess_h2o_pose_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把 H2O pose-only 数据转换为单右手 9 通道时序数据")
    parser.add_argument("--h2o-root", type=Path, required=True, help="已解压且包含 subject1...subject4 的 H2O 根目录")
    parser.add_argument("--output-root", type=Path, required=True, help="必须为空的新输出目录")
    parser.add_argument(
        "--mapping-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "svh_9ch_preview.yaml",
        help="现有 svh_9ch 映射配置",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="H2O pose 序列帧率")
    parser.add_argument(
        "--split-policy",
        choices=("cross_subject", "official"),
        default="cross_subject",
        help="默认按 subject1-2/3/4 切成 train/val/test，避免随机拆帧泄漏",
    )
    parser.add_argument("--min-segment-frames", type=int, default=40, help="短于此值的连续有效片段会丢弃")
    parser.add_argument("--limit-takes", type=int, default=None, help="仅调试时限制处理的 take 数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = preprocess_h2o_pose_dataset(
        h2o_root=args.h2o_root,
        output_root=args.output_root,
        mapping_config_path=args.mapping_config,
        fps=args.fps,
        split_policy=args.split_policy,
        min_segment_frames=args.min_segment_frames,
        limit_takes=args.limit_takes,
    )
    print(f"H2O 单右手预处理完成：{manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
