from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from intent_prediction.second_round import run_second_round  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单右手控制意图预测第二轮 validation 选型 + test 一次评估")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "h2o_second_round.json",
        help="第二轮 JSON 配置",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="preprocess_h2o.py 生成的数据目录")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_ROOT / "outputs" / "second_round",
        help="每次运行会在这里新建 UTC 时间戳目录",
    )
    parser.add_argument("--synthetic-smoke", action="store_true", help="只验证流程；结果禁止作为研究结论")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.synthetic_smoke and args.data_root is None:
        raise SystemExit("真实第二轮必须提供 --data-root；代码自检可改用 --synthetic-smoke")
    report_path = run_second_round(
        config_path=args.config,
        data_root=args.data_root,
        output_root=args.output_root,
        synthetic_smoke=args.synthetic_smoke,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "report": str(report_path),
                "selected": report["selection"]["selected_label"],
                "offline_gate_passed": report["acceptance"]["offline_gate_passed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
