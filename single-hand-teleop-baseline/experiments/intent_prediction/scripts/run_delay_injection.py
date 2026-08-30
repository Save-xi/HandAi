from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intent_prediction.delay_injection import (  # noqa: E402
    build_runtime_jsonl_forecast_traces,
    run_delay_injection,
)
from prediction.shadow_predictor import build_prediction_shadow  # noqa: E402
from utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单右手意图预测的冻结延迟/抖动/丢包回放")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "delay_injection_v1.json",
        help="运行前冻结的扰动矩阵与 retention gate 配置",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="H2O v2 预处理数据目录")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_ROOT / "outputs" / "delay_injection_v1",
        help="每次运行会在这里新建 UTC 时间戳目录",
    )
    parser.add_argument(
        "--runtime-jsonl",
        type=Path,
        action="append",
        default=[],
        help="可重复提供真实摄像头 baseline JSONL；结果只作跨域探索，不参与 H2O gate",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "unity_udp_preview.yaml",
        help="加载现役 shadow checkpoint 与映射契约的 YAML",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_groups = []
    if args.runtime_jsonl:
        logger = logging.getLogger("delay-injection-runtime-replay")
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
        for jsonl_path in args.runtime_jsonl:
            runtime_cfg = load_config(args.runtime_config)
            runtime_cfg["prediction_shadow_enabled"] = True
            predictor = build_prediction_shadow(runtime_cfg, logger=logger)
            if predictor is None:
                raise SystemExit("runtime shadow predictor 未启用")
            runtime_groups.append(
                build_runtime_jsonl_forecast_traces(
                    jsonl_path,
                    predictor=predictor,
                    recent_frames=int(predictor.gate_recent_frames),
                )
            )
    report_path = run_delay_injection(
        config_path=args.config,
        data_root=args.data_root,
        output_root=args.output_root,
        runtime_trace_groups=runtime_groups,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "report": str(report_path),
                "retention_gate_passed": report["retention"]["retention_gate_passed"],
                "decision": report["retention"]["decision"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
