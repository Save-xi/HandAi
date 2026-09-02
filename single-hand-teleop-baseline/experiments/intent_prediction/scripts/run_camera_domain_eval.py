from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intent_prediction.camera_domain_eval import (  # noqa: E402
    DEFAULT_BLIND_CONFIG_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    parse_video_specs,
    run_camera_domain_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按视频媒体时间轴运行单右手摄像头域确定性评测；不会创建 Unity UDP 输出。"
    )
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="固定视频，可重复提供，例如 --video V1=D:\\videos\\open_fist.mp4",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="机器可读 camera-domain 评测配置",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="每次运行会在这里新建 UTC 时间戳目录",
    )
    parser.add_argument("--role", choices=("development", "blind"), default="development")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="只允许 development 冒烟时缺少部分 V1-V7；盲测永远禁止",
    )
    parser.add_argument("--runtime-config", type=Path, default=None, help="覆盖现役 Unity preview YAML")
    parser.add_argument("--protocol", type=Path, default=None, help="覆盖人类可读准入协议路径")
    mirror = parser.add_mutually_exclusive_group()
    mirror.add_argument("--input-mirrored", action="store_true", help="输入视频已经镜像/自拍化")
    mirror.add_argument("--input-not-mirrored", action="store_true", help="输入视频未镜像")
    parser.add_argument("--expected-config-sha256", default=None)
    parser.add_argument("--expected-protocol-sha256", default=None)
    parser.add_argument(
        "--blind-freeze-manifest",
        type=Path,
        default=None,
        help="正式盲测必填：合并后在 Git 树外生成的冻结清单",
    )
    parser.add_argument(
        "--expected-blind-freeze-manifest-sha256",
        default=None,
        help="正式盲测必填：冻结清单的外部预期 SHA-256",
    )
    parser.add_argument(
        "--live-baseline-jsonl",
        type=Path,
        default=None,
        help="可选：真实摄像头主链 JSONL，用于异步运行能力分析",
    )
    parser.add_argument(
        "--live-prediction-jsonl",
        type=Path,
        default=None,
        help="可选：与 live baseline 对应的 prediction JSONL",
    )
    parser.add_argument(
        "--live-session-manifest",
        type=Path,
        default=None,
        help="可选：runtime_session_<run_id>.json；未给出时从 baseline 同目录严格推导",
    )
    parser.add_argument(
        "--live-unity-timing-json",
        type=Path,
        default=None,
        help="可选：Unity 退出 Play 后生成的有界 timing summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    mirrored_override = None
    if args.input_mirrored:
        mirrored_override = True
    elif args.input_not_mirrored:
        mirrored_override = False
    if args.role == "blind" and args.config.resolve() == DEFAULT_CONFIG_PATH.resolve():
        args.config = DEFAULT_BLIND_CONFIG_PATH
    report_path = run_camera_domain_evaluation(
        video_specs=parse_video_specs(args.video),
        evaluation_config_path=args.config,
        output_root=args.output_root,
        role=args.role,
        allow_partial=args.allow_partial,
        runtime_config_override=args.runtime_config,
        protocol_override=args.protocol,
        input_mirrored_override=mirrored_override,
        expected_config_sha256=args.expected_config_sha256,
        expected_protocol_sha256=args.expected_protocol_sha256,
        live_baseline_jsonl=args.live_baseline_jsonl,
        live_prediction_jsonl=args.live_prediction_jsonl,
        live_session_manifest=args.live_session_manifest,
        live_unity_timing_json=args.live_unity_timing_json,
        blind_freeze_manifest_path=args.blind_freeze_manifest,
        expected_blind_freeze_manifest_sha256=(
            args.expected_blind_freeze_manifest_sha256
        ),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "report": str(report_path),
                "role": report["protocol"]["role"],
                "decision": report["decision"]["status"],
                "udp_created": report["safety"]["udp_created"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
