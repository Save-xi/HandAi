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

from intent_prediction.camera_domain_blind_freeze import (  # noqa: E402
    create_blind_freeze_manifest,
    resolve_relative,
)
from intent_prediction.camera_domain_eval import (  # noqa: E402
    DEFAULT_BLIND_CONFIG_PATH,
    load_evaluation_config,
    parse_video_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在 HEAD=origin/main 且完整工作树 clean 的最终合并 revision 上生成摄像头域盲测冻结清单；"
            "该清单放在 Git 树外，避免 commit SHA 自引用。"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_BLIND_CONFIG_PATH,
        help="已启用 blind_frozen 的机器可读配置",
    )
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="必须完整提供 B1-B7；封存输入并核对模型/环境身份，不运行手部检测或预测回放",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    logger = logging.getLogger("camera-domain-blind-freeze")
    config_path = args.config.resolve()
    config = load_evaluation_config(config_path)
    video_specs = parse_video_specs(args.video)
    protocol_path = resolve_relative(config_path, config["protocol_document_path"])
    runtime_config_path = resolve_relative(config_path, config["runtime_config_path"])
    delay_config_path = resolve_relative(config_path, config["delay_config_path"])
    result = create_blind_freeze_manifest(
        evaluation_config_path=config_path,
        protocol_path=protocol_path,
        runtime_config_path=runtime_config_path,
        delay_config_path=delay_config_path,
        blind_video_paths={spec.video_id: spec.path for spec in video_specs},
        logger=logger,
    )
    print(
        json.dumps(
            {
                "manifest": result["path"],
                "sha256": result["sha256"],
                "git_revision": result["manifest"]["git"]["revision"],
                "attempt_token": result["manifest"]["attempt_token"],
                "attempt_receipt": result["attempt_receipt_path"],
                "evaluation_config_sha256": result["manifest"]["artifacts"][
                    "evaluation_config"
                ]["sha256"],
                "protocol_sha256": result["manifest"]["artifacts"]["protocol"][
                    "sha256"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
