"""M3-C 本地视频导入、人工标注与真实标签评测。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(EXPERIMENT)]

from intent_prediction.camera_m3c import evaluate_dataset, import_clip  # noqa: E402
from intent_prediction.m3c_annotator import serve  # noqa: E402
from utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs" / "datasets" / "camera_m3c_v1",
                        help="本机 ignored 数据集目录")
    sub = parser.add_subparsers(dest="action", required=True)
    add = sub.add_parser("import", help="导入视频并记录人工声明的身份/用途，不推测身份")
    add.add_argument("--video", type=Path, required=True)
    for key in ("person-id", "session-id", "clip-id", "scenario"):
        add.add_argument("--" + key, required=True)
    add.add_argument("--role", choices=("development", "heldout"), required=True)
    add.add_argument("--synthetic", action="store_true", help="明确标记为合成样例，永不作为独立效果证据")
    add.add_argument("--split-policy", choices=("independent_person", "cross_session"),
                     default="independent_person")
    annotate = sub.add_parser("annotate", help="打开 127.0.0.1 本地标注器")
    annotate.add_argument("--clip-id", required=True)
    annotate.add_argument("--port", type=int, default=8765)
    evaluate = sub.add_parser("evaluate", help="用人工 GT 对公共流水线与三种固定基线评测")
    evaluate.add_argument("--config", type=Path, default=ROOT / "configs" / "ai_m3a.yaml")
    evaluate.add_argument("--pck-px", type=float, default=20.0)
    evaluate.add_argument("--role", choices=("development", "heldout"), default="development",
                          help="默认只评开发集；留出须明确指定")
    evaluate.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.action == "import":
        location = import_clip(args.dataset, video=args.video, person_id=args.person_id,
                               session_id=args.session_id, clip_id=args.clip_id,
                               scenario=args.scenario, role=args.role, split_policy=args.split_policy,
                               synthetic=args.synthetic)
        print(f"已导入 {location}；标注仍为空，不代表 GT")
    elif args.action == "annotate":
        serve(args.dataset / args.clip_id, port=args.port)
    else:
        cfg = load_config(str(args.config.resolve()))
        result = evaluate_dataset(args.dataset, cfg, threshold_px=args.pck_px,
                                  output_root=args.output_root, role=args.role)
        print(f"真实标签报告：{result / 'report.json'}")


if __name__ == "__main__":
    main()
