from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from freihand.io import inspect_configured_annotations, load_config, resolve_path, write_text
from freihand.report import render_annotation_inspection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect FreiHAND annotation JSON files.")
    parser.add_argument("--config", default="../configs/freihand_eval.yaml", help="Path to freihand_eval.yaml")
    return parser.parse_args()


def resolve_config_arg(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (THIS_DIR / path).resolve()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_config_arg(args.config))
    result = inspect_configured_annotations(config)
    report_path = resolve_path(config, config.data["paths"]["annotation_inspection_report"])
    write_text(report_path, render_annotation_inspection(result))
    print(f"annotation inspection saved to {report_path}")


if __name__ == "__main__":
    main()
