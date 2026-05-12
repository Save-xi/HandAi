from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from freihand.io import load_config, read_json, resolve_path, write_text
from freihand.report import render_ppt_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a PPT-ready Markdown table from eval_metrics.json.")
    parser.add_argument("--config", default="../configs/freihand_eval.yaml", help="Path to freihand_eval.yaml")
    parser.add_argument("--metrics", default=None, help="Override eval_metrics.json path")
    return parser.parse_args()


def resolve_config_arg(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (THIS_DIR / path).resolve()


def resolve_cli_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_config_arg(args.config))
    metrics_path = resolve_cli_path(args.metrics) if args.metrics else resolve_path(config, config.data["paths"]["eval_metrics_json"])
    output_path = resolve_path(config, config.data["paths"]["ppt_table_md"])
    metrics = read_json(metrics_path)
    write_text(output_path, render_ppt_table(metrics))
    print(f"PPT table saved to {output_path}")


if __name__ == "__main__":
    main()
