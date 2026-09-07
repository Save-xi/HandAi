from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prediction.model_config import export_model_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="从第二轮训练报告导出可直接加载的预测模型配置")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_model_config(args.report.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
