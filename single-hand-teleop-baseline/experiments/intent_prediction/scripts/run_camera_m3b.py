"""运行 M3-B 历史摄像头开发诊断；不启动设备控制。"""
from pathlib import Path
import argparse
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT))
from intent_prediction.camera_m3b import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "configs/camera_m3b.json", help="预先声明用途的开发配置")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/m3b", help="结果根目录")
    parser.add_argument("--observations-root", type=Path, help="复用本工具已保存的观测及检测耗时，仍重新计算预测和评分")
    args = parser.parse_args()
    run(args.config, args.output_root, observations_root=args.observations_root)


if __name__ == "__main__":
    main()
