from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from freihand.io import limit_split, load_config, load_split_annotations, resolve_path, write_json
from freihand.projection import numpy_to_jsonable_points, project_xyz_to_uv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project FreiHAND xyz annotations to 2D uv keypoints.")
    parser.add_argument("--config", default="../configs/freihand_eval.yaml", help="Path to freihand_eval.yaml")
    parser.add_argument("--split", default=None, help="Override split name, e.g. training or evaluation")
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
    split_name = args.split or str(config.data.get("project", {}).get("default_split", "evaluation"))
    max_samples = config.data.get("runtime", {}).get("max_samples")
    split = limit_split(load_split_annotations(config, split_name, require_xyz=True, require_K=True), max_samples)
    if split.xyz is None or split.K is None:
        raise RuntimeError(f"split '{split_name}' does not contain xyz and K annotations")

    uv = project_xyz_to_uv(
        split.xyz,
        split.K,
        z_epsilon=float(config.data.get("projection", {}).get("z_epsilon", 1.0e-8)),
    )
    output = {sid: numpy_to_jsonable_points(uv[i]) for i, sid in enumerate(split.sample_ids)}
    output_path = resolve_path(config, config.data["paths"]["projected_2d_keypoints_json"])
    write_json(output_path, output)
    print(f"projected {len(output)} samples from split '{split_name}' to {output_path}")


if __name__ == "__main__":
    main()
