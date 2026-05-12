"""FreiHAND 离线评估工具包。"""

from .io import load_config, load_split_annotations
from .metrics import evaluate_predictions
from .projection import project_xyz_to_uv

__all__ = [
    "evaluate_predictions",
    "load_config",
    "load_split_annotations",
    "project_xyz_to_uv",
]
