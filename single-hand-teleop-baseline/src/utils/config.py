from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_config_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    return PROJECT_ROOT / candidate


def load_config(path: str) -> Dict[str, Any]:
    config_path = _resolve_config_path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in ("video_file_path", "output_json_path"):
        value = cfg.get(key)
        if isinstance(value, str) and value:
            path_value = Path(value)
            if not path_value.is_absolute():
                cfg[key] = str(PROJECT_ROOT / path_value)

    return cfg
