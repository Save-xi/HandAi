from __future__ import annotations

"""配置加载与路径解析。

配置文件里的相对路径统一解析到项目根目录，避免从仓库根目录或子项目目录
运行时得到不同输出位置。
"""

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""single-hand-teleop-baseline 子项目根目录。"""


def _resolve_config_path(path: str) -> Path:
    """解析配置文件路径。

    优先使用调用者当前目录下的路径；找不到时再回到项目根目录下查找。
    这样既支持在子项目目录运行，也支持从仓库根目录运行。
    """

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    return PROJECT_ROOT / candidate


def load_config(path: str) -> Dict[str, Any]:
    """读取 yaml 配置，并规范化常用输出路径。"""

    config_path = _resolve_config_path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in ("video_file_path", "output_json_path", "jsonl_output_dir"):
        value = cfg.get(key)
        if isinstance(value, str) and value:
            path_value = Path(value)
            if not path_value.is_absolute():
                # 这些路径是运行产物位置，固定到项目根目录能减少“从哪里启动就写到哪里”的混乱。
                cfg[key] = str(PROJECT_ROOT / path_value)

    return cfg
