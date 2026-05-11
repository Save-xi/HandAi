from __future__ import annotations

"""OpenCV 预览窗口右侧状态面板。

GUI 不是控制链路必需部分，但它对调试很重要：可以实时看到检测状态、
手势、连续控制量、SVH preview 和性能指标。
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from output.frame_payload_contract import get_stable_gesture, get_svh_preview

_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
    Path("C:/Windows/Fonts/simkai.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)
"""优先使用中文字体，找不到时再退回通用字体。"""


def _fmt_float(value) -> str:
    """把 None 显示成“无”，数值显示成短小的小数。"""

    if value is None:
        return "无"
    return f"{float(value):.3f}"


def _fmt_list_preview(values, max_items: int = 3) -> str:
    """只展示列表前几个值，避免状态面板太拥挤。"""

    if not values:
        return "[]"
    preview = ", ".join(f"{float(v):.2f}" for v in values[:max_items])
    if len(values) > max_items:
        preview += ", ..."
    return f"[{preview}]"


@lru_cache(maxsize=4)
def _load_font(size: int) -> ImageFont.ImageFont:
    """缓存字体加载结果，避免每帧重新扫描字体文件。"""

    for candidate in _FONT_CANDIDATES:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _draw_text_rows(panel: np.ndarray, rows: list[str]) -> np.ndarray:
    """用 PIL 绘制中文文本，再转回 OpenCV 的 BGR 图像。"""

    rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = _load_font(22)
    y = 18
    for row in rows:
        draw.text((10, y), row, font=font, fill=(0, 255, 255))
        y += 30
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def build_status_panel(height: int, width: int, data: Dict) -> np.ndarray:
    """根据当前 payload 构建一张状态面板图像。"""

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    svh = get_svh_preview(data)
    control = data.get("control_representation", {})
    text_rows = [
        f"检测到手: {data.get('detected')}",
        f"左右手标签: {data.get('handedness')}",
        f"稳定手势: {get_stable_gesture(data)}",
        f"原始手势: {data.get('gesture_raw')}",
        f"控制就绪: {data.get('control_ready')}",
        f"归一化捏合距离: {_fmt_float(data.get('pinch_distance_norm'))}",
        f"手掌张开比例: {_fmt_float(data.get('hand_open_ratio'))}",
        f"控制特征有效: {control.get('features_valid')}",
        f"抓握闭合度: {_fmt_float(control.get('grasp_close'))}",
        f"有效捏合强度: {_fmt_float(control.get('effective_pinch_strength'))}",
        f"拇指-食指接近度: {_fmt_float(control.get('thumb_index_proximity'))}",
        f"拇指 curl: {_fmt_float(data.get('finger_curl', {}).get('thumb'))}",
        f"食指 curl: {_fmt_float(data.get('finger_curl', {}).get('index'))}",
        f"中指 curl: {_fmt_float(data.get('finger_curl', {}).get('middle'))}",
        f"无名指 curl: {_fmt_float(data.get('finger_curl', {}).get('ring'))}",
        f"小指 curl: {_fmt_float(data.get('finger_curl', {}).get('little'))}",
        f"svh_preview 是否有效: {svh.get('valid')}",
        f"svh_preview 模式: {svh.get('mode')}",
        f"svh_preview 目标: {_fmt_list_preview(svh.get('target_positions', []))}",
        f"帧率 fps: {data.get('fps', 0.0):.2f}",
        f"时延 ms: {data.get('latency_ms', 0.0):.2f}",
    ]
    return _draw_text_rows(panel, text_rows)
