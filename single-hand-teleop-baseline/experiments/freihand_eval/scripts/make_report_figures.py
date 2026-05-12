from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any, Dict


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from freihand.io import load_config, read_json, resolve_path, write_text
from freihand.report import get_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PPT-ready SVG figures from FreiHAND 2D metrics.")
    parser.add_argument("--config", default="../configs/freihand_eval.yaml", help="Path to freihand_eval.yaml")
    parser.add_argument("--metrics", default=None, help="Override eval_metrics.json path")
    parser.add_argument("--output-dir", default=None, help="Override figure output directory")
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


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def svg_text(x: float, y: float, text: Any, *, size: int = 16, weight: str = "400", color: str = "#1f2937", anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Microsoft YaHei, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc(text)}</text>'
    )


def svg_card(x: float, y: float, w: float, h: float, *, fill: str = "#ffffff", stroke: str = "#d7dee8") -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'


def render_detection_matrix(metrics: Dict[str, Any]) -> str:
    counts = metrics.get("sample_counts", {})
    total = int(counts.get("ground_truth_samples", 0) or 0)
    complete = int(counts.get("evaluated_2d_samples", 0) or 0)
    missing = int(counts.get("invalid_2d_prediction_samples", 0) or 0) + int(counts.get("missing_prediction_samples", 0) or 0)
    complete_rate = metrics.get("keypoint_complete_rate")
    missing_rate = None if total == 0 else missing / total

    cell_w = 220
    cell_h = 118
    x0 = 210
    y0 = 135
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="520" viewBox="0 0 860 520">',
        '<rect width="860" height="520" fill="#f7f9fc"/>',
        svg_text(40, 54, "检测结果矩阵", size=30, weight="700"),
        svg_text(40, 86, "FreiHAND 为有手数据集；这里展示当前管线是否完整输出 21 个 2D 点。", size=15, color="#5b6472"),
        svg_text(x0 + cell_w / 2, 120, "预测：完整 21 点", size=16, weight="700", anchor="middle"),
        svg_text(x0 + cell_w * 1.5, 120, "预测：未完整输出", size=16, weight="700", anchor="middle"),
        svg_text(55, y0 + cell_h / 2, "GT：有手", size=17, weight="700"),
        svg_text(55, y0 + cell_h * 1.5, "GT：无手", size=17, weight="700", color="#8a94a3"),
        f'<rect x="{x0}" y="{y0}" width="{cell_w}" height="{cell_h}" rx="12" fill="#dff6e8" stroke="#82c89c"/>',
        f'<rect x="{x0 + cell_w}" y="{y0}" width="{cell_w}" height="{cell_h}" rx="12" fill="#fde7df" stroke="#f0a187"/>',
        f'<rect x="{x0}" y="{y0 + cell_h}" width="{cell_w}" height="{cell_h}" rx="12" fill="#eef2f6" stroke="#d7dee8"/>',
        f'<rect x="{x0 + cell_w}" y="{y0 + cell_h}" width="{cell_w}" height="{cell_h}" rx="12" fill="#eef2f6" stroke="#d7dee8"/>',
        svg_text(x0 + cell_w / 2, y0 + 48, complete, size=34, weight="700", color="#116b38", anchor="middle"),
        svg_text(x0 + cell_w / 2, y0 + 80, pct(complete_rate), size=18, weight="700", color="#116b38", anchor="middle"),
        svg_text(x0 + cell_w * 1.5, y0 + 48, missing, size=34, weight="700", color="#9d3018", anchor="middle"),
        svg_text(x0 + cell_w * 1.5, y0 + 80, pct(missing_rate), size=18, weight="700", color="#9d3018", anchor="middle"),
        svg_text(x0 + cell_w / 2, y0 + cell_h + 62, "N/A", size=28, weight="700", color="#8a94a3", anchor="middle"),
        svg_text(x0 + cell_w * 1.5, y0 + cell_h + 62, "N/A", size=28, weight="700", color="#8a94a3", anchor="middle"),
        svg_text(40, 430, f"总样本：{total}    完整输出：{complete}    未完整输出：{missing}", size=18, weight="700"),
        svg_text(40, 462, "说明：FreiHAND 不包含无手负样本，因此不能用它评估无手误检率。", size=15, color="#5b6472"),
        "</svg>",
    ]
    return "\n".join(lines)


def render_pck_curve(metrics: Dict[str, Any]) -> str:
    pck_all = metrics.get("pck_2d_at_thresholds_all_gt", {})
    pck_valid = metrics.get("pck_2d_at_thresholds", {})
    thresholds = [5, 10, 20, 30]
    x_values = [135, 295, 455, 615]
    y_base = 380
    chart_h = 260

    def y_pos(value: Any) -> float:
        if value is None:
            return y_base
        return y_base - float(value) * chart_h

    def polyline(mapping: Dict[str, Any], color: str) -> str:
        points = " ".join(f"{x},{y_pos(get_threshold(mapping, t)):.2f}" for x, t in zip(x_values, thresholds))
        return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="520" viewBox="0 0 860 520">',
        '<rect width="860" height="520" fill="#f7f9fc"/>',
        svg_text(40, 54, "PCK 曲线", size=30, weight="700"),
        svg_text(40, 86, "全数据口径会把未检测帧计为失败；有效预测口径只看成功输出 21 点的帧。", size=15, color="#5b6472"),
        f'<line x1="95" y1="{y_base}" x2="680" y2="{y_base}" stroke="#aeb8c5" stroke-width="1.5"/>',
        f'<line x1="95" y1="{y_base - chart_h}" x2="95" y2="{y_base}" stroke="#aeb8c5" stroke-width="1.5"/>',
    ]
    for rate in [0.25, 0.5, 0.75, 1.0]:
        y = y_base - rate * chart_h
        lines.append(f'<line x1="95" y1="{y}" x2="680" y2="{y}" stroke="#e3e8ef" stroke-width="1"/>')
        lines.append(svg_text(76, y + 5, f"{int(rate * 100)}%", size=13, color="#667085", anchor="end"))
    lines.extend([polyline(pck_all, "#2563eb"), polyline(pck_valid, "#16a34a")])
    for x, threshold in zip(x_values, thresholds):
        lines.append(svg_text(x, y_base + 32, f"{threshold}px", size=14, color="#475467", anchor="middle"))
        for mapping, color in [(pck_all, "#2563eb"), (pck_valid, "#16a34a")]:
            value = get_threshold(mapping, threshold)
            y = y_pos(value)
            lines.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
    lines.extend(
        [
            f'<rect x="705" y="160" width="18" height="18" rx="4" fill="#2563eb"/>',
            svg_text(732, 175, "全数据", size=15),
            f'<rect x="705" y="198" width="18" height="18" rx="4" fill="#16a34a"/>',
            svg_text(732, 213, "有效预测", size=15),
            svg_text(95, 450, f"PCK@20px 全数据：{pct(get_threshold(pck_all, 20))}", size=18, weight="700", color="#2563eb"),
            svg_text(390, 450, f"PCK@20px 有效预测：{pct(get_threshold(pck_valid, 20))}", size=18, weight="700", color="#16a34a"),
            "</svg>",
        ]
    )
    return "\n".join(lines)


def render_latency_card(metrics: Dict[str, Any]) -> str:
    latency = metrics.get("latency_ms", {})
    mean = latency.get("mean")
    p95 = latency.get("p95")
    fps = None if mean in (None, 0) else 1000.0 / float(mean)
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="360" viewBox="0 0 860 360">',
        '<rect width="860" height="360" fill="#f7f9fc"/>',
        svg_text(40, 54, "实时性指标", size=30, weight="700"),
        svg_card(45, 105, 230, 155, fill="#ffffff"),
        svg_card(315, 105, 230, 155, fill="#ffffff"),
        svg_card(585, 105, 230, 155, fill="#ffffff"),
        svg_text(70, 150, "平均 latency", size=16, color="#667085"),
        svg_text(70, 205, f"{num(mean, 3)} ms", size=34, weight="700", color="#1d4ed8"),
        svg_text(340, 150, "P95 latency", size=16, color="#667085"),
        svg_text(340, 205, f"{num(p95, 3)} ms", size=34, weight="700", color="#7c3aed"),
        svg_text(610, 150, "估算 FPS", size=16, color="#667085"),
        svg_text(610, 205, f"{num(fps, 1)}", size=34, weight="700", color="#0f766e"),
        svg_text(45, 305, "结论：当前管线平均耗时约 19 ms/frame，满足初期实时遥操作 baseline 的速度要求。", size=17, color="#344054"),
        "</svg>",
    ]
    return "\n".join(lines)


def render_dashboard(metrics: Dict[str, Any]) -> str:
    counts = metrics.get("sample_counts", {})
    latency = metrics.get("latency_ms", {})
    pck_all = metrics.get("pck_2d_at_thresholds_all_gt", {})
    cards = [
        ("完整率", pct(metrics.get("keypoint_complete_rate")), "输出 21 个 2D 点", "#1d4ed8"),
        ("PCK@20px", pct(get_threshold(pck_all, 20)), "全数据口径", "#16a34a"),
        ("2D MPJPE", f"{num(metrics.get('mpjpe_2d_px'), 3)} px", "有效预测平均误差", "#ea580c"),
        ("平均耗时", f"{num(latency.get('mean'), 3)} ms", "单帧处理时间", "#7c3aed"),
    ]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="520" viewBox="0 0 1180 520">',
        '<rect width="1180" height="520" fill="#f7f9fc"/>',
        svg_text(50, 60, "当前 CV 管线 FreiHAND 2D 评估总览", size=32, weight="700"),
        svg_text(50, 94, f"evaluation RGB：{counts.get('ground_truth_samples', 0)} 张；完整输出：{counts.get('evaluated_2d_samples', 0)} 张", size=16, color="#5b6472"),
    ]
    for i, (title, value, subtitle, color) in enumerate(cards):
        x = 50 + i * 275
        lines.extend(
            [
                svg_card(x, 140, 245, 165),
                f'<circle cx="{x + 34}" cy="174" r="10" fill="{color}"/>',
                svg_text(x + 55, 181, title, size=17, weight="700"),
                svg_text(x + 28, 238, value, size=34, weight="700", color=color),
                svg_text(x + 28, 274, subtitle, size=14, color="#667085"),
            ]
        )
    lines.extend(
        [
            svg_text(50, 380, "推荐汇报口径", size=20, weight="700"),
            svg_text(50, 416, "当前 MediaPipe 单右手管线在 FreiHAND evaluation RGB 图像上，完整率 85.7%，全数据 PCK@20px 80.8%，平均耗时约 19.1 ms/frame。", size=18, color="#344054"),
            "</svg>",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_config(resolve_config_arg(args.config))
    metrics_path = (
        resolve_cli_path(args.metrics)
        if args.metrics
        else resolve_path(config, config.data["paths"]["eval_metrics_json"])
    )
    output_dir = (
        resolve_cli_path(args.output_dir)
        if args.output_dir
        else resolve_path(config, "../reports/figures")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_json(metrics_path)

    figures = {
        "detection_matrix.svg": render_detection_matrix(metrics),
        "pck_curve.svg": render_pck_curve(metrics),
        "latency_card.svg": render_latency_card(metrics),
        "pipeline_eval_dashboard.svg": render_dashboard(metrics),
    }
    for name, content in figures.items():
        write_text(output_dir / name, content)
        print(f"figure saved to {output_dir / name}")


if __name__ == "__main__":
    main()
