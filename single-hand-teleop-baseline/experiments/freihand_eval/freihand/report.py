from __future__ import annotations

from typing import Any, Dict


def format_number(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return str(value)


def format_percent(value: Any, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def get_threshold(mapping: Dict[str, Any], threshold: int | float) -> Any:
    """兼容 JSON 里可能出现的 20 / 20.0 两种阈值 key。"""

    keys = [f"{float(threshold):g}", str(threshold), str(float(threshold))]
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def render_annotation_inspection(result: Dict[str, Any]) -> str:
    lines = [
        "# FreiHAND Annotation Inspection",
        "",
        "本报告由 `inspect_annotations.py` 生成，用于确认 FreiHAND annotation 是否适合离线 2D 关键点评估。",
        "",
    ]
    for split, info in result.items():
        lines.extend(
            [
                f"## {split}",
                "",
                f"- split 名称：`{split}`",
                f"- 样本数量：{info.get('sample_count', 0)}",
                f"- xyz 是否全部为 21 个关键点：{info.get('all_xyz_samples_have_21_keypoints')}",
                "",
                "| 文件 | 是否存在 | 数量 | 维度统计 | 非法维度 | None | NaN | 路径 |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for key in ["K", "xyz", "scale"]:
            file_info = info.get(key, {})
            shape_counts = ", ".join(f"{shape}: {count}" for shape, count in file_info.get("shape_counts", {}).items())
            lines.append(
                "| {key} | {exists} | {count} | {shape_counts} | {invalid} | {none} | {nan} | `{path}` |".format(
                    key=key,
                    exists=file_info.get("exists", False),
                    count=file_info.get("count", 0),
                    shape_counts=shape_counts or "N/A",
                    invalid=file_info.get("invalid_shape_count", "N/A"),
                    none=file_info.get("contains_none_count", "N/A"),
                    nan=file_info.get("contains_nan_count", "N/A"),
                    path=file_info.get("path") or "N/A",
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_eval_report(metrics: Dict[str, Any], *, split: str, prediction_path: str) -> str:
    counts = metrics.get("sample_counts", {})
    pck2d = metrics.get("pck_2d_at_thresholds", {})
    pck2d_all = metrics.get("pck_2d_at_thresholds_all_gt", pck2d)
    latency = metrics.get("latency_ms", {})

    lines = [
        "# FreiHAND 2D Evaluation Report",
        "",
        f"- split：`{split}`",
        f"- predictions：`{prediction_path}`",
        f"- ground truth samples：{counts.get('ground_truth_samples', 0)}",
        f"- matched prediction samples：{counts.get('matched_prediction_samples', 0)}",
        f"- 2D keypoint complete rate：{format_percent(metrics.get('keypoint_complete_rate'))}",
        "",
        "## 核心指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 2D MPJPE | {format_number(metrics.get('mpjpe_2d_px'), 3, ' px')} |",
        f"| PCK@20px（全数据口径） | {format_percent(get_threshold(pck2d_all, 20))} |",
        f"| 检测器+手选择平均耗时 | {format_number(latency.get('mean'), 3, ' ms')} |",
        f"| 检测器+手选择 P95 | {format_number(latency.get('p95'), 3, ' ms')} |",
        f"| 检测器+手选择 P99 | {format_number(latency.get('p99'), 3, ' ms')} |",
        f"| 检测器+手选择最大耗时 | {format_number(latency.get('max'), 3, ' ms')} |",
        f"| 超过 {format_number(latency.get('threshold_ms'), 1, ' ms')} | {latency.get('over_threshold_count', 0)} / {latency.get('count', 0)} |",
        "",
        "## PCK 细分",
        "",
        "| 类型 | 阈值 | 数值 |",
        "| --- | ---: | ---: |",
    ]
    for threshold, value in pck2d_all.items():
        lines.append(f"| 2D 全数据 | {threshold}px | {format_percent(value)} |")
    for threshold, value in pck2d.items():
        lines.append(f"| 2D 有效预测 | {threshold}px | {format_percent(value)} |")

    lines.extend(
        [
            "",
            "## 数据质量提示",
            "",
            f"- 缺失预测样本：{counts.get('missing_prediction_samples', 0)}",
            f"- 额外预测样本：{counts.get('extra_prediction_samples', 0)}",
            f"- 2D 非法预测样本：{counts.get('invalid_2d_prediction_samples', 0)}",
            f"- 参与 2D 指标计算的样本 / 关键点：{counts.get('evaluated_2d_samples', 0)} / {counts.get('evaluated_2d_keypoints', 0)}",
            "",
        ]
    )
    return "\n".join(lines)


def render_ppt_table(metrics: Dict[str, Any]) -> str:
    pck2d_all = metrics.get("pck_2d_at_thresholds_all_gt", metrics.get("pck_2d_at_thresholds", {}))
    latency = metrics.get("latency_ms", {})
    lines = [
        "| 指标 | 数值 | PPT 口径 |",
        "| --- | ---: | --- |",
        f"| 关键点完整率 | {format_percent(metrics.get('keypoint_complete_rate'))} | 预测是否完整输出 21 个 2D 关键点 |",
        f"| PCK@20px | {format_percent(get_threshold(pck2d_all, 20))} | 全数据口径，未检测帧计为失败 |",
        f"| 2D MPJPE | {format_number(metrics.get('mpjpe_2d_px'), 3, ' px')} | 2D 平均关键点误差 |",
        f"| 检测器+手选择平均耗时 | {format_number(latency.get('mean'), 3, ' ms')} | 不含读图、映射、UDP、Unity |",
        f"| 检测器+手选择 P95 | {format_number(latency.get('p95'), 3, ' ms')} | 95% 样本不超过该耗时 |",
        f"| 检测器+手选择 P99 | {format_number(latency.get('p99'), 3, ' ms')} | 99% 样本不超过该耗时 |",
    ]
    return "\n".join(lines) + "\n"
