from __future__ import annotations

"""把训练评测结果导出成独立运行所需的简短模型配置。"""

import json
from pathlib import Path

from svh.mapping_contract import MAPPING_CONTRACT_VERSION, legacy_v2_mapping_contract

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def export_model_config(report_path: Path, output_path: Path) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection = report["selection"]
    if selection["selected_label"] == "hold_last":
        raise ValueError("该轮选择 hold_last，没有神经网络需要导出")
    selection_path = Path(report["selection_path"])
    if not selection_path.is_absolute():
        selection_path = report_path.parent / selection_path
    selection_record = json.loads(selection_path.read_text(encoding="utf-8"))
    contract = selection_record["data_contract"]
    mapping = contract.get("mapping_contract")
    if mapping is None:
        if contract.get("mapping_contract_version") != MAPPING_CONTRACT_VERSION:
            raise ValueError("数据未记录可用于运行的映射参数；请先按当前流程预处理")
        mapping = legacy_v2_mapping_contract()
    checkpoint = Path(selection["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = report_path.parent / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint}")
    import torch

    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if metadata["model_name"] != selection["selected_model"]:
        raise ValueError("报告与 checkpoint 的模型类型不一致")
    horizons = list(report["horizon_ms"])
    if metadata["horizon_count"] != len(horizons):
        raise ValueError("报告与 checkpoint 的预测距离不一致")
    try:
        checkpoint_text = checkpoint.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        checkpoint_text = str(checkpoint.resolve())
    model = {
        "schema_version": "handai-prediction-model-v1",
        "label": selection["selected_label"],
        "model_name": selection["selected_model"],
        "checkpoint": checkpoint_text,
        "history_frames": metadata["history_frames"],
        "horizon_ms": horizons,
        "target_fps": contract["dataset_fps"],
        "mapping_contract": mapping,
        "gate": selection["gate_parameters"],
        "validation_dynamic_threshold": report["motion_strata_thresholds_from_validation"]["q90"],
        "offline_gate_passed": report["acceptance"]["offline_gate_passed"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path.resolve()
