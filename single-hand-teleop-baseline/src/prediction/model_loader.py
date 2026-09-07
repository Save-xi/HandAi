from __future__ import annotations

"""按可读模型配置加载权重；只检查影响推理正确性的兼容条件。"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Callable

import numpy as np

from svh.mapping_contract import MAPPING_CONTRACT_VERSION, assert_mapping_compatible
from svh.svh_layout import SVH_9CH_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LoadedPredictionModel:
    spec: dict
    checkpoint_path: Path
    device: str
    predict: Callable[[np.ndarray], np.ndarray]


def load_prediction_model(model_path: Path, cfg: dict) -> LoadedPredictionModel:
    spec = json.loads(model_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != "handai-prediction-model-v1":
        raise ValueError("不支持的预测模型配置版本")
    assert_mapping_compatible(cfg, spec["mapping_contract"])
    history_frames = spec["history_frames"]
    horizons = spec["horizon_ms"]
    if not isinstance(history_frames, int) or isinstance(history_frames, bool) or history_frames < 2:
        raise ValueError("history_frames 必须是至少为 2 的整数")
    if (
        not isinstance(horizons, list)
        or not horizons
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in horizons)
        or horizons != sorted(set(horizons))
    ):
        raise ValueError("horizon_ms 必须是严格递增的正整数列表")
    fps = float(spec["target_fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("target_fps 必须是有限正数")
    if not math.isfinite(float(spec["validation_dynamic_threshold"])):
        raise ValueError("validation_dynamic_threshold 必须是有限数字")
    gate = spec["gate"]
    for key in ("threshold", "temperature"):
        value = float(gate[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"gate.{key} 必须是有限非负数")
    alpha = gate["alpha_by_horizon"]
    if len(alpha) != len(horizons) or any(not 0 <= float(value) <= 1 for value in alpha):
        raise ValueError("gate.alpha_by_horizon 必须与预测距离等长且位于 [0,1]")
    for key, expected in (("prediction_shadow_horizon_ms", horizons), ("prediction_shadow_target_fps", fps)):
        if key in cfg and cfg[key] != expected:
            raise ValueError(f"{key} 与模型训练配置不一致")

    import torch

    requested = str(cfg.get("prediction_shadow_device", "auto"))
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无可用 CUDA")
    from prediction.shadow_predictor import _configure_torch_determinism

    _configure_torch_determinism(torch, device)
    checkpoint_path = Path(spec["checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key, expected in (
        ("model_name", spec["model_name"]),
        ("history_frames", history_frames),
        ("horizon_count", len(horizons)),
    ):
        if checkpoint.get(key) != expected:
            raise ValueError(f"checkpoint.{key} 与模型配置不一致")
    if "horizon_ms" in checkpoint and checkpoint["horizon_ms"] != horizons:
        raise ValueError("checkpoint.horizon_ms 与模型配置不一致")
    contract = checkpoint.get("data_contract", {})
    if contract.get("mapping_contract_version") != MAPPING_CONTRACT_VERSION:
        raise ValueError("checkpoint 映射版本不兼容")
    if float(contract.get("dataset_fps", 0)) != fps:
        raise ValueError("checkpoint 采样率与模型配置不一致")
    if contract.get("mapping_contract") is not None:
        assert_mapping_compatible(cfg, contract["mapping_contract"])

    experiment_root = str(PROJECT_ROOT / "experiments" / "intent_prediction")
    if experiment_root not in sys.path:
        sys.path.insert(0, experiment_root)
    from intent_prediction.models import build_model

    model = build_model(
        spec["model_name"],
        history_frames=history_frames,
        horizon_count=len(horizons),
        architecture=checkpoint["architecture"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    def predict(history: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(np.asarray(history, dtype=np.float32)[None], device=device)
        with torch.inference_mode():
            output = model(tensor)
        return output.detach().cpu().numpy()[0]

    warmup = predict(np.zeros((history_frames, len(SVH_9CH_NAMES)), dtype=np.float32))
    if warmup.shape != (len(horizons), len(SVH_9CH_NAMES)) or not np.isfinite(warmup).all():
        raise ValueError("模型预热输出维度错误或包含非有限值")
    return LoadedPredictionModel(spec, checkpoint_path.resolve(), str(device), predict)
