"""加载 M1 原生 H2O 21 点候选；与现有 9 通道 shadow 接口分开。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2] / "experiments/intent_prediction"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from intent_prediction.keypoint_data import denormalize, make_sequence, prepare_history, task_contract  # noqa: E402
from intent_prediction.keypoint_metrics import query_trajectory  # noqa: E402
from intent_prediction.models import build_model  # noqa: E402
from intent_prediction.training import _resolve_device  # noqa: E402


class LoadedKeypointModel:
    def __init__(self, model_path: Path, *, expected_task_id: str, device: str = "auto") -> None:
        import torch
        from prediction.shadow_predictor import _configure_torch_determinism

        self.spec = json.loads(model_path.read_text(encoding="utf-8"))
        if self.spec.get("schema_version") != "handai-keypoint-model-v1":
            raise ValueError("需要关键点模型配置，不接受旧 9 通道配置")
        self.task = self.spec["task"]
        self.s_min_m = float(self.spec["s_min_m"])
        self.contract = task_contract(self.task, self.s_min_m)
        if expected_task_id != self.contract["task_id"]:
            raise ValueError("调用方任务与模型坐标任务不一致")
        self.device = _resolve_device(device)
        _configure_torch_determinism(torch, torch.device(self.device))
        checkpoint_path = Path(self.spec["checkpoint"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = model_path.parent / checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("data_contract") != self.contract:
            raise ValueError("checkpoint 的坐标、点序、时间或归一化参数与模型配置不一致")
        timing = self.task["time"]
        options = {"input_size": 63, "output_size": 63, "output_bounds": None}
        if checkpoint.get("model_options") != options:
            raise ValueError("关键点模型必须使用 63 维带符号输出")
        if checkpoint["model_name"] != "residual_gru" or checkpoint["history_frames"] != timing["history_frames"]:
            raise ValueError("模型类型或历史长度不匹配")
        if checkpoint["horizon_count"] != len(timing["forecast_steps"]):
            raise ValueError("模型未来步数不匹配")
        if checkpoint["architecture"] != self.spec["architecture"]:
            raise ValueError("模型结构参数与 checkpoint 不一致")
        self.model = build_model("residual_gru", history_frames=timing["history_frames"],
                                  horizon_count=len(timing["forecast_steps"]),
                                  architecture=self.spec["architecture"], **options)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.grid_ms = tuple(np.asarray(timing["forecast_steps"]) / timing["fps"] * 1000)

    def predict_normalized(self, history: np.ndarray) -> np.ndarray:
        """实验批量接口；输入已采用本模型锚点变换的 [N,T,63]。"""
        import torch

        x = np.asarray(history, dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (self.task["time"]["history_frames"], 63) or not np.isfinite(x).all():
            raise ValueError("归一化历史形状不匹配或包含非有限值")
        with torch.inference_mode():
            output = self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()
        if not np.isfinite(output).all():
            raise ValueError("模型输出非有限值")
        return output

    def predict(self, points: np.ndarray, timestamps_s: np.ndarray, *, input_task: dict,
                frame_ids: np.ndarray | None = None, mask: np.ndarray | None = None,
                frame_valid: np.ndarray | None = None, segment_ids: np.ndarray | None = None) -> dict:
        """原始米坐标接口。完整历史不足时明确报错，由上层选择回退。"""
        if task_contract(input_task, self.s_min_m) != self.contract:
            raise ValueError("输入任务的坐标、点序或时间定义与模型不兼容")
        seq = make_sequence("inference", "unspecified", timestamps_s, points,
                            max_gap_s=self.task["time"]["max_label_gap_s"], frame_ids=frame_ids,
                            mask=mask, frame_valid=frame_valid, segment_ids=segment_ids)
        history, reason = prepare_history(seq, len(seq.timestamps) - 1, self.task, self.s_min_m)
        if history is None:
            raise ValueError(f"无法预测：{reason}")
        normalized = self.predict_normalized(history["x"][None])
        horizons = tuple(self.task["time"]["evaluation_horizons_ms"])
        query = query_trajectory(normalized, self.grid_ms, horizons)
        wrists, scales = history["wrist"][None], np.asarray([history["scale"]])
        dense = denormalize(normalized.reshape(1, len(self.grid_ms), 21, 3), wrists, scales)[0]
        return {"task_id": self.contract["task_id"], "source_timestamp_s": float(seq.timestamps[-1]),
                "forecast_timestamps_s": seq.timestamps[-1] + np.asarray(self.grid_ms) / 1000,
                "landmarks_3d_m": dense, "evaluation_horizons_ms": list(horizons),
                "evaluation_timestamps_s": seq.timestamps[-1] + np.asarray(horizons) / 1000,
                "evaluation_landmarks_3d_m": denormalize(query, wrists, scales)[0],
                "anchor_scale_m": history["scale"], "valid": True}


def load_keypoint_prediction_model(model_path: Path, *, expected_task_id: str, device: str = "auto") -> LoadedKeypointModel:
    return LoadedKeypointModel(model_path, expected_task_id=expected_task_id, device=device)
