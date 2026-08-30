from __future__ import annotations

"""单右手 9 通道预测的默认关闭影子模式。

影子模式只观察已经生成的 ``svh_preview.target_positions``，并返回可选
``prediction_diagnostics``。它既不改写 ``svh_preview``，也不参与 UDP 发送；
checkpoint、依赖或推理异常都会收敛为诊断状态，而不是中断 baseline。

PyTorch 只在显式启用影子模式时延迟导入，保证原 baseline 环境和启动路径不
因为研究支线而增加硬依赖。
"""

from collections import deque
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Deque, Dict, Sequence

import numpy as np

from svh.svh_layout import SVH_9CH_LAYOUT, SVH_9CH_NAMES
from svh.mapping_contract import (
    MAPPING_CONTRACT_VERSION,
    assert_mapping_implementation_compatible,
    mapping_contract_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTION_PATH = (
    "experiments/intent_prediction/reports/second_round/"
    "20260829T024400_143036Z_selection.json"
)
DEFAULT_CHECKPOINT_PATH = (
    "experiments/intent_prediction/outputs/second_round_v2_open_release/"
    "20260829T024400_143036Z/checkpoints/residual_motion4.pt"
)
DEFAULT_REPORT_PATH = (
    "experiments/intent_prediction/reports/second_round/"
    "20260829T024400_143036Z_report.json"
)

PredictFunction = Callable[[np.ndarray], np.ndarray]


def _unix_ms() -> float:
    return time.time() * 1000.0


def _summarize_exception(exc: Exception, *, max_length: int = 240) -> str:
    detail = str(exc).strip()
    summary = type(exc).__name__ if not detail else f"{type(exc).__name__}: {detail}"
    if len(summary) <= max_length:
        return summary
    return summary[: max_length - 3] + "..."


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


class PredictionShadow:
    """维护连续 9 通道历史并生成 hold/raw/gated 影子诊断。"""

    def __init__(
        self,
        *,
        predict_fn: PredictFunction | None,
        history_frames: int,
        horizon_ms: Sequence[int],
        gate_recent_frames: int,
        gate_threshold: float,
        gate_temperature: float,
        gate_alpha_by_horizon: Sequence[float],
        max_frame_gap_ms: float,
        device: str | None,
        model_label: str | None,
        selection_sha256: str | None,
        checkpoint_sha256: str | None,
        target_fps: float = 30.0,
        initialization_error: str | None = None,
    ) -> None:
        if history_frames < 2:
            raise ValueError("history_frames 必须至少为 2")
        if not horizon_ms or any(int(value) <= 0 for value in horizon_ms):
            raise ValueError("horizon_ms 必须包含正整数")
        if len(gate_alpha_by_horizon) != len(horizon_ms):
            raise ValueError("gate alpha 数量必须与 horizon 数量一致")
        if any(not 0.0 <= float(value) <= 1.0 for value in gate_alpha_by_horizon):
            raise ValueError("gate alpha 必须位于 [0, 1]")
        if gate_temperature < 0.0:
            raise ValueError("gate temperature 不能为负数")
        if max_frame_gap_ms <= 0.0:
            raise ValueError("max_frame_gap_ms 必须为正数")
        if target_fps <= 0.0:
            raise ValueError("target_fps 必须为正数")

        self.predict_fn = predict_fn
        self.history_frames = int(history_frames)
        self.horizon_ms = [int(value) for value in horizon_ms]
        self.gate_recent_frames = max(2, int(gate_recent_frames))
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.gate_alpha_by_horizon = np.asarray(gate_alpha_by_horizon, dtype=np.float64)
        self.max_frame_gap_ms = float(max_frame_gap_ms)
        self.target_fps = float(target_fps)
        self.target_period_ms = 1000.0 / self.target_fps
        self.device = device
        self.model_label = model_label
        self.selection_sha256 = selection_sha256
        self.checkpoint_sha256 = checkpoint_sha256
        self.initialization_error = initialization_error
        self._runtime_error: str | None = None
        # 原始摄像头帧率并不稳定。保留比模型窗口更长的有界原始时间窗，
        # 推理前再按时间戳插值到训练时的固定 30 Hz（可配置 target_fps）。
        self._history: Deque[tuple[int, float, np.ndarray]] = deque(
            maxlen=max(self.history_frames * 4, self.history_frames + 2)
        )

    @classmethod
    def unavailable(
        cls,
        error: str,
        *,
        history_frames: int = 30,
        horizon_ms: Sequence[int] = (50, 100, 150),
        model_label: str | None = None,
        selection_sha256: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> "PredictionShadow":
        """构造不抛异常的初始化失败对象，供主循环持续输出诊断。"""

        return cls(
            predict_fn=None,
            history_frames=history_frames,
            horizon_ms=horizon_ms,
            gate_recent_frames=8,
            gate_threshold=0.0,
            gate_temperature=0.0,
            gate_alpha_by_horizon=[0.0] * len(horizon_ms),
            max_frame_gap_ms=100.0,
            device=None,
            model_label=model_label,
            selection_sha256=selection_sha256,
            checkpoint_sha256=checkpoint_sha256,
            initialization_error=error,
        )

    def _diagnostic(
        self,
        payload: Dict[str, Any],
        *,
        status: str,
        history_available: int,
        fallback_reason: str | None,
        history_span_ms: float | None = None,
        observed_fps: float | None = None,
        hold_last: list[list[float]] | None = None,
        raw_prediction: list[list[float]] | None = None,
        gated_prediction: list[list[float]] | None = None,
        motion_score: float | None = None,
        base_gate: float | None = None,
        effective_gate_by_horizon: list[float] | None = None,
        inference_started_unix_ms: float | None = None,
        inference_completed_unix_ms: float | None = None,
        inference_ms: float | None = None,
        gating_completed_unix_ms: float | None = None,
        gating_ms: float | None = None,
        raw_range_violation_count: int = 0,
    ) -> Dict[str, Any]:
        frame_index = payload.get("frame_index")
        timestamp = payload.get("timestamp")
        timestamp_unix_ms = (
            float(timestamp) * 1000.0
            if isinstance(timestamp, (int, float))
            and not isinstance(timestamp, bool)
            and math.isfinite(float(timestamp))
            else None
        )
        return {
            "schema_version": 1,
            "mode": "shadow",
            "enabled": True,
            "status": status,
            "ready": status == "predicted",
            "source_frame_index": int(frame_index) if isinstance(frame_index, int) and not isinstance(frame_index, bool) else -1,
            "source_timestamp_unix_ms": timestamp_unix_ms,
            "history_frames_required": self.history_frames,
            "history_frames_available": int(history_available),
            "history_span_ms": history_span_ms,
            "observed_fps": observed_fps,
            "horizon_ms": list(self.horizon_ms),
            "channel_order": list(SVH_9CH_NAMES),
            "hold_last": hold_last or [],
            "raw_prediction": raw_prediction or [],
            "gated_prediction": gated_prediction or [],
            "motion_score": motion_score,
            "base_gate": base_gate,
            "effective_gate_by_horizon": effective_gate_by_horizon or [],
            "inference_started_unix_ms": inference_started_unix_ms,
            "inference_completed_unix_ms": inference_completed_unix_ms,
            "inference_ms": inference_ms,
            "gating_completed_unix_ms": gating_completed_unix_ms,
            "gating_ms": gating_ms,
            "raw_range_violation_count": int(raw_range_violation_count),
            "device": self.device,
            "model_label": self.model_label,
            "selection_sha256": self.selection_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "fallback_reason": fallback_reason,
        }

    def _history_timing(self) -> tuple[float | None, float | None]:
        if len(self._history) < 2:
            return None, None
        span_ms = float(self._history[-1][1] - self._history[0][1])
        if span_ms <= 0.0:
            return None, None
        observed_fps = (len(self._history) - 1) * 1000.0 / span_ms
        return span_ms, float(observed_fps)

    def _resample_history(self) -> tuple[np.ndarray | None, int, float | None, float | None]:
        """把不规则原始帧按时间戳重采样成模型所需固定频率窗口。"""

        raw_span_ms, observed_fps = self._history_timing()
        if not self._history or raw_span_ms is None:
            return None, min(len(self._history), self.history_frames), raw_span_ms, observed_fps

        available = min(
            self.history_frames,
            int(math.floor((raw_span_ms + 1e-6) / self.target_period_ms)) + 1,
        )
        target_end_ms = self._history[-1][1]
        target_span_ms = (self.history_frames - 1) * self.target_period_ms
        target_start_ms = target_end_ms - target_span_ms
        if self._history[0][1] > target_start_ms + 1e-6:
            return None, available, raw_span_ms, observed_fps

        timestamps = np.asarray([item[1] for item in self._history], dtype=np.float64)
        values = np.stack([item[2] for item in self._history], axis=0).astype(np.float64)
        target_timestamps = np.linspace(
            target_start_ms,
            target_end_ms,
            self.history_frames,
            dtype=np.float64,
        )
        resampled = np.empty((self.history_frames, len(SVH_9CH_NAMES)), dtype=np.float32)
        for channel in range(len(SVH_9CH_NAMES)):
            resampled[:, channel] = np.interp(
                target_timestamps,
                timestamps,
                values[:, channel],
            ).astype(np.float32)
        return resampled, self.history_frames, float(target_span_ms), observed_fps

    @staticmethod
    def _extract_preview(payload: Dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
        preview = payload.get("svh_preview")
        if not isinstance(preview, dict) or preview.get("valid") is not True:
            return None, "svh_preview_not_valid"
        protocol_hint = preview.get("protocol_hint")
        if not isinstance(protocol_hint, dict) or protocol_hint.get("channel_layout") != SVH_9CH_LAYOUT:
            return None, "svh_preview_layout_not_svh_9ch"
        expected_order = ",".join(SVH_9CH_NAMES)
        if protocol_hint.get("channel_order") != expected_order:
            return None, "svh_preview_channel_order_mismatch"
        positions = preview.get("target_positions")
        if not isinstance(positions, list) or len(positions) != len(SVH_9CH_NAMES):
            return None, "svh_preview_requires_9_positions"
        try:
            values = np.asarray(positions, dtype=np.float64)
        except (TypeError, ValueError):
            return None, "svh_preview_positions_not_numeric"
        if values.shape != (len(SVH_9CH_NAMES),) or not np.all(np.isfinite(values)):
            return None, "svh_preview_positions_not_finite"
        if np.any(values < 0.0) or np.any(values > 1.0):
            return None, "svh_preview_positions_out_of_range"
        return values, None

    @classmethod
    def _extract_history_input(
        cls,
        payload: Dict[str, Any],
    ) -> tuple[np.ndarray | None, int | None, float | None, str | None]:
        """统一校验一帧是否能进入连续历史，并返回毫秒时间戳。"""

        values, invalid_reason = cls._extract_preview(payload)
        timestamp = payload.get("timestamp")
        frame_index = payload.get("frame_index")
        timestamp_valid = (
            isinstance(timestamp, (int, float))
            and not isinstance(timestamp, bool)
            and math.isfinite(float(timestamp))
            and float(timestamp) >= 0.0
        )
        frame_index_valid = (
            isinstance(frame_index, int)
            and not isinstance(frame_index, bool)
            and frame_index >= 0
        )
        if values is None:
            return None, None, None, invalid_reason
        if not timestamp_valid:
            return None, None, None, "source_timestamp_invalid"
        if not frame_index_valid:
            return None, None, None, "source_frame_index_invalid"
        return values, int(frame_index), float(timestamp) * 1000.0, None

    @classmethod
    def requires_history_reset(cls, payload: Dict[str, Any]) -> bool:
        """供 latest-only worker 标记不可被队列覆盖掉的历史断点。"""

        return cls._extract_history_input(payload)[3] is not None

    def reset_history(self) -> None:
        """只清空时序历史，不改变初始化/推理错误的 fail-closed 状态。"""

        self._history.clear()

    def observe(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """观察一帧，但绝不修改传入 payload 或其中的 preview。"""

        if self.initialization_error is not None:
            return self._diagnostic(
                payload,
                status="initialization_error",
                history_available=0,
                fallback_reason=self.initialization_error,
            )
        if self._runtime_error is not None:
            return self._diagnostic(
                payload,
                status="inference_error",
                history_available=0,
                fallback_reason=self._runtime_error,
            )

        values, frame_index, timestamp_ms, invalid_reason = self._extract_history_input(payload)
        if invalid_reason is not None:
            self.reset_history()
            return self._diagnostic(
                payload,
                status="invalid_input",
                history_available=0,
                fallback_reason=invalid_reason,
            )

        assert values is not None and frame_index is not None and timestamp_ms is not None
        reset_reason: str | None = None
        if self._history:
            previous_frame_index, previous_timestamp_ms, _ = self._history[-1]
            if frame_index <= previous_frame_index:
                reset_reason = "history_reset_non_monotonic_frame_index"
            elif timestamp_ms <= previous_timestamp_ms:
                reset_reason = "history_reset_non_monotonic_timestamp"
            elif timestamp_ms - previous_timestamp_ms > self.max_frame_gap_ms:
                reset_reason = "history_reset_frame_gap"
        if reset_reason is not None:
            self.reset_history()

        self._history.append((frame_index, timestamp_ms, values.copy()))
        history, history_available, history_span_ms, observed_fps = self._resample_history()
        if history is None:
            return self._diagnostic(
                payload,
                status="warming_up",
                history_available=history_available,
                fallback_reason=reset_reason or "history_not_ready",
                history_span_ms=history_span_ms,
                observed_fps=observed_fps,
            )

        inference_started_unix_ms = _unix_ms()
        perf_start = time.perf_counter()
        try:
            if self.predict_fn is None:
                raise RuntimeError("predict_fn 不可用")
            raw = np.asarray(self.predict_fn(history.copy()), dtype=np.float64)
            if raw.shape == (1, len(self.horizon_ms), len(SVH_9CH_NAMES)):
                raw = raw[0]
            expected_shape = (len(self.horizon_ms), len(SVH_9CH_NAMES))
            if raw.shape != expected_shape:
                raise ValueError(f"预测输出必须为 {expected_shape}，实际为 {raw.shape}")
            if not np.all(np.isfinite(raw)):
                raise ValueError("预测输出包含非有限值")
        except Exception as exc:
            inference_ms = (time.perf_counter() - perf_start) * 1000.0
            inference_completed_unix_ms = max(
                _unix_ms(),
                inference_started_unix_ms + inference_ms,
            )
            self._runtime_error = _summarize_exception(exc)
            self._history.clear()
            return self._diagnostic(
                payload,
                status="inference_error",
                history_available=self.history_frames,
                fallback_reason=self._runtime_error,
                history_span_ms=history_span_ms,
                observed_fps=observed_fps,
                inference_started_unix_ms=inference_started_unix_ms,
                inference_completed_unix_ms=inference_completed_unix_ms,
                inference_ms=float(inference_ms),
            )

        inference_ms = (time.perf_counter() - perf_start) * 1000.0
        inference_completed_unix_ms = max(
            _unix_ms(),
            inference_started_unix_ms + inference_ms,
        )
        gating_start = time.perf_counter()
        raw_range_violation_count = int(np.count_nonzero((raw < 0.0) | (raw > 1.0)))
        raw_clipped = np.clip(raw, 0.0, 1.0)
        hold = np.repeat(history[-1:, :], len(self.horizon_ms), axis=0).astype(np.float64)
        recent_count = min(max(2, self.gate_recent_frames), len(history))
        motion_score = float(np.mean(np.abs(np.diff(history[-recent_count:, :], axis=0))))
        if self.gate_temperature <= 0.0:
            base_gate = 1.0 if motion_score >= self.gate_threshold else 0.0
        else:
            z = float(np.clip((motion_score - self.gate_threshold) / self.gate_temperature, -40.0, 40.0))
            base_gate = 1.0 / (1.0 + math.exp(-z))
        effective_gate = base_gate * self.gate_alpha_by_horizon
        gated = np.clip(
            hold + effective_gate[:, None] * (raw_clipped - hold),
            0.0,
            1.0,
        )
        gating_ms = (time.perf_counter() - gating_start) * 1000.0
        gating_completed_unix_ms = max(
            _unix_ms(),
            inference_completed_unix_ms + gating_ms,
        )

        return self._diagnostic(
            payload,
            status="predicted",
            history_available=self.history_frames,
            fallback_reason=None,
            history_span_ms=history_span_ms,
            observed_fps=observed_fps,
            hold_last=hold.tolist(),
            raw_prediction=raw_clipped.tolist(),
            gated_prediction=gated.tolist(),
            motion_score=motion_score,
            base_gate=float(base_gate),
            effective_gate_by_horizon=[float(value) for value in effective_gate],
            inference_started_unix_ms=inference_started_unix_ms,
            inference_completed_unix_ms=inference_completed_unix_ms,
            inference_ms=float(inference_ms),
            gating_completed_unix_ms=gating_completed_unix_ms,
            gating_ms=float(gating_ms),
            raw_range_violation_count=raw_range_violation_count,
        )

    def record_runtime_failure(self, payload: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        """把影子层边界外的意外异常转成永久安全回退诊断。"""

        self._runtime_error = _summarize_exception(exc)
        self._history.clear()
        return self._diagnostic(
            payload,
            status="inference_error",
            history_available=0,
            fallback_reason=self._runtime_error,
        )


def _configure_torch_determinism(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def build_prediction_shadow(cfg: Dict[str, Any], *, logger) -> PredictionShadow | None:
    """按冻结 selection/checkpoint 构造影子预测器；默认关闭时返回 None。"""

    if not bool(cfg.get("prediction_shadow_enabled", False)):
        return None

    configured_horizons = cfg.get("prediction_shadow_horizon_ms", [50, 100, 150])
    try:
        horizon_ms = [int(value) for value in configured_horizons]
    except (TypeError, ValueError):
        horizon_ms = [50, 100, 150]
    if not horizon_ms or any(value <= 0 for value in horizon_ms):
        horizon_ms = [50, 100, 150]

    model_label: str | None = None
    selection_sha256: str | None = None
    checkpoint_sha256: str | None = None
    offline_gate_passed: bool | None = None
    try:
        selection_path = _resolve_project_path(
            str(cfg.get("prediction_shadow_selection_path", DEFAULT_SELECTION_PATH))
        )
        selection_bytes = selection_path.read_bytes()
        selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
        selection = json.loads(selection_bytes.decode("utf-8"))
        if selection.get("schema_version") != "intent-second-round-selection-v1":
            raise ValueError("selection schema_version 不受支持")
        if selection.get("selection_fit_split") != "validation":
            raise ValueError("selection 必须只在 validation 上拟合")
        if selection.get("test_loaded") is not False:
            raise ValueError("selection 文件必须在加载 test 前冻结")
        data_contract = selection.get("data_contract")
        if not isinstance(data_contract, dict):
            raise ValueError("selection 缺少 data_contract；旧映射模型必须重新预处理/评估后才能用于当前影子模式")
        expected_mapping_version = data_contract.get("mapping_contract_version")
        expected_mapping_sha256 = data_contract.get("mapping_contract_sha256")
        current_mapping_sha256 = mapping_contract_sha256(cfg)
        if expected_mapping_version != MAPPING_CONTRACT_VERSION:
            raise ValueError(
                "mapping contract 版本不匹配："
                f"expected={expected_mapping_version}, current={MAPPING_CONTRACT_VERSION}"
            )
        if expected_mapping_sha256 != current_mapping_sha256:
            raise ValueError(
                "mapping contract SHA-256 不匹配；当前控制/SVH 映射已变化，"
                "必须重新生成标签并重新评估模型："
                f"expected={expected_mapping_sha256}, current={current_mapping_sha256}"
            )
        assert_mapping_implementation_compatible(cfg, data_contract)

        report_path = _resolve_project_path(
            str(cfg.get("prediction_shadow_report_path", DEFAULT_REPORT_PATH))
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("selection_sha256") != selection_sha256:
            raise ValueError("第二轮 report 引用的 selection SHA-256 与当前 selection 不一致")
        offline_gate_passed = bool(
            report.get("acceptance", {}).get("offline_gate_passed", False)
        )
        if bool(cfg.get("prediction_shadow_require_offline_gate", False)) and not offline_gate_passed:
            raise ValueError("第二轮离线 acceptance gate 未通过，当前配置禁止加载该影子模型")
        model_label = str(selection["selected_label"])

        checkpoint_value = cfg.get("prediction_shadow_checkpoint_path") or selection.get("checkpoint") or DEFAULT_CHECKPOINT_PATH
        checkpoint_path = _resolve_project_path(str(checkpoint_value))
        checkpoint_bytes = checkpoint_path.read_bytes()
        checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        expected_checkpoint_sha256 = str(selection["checkpoint_sha256"])
        if checkpoint_sha256 != expected_checkpoint_sha256:
            raise ValueError(
                "checkpoint SHA-256 不匹配："
                f"expected={expected_checkpoint_sha256}, actual={checkpoint_sha256}"
            )

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "影子模式需要 PyTorch；请使用 handai-intent-prediction 环境"
            ) from exc

        requested_device = str(cfg.get("prediction_shadow_device", "auto")).strip().lower() or "auto"
        if requested_device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("配置请求 CUDA，但当前 PyTorch 无可用 CUDA")
        _configure_torch_determinism(torch, device)

        experiment_root = PROJECT_ROOT / "experiments" / "intent_prediction"
        experiment_root_text = str(experiment_root)
        inserted_path = experiment_root_text not in sys.path
        if inserted_path:
            sys.path.insert(0, experiment_root_text)
        try:
            from intent_prediction.models import build_model
        finally:
            if inserted_path and sys.path and sys.path[0] == experiment_root_text:
                sys.path.pop(0)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("data_contract") != data_contract:
            raise ValueError("checkpoint data_contract 与冻结 selection 不一致")
        selected_model = str(selection["selected_model"])
        if str(checkpoint["model_name"]) != selected_model:
            raise ValueError("selection 与 checkpoint 的模型名不一致")
        history_frames = int(checkpoint["history_frames"])
        horizon_count = int(checkpoint["horizon_count"])
        if horizon_count != len(horizon_ms):
            raise ValueError("配置 horizon 数量与 checkpoint 不一致")
        model = build_model(
            selected_model,
            history_frames=history_frames,
            horizon_count=horizon_count,
            architecture=dict(checkpoint["architecture"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        with torch.no_grad():
            _ = model(torch.zeros((1, history_frames, len(SVH_9CH_NAMES)), dtype=torch.float32, device=device))
            if device.type == "cuda":
                torch.cuda.synchronize()

        def _predict(history: np.ndarray) -> np.ndarray:
            tensor = torch.from_numpy(np.asarray(history, dtype=np.float32)[None, :, :]).to(device)
            with torch.no_grad():
                output = model(tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            return output.detach().cpu().numpy()[0]

        gate = dict(selection["gate_parameters"])
        predictor = PredictionShadow(
            predict_fn=_predict,
            history_frames=history_frames,
            horizon_ms=horizon_ms,
            gate_recent_frames=int(gate["recent_frames"]),
            gate_threshold=float(gate["threshold"]),
            gate_temperature=float(gate["temperature"]),
            gate_alpha_by_horizon=[float(value) for value in gate["alpha_by_horizon"]],
            max_frame_gap_ms=float(cfg.get("prediction_shadow_max_frame_gap_ms", 100.0)),
            device=str(device),
            model_label=model_label,
            selection_sha256=selection_sha256,
            checkpoint_sha256=checkpoint_sha256,
            target_fps=float(cfg.get("prediction_shadow_target_fps", 30.0)),
        )
        logger.info(
            "预测影子模式已加载：label=%s, device=%s, history=%d, horizons=%s, offline_gate_passed=%s；只写诊断，不改 UDP。",
            model_label,
            device,
            history_frames,
            horizon_ms,
            offline_gate_passed,
        )
        if offline_gate_passed is False:
            logger.warning(
                "当前 v2 模型仅作影子诊断：离线 gate 未通过（不会进入 UDP，也不能宣称延迟补偿有效）。"
            )
        return predictor
    except Exception as exc:
        error = _summarize_exception(exc)
        logger.warning("预测影子模式初始化失败（%s）；baseline 与 Unity UDP 将继续运行。", error)
        return PredictionShadow.unavailable(
            error,
            horizon_ms=horizon_ms or (50, 100, 150),
            model_label=model_label,
            selection_sha256=selection_sha256,
            checkpoint_sha256=checkpoint_sha256,
        )
