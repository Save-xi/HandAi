from __future__ import annotations

"""可独立调用的单右手 AI 流水线，不依赖摄像头、窗口或网络发送。"""

from dataclasses import dataclass
from copy import deepcopy
import logging
import math
import time
from typing import Any, Dict

from control.control_representation import build_control_representation, empty_control_representation
from control.open_release import OpenReleaseState
from features.hand_features import empty_features, extract_hand_features
from gesture.rule_based_gesture import GestureStabilizer, infer_gesture_raw
from output.frame_payload_contract import prepare_frame_payload
from perception.base import HandDetection, HandDetector
from perception.hand_filter import select_right_hand
from perception.geometry import prepare_geometry
from svh.mapping_contract import mapping_contract_payload
from svh.svh_adapter import build_svh_command_preview, empty_svh_preview
from utils.config import validate_config

ExtensionDiagnostics = list[dict[str, str]]


@dataclass(frozen=True)
class RuntimeMode:
    """一次运行里真正生效的模式开关。

    配置文件和 CLI 参数会先合并成 cfg，然后再收敛成 RuntimeMode。
    这样主循环不用到处判断原始配置字段，也能避免“SVH preview 开了但
    control 没开”这类组合状态在代码里散落。
    """

    gui_enabled: bool
    headless: bool
    input_source_type: str
    input_mirrored: bool
    control_extension_enabled: bool
    svh_preview_enabled: bool
    video_file_path: str | None


def _build_runtime_mode(cfg: Dict[str, Any]) -> RuntimeMode:
    """把松散配置收敛成主循环实际使用的运行模式。"""

    svh_preview_enabled = bool(cfg.get("svh_enable_preview", False))
    # SVH preview 依赖 control_representation；因此 preview 开启时隐式开启 control。
    control_extension_enabled = bool(cfg.get("enable_control_extension", False) or svh_preview_enabled)
    gui_enabled = bool(cfg.get("gui_enabled", True)) and not bool(cfg.get("headless", False))
    headless = not gui_enabled
    input_source_type = str(cfg.get("input_source_type", "webcam")).strip().lower() or "webcam"
    video_file_path = str(cfg.get("video_file_path", "") or "").strip() or None
    return RuntimeMode(
        gui_enabled=gui_enabled,
        headless=headless,
        input_source_type=input_source_type,
        input_mirrored=bool(cfg.get("input_mirrored", False)),
        control_extension_enabled=control_extension_enabled,
        svh_preview_enabled=svh_preview_enabled,
        video_file_path=video_file_path,
    )


def _build_baseline_payload(
    frame,
    detector: HandDetector,
    cfg: Dict[str, Any],
    *,
    draw_landmarks: bool,
    timestamp: float | None = None,
) -> Dict[str, Any]:
    """从单帧图像生成 baseline payload。

    这一层只做视觉 baseline：检测右手、检查输入、提取特征和 raw 手势。
    control_representation 和 svh_preview 在后续扩展链路里再补上。
    """

    detections = detector.detect(frame)
    right = select_right_hand(detections)
    ts = time.time() if timestamp is None else float(timestamp)

    diagnostics = prepare_geometry(right, cfg, frame=frame)
    payload = empty_features(ts)
    if diagnostics["geometry_landmarks"]:
        raw_xy = [[float(v) for v in p] for p in right.landmarks_2d]
        raw_xyz = [[float(v) for v in p] for p in right.landmarks_xyz]
    if diagnostics["input_valid"]:
        payload = extract_hand_features(
            raw_xy,
            "Right",
            right.confidence,
            ts,
            landmarks_xyz=raw_xyz,
            geometry_xyz=(diagnostics["geometry_landmarks"] if cfg.get("geometry_mode") == "image_width_xyz_v1" else None),
        )
    elif diagnostics["reason"] == "out_of_bounds":
        # 数组已经检查过；保留图像观察，但不计算无效帧的控制特征。
        payload.update(detected=True, handedness="Right", confidence=float(right.confidence),
                       landmarks_2d=raw_xy, landmarks_3d=raw_xyz)
    if diagnostics["geometry_landmarks"]:
        if draw_landmarks and hasattr(detector, "draw_landmarks"):
            detector.draw_landmarks(frame, raw_xy)

    payload["input_diagnostics"] = diagnostics
    payload["gesture_raw"] = infer_gesture_raw(payload, cfg)
    return payload


def _summarize_exception(exc: Exception, *, max_length: int = 160) -> str:
    """把异常压成单行摘要，避免实时日志被长 traceback 淹没。"""

    detail = str(exc).strip()
    summary = type(exc).__name__ if not detail else f"{type(exc).__name__}: {detail}"
    if len(summary) <= max_length:
        return summary
    return summary[: max_length - 3] + "..."


def _record_extension_failure(
    diagnostics: ExtensionDiagnostics,
    *,
    extension_name: str,
    exc: Exception,
    logger,
    fallback_summary: str,
) -> None:
    """记录扩展失败，但不让扩展失败中断 baseline 主循环。"""

    summary = _summarize_exception(exc)
    diagnostics.append({"extension": extension_name, "error": summary})
    logger.warning("%s 扩展失败（%s）；%s", extension_name, summary, fallback_summary)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "%s 扩展的 traceback 如下。",
            extension_name,
            exc_info=(type(exc), exc, exc.__traceback__),
        )


def _apply_extension_chain(
    payload: Dict[str, Any],
    cfg: Dict[str, Any],
    runtime: RuntimeMode,
    *,
    logger,
    release_state: OpenReleaseState | None = None,
) -> ExtensionDiagnostics:
    """依次运行可选扩展层，并在失败时退回规范占位对象。

    设计原则：
    - baseline 视觉链路优先保持可运行；
    - control / SVH preview 都是可选层；
    - 扩展失败时 payload 仍要满足输出结构。
    """

    diagnostics: ExtensionDiagnostics = []
    if runtime.control_extension_enabled:
        try:
            control_representation = build_control_representation(payload, cfg, release_state=release_state)
        except Exception as exc:
            _record_extension_failure(
                diagnostics,
                extension_name="control_representation",
                exc=exc,
                logger=logger,
                fallback_summary="将继续使用规范的空 control 占位对象。",
            )
            control_representation = empty_control_representation()
    else:
        control_representation = empty_control_representation()

    payload["control_representation"] = control_representation
    # 顶层 control_ready 是给下游快速门控用的镜像字段。
    payload["control_ready"] = bool(control_representation.get("command_ready", False))

    if runtime.svh_preview_enabled:
        try:
            svh_preview = build_svh_command_preview(payload, cfg)
        except Exception as exc:
            _record_extension_failure(
                diagnostics,
                extension_name="svh_preview",
                exc=exc,
                logger=logger,
                fallback_summary="将继续使用规范的空 SVH 预览占位对象。",
            )
            svh_preview = empty_svh_preview(cfg, enabled=True, mode=str(cfg.get("svh_preview_mode", "preview")))
    else:
        svh_preview = empty_svh_preview(cfg, enabled=False, mode="disabled")

    payload["svh_preview"] = svh_preview
    return diagnostics


class _SuppliedDetections:
    def __init__(self, detections: list[HandDetection]):
        self.detections = detections

    def detect(self, _frame):
        return self.detections

    def close(self):
        pass


class HandPipeline:
    """一条输入流对应一个实例，实例持有该流的手势去抖状态。

    注入 detector 可替换姿态模型；process_detections 可直接接设备提供的
    21 点。返回普通字典，由调用方选择 JSON、UDP 或自己的下游适配器。
    detector 的生命周期由创建它的调用方管理。
    """

    def __init__(self, cfg: dict[str, Any], *, detector: HandDetector | None = None, logger=None):
        validate_config(cfg)
        self.cfg = deepcopy(cfg)
        self.runtime = _build_runtime_mode(self.cfg)
        self.detector = detector
        self.logger = logger or logging.getLogger("handai.pipeline")
        self.last_stage_timing: dict[str, float] = {}
        self.stabilizer = GestureStabilizer(
            confirm_frames=int(cfg.get("stable_gesture_min_consecutive", 2)),
            unknown_confirm_frames=int(cfg.get("stable_unknown_consecutive", 1)),
        )
        self.release_state = OpenReleaseState()
        self._last_timestamp: float | None = None
        self._last_image_size: tuple | None = None
        self._mapping_version = mapping_contract_payload(self.cfg)["version"]

    def reset(self) -> None:
        """换流或明确中断时丢弃全部映射历史；预测器历史由调用方同步重置。"""
        self.stabilizer.reset()
        self.release_state.reset()
        self._last_timestamp = None
        self._last_image_size = None

    def fork_mapping(self) -> HandPipeline:
        """复制全部映射状态用于独立姿态分支；复制本身不推进任何计数。

        不复制检测器。未来查询须在副本上按源采样网格推进；M3-B 的
        分数时刻插值/查询调度尚不属于此接口。
        """
        fork = HandPipeline(self.cfg, logger=self.logger)
        fork.stabilizer = deepcopy(self.stabilizer)
        fork.release_state = deepcopy(self.release_state)
        fork._last_timestamp = self._last_timestamp
        fork._last_image_size = self._last_image_size
        return fork

    def _process(self, frame, detector, *, frame_index, timestamp, fps, draw_landmarks):
        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
            raise ValueError("frame_index 必须是非负整数")
        if timestamp is not None and (not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError("timestamp 必须是有限非负秒数")
        started = time.perf_counter()
        payload = _build_baseline_payload(
            frame,
            detector,
            self.cfg,
            draw_landmarks=draw_landmarks,
            timestamp=timestamp,
        )
        diagnostics = payload["input_diagnostics"]
        image_size = (diagnostics["image_width"], diagnostics["image_height"])
        # 旧标签不加入新的时序策略；新流遇到断帧/倒序/换尺寸即重启状态。
        discontinuity = self.cfg.get("geometry_mode") == "image_width_xyz_v1" and (
            (self._last_timestamp is not None and not 0 < payload["timestamp"] - self._last_timestamp
             <= float(self.cfg.get("mapping_max_frame_gap_ms", 100.0)) / 1000.0)
            or (self._last_image_size is not None and image_size != self._last_image_size)
        )
        reset = not diagnostics["input_valid"] or discontinuity
        if reset:
            self.reset()
        if diagnostics["input_valid"]:
            payload["gesture_stable"] = self.stabilizer.update(payload["gesture_raw"])
            self._last_timestamp = payload["timestamp"]
            self._last_image_size = image_size
        else:
            payload["gesture_stable"] = "unknown"
        baseline_end = time.time() * 1000.0
        _apply_extension_chain(payload, self.cfg, self.runtime, logger=self.logger, release_state=self.release_state)
        diagnostics.update(mapping_version=self._mapping_version,
                           release_weight=self.release_state.weight if self.cfg.get("control_open_release_mode") == "continuous_v1" else None,
                           state_reset=bool(reset))
        self.last_stage_timing = {
            "baseline_end_unix_ms": baseline_end,
            "preview_end_unix_ms": time.time() * 1000.0,
        }
        payload.update(frame_index=frame_index, fps=float(fps), latency_ms=(time.perf_counter() - started) * 1000.0)
        return prepare_frame_payload(payload, include_deprecated_aliases=False)

    def process_frame(
        self, frame, *, frame_index: int, timestamp: float | None = None, fps: float = 0.0, draw_landmarks: bool = False
    ) -> dict[str, Any]:
        """处理 BGR 图像；视频评测应显式传媒体时间，实时输入可用默认时钟。"""
        if self.detector is None:
            raise ValueError("process_frame 需要注入 detector；已有关键点请用 process_detections")
        return self._process(
            frame, self.detector, frame_index=frame_index, timestamp=timestamp, fps=fps, draw_landmarks=draw_landmarks
        )

    def process_detections(
        self, detections: list[HandDetection], *, frame_index: int, timestamp: float, fps: float = 0.0
    ) -> dict[str, Any]:
        """HoloLens/Kinect/其他模型的关键点入口，坐标要求见 HandDetection。"""
        return self._process(
            None,
            _SuppliedDetections(detections),
            frame_index=frame_index,
            timestamp=timestamp,
            fps=fps,
            draw_landmarks=False,
        )
