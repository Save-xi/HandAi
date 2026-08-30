from __future__ import annotations

"""单右手遥操作 baseline 的运行入口。

这个文件负责把各个模块串成一条实时 pipeline：

摄像头/视频 -> MediaPipe 检测 -> 右手筛选 -> 特征提取 -> 手势稳定
-> 可选 control 表示 -> 可选 SVH preview -> JSON/JSONL/UDP/GUI 输出。

注意：这里尽量只做“编排”。具体算法逻辑分别放在 perception、
features、gesture、control、svh、output 等子模块里，方便单独测试和替换。
"""

import argparse
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2

from capture.input_source import InputSource
from capture.video_file import VideoFileSource
from capture.webcam import WebcamSource
from control.control_representation import build_control_representation, empty_control_representation
from features.hand_features import empty_features, extract_hand_features, invalidate_control_features
from gesture.rule_based_gesture import GestureStabilizer, infer_gesture_raw
from output.frame_payload_contract import assert_valid_frame_payload, prepare_frame_payload
from output.json_exporter import JsonExporter
from perception.hand_filter import select_right_hand
from perception.landmark_quality import assess_control_readiness
from perception.mediapipe_hand import MediaPipeHandDetector
from prediction.shadow_predictor import build_prediction_shadow
from prediction.shadow_worker import PredictionShadowWorker
from svh.svh_adapter import build_svh_command_preview, empty_svh_preview
from utils.config import load_config
from utils.logger import get_logger
from utils.recent_frames import RecentFrameBuffer
from utils.timer import FrameTimer, now_ts
from visualize.overlay_2d import compose_view


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


ExtensionDiagnostics = List[Dict[str, str]]
"""扩展链路的非致命错误记录。

baseline 的设计目标是：control / SVH preview 失败时，主感知链路仍然运行。
这里记录的诊断信息主要给测试、日志和后续调试使用。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单右手遥操作 baseline 演示")
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    parser.add_argument("--camera-index", default=None, type=int, help="覆盖默认摄像头的相机索引")
    parser.add_argument("--video-file", default=None, type=str, help="从本地视频文件读取帧，而不是使用摄像头")
    parser.add_argument("--input-mirrored", action="store_true", help="把输入视为已经镜像/自拍视角")
    parser.add_argument("--enable-control", action="store_true", help="启用 control_representation 扩展层")
    parser.add_argument("--preview-svh", action="store_true", help="启用 SVH 预览扩展层（会隐式开启 --enable-control）")
    parser.add_argument("--no-gui", action="store_true", help="关闭 OpenCV 预览窗口，但保持实时处理")
    parser.add_argument("--headless", action="store_true", help="完全无窗口运行；适合日志、JSONL 或视频文件处理")
    parser.add_argument("--max-frames", default=None, type=int, help="处理到 N 帧后自动停止")
    parser.add_argument("--print-json", action="store_true", help="把逐帧 JSON 打印到控制台")
    parser.add_argument("--save-jsonl", action="store_true", help="为本次运行启用逐帧 JSONL 日志")
    parser.add_argument(
        "--prediction-shadow",
        action="store_true",
        help="启用默认关闭的 9 通道预测影子诊断；不改变 Unity UDP 或 svh_preview",
    )
    return parser.parse_args()


def _resolve_user_path(path_str: str) -> str:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path.cwd() / candidate).resolve())


def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """把命令行参数覆盖到配置字典上。

    配置文件保留默认运行方式，CLI 用来做一次性实验覆盖。比如临时换摄像头、
    临时读取视频、临时启用 SVH preview，都不需要改 yaml。
    """

    cfg = dict(cfg)
    if args.camera_index is not None:
        cfg["camera_index"] = args.camera_index
    if args.video_file:
        cfg["input_source_type"] = "video_file"
        cfg["video_file_path"] = _resolve_user_path(args.video_file)
    if args.input_mirrored:
        cfg["input_mirrored"] = True
    if args.enable_control:
        cfg["enable_control_extension"] = True
    if args.preview_svh:
        cfg["enable_control_extension"] = True
        cfg["svh_enable_preview"] = True
    if args.no_gui:
        cfg["gui_enabled"] = False
    if args.headless:
        cfg["headless"] = True
        cfg["gui_enabled"] = False
    if args.save_jsonl:
        cfg["save_jsonl"] = True
    if getattr(args, "prediction_shadow", False):
        cfg["prediction_shadow_enabled"] = True
    return cfg


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


def _build_jsonl_session_path(
    cfg: Dict[str, Any],
    *,
    output_dir_key: str = "jsonl_output_dir",
    prefix: str = "session",
) -> str:
    """为本次运行生成 JSONL 会话日志路径。"""

    output_dir = Path(str(cfg.get(output_dir_key, "outputs")))
    # 微秒 + PID 避免同一秒内启动两次或并行启动时互相追加到同一文件。
    nonce = uuid.uuid4().hex[:8]
    return str(
        output_dir
        / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}_{nonce}.jsonl"
    )


def _build_input_source(cfg: Dict[str, Any], runtime: RuntimeMode, logger) -> InputSource | None:
    """根据运行模式创建输入源。

    返回 None 表示输入源无法安全打开，调用方应直接退出，而不是进入主循环。
    """

    source_type = runtime.input_source_type
    if source_type == "video_file":
        if not runtime.video_file_path:
            logger.error("请求了视频文件模式，但没有提供视频路径；程序将安全退出。")
            return None
        source = VideoFileSource(runtime.video_file_path)
        if not source.is_opened():
            logger.error("无法打开视频文件 '%s'；程序将安全退出。", runtime.video_file_path)
            source.release()
            return None
        logger.info("当前使用视频文件输入：%s", runtime.video_file_path)
        return source

    if source_type != "webcam":
        logger.warning("不支持的 input_source_type '%s'；将回退到默认摄像头输入。", source_type)

    camera_index = int(cfg.get("camera_index", 0))
    source = WebcamSource(
        camera_index=camera_index,
        width=int(cfg["display_width"]),
        height=int(cfg["display_height"]),
    )
    if not source.is_opened():
        logger.error("无法打开 camera_index=%s 对应的摄像头；程序将安全退出。", camera_index)
        source.release()
        return None
    logger.info("当前使用摄像头输入，camera_index=%s。", camera_index)
    return source


def _build_detector(cfg: Dict[str, Any], runtime: RuntimeMode) -> MediaPipeHandDetector:
    """创建 MediaPipe 手部检测器，并把镜像视角信息传进去。"""

    return MediaPipeHandDetector(
        max_num_hands=int(cfg.get("max_num_hands", 2)),
        min_detection_confidence=float(cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(cfg.get("min_tracking_confidence", 0.5)),
        input_mirrored=runtime.input_mirrored,
    )


def _build_exporter(cfg: Dict[str, Any], logger) -> JsonExporter:
    """创建统一导出器。

    JsonExporter 同时负责 last-frame JSON、可选 JSONL、可选 Unity UDP。
    主循环只把规范化后的 payload 交给它，不关心具体输出通道。
    """

    return JsonExporter(
        output_path=str(cfg.get("output_json_path", "outputs/latest_frame.json")),
        save_last_json=bool(cfg.get("save_last_json", True)),
        jsonl_path=_build_jsonl_session_path(cfg) if bool(cfg.get("save_jsonl", False)) else None,
        export_last_every_n_frames=int(cfg.get("export_last_every_n_frames", 1)),
        jsonl_flush_interval=int(cfg.get("jsonl_flush_interval", 1)),
        unity_udp_enabled=bool(cfg.get("unity_udp_enabled", False)),
        unity_udp_host=str(cfg.get("unity_udp_host", "127.0.0.1")),
        unity_udp_port=int(cfg.get("unity_udp_port", 18080)),
        logger=logger,
    )


def _build_prediction_result_exporter(cfg: Dict[str, Any], logger) -> JsonExporter:
    """为后台影子结果创建独立本机输出；绝不启用 UDP。"""

    return JsonExporter(
        output_path=str(
            cfg.get(
                "prediction_shadow_output_json_path",
                "outputs/latest_prediction_shadow.json",
            )
        ),
        save_last_json=bool(cfg.get("save_last_json", True)),
        jsonl_path=(
            _build_jsonl_session_path(
                cfg,
                output_dir_key="prediction_shadow_jsonl_output_dir",
                prefix="prediction_session",
            )
            if bool(cfg.get("save_jsonl", False))
            else None
        ),
        export_last_every_n_frames=1,
        jsonl_flush_interval=int(cfg.get("jsonl_flush_interval", 1)),
        unity_udp_enabled=False,
        logger=logger,
    )


def _build_svh_transport(cfg: Dict[str, Any], runtime: RuntimeMode, logger):
    """创建 SVH preview 传输层。

    当前只有 mock transport。真实 TCP / 串口 / RS485 还没有接入，
    因此这里不能把非 mock 配置伪装成可用硬件链路。
    """

    if not runtime.svh_preview_enabled:
        return None
    svh_transport_name = str(cfg.get("svh_transport", "mock"))
    if svh_transport_name == "mock":
        from svh.svh_transport_mock import MockSvhTransport

        logger.info("SVH 预览扩展已启用，当前使用 mock 传输。")
        return MockSvhTransport(
            logger=logger,
            history_size=int(cfg.get("svh_mock_history_size", 32)),
        )
    logger.warning(
        "不支持的 SVH transport '%s'；将继续以纯预览模式运行，不发送传输命令。",
        svh_transport_name,
    )
    return None


def _log_runtime_mode(runtime: RuntimeMode, cfg: Dict[str, Any], logger) -> None:
    """启动时集中打印运行模式，方便排查“我到底开了哪些扩展”。"""

    if runtime.gui_enabled:
        logger.info("GUI 已启用。请在 OpenCV 窗口中按 q 退出。")
    else:
        logger.info("GUI 已关闭；当前以无界面模式运行。")

    if runtime.input_mirrored:
        logger.info("输入将按镜像/自拍视角处理。")

    if not runtime.control_extension_enabled and not runtime.svh_preview_enabled:
        logger.info("当前运行在纯 baseline 模式；control 与 SVH 扩展均已关闭。")
    elif runtime.control_extension_enabled and not runtime.svh_preview_enabled:
        logger.info("control 扩展已启用；SVH 预览扩展保持关闭。")
    elif runtime.svh_preview_enabled:
        logger.info("SVH 预览扩展已启用，当前以纯预览模式运行。")

    if bool(cfg.get("save_jsonl", False)):
        logger.info("本次运行已启用逐帧 JSONL 日志。")
    if bool(cfg.get("prediction_shadow_enabled", False)):
        logger.info(
            "预测影子模式已请求；UDP 后仅做非阻塞提交，后台结果写入独立 prediction JSON/JSONL，"
            "不会改写 svh_preview 或 UDP payload。"
        )
        if not runtime.svh_preview_enabled:
            logger.warning(
                "预测影子模式需要有效的 svh_9ch preview；当前配置未启用 SVH preview，"
                "因此只会记录 invalid_input。"
            )


def _build_baseline_payload(
    frame,
    detector: MediaPipeHandDetector,
    cfg: Dict[str, Any],
    stabilizer: GestureStabilizer,
    *,
    draw_landmarks: bool,
) -> Dict[str, Any]:
    """从单帧图像生成 baseline payload。

    这一层只做视觉 baseline：检测右手、提取几何特征、判断并稳定手势。
    control_representation 和 svh_preview 在后续扩展链路里再补上。
    """

    detections = detector.detect(frame)
    right = select_right_hand(detections)
    ts = now_ts()

    if right is None:
        # 没有右手时仍返回规范形状，保证下游不会因为字段缺失崩掉。
        payload = empty_features(ts)
    else:
        payload = extract_hand_features(
            right.landmarks_2d,
            right.handedness,
            right.confidence,
            ts,
            landmarks_xyz=right.landmarks_xyz,
        )
        quality = assess_control_readiness(right.landmarks_2d, cfg)
        if not bool(quality["control_ready"]):
            # 低质量帧仍保留 detected / landmarks，便于调试；
            # 但清空面向控制的连续特征，避免下游误用不稳定数据。
            payload = invalidate_control_features(payload)
        if draw_landmarks:
            detector.draw_landmarks(frame, right.landmarks_2d)

    payload["gesture_raw"] = infer_gesture_raw(payload, cfg)
    payload["gesture_stable"] = stabilizer.update(payload["gesture_raw"])
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


def _unix_ms() -> float:
    """返回 Unix epoch 毫秒；用于同机 Python/Unity 阶段诊断。"""

    return time.time() * 1000.0


def _send_prepared_udp(payload: Dict[str, Any], *, exporter: JsonExporter) -> None:
    """优先发送 frozen payload；影子预测必须排在这个函数之后。"""

    if exporter.unity_udp_enabled:
        timing = payload.get("timing")
        if isinstance(timing, dict):
            timing["udp_send_attempt_unix_ms"] = _unix_ms()
            assert_valid_frame_payload(payload)
        exporter.send_prepared_frame(payload)


def _emit_prepared_debug_outputs(
    payload: Dict[str, Any],
    *,
    frame_index: int,
    exporter: JsonExporter,
    print_json: bool,
    print_every_n_frames: int,
    landmarks_preview_count: int,
) -> None:
    """在 UDP 之后输出控制台和 JSON/JSONL。"""

    if print_json and frame_index % print_every_n_frames == 0:
        exporter.print_console(payload, landmarks_preview_count=landmarks_preview_count)
    exporter.export_prepared_frame(payload, frame_index=frame_index)


def _emit_prepared_frame(
    payload: Dict[str, Any],
    *,
    frame_index: int,
    exporter: JsonExporter,
    print_json: bool,
    print_every_n_frames: int,
    landmarks_preview_count: int,
) -> None:
    """兼容入口：按实时优先级先 UDP，再控制台，最后落盘。

    UDP 必须排在控制台和 JSON/JSONL I/O 前面，避免调试输出把 Unity 预览
    无谓推迟。实时影子模式另用后台 worker，不经过这个同步兼容入口。
    """

    _send_prepared_udp(payload, exporter=exporter)
    _emit_prepared_debug_outputs(
        payload,
        frame_index=frame_index,
        exporter=exporter,
        print_json=print_json,
        print_every_n_frames=print_every_n_frames,
        landmarks_preview_count=landmarks_preview_count,
    )


def _drain_prediction_results(
    worker: PredictionShadowWorker | None,
    exporter: JsonExporter | None,
) -> int:
    """把后台已完成诊断写入独立本机文件；调用本身不等待推理。"""

    if worker is None or exporter is None:
        return 0
    results = worker.drain_results()
    for result in results:
        exporter.export_prepared_frame(
            result,
            frame_index=int(result["frame_index"]),
        )
    return len(results)


def _apply_extension_chain(
    payload: Dict[str, Any],
    cfg: Dict[str, Any],
    runtime: RuntimeMode,
    *,
    svh_transport,
    logger,
) -> ExtensionDiagnostics:
    """依次运行可选扩展层，并在失败时退回规范占位对象。

    设计原则：
    - baseline 视觉链路优先保持可运行；
    - control / SVH preview 都是可选层；
    - 扩展失败时 payload 仍要满足 frozen contract。
    """

    diagnostics: ExtensionDiagnostics = []
    if runtime.control_extension_enabled:
        try:
            control_representation = build_control_representation(payload, cfg)
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
        if svh_transport is not None and svh_preview.get("valid"):
            try:
                svh_transport.send(svh_preview)
            except Exception as exc:
                _record_extension_failure(
                    diagnostics,
                    extension_name="svh_transport",
                    exc=exc,
                    logger=logger,
                    fallback_summary="将保留预览 payload，但跳过这一帧的传输发送。",
                )
    else:
        svh_preview = empty_svh_preview(cfg, enabled=False, mode="disabled")

    payload["svh_preview"] = svh_preview
    return diagnostics


def main() -> None:
    """实时运行主循环。"""

    args = parse_args()
    cfg = _apply_cli_overrides(load_config(args.config), args)
    runtime = _build_runtime_mode(cfg)
    logger = get_logger()

    _log_runtime_mode(runtime, cfg, logger)

    source = _build_input_source(cfg, runtime, logger)
    if source is None:
        return

    detector = None
    exporter = None
    prediction_worker = None
    prediction_result_exporter = None
    try:
        detector = _build_detector(cfg, runtime)
        exporter = _build_exporter(cfg, logger)
        prediction_shadow = build_prediction_shadow(cfg, logger=logger)
        if prediction_shadow is not None:
            prediction_worker = PredictionShadowWorker(
                prediction_shadow,
                logger=logger,
                input_queue_size=1,
                result_queue_size=int(cfg.get("prediction_shadow_result_queue_size", 8)),
            )
            prediction_result_exporter = _build_prediction_result_exporter(cfg, logger)
        # history 当前只保留最近帧摘要，方便未来做时序模型或更复杂去抖。
        history = RecentFrameBuffer(maxlen=int(cfg.get("recent_frames_buffer_size", 10)))
        svh_transport = _build_svh_transport(cfg, runtime, logger)
        stabilizer = GestureStabilizer(
            confirm_frames=int(cfg.get("stable_gesture_min_consecutive", 2)),
            unknown_confirm_frames=int(cfg.get("stable_unknown_consecutive", 1)),
        )
        print_every_n_frames = max(1, int(cfg.get("console_print_every_n_frames", 5)))
        landmarks_preview_count = max(0, int(cfg.get("console_landmarks_preview_count", 3)))
        draw_landmarks = runtime.gui_enabled and bool(cfg.get("draw_landmarks", True))

        timer = FrameTimer()
        frame_index = 0

        while True:
            source_read_start_unix_ms = _unix_ms()
            ok, frame = source.read()
            source_read_end_unix_ms = _unix_ms()
            if not ok or frame is None:
                if frame_index == 0:
                    logger.warning("输入源没有产出任何帧；程序将安全退出。")
                else:
                    logger.info("输入源已耗尽，共处理 %d 帧。", frame_index)
                break

            t0 = time.perf_counter()
            # 1. 先生成纯视觉 baseline payload。
            payload = _build_baseline_payload(
                frame,
                detector,
                cfg,
                stabilizer,
                draw_landmarks=draw_landmarks,
            )
            baseline_end_unix_ms = _unix_ms()
            # 2. 再按运行模式追加 control / SVH preview 扩展字段。
            _apply_extension_chain(
                payload,
                cfg,
                runtime,
                svh_transport=svh_transport,
                logger=logger,
            )
            preview_end_unix_ms = _unix_ms()
            payload["frame_index"] = frame_index
            dt = timer.tick()
            payload["fps"] = 1.0 / dt if dt > 1e-6 else 0.0
            payload["latency_ms"] = (time.perf_counter() - t0) * 1000.0
            payload["timing"] = {
                "schema_version": 1,
                "clock": "unix_epoch_ms",
                "source_read_start_unix_ms": source_read_start_unix_ms,
                "source_read_end_unix_ms": source_read_end_unix_ms,
                # _build_baseline_payload 在 detect + 右手筛选后生成 timestamp。
                "detection_end_unix_ms": float(payload["timestamp"]) * 1000.0,
                "baseline_end_unix_ms": baseline_end_unix_ms,
                "preview_end_unix_ms": preview_end_unix_ms,
                "payload_ready_unix_ms": _unix_ms(),
                "udp_send_attempt_unix_ms": None,
            }
            # 3. 最后统一规范化并校验，保证输出满足 contract。
            payload = prepare_frame_payload(payload, include_deprecated_aliases=False)
            history.append(payload)

            # 4. frozen UDP 先发送；影子推理永远不能推迟或改写 Unity 包。
            _send_prepared_udp(payload, exporter=exporter)
            # 5. 只做 latest-only 非阻塞提交；当前帧 baseline 照常立即落盘/显示。
            if prediction_worker is not None:
                prediction_worker.submit(payload)
            _emit_prepared_debug_outputs(
                payload,
                frame_index=frame_index,
                exporter=exporter,
                print_json=bool(args.print_json),
                print_every_n_frames=print_every_n_frames,
                landmarks_preview_count=landmarks_preview_count,
            )
            _drain_prediction_results(
                prediction_worker,
                prediction_result_exporter,
            )

            if runtime.gui_enabled:
                view = compose_view(frame, payload)
                cv2.imshow("single-right-hand-teleop-baseline", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            if args.max_frames is not None and frame_index >= args.max_frames:
                logger.info("已达到 max_frames=%d；程序退出。", args.max_frames)
                break
    except KeyboardInterrupt:
        logger.info("用户中断运行。")
    finally:
        if prediction_worker is not None:
            prediction_worker.close(
                timeout_s=float(cfg.get("prediction_shadow_worker_shutdown_timeout_s", 2.0))
            )
            _drain_prediction_results(
                prediction_worker,
                prediction_result_exporter,
            )
        if prediction_result_exporter is not None:
            prediction_result_exporter.close()
        if exporter is not None:
            exporter.close()
        if detector is not None:
            detector.close()
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
