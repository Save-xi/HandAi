from __future__ import annotations

"""单右手遥操作 baseline 的运行入口。

这个文件负责把各个模块串成一条实时 pipeline：

摄像头/视频 -> MediaPipe 检测 -> 右手筛选 -> 特征提取 -> 手势稳定
-> 可选 control 表示 -> 可选 SVH preview -> JSON/JSONL/UDP/GUI 输出。

注意：这里尽量只做“编排”。具体算法逻辑分别放在 perception、
features、gesture、control、svh、output 等子模块里，方便单独测试和替换。
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import cv2

from capture.input_source import InputSource
from capture.video_file import VideoFileSource
from capture.webcam import WebcamSource
from output.frame_payload_contract import assert_valid_frame_payload, prepare_frame_payload
from output.json_exporter import JsonExporter
from perception.mediapipe_hand import MediaPipeHandDetector
from pipeline import HandPipeline, RuntimeMode, _build_runtime_mode
from prediction.shadow_predictor import build_prediction_shadow
from prediction.shadow_worker import PredictionShadowWorker
from utils.config import load_config
from utils.logger import get_logger
from utils.runtime_session import (
    RuntimeSessionRecorder,
    build_jsonl_session_path,
    create_runtime_session_artifacts,
)
from utils.timer import FrameTimer
from visualize.overlay_2d import compose_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单右手 AI：姿态、手势、连续表示与可选预测")
    parser.add_argument("--config", default="configs/ai.yaml", type=str)
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
    parser.add_argument("--prediction-model", help="预测模型 JSON 配置路径")
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
    if getattr(args, "prediction_model", None):
        cfg["prediction_shadow_model_path"] = _resolve_user_path(args.prediction_model)
    return cfg


def _build_jsonl_session_path(
    cfg: Dict[str, Any],
    *,
    output_dir_key: str = "jsonl_output_dir",
    prefix: str = "session",
    run_id: str | None = None,
) -> str:
    """为本次运行生成 JSONL 会话日志路径。"""

    return build_jsonl_session_path(
        cfg,
        output_dir_key=output_dir_key,
        prefix=prefix,
        run_id=run_id,
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


def _build_exporter(
    cfg: Dict[str, Any],
    logger,
    *,
    jsonl_path: str | None = None,
) -> JsonExporter:
    """创建统一导出器。

    JsonExporter 同时负责 last-frame JSON、可选 JSONL、可选 Unity UDP。
    主循环只把规范化后的 payload 交给它，不关心具体输出通道。
    """

    resolved_jsonl_path = None
    if bool(cfg.get("save_jsonl", False)):
        resolved_jsonl_path = jsonl_path or _build_jsonl_session_path(cfg)

    return JsonExporter(
        output_path=str(cfg.get("output_json_path", "outputs/latest_frame.json")),
        save_last_json=bool(cfg.get("save_last_json", True)),
        jsonl_path=resolved_jsonl_path,
        export_last_every_n_frames=int(cfg.get("export_last_every_n_frames", 1)),
        jsonl_flush_interval=int(cfg.get("jsonl_flush_interval", 1)),
        unity_udp_enabled=bool(cfg.get("unity_udp_enabled", False)),
        unity_udp_host=str(cfg.get("unity_udp_host", "127.0.0.1")),
        unity_udp_port=int(cfg.get("unity_udp_port", 18080)),
        logger=logger,
    )


def _build_prediction_result_exporter(
    cfg: Dict[str, Any],
    logger,
    *,
    jsonl_path: str | None = None,
) -> JsonExporter:
    """为后台影子结果创建独立本机输出；绝不启用 UDP。"""

    resolved_jsonl_path = None
    if bool(cfg.get("save_jsonl", False)):
        resolved_jsonl_path = jsonl_path or _build_jsonl_session_path(
            cfg,
            output_dir_key="prediction_shadow_jsonl_output_dir",
            prefix="prediction_session",
        )

    return JsonExporter(
        output_path=str(
            cfg.get(
                "prediction_shadow_output_json_path",
                "outputs/latest_prediction_shadow.json",
            )
        ),
        save_last_json=bool(cfg.get("save_last_json", True)),
        jsonl_path=resolved_jsonl_path,
        export_last_every_n_frames=1,
        jsonl_flush_interval=int(cfg.get("jsonl_flush_interval", 1)),
        unity_udp_enabled=False,
        logger=logger,
    )


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
    prediction_shadow = None
    session_recorder = None
    session_artifacts = None
    session_status = "completed"
    session_error: BaseException | None = None
    prediction_worker_stopped: bool | None = None
    if bool(cfg.get("save_jsonl", False)):
        session_artifacts = create_runtime_session_artifacts(
            cfg,
            prediction_requested=bool(cfg.get("prediction_shadow_enabled", False)),
        )
        try:
            session_recorder = RuntimeSessionRecorder(
                session_artifacts,
                config_path=Path(args.config),
                cfg=cfg,
                runtime=runtime,
            )
            logger.info(
                "运行会话已创建：run_id=%s，manifest=%s。",
                session_artifacts.run_id,
                session_artifacts.manifest_path,
            )
        except OSError as exc:
            logger.warning("无法创建运行 manifest；逐帧链路继续运行：%s", exc)
    try:
        detector = _build_detector(cfg, runtime)
        exporter = _build_exporter(
            cfg,
            logger,
            jsonl_path=(
                str(session_artifacts.baseline_jsonl_path)
                if session_artifacts is not None
                else None
            ),
        )
        prediction_shadow = build_prediction_shadow(cfg, logger=logger)
        if session_recorder is not None:
            try:
                session_recorder.record_prediction_identity(prediction_shadow)
            except OSError as exc:
                logger.warning("更新运行 manifest 失败；逐帧链路继续运行：%s", exc)
                session_recorder = None
        if prediction_shadow is not None:
            prediction_worker = PredictionShadowWorker(
                prediction_shadow,
                logger=logger,
                input_queue_size=1,
                result_queue_size=int(cfg.get("prediction_shadow_result_queue_size", 8)),
            )
            prediction_result_exporter = _build_prediction_result_exporter(
                cfg,
                logger,
                jsonl_path=(
                    str(session_artifacts.prediction_jsonl_path)
                    if session_artifacts is not None
                    and session_artifacts.prediction_jsonl_path is not None
                    else None
                ),
            )
        pipeline = HandPipeline(cfg, detector=detector, logger=logger)
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
            payload = pipeline.process_frame(
                frame, frame_index=frame_index, draw_landmarks=draw_landmarks,
            )
            baseline_end_unix_ms = pipeline.last_stage_timing["baseline_end_unix_ms"]
            preview_end_unix_ms = pipeline.last_stage_timing["preview_end_unix_ms"]
            payload["frame_index"] = frame_index
            dt = timer.tick()
            payload["fps"] = 1.0 / dt if dt > 1e-6 else 0.0
            payload["latency_ms"] = (time.perf_counter() - t0) * 1000.0
            payload["timing"] = {
                "schema_version": 1,
                "clock": "unix_epoch_ms",
                "source_read_start_unix_ms": source_read_start_unix_ms,
                "source_read_end_unix_ms": source_read_end_unix_ms,
                # pipeline 在检测完成后生成 timestamp。
                "detection_end_unix_ms": float(payload["timestamp"]) * 1000.0,
                "baseline_end_unix_ms": baseline_end_unix_ms,
                "preview_end_unix_ms": preview_end_unix_ms,
                "payload_ready_unix_ms": _unix_ms(),
                "udp_send_attempt_unix_ms": None,
            }
            # 3. 最后统一规范化并校验，保证输出满足 contract。
            payload = prepare_frame_payload(payload, include_deprecated_aliases=False)
            if session_recorder is not None:
                session_recorder.observe_baseline(payload)

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
        session_status = "interrupted"
        logger.info("用户中断运行。")
    except BaseException as exc:
        session_status = "failed"
        session_error = exc
        raise
    finally:
        if prediction_worker is not None:
            prediction_worker_stopped = prediction_worker.close(
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
        if session_recorder is not None:
            try:
                session_recorder.finalize(
                    status=session_status,
                    error=session_error,
                    baseline_exporter=exporter,
                    prediction_exporter=prediction_result_exporter,
                    prediction_worker=prediction_worker,
                    prediction_worker_stopped=prediction_worker_stopped,
                )
                logger.info("运行会话信息已保存：%s。", session_recorder.manifest_path)
            except OSError as exc:
                logger.warning("保存运行 manifest 失败：%s", exc)
        if detector is not None:
            detector.close()
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
