from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cv2

from capture.input_source import InputSource
from capture.video_file import VideoFileSource
from capture.webcam import WebcamSource
from features.hand_features import empty_features, extract_hand_features, invalidate_control_features
from gesture.rule_based_gesture import GestureStabilizer, infer_gesture_raw
from output.frame_payload_contract import normalize_frame_payload
from output.json_exporter import JsonExporter
from perception.hand_filter import select_right_hand
from perception.landmark_quality import assess_control_readiness
from perception.mediapipe_hand import MediaPipeHandDetector
from utils.config import load_config
from utils.logger import get_logger
from utils.recent_frames import RecentFrameBuffer
from utils.timer import FrameTimer, now_ts
from visualize.overlay_2d import compose_view


@dataclass(frozen=True)
class RuntimeMode:
    gui_enabled: bool
    headless: bool
    input_source_type: str
    input_mirrored: bool
    control_extension_enabled: bool
    svh_preview_enabled: bool
    video_file_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single right hand teleop baseline demo")
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    parser.add_argument("--camera-index", default=None, type=int, help="Override webcam camera index")
    parser.add_argument("--video-file", default=None, type=str, help="Read frames from a local video file instead of a webcam")
    parser.add_argument("--input-mirrored", action="store_true", help="Treat input as already mirrored/selfie-style")
    parser.add_argument("--enable-control", action="store_true", help="Enable the control_representation extension layer")
    parser.add_argument("--preview-svh", action="store_true", help="Enable the SVH preview extension layer (implies --enable-control)")
    parser.add_argument("--no-gui", action="store_true", help="Disable the OpenCV preview window but keep live processing")
    parser.add_argument("--headless", action="store_true", help="Run without any OpenCV window; useful for logs, JSONL, or video-file processing")
    parser.add_argument("--max-frames", default=None, type=int, help="Stop after processing N frames")
    parser.add_argument("--print-json", action="store_true", help="Print frame JSON to console")
    parser.add_argument("--save-jsonl", action="store_true", help="Enable per-frame JSONL logging for this session")
    return parser.parse_args()


def _resolve_user_path(path_str: str) -> str:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path.cwd() / candidate).resolve())


def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
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
    return cfg


def _build_runtime_mode(cfg: Dict[str, Any]) -> RuntimeMode:
    svh_preview_enabled = bool(cfg.get("svh_enable_preview", False))
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


def _build_jsonl_session_path(cfg: Dict[str, Any]) -> str:
    output_dir = Path(str(cfg.get("jsonl_output_dir", "outputs")))
    return str(output_dir / f"session_{datetime.now():%Y%m%d_%H%M%S}.jsonl")


def _build_input_source(cfg: Dict[str, Any], runtime: RuntimeMode, logger) -> InputSource | None:
    source_type = runtime.input_source_type
    if source_type == "video_file":
        if not runtime.video_file_path:
            logger.error("Video-file mode requested but no video path was provided; exiting gracefully.")
            return None
        source = VideoFileSource(runtime.video_file_path)
        if not source.is_opened():
            logger.error("Unable to open video file '%s'; exiting gracefully.", runtime.video_file_path)
            source.release()
            return None
        logger.info("Using video file input: %s", runtime.video_file_path)
        return source

    if source_type != "webcam":
        logger.warning("Unsupported input_source_type '%s'; falling back to webcam.", source_type)

    camera_index = int(cfg.get("camera_index", 0))
    source = WebcamSource(
        camera_index=camera_index,
        width=int(cfg["display_width"]),
        height=int(cfg["display_height"]),
    )
    if not source.is_opened():
        logger.error("Unable to open webcam at camera_index=%s; exiting gracefully.", camera_index)
        source.release()
        return None
    logger.info("Using webcam input at camera_index=%s.", camera_index)
    return source


def _build_detector(cfg: Dict[str, Any], runtime: RuntimeMode) -> MediaPipeHandDetector:
    return MediaPipeHandDetector(
        max_num_hands=int(cfg.get("max_num_hands", 2)),
        min_detection_confidence=float(cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(cfg.get("min_tracking_confidence", 0.5)),
        input_mirrored=runtime.input_mirrored,
    )


def _build_exporter(cfg: Dict[str, Any], logger) -> JsonExporter:
    return JsonExporter(
        output_path=str(cfg.get("output_json_path", "examples/sample_output.json")),
        save_last_json=bool(cfg.get("save_last_json", True)),
        jsonl_path=_build_jsonl_session_path(cfg) if bool(cfg.get("save_jsonl", False)) else None,
        logger=logger,
    )


def _build_svh_transport(cfg: Dict[str, Any], runtime: RuntimeMode, logger):
    if not runtime.svh_preview_enabled:
        return None
    svh_transport_name = str(cfg.get("svh_transport", "mock"))
    if svh_transport_name == "mock":
        from svh.svh_transport_mock import MockSvhTransport

        logger.info("SVH preview extension enabled with mock transport.")
        return MockSvhTransport(logger=logger)
    logger.warning(
        "Unsupported SVH transport '%s'; continuing in preview-only mode without transport send.",
        svh_transport_name,
    )
    return None


def _log_runtime_mode(runtime: RuntimeMode, cfg: Dict[str, Any], logger) -> None:
    if runtime.gui_enabled:
        logger.info("GUI enabled. Press q in the OpenCV window to quit.")
    else:
        logger.info("GUI disabled; running headless.")

    if runtime.input_mirrored:
        logger.info("Input is treated as mirrored/selfie-style.")

    if not runtime.control_extension_enabled and not runtime.svh_preview_enabled:
        logger.info("Running in baseline-only mode; control and SVH extensions are disabled.")
    elif runtime.control_extension_enabled and not runtime.svh_preview_enabled:
        logger.info("Control extension enabled; SVH preview extension remains disabled.")
    elif runtime.svh_preview_enabled:
        logger.info("SVH preview extension is enabled in preview-only mode.")

    if bool(cfg.get("save_jsonl", False)):
        logger.info("Per-frame JSONL logging is enabled for this session.")


def _build_baseline_payload(frame, detector: MediaPipeHandDetector, cfg: Dict[str, Any], stabilizer: GestureStabilizer, *, draw_landmarks: bool) -> Dict[str, Any]:
    detections = detector.detect(frame)
    right = select_right_hand(detections)
    ts = now_ts()

    if right is None:
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
            payload = invalidate_control_features(payload)
        if draw_landmarks:
            detector.draw_landmarks(frame, right.landmarks_2d)

    payload["gesture_raw"] = infer_gesture_raw(payload, cfg)
    payload["gesture_stable"] = stabilizer.update(payload["gesture_raw"])
    return payload


def _empty_control_representation_payload() -> Dict[str, Any]:
    return {
        "valid": False,
        "features_valid": False,
        "command_ready": False,
        "source": None,
        "gesture_context": None,
        "preferred_mapping": None,
        "grasp_close": None,
        "thumb_index_proximity": None,
        "effective_pinch_strength": None,
        "pinch_strength": None,
        "support_flex": None,
        "finger_flex": {
            "thumb": None,
            "index": None,
            "middle": None,
            "ring": None,
            "little": None,
        },
    }


def _empty_svh_preview_payload(cfg: Dict[str, Any], *, enabled: bool, mode: str) -> Dict[str, Any]:
    channel_layout = str(cfg.get("svh_preview_layout", "compact5"))
    channel_order = (
        "thumb_flexion,thumb_opposition,index_finger_distal,index_finger_proximal,middle_finger_distal,middle_finger_proximal,ring_finger,pinky,finger_spread"
        if channel_layout == "svh_9ch"
        else "thumb,index,middle,ring,little"
    )
    return {
        "enabled": enabled,
        "mode": mode,
        "valid": False,
        "command_source": None,
        "target_channels": [],
        "target_positions": [],
        "target_ticks_preview": [],
        "protocol_hint": {
            "set_control_state_addr": "0x09",
            "set_all_channels_addr": "0x03",
            "transport": str(cfg.get("svh_transport", "mock")),
            "channel_layout": channel_layout,
            "channel_order": channel_order,
            "position_units": "normalized_preview",
            "target_tick_units": "encoder_ticks_preview" if channel_layout == "svh_9ch" else "none",
        },
    }


def _apply_extension_chain(payload: Dict[str, Any], cfg: Dict[str, Any], runtime: RuntimeMode, *, svh_transport, logger) -> None:
    if runtime.control_extension_enabled:
        try:
            from control.control_representation import build_control_representation, empty_control_representation

            control_representation = build_control_representation(payload, cfg)
        except Exception:
            logger.exception("control_representation extension failed; continuing with baseline-only payload for this frame.")
            try:
                from control.control_representation import empty_control_representation

                control_representation = empty_control_representation()
            except Exception:
                control_representation = _empty_control_representation_payload()
    else:
        control_representation = _empty_control_representation_payload()

    payload["control_representation"] = control_representation
    payload["control_ready"] = bool(control_representation.get("command_ready", False))

    if runtime.svh_preview_enabled:
        try:
            from svh.svh_adapter import build_svh_command_preview, empty_svh_preview

            svh_preview = build_svh_command_preview(payload, cfg)
        except Exception:
            logger.exception("svh_preview extension failed; continuing without SVH preview output for this frame.")
            try:
                from svh.svh_adapter import empty_svh_preview

                svh_preview = empty_svh_preview(cfg, enabled=True, mode=str(cfg.get("svh_preview_mode", "preview")))
            except Exception:
                svh_preview = _empty_svh_preview_payload(cfg, enabled=True, mode=str(cfg.get("svh_preview_mode", "preview")))
        if svh_transport is not None and svh_preview.get("valid"):
            try:
                svh_transport.send(svh_preview)
            except Exception:
                logger.exception("SVH transport send failed; keeping preview output but skipping transport for this frame.")
    else:
        svh_preview = _empty_svh_preview_payload(cfg, enabled=False, mode="disabled")

    payload["svh_preview"] = svh_preview


def main() -> None:
    args = parse_args()
    cfg = _apply_cli_overrides(load_config(args.config), args)
    runtime = _build_runtime_mode(cfg)
    logger = get_logger()

    _log_runtime_mode(runtime, cfg, logger)

    source = _build_input_source(cfg, runtime, logger)
    if source is None:
        return

    detector = None
    try:
        detector = _build_detector(cfg, runtime)
        exporter = _build_exporter(cfg, logger)
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
            ok, frame = source.read()
            if not ok or frame is None:
                if frame_index == 0:
                    logger.warning("Input source did not yield any frames; exiting gracefully.")
                else:
                    logger.info("Input source exhausted after %d processed frames.", frame_index)
                break

            t0 = time.perf_counter()
            payload = _build_baseline_payload(
                frame,
                detector,
                cfg,
                stabilizer,
                draw_landmarks=draw_landmarks,
            )
            _apply_extension_chain(
                payload,
                cfg,
                runtime,
                svh_transport=svh_transport,
                logger=logger,
            )
            payload["frame_index"] = frame_index
            dt = timer.tick()
            payload["fps"] = 1.0 / dt if dt > 1e-6 else 0.0
            payload["latency_ms"] = (time.perf_counter() - t0) * 1000.0
            payload = normalize_frame_payload(payload, include_deprecated_aliases=False)
            history.append(payload)

            if args.print_json and frame_index % print_every_n_frames == 0:
                exporter.print_console(payload, landmarks_preview_count=landmarks_preview_count)
            exporter.save_last_frame(payload)
            exporter.append_jsonl(payload)
            exporter.send(payload)

            if runtime.gui_enabled:
                view = compose_view(frame, payload)
                cv2.imshow("single-right-hand-teleop-baseline", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            if args.max_frames is not None and frame_index >= args.max_frames:
                logger.info("Reached max_frames=%d; exiting.", args.max_frames)
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if detector is not None:
            detector.close()
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
