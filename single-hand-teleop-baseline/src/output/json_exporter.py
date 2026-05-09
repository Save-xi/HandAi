from __future__ import annotations

import json
import logging
import socket
from pathlib import Path
from typing import Any, Dict, TextIO

from output.frame_payload_contract import prepare_frame_payload


class JsonExporter:
    def __init__(
        self,
        output_path: str,  
        save_last_json: bool = True,
        jsonl_path: str | None = None,
        export_last_every_n_frames: int = 1,
        jsonl_flush_interval: int = 1,
        unity_udp_enabled: bool = True,
        unity_udp_host: str = "127.0.0.1",
        unity_udp_port: int = 18080,
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_path = output_path
        self.save_last_json = save_last_json
        self.jsonl_path = jsonl_path
        self.export_last_every_n_frames = max(1, int(export_last_every_n_frames))
        self.jsonl_flush_interval = max(1, int(jsonl_flush_interval))
        self.unity_udp_enabled = bool(unity_udp_enabled)
        self.unity_udp_host = str(unity_udp_host)
        self.unity_udp_port = int(unity_udp_port)
        self.logger = logger
        self._jsonl_failed = False
        self._jsonl_handle: TextIO | None = None
        self._unity_udp_failed = False
        self._unity_udp_socket: socket.socket | None = None
        self._jsonl_pending_lines = 0
        self._last_frame_dirty = False
        self._latest_prepared_payload: Dict[str, Any] | None = None
        self._last_frame_write_count = 0
        self._jsonl_write_count = 0
        self._jsonl_flush_count = 0
        self._unity_udp_send_count = 0

    def to_json_str(self, obj: Dict[str, Any]) -> str:
        prepared = prepare_frame_payload(obj, include_deprecated_aliases=False)
        return json.dumps(prepared, ensure_ascii=False, indent=2)

    def _round_value(self, value: Any, ndigits: int = 3) -> Any:
        if isinstance(value, float):
            return round(value, ndigits)
        if isinstance(value, list):
            return [self._round_value(item, ndigits=ndigits) for item in value]
        if isinstance(value, dict):
            return {key: self._round_value(item, ndigits=ndigits) for key, item in value.items()}
        return value

    def to_console_obj(self, obj: Dict[str, Any], landmarks_preview_count: int = 3) -> Dict[str, Any]:
        obj = prepare_frame_payload(obj, include_deprecated_aliases=False)
        console_obj: Dict[str, Any] = {}
        for key, value in obj.items():
            if key == "landmarks_2d":
                console_obj["landmarks_count"] = len(value)
                console_obj["landmarks_2d_preview"] = value[:landmarks_preview_count]
                continue
            if key == "landmarks_3d":
                console_obj["landmarks_3d_count"] = len(value)
                console_obj["landmarks_3d_preview"] = value[:landmarks_preview_count]
                continue
            if key == "svh_preview" and isinstance(value, dict):
                svh_preview = dict(value)
                positions = list(svh_preview.get("target_positions", []))
                ticks = list(svh_preview.get("target_ticks_preview", []))
                svh_preview["target_positions_count"] = len(positions)
                svh_preview["target_positions_preview"] = positions[:landmarks_preview_count]
                svh_preview["target_ticks_count"] = len(ticks)
                svh_preview["target_ticks_preview_short"] = ticks[:landmarks_preview_count]
                svh_preview.pop("target_positions", None)
                svh_preview.pop("target_ticks_preview", None)
                console_obj[key] = svh_preview
                continue
            console_obj[key] = value
        return self._round_value(console_obj)

    def print_console(self, obj: Dict[str, Any], landmarks_preview_count: int = 3) -> None:
        print(json.dumps(self.to_console_obj(obj, landmarks_preview_count=landmarks_preview_count), ensure_ascii=False))

    def _warn(self, message: str) -> None:
        if self.logger is not None:
            self.logger.warning(message)

    def _debug(self, message: str, *args: Any) -> None:
        if self.logger is not None:
            self.logger.debug(message, *args)

    def _write_last_prepared_frame(self, prepared: Dict[str, Any]) -> None:
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(prepared, f, ensure_ascii=False, indent=2)
        self._last_frame_write_count += 1
        self._last_frame_dirty = False

    def _ensure_jsonl_handle(self) -> TextIO | None:
        if not self.jsonl_path or self._jsonl_failed:
            return None
        if self._jsonl_handle is None:
            path = Path(self.jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_handle = path.open("a", encoding="utf-8")
        return self._jsonl_handle

    def _flush_jsonl_handle(self) -> None:
        if self._jsonl_handle is None:
            return
        self._jsonl_handle.flush()
        self._jsonl_flush_count += 1
        self._jsonl_pending_lines = 0

    def _ensure_unity_udp_socket(self) -> socket.socket | None:
        if not self.unity_udp_enabled or self._unity_udp_failed:
            return None
        if self._unity_udp_socket is None:
            self._unity_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return self._unity_udp_socket

    def _append_prepared_jsonl(self, prepared: Dict[str, Any], *, force_flush: bool) -> None:
        handle = self._ensure_jsonl_handle()
        if handle is None:
            return
        handle.write(json.dumps(prepared, ensure_ascii=False))
        handle.write("\n")
        self._jsonl_write_count += 1
        self._jsonl_pending_lines += 1
        if force_flush or self._jsonl_pending_lines >= self.jsonl_flush_interval:
            self._flush_jsonl_handle()

    def save_last_frame(self, obj: Dict[str, Any]) -> None:
        if not self.save_last_json:
            return
        try:
            prepared = prepare_frame_payload(obj, include_deprecated_aliases=False)
            self._latest_prepared_payload = prepared
            self._last_frame_dirty = True
            self._write_last_prepared_frame(prepared)
        except OSError as exc:
            self._warn(f"保存最后一帧 JSON 失败：{exc}")

    def append_jsonl(self, obj: Dict[str, Any]) -> None:
        if not self.jsonl_path or self._jsonl_failed:
            return
        try:
            prepared = prepare_frame_payload(obj, include_deprecated_aliases=False)
            self._append_prepared_jsonl(prepared, force_flush=True)
        except OSError as exc:
            self._jsonl_failed = True
            if self._jsonl_handle is not None:
                self._jsonl_handle.close()
                self._jsonl_handle = None
            self._warn(f"追加 JSONL 日志失败；本次运行将停用 JSONL：{exc}")

    def export_prepared_frame(self, prepared: Dict[str, Any], *, frame_index: int) -> None:
        if self.save_last_json:
            self._latest_prepared_payload = prepared
            self._last_frame_dirty = True
            if frame_index % self.export_last_every_n_frames == 0:
                try:
                    self._write_last_prepared_frame(prepared)
                except OSError as exc:
                    self._warn(f"保存最后一帧 JSON 失败：{exc}")
        if self.jsonl_path and not self._jsonl_failed:
            try:
                self._append_prepared_jsonl(prepared, force_flush=False)
            except OSError as exc:
                self._jsonl_failed = True
                if self._jsonl_handle is not None:
                    self._jsonl_handle.close()
                    self._jsonl_handle = None
                self._warn(f"追加 JSONL 日志失败；本次运行将停用 JSONL：{exc}")

    def send(self, obj: Dict[str, Any]) -> None:
        """为未来网络 / 控制集成预留的输出接口。"""
        prepared = prepare_frame_payload(obj, include_deprecated_aliases=False)
        self.send_prepared_frame(prepared)

    def send_prepared_frame(self, prepared: Dict[str, Any]) -> None:
        if not self.unity_udp_enabled:
            return
        if self._unity_udp_failed:
            return

        udp_socket = self._ensure_unity_udp_socket()
        print (f"UDP Socket: {udp_socket}")
        if udp_socket is None:
            return

        try:
            payload = json.dumps(prepared, ensure_ascii=False).encode("utf-8")
            suc = udp_socket.sendto(payload, (self.unity_udp_host, self.unity_udp_port))
            self._unity_udp_send_count += 1
        except OSError as exc:
            self._unity_udp_failed = True
            if self._unity_udp_socket is not None:
                self._unity_udp_socket.close()
                self._unity_udp_socket = None
            self._warn(f"发送 Unity UDP 预览数据失败；本次运行将停用该通道：{exc}")

    def close(self) -> None:
        if self.save_last_json and self._last_frame_dirty and self._latest_prepared_payload is not None:
            try:
                self._write_last_prepared_frame(self._latest_prepared_payload)
            except OSError as exc:
                self._warn(f"关闭导出器时保存最后一帧 JSON 失败：{exc}")

        if self._jsonl_handle is not None:
            try:
                if self._jsonl_pending_lines > 0:
                    self._flush_jsonl_handle()
            finally:
                self._jsonl_handle.close()
                self._jsonl_handle = None

        if self._unity_udp_socket is not None:
            self._unity_udp_socket.close()
            self._unity_udp_socket = None

        self._debug(
            "导出器摘要：last-frame 写入=%d，jsonl 行数=%d，jsonl flush 次数=%d，Unity UDP 发送次数=%d",
            self._last_frame_write_count,
            self._jsonl_write_count,
            self._jsonl_flush_count,
            self._unity_udp_send_count,
        )
