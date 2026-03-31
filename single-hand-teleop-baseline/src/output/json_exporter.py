from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict


class JsonExporter:
    def __init__(
        self,
        output_path: str,
        save_last_json: bool = True,
        jsonl_path: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_path = output_path
        self.save_last_json = save_last_json
        self.jsonl_path = jsonl_path
        self.logger = logger
        self._jsonl_failed = False

    def to_json_str(self, obj: Dict[str, Any]) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2)

    def _round_value(self, value: Any, ndigits: int = 3) -> Any:
        if isinstance(value, float):
            return round(value, ndigits)
        if isinstance(value, list):
            return [self._round_value(item, ndigits=ndigits) for item in value]
        if isinstance(value, dict):
            return {key: self._round_value(item, ndigits=ndigits) for key, item in value.items()}
        return value

    def to_console_obj(self, obj: Dict[str, Any], landmarks_preview_count: int = 3) -> Dict[str, Any]:
        console_obj: Dict[str, Any] = {}
        for key, value in obj.items():
            if key == "landmarks_2d":
                console_obj["landmarks_count"] = len(value)
                console_obj["landmarks_2d_preview"] = value[:landmarks_preview_count]
                continue
            if key == "svh" and isinstance(value, dict):
                svh_preview = dict(value)
                positions = list(svh_preview.get("target_positions", []))
                svh_preview["target_positions_count"] = len(positions)
                svh_preview["target_positions_preview"] = positions[:landmarks_preview_count]
                svh_preview.pop("target_positions", None)
                console_obj[key] = svh_preview
                continue
            console_obj[key] = value
        return self._round_value(console_obj)

    def print_console(self, obj: Dict[str, Any], landmarks_preview_count: int = 3) -> None:
        print(json.dumps(self.to_console_obj(obj, landmarks_preview_count=landmarks_preview_count), ensure_ascii=False))

    def _warn(self, message: str) -> None:
        if self.logger is not None:
            self.logger.warning(message)

    def save_last_frame(self, obj: Dict[str, Any]) -> None:
        if not self.save_last_json:
            return
        try:
            path = Path(self.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            self._warn(f"Failed to save last-frame JSON: {exc}")

    def append_jsonl(self, obj: Dict[str, Any]) -> None:
        if not self.jsonl_path or self._jsonl_failed:
            return
        try:
            path = Path(self.jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False))
                f.write("\n")
        except OSError as exc:
            self._jsonl_failed = True
            self._warn(f"Failed to append JSONL log; disabling JSONL for this session: {exc}")

    def send(self, obj: Dict[str, Any]) -> None:
        """Reserved output interface for future network/control integration."""
        _ = obj
