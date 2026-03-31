from __future__ import annotations

import json
from typing import Any, Dict


class JsonExporter:
    def __init__(self, output_path: str, save_last_json: bool = True) -> None:
        self.output_path = output_path
        self.save_last_json = save_last_json

    def to_json_str(self, obj: Dict[str, Any]) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2)

    def print_console(self, obj: Dict[str, Any]) -> None:
        print(self.to_json_str(obj))

    def save_last_frame(self, obj: Dict[str, Any]) -> None:
        if not self.save_last_json:
            return
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def send(self, obj: Dict[str, Any]) -> None:
        """Reserved output interface for future network/control integration."""
        _ = obj
