"""仅监听本机的轻量标注服务；保存前验证标注契约。"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from .camera_m3c import load_timeline, read_json, save_annotations

HTML_PATH = Path(__file__).with_name("m3c_annotator.html")


def serve(clip_dir: Path, *, port: int = 8765) -> None:
    manifest = read_json(clip_dir / "manifest.json")
    video = Path(manifest["video_path"])
    if not video.is_file():
        raise FileNotFoundError(video)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path
            if route == "/":
                self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            elif route == "/api":
                value = {"manifest": manifest, "timeline": load_timeline(clip_dir),
                         "annotations": read_json(clip_dir / "annotations.json")}
                self._send(200, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")
            elif route == "/video":
                size = video.stat().st_size
                range_header = self.headers.get("Range")
                start, end = 0, size - 1
                if range_header:
                    try:
                        value = range_header.removeprefix("bytes=").split(",", 1)[0]
                        a, b = value.split("-", 1)
                        start = int(a) if a else 0
                        end = min(int(b), size - 1) if b else size - 1
                    except (ValueError, AttributeError):
                        self.send_error(416)
                        return
                if not 0 <= start <= end < size:
                    self.send_error(416)
                    return
                self.send_response(206 if range_header else 200)
                self.send_header("Content-Type", "video/x-msvideo" if video.suffix.lower() == ".avi" else "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if range_header:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                with video.open("rb") as handle:
                    handle.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = handle.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            elif route == "/frame":
                try:
                    index = int(parse_qs(parsed.query)["index"][0])
                    if not 0 <= index < manifest["frame_count"]:
                        raise ValueError("帧号越界")
                    cap = cv2.VideoCapture(str(video))
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                        ok, frame = cap.read()
                        actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                        if not ok or actual != index:
                            raise ValueError("无法准确解码指定帧")
                        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        if not encoded:
                            raise ValueError("JPEG 编码失败")
                    finally:
                        cap.release()
                    self._send(200, buffer.tobytes(), "image/jpeg")
                except (KeyError, IndexError, ValueError) as exc:
                    self._send(400, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("标注 JSON 大小无效")
                value = json.loads(self.rfile.read(length))
                save_annotations(clip_dir, value)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._send(400, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                return
            self._send(200, b'{"saved":true}', "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"本地标注器：http://127.0.0.1:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
