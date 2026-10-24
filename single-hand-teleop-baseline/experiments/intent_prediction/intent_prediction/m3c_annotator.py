"""仅监听本机的轻量标注服务；保存前验证标注契约。"""
from __future__ import annotations

from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from capture.timeline import resolve_media_timestamp_ms
from .camera_m3c import load_timeline, read_json, save_annotations

HTML_PATH = Path(__file__).with_name("m3c_annotator.html")
CONTAINER_PTS = "container_pts_ms"
PTS_TOLERANCE_MS = 1e-3
# 在目标 PTS 之前这么多名义帧处 seek；越过目标就换更早起点，最后从头顺序计数。
SEEK_BACKOFF_FRAMES = (8, 32)


def _seek_by_pts(video: Path, timeline: list[dict], index: int) -> np.ndarray | None:
    """在目标前 seek 后逐帧解码；相邻两帧 PTS 都与时间轴一致才认定为第 index 帧。"""
    target = timeline[index]["pts_ms"]
    previous = timeline[index - 1]["pts_ms"]
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            return None
        # 起点按目标 PTS 换算而非按帧号，长片段累计的帧号漂移不会把落点推过目标。
        nominal = int(target * fps / 1000)
        for back in SEEK_BACKOFF_FRAMES:
            start = nominal - back
            if start <= 0 or not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
                return None
            last = None
            while cap.grab():
                pts = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                if pts > target - PTS_TOLERANCE_MS:
                    if (abs(pts - target) < PTS_TOLERANCE_MS and last is not None
                            and abs(last - previous) < PTS_TOLERANCE_MS):
                        ok, frame = cap.retrieve()
                        return frame if ok else None
                    break  # 越过目标，或落点恰在目标上无法确认前一帧
                last = pts
        return None
    finally:
        cap.release()


def _decode_sequential(video: Path, timeline: list[dict], index: int) -> np.ndarray:
    """与导入相同地从头顺序解码，并逐帧核对媒体时间及来源。"""
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"无法打开视频：{video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        previous = None
        for frame_id in range(index + 1):
            if not cap.grab():
                raise ValueError(f"视频在第 {frame_id} 帧提前结束，与导入时间轴不一致")
            decision = resolve_media_timestamp_ms(frame_id, raw_pts_ms=float(cap.get(cv2.CAP_PROP_POS_MSEC)),
                                                  nominal_fps=fps, previous_timestamp_ms=previous)
            expected = timeline[frame_id]
            if (decision.source != expected["timestamp_source"]
                    or abs(decision.timestamp_ms - expected["pts_ms"]) >= PTS_TOLERANCE_MS):
                raise ValueError(f"第 {frame_id} 帧媒体时间与导入时间轴不一致，视频可能已更换，请重新导入")
            previous = decision.timestamp_ms
        ok, frame = cap.retrieve()
        if not ok:
            raise ValueError(f"无法解码第 {index} 帧")
        return frame
    finally:
        cap.release()


def read_exact_frame(video: Path, timeline: list[dict], index: int) -> np.ndarray:
    """返回与导入时顺序解码一致的第 index 帧。

    OpenCV 的 FFmpeg 后端按“帧号/名义 FPS”换算时间去 seek，并按同一换算回报帧号；
    可变帧率视频上会取到别的帧，帧号自检却照样通过。故以时间轴的容器 PTS 为准定位。
    本帧或前一帧是回退时间戳时无 PTS 可比对，与 seek 失败时一样从头顺序计数。
    """
    if not 0 <= index < len(timeline):
        raise ValueError("帧号越界")
    frame = None
    if index > 0 and all(timeline[i]["timestamp_source"] == CONTAINER_PTS for i in (index - 1, index)):
        frame = _seek_by_pts(video, timeline, index)
    if frame is None:
        frame = _decode_sequential(video, timeline, index)
    if frame.shape[:2] != (timeline[index]["height"], timeline[index]["width"]):
        raise ValueError("解码帧尺寸与导入时间轴不一致")
    return frame


def parse_byte_range(header: str, size: int) -> tuple[int, int] | None:
    """解析单段 Range；`bytes=-N` 按 RFC 7233 取末尾 N 字节。不可满足时返回 None。"""
    match = re.fullmatch(r"bytes=([0-9]*)-([0-9]*)", header.split(",", 1)[0].strip())
    if match is None or match.group(1) == match.group(2) == "":
        return None
    first, last = match.groups()
    if first:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
    elif int(last) == 0:
        return None
    else:
        start, end = max(size - int(last), 0), size - 1
    return (start, end) if 0 <= start <= end < size else None


def make_server(clip_dir: Path, *, port: int = 8765) -> ThreadingHTTPServer:
    manifest = read_json(clip_dir / "manifest.json")
    video = Path(manifest["video_path"])
    if not video.is_file():
        raise FileNotFoundError(video)
    timeline = load_timeline(clip_dir)

    @lru_cache(maxsize=32)
    def frame_jpeg(index: int) -> bytes:
        frame = read_exact_frame(video, timeline, index)
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not encoded:
            raise ValueError("JPEG 编码失败")
        return buffer.tobytes()

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
                    span = parse_byte_range(range_header, size)
                    if span is None:
                        self.send_error(416)
                        return
                    start, end = span
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
                    body = frame_jpeg(int(parse_qs(parsed.query)["index"][0]))
                except (KeyError, IndexError, ValueError) as exc:
                    self._send(400, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                    return
                self._send(200, body, "image/jpeg")
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

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(clip_dir: Path, *, port: int = 8765) -> None:
    server = make_server(clip_dir, port=port)
    print(f"本地标注器：http://127.0.0.1:{server.server_address[1]}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
