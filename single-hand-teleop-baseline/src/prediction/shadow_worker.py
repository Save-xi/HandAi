from __future__ import annotations

"""有界、latest-only 的预测影子后台线程。

主控制循环只负责在 UDP 发送后非阻塞提交 canonical payload。真正的模型推理、
门控和诊断拼接都在后台完成；结果由主线程稍后写入独立的 prediction JSON/JSONL，
因此无论模型偶发变慢还是失败，都不会推迟下一帧 Unity baseline UDP。
"""

from copy import deepcopy
import logging
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Tuple

from output.frame_payload_contract import prepare_frame_payload
from prediction.shadow_predictor import PredictionShadow


class PredictionShadowWorker:
    """单 worker、有界输入/输出、队满时保留最新帧的影子执行器。"""

    def __init__(
        self,
        shadow: PredictionShadow,
        *,
        logger: logging.Logger | None = None,
        input_queue_size: int = 1,
        result_queue_size: int = 8,
    ) -> None:
        self.shadow = shadow
        self.logger = logger
        self._input: Queue[Tuple[Dict[str, Any], int]] = Queue(maxsize=max(1, int(input_queue_size)))
        self._results: Queue[Dict[str, Any]] = Queue(maxsize=max(1, int(result_queue_size)))
        self._stop_requested = Event()
        self._closed = False
        self._lock = Lock()
        self.submitted_count = 0
        self.completed_count = 0
        self.dropped_input_count = 0
        self.dropped_result_count = 0
        # latest-only 可以丢普通有效帧，但绝不能丢掉“手消失/输入无效”这一
        # 时序断点。generation 会随下一帧一起传给 worker，确保先清空历史。
        self._reset_generation = 0
        self._applied_reset_generation = 0
        self._thread = Thread(
            target=self._run,
            name="prediction-shadow-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, payload: Dict[str, Any]) -> bool:
        """非阻塞提交；队满时丢弃等待中的旧帧并保留当前最新帧。"""

        if self._closed or self._stop_requested.is_set():
            return False
        item = deepcopy(payload)
        with self._lock:
            if self.shadow.requires_history_reset(item):
                self._reset_generation += 1
            reset_generation = self._reset_generation
        queued_item = (item, reset_generation)
        try:
            self._input.put_nowait(queued_item)
        except Full:
            try:
                self._input.get_nowait()
                self._input.task_done()
                with self._lock:
                    self.dropped_input_count += 1
            except Empty:
                pass
            try:
                self._input.put_nowait(queued_item)
            except Full:
                with self._lock:
                    self.dropped_input_count += 1
                return False
        with self._lock:
            self.submitted_count += 1
        return True

    def _record_result(self, payload: Dict[str, Any]) -> None:
        try:
            self._results.put_nowait(payload)
        except Full:
            # 主循环正常会逐帧 drain；若磁盘/GUI 卡顿，仍限制内存并保留最新结果。
            try:
                self._results.get_nowait()
                self._results.task_done()
                with self._lock:
                    self.dropped_result_count += 1
            except Empty:
                pass
            try:
                self._results.put_nowait(payload)
            except Full:
                with self._lock:
                    self.dropped_result_count += 1

    def _run(self) -> None:
        while not self._stop_requested.is_set() or not self._input.empty():
            try:
                payload, reset_generation = self._input.get(timeout=0.05)
            except Empty:
                continue
            try:
                try:
                    if reset_generation > self._applied_reset_generation:
                        self.shadow.reset_history()
                        self._applied_reset_generation = reset_generation
                    diagnostics = self.shadow.observe(payload)
                except Exception as exc:
                    if self.logger is not None:
                        self.logger.warning("预测影子 worker 推理失败：%s", exc)
                    diagnostics = self.shadow.record_runtime_failure(payload, exc)
                augmented = dict(payload)
                augmented["prediction_diagnostics"] = diagnostics
                prepared = prepare_frame_payload(
                    augmented,
                    include_deprecated_aliases=False,
                )
                self._record_result(prepared)
                with self._lock:
                    self.completed_count += 1
            except Exception as exc:
                # 结果层异常也只停用当前记录，不反向影响 baseline/UDP。
                if self.logger is not None:
                    self.logger.warning("预测影子 worker 无法生成 canonical 诊断帧：%s", exc)
            finally:
                self._input.task_done()

    def drain_results(self, *, max_items: int | None = None) -> List[Dict[str, Any]]:
        """非阻塞取走当前已完成结果。"""

        results: List[Dict[str, Any]] = []
        while max_items is None or len(results) < max_items:
            try:
                results.append(self._results.get_nowait())
                self._results.task_done()
            except Empty:
                break
        return results

    def close(self, *, timeout_s: float = 2.0) -> bool:
        """停止接收新帧并有界等待在途推理；返回线程是否已正常结束。"""

        self._closed = True
        self._stop_requested.set()
        self._thread.join(max(0.0, float(timeout_s)))
        stopped = not self._thread.is_alive()
        if self.logger is not None:
            self.logger.info(
                "预测影子 worker 摘要：提交=%d，完成=%d，输入丢弃=%d，结果丢弃=%d，正常结束=%s。",
                self.submitted_count,
                self.completed_count,
                self.dropped_input_count,
                self.dropped_result_count,
                stopped,
            )
        return stopped
