# -*- coding: utf-8 -*-
"""病害库增量轮询器。

界面服务只读轮询配置的病害库，并通过事件总线推送外部写入的新记录。轮询
只读、幂等，不修改数据库，也不接收 AI 网关回调。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .events import EventBus
from .records import RecordRepository


LOGGER = logging.getLogger(__name__)


class RecordWatcher:
    """后台线程轮询病害库，把新记录和描述更新发布到事件总线。"""

    def __init__(
        self,
        repository: RecordRepository,
        bus: EventBus,
        throttle_per_second: float = 4,
    ) -> None:
        """创建轮询器但不启动。

        Args:
            repository: 只读病害库访问层。
            bus: 事件总线。
            throttle_per_second: 每个会话每秒最多推送条数，避免高频入库时
                把 WebSocket 打满；超出部分仍可在历史列表中查到。
        """
        if throttle_per_second <= 0:
            raise ValueError("推送节流必须大于零")
        self._repository = repository
        self._bus = bus
        self._min_interval = 1.0 / throttle_per_second
        self._last_id = 0
        self._last_publish: dict[str, float] = {}
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="record-watcher", daemon=True
        )

    def prime(self) -> None:
        """把游标定位到当前最大 id，避免启动时把历史记录全部推送一遍。"""
        self._last_id = self._repository.max_record_id()

    def start(self) -> None:
        """启动后台轮询线程。"""
        self._thread.start()

    def close(self) -> None:
        """停止轮询线程。"""
        self._closed.set()
        self._thread.join(timeout=5)

    def poll_once(self) -> list[dict[str, Any]]:
        """执行一次增量轮询并发布新事件。

        Returns:
            本次发现的新记录列表（已按 id 升序）。

        Example:
            >>> watcher.poll_once()
            []
        """
        if not self._repository.available():
            return []
        rows = self._repository.list_records(limit=200, after_id=self._last_id)
        if not rows:
            return []
        fresh = sorted(rows, key=lambda item: int(item["id"]))
        self._last_id = max(self._last_id, int(fresh[-1]["id"]))
        published: list[dict[str, Any]] = []
        for record in fresh:
            if not self._allow(record):
                continue
            published.append(record)
            self._bus.publish({"type": "record", "record": record})
        return published

    def _allow(self, record: dict[str, Any]) -> bool:
        """按会话做推送节流。

        Args:
            record: 病害记录字典。

        Returns:
            允许推送时为 True。
        """
        session_id = str(record.get("session_id") or "")
        now = time.monotonic()
        last = self._last_publish.get(session_id)
        if last is not None and now - last < self._min_interval:
            return False
        self._last_publish[session_id] = now
        return True

    def _run(self) -> None:
        """按配置间隔循环轮询，直到被关闭。"""
        interval = self._repository.poll_interval_s
        while not self._closed.is_set():
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 - 轮询异常不应终止线程
                LOGGER.exception("病害库轮询失败")
            self._closed.wait(interval)
