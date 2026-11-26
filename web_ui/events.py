# -*- coding: utf-8 -*-
"""线程安全的事件总线，用于把检测结果推送给浏览器。

发布方是病害库轮询器，订阅方是 FastAPI 的 WebSocket 协程。每个订阅者持有
独立的有界队列，队列满时丢弃最旧事件，避免慢客户端拖住轮询线程。
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Any


class Subscription:
    """单个订阅者的有界事件队列。"""

    def __init__(self, maxsize: int) -> None:
        """创建订阅者队列。

        Args:
            maxsize: 队列长度上限，必须大于零。
        """
        if maxsize <= 0:
            raise ValueError("订阅队列长度必须大于零")
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=maxsize)
        self._dropped = 0
        self._closed = False

    @property
    def dropped(self) -> int:
        """返回因队列满而被丢弃的事件数。"""
        return self._dropped

    def offer(self, event: dict[str, Any]) -> None:
        """放入事件；队列满时丢弃最旧事件后重试。"""
        if self._closed:
            return
        try:
            self._queue.put_nowait(event)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._dropped += 1
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            pass

    def close(self) -> None:
        """标记订阅结束；后续 ``offer`` 直接丢弃。"""
        self._closed = True

    async def get(self, timeout: float) -> dict[str, Any] | None:
        """异步取一个事件，超时返回 None。

        使用短间隔轮询而非线程池，避免每个 WebSocket 连接占用一个线程。

        Args:
            timeout: 最长等待秒数。

        Returns:
            事件字典；超时或订阅已关闭时返回 None。
        """
        deadline = time.monotonic() + timeout
        while True:
            if self._closed:
                return None
            try:
                return self._queue.get_nowait()
            except queue.Empty:
                pass
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(0.02)


class EventBus:
    """向所有订阅者广播事件的线程安全总线。"""

    def __init__(self, queue_size: int = 256) -> None:
        """创建事件总线。

        Args:
            queue_size: 每个订阅者的队列长度上限。
        """
        if queue_size <= 0:
            raise ValueError("事件队列长度必须大于零")
        self._queue_size = queue_size
        self._subscribers: set[Subscription] = set()
        self._lock = threading.Lock()

    @property
    def subscriber_count(self) -> int:
        """返回当前订阅者数量。"""
        with self._lock:
            return len(self._subscribers)

    def subscribe(self) -> Subscription:
        """注册新订阅者并返回其队列。"""
        subscription = Subscription(self._queue_size)
        with self._lock:
            self._subscribers.add(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        """注销订阅者并关闭其队列。"""
        subscription.close()
        with self._lock:
            self._subscribers.discard(subscription)

    def publish(self, event: dict[str, Any]) -> None:
        """向所有订阅者广播事件；无订阅者时立即返回。"""
        with self._lock:
            subscribers = tuple(self._subscribers)
        for subscription in subscribers:
            subscription.offer(event)
