# -*- coding: utf-8 -*-
"""管理网关检测会话和会话下的标注流任务。"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlsplit
from uuid import uuid4


LOGGER = logging.getLogger(__name__)
TERMINAL_STATES = {"completed", "stopped", "failed"}
OUTPUT_TERMINAL_STATES = {"stopped", "failed"}


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _valid_url(value: str, schemes: set[str], field_name: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in schemes or not parsed.hostname:
        raise ValueError(f"{field_name} 地址无效")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"{field_name} 端口格式无效") from exc


@dataclass(frozen=True)
class DetectionTask:
    """业务系统提交的实时检测会话。"""

    session_id: str
    stream_url: str
    algorithm_ids: tuple[str, ...]
    callback_url: str
    job_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id.strip() or len(self.session_id) > 128:
            raise ValueError("sessionId 长度必须在 1 到 128 之间")
        _valid_url(
            self.stream_url,
            {"rtsp", "rtmp", "http", "https"},
            "streamUrl",
        )
        _valid_url(self.callback_url, {"http", "https"}, "callbackUrl")
        if not self.algorithm_ids or len(set(self.algorithm_ids)) != len(
            self.algorithm_ids
        ):
            raise ValueError("algorithmIds 必须是非空且不重复的算法 ID 列表")
        if any(not item.strip() or len(item) > 128 for item in self.algorithm_ids):
            raise ValueError("algorithmIds 中包含无效算法 ID")


@dataclass(frozen=True)
class SessionSnapshot:
    """检测会话的幂等受理记录。"""

    session_id: str
    gateway_session_id: str
    accepted_at: int
    status: str
    error: str | None


@dataclass
class _SessionState:
    task: DetectionTask
    gateway_session_id: str = field(default_factory=lambda: f"gw-{uuid4().hex}")
    accepted_at: int = field(default_factory=_now_ms)
    stop_event: threading.Event = field(default_factory=threading.Event)
    status: str = "accepted"
    error: str | None = None
    future: Future[None] | None = None


class TaskAlreadyExists(Exception):
    """sessionId 已被另一组参数占用。"""


class TaskCapacityExceeded(Exception):
    """达到配置的并发任务上限。"""


SessionWorker = Callable[[DetectionTask, threading.Event], None]
SessionFinished = Callable[[str], None]


class TaskManager:
    """管理可重试创建的会话任务及其停止信号。"""

    def __init__(
        self,
        worker: SessionWorker,
        max_concurrent_tasks: int,
        history_limit: int = 1000,
        on_finished: SessionFinished | None = None,
    ) -> None:
        if max_concurrent_tasks <= 0 or history_limit < max_concurrent_tasks:
            raise ValueError("并发数和历史记录上限配置无效")
        self._worker = worker
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_tasks,
            thread_name_prefix="gateway-session",
        )
        self._max_concurrent_tasks = max_concurrent_tasks
        self._history_limit = history_limit
        self._on_finished = on_finished
        self._sessions: dict[str, _SessionState] = {}
        self._lock = threading.Lock()
        self._closed = False

    def start(self, task: DetectionTask) -> SessionSnapshot:
        """接受新会话；同 sessionId、同参数的重试返回已有受理记录。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("会话管理器已关闭")
            existing = self._sessions.get(task.session_id)
            if existing is not None:
                if existing.task == task:
                    return self._snapshot(existing)
                raise TaskAlreadyExists(
                    f"sessionId 已被不同请求占用: {task.session_id}"
                )
            active_count = sum(
                state.status not in TERMINAL_STATES
                for state in self._sessions.values()
            )
            if active_count >= self._max_concurrent_tasks:
                raise TaskCapacityExceeded("已达到最大并发检测会话数")
            self._prune_history()
            state = _SessionState(task=task)
            self._sessions[task.session_id] = state
            state.future = self._executor.submit(self._run, task.session_id)
            return self._snapshot(state)

    def get(self, session_id: str) -> SessionSnapshot | None:
        with self._lock:
            state = self._sessions.get(session_id)
            return self._snapshot(state) if state else None

    def stop(self, session_id: str) -> SessionSnapshot | None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            if state.status not in TERMINAL_STATES:
                state.stop_event.set()
                state.status = "stopping"
            return self._snapshot(state)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for state in self._sessions.values():
                if state.status not in TERMINAL_STATES:
                    state.stop_event.set()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _run(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions[session_id]
            task = state.task
            stop_event = state.stop_event
        failed = False
        try:
            self._worker(task, stop_event)
        except Exception as exc:
            LOGGER.exception("网关检测会话失败: session_id=%s", session_id)
            with self._lock:
                state.error = f"{type(exc).__name__}: {exc}"
                state.status = "failed"
            failed = True
        if self._on_finished is not None:
            try:
                self._on_finished(session_id)
            except Exception as exc:
                LOGGER.exception("清理会话关联资源失败: session_id=%s", session_id)
                with self._lock:
                    state.error = f"{type(exc).__name__}: {exc}"
                    state.status = "failed"
                failed = True
        if failed:
            return
        with self._lock:
            state.status = "stopped" if stop_event.is_set() else "completed"

    def _prune_history(self) -> None:
        while len(self._sessions) >= self._history_limit:
            removable = next(
                (
                    session_id
                    for session_id, state in self._sessions.items()
                    if state.status in TERMINAL_STATES
                ),
                None,
            )
            if removable is None:
                raise TaskCapacityExceeded("会话历史记录已满且没有可清理的终态任务")
            self._sessions.pop(removable)

    @staticmethod
    def _snapshot(state: _SessionState) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=state.task.session_id,
            gateway_session_id=state.gateway_session_id,
            accepted_at=state.accepted_at,
            status=state.status,
            error=state.error,
        )


@dataclass(frozen=True)
class OutputTask:
    """一个会话的标注流发布任务。"""

    session_id: str
    stream_url: str
    algorithm_ids: tuple[str, ...]
    output_stream_url: str

    def __post_init__(self) -> None:
        _valid_url(
            self.stream_url,
            {"rtsp", "rtmp", "http", "https"},
            "streamUrl",
        )
        _valid_url(self.output_stream_url, {"rtmp"}, "outputStreamUrl")
        if urlsplit(self.output_stream_url).query:
            raise ValueError("outputStreamUrl 必须是不含推流密钥的干净地址")
        if not self.algorithm_ids or len(set(self.algorithm_ids)) != len(
            self.algorithm_ids
        ):
            raise ValueError("algorithmIds 必须是非空且不重复的算法 ID 列表")


@dataclass(frozen=True)
class OutputSnapshot:
    session_id: str
    output_task_id: str
    status: str
    error: str | None


@dataclass
class _OutputState:
    task: OutputTask
    output_task_id: str = field(default_factory=lambda: f"gw-out-{uuid4().hex}")
    stop_event: threading.Event = field(default_factory=threading.Event)
    status: str = "starting"
    error: str | None = None
    future: Future[None] | None = None


OutputWorker = Callable[[OutputTask, threading.Event, Callable[[str], None]], None]


class OutputAlreadyExists(Exception):
    """会话已经存在不同参数的标注流任务。"""


class OutputManager:
    """独立管理标注流发布状态，确保 DELETE 等待 FFmpeg 断开。"""

    def __init__(
        self,
        worker: OutputWorker,
        max_concurrent_tasks: int,
        history_limit: int = 1000,
    ) -> None:
        if max_concurrent_tasks <= 0 or history_limit < max_concurrent_tasks:
            raise ValueError("并发数和推流历史记录上限配置无效")
        self._worker = worker
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_tasks,
            thread_name_prefix="gateway-output",
        )
        self._history_limit = history_limit
        self._states: dict[str, _OutputState] = {}
        self._lock = threading.Lock()
        self._closed = False

    def start(self, task: OutputTask) -> OutputSnapshot:
        with self._lock:
            if self._closed:
                raise RuntimeError("推流管理器已关闭")
            existing = self._states.get(task.session_id)
            if existing is not None:
                if existing.task != task:
                    raise OutputAlreadyExists(
                        f"sessionId 已被不同输出请求占用: {task.session_id}"
                    )
                if existing.status not in OUTPUT_TERMINAL_STATES:
                    return self._snapshot(existing)
                if existing.status == "stopped":
                    return self._snapshot(existing)
            while len(self._states) >= self._history_limit:
                removable = next(
                    (
                        session_id
                        for session_id, state in self._states.items()
                        if state.status in OUTPUT_TERMINAL_STATES
                    ),
                    None,
                )
                if removable is None:
                    raise RuntimeError("推流历史记录已满且没有可清理的终态任务")
                self._states.pop(removable)
            state = _OutputState(task=task)
            self._states[task.session_id] = state
            state.future = self._executor.submit(self._run, task.session_id)
            return self._snapshot(state)

    def get(self, session_id: str) -> OutputSnapshot | None:
        with self._lock:
            state = self._states.get(session_id)
            return self._snapshot(state) if state else None

    def stop(self, session_id: str, timeout_s: float = 10) -> OutputSnapshot | None:
        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                return None
            state.stop_event.set()
            future = state.future
        if future is not None:
            try:
                future.result(timeout=timeout_s)
            except TimeoutError as exc:
                raise RuntimeError("等待 RTMP 推流断开超时") from exc
        return self.get(session_id)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for state in self._states.values():
                state.stop_event.set()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _set_status(self, session_id: str, status: str) -> None:
        with self._lock:
            state = self._states[session_id]
            if state.status not in OUTPUT_TERMINAL_STATES:
                state.status = status

    def _run(self, session_id: str) -> None:
        with self._lock:
            state = self._states[session_id]
            task = state.task
            stop_event = state.stop_event
        try:
            self._worker(
                task,
                stop_event,
                lambda status: self._set_status(session_id, status),
            )
        except Exception as exc:
            LOGGER.exception("标注流任务失败: session_id=%s", session_id)
            with self._lock:
                state.error = f"{type(exc).__name__}: {exc}"
                state.status = "failed"
            return
        with self._lock:
            state.status = "stopped"

    @staticmethod
    def _snapshot(state: _OutputState) -> OutputSnapshot:
        return OutputSnapshot(
            session_id=state.task.session_id,
            output_task_id=state.output_task_id,
            status=state.status,
            error=state.error,
        )
