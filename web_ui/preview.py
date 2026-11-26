# -*- coding: utf-8 -*-
"""独立 MJPEG 预览拉流。

``stream_service`` 的 FFmpeg 管道在抽帧后只输出 ``sample_fps`` 帧率的
rawvideo，Python 侧拿不到全帧率画面。本模块另起一路 FFmpeg，直接输出
MJPEG 字节流，按 JPEG 起止标记切帧后广播给所有浏览器订阅者，因此预览
帧率与检测链路完全解耦，不影响现有功能。
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

from stream_service.stream import resolve_ffmpeg_path

from .config import PreviewSettings


LOGGER = logging.getLogger(__name__)

JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"
READ_CHUNK = 64 * 1024
MAX_BUFFER_BYTES = 8 * 1024 * 1024


@dataclass
class PreviewStats:
    """预览流的运行统计，供界面显示状态。"""

    frames: int = 0
    subscribers: int = 0
    restarts: int = 0
    last_frame_at: float | None = None
    error: str | None = None


class JpegSplitter:
    """把连续的 MJPEG 字节流切分为单帧 JPEG。

    JPEG 熵编码段内的 ``0xFF`` 会被填充为 ``0xFF00``，因此 ``0xFFD9``
    只会作为结束标记出现，按标记切分是安全的。
    """

    def __init__(self) -> None:
        """创建空缓冲区。"""
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[bytes]:
        """追加字节并返回本次解析出的完整 JPEG 帧。

        Args:
            chunk: 从 FFmpeg 管道读到的任意长度字节。

        Returns:
            按出现顺序排列的完整 JPEG 帧列表。

        Example:
            >>> JpegSplitter().feed(b"\\xff\\xd8data\\xff\\xd9")
            [b'\\xff\\xd8data\\xff\\xd9']
        """
        self._buffer.extend(chunk)
        frames: list[bytes] = []
        while True:
            start = self._buffer.find(JPEG_START)
            if start < 0:
                # 没有起始标记时保留最后 1 字节，避免标记被切断
                if len(self._buffer) > 1:
                    del self._buffer[:-1]
                break
            if start > 0:
                del self._buffer[:start]
            end = self._buffer.find(JPEG_END, len(JPEG_START))
            if end < 0:
                if len(self._buffer) > MAX_BUFFER_BYTES:
                    LOGGER.warning("MJPEG 缓冲区超限，丢弃未完成帧")
                    self._buffer.clear()
                break
            frames.append(bytes(self._buffer[: end + len(JPEG_END)]))
            del self._buffer[: end + len(JPEG_END)]
        return frames


@dataclass(eq=False)
class _Subscriber:
    """单个预览订阅者的有界帧队列。

    队列是 ``asyncio.Queue``，由拉流线程通过 ``loop.call_soon_threadsafe``
    投递，避免每帧占用线程池。``eq=False`` 保留默认的对象哈希，使其可放入集合。
    """

    loop: asyncio.AbstractEventLoop
    maxsize: int = 2
    queue: asyncio.Queue[bytes] = field(init=False)

    def __post_init__(self) -> None:
        """在所属事件循环中创建队列。"""
        self.queue = asyncio.Queue(maxsize=self.maxsize)

    def offer(self, frame: bytes) -> None:
        """从任意线程投递最新帧；队列满时丢弃最旧帧。"""
        try:
            self.loop.call_soon_threadsafe(self._push, frame)
        except RuntimeError:
            # 事件循环已关闭，订阅者已失效
            pass

    def _push(self, frame: bytes) -> None:
        """在事件循环线程内写入队列，保证订阅者总是看到最新画面。"""
        while True:
            try:
                self.queue.put_nowait(frame)
                return
            except asyncio.QueueFull:
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    return


class PreviewStream:
    """单路视频源的 MJPEG 预览流，自带断流重连。"""

    def __init__(self, session_id: str, source: str, settings: PreviewSettings) -> None:
        """创建预览流但不启动拉流。

        Args:
            session_id: 关联的检测会话 ID。
            source: RTSP/RTMP/HTTP(S) 视频源地址。
            settings: 预览分辨率、帧率和质量配置。
        """
        self.session_id = session_id
        self.source = source
        self._settings = settings
        self._subscribers: set[_Subscriber] = set()
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._last_frame: bytes | None = None
        self._stats = PreviewStats()
        self._thread = threading.Thread(
            target=self._run,
            name=f"preview-{session_id}",
            daemon=True,
        )

    @property
    def stats(self) -> PreviewStats:
        """返回当前统计快照。"""
        with self._lock:
            return PreviewStats(
                frames=self._stats.frames,
                subscribers=len(self._subscribers),
                restarts=self._stats.restarts,
                last_frame_at=self._stats.last_frame_at,
                error=self._stats.error,
            )

    @property
    def last_frame(self) -> bytes | None:
        """返回最近一帧 JPEG，用于订阅者首帧立即显示。"""
        with self._lock:
            return self._last_frame

    def start(self) -> None:
        """启动后台拉流线程。"""
        self._thread.start()

    def close(self) -> None:
        """停止拉流并关闭所有订阅者。"""
        self._closed.set()
        with self._lock:
            subscribers = tuple(self._subscribers)
            self._subscribers.clear()
        for subscriber in subscribers:
            subscriber.offer(b"")
        self._thread.join(timeout=5)

    def subscribe(self) -> _Subscriber:
        """注册订阅者并立即推送最近一帧。

        Raises:
            RuntimeError: 当前线程没有运行中的事件循环。
        """
        loop = asyncio.get_running_loop()
        subscriber = _Subscriber(loop=loop)
        with self._lock:
            self._subscribers.add(subscriber)
            last = self._last_frame
        if last is not None:
            subscriber.offer(last)
        return subscriber

    def unsubscribe(self, subscriber: _Subscriber) -> None:
        """注销订阅者。"""
        with self._lock:
            self._subscribers.discard(subscriber)

    async def iter_frames(
        self,
        subscriber: _Subscriber,
        is_cancelled: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[bytes]:
        """异步迭代订阅者收到的 JPEG 帧，直到流关闭。

        Args:
            subscriber: ``subscribe`` 返回的订阅者。
            is_cancelled: 可选的取消检查；在等待帧超时时调用，用于在
                视频源长时间无帧时也能及时释放订阅者。

        Yields:
            完整 JPEG 帧字节。
        """
        while not self._closed.is_set():
            try:
                frame = await asyncio.wait_for(subscriber.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if is_cancelled is not None and await is_cancelled():
                    return
                continue
            if not frame:
                return
            yield frame

    def build_command(self) -> list[str]:
        """构建输出 MJPEG 到 stdout 的 FFmpeg 命令。

        Returns:
            FFmpeg 参数列表。

        Example:
            >>> PreviewStream("s", "rtsp://c/live", settings).build_command()[-2:]
            ['mjpeg', 'pipe:1']
        """
        settings = self._settings
        command = [
            resolve_ffmpeg_path(settings.ffmpeg_path),
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-rw_timeout", str(settings.read_timeout_us),
        ]
        if self.source.lower().startswith("rtsp://"):
            command.extend(["-rtsp_transport", settings.rtsp_transport])
        if _is_local_file(self.source):
            # 本地文件默认全速解码，加 -re 才能按实时速率输出，避免刷爆浏览器
            command.append("-re")
        command.extend(
            [
                "-i", self.source,
                "-an",
                "-vf",
                f"scale={settings.width}:{settings.height},fps={settings.fps:g}",
                "-q:v", str(settings.quality),
                "-f", "image2pipe",
                "-c:v", "mjpeg",
                "pipe:1",
            ]
        )
        return command

    def _run(self) -> None:
        """循环拉流；FFmpeg 退出后按配置间隔重连，直到被关闭。"""
        while not self._closed.is_set():
            process: subprocess.Popen[bytes] | None = None
            stderr_lines: deque[str] = deque(maxlen=8)
            stderr_thread: threading.Thread | None = None
            exit_code: int | None = None
            try:
                process = subprocess.Popen(
                    self.build_command(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                stderr_thread = threading.Thread(
                    target=_drain_stderr,
                    args=(process, stderr_lines),
                    name=f"preview-stderr-{self.session_id}",
                    daemon=True,
                )
                stderr_thread.start()
                self._pump(process)
                # stdout 已 EOF，等 FFmpeg 真正退出才能拿到有意义的退出码
                try:
                    exit_code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    exit_code = None
            except FileNotFoundError:
                with self._lock:
                    self._stats.error = "找不到 FFmpeg 可执行文件"
                LOGGER.exception("预览拉流失败: session_id=%s", self.session_id)
                return
            except Exception as exc:  # noqa: BLE001 - 拉流异常需重连而非崩溃
                with self._lock:
                    self._stats.error = f"{type(exc).__name__}: {exc}"
                LOGGER.warning(
                    "预览拉流异常: session_id=%s error=%s", self.session_id, exc
                )
            finally:
                _terminate(process)
                if stderr_thread is not None:
                    stderr_thread.join(timeout=2)
                _close_pipes(process)
            if self._closed.is_set():
                return
            # FFmpeg 自己退出（非正常结束）说明拉流失败，把 stderr 原因暴露给界面
            if exit_code not in (0, None):
                message = _last_error(stderr_lines)
                with self._lock:
                    self._stats.error = message or f"FFmpeg 退出码 {exit_code}"
                LOGGER.warning(
                    "预览拉流中断: session_id=%s error=%s",
                    self.session_id,
                    self._stats.error,
                )
            with self._lock:
                self._stats.restarts += 1
            self._closed.wait(self._settings.retry_interval_s)

    def _pump(self, process: subprocess.Popen[bytes]) -> None:
        """读取 FFmpeg stdout 并广播切分出的 JPEG 帧。"""
        stdout = process.stdout
        if stdout is None:
            return
        splitter = JpegSplitter()
        while not self._closed.is_set():
            chunk = stdout.read(READ_CHUNK)
            if not chunk:
                return
            for frame in splitter.feed(chunk):
                self._broadcast(frame)

    def _broadcast(self, frame: bytes) -> None:
        """把一帧 JPEG 推送给所有订阅者并更新统计。"""
        with self._lock:
            self._last_frame = frame
            self._stats.frames += 1
            self._stats.last_frame_at = time.time()
            # 真正收到画面才清除上一次的失败原因，避免重连期间错误信息闪烁
            self._stats.error = None
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber.offer(frame)


class PreviewManager:
    """按 session_id 管理多路预览流。"""

    def __init__(self, settings: PreviewSettings) -> None:
        """创建预览管理器。

        Args:
            settings: 预览拉流配置。
        """
        self._settings = settings
        self._streams: dict[str, PreviewStream] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """返回预览功能是否启用。"""
        return self._settings.enabled

    def open(self, session_id: str, source: str) -> PreviewStream:
        """获取或创建指定会话的预览流。

        Args:
            session_id: 检测会话 ID。
            source: 视频源地址；与已存在流不同时会重建。

        Returns:
            已启动的预览流。

        Raises:
            RuntimeError: 预览功能被配置关闭。
        """
        if not self._settings.enabled:
            raise RuntimeError("预览功能已在配置中关闭")
        with self._lock:
            existing = self._streams.get(session_id)
            if existing is not None and existing.source == source:
                return existing
            if existing is not None:
                existing.close()
            stream = PreviewStream(session_id, source, self._settings)
            self._streams[session_id] = stream
        stream.start()
        return stream

    def get(self, session_id: str) -> PreviewStream | None:
        """返回已存在的预览流；不存在时返回 None。"""
        with self._lock:
            return self._streams.get(session_id)

    def session_ids(self) -> list[str]:
        """返回当前所有预览流的会话 ID。"""
        with self._lock:
            return list(self._streams)

    def close(self, session_id: str) -> bool:
        """关闭并移除指定会话的预览流。

        Returns:
            实际关闭了流时为 True。
        """
        with self._lock:
            stream = self._streams.pop(session_id, None)
        if stream is None:
            return False
        stream.close()
        return True

    def close_all(self) -> None:
        """关闭全部预览流。"""
        with self._lock:
            streams = tuple(self._streams.values())
            self._streams.clear()
        for stream in streams:
            stream.close()


def _is_local_file(source: str) -> bool:
    """判断视频源是否为本地文件路径。

    Args:
        source: 视频源地址。

    Returns:
        是本地文件路径时为 True。

    Example:
        >>> _is_local_file("C:/tmp/a.mp4")
        True
    """
    lowered = source.lower()
    if lowered.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://")):
        return False
    return Path(source).is_file()


def _drain_stderr(
    process: subprocess.Popen[bytes], lines: deque[str]
) -> None:
    """持续读取 FFmpeg stderr，保留最后若干行用于诊断拉流失败原因。

    Args:
        process: FFmpeg 子进程。
        lines: 有界行缓冲，超出长度自动丢弃最旧行。
    """
    stream = process.stderr
    if stream is None:
        return
    try:
        for raw in iter(stream.readline, b""):
            text = raw.decode("utf-8", "replace").strip()
            if text:
                lines.append(text)
    except (ValueError, OSError):
        # 进程被终止时管道会关闭，属正常结束
        return
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _last_error(lines: deque[str]) -> str | None:
    """从 stderr 缓冲里挑出最能说明失败原因的一行。

    FFmpeg 失败时通常先打印具体原因（如 ``Error opening input files: I/O error``），
    最后再补一句通用的 ``Conversion failed!``。因此优先取含 ``error``/``invalid``
    的行，其次取含 ``failed`` 的行，都没有时退回最后一行。

    Args:
        lines: ``_drain_stderr`` 收集的行。

    Returns:
        最能说明失败原因的一行；缓冲为空时为 None。

    Example:
        >>> _last_error(deque(["Error opening input: I/O error", "Conversion failed!"]))
        'Error opening input: I/O error'
    """
    for keyword in ("error", "invalid", "failed"):
        for line in lines:
            if keyword in line.lower():
                return _strip_ffmpeg_context(line)
    return _strip_ffmpeg_context(lines[-1]) if lines else None


def _strip_ffmpeg_context(line: str) -> str:
    """去掉 FFmpeg 日志行里的上下文前缀，如 ``[in#0 @ 000002be018703c0] ``。

    这些前缀含进程内指针地址，对用户没有意义，且每次重连都会变化。

    Args:
        line: 原始 stderr 行。

    Returns:
        去掉前缀后的消息。

    Example:
        >>> _strip_ffmpeg_context("[in#0 @ 0x1f80] Error opening input: I/O error")
        'Error opening input: I/O error'
    """
    text = line.strip()
    if text.startswith("["):
        end = text.find("]")
        if end != -1:
            text = text[end + 1 :].strip()
    return text or line.strip()


def _terminate(process: subprocess.Popen[bytes] | None) -> None:
    """终止 FFmpeg 子进程（不关闭管道，交由 ``_close_pipes`` 处理）。"""
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _close_pipes(process: subprocess.Popen[bytes] | None) -> None:
    """关闭子进程管道；须在 stderr 读取线程结束后调用。"""
    if process is None:
        return
    for stream in (process.stdout, process.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            pass
