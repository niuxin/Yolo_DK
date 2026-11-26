# -*- coding: utf-8 -*-
"""视频流量探测和 FFmpeg rawvideo 拉流。"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from fractions import Fraction
import json
import math
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from .config import StreamSettings


@dataclass(frozen=True)
class VideoInfo:
    """输入视频流的原始尺寸和帧率。"""

    width: int
    height: int
    frame_rate: float
    start_time_s: float


@dataclass(frozen=True)
class VideoFrame:
    """带源媒体时间戳的一帧 BGR 视频。"""

    image: np.ndarray
    media_timestamp_ms: float


@lru_cache(maxsize=8)
def resolve_ffmpeg_path(configured: str) -> str:
    """把配置中的 FFmpeg 路径解析为可直接执行的路径。

    依次尝试：配置值本身（绝对路径或相对路径）、系统 PATH、以及
    ``imageio-ffmpeg`` 随包附带的二进制。这样在未安装系统 FFmpeg 的
    机器上也能直接运行。

    Args:
        configured: 配置中的 ``stream.ffmpeg_path``。

    Returns:
        可执行的 FFmpeg 路径。

    Raises:
        RuntimeError: 三种方式都找不到 FFmpeg 时抛出。

    Example:
        >>> resolve_ffmpeg_path("ffmpeg").endswith("ffmpeg.exe")
        True
    """
    candidate = Path(configured).expanduser()
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(configured)
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        pass
    else:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001 - 附带二进制缺失时继续走报错分支
            pass
    raise RuntimeError(
        f"找不到 FFmpeg 可执行文件: {configured}。"
        "请安装 FFmpeg 并加入 PATH，或把 stream.ffmpeg_path 指向可执行文件，"
        "或安装 imageio-ffmpeg（pip install imageio-ffmpeg）。"
    )


@lru_cache(maxsize=8)
def resolve_ffprobe_path(configured: str) -> str:
    candidate = Path(configured).expanduser()
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(configured)
    if found:
        return found
    raise RuntimeError(
        f"找不到 ffprobe 可执行文件: {configured}。请安装 FFmpeg 工具集，"
        "或设置 stream.ffprobe_path。"
    )


def probe_video_info(settings: StreamSettings) -> VideoInfo:
    """查询输入视频原生尺寸、帧率和起始 PTS。"""
    ffprobe = resolve_ffprobe_path(settings.ffprobe_path)
    command = [
        ffprobe,
        "-v", "error",
        "-rw_timeout", str(settings.read_timeout_us),
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,start_time",
        "-of", "json",
    ]
    if settings.source.lower().startswith("rtsp://"):
        command.extend(["-rtsp_transport", settings.rtsp_transport])
    command.append(settings.source)
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=settings.probe_timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("查询视频流参数超时") from exc
    if result.returncode != 0:
        raise RuntimeError(f"查询视频流参数失败: {result.stderr.strip()}")
    try:
        streams = json.loads(result.stdout)["streams"]
        stream = streams[0]
        width = int(stream["width"])
        height = int(stream["height"])
        frame_rate = 0.0
        for key in ("avg_frame_rate", "r_frame_rate"):
            try:
                candidate = float(Fraction(stream.get(key, "")))
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if candidate > 0 and math.isfinite(candidate):
                frame_rate = candidate
                break
        try:
            start_time_s = float(stream.get("start_time") or 0)
        except (TypeError, ValueError):
            start_time_s = 0.0
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("ffprobe 返回了无效的视频参数") from exc
    if width <= 0 or height <= 0 or frame_rate <= 0 or not math.isfinite(frame_rate):
        raise RuntimeError("视频流尺寸和帧率必须大于零")
    return VideoInfo(width, height, frame_rate, start_time_s)


def _with_resolved_ffmpeg(settings: StreamSettings) -> StreamSettings:
    """返回把 ``ffmpeg_path`` 换成可执行绝对路径的配置副本。"""
    resolved = resolve_ffmpeg_path(settings.ffmpeg_path)
    if resolved == settings.ffmpeg_path:
        return settings
    return replace(settings, ffmpeg_path=resolved)


def rtmp_to_hls_url(source: str, hls_http_port: int) -> str | None:
    """把 RTMP 推流地址映射为同源 SRS HLS 播放地址。

    SRS 默认在 8080 端口提供 HLS，路径与 RTMP 的 ``app/stream`` 一致，
    仅扩展名变为 ``.m3u8``。

    Args:
        source: 原始流地址。
        hls_http_port: SRS HTTP 端口。

    Returns:
        HLS 地址；非 RTMP 地址或缺少 app/stream 路径时返回 None。

    Example:
        >>> rtmp_to_hls_url("rtmp://10.0.0.1:1935/live/cam-1", 8080)
        'http://10.0.0.1:8080/live/cam-1.m3u8'
    """
    parsed = urlsplit(source)
    if parsed.scheme.lower() != "rtmp" or not parsed.hostname:
        return None
    app_path = parsed.path.strip("/")
    if not app_path:
        return None
    return urlunsplit(
        ("http", f"{parsed.hostname}:{hls_http_port}", f"/{app_path}.m3u8", "", "")
    )


def build_source_candidates(settings: StreamSettings) -> list[str]:
    """按优先级返回待探测的流地址列表。

    Args:
        settings: 含原始地址和 HLS 回退开关的流配置。

    Returns:
        原始地址在前；启用回退且地址为 RTMP 时追加同源 HLS 地址。

    Example:
        >>> build_source_candidates(settings.stream)[0]
        'rtmp://10.0.0.1:1935/live/cam-1'
    """
    candidates = [settings.source]
    if settings.hls_fallback:
        fallback = rtmp_to_hls_url(settings.source, settings.hls_http_port)
        if fallback is not None:
            candidates.append(fallback)
    return candidates


def select_stream_source(settings: StreamSettings) -> str | None:
    """依次探测候选地址，返回第一个能解码出视频帧的地址。

    Args:
        settings: 含候选地址、FFmpeg 路径和探测超时的配置。

    Returns:
        可用的流地址；全部候选都无流量时返回 None。

    Example:
        >>> select_stream_source(settings.stream)
        'http://10.0.0.1:8080/live/cam-1.m3u8'
    """
    for candidate in build_source_candidates(settings):
        if probe_video_stream(replace(settings, source=candidate)):
            return candidate
    return None


def probe_video_stream(settings: StreamSettings) -> bool:
    """通过解码一帧确认地址当前确实有视频流量。

    Args:
        settings: 包含流地址、FFmpeg 路径和探测超时的配置。

    Returns:
        能成功解码一帧视频时为 True，否则为 False。无人机未起飞或未推流时
        应返回 False，调用方会等待后再次探测。

    Example:
        >>> probe_video_stream(settings.stream)
        False
    """
    try:
        result = subprocess.run(
            build_probe_command(_with_resolved_ffmpeg(settings)),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=settings.probe_timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def build_probe_command(settings: StreamSettings) -> list[str]:
    """构建只解码一帧后退出的 FFmpeg 探测命令。

    Args:
        settings: 输入流配置。

    Returns:
        FFmpeg 参数列表。

    Example:
        >>> build_probe_command(settings.stream)[-6:]
        ['-map', '0:v:0', '-frames:v', '1', '-f', 'null']
    """
    return [
        *_network_options(settings, "error"),
        "-i", settings.source,
        "-map", "0:v:0",
        "-frames:v", "1",
        "-f", "null",
        "-",
    ]


def start_stream_puller(
    settings: StreamSettings,
    *,
    every_frame: bool = False,
    native_size: bool = False,
    with_timestamps: bool = False,
) -> subprocess.Popen[bytes]:
    """启动持续输出 BGR rawvideo 的 FFmpeg 拉流进程。

    Args:
        settings: 输出尺寸、抽帧帧率与流地址配置。

    Returns:
        正在运行的 FFmpeg 子进程。

    Example:
        >>> puller = start_stream_puller(settings.stream)
        >>> stop_stream_puller(puller)
    """
    try:
        return subprocess.Popen(
            build_puller_command(
                _with_resolved_ffmpeg(settings),
                every_frame=every_frame,
                native_size=native_size,
                with_timestamps=with_timestamps,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if with_timestamps else None,
            bufsize=0,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"找不到 FFmpeg 可执行文件: {settings.ffmpeg_path}") from exc


def build_puller_command(
    settings: StreamSettings,
    *,
    every_frame: bool = False,
    native_size: bool = False,
    with_timestamps: bool = False,
) -> list[str]:
    """构建 rawvideo BGR 拉流命令。

    默认维持旧配置的尺寸和抽帧行为。网关会话使用 ``native_size`` 保留源
    分辨率，并通过 ``with_timestamps`` 输出每帧源 PTS。

    Args:
        settings: 输入流、目标尺寸和抽帧帧率。
        every_frame: 是否禁止按 ``sample_fps`` 抽帧。
        native_size: 是否保留流的原始分辨率。
        with_timestamps: 是否通过 showinfo 输出源帧 PTS。

    Returns:
        FFmpeg 参数列表。

    Example:
        >>> build_puller_command(settings.stream)[-2:]
        ['rawvideo', 'pipe:1']
    """
    filters = [] if native_size else [f"scale={settings.width}:{settings.height}"]
    if not every_frame:
        filters.append(f"fps={settings.sample_fps:g}")
    if with_timestamps:
        filters.append("showinfo")
    command = [
        *_network_options(settings, "info" if with_timestamps else "warning"),
        "-i", settings.source,
        "-an",
        "-fps_mode", "passthrough",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1",
    ]
    if filters:
        command[command.index("-pix_fmt"):command.index("-pix_fmt")] = [
            "-vf", ",".join(filters)
        ]
    return command


def build_pusher_command(
    settings: StreamSettings,
    publish_url: str,
    *,
    width: int,
    height: int,
    frame_rate: float,
) -> list[str]:
    """构建将 BGR 帧编码为 H.264/FLV 并推送到 RTMP 的 FFmpeg 命令。"""
    keyframe_interval = max(1, round(frame_rate * 2))
    return [
        resolve_ffmpeg_path(settings.ffmpeg_path),
        "-hide_banner",
        "-loglevel", "warning",
        "-nostdin",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", f"{frame_rate:g}",
        "-i", "pipe:0",
        "-an",
        "-fps_mode", "passthrough",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-g", str(keyframe_interval),
        "-keyint_min", str(keyframe_interval),
        "-f", "flv",
        publish_url,
    ]


def start_stream_pusher(
    settings: StreamSettings,
    publish_url: str,
    *,
    width: int,
    height: int,
    frame_rate: float,
) -> subprocess.Popen[bytes]:
    """启动 RTMP 推流 FFmpeg 进程并返回可写入帧数据的进程。"""
    try:
        return subprocess.Popen(
            build_pusher_command(
                settings,
                publish_url,
                width=width,
                height=height,
                frame_rate=frame_rate,
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"找不到 FFmpeg 可执行文件: {settings.ffmpeg_path}") from exc


def write_stream_frame(pusher: subprocess.Popen[bytes], frame: np.ndarray) -> None:
    """把一帧完整 BGR 图像写入推流进程；推流进程退出时明确报错。"""
    if pusher.stdin is None:
        raise RuntimeError("FFmpeg 推流进程没有 stdin 管道")
    if pusher.poll() is not None:
        raise RuntimeError(f"FFmpeg 推流进程已退出，退出码: {pusher.returncode}")
    try:
        pusher.stdin.write(frame.tobytes())
    except (BrokenPipeError, OSError) as exc:
        raise RuntimeError("写入 FFmpeg 推流管道失败") from exc


def stop_stream_pusher(pusher: subprocess.Popen[bytes]) -> None:
    """关闭推流输入并等待 FFmpeg 完成清理。"""
    if pusher.stdin is not None:
        try:
            pusher.stdin.close()
        except BrokenPipeError:
            pass
    try:
        pusher.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pusher.kill()
        pusher.wait(timeout=5)
    if pusher.returncode != 0:
        raise RuntimeError(
            f"FFmpeg 推流进程异常退出，退出码: {pusher.returncode}"
        )


def stop_stream_puller(puller: subprocess.Popen[bytes]) -> None:
    """终止 FFmpeg 拉流进程并释放管道。

    Args:
        puller: 由 start_stream_puller 创建的进程。

    Example:
        >>> stop_stream_puller(puller)
    """
    if puller.poll() is None:
        puller.terminate()
    try:
        puller.wait(timeout=5)
    except subprocess.TimeoutExpired:
        puller.kill()
        puller.wait(timeout=5)
    finally:
        if puller.stdout is not None:
            puller.stdout.close()
        if puller.stderr is not None:
            puller.stderr.close()


def read_video_frame(
    puller: subprocess.Popen[bytes],
    settings: StreamSettings,
    stop_event: threading.Event | None = None,
) -> np.ndarray | None:
    """从 FFmpeg stdout 读取一帧完整 BGR 图像。

    单次调用会新建读取线程，仅适合一次性读取；持续拉流请使用
    ``FrameReader``，避免每帧创建线程。

    Args:
        puller: 已经开始输出的 FFmpeg 进程。
        settings: 决定单帧字节数和图像尺寸。
        stop_event: 置位后尽快返回 None。

    Returns:
        HxWx3 BGR uint8 数组；流结束或不完整时返回 None。

    Example:
        >>> frame = read_video_frame(puller, settings.stream)
        >>> frame.shape if frame is not None else None
        (720, 1280, 3)
    """
    reader = FrameReader(puller, settings)
    try:
        return reader.read(stop_event)
    finally:
        reader.close()


class FrameReader:
    """在后台线程读取 FFmpeg rawvideo 管道。

    Windows 的 ``select`` 只支持套接字，不能用于管道，因此用独立线程阻塞
    读取并把完整帧放入队列，主线程按超时取帧以便响应停止事件。
    """

    def __init__(
        self,
        puller: subprocess.Popen[bytes],
        settings: StreamSettings,
        *,
        width: int | None = None,
        height: int | None = None,
        frame_rate: float | None = None,
        start_time_s: float = 0,
        with_timestamps: bool = False,
    ) -> None:
        """启动读取线程。

        Args:
            puller: 由 start_stream_puller 创建、stdout 为 rawvideo 的进程。
            settings: 决定单帧字节数和图像尺寸。

        Example:
            >>> reader = FrameReader(puller, settings.stream)
        """
        if puller.stdout is None:
            raise RuntimeError("FFmpeg 拉流进程没有 stdout 管道")
        self._puller = puller
        self._settings = settings
        self._width = width or settings.width
        self._height = height or settings.height
        self._frame_rate = frame_rate or settings.sample_fps
        self._start_time_s = start_time_s
        self._frame_size = self._width * self._height * 3
        self._queue: queue.Queue[VideoFrame | None] = queue.Queue(maxsize=2)
        self._timestamps: queue.Queue[float] = queue.Queue()
        self._with_timestamps = with_timestamps and puller.stderr is not None
        self._closed = threading.Event()
        self._timestamp_thread = None
        if self._with_timestamps:
            self._timestamp_thread = threading.Thread(
                target=self._read_timestamps,
                name="ffmpeg-pts-reader",
                daemon=True,
            )
            self._timestamp_thread.start()
        self._thread = threading.Thread(
            target=self._pump,
            name="ffmpeg-frame-reader",
            daemon=True,
        )
        self._thread.start()

    def read(self, stop_event: threading.Event | None = None) -> np.ndarray | None:
        """取下一帧；流结束、读取线程退出或收到停止信号时返回 None。

        Args:
            stop_event: 置位后尽快返回 None。

        Returns:
            HxWx3 BGR uint8 数组或 None。
        """
        while not self._closed.is_set():
            if stop_event is not None and stop_event.is_set():
                return None
            try:
                item = self._queue.get(timeout=0.1)
                return item.image if item is not None else None
            except queue.Empty:
                continue
        return None

    def read_timed(
        self, stop_event: threading.Event | None = None
    ) -> VideoFrame | None:
        while not self._closed.is_set():
            if stop_event is not None and stop_event.is_set():
                return None
            try:
                return self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
        return None

    def close(self) -> None:
        """停止读取线程并等待其退出。

        读取线程可能阻塞在管道上，因此调用方应先终止 FFmpeg 进程
        （``stop_stream_puller``）再调用本方法，否则最多等待 5 秒。
        """
        self._closed.set()
        self._thread.join(timeout=5)
        if self._timestamp_thread is not None:
            self._timestamp_thread.join(timeout=5)

    def _pump(self) -> None:
        """持续读取完整帧并放入队列，直到管道结束或读取被关闭。"""
        stdout = self._puller.stdout
        if stdout is None:
            self._offer(None)
            return
        try:
            frame_index = 0
            while not self._closed.is_set():
                chunks = bytearray()
                while len(chunks) < self._frame_size:
                    chunk = stdout.read(self._frame_size - len(chunks))
                    if not chunk:
                        self._offer(None)
                        return
                    chunks.extend(chunk)
                image = np.frombuffer(chunks, dtype=np.uint8).reshape(
                    (self._height, self._width, 3)
                ).copy()
                try:
                    pts_s = self._timestamps.get(timeout=2) if self._with_timestamps else None
                except queue.Empty:
                    pts_s = None
                if pts_s is None:
                    pts_s = self._start_time_s + frame_index / self._frame_rate
                frame_index += 1
                if not self._offer(VideoFrame(image, pts_s * 1000)):
                    return
        except (OSError, ValueError):
            self._offer(None)

    def _read_timestamps(self) -> None:
        stderr = self._puller.stderr
        if stderr is None:
            return
        pattern = re.compile(rb"pts_time:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))")
        try:
            for line in iter(stderr.readline, b""):
                match = pattern.search(line)
                if match is not None:
                    self._timestamps.put(float(match.group(1)))
        except (OSError, ValueError):
            return

    def _offer(self, item: VideoFrame | None) -> bool:
        """把帧放入队列；队列满时等待，关闭后放弃。"""
        while not self._closed.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False


def _network_options(settings: StreamSettings, loglevel: str) -> list[str]:
    """生成 FFmpeg 网络输入公共参数。

    Args:
        settings: FFmpeg 路径、超时和 RTSP 传输协议。
        loglevel: FFmpeg 日志级别。

    Returns:
        FFmpeg 公共参数列表。

    Example:
        >>> _network_options(settings.stream, "error")[:2]
        ['ffmpeg', '-hide_banner']
    """
    options = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-loglevel", loglevel,
        "-nostdin",
        "-rw_timeout", str(settings.read_timeout_us),
    ]
    if settings.source.lower().startswith("rtsp://"):
        options.extend(["-rtsp_transport", settings.rtsp_transport])
    return options