# -*- coding: utf-8 -*-
"""可视化界面服务配置。

职责：读取 YAML、解析相对路径并校验预览拉流、病害库读取和事件推送参数。
``load_settings`` 接收配置文件路径并返回类型化配置对象。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any
import unicodedata

import yaml


@dataclass(frozen=True)
class PreviewSettings:
    """独立 MJPEG 预览拉流参数。"""

    enabled: bool
    ffmpeg_path: str
    rtsp_transport: str
    read_timeout_us: int
    retry_interval_s: float
    width: int
    height: int
    fps: float
    quality: int


@dataclass(frozen=True)
class UiStorageSettings:
    """病害库和图片目录；默认指向 ``stream_service`` 的 data 目录。"""

    image_dir: Path
    database_path: Path
    poll_interval_s: float


@dataclass(frozen=True)
class UiServerSettings:
    """界面服务监听地址和前端分页参数。"""

    host: str
    port: int
    page_size: int
    max_page_size: int
    event_queue_size: int
    record_throttle_per_second: float
    default_source: str


@dataclass(frozen=True)
class UiSettings:
    """可视化界面服务完整配置。"""

    preview: PreviewSettings
    storage: UiStorageSettings
    server: UiServerSettings


def load_settings(config_path: str | Path) -> UiSettings:
    """读取界面服务 YAML 并校验参数。

    Args:
        config_path: YAML 配置路径。

    Returns:
        校验后的预览、存储和监听配置。相对路径以配置文件目录为基准。

    Example:
        >>> settings = load_settings("web_ui/config.yaml")
        >>> settings.server.port
        8085
    """
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"界面服务配置不存在: {path}")
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("界面服务配置根节点必须是 YAML 对象")

    preview_data = _mapping(data.get("preview"), "preview")
    storage_data = _mapping(data.get("storage"), "storage")
    server_data = _mapping(data.get("server"), "server")

    transport = str(preview_data.get("rtsp_transport", "tcp")).lower()
    if transport not in {"tcp", "udp"}:
        raise ValueError("preview.rtsp_transport 必须是 tcp 或 udp")

    image_dir = Path(str(storage_data.get("image_dir", "data/images"))).expanduser()
    database_path = Path(
        str(storage_data.get("database_path", "data/disease.db"))
    ).expanduser()
    if not image_dir.is_absolute():
        image_dir = path.parent / image_dir
    if not database_path.is_absolute():
        database_path = path.parent / database_path

    page_size = _positive_int(server_data.get("page_size", 50), "server.page_size")
    max_page_size = _positive_int(
        server_data.get("max_page_size", 200), "server.max_page_size"
    )
    if max_page_size < page_size:
        raise ValueError("server.max_page_size 不能小于 server.page_size")

    preview = PreviewSettings(
        enabled=_boolean(preview_data.get("enabled", True), "preview.enabled"),
        ffmpeg_path=str(preview_data.get("ffmpeg_path", "ffmpeg")),
        rtsp_transport=transport,
        read_timeout_us=_positive_int(
            preview_data.get("read_timeout_us", 10_000_000),
            "preview.read_timeout_us",
        ),
        retry_interval_s=_positive_float(
            preview_data.get("retry_interval_s", 3), "preview.retry_interval_s"
        ),
        width=_positive_int(preview_data.get("width", 1280), "preview.width"),
        height=_positive_int(preview_data.get("height", 720), "preview.height"),
        fps=_positive_float(preview_data.get("fps", 25), "preview.fps"),
        quality=_quality(preview_data.get("quality", 7), "preview.quality"),
    )
    storage = UiStorageSettings(
        image_dir=image_dir.resolve(),
        database_path=database_path.resolve(),
        poll_interval_s=_positive_float(
            storage_data.get("poll_interval_s", 1), "storage.poll_interval_s"
        ),
    )
    server = UiServerSettings(
        host=str(server_data.get("host", "0.0.0.0")),
        port=_positive_int(server_data.get("port", 8085), "server.port"),
        page_size=page_size,
        max_page_size=max_page_size,
        event_queue_size=_positive_int(
            server_data.get("event_queue_size", 256), "server.event_queue_size"
        ),
        record_throttle_per_second=_positive_float(
            server_data.get("record_throttle_per_second", 4),
            "server.record_throttle_per_second",
        ),
        default_source=_clean_source(server_data.get("default_source", "")),
    )
    return UiSettings(
        preview=preview,
        storage=storage,
        server=server,
    )


def _clean_source(value: Any) -> str:
    """清洗视频源字符串，去掉首尾空白与不可见格式字符。

    从网页或文档复制 URL 时常会带入零宽空格（U+200B）、零宽连接符（U+200D）
    等 ``Cf`` 类字符，``str.strip()`` 不会移除它们，会导致 FFmpeg 拉流失败。

    Args:
        value: 配置中的原始值。

    Returns:
        去掉首尾空白与不可见格式字符后的字符串。

    Example:
        >>> _clean_source("\u200brtmp://host/live/x\u200b")
        'rtmp://host/live/x'
    """
    text = str(value)
    cleaned = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return cleaned.strip()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """校验并返回 YAML 字典节点。

    Args:
        value: 原始 YAML 节点。
        name: 字段路径。

    Returns:
        字典节点；None 转换为空字典。

    Example:
        >>> _mapping(None, "preview")
        {}
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是 YAML 对象")
    return value


def _positive_int(value: Any, name: str) -> int:
    """校验正整数配置。

    Args:
        value: 原始配置值。
        name: 字段路径。

    Returns:
        正整数。

    Example:
        >>> _positive_int("8085", "server.port")
        8085
    """
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正整数") from exc
    if number <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return number


def _positive_float(value: Any, name: str) -> float:
    """校验正浮点数配置。

    Args:
        value: 原始配置值。
        name: 字段路径。

    Returns:
        正浮点数。

    Example:
        >>> _positive_float("25", "preview.fps")
        25.0
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正数") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} 必须是正数")
    return number


def _quality(value: Any, name: str) -> int:
    """校验 FFmpeg ``-q:v`` 质量参数范围。

    Args:
        value: 原始配置值。
        name: 字段路径。

    Returns:
        1 到 31 之间的整数，数值越小画质越好。

    Example:
        >>> _quality(7, "preview.quality")
        7
    """
    number = _positive_int(value, name)
    if number > 31:
        raise ValueError(f"{name} 必须在 1 到 31 之间")
    return number


def _boolean(value: Any, name: str) -> bool:
    """校验 YAML 布尔配置。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{name} 必须是 true 或 false")
