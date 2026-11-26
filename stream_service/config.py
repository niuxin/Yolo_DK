# -*- coding: utf-8 -*-
"""流媒体服务配置。

职责：读取 YAML、解析相对路径并校验流媒体、任务 API、检测 API、
存储和 VLM 参数。``load_settings`` 接收配置文件路径并返回类型化配置对象。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StreamSettings:
    """无人机输入流和 FFmpeg 参数。"""

    source: str
    ffmpeg_path: str
    ffprobe_path: str
    rtsp_transport: str
    read_timeout_us: int
    probe_timeout_s: float
    retry_interval_s: float
    width: int
    height: int
    sample_fps: float
    jpeg_quality: int
    hls_fallback: bool
    hls_http_port: int


@dataclass(frozen=True)
class DiseaseApiSettings:
    """GPU 病害检测 HTTP 服务参数。"""

    base_url: str
    timeout_s: float
    upload_jpeg_quality: int


@dataclass(frozen=True)
class StorageSettings:
    """逐帧/代表帧图片及病害库路径。"""

    image_dir: Path
    database_path: Path


@dataclass(frozen=True)
class UniqueTargetApiSettings:
    """可选的跨帧去重服务参数。"""

    enabled: bool
    base_url: str
    timeout_s: float


@dataclass(frozen=True)
class ControlApiSettings:
    """任务控制 HTTP API 配置。"""

    host: str
    port: int
    max_concurrent_tasks: int
    task_history_limit: int


@dataclass(frozen=True)
class VLMSettings:
    """可选的独立 VLM 服务参数。"""

    enabled: bool
    base_url: str
    timeout_s: float


@dataclass(frozen=True)
class GatewaySettings:
    """AI 网关回调鉴权、重试和推流密钥配置。"""

    callback_app_id_env: str
    callback_app_secret_env: str
    output_secret_env: str
    callback_allowed_hosts: tuple[str, ...]
    output_allowed_hosts: tuple[str, ...]
    allow_insecure_http: bool
    callback_retry_attempts: int
    callback_retry_backoff_s: float


@dataclass(frozen=True)
class StreamServiceSettings:
    """流媒体服务完整配置。"""

    stream: StreamSettings
    disease_api: DiseaseApiSettings
    unique_target_api: UniqueTargetApiSettings
    storage: StorageSettings
    control_api: ControlApiSettings
    vlm: VLMSettings
    gateway: GatewaySettings

    @property
    def frame_size_bytes(self) -> int:
        """返回 FFmpeg rawvideo 单帧字节数。"""
        return self.stream.width * self.stream.height * 3


def load_settings(config_path: str | Path) -> StreamServiceSettings:
    """读取流媒体服务 YAML 并校验参数。

    Args:
        config_path: YAML 配置路径。

    Returns:
        校验后的流媒体、API 和存储配置。相对路径以配置文件目录为基准。

    Example:
        >>> settings = load_settings("code/stream_service/config.yaml")
        >>> settings.stream.width
        1280
    """
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"流媒体服务配置不存在: {path}")
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("流媒体服务配置根节点必须是 YAML 对象")

    stream_data = _mapping(data.get("stream"), "stream")
    api_data = _mapping(data.get("disease_api"), "disease_api")
    target_data = _mapping(data.get("unique_target_api"), "unique_target_api")
    storage_data = _mapping(data.get("storage"), "storage")
    control_data = _mapping(data.get("control_api"), "control_api")
    vlm_data = _mapping(data.get("vlm"), "vlm")
    gateway_data = _mapping(data.get("gateway"), "gateway")

    source = str(stream_data.get("source", "")).strip()
    if source and not source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    ):
        raise ValueError("stream.source 必须为空或 RTSP/RTMP/HTTP(S) 地址")
    transport = str(stream_data.get("rtsp_transport", "tcp")).lower()
    if transport not in {"tcp", "udp"}:
        raise ValueError("stream.rtsp_transport 必须是 tcp 或 udp")

    image_dir = Path(str(storage_data.get("image_dir", "data/images"))).expanduser()
    database_path = Path(
        str(storage_data.get("database_path", "data/disease.db"))
    ).expanduser()
    if not image_dir.is_absolute():
        image_dir = path.parent / image_dir
    if not database_path.is_absolute():
        database_path = path.parent / database_path

    stream = StreamSettings(
        source=source,
        ffmpeg_path=str(stream_data.get("ffmpeg_path", "ffmpeg")),
        ffprobe_path=str(stream_data.get("ffprobe_path", "ffprobe")),
        rtsp_transport=transport,
        read_timeout_us=_positive_int(
            stream_data.get("read_timeout_us", 10_000_000),
            "stream.read_timeout_us",
        ),
        probe_timeout_s=_positive_float(
            stream_data.get("probe_timeout_s", 5), "stream.probe_timeout_s"
        ),
        retry_interval_s=_positive_float(
            stream_data.get("retry_interval_s", 5), "stream.retry_interval_s"
        ),
        width=_positive_int(stream_data.get("width"), "stream.width"),
        height=_positive_int(stream_data.get("height"), "stream.height"),
        sample_fps=_positive_float(
            stream_data.get("sample_fps", 5), "stream.sample_fps"
        ),
        jpeg_quality=_quality(stream_data.get("jpeg_quality", 90), "stream.jpeg_quality"),
        hls_fallback=_boolean(
            stream_data.get("hls_fallback", True), "stream.hls_fallback"
        ),
        hls_http_port=_positive_int(
            stream_data.get("hls_http_port", 8080), "stream.hls_http_port"
        ),
    )
    disease_api = DiseaseApiSettings(
        base_url=str(api_data.get("base_url", "http://127.0.0.1:8081")).rstrip("/"),
        timeout_s=_positive_float(
            api_data.get("timeout_s", 10), "disease_api.timeout_s"
        ),
        upload_jpeg_quality=_quality(
            api_data.get("upload_jpeg_quality", 90),
            "disease_api.upload_jpeg_quality",
        ),
    )
    unique_target_api = UniqueTargetApiSettings(
        enabled=_boolean(target_data.get("enabled", True), "unique_target_api.enabled"),
        base_url=str(
            target_data.get("base_url", "http://127.0.0.1:8083")
        ).rstrip("/"),
        timeout_s=_positive_float(
            target_data.get("timeout_s", 10), "unique_target_api.timeout_s"
        ),
    )
    storage = StorageSettings(
        image_dir=image_dir.resolve(),
        database_path=database_path.resolve(),
    )
    control_api = ControlApiSettings(
        host=str(control_data.get("host", "0.0.0.0")),
        port=_positive_int(control_data.get("port", 8082), "control_api.port"),
        max_concurrent_tasks=_positive_int(
            control_data.get("max_concurrent_tasks", 4),
            "control_api.max_concurrent_tasks",
        ),
        task_history_limit=_positive_int(
            control_data.get("task_history_limit", 1000),
            "control_api.task_history_limit",
        ),
    )
    vlm = VLMSettings(
        enabled=_boolean(vlm_data.get("enabled", False), "vlm.enabled"),
        base_url=str(
            vlm_data.get("base_url", "http://127.0.0.1:8084")
        ).rstrip("/"),
        timeout_s=_positive_float(vlm_data.get("timeout_s", 60), "vlm.timeout_s"),
    )
    allowed_hosts_value = gateway_data.get("callback_allowed_hosts", [])
    if not isinstance(allowed_hosts_value, list) or any(
        not isinstance(host, str) or not host.strip()
        for host in allowed_hosts_value
    ):
        raise ValueError("gateway.callback_allowed_hosts 必须是字符串数组")
    output_hosts_value = gateway_data.get("output_allowed_hosts", [])
    if not isinstance(output_hosts_value, list) or any(
        not isinstance(host, str) or not host.strip()
        for host in output_hosts_value
    ):
        raise ValueError("gateway.output_allowed_hosts 必须是字符串数组")
    gateway = GatewaySettings(
        callback_app_id_env=str(
            gateway_data.get("callback_app_id_env", "LIVE_DETECTION_APP_ID")
        ).strip(),
        callback_app_secret_env=str(
            gateway_data.get(
                "callback_app_secret_env", "LIVE_DETECTION_APP_SECRET"
            )
        ).strip(),
        output_secret_env=str(
            gateway_data.get("output_secret_env", "LIVE_OUTPUT_RTMP_SECRET")
        ).strip(),
        callback_allowed_hosts=tuple(
            host.strip().lower() for host in allowed_hosts_value
        ),
        output_allowed_hosts=tuple(
            host.strip().lower() for host in output_hosts_value
        ),
        allow_insecure_http=_boolean(
            gateway_data.get("allow_insecure_http", False),
            "gateway.allow_insecure_http",
        ),
        callback_retry_attempts=_positive_int(
            gateway_data.get("callback_retry_attempts", 4),
            "gateway.callback_retry_attempts",
        ),
        callback_retry_backoff_s=_positive_float(
            gateway_data.get("callback_retry_backoff_s", 1),
            "gateway.callback_retry_backoff_s",
        ),
    )
    return StreamServiceSettings(
        stream=stream,
        disease_api=disease_api,
        unique_target_api=unique_target_api,
        storage=storage,
        control_api=control_api,
        vlm=vlm,
        gateway=gateway,
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """校验并返回 YAML 字典节点。

    Args:
        value: 原始 YAML 节点。
        name: 字段路径。

    Returns:
        字典节点；None 转换为空字典。

    Example:
        >>> _mapping(None, "stream")
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
        >>> _positive_int("1280", "stream.width")
        1280
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
        >>> _positive_float("2.5", "stream.retry_interval_s")
        2.5
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正数") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} 必须是正数")
    return number


def _quality(value: Any, name: str) -> int:
    """校验 JPEG 质量范围。

    Args:
        value: 原始配置值。
        name: 字段路径。

    Returns:
        1 到 100 之间的整数。

    Example:
        >>> _quality(90, "stream.jpeg_quality")
        90
    """
    number = _positive_int(value, name)
    if number > 100:
        raise ValueError(f"{name} 必须在 1 到 100 之间")
    return number


def _boolean(value: Any, name: str) -> bool:
    """校验 YAML 布尔配置。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{name} 必须是 true 或 false")
