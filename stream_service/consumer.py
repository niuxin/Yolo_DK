# -*- coding: utf-8 -*-
"""实时会话检测、代表帧回调和标注流输出编排。"""
from __future__ import annotations

import argparse
import base64
import ipaddress
import logging
import os
import socket
import threading
from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode, urlsplit, urlunsplit

import numpy as np
from PIL import Image, ImageDraw
import requests

from .api_client import (
    BBox,
    DiseaseApiClient,
    DiseaseDetection,
    UniqueTargetApiClient,
    VLMApiClient,
)
from .callback_client import GatewayCallbackClient
from .config import StreamServiceSettings, load_settings
from .control_api import create_app
from .stream import (
    FrameReader,
    VideoInfo,
    VideoFrame,
    probe_video_info,
    select_stream_source,
    start_stream_puller,
    start_stream_pusher,
    stop_stream_puller,
    stop_stream_pusher,
    write_stream_frame,
)
from .task_manager import (
    DetectionTask,
    OutputManager,
    OutputTask,
    OutputWorker,
    TaskManager,
    TaskWorker,
)


LOGGER = logging.getLogger(__name__)


def create_session_worker(settings: StreamServiceSettings) -> TaskWorker:
    """创建按配置抽帧、调用模型并回调最优代表帧的会话 worker。"""

    def run_session(task: DetectionTask, stop_event: threading.Event) -> None:
        if not settings.vlm.enabled:
            raise RuntimeError("网关协议要求启用结构化 VLM 描述服务")
        callback = GatewayCallbackClient(
            task.callback_url,
            settings.gateway,
            timeout_s=settings.disease_api.timeout_s,
        )
        detector = DiseaseApiClient(settings.disease_api)
        target_client = UniqueTargetApiClient(settings.unique_target_api)
        vlm = VLMApiClient(settings.vlm)
        active_target_sessions = [
            f"{task.session_id}__{algorithm_id}"
            for algorithm_id in task.algorithm_ids
        ]
        frame_seq = 0
        captured_at_by_seq: dict[int, int] = {}
        active_source: str | None = None
        last_media_timestamp_ms: float | None = None
        timestamp_offset_ms = 0.0

        try:
            while not stop_event.is_set():
                if active_source is None:
                    active_source = select_stream_source(
                        replace(settings.stream, source=task.stream_url)
                    )
                    if active_source is None:
                        stop_event.wait(settings.stream.retry_interval_s)
                        continue

                stream_settings = replace(settings.stream, source=active_source)
                video_info = probe_video_info(stream_settings)
                puller = start_stream_puller(
                    stream_settings,
                    native_size=True,
                    with_timestamps=True,
                )
                reader = FrameReader(
                    puller,
                    stream_settings,
                    width=video_info.width,
                    height=video_info.height,
                    frame_rate=video_info.frame_rate,
                    start_time_s=video_info.start_time_s,
                    with_timestamps=True,
                )
                frames_read = 0
                try:
                    while not stop_event.is_set():
                        video_frame = reader.read_timed(stop_event)
                        if video_frame is None:
                            break
                        media_timestamp_ms, timestamp_offset_ms = (
                            _monotonic_media_timestamp(
                                video_frame.media_timestamp_ms,
                                last_media_timestamp_ms,
                                timestamp_offset_ms,
                                min(
                                    video_info.frame_rate,
                                    settings.stream.sample_fps,
                                ),
                            )
                        )
                        last_media_timestamp_ms = media_timestamp_ms
                        video_frame = VideoFrame(
                            video_frame.image, media_timestamp_ms
                        )
                        frame_seq += 1
                        frames_read += 1
                        captured_at = _now_ms()
                        captured_at_by_seq[frame_seq] = captured_at
                        if frame_seq > 2000:
                            captured_at_by_seq.pop(frame_seq - 2000, None)
                        target_events = _detect_and_track(
                            detector,
                            target_client,
                            task,
                            video_frame,
                            frame_seq,
                        )
                        if target_events:
                            frame_payloads, descriptions = _build_callback_frames(
                                target_events,
                                captured_at_by_seq,
                                video_info,
                            )
                            if frame_payloads:
                                callback.send_detection(task.session_id, frame_payloads)
                                _send_descriptions(
                                    task.session_id,
                                    descriptions,
                                    vlm,
                                    callback,
                                )
                finally:
                    stop_stream_puller(puller)
                    reader.close()
                if frames_read == 0:
                    active_source = None
                if not stop_event.is_set():
                    stop_event.wait(settings.stream.retry_interval_s)
        finally:
            try:
                for target_session in active_target_sessions:
                    target_client.close_session(target_session)
            finally:
                try:
                    target_client.close()
                finally:
                    try:
                        detector.close()
                    finally:
                        try:
                            vlm.close()
                        finally:
                            callback.close()

    return run_session


def _monotonic_media_timestamp(
    timestamp_ms: float,
    previous_ms: float | None,
    offset_ms: float,
    frame_rate: float,
) -> tuple[float, float]:
    adjusted_ms = timestamp_ms + offset_ms
    if previous_ms is None and adjusted_ms < 0:
        offset_ms -= adjusted_ms
        adjusted_ms = 0.0
    elif previous_ms is not None and adjusted_ms <= previous_ms:
        offset_ms += previous_ms + 1000 / frame_rate - adjusted_ms
        adjusted_ms = timestamp_ms + offset_ms
    return adjusted_ms, offset_ms


def _detect_and_track(
    detector: DiseaseApiClient,
    target_client: UniqueTargetApiClient,
    task: DetectionTask,
    video_frame: VideoFrame,
    frame_seq: int,
) -> list[tuple[str, dict[str, object]]]:
    grouped: list[tuple[str, dict[str, object]]] = []
    for algorithm_id in task.algorithm_ids:
        detections = detector.detect(
            video_frame.image,
            task.session_id,
            frame_seq,
            algorithm_id,
        )
        events = target_client.process(
            video_frame.image,
            f"{task.session_id}__{algorithm_id}",
            frame_seq,
            video_frame.media_timestamp_ms,
            detections,
        )
        grouped.extend((algorithm_id, event) for event in events)
    return grouped


def _build_callback_frames(
    target_events: list[tuple[str, dict[str, object]]],
    captured_at_by_seq: dict[int, int],
    video_info: VideoInfo,
) -> tuple[list[dict[str, object]], list[tuple[str, dict[str, object]]]]:
    grouped: dict[int, dict[str, object]] = {}
    descriptions: list[tuple[str, dict[str, object]]] = []
    for algorithm_id, event in target_events:
        frame_seq = int(event["frame_id"])
        frame_id = f"frame-{frame_seq}"
        image_bytes = bytes(event["image_bytes"])
        frame = grouped.setdefault(
            frame_seq,
            {
                "frameId": frame_id,
                "frameSeq": frame_seq,
                "mediaTimestampMs": float(event["timestamp_ms"]),
                "capturedAt": captured_at_by_seq.get(frame_seq, _now_ms()),
                "width": video_info.width,
                "height": video_info.height,
                "frameObjectKey": (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(image_bytes).decode("ascii")
                ),
                "algorithmResults": {},
            },
        )
        algorithm_results = frame["algorithmResults"]
        if not isinstance(algorithm_results, dict):
            raise TypeError("algorithmResults 结构无效")
        result = algorithm_results.setdefault(
            algorithm_id,
            {"algorithmId": algorithm_id, "status": "success", "detections": []},
        )
        detections = result["detections"]
        if not isinstance(detections, list):
            raise TypeError("detections 结构无效")
        bbox = event["bbox_xyxy"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("唯一目标事件 bbox_xyxy 格式无效")
        x1, y1, x2, y2 = (float(value) for value in bbox)
        target_id = str(event["target_id"])
        result["detections"].append(
            {
                "detectionKey": f"{frame_id}-{algorithm_id}-{target_id}",
                "findingId": target_id,
                "classKey": str(event["class_name"]),
                "className": str(event["class_name"]),
                "score": float(event["confidence"]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            }
        )
        descriptions.append((algorithm_id, event))
    frames: list[dict[str, object]] = []
    for frame_seq, frame in sorted(grouped.items()):
        frame["algorithmResults"] = list(frame["algorithmResults"].values())
        frames.append(frame)
    return frames, descriptions


def _send_descriptions(
    session_id: str,
    events: list[tuple[str, dict[str, object]]],
    vlm: VLMApiClient,
    callback: GatewayCallbackClient,
) -> None:
    for algorithm_id, event in events:
        image_bytes = bytes(event["image_bytes"])
        with Image.open(BytesIO(image_bytes)) as image:
            image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        bbox_values = event["bbox_xyxy"]
        if not isinstance(bbox_values, list) or len(bbox_values) != 4:
            raise ValueError("唯一目标事件 bbox_xyxy 格式无效")
        x1, y1, x2, y2 = (float(value) for value in bbox_values)
        detection = DiseaseDetection(
            disease_type=str(event["class_name"]),
            confidence=float(event["confidence"]),
            bbox=BBox(x1, y1, x2, y2),
            class_id=int(event["class_id"]),
        )
        try:
            result = vlm.describe(image_bgr, detection)
            callback.send_description(
                session_id,
                algorithm_id,
                str(event["target_id"]),
                result.summary,
                result.severity,
                result.suggestion,
            )
        except (
            OSError,
            ValueError,
            RuntimeError,
            requests.RequestException,
        ):
            LOGGER.exception(
                "代表帧描述生成或回调失败: session_id=%s algorithm_id=%s finding_id=%s",
                session_id,
                algorithm_id,
                event["target_id"],
            )


def create_output_worker(settings: StreamServiceSettings) -> OutputWorker:
    """创建保持源尺寸和帧率的多算法标注流发布 worker。"""

    def run_output(
        task: OutputTask,
        stop_event: threading.Event,
        set_status: Callable[[str], None],
    ) -> None:
        secret = os.environ.get(settings.gateway.output_secret_env, "")
        if not secret:
            raise RuntimeError(
                f"RTMP 推流密钥环境变量未设置: {settings.gateway.output_secret_env}"
            )
        _validate_output_host(
            task.output_stream_url, settings.gateway.output_allowed_hosts
        )
        detector = DiseaseApiClient(settings.disease_api)
        active_source: str | None = None
        try:
            while not stop_event.is_set():
                if active_source is None:
                    active_source = select_stream_source(
                        replace(settings.stream, source=task.stream_url)
                    )
                    if active_source is None:
                        set_status("starting")
                        stop_event.wait(settings.stream.retry_interval_s)
                        continue
                stream_settings = replace(settings.stream, source=active_source)
                video_info = probe_video_info(stream_settings)
                if video_info.width % 2 or video_info.height % 2:
                    raise ValueError("H.264/RTMP 输出要求源分辨率为偶数")
                puller = start_stream_puller(
                    stream_settings,
                    every_frame=True,
                    native_size=True,
                    with_timestamps=True,
                )
                reader = FrameReader(
                    puller,
                    stream_settings,
                    width=video_info.width,
                    height=video_info.height,
                    frame_rate=video_info.frame_rate,
                    start_time_s=video_info.start_time_s,
                    with_timestamps=True,
                )
                pusher = None
                frames_read = 0
                try:
                    while not stop_event.is_set():
                        video_frame = reader.read_timed(stop_event)
                        if video_frame is None:
                            break
                        frames_read += 1
                        detections_by_algorithm = []
                        for algorithm_id in task.algorithm_ids:
                            detections = detector.detect(
                                video_frame.image,
                                task.session_id,
                                frames_read,
                                algorithm_id,
                            )
                            detections_by_algorithm.append((algorithm_id, detections))
                        annotated = _draw_detections(
                            video_frame.image, detections_by_algorithm
                        )
                        if pusher is None:
                            output_url = _with_publish_secret(
                                task.output_stream_url, secret
                            )
                            pusher = start_stream_pusher(
                                stream_settings,
                                output_url,
                                width=video_info.width,
                                height=video_info.height,
                                frame_rate=video_info.frame_rate,
                            )
                        write_stream_frame(pusher, annotated)
                        set_status("pushing")
                finally:
                    if pusher is not None:
                        stop_stream_pusher(pusher)
                    stop_stream_puller(puller)
                    reader.close()
                if frames_read == 0:
                    active_source = None
                if not stop_event.is_set():
                    set_status("starting")
                    stop_event.wait(settings.stream.retry_interval_s)
        finally:
            detector.close()

    return run_output


def _with_publish_secret(output_url: str, secret: str) -> str:
    parsed = urlsplit(output_url)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode({"secret": secret}),
            "",
        )
    )


def _validate_output_host(output_url: str, allowed_hosts: tuple[str, ...]) -> None:
    from urllib.parse import urlsplit

    parsed = urlsplit(output_url)
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if not hostname:
        raise ValueError("outputStreamUrl 缺少主机名")
    if hostname in allowed_hosts:
        return
    try:
        addresses = {
            ipaddress.ip_address(result[4][0])
            for result in socket.getaddrinfo(
                hostname,
                parsed.port or 1935,
                type=socket.SOCK_STREAM,
            )
        }
    except OSError as exc:
        raise ValueError("outputStreamUrl 主机名无法解析") from exc
    if not addresses or any(
        not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
        )
        for address in addresses
    ):
        raise ValueError("outputStreamUrl 主机必须位于白名单或解析到私有地址")


def _draw_detections(
    frame: np.ndarray,
    detections_by_algorithm: list[tuple[str, list[DiseaseDetection]]],
) -> np.ndarray:
    """把所有算法检测框叠加到同一帧，并使用算法 ID 区分类别。"""
    image = Image.fromarray(frame[:, :, ::-1])
    draw = ImageDraw.Draw(image)
    colors = ((255, 0, 0), (0, 200, 0), (0, 80, 255), (255, 160, 0))
    for algorithm_index, (algorithm_id, detections) in enumerate(
        detections_by_algorithm
    ):
        color = colors[algorithm_index % len(colors)]
        for detection in detections:
            box = (
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.x2,
                detection.bbox.y2,
            )
            draw.rectangle(box, outline=color, width=3)
            draw.text(
                (detection.bbox.x1, max(0, detection.bbox.y1 - 14)),
                f"{algorithm_id}:{detection.class_id} {detection.confidence:.2f}",
                fill=color,
            )
    return np.ascontiguousarray(np.asarray(image)[:, :, ::-1])


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description="实时视频 AI 网关")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="服务 YAML 配置路径",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = load_settings(args.config)
    output_manager = OutputManager(
        worker=create_output_worker(settings),
        max_concurrent_tasks=settings.control_api.max_concurrent_tasks,
        history_limit=settings.control_api.task_history_limit,
    )
    manager = TaskManager(
        worker=create_session_worker(settings),
        max_concurrent_tasks=settings.control_api.max_concurrent_tasks,
        history_limit=settings.control_api.task_history_limit,
        on_finished=output_manager.stop,
    )
    app = create_app(manager, output_manager)

    import uvicorn

    uvicorn.run(
        app,
        host=settings.control_api.host,
        port=settings.control_api.port,
    )


if __name__ == "__main__":
    main()
