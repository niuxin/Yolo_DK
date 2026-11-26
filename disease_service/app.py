# -*- coding: utf-8 -*-
"""无状态病害检测 HTTP 接口。"""
from __future__ import annotations

import io
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import DiseaseServiceSettings, load_settings
from .registry import DetectorRegistry
from .types import Detection, DetectorBackend


LOGGER = logging.getLogger(__name__)
METRICS_LOG_INTERVAL_SECONDS = 10.0


class BBoxResponse(BaseModel):
    """单帧检测框响应，坐标原点为图像左上角。"""

    x1: float = Field(description="框左上角 x 坐标")
    y1: float = Field(description="框左上角 y 坐标")
    x2: float = Field(description="框右下角 x 坐标")
    y2: float = Field(description="框右下角 y 坐标")


class DetectionResponse(BaseModel):
    """单帧模型检测结果。"""

    class_id: int = Field(description="模型输出的类别编号")
    disease_type: str = Field(description="模型输出的类别名称")
    confidence: float = Field(description="当前帧模型置信度")
    bbox: BBoxResponse = Field(description="当前帧原始检测框")


class DetectResponse(BaseModel):
    """无状态单帧检测响应。"""

    session_id: str = Field(description="调用方提供的检测会话 ID")
    frame_id: int = Field(description="调用方提供的帧序号")
    image_width: int = Field(description="输入图像宽度")
    image_height: int = Field(description="输入图像高度")
    detections: list[DetectionResponse] = Field(description="当前帧原始检测框列表")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str
    model: str
    detector_ids: list[str]


@dataclass
class _InferenceWindow:
    """一个任务与模型在当前统计窗口内的检测性能数据。"""

    started_at: float
    last_request_at: float
    frame_count: int = 0
    decode_seconds: float = 0.0
    inference_seconds: float = 0.0
    request_seconds: float = 0.0


class _InferenceMetrics:
    """按任务和模型定期汇总实际检测吞吐及请求耗时。"""

    def __init__(
        self,
        log_interval_seconds: float,
        session_ttl_seconds: int,
    ) -> None:
        self._log_interval_seconds = log_interval_seconds
        self._session_ttl_seconds = session_ttl_seconds
        self._windows: dict[tuple[str, str], _InferenceWindow] = {}
        self._lock = threading.Lock()

    def record(
        self,
        session_id: str,
        detector_id: str,
        decode_seconds: float,
        inference_seconds: float,
        request_seconds: float,
    ) -> None:
        """记录一次成功处理的帧，达到窗口时长后输出性能摘要。"""
        now = time.monotonic()
        key = (session_id, detector_id)
        summary: tuple[int, float, float, float, float] | None = None

        with self._lock:
            stale = [
                session_key
                for session_key, window in self._windows.items()
                if now - window.last_request_at > self._session_ttl_seconds
            ]
            for session_key in stale:
                self._windows.pop(session_key)

            window = self._windows.get(key)
            if window is None:
                window = _InferenceWindow(now, now)
                self._windows[key] = window

            window.last_request_at = now
            window.frame_count += 1
            window.decode_seconds += decode_seconds
            window.inference_seconds += inference_seconds
            window.request_seconds += request_seconds
            elapsed = now - window.started_at
            if elapsed >= self._log_interval_seconds:
                summary = (
                    window.frame_count,
                    elapsed,
                    window.decode_seconds,
                    window.inference_seconds,
                    window.request_seconds,
                )
                window.started_at = now
                window.frame_count = 0
                window.decode_seconds = 0.0
                window.inference_seconds = 0.0
                window.request_seconds = 0.0

        if summary is None:
            return
        frames, elapsed, decode, inference, request = summary
        LOGGER.info(
            "检测性能 session_id=%s detector_id=%s window=%.1fs frames=%d "
            "throughput=%.2f frames/s avg_decode=%.1fms avg_model=%.1fms "
            "avg_handler=%.1fms",
            session_id,
            detector_id,
            elapsed,
            frames,
            frames / elapsed,
            decode * 1000 / frames,
            inference * 1000 / frames,
            request * 1000 / frames,
        )


class _AppState:
    """应用级共享资源。"""

    def __init__(
        self,
        settings: DiseaseServiceSettings,
        detector: DetectorRegistry,
    ) -> None:
        self.settings = settings
        self.detector = detector
        self.metrics = _InferenceMetrics(METRICS_LOG_INTERVAL_SECONDS, 300)


def create_app(
    config_path: str | Path,
    detector: DetectorBackend | None = None,
) -> FastAPI:
    """创建病害检测 FastAPI 应用。

    Args:
        config_path: 病害服务 YAML 配置路径。
        detector: 可选的模型后端。生产环境留空以加载 Ultralytics；
            测试或替换模型时可传入实现 DetectorBackend 的对象。

    Returns:
        已注册检测和健康检查接口的 FastAPI 应用。

    Example:
        >>> app = create_app("code/disease_service/config.yaml")
        >>> app = create_app("config.yaml", detector=MyDetector())
    """
    settings = load_settings(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        overrides = (
            {settings.default_model_id: detector} if detector is not None else None
        )
        app.state.service = _AppState(
            settings,
            DetectorRegistry(settings.models, backend_overrides=overrides),
        )
        yield

    app = FastAPI(
        title="无状态无人机病害检测服务",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """返回服务和已注册模型状态。

        Returns:
            健康检查信息。

        Example:
            >>> response = client.get("/health")
            >>> response.json()["status"]
            'ok'
        """
        state: _AppState = app.state.service
        return HealthResponse(
            status="ok",
            model=state.settings.model.weights.name,
            detector_ids=list(state.detector.model_ids),
        )

    @app.post("/v1/detect", response_model=DetectResponse)
    async def detect(
        session_id: str = Form(..., description="检测会话，例如无人机任务 ID"),
        frame_id: int = Form(..., ge=0, description="连续帧序号，从 0 或 1 开始"),
        detector_id: str | None = Form(
            None, description="服务端已配置的病害模型 ID"
        ),
        image: UploadFile = File(..., description="JPEG 或 PNG 图像"),
    ) -> DetectResponse:
        """检测单帧图像并返回原始检测框，不维护跨帧状态。

        Args:
            session_id: 调用方会话标识，用于日志和响应关联。
            frame_id: 调用方帧序号，用于响应关联。
            detector_id: 可选模型 ID；未提供时使用服务默认模型。
            image: JPEG 或 PNG 文件。

        Returns:
            当前帧模型检测列表。

        Example:
            >>> files = {"image": ("frame.jpg", jpeg_bytes, "image/jpeg")}
            >>> data = {"session_id": "drone-01", "frame_id": "1"}
            >>> client.post("/v1/detect", files=files, data=data).json()["detections"]
            []
        """
        request_started_at = time.perf_counter()
        state: _AppState = app.state.service
        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            raise HTTPException(status_code=422, detail="session_id 不能为空")

        decode_started_at = time.perf_counter()
        raw = await image.read()
        max_bytes = state.settings.api.max_upload_mb * 1024 * 1024
        if not raw or len(raw) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"图像大小必须在 1 到 {state.settings.api.max_upload_mb} MB 之间",
            )
        try:
            with Image.open(io.BytesIO(raw)) as source:
                source.load()
                width, height = source.size
                image_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=415, detail="无法解析上传图像") from exc
        decode_seconds = time.perf_counter() - decode_started_at

        if width == 0 or height == 0:
            raise HTTPException(status_code=422, detail="图像宽高必须大于零")

        try:
            selected_detector_id = (detector_id or state.settings.default_model_id).strip()
            inference_started_at = time.perf_counter()
            raw_detections = await run_in_threadpool(
                state.detector.detect, selected_detector_id, image_rgb
            )
            inference_seconds = time.perf_counter() - inference_started_at
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        response = DetectResponse(
            session_id=normalized_session_id,
            frame_id=frame_id,
            image_width=width,
            image_height=height,
            detections=[
                DetectionResponse(
                    disease_type=item.disease_type,
                    class_id=item.class_id,
                    confidence=item.confidence,
                    bbox=BBoxResponse(
                        x1=item.bbox[0],
                        y1=item.bbox[1],
                        x2=item.bbox[2],
                        y2=item.bbox[3],
                    ),
                )
                for item in raw_detections
            ],
        )
        state.metrics.record(
            session_id=normalized_session_id,
            detector_id=selected_detector_id,
            decode_seconds=decode_seconds,
            inference_seconds=inference_seconds,
            request_seconds=time.perf_counter() - request_started_at,
        )
        return response

    return app