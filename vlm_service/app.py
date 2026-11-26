"""HTTP API for image-region descriptions."""
from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .domain import (
    BBox,
    VLMProviderFactory,
    VLMService,
    VLMSettings,
    create_vlm_service,
)


class DescriptionResponse(BaseModel):
    disease_type: str
    summary: str
    severity: Literal["low", "medium", "high"]
    suggestion: str


def create_app(
    settings: VLMSettings,
    provider_factory: VLMProviderFactory | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = (
            create_vlm_service(settings)
            if provider_factory is None
            else create_vlm_service(settings, provider_factory)
        )
        try:
            yield
        finally:
            service: VLMService = app.state.service
            service.close()

    app = FastAPI(title="VLM 图像描述服务", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/describe", response_model=DescriptionResponse)
    async def describe(
        disease_type: str = Form(..., min_length=1, max_length=128),
        x1: float = Form(...),
        y1: float = Form(...),
        x2: float = Form(...),
        y2: float = Form(...),
        image: UploadFile = File(...),
    ) -> DescriptionResponse:
        raw = await image.read(20 * 1024 * 1024 + 1)
        if not raw or len(raw) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="图像大小必须在 1 到 20 MiB 之间")
        try:
            with Image.open(io.BytesIO(raw)) as source:
                source.load()
                image_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=415, detail="无法解析上传图像") from exc
        bbox = BBox(x1, y1, x2, y2)
        service: VLMService = app.state.service
        try:
            result = await run_in_threadpool(
                service.describe, image_rgb, bbox, disease_type.strip()
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return DescriptionResponse(
            disease_type=disease_type.strip(),
            summary=result.summary,
            severity=result.severity,
            suggestion=result.suggestion,
        )

    return app
