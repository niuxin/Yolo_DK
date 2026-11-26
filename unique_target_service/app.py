# -*- coding: utf-8 -*-
"""接收单帧检测结果并输出跨帧唯一目标事件。"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .unique_target_core import UniqueTargetCore, export_event


class TargetDetection(BaseModel):
    """检测服务返回的单个原始框。"""

    class_id: int = 0
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: tuple[float, float, float, float]


@dataclass
class _Session:
    core: UniqueTargetCore
    lock: threading.Lock
    last_frame_id: int = -1
    last_timestamp_ms: float = -1


class UniqueTargetEngine:
    """按会话顺序处理图像帧及外部检测框。"""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        core_factory: Callable[..., UniqueTargetCore] = UniqueTargetCore,
    ) -> None:
        self.output_dir = Path(
            output_dir
            or Path(__file__).with_name("data")
        ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._core_factory = core_factory
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    @staticmethod
    def session_key(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]

    def _session(self, session_id: str) -> tuple[str, _Session]:
        normalized = session_id.strip()
        if not normalized or len(normalized) > 128:
            raise ValueError("session_id 长度必须在 1 到 128 之间")
        key = self.session_key(normalized)
        with self._lock:
            state = self._sessions.get(key)
            if state is None:
                state = _Session(
                    core=self._core_factory(stream_id=f"s{key[:16]}"),
                    lock=threading.Lock(),
                )
                self._sessions[key] = state
        return key, state

    def process(
        self,
        session_id: str,
        frame_id: int,
        timestamp_ms: float,
        image_bgr: np.ndarray,
        detections: list[TargetDetection],
    ) -> dict[str, object]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image 必须是 HxWx3 BGR 图像")
        if not math.isfinite(timestamp_ms) or timestamp_ms < 0:
            raise ValueError("timestamp_ms 必须是非负有限数")
        key, state = self._session(session_id)
        with state.lock:
            if frame_id <= state.last_frame_id or timestamp_ms <= state.last_timestamp_ms:
                raise ValueError("frame_id 和 timestamp_ms 必须严格递增")
            core_detections = [
                {
                    "class_id": item.class_id,
                    "class_name": item.class_name,
                    "confidence": item.confidence,
                    "bbox_xyxy": item.bbox_xyxy,
                }
                for item in detections
            ]
            events = state.core.push(
                image_bgr, frame_id, timestamp_ms, core_detections
            )
            response_events = []
            for event in events:
                target_id = str(event["target_id"])
                export_event(event, self.output_dir / key / "targets")
                response_events.append(
                    {
                        "target_id": target_id,
                        "class_id": int(event["class_id"]),
                        "class_name": str(event["class_name"]),
                        "frame_id": int(event["frame_id"]),
                        "timestamp_ms": float(event["timestamp_ms"]),
                        "confirmed_frame_id": int(event["confirmed_frame_id"]),
                        "confirmed_timestamp_ms": float(
                            event["confirmed_timestamp_ms"]
                        ),
                        "bbox_xyxy": [float(value) for value in event["bbox_xyxy"]],
                        "confidence": float(event["confidence"]),
                        "quality": event["quality"],
                        "evidence_frames": event["evidence_frames"],
                        "original_image_url": (
                            f"/v1/assets/{key}/{target_id}/original.jpg"
                        ),
                    }
                )
            state.last_frame_id = frame_id
            state.last_timestamp_ms = timestamp_ms
            return {
                "session_id": session_id,
                "frame_id": frame_id,
                "events": response_events,
            }

    def close(self, session_id: str) -> dict[str, object]:
        key = self.session_key(session_id.strip())
        with self._lock:
            state = self._sessions.get(key)
        if state is None:
            return {"session_id": session_id, "status": "not_active"}
        with state.lock:
            state.core.flush()
            review_path = self.output_dir / key / "pending_review.json"
            review_path.parent.mkdir(parents=True, exist_ok=True)
            review_path.write_text(
                json.dumps(state.core.review_items, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary = state.core.stats()
        with self._lock:
            self._sessions.pop(key, None)
        return {"session_id": session_id, "status": "closed", "summary": summary}


def create_app(
    engine: UniqueTargetEngine | None = None,
) -> FastAPI:
    engine = engine or UniqueTargetEngine()
    app = FastAPI(title="唯一目标与代表帧服务", version="1.0.0")
    app.state.engine = engine

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/process")
    async def process_frame(
        session_id: str = Form(...),
        frame_id: int = Form(..., ge=0),
        timestamp_ms: float = Form(..., ge=0, allow_inf_nan=False),
        detections: str = Form(...),
        image: UploadFile = File(...),
    ) -> dict[str, object]:
        try:
            parsed = json.loads(detections)
            if not isinstance(parsed, list):
                raise ValueError("detections 必须是 JSON 数组")
            boxes = [TargetDetection.model_validate(item) for item in parsed]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        raw = await image.read(20 * 1024 * 1024 + 1)
        if not raw or len(raw) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="图像大小必须在 1 到 20 MiB 之间")
        image_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(status_code=415, detail="无法解析上传图像")
        if image_bgr.shape[0] * image_bgr.shape[1] > 40_000_000:
            raise HTTPException(status_code=413, detail="图像尺寸超过限制")
        try:
            return engine.process(
                session_id, frame_id, timestamp_ms, image_bgr, boxes
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/close")
    def close_session(session_id: str = Form(...)) -> dict[str, object]:
        return engine.close(session_id)

    @app.get("/v1/assets/{session_key}/{target_id}/original.jpg")
    def original_image(session_key: str, target_id: str) -> FileResponse:
        if not re.fullmatch(r"[a-f0-9]{32}", session_key) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,80}", target_id
        ):
            raise HTTPException(status_code=404, detail="图片不存在")
        path = (
            engine.output_dir / session_key / "targets" / target_id / "original.jpg"
        ).resolve()
        if not path.is_relative_to(engine.output_dir) or not path.is_file():
            raise HTTPException(status_code=404, detail="图片不存在")
        return FileResponse(path, media_type="image/jpeg")

    return app
