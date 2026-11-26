# -*- coding: utf-8 -*-
"""AI 网关会话和标注流控制 API。"""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .task_manager import (
    DetectionTask,
    OutputAlreadyExists,
    OutputManager,
    OutputSnapshot,
    OutputTask,
    SessionSnapshot,
    TaskAlreadyExists,
    TaskCapacityExceeded,
    TaskManager,
)

LOGGER = logging.getLogger(__name__)


class ContractModel(BaseModel):
    """允许协议 camelCase 字段和 Python snake_case 字段互通。"""

    model_config = ConfigDict(populate_by_name=True)


def _validate_stream_url(value: str, field_name: str) -> str:
    normalized = value.strip()
    parsed = urlsplit(normalized)
    if parsed.scheme.lower() not in {"rtsp", "rtmp", "http", "https"} or not parsed.hostname:
        raise ValueError(f"{field_name} 必须是 RTSP、RTMP 或 HTTP(S) 地址")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"{field_name} 端口格式无效") from exc
    return normalized


def _validate_algorithms(value: list[str]) -> list[str]:
    algorithms = [item.strip() for item in value]
    if not algorithms or any(
        not item or len(item) > 128 or "/" in item or "\\" in item
        for item in algorithms
    ):
        raise ValueError("algorithmIds 必须包含有效算法 ID")
    if len(set(algorithms)) != len(algorithms):
        raise ValueError("algorithmIds 不能包含重复值")
    return algorithms


class CreateSessionRequest(ContractModel):
    """创建实时检测会话请求。"""

    session_id: str = Field(alias="sessionId", min_length=1, max_length=128)
    job_id: str | None = Field(default=None, alias="jobId", max_length=128)
    stream_url: str = Field(alias="streamUrl", min_length=1, max_length=2048)
    algorithm_ids: list[str] = Field(alias="algorithmIds", min_length=1, max_length=32)
    callback_url: str = Field(alias="callbackUrl", min_length=1, max_length=2048)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "/" in normalized or "\\" in normalized:
            raise ValueError("sessionId 无效")
        return normalized

    @field_validator("job_id")
    @classmethod
    def normalize_job_id(cls, value: str | None) -> str | None:
        normalized = value.strip() if value is not None else None
        return normalized or None

    @field_validator("stream_url")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _validate_stream_url(value, "streamUrl")

    @field_validator("algorithm_ids")
    @classmethod
    def validate_algorithm_list(cls, value: list[str]) -> list[str]:
        return _validate_algorithms(value)

    @field_validator("callback_url")
    @classmethod
    def validate_callback(cls, value: str) -> str:
        normalized = value.strip()
        parsed = urlsplit(normalized)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("callbackUrl 必须是 HTTP(S) 完整地址")
        if parsed.path != "/manage/api/v1/ai-gateway/events":
            raise ValueError("callbackUrl path 必须是 /manage/api/v1/ai-gateway/events")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError("callbackUrl 不能包含凭证或 fragment")
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("callbackUrl 端口格式无效") from exc
        return normalized


class StopSessionRequest(ContractModel):
    """停止实时检测会话请求。"""

    reason: str = Field(default="operator_requested", min_length=1, max_length=128)


class OutputRequest(ContractModel):
    """开始标注流发布请求。"""

    stream_url: str = Field(alias="streamUrl", min_length=1, max_length=2048)
    algorithm_ids: list[str] = Field(alias="algorithmIds", min_length=1, max_length=32)
    output_stream_url: str = Field(
        alias="outputStreamUrl", min_length=1, max_length=2048
    )

    @field_validator("stream_url")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _validate_stream_url(value, "streamUrl")

    @field_validator("algorithm_ids")
    @classmethod
    def validate_algorithm_list(cls, value: list[str]) -> list[str]:
        return _validate_algorithms(value)

    @field_validator("output_stream_url")
    @classmethod
    def validate_output(cls, value: str) -> str:
        normalized = value.strip()
        parsed = urlsplit(normalized)
        if parsed.scheme.lower() != "rtmp" or not parsed.hostname:
            raise ValueError("outputStreamUrl 必须是 RTMP 地址")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("outputStreamUrl 必须是不含密钥的干净地址")
        path_parts = parsed.path.strip("/").split("/")
        if (
            len(path_parts) < 2
            or path_parts[0] != "ai"
            or any(not part for part in path_parts)
        ):
            raise ValueError("outputStreamUrl 必须使用 ai app 路径")
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("outputStreamUrl 端口格式无效") from exc
        return normalized


class SessionResponse(ContractModel):
    session_id: str = Field(alias="sessionId")
    gateway_session_id: str = Field(alias="gatewaySessionId")
    status: str
    accepted_at: int = Field(alias="acceptedAt")


class OutputResponse(ContractModel):
    session_id: str = Field(alias="sessionId")
    output_task_id: str = Field(alias="outputTaskId")
    status: str


def create_app(manager: TaskManager, output_manager: OutputManager) -> FastAPI:
    """创建实现网关会话和标注流协议的应用。"""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        output_manager.shutdown()
        manager.shutdown()

    app = FastAPI(title="实时视频 AI 网关", version="2.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/api/v1/live-sessions",
        response_model=SessionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_session(request: CreateSessionRequest) -> SessionResponse:
        try:
            snapshot = manager.start(
                DetectionTask(
                    session_id=request.session_id,
                    job_id=request.job_id,
                    stream_url=request.stream_url,
                    algorithm_ids=tuple(request.algorithm_ids),
                    callback_url=request.callback_url,
                )
            )
        except TaskAlreadyExists as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TaskCapacityExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _session_response(snapshot, status_override="accepted")

    @app.post(
        "/api/v1/live-sessions/{session_id}:stop",
        response_model=SessionResponse,
    )
    async def stop_session(
        session_id: str, request: StopSessionRequest | None = None
    ) -> SessionResponse:
        reason = request.reason if request is not None else "operator_requested"
        LOGGER.info("收到会话停止请求: session_id=%s reason=%s", session_id, reason)
        snapshot = manager.get(session_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        try:
            await run_in_threadpool(output_manager.stop, session_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        snapshot = manager.stop(session_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return _session_response(snapshot)

    @app.post(
        "/api/v1/live-sessions/{session_id}/output",
        response_model=OutputResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def start_output(
        session_id: str, request: OutputRequest
    ) -> OutputResponse:
        session = manager.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="检测会话不存在")
        if session.status in {"completed", "stopped", "failed", "stopping"}:
            raise HTTPException(status_code=409, detail="检测会话已结束或正在停止")
        try:
            snapshot = output_manager.start(
                OutputTask(
                    session_id=session_id,
                    stream_url=request.stream_url,
                    algorithm_ids=tuple(request.algorithm_ids),
                    output_stream_url=request.output_stream_url,
                )
            )
        except OutputAlreadyExists as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _output_response(snapshot)

    @app.get(
        "/api/v1/live-sessions/{session_id}/output",
        response_model=OutputResponse,
    )
    async def get_output(session_id: str) -> OutputResponse:
        snapshot = output_manager.get(session_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="标注流任务不存在")
        return _output_response(snapshot)

    @app.delete(
        "/api/v1/live-sessions/{session_id}/output",
        response_model=OutputResponse,
    )
    async def stop_output(session_id: str) -> OutputResponse:
        try:
            snapshot = await run_in_threadpool(output_manager.stop, session_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if snapshot is None:
            raise HTTPException(status_code=404, detail="标注流任务不存在")
        return _output_response(snapshot)

    return app


def _session_response(
    snapshot: SessionSnapshot, status_override: str | None = None
) -> SessionResponse:
    return SessionResponse(
        session_id=snapshot.session_id,
        gateway_session_id=snapshot.gateway_session_id,
        status=status_override or snapshot.status,
        accepted_at=snapshot.accepted_at,
    )


def _output_response(snapshot: OutputSnapshot) -> OutputResponse:
    return OutputResponse(
        session_id=snapshot.session_id,
        output_task_id=snapshot.output_task_id,
        status=snapshot.status,
    )
