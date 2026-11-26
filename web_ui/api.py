# -*- coding: utf-8 -*-
"""可视化界面 HTTP/WebSocket 接口。

提供历史记录查询和独立视频预览，不控制 AI 网关检测会话，也不接收网关回调。
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

from .config import UiSettings
from .events import EventBus
from .preview import PreviewManager
from .records import RecordRepository


LOGGER = logging.getLogger(__name__)

MJPEG_BOUNDARY = "frame"


class OpenPreviewRequest(BaseModel):
    """打开预览流请求。"""

    session_id: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=2048)


def create_router(
    settings: UiSettings,
    repository: RecordRepository,
    bus: EventBus,
    previews: PreviewManager,
) -> APIRouter:
    """创建界面服务路由。

    Args:
        settings: 界面服务配置。
        repository: 只读病害库访问层。
        bus: 事件总线。
        previews: 预览流管理器。
    Returns:
        挂载到 FastAPI 应用的路由对象。
    """
    router = APIRouter()

    @router.get("/api/health")
    async def health() -> dict[str, Any]:
        """返回界面服务及依赖状态。"""
        return {
            "status": "ok",
            "database": {
                "path": str(repository.database_path),
                "available": repository.available(),
            },
            "preview": {
                "enabled": previews.enabled,
                "streams": _preview_status(previews),
            },
            "default_source": settings.server.default_source,
            "subscribers": bus.subscriber_count,
        }

    @router.get("/api/records")
    async def list_records(
        limit: int = Query(default=0, ge=0),
        offset: int = Query(default=0, ge=0),
        session_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """分页查询病害记录，按 id 倒序。"""
        page_size = limit or settings.server.page_size
        if page_size > settings.server.max_page_size:
            page_size = settings.server.max_page_size
        records = await asyncio.to_thread(
            repository.list_records, page_size, offset, session_id, None
        )
        total = await asyncio.to_thread(repository.count_records, session_id)
        return {
            "records": records,
            "total": total,
            "limit": page_size,
            "offset": offset,
        }

    @router.get("/api/records/{record_id}/image")
    async def record_image(record_id: int) -> FileResponse:
        """返回病害记录对应的代表帧图片。"""
        path = await asyncio.to_thread(repository.get_image_path, record_id)
        if path is None:
            raise HTTPException(status_code=404, detail="代表帧图片不存在")
        return FileResponse(path, media_type="image/jpeg")

    @router.get("/api/sessions")
    async def list_sessions(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        """列出最近的检测会话。"""
        sessions = await asyncio.to_thread(repository.list_sessions, limit)
        return {"sessions": sessions}

    @router.post("/api/preview/open")
    async def open_preview(request: OpenPreviewRequest) -> dict[str, Any]:
        """为指定会话打开一路独立 MJPEG 预览流。"""
        if not previews.enabled:
            raise HTTPException(status_code=503, detail="预览功能已在配置中关闭")
        try:
            stream = previews.open(request.session_id, request.source)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "session_id": stream.session_id,
            "source": stream.source,
            "url": f"/api/preview/{stream.session_id}.mjpg",
        }

    @router.delete("/api/preview/{session_id}")
    async def close_preview(session_id: str) -> dict[str, Any]:
        """关闭指定会话的预览流。"""
        closed = previews.close(session_id)
        return {"session_id": session_id, "closed": closed}

    @router.get("/api/preview/{session_id}.mjpg")
    async def preview_stream(session_id: str, request: Request) -> StreamingResponse:
        """以 ``multipart/x-mixed-replace`` 输出预览帧，浏览器 ``<img>`` 可直接显示。"""
        stream = previews.get(session_id)
        if stream is None:
            raise HTTPException(status_code=404, detail="预览流未打开")
        subscriber = stream.subscribe()

        async def generate():
            try:
                async for frame in stream.iter_frames(
                    subscriber, is_cancelled=request.is_disconnected
                ):
                    yield (
                        b"--" + MJPEG_BOUNDARY.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                        + frame + b"\r\n"
                    )
            finally:
                stream.unsubscribe(subscriber)

        return StreamingResponse(
            generate(),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @router.websocket("/ws/events")
    async def events_socket(websocket: WebSocket) -> None:
        """推送病害记录事件；客户端可发送任意文本作为心跳。"""
        await websocket.accept()
        subscription = bus.subscribe()
        # 必须并发读取客户端消息，否则 starlette 不会感知断开，
        # 页面刷新后旧订阅会一直留在总线上造成泄漏。
        receiver = asyncio.create_task(_drain(websocket))
        try:
            await websocket.send_text(
                json.dumps(
                    {"type": "hello", "subscribers": bus.subscriber_count},
                    ensure_ascii=False,
                )
            )
            while True:
                if receiver.done():
                    return
                event = await subscription.get(timeout=1.0)
                if event is None:
                    continue
                await websocket.send_text(json.dumps(event, ensure_ascii=False))
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001 - 连接异常直接结束该订阅
            LOGGER.debug("WebSocket 订阅结束", exc_info=True)
        finally:
            receiver.cancel()
            bus.unsubscribe(subscription)

    return router


async def _drain(websocket: WebSocket) -> None:
    """持续读取并丢弃客户端消息，用于及时感知连接断开。"""
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001 - 断开方式多样，统一视为结束
        return


def _preview_status(previews: PreviewManager) -> list[dict[str, Any]]:
    """汇总所有预览流的运行状态。"""
    result: list[dict[str, Any]] = []
    for session_id in previews.session_ids():
        stream = previews.get(session_id)
        if stream is None:
            continue
        stats = stream.stats
        result.append(
            {
                "session_id": session_id,
                "source": stream.source,
                "frames": stats.frames,
                "subscribers": stats.subscribers,
                "restarts": stats.restarts,
                "last_frame_at": stats.last_frame_at,
                "error": stats.error,
            }
        )
    return result


def resolve_static_dir() -> Path:
    """返回前端静态资源目录。"""
    return Path(__file__).with_name("static")
