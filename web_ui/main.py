# -*- coding: utf-8 -*-
"""可视化界面服务入口。

独立进程、独立端口（默认 8085），只读病害库并自行拉一路 MJPEG 预览，
不修改 ``stream_service`` 的任何代码或运行行为。启动方式::

    python -m web_ui.main --config web_ui/config.yaml
"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import create_router, resolve_static_dir
from .config import UiSettings, load_settings
from .events import EventBus
from .preview import PreviewManager
from .records import RecordRepository
from .watcher import RecordWatcher


LOGGER = logging.getLogger(__name__)


def create_app(settings: UiSettings) -> FastAPI:
    """创建界面服务应用。

    Args:
        settings: 界面服务配置。

    Returns:
        挂载了 API、WebSocket 和静态前端的 FastAPI 应用。

    Example:
        >>> app = create_app(load_settings("web_ui/config.yaml"))
    """
    repository = RecordRepository(settings.storage)
    bus = EventBus(settings.server.event_queue_size)
    previews = PreviewManager(settings.preview)
    watcher = RecordWatcher(
        repository, bus, settings.server.record_throttle_per_second
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        watcher.prime()
        watcher.start()
        LOGGER.info(
            "界面服务已启动: 病害库=%s 预览=%s",
            repository.database_path,
            "启用" if previews.enabled else "关闭",
        )
        try:
            yield
        finally:
            watcher.close()
            previews.close_all()

    app = FastAPI(
        title="无人机病害检测可视化界面",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(
        create_router(settings, repository, bus, previews)
    )

    static_dir = resolve_static_dir()
    if static_dir.is_dir():
        app.mount(
            "/static", StaticFiles(directory=str(static_dir)), name="static"
        )

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            """返回界面首页。"""
            return FileResponse(static_dir / "index.html")

    return app


def main() -> None:
    """解析参数、加载配置并启动界面服务。"""
    parser = argparse.ArgumentParser(description="无人机病害检测可视化界面服务")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="界面服务 YAML 配置路径",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = load_settings(args.config)
    app = create_app(settings)

    import uvicorn

    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
