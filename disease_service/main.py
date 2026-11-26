# -*- coding: utf-8 -*-
"""启动病害检测 HTTP 服务。"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from .app import METRICS_LOG_INTERVAL_SECONDS, create_app
from .config import load_settings


LOGGER = logging.getLogger(__name__)


def main() -> None:
    """读取命令行参数并启动 GPU 病害检测服务。

    Example:
        python -m code.disease_service.main --config code/disease_service/config.yaml
    """
    parser = argparse.ArgumentParser(description="无状态无人机病害检测服务")
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
    LOGGER.info(
        "病害检测服务启动: host=%s port=%d default_detector=%s device=%s "
        "image_size=%d metrics_interval=%.0fs",
        settings.api.host,
        settings.api.port,
        settings.default_model_id,
        settings.model.device,
        settings.model.image_size,
        METRICS_LOG_INTERVAL_SECONDS,
    )
    uvicorn.run(create_app(args.config), host=settings.api.host, port=settings.api.port)


if __name__ == "__main__":
    main()