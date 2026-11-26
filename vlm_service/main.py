"""Start the independent VLM HTTP service."""
from __future__ import annotations

import argparse
import os

import uvicorn

from .app import create_app
from .domain import VLMSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM 图像描述服务")
    parser.add_argument("--host", default=os.environ.get("VLM_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLM_PORT", "8084")))
    parser.add_argument(
        "--provider-base-url",
        default=os.environ.get("VLM_PROVIDER_BASE_URL", ""),
    )
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", ""))
    parser.add_argument(
        "--api-key-env",
        default=os.environ.get("VLM_API_KEY_ENV", "VLM_API_KEY"),
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=float(os.environ.get("VLM_TIMEOUT_S", "60")),
    )
    parser.add_argument(
        "--crop-margin-ratio",
        type=float,
        default=float(os.environ.get("VLM_CROP_MARGIN_RATIO", "0.1")),
    )
    args = parser.parse_args()
    if not args.provider_base_url or not args.model:
        parser.error("通过参数或环境变量提供 provider base URL 和 model")
    settings = VLMSettings(
        enabled=True,
        base_url=args.provider_base_url.rstrip("/"),
        model=args.model,
        api_key_env=args.api_key_env,
        timeout_s=args.timeout_s,
        crop_margin_ratio=args.crop_margin_ratio,
        jpeg_quality=90,
    )
    uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
