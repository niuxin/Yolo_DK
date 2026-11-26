"""启动独立唯一目标服务。"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import UniqueTargetEngine, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="唯一目标与代表帧服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("data"),
    )
    args = parser.parse_args()
    uvicorn.run(
        create_app(UniqueTargetEngine(args.output_dir)),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
