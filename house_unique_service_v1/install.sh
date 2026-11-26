#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-gpu}"
case "$MODE" in gpu) TORCH_INDEX=https://download.pytorch.org/whl/cu121;; cpu) TORCH_INDEX=https://download.pytorch.org/whl/cpu;; *) echo '用法: bash install.sh [gpu|cpu]' >&2; exit 2;; esac
BOOTSTRAP_PYTHON="${PYTHON_BIN:-python3}"
"$BOOTSTRAP_PYTHON" -c 'import sys; assert (3,10)<=sys.version_info[:2]<(3,13), "Use Python 3.10 (recommended), 3.11 or 3.12"'
mkdir -p "$PACKAGE_DIR/.cache/pip" "$PACKAGE_DIR/.tmp"
export PIP_CACHE_DIR="$PACKAGE_DIR/.cache/pip" TMPDIR="$PACKAGE_DIR/.tmp"
"$BOOTSTRAP_PYTHON" -m venv "$PACKAGE_DIR/.venv"
"$PACKAGE_DIR/.venv/bin/python" -m pip install --upgrade pip
"$PACKAGE_DIR/.venv/bin/python" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url "$TORCH_INDEX"
"$PACKAGE_DIR/.venv/bin/python" -m pip install -r "$PACKAGE_DIR/requirements.txt"
"$PACKAGE_DIR/.venv/bin/python" -m pip check
"$PACKAGE_DIR/.venv/bin/python" -m pip freeze > "$PACKAGE_DIR/verification/installed_environment.txt"
echo '安装完成。GPU: ./start.sh --gpu 0；CPU: ./start.sh --gpu cpu'
