#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_PYTHON="${HOUSE_PYTHON:-$PACKAGE_DIR/.venv/bin/python}"
if [[ ! -x "$SERVICE_PYTHON" ]]; then
  echo "请先运行 bash install.sh gpu（或 cpu），或用 HOUSE_PYTHON 指定已有兼容环境。" >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$SERVICE_PYTHON" "$PACKAGE_DIR/serve.py" "$@"
