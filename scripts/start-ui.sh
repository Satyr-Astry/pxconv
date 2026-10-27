#!/usr/bin/env sh
# pxconv local web UI launcher (Linux / macOS)
cd "$(dirname "$0")/.." || exit 1
exec "${PYTHON:-python3}" ui/server.py "${1:-8765}"
