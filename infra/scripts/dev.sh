#!/usr/bin/env bash
# 本地开发启动。模型冷加载约 60s，启动完成前 /healthz 返回 status=loading。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec .venv/bin/uvicorn apps.gateway.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" "$@"
