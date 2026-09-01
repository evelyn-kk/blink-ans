#!/usr/bin/env bash
# 重新生成 requirements.lock.txt。
# 用 importlib.metadata 而非 pip freeze —— uv 建的虚拟环境里没有 pip。
set -euo pipefail
cd "$(dirname "$0")/../.."

{
  echo "# 完整环境快照，含全部传递依赖。用于换机精确复现。"
  echo "# 生成: ./infra/scripts/freeze.sh   （$(date -u +%Y-%m-%d)，Python $(.venv/bin/python -c 'import sys;print("%d.%d"%sys.version_info[:2])')，$(uname -sm)）"
  echo "# 日常开发装 requirements.txt 即可；本文件只在需要复现同一环境时使用。"
  echo
  .venv/bin/python - <<'PY'
import importlib.metadata as m
for n, v in sorted({(d.metadata["Name"], d.version) for d in m.distributions() if d.metadata["Name"]},
                   key=lambda x: x[0].lower()):
    print(f"{n}=={v}")
PY
} > requirements.lock.txt

echo "已写入 requirements.lock.txt（$(grep -c '==' requirements.lock.txt) 个包）"
