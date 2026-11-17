#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p downloads logs models
exec 9>downloads/setup.lock
flock -n 9 || { echo '已有安装任务正在运行。' >&2; exit 1; }
docker image inspect lmsysorg/sglang:qwen38-27b >/dev/null
python3 -u scripts/download_model.py 2>&1 | tee logs/model-download.log
./start.sh
python3 -u scripts/verify_service.py 2>&1 | tee logs/service-verification.log
