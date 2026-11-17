#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
docker compose ps -a
if [[ -f logs/model-download.log ]]; then python3 scripts/progress.py; fi
curl -fsS --max-time 5 http://127.0.0.1:8000/health || true
echo
