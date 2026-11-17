#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 -c 'import hashlib,json; from pathlib import Path; m=json.loads(Path("experiment/files.sha256.json").read_text()); assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in m.items()), "Pinned template/tokenizer changed"'
docker compose -f compose.yaml up -d --pull never "$@"
