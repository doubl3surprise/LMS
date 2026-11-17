"""Read back original source collection and compare to pre-run manifest."""
import sys,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from task5_run import _read_source_records,_json_hash
from tracer.config import get_config
from qdrant_client import QdrantClient
host,run=sys.argv[1:3];cfg=get_config(host);manifest=json.loads((Path(run)/'manifest.json').read_text())
client=QdrantClient(url=cfg.qdrant_url) if cfg.qdrant_mode=='server' else QdrantClient(path=cfg.qdrant_path)
try:records=_read_source_records(client,cfg.collection,cfg.user_id)
finally:client.close()
result={'records':len(records),'source_sha256_after':_json_hash(records),'source_sha256_before':manifest['source_records_sha256'],'same':len(records)==879 and _json_hash(records)==manifest['source_records_sha256']}
assert result['same'],result
(Path(run)/'source_readback.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
