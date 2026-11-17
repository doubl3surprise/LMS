"""Stream compact the Task5 KV/Mamba metrics without changing raw samples."""
from pathlib import Path
import json,gzip,shutil,sys,hashlib
p=Path(sys.argv[1]);assert (p/'DONE').exists();source=p/'metrics_samples.jsonl';target=p/'metrics_task5.jsonl'
n=0
with source.open() as f,target.open('w') as g:
 for line in f:
  s=json.loads(line);s['metrics']=[m for m in s.get('metrics',[]) if any(t in m.get('name','') for t in ['kv_','mamba_'])];g.write(json.dumps(s)+'\n');n+=1
with source.open('rb') as f,gzip.open(p/'metrics_samples.full.jsonl.gz','wb',compresslevel=3) as g:shutil.copyfileobj(f,g)
(p/'metrics_provenance.json').write_text(json.dumps({'raw_name':source.name,'analysis_name':target.name,'raw_bytes':source.stat().st_size,'raw_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'analysis_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'samples':n,'filter':'metric name contains kv_ or mamba_; all sample timestamps/other fields retained; raw unchanged'},indent=2))
print('METRICS_COMPACTED',n,target.stat().st_size,flush=True)
