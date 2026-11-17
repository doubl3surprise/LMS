#!/usr/bin/env python3
"""Join component counters to request windows. Exclusive/overlap scope remains explicit."""
import argparse,json
from pathlib import Path

def join_requests(rows,samples):
 output=[]
 for row in rows:
  lo=row['wall_start'];hi=row.get('answer_end',row['wall_end']);window=[s for s in samples if lo<=s['timestamp']<=hi]
  entry={'request_id':row['request_id'],'start':lo,'end':hi,'sample_count':len(window),'components':{},'scope':'sample-window component deltas, excludes Judge; not exclusive causal attribution; endpoints undersample'}
  for name in ['sglang','qdrant','benchmark']:
   values=[s['components'].get(name,{}) for s in window];result={}
   for key in ['cpu_user_s','cpu_system_s','read_bytes','write_bytes','rchar','wchar']:
    v=[x[key] for x in values if isinstance(x.get(key),(int,float))];result[key]=v[-1]-v[0] if len(v)>1 and v[-1]>=v[0] else None
   entry['components'][name]=result
  output.append(entry)
 return {'scope':'sampled_request_windows','requests':output,'limitations':['Short requests may contain fewer than two host samples; missing remains null.','GPU kernel request attribution is in request_profiles.json, separate from sampler windows.']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();d=Path(a.run_dir)
 read=lambda f:[json.loads(l) for l in (d/f).read_text().splitlines() if l.strip()]
 (d/'request_resources.json').write_text(json.dumps(join_requests(read('trace.jsonl'),read('resource_samples.jsonl')),indent=2))
if __name__=='__main__':main()
