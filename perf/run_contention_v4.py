#!/usr/bin/env python3
"""Bounded component co-run experiment; does not alter memories or server settings.
No same-request dependency is bypassed: auxiliary work belongs to other requests.
"""
import argparse,concurrent.futures,json,random,threading,time
from pathlib import Path
from run_trace_v4 import generate,tokenize
from trace_core_v4 import slowdown

def main():
 from tracer.config import get_config
 from tracer.retriever import retrieve,get_embedder
 from qdrant_client import QdrantClient
 from qdrant_client.models import Filter,FieldCondition,MatchValue
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--platform',choices=['dgx','5880ada'],required=True);p.add_argument('--out-dir',required=True);p.add_argument('--repetitions',type=int,default=3);p.add_argument('--max-tokens',type=int,default=32);a=p.parse_args()
 cfg=get_config(a.platform);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
 if (out/'contention.json').exists():raise FileExistsError('Refuse overwrite contention result')
 qc=QdrantClient(url=cfg.qdrant_url) if cfg.qdrant_mode=='server' else QdrantClient(path=cfg.qdrant_path)
 try:
  q=json.loads(Path(cfg.dataset_path).read_text())[0]['qa'][0]['question'];embed=get_embedder();vec=list(embed.embed([q]))[0].tolist();flt=Filter(must=[FieldCondition(key='user_id',match=MatchValue(value=cfg.user_id))])
  ids=tokenize(cfg.sglang_url,cfg.model,messages=[{'role':'system','content':'Answer concisely.'},{'role':'user','content':'Describe the importance of keeping accurate long-term memory in an assistant in three sentences.'}])
  # Warm all paths; profiling is not active here.
  generate(cfg.sglang_url,ids,'contention-warmup',a.max_tokens);retrieve(q,qc,cfg.collection,cfg.user_id)
  conditions={name:{'name':name,'mode':'baseline' if name=='llm_only' else 'co_run','pair_id':'fixed_warm_prefix','repetitions':a.repetitions,'llm_ms':[],'auxiliary_ops':0,'auxiliary_ms':[],'wall_s':0,'max_tokens':a.max_tokens} for name in ['llm_only','embedding_llm','retrieval_llm','full_retrieval_llm']}
  def operation(name):
   if name=='embedding_llm':list(embed.embed([q]))
   elif name=='retrieval_llm':qc.query_points(cfg.collection,query=vec,query_filter=flt,limit=50,with_vectors=True,with_payload=True)
   elif name=='full_retrieval_llm':retrieve(q,qc,cfg.collection,cfg.user_id)
  # Alternating shuffled rounds reduce order/temperature effects. Prefix is warm in all cases.
  rng=random.Random(42)
  for rep in range(a.repetitions):
   order=list(conditions);rng.shuffle(order)
   for name in order:
    stop=threading.Event();ready=threading.Event();durations=[];errors=[]
    def worker():
     try:
      ready.set()
      while not stop.is_set():
       t=time.perf_counter();operation(name);durations.append((time.perf_counter()-t)*1000)
     except Exception as e:errors.append(str(e));ready.set()
    th=None;t0=time.perf_counter()
    if name!='llm_only':th=threading.Thread(target=worker);th.start();ready.wait(10)
    try:res=generate(cfg.sglang_url,ids,f'contention-{name}-{rep}',a.max_tokens)
    finally:
     stop.set()
     if th:th.join(timeout=30)
    if th and th.is_alive():raise RuntimeError('auxiliary worker did not stop')
    if errors:raise RuntimeError(errors)
    c=conditions[name];c['llm_ms'].append(res['total_ms']);c['auxiliary_ops']+=len(durations);c['auxiliary_ms']+=durations;c['wall_s']+=time.perf_counter()-t0
    print(name,rep+1,f"llm={res['total_ms']:.1f}ms aux_ops={len(durations)}",flush=True)
  for name,c in conditions.items():c['slowdown_vs_llm_only']=slowdown(conditions['llm_only']['llm_ms'],c['llm_ms'])
  result={'platform':a.platform,'conditions':list(conditions.values()),'limitations':['Small warm-prefix co-run characterization; no claim of statistically significant slowdown without repeated broader workloads.','Auxiliary request rate is saturation loop, not natural production arrival rate.','Slowdown alone does not prove memory bandwidth causality.','Judge excluded; underlying SGLang/runtime and Qdrant deployment differ across hosts.'],'warm_cache':True}
  (out/'contention.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('saved',out/'contention.json')
 finally:qc.close()
if __name__=='__main__':main()
