import os,json,time,sys
from pathlib import Path
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root));os.environ['FASTEMBED_CACHE_PATH']=str(root/'models');os.environ['HF_HUB_OFFLINE']='1';os.environ['MEM0_TELEMETRY']='false'
from tracer.config import get_config
from qdrant_client import QdrantClient,models
from mem0 import Memory
h=sys.argv[1];out=Path(sys.argv[2]);cfg=get_config(h)
source=QdrantClient(url=cfg.qdrant_url) if cfg.qdrant_mode=='server' else QdrantClient(path=cfg.qdrant_path)
pts=[];offset=None
try:
 while True:
  batch,offset=source.scroll(cfg.collection,scroll_filter=models.Filter(must=[models.FieldCondition(key='user_id',match=models.MatchValue(value=cfg.user_id))]),limit=100,offset=offset,with_payload=True,with_vectors=True);pts+=batch
  if offset is None:break
finally:source.close()
qc=QdrantClient(':memory:');qc.create_collection('task3_native_probe',vectors_config=models.VectorParams(size=384,distance=models.Distance.COSINE))
for start in range(0,len(pts),100):qc.upsert('task3_native_probe',[models.PointStruct(id=p.id,vector=p.vector if isinstance(p.vector,list) else next(iter(p.vector.values())),payload=p.payload) for p in pts[start:start+100]])
config={'version':'v1.1','history_db_path':str(out/'native_probe_history.db'),'llm':{'provider':'openai','config':{'model':cfg.model,'openai_base_url':cfg.sglang_url+'/v1','api_key':'local-unused'}},'embedder':{'provider':'fastembed','config':{'model':'sentence-transformers/all-MiniLM-L6-v2'}},'vector_store':{'provider':'qdrant','config':{'collection_name':'task3_native_probe','embedding_model_dims':384,'path':str(out/'unused_probe_path'),'client':qc}}}
results=[]
try:
 m=Memory.from_config(config)
 data=json.loads(Path(cfg.dataset_path).read_text())[0]
 for cat in [1,2,3,4]:
  i,q=next((i,q) for i,q in enumerate(data['qa']) if q['category']==cat);t=time.perf_counter();res=m.search(q['question'],filters={'user_id':cfg.user_id},top_k=10,rerank=False)
  results.append({'source_q_idx':i,'category':cat,'question':q['question'],'search_ms':(time.perf_counter()-t)*1000,'result':res})
 report={'status':'success','memory_count':len(pts),'scope':'native Memory.search API on isolated in-memory copy of existing dense records; NOT production retrieval timing; initialization/warmup may affect first call; no add/update or LLM answer calls','collection_has_bm25_sparse_slot':False,'rerank':False,'results':results}
 (out/'native_search_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'status':'success','memories':len(pts),'counts':[len(x['result'].get('results',[])) for x in results]}))
finally:qc.close()
