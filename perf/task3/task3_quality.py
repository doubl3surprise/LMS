#!/usr/bin/env python3
"""Native Mem0 single-factor task3 matrix on isolated original-record copies."""
import os,sys,json,time,hashlib,random,argparse,contextlib,statistics
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--platform',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--nonce',required=True);p.add_argument('--repeats',type=int,default=2);p.add_argument('--limit',type=int);a=p.parse_args()
 root=Path(__file__).resolve().parent;sys.path.insert(0,str(root));out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
 assert not (out/'trace.jsonl').exists(),'Refuse overwrite'
 os.environ['FASTEMBED_CACHE_PATH']=str(root/'models');os.environ['MEM0_TELEMETRY']='false';os.environ['HF_HUB_OFFLINE']='1'
 from tracer.config import get_config
 from tracer.retriever import retrieve,get_embedder
 from tracer.timer import PhasedTimer
 from tracer.observability_v4 import MetricsSampler
 from run_trace_v4 import tokenize
 from task3_pilot import history_text
 from task3_matrix_core import subsets,fit_budget
 from qdrant_client import QdrantClient,models
 from mem0 import Memory
 from tokenizers import Tokenizer
 import requests
 cfg=get_config(a.platform);dataset=Path(cfg.dataset_path);conv=json.loads(dataset.read_text())[0];history=history_text(conv['conversation'])
 tokenizer=Tokenizer.from_file(str(root/'model-serving/experiment/tokenizer/tokenizer.json'))
 encode=lambda t:tokenizer.encode(t,add_special_tokens=False).ids
 decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=False)
 source=QdrantClient(url=cfg.qdrant_url) if cfg.qdrant_mode=='server' else QdrantClient(path=cfg.qdrant_path)
 points=[];offset=None
 try:
  while True:
   pts,offset=source.scroll(cfg.collection,scroll_filter=models.Filter(must=[models.FieldCondition(key='user_id',match=models.MatchValue(value=cfg.user_id))]),limit=100,offset=offset,with_vectors=True,with_payload=True)
   points.extend({'id':str(x.id),'vector':x.vector if isinstance(x.vector,list) else next(iter(x.vector.values())),'payload':x.payload} for x in pts)
   if offset is None:break
 finally:source.close()
 assert len(points)==879
 groups=subsets(points,[879]);memory=[];clients={};instances={}
 # Use original per-platform deployment form: server collections vs embedded files.
 for n,records in groups.items():
  name='t3_'+a.nonce.replace('-','_')+'_'+str(n)
  qc=QdrantClient(url=cfg.qdrant_url) if cfg.qdrant_mode=='server' else QdrantClient(path=str(out/('qdrant_'+str(n))))
  assert not qc.collection_exists(name),'Refuse existing experimental collection'
  qc.create_collection(name,vectors_config=models.VectorParams(size=384,distance=models.Distance.COSINE))
  for i in range(0,n,100):qc.upsert(name,[models.PointStruct(**x) for x in records[i:i+100]])
  conf={'history_db_path':str(out/('history_'+str(n)+'.db')),'llm':{'provider':'openai','config':{'model':cfg.model,'openai_base_url':cfg.sglang_url+'/v1','api_key':'unused-local'}},'embedder':{'provider':'fastembed','config':{'model':'sentence-transformers/all-MiniLM-L6-v2'}},'vector_store':{'provider':'qdrant','config':{'client':qc,'collection_name':name,'embedding_model_dims':384,'path':str(out/'unused')}}}
  instances[n]=Memory.from_config(conf);clients[n]=(qc,name)
  memory.append({'n':n,'collection':name,'records_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),'ordered_ids':[x['id'] for x in records]})
 questions=[]
 for cat in [1,2,3,4]:
  questions.extend({'source_q_idx':i,**q} for i,q in [(i,q) for i,q in enumerate(conv['qa']) if q['category']==cat][:8])
 # Shared baseline: native,N879,K10,no cap. Other axes vary separately.
 conditions=[{'id':'no_memory','mode':'no_memory'},{'id':'full_history','mode':'full_history'}, {'id':'native_n879_k10','mode':'native','n':879,'k':10}]
 jobs=[(rep,q,cond) for rep in range(a.repeats) for q in questions for cond in conditions];random.Random(42).shuffle(jobs)
 if a.limit:jobs=jobs[:a.limit]
 info=requests.get(cfg.sglang_url+'/get_server_info',timeout=20).json()
 manifest={'schema_version':'task3-quality-1.0','platform':a.platform,'nonce':a.nonce,'expected_requests':len(jobs),'repeats':a.repeats,'conditions':conditions,'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),'questions':questions,'collections':memory,'server_info':info,'scope':'native SDK search on isolated original-record nested subsets in original per-platform Qdrant deployment form; MMR separate; no ingestion. SDK BM25 disabled because original records lack sparse slot.','output_policy':{'temperature':0,'top_p':1,'top_k':-1,'max_new_tokens':128,'min_new_tokens':0,'ignore_eos':False},'cache_policy':'unique leading nonce marker per request, matched on both hosts; cached_tokens recorded','quality':'natural stop max128, normalized EM and token F1 diagnostic scoring, not official LoCoMo evaluator','budget_policy':'rank-ordered concatenated memory labels/text; truncate block token prefix then re-encode, no padding; whole prompt exact token count saved'}
 (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 # Wrap native boundary methods without altering return values or scoring.
 spans={}
 def instrument(obj,name,key):
  original=getattr(obj,name)
  def wrapped(*args,**kwargs):
   t=time.perf_counter()
   try:return original(*args,**kwargs)
   finally:spans[key]=spans.get(key,0)+(time.perf_counter()-t)*1000
  setattr(obj,name,wrapped)
 for n,m in instances.items():
  instrument(m.embedding_model,'embed','embed_ms');instrument(m.vector_store,'search','semantic_search_ms');instrument(m.vector_store,'keyword_search','keyword_search_ms')
  instrument(m,'_compute_entity_boosts','entity_boost_ms')
  m.search(questions[0]['question'],top_k=10,filters={'user_id':cfg.user_id},rerank=False)
 get_embedder();qc,name=clients[879];retrieve(questions[0]['question'],qc,name,cfg.user_id,50,10,.5)
 sampler=MetricsSampler(cfg.sglang_url,out/'metrics_samples.jsonl',interval=.2);sampler.start()
 try:
  with (out/'trace.jsonl').open('w') as f:
   for idx,(rep,q,cond) in enumerate(jobs):
    rid=a.nonce+':'+str(idx);spans.clear();timer=PhasedTimer();t0=time.perf_counter();wall=time.time();mem=[];search_ms=0
    if cond['mode']=='native':
     ts=time.perf_counter();res=instances[cond['n']].search(q['question'],top_k=cond['k'],filters={'user_id':cfg.user_id},rerank=False);search_ms=(time.perf_counter()-ts)*1000
     mem=[{'id':r['id'],'text':r['memory'],'score':r['score'],'rank':i+1} for i,r in enumerate(res['results'])]
    elif cond['mode']=='mmr':
     ts=time.perf_counter();qc,name=clients[cond['n']];rs,_=retrieve(q['question'],qc,name,cfg.user_id,50,cond['k'],.5,timer);search_ms=(time.perf_counter()-ts)*1000
     mem=[{'id':r.memory_id,'text':r.text,'score':r.score,'rank':r.rank} for r in rs]
    ta=time.perf_counter()
    block=history if cond['mode']=='full_history' else '\n'.join('[Memory '+str(i+1)+'] '+r['text'] for i,r in enumerate(mem))
    original_ids=encode(block);block,block_ids=fit_budget(block,cond.get('budget'),encode,decode)
    messages=[{'role':'system','content':a.nonce+' request '+str(idx)+'. Answer the question using supplied context when available. Be concise.'},{'role':'user','content':'Context:\n'+block+'\n\nQuestion: '+q['question']+'\nAnswer:'}]
    assembly_ms=(time.perf_counter()-ta)*1000;ts=time.perf_counter();ids=tokenize(cfg.sglang_url,cfg.model,messages=messages);tokenize_ms=(time.perf_counter()-ts)*1000
    assert len(ids)+128<=info['context_length']
    payload={'rid':rid,'input_ids':ids,'sampling_params':manifest['output_policy'],'stream':True};first=None;meta={};answer='';gt=time.perf_counter()
    with requests.post(cfg.sglang_url+'/generate',json=payload,stream=True,timeout=300) as r:
     r.raise_for_status()
     for raw in r.iter_lines(chunk_size=1,decode_unicode=True):
      if not raw or not raw.startswith('data: '):continue
      if raw[6:].strip()=='[DONE]':break
      chunk=json.loads(raw[6:]);now=time.perf_counter()
      if chunk.get('text') and first is None:first=now
      answer=chunk.get('text','');meta=chunk.get('meta_info') or meta
    end=time.perf_counter();wall_end=time.time()
    assert meta['completion_tokens']<=128
    timings={'retrieval_total_ms':search_ms,'assembly_ms':assembly_ms,'prompt_tokenize_ms':tokenize_ms,'server_prefill_elapsed_ms':(meta['prefill_finished_time']-meta['forward_entry_time'])*1000,'server_queue_ms':meta.get('queue_time')*1000 if meta.get('queue_time') is not None else None,'ttft_ms':(first-gt)*1000 if first else None,'decode_ms':(end-first)*1000 if first else None,'generate_total_ms':(end-gt)*1000,'response_ms':(end-t0)*1000,**spans}
    if cond['mode']=='native':timings['sdk_other_ms']=search_ms-sum(spans.values())
    if cond['mode']=='mmr':timings.update({k:timer.ms(k[:-3]) for k in ['embed_ms','search_ms','mmr_ms']})
    row={'schema_version':'task3-quality-1.0','request_id':rid,'platform':a.platform,'repetition':rep,'condition':cond,'source_q_idx':q['source_q_idx'],'category':q['category'],'question':q['question'],'gold':q['answer'],'answer':answer,'messages':messages,'memories':mem,'selected_memory_count':len(mem),'tokens':{'prompt':len(ids),'context_actual':len(block_ids),'context_before_budget':len(original_ids),'output':meta['completion_tokens'],'cached':meta.get('cached_tokens'),'server_decode_tps':meta.get('decode_throughput')},'input_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'timing':timings,'wall_start':wall,'answer_end':wall_end,'native_meta':meta}
    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();print(json.dumps({'request':idx+1,'total':len(jobs),'condition':cond['id'],'response_ms':timings['response_ms'],'input':len(ids),'context':len(block_ids),'cached':meta.get('cached_tokens')}),flush=True)
  (out/'DONE').write_text(str(len(jobs))+' completed\n')
 finally:
  sampler.stop()
  for qc,name in clients.values():qc.close()
if __name__=='__main__':main()
