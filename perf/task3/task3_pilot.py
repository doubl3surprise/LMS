#!/usr/bin/env python3
"""Task3 bounded pilot. Four original QA x three modes; controlled 16-step decoding.
Reuse installed collectors. No memory writes, no service restarts/cache flushing.
"""
import argparse,json,os,sys,time,hashlib,random,statistics
from pathlib import Path

def history_text(conv):
 out=[]
 for k in sorted([k for k,v in conv.items() if k.startswith('session_') and isinstance(v,list)],key=lambda k:int(k.split('_')[1])):
  out.append('Session date: '+str(conv.get(k+'_date_time','')))
  out.extend(t['speaker']+': '+t.get('text','') for t in conv[k])
 return '\n'.join(out)

def main():
 p=argparse.ArgumentParser();p.add_argument('--platform',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--nonce',required=True);a=p.parse_args()
 root=Path(__file__).resolve().parent;sys.path.insert(0,str(root));os.environ['FASTEMBED_CACHE_PATH']=str(root/'models');os.environ['HF_HUB_OFFLINE']='1';os.environ['MEM0_TELEMETRY']='false'
 from tracer.config import get_config
 from tracer.retriever import get_embedder,retrieve
 from tracer.timer import PhasedTimer
 from tracer.observability_v4 import MetricsSampler
 from qdrant_client import QdrantClient
 from run_trace_v4 import tokenize,post
 import requests
 cfg=get_config(a.platform);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
 assert not (out/'trace.jsonl').exists(),'refusing overwrite'
 source=Path(cfg.dataset_path);c=json.loads(source.read_text())[0];history=history_text(c['conversation']);questions=[]
 for cat in [1,2,3,4]:
  i,q=next((i,q) for i,q in enumerate(c['qa']) if q['category']==cat);questions.append({'source_q_idx':i,**q})
 jobs=[(q,m) for q in questions for m in ['no_memory','full_history','retrieval_mmr']];random.Random(42).shuffle(jobs)
 info=requests.get(cfg.sglang_url+'/get_server_info',timeout=20).json()
 manifest={'platform':a.platform,'run_id':out.name,'nonce':a.nonce,'sample_id':c.get('sample_id'),'dataset_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'question_indices':[q['source_q_idx'] for q in questions],'request_count_expected':len(jobs),'modes':['no_memory','full_history','retrieval_mmr'],'history_scope':'all 19 sessions chronological speaker-labeled text only, excludes images/captions, summaries, QA and gold','history_chars':len(history),'output_policy':{'max_new_tokens':16,'min_new_tokens':16,'ignore_eos':True,'temperature':0,'top_p':1,'top_k':-1},'quality_scoring':False,'cache_policy':'distinct leading system request marker per job, same markers/order on both hosts; observed cached_tokens required, no cache flush','server_info':info,'measurement_scope':'task3 application query path; custom MMR uses direct components, no Mem0 SDK; no ingestion or Judge; prefill elapsed not pure GPU time','order':[{'q':q['source_q_idx'],'mode':m} for q,m in jobs]}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));qc=QdrantClient(url=cfg.qdrant_url) if cfg.qdrant_mode=='server' else QdrantClient(path=cfg.qdrant_path)
 sampler=None
 try:
  get_embedder();retrieve(questions[0]['question'],qc,cfg.collection,cfg.user_id,50,10,.5)
  sampler=MetricsSampler(cfg.sglang_url,out/'metrics_samples.jsonl',interval=.2);sampler.start()
  with (out/'trace.jsonl').open('w') as f:
   for index,(q,mode) in enumerate(jobs):
    rid=a.nonce+':'+str(index);timer=PhasedTimer();start=time.perf_counter();wall=time.time();memories=[]
    if mode=='retrieval_mmr':
     mm,_=retrieve(q['question'],qc,cfg.collection,cfg.user_id,50,10,.5,timer)
     memories=[{'memory_id':m.memory_id,'rank':m.rank,'score':m.score,'text':m.text} for m in mm]
    with timer.phase('assembly'):
     if mode=='full_history':block=history
     elif mode=='retrieval_mmr':block='\n'.join('[Memory '+str(i+1)+'] '+m['text'] for i,m in enumerate(memories))
     else:block=''
     messages=[{'role':'system','content':a.nonce+' request '+str(index)+'. Answer the question using the supplied context when available. Be concise.'},{'role':'user','content':'Context:\n'+block+'\n\nQuestion: '+q['question']+'\nAnswer:'}]
    t=time.perf_counter();ids=tokenize(cfg.sglang_url,cfg.model,messages=messages);tokenize_ms=(time.perf_counter()-t)*1000
    assert len(ids)+16<=info['context_length'],'context overflow'
    payload={'input_ids':ids,'rid':rid,'sampling_params':manifest['output_policy'],'stream':True}
    genstart=time.perf_counter();first=None;text='';meta={}
    with requests.post(cfg.sglang_url+'/generate',json=payload,stream=True,timeout=300) as resp:
     resp.raise_for_status()
     for raw in resp.iter_lines(chunk_size=1,decode_unicode=True):
      if not raw or not raw.startswith('data: '):continue
      if raw[6:].strip()=='[DONE]':break
      chunk=json.loads(raw[6:]);now=time.perf_counter()
      if chunk.get('text') and first is None:first=now
      text=chunk.get('text','');meta=chunk.get('meta_info') or meta
    end=time.perf_counter();wallend=time.time();ttft=(first-genstart)*1000 if first else None
    record={'schema_version':'task3-pilot-1.0','request_id':rid,'platform':a.platform,'mode':mode,'source_q_idx':q['source_q_idx'],'category':q['category'],'question':q['question'],'answer':text,'messages':messages,'input_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'wall_start':wall,'answer_end':wallend,'wall_end':wallend,'timing':{'embed_ms':timer.ms('embed'),'search_ms':timer.ms('search'),'mmr_ms':timer.ms('mmr'),'assembly_ms':timer.ms('assembly'),'prompt_tokenize_ms':tokenize_ms,'server_prefill_elapsed_ms':(meta['prefill_finished_time']-meta['forward_entry_time'])*1000,'server_queue_ms':meta.get('queue_time',0)*1000,'ttft_ms':ttft,'decode_ms':(end-first)*1000 if first else None,'generate_total_ms':(end-genstart)*1000,'response_ms':(end-start)*1000},'tokens':{'prompt_tokens':meta.get('prompt_tokens'),'completion_tokens':meta.get('completion_tokens'),'cached_tokens':meta.get('cached_tokens'),'server_decode_tps':meta.get('decode_throughput')},'memories':memories,'native_meta':meta}
    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();print(json.dumps({'index':index+1,'mode':mode,'input_tokens':len(ids),'output_tokens':record['tokens']['completion_tokens'],'cached_tokens':record['tokens']['cached_tokens'],'response_ms':record['timing']['response_ms'],'prefill_ms':record['timing']['server_prefill_elapsed_ms']}),flush=True)
  (out/'DONE').write_text('12 requests completed\n')
 finally:
  if sampler:sampler.stop()
  qc.close()
if __name__=='__main__':main()
