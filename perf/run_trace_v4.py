#!/usr/bin/env python3
"""Native SGLang tracing. Does not modify memories or use reference answers for generation.
Example: python run_trace_v4.py --platform dgx --questions 6 --session-replays 2
A replay creates explicitly synthetic independent request sessions, not LoCoMo source sessions.
"""
import argparse,dataclasses,hashlib,json,os,resource,subprocess,sys,time,uuid
from pathlib import Path
import requests
from trace_core_v4 import build_prompt,native_timing,with_session_history

BASE=Path(__file__).resolve().parent
os.environ.setdefault('FASTEMBED_CACHE_PATH',str(BASE/'models'))
os.environ.setdefault('MEM0_TELEMETRY','false')
os.environ.setdefault('HF_HUB_OFFLINE','1')

def post(base,path,body):
 r=requests.post(base+path,json=body,timeout=120);r.raise_for_status();return r.json()

def tokenize(base,model,**content):
 d=post(base,'/v1/tokenize',{'model':model,'add_special_tokens':False,'chat_template_kwargs':{'enable_thinking':False},**content})
 if not isinstance(d.get('tokens'),list):raise ValueError('Missing tokenizer token IDs')
 return d['tokens']

def generate(base,ids,rid,max_tokens):
 """SSE contains cumulative text: capture first nonempty text, use last server meta."""
 t0=time.perf_counter();first=None;text='';meta={};arrival=[]
 with requests.post(base+'/generate',json={'input_ids':ids,'rid':rid,'sampling_params':{'temperature':0,'top_p':1,'top_k':-1,'max_new_tokens':max_tokens},'stream':True},stream=True,timeout=120) as resp:
  resp.raise_for_status()
  for raw in resp.iter_lines(chunk_size=1,decode_unicode=True):
   if not raw or not raw.startswith('data: '):continue
   if raw[6:].strip()=='[DONE]':break
   chunk=json.loads(raw[6:]);now=time.perf_counter();new=chunk.get('text','')
   if new and first is None:first=now
   if new!=text:arrival.append({'elapsed_ms':(now-t0)*1000,'text_chars':len(new)})
   text=new;meta=chunk.get('meta_info') or meta
 total=(time.perf_counter()-t0)*1000;ttft=(first-t0)*1000 if first else None
 return {'text':text,'meta':meta,'ttft_ms':ttft,'total_ms':total,'decode_ms':total-ttft if ttft is not None else None,'arrivals':arrival}

def self_cpu():
 r=resource.getrusage(resource.RUSAGE_SELF)
 return {'user_s':r.ru_utime,'system_s':r.ru_stime}

def run(args):
 from tracer.config import get_config
 from tracer.retriever import retrieve,get_embedder
 from tracer.timer import PhasedTimer
 from qdrant_client import QdrantClient
 from tracer.observability_v4 import MetricsSampler
 cfg=get_config(args.platform);os.environ['MEM0_TELEMETRY']='false'
 dataset=Path(cfg.dataset_path);conv=json.loads(dataset.read_text())[args.conv]
 questions=[(i,q) for i,q in enumerate(conv['qa']) if q['category'] in (1,2,3,4)]
 if args.questions!='all':questions=questions[:int(args.questions)]
 session_workload=json.loads(Path(args.sessions).read_text()) if args.sessions else None
 if session_workload:
  if args.session_replays!=1:raise ValueError('sessions workload requires session-replays=1')
  sessions=session_workload['sessions']
  questions=[(i,q) for session in sessions for i,q in enumerate(session['turns'])]
 else:sessions=[{'session_id':f'replay-{i+1}','turns':[q for _,q in questions]} for i in range(args.session_replays)]
 if not questions:raise ValueError('Empty question set')
 run_id=args.run_id or time.strftime('%Y%m%d_%H%M%S')+'_'+args.platform+'_v4'
 out=Path(args.out_dir) if args.out_dir else Path(cfg.trace_results_dir)/run_id
 out.mkdir(parents=True,exist_ok=False) if not out.exists() else None
 if (out/'trace.jsonl').exists():raise FileExistsError('Refuse to overwrite existing trace')
 (out/'RUNNING').write_text(str(os.getpid()))
 qc=QdrantClient(url=cfg.qdrant_url) if cfg.qdrant_mode=='server' else QdrantClient(path=cfg.qdrant_path)
 sampler=None
 try:
  # Exclude embedding model load and one warmup query from timed requests.
  get_embedder();retrieve(questions[0][1]['question'],qc,cfg.collection,cfg.user_id,cfg.cand_k,cfg.final_k,cfg.mmr_lambda)
  info=requests.get(cfg.sglang_url+'/get_server_info',timeout=15).json()
  meta={'schema_version':'4.0','run_id':run_id,'platform':args.platform,'n_questions':len(questions),'request_count_expected':len(questions)*args.session_replays,'session_source':'synthetic_replay' if args.session_replays>1 else 'single_benchmark_session','dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),'config':dataclasses.asdict(cfg),'server_info':info,'arguments':vars(args),'measurement_scope':'FastEmbed + Qdrant + MMR + native SGLang; bypasses Mem0 SDK; Judge after response','clock':{'wall_time':time.time(),'monotonic_ns':time.monotonic_ns()}}
  meta['session_source']='curated_multi_turn_sessions' if session_workload else meta['session_source']
  meta['workload']=session_workload
  meta['request_count_expected']=sum(len(s['turns']) for s in sessions)
  (out/'run_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
  sampler=MetricsSampler(cfg.sglang_url,out/'metrics_samples.jsonl',interval=args.sample_interval);sampler.start()
  with (out/'trace.jsonl').open('w') as f:
   for replay,session in enumerate(sessions):
    session_id=f"{run_id}:{session['session_id']}";history=[]
    for idx,q in enumerate(session['turns']):
     source_idx=q.get('source_q_idx',idx)
     timer=PhasedTimer();cpu0=self_cpu();wall=time.time();t0=time.perf_counter();rid=f'{run_id}:{replay}:{idx}'
     memories,_=retrieve(q['question'],qc,cfg.collection,cfg.user_id,cfg.cand_k,cfg.final_k,cfg.mmr_lambda,timer)
     with timer.phase('assembly'):
      prompt=build_prompt(q['question'],[{'rank':m.rank,'memory_id':m.memory_id,'score':m.score,'text':m.text,'text_len':m.text_len,'metadata':m.metadata} for m in memories])
      if session_workload:prompt=with_session_history(prompt,history)
     ta=time.perf_counter();ids=tokenize(cfg.sglang_url,cfg.model,messages=prompt['messages']);render_tokenize_ms=(time.perf_counter()-ta)*1000
     result=generate(cfg.sglang_url,ids,rid,args.max_tokens);response_ms=(time.perf_counter()-t0)*1000;answer_end=time.time();cpu1=self_cpu()
     # Token accounting is outside response timing. Count exactly the actual injected block,
     # including Memory labels/separators, without BOS/EOS; boundary count is declared.
     mem_ids=tokenize(cfg.sglang_url,cfg.model,prompt=prompt['memory_block'])
     timing={k:timer.ms(k[:-3]) for k in ['embed_ms','search_ms','mmr_ms','assembly_ms']}
     timing.update(native_timing(result['meta'],result['ttft_ms']))
     timing.update({'ttft_ms':result['ttft_ms'],'decode_ms':result['decode_ms'],'generate_total_ms':result['total_ms'],'response_ms':response_ms,'prompt_tokenize_ms':render_tokenize_ms,'judge_total_ms':None,'end_to_end_ms':response_ms})
     correct=None;judge_raw=None
     if args.judge:
      jt=time.perf_counter();j=post(cfg.sglang_url,'/v1/chat/completions',{'model':cfg.model,'messages':[{'role':'user','content':f"Question: {q['question']}\nGold answer: {q['answer']}\nPredicted answer: {result['text']}\nDoes the prediction convey the same key information? Answer only yes or no."}],'temperature':0,'top_p':1,'max_tokens':8,'seed':42,'chat_template_kwargs':{'enable_thinking':False}})
      judge_raw=j['choices'][0]['message']['content'];correct=judge_raw.strip().lower().startswith('yes');timing['judge_total_ms']=(time.perf_counter()-jt)*1000;timing['end_to_end_ms']=(time.perf_counter()-t0)*1000
     result['meta']['engine']=info.get('version') or ('SGLang '+args.platform+' pinned-image; see run_meta.json')
     n=result['meta'].get('completion_tokens');dec=result['decode_ms']
     record={'schema_version':'4.0','run_id':run_id,'platform':args.platform,'request_id':rid,'q_idx':idx,'source_q_idx':source_idx,'request_session_id':session_id,'session_source':meta['session_source'],'user_id':cfg.user_id,'category':q['category'],'question':q['question'],'gold':q['answer'],'answer':result['text'],'correct':correct,'judge_raw':judge_raw,'wall_start':wall,'answer_end':answer_end,'wall_end':time.time(),'timing':timing,'timing_source':'SGLang native meta_info; prefill elapsed is scheduler wall time, not GPU kernel sum','tokens':{'prompt_tokens':result['meta'].get('prompt_tokens'),'completion_tokens':n,'cached_tokens':result['meta'].get('cached_tokens'),'memory_tokens_actual':len(mem_ids),'memory_token_method':'tokenize_memory_block_no_special_tokens','memory_token_scope':'includes Memory labels and newlines; separately tokenized block, not additive prompt span','decode_tps':(n-1)/(dec/1000) if n and n>1 and dec and dec>0 else None},'memories':prompt['memories'],'messages':prompt['messages'],'prompt_token_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'native_meta':result['meta'],'client_token_arrivals':result['arrivals'],'local_cpu':{'user_ms':(cpu1['user_s']-cpu0['user_s'])*1000,'system_ms':(cpu1['system_s']-cpu0['system_s'])*1000,'scope':'benchmark process, excludes remote engine/Qdrant server'}}
     if session_workload:history += [{'role':'user','content':q['question']},{'role':'assistant','content':result['text']}]
     f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
     print(f"[{replay+1}/{len(sessions)} {idx+1}/{len(session['turns'])}] prefill={timing['server_prefill_elapsed_ms']} queue={timing['server_queue_ms']} response={response_ms:.1f}ms mem_tokens={len(mem_ids)}",flush=True)
  (out/'DONE').write_text('completed\n');print('TRACE_DIR='+str(out),flush=True)
 finally:
  if sampler:sampler.stop()
  qc.close();(out/'RUNNING').unlink(missing_ok=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--platform',choices=['dgx','5880ada'],required=True);p.add_argument('--questions',default='6');p.add_argument('--conv',type=int,default=0);p.add_argument('--session-replays',type=int,default=2);p.add_argument('--max-tokens',type=int,default=100);p.add_argument('--sample-interval',type=float,default=.1);p.add_argument('--run-id');p.add_argument('--out-dir');p.add_argument('--sessions',help='JSON multi-turn access-session workload');p.add_argument('--judge',action='store_true');a=p.parse_args()
 if a.session_replays<1:p.error('session-replays must be positive')
 run(a)
if __name__=='__main__':main()
