#!/usr/bin/env python3
"""Host-sampler on/off paired calibration, independent of inference client startup."""
import argparse,json,subprocess,time,urllib.request,random
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);p.add_argument('--repetitions',type=int,default=3);a=p.parse_args();d=Path(a.out_dir);root=Path(__file__).resolve().parent
 def post(path,payload):
  req=urllib.request.Request('http://127.0.0.1:8000'+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=90) as r:return json.load(r)
 tok=post('/v1/tokenize',{'model':'qwen3.8-27b-zh','messages':[{'role':'user','content':'List twenty useful facts about computer memory.'}],'add_special_tokens':False,'chat_template_kwargs':{'enable_thinking':False}})['tokens']
 payload={'input_ids':tok,'sampling_params':{'temperature':0,'top_p':1,'max_new_tokens':16},'stream':False};post('/generate',payload)
 samples=[];rng=random.Random(42)
 for repeat in range(a.repetitions):
  modes=['off','on'];rng.shuffle(modes)
  for mode in modes:
   proc=None
   if mode=='on':
    proc=subprocess.Popen([str(root/'.venv/bin/python'),str(root/'host_resources_v4.py'),'--out',str(d/'calibration_host_samples.jsonl'),'--interval','.5'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);time.sleep(.5)
   try:
    start=time.perf_counter();response=post('/generate',payload);elapsed=(time.perf_counter()-start)*1000
    samples.append({'repetition':repeat,'host_sampler':mode,'elapsed_ms':elapsed,'output_tokens':response['meta_info']['completion_tokens'],'queue_s':response['meta_info'].get('queue_time')})
   finally:
    if proc:
     proc.terminate()
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
 off=[x['elapsed_ms'] for x in samples if x['host_sampler']=='off'];on=[x['elapsed_ms'] for x in samples if x['host_sampler']=='on'];ratio=(sum(on)/len(on))/(sum(off)/len(off))
 result={'samples':samples,'on_off_ratio':ratio,'repetitions':a.repetitions,'scope':'host CPU/IO/GPU sampler only; no metrics sampler, fixed warm native request, includes no Python client startup','limitations':['Small sample does not establish statistical significance.','Profiler off in both; on/off sampler startup excluded; actual sampling interval and command overhead retained.']}
 (d/'calibration_host.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
