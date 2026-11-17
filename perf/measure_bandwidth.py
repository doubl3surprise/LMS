#!/usr/bin/env python3
"""Capture device-wide GPU metric counters without changing host driver policy.
Uses one docker exec --privileged command (existing Docker authorization). Percentage
is measured throughput utilization, NOT bytes/s. GB10 may expose no DRAM metric.
"""
import argparse,json,subprocess,time,urllib.request
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',required=True);p.add_argument('--container',default='jiangchuan-qwen38-zh');p.add_argument('--duration',type=int,default=12);p.add_argument('--base-url',default='http://127.0.0.1:8000');a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
 remote='/root/.cache/task2_gpu_metrics/'+str(time.time_ns());c=a.container
 def run(args,timeout=60):return subprocess.run(args,capture_output=True,text=True,timeout=timeout,check=True)
 run(['docker','exec',c,'mkdir','-p',remote])
 cmd=['docker','exec','--privileged','-e','TMPDIR='+remote,c,'nsys','profile','--trace=none','--sample=none','--cpuctxsw=none','--gpu-metrics-devices=all','--gpu-metrics-frequency=1000','--duration='+str(a.duration),'--output='+remote+'/metrics','sleep',str(a.duration)]
 process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);time.sleep(2)
 payload={'model':'qwen3.8-27b-zh','messages':[{'role':'user','content':'List twenty important facts about computer systems.'}],'temperature':0,'max_tokens':24,'chat_template_kwargs':{'enable_thinking':False}}
 req=urllib.request.Request(a.base_url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=60) as r:response=json.load(r)
 log=process.communicate(timeout=90)[0];(out/'gpu_metrics_collect.log').write_text(log)
 if process.returncode:raise RuntimeError(log[-2000:])
 run(['docker','exec',c,'nsys','export','--type=sqlite','--output='+remote+'/metrics.sqlite',remote+'/metrics.nsys-rep'])
 probe="""import sqlite3,json,sys
c=sqlite3.connect(sys.argv[1]);names=list(c.execute('select typeId,metricId,metricName from TARGET_INFO_GPU_METRICS'));series=[]
for typ,mid,name in names:
 if not any(x in name for x in ['DRAM','SMs Active','GR Active','Tensor Active']):continue
 rows=list(c.execute('select timestamp,value from GPU_METRICS where typeId=? and metricId=? order by timestamp',(typ,mid)));v=[float(y) for x,y in rows]
 series.append({'type_id':typ,'metric_id':mid,'name':name,'unit':'percent_of_peak' if 'Throughput %' in name else 'as_exported','count':len(v),'min':min(v) if v else None,'max':max(v) if v else None,'avg':sum(v)/len(v) if v else None,'samples':rows})
print(json.dumps({'all_metric_names':[n for _,_,n in names],'series':series}))
"""
 data=json.loads(run(['docker','exec',c,'python3','-c',probe,remote+'/metrics.sqlite']).stdout)
 data.update({'status':'measured_percent' if any('DRAM' in x['name'] for x in data['series']) else 'unsupported_dram_metrics','scope':'device-wide sampler during independent representative LLM request, includes idle portions; not request-isolated','limitations':['Percent of peak sustained throughput is not GB/s.','GB10 GPU metrics do not establish CPU+GPU total shared DRAM traffic.'],'usage':response.get('usage'),'remote_dir':remote})
 (out/'bandwidth.json').write_text(json.dumps(data,indent=2));run(['docker','cp',c+':'+remote+'/metrics.nsys-rep',str(out/'gpu_metrics.nsys-rep')]);run(['docker','cp',c+':'+remote+'/metrics.sqlite',str(out/'gpu_metrics.sqlite')])
 print(json.dumps({'status':data['status'],'metrics':[(x['name'],x['avg'],x['max']) for x in data['series']]}))
if __name__=='__main__':main()
