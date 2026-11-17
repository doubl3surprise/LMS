#!/usr/bin/env python3
"""Single current host entry point. Run `python3 project.py --help`."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','validate','profile','bandwidth','report']);p.add_argument('--platform',required=True,choices=['dgx','5880ada']);p.add_argument('--run-dir');p.add_argument('--questions',default='6');p.add_argument('--sessions',action='store_true');p.add_argument('--judge',action='store_true');a=p.parse_args()
 if a.run_dir:d=Path(a.run_dir).resolve()
 elif a.action!='run' and (ROOT/'trace_results/LATEST').exists():d=ROOT.parent/'trace_results'/(ROOT/'trace_results/LATEST').read_text().strip()
 else:d=ROOT.parent/'trace_results'/(time.strftime('%Y%m%d_%H%M%S')+'_'+a.platform+'_current')
 py=str(ROOT/'.venv/bin/python');bench=[py] if a.platform=='dgx' else ['docker','exec','mem0-benchmark-python','python3'];prefix=str(ROOT) if a.platform=='dgx' else '/workspace';target=str(d) if a.platform=='dgx' else '/workspace/'+str(d.relative_to(ROOT))
 if a.action=='run':
  d.mkdir(parents=True,exist_ok=False);log=(d/'sampler.log').open('w');sampler=subprocess.Popen([py,str(ROOT/'host_resources_v4.py'),'--out',str(d/'resource_samples.jsonl'),'--interval','0.5'],stdout=log,stderr=subprocess.STDOUT)
  cmd=bench+[prefix+'/run_trace_v4.py','--platform',a.platform,'--questions',a.questions,'--session-replays','1','--max-tokens','40','--out-dir',target,'--run-id',d.name]
  if a.sessions:cmd+=['--sessions',prefix+'/multi_session_workload.json']
  if a.judge:cmd+=['--judge']
  try:subprocess.run(cmd,cwd=ROOT,check=True)
  finally:
   sampler.terminate()
   try:sampler.wait(timeout=15)
   except subprocess.TimeoutExpired:sampler.kill();sampler.wait()
   log.close()
  (ROOT/'trace_results/LATEST').write_text(d.name+'\n')
 elif a.action=='validate':subprocess.run(bench+[prefix+'/validate_workloads.py','--platform',a.platform,'--out-dir',target],cwd=ROOT,check=True)
 elif a.action=='profile':subprocess.run([sys.executable,str(ROOT/'request_profile.py'),'--platform',a.platform,'--out-dir',str(d),'--requests',str(d/'trace.jsonl'),'--limit','2'],cwd=ROOT,check=True)
 elif a.action=='bandwidth':subprocess.run([sys.executable,str(ROOT/'measure_bandwidth.py'),'--out-dir',str(d)],check=True)
 elif a.action=='report':
  subprocess.run([sys.executable,str(ROOT/'request_resources.py'),'--run-dir',str(d)],check=True)
  subprocess.run([sys.executable,str(ROOT/'render_trace_charts_v4.py'),'--run-dir',str(d)],check=True)
  subprocess.run([sys.executable,str(ROOT/'report_current.py'),'--run-dir',str(d),'--platform',a.platform],check=True)
 print('RUN_DIR='+str(d))
if __name__=='__main__':main()
