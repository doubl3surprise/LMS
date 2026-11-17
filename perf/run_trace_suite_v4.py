#!/usr/bin/env python3
"""Host entry point: one command collects native trace + resources and renders Markdown.
Example: python3 run_trace_suite_v4.py --platform dgx --questions 6 --session-replays 2 --judge --profile --contention
"""
import argparse,json,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--platform',required=True,choices=['dgx','5880ada']);p.add_argument('--questions',default='6');p.add_argument('--session-replays',type=int,default=2);p.add_argument('--max-tokens',type=int,default=64);p.add_argument('--judge',action='store_true');p.add_argument('--profile',action='store_true');p.add_argument('--contention',action='store_true');a=p.parse_args()
 root=Path(__file__).resolve().parent;rid=time.strftime('%Y%m%d_%H%M%S')+'_'+a.platform+'_v4';out=root/'trace_results'/rid;out.mkdir(parents=True)
 hostpython=str(root/'.venv/bin/python');bench=[hostpython] if a.platform=='dgx' else ['docker','exec','mem0-benchmark-python','python3'];prefix=str(root) if a.platform=='dgx' else '/workspace';runpath=str(out) if a.platform=='dgx' else '/workspace/trace_results/'+rid
 samplelog=(out/'sampler.log').open('w');sampler=subprocess.Popen([hostpython,str(root/'host_resources_v4.py'),'--out',str(out/'resource_samples.jsonl'),'--interval','0.5'],stdout=samplelog,stderr=subprocess.STDOUT)
 try:
  cmd=bench+[prefix+'/run_trace_v4.py','--platform',a.platform,'--questions',a.questions,'--session-replays',str(a.session_replays),'--max-tokens',str(a.max_tokens),'--run-id',rid,'--out-dir',runpath]
  if a.judge:cmd+=['--judge']
  subprocess.run(cmd,cwd=root,check=True)
 finally:
  sampler.terminate()
  try:sampler.wait(timeout=10)
  except subprocess.TimeoutExpired:sampler.kill();sampler.wait()
  samplelog.close()
 if a.profile:
  subprocess.run([sys.executable,str(root/'profile_trace_v4.py'),'--platform',a.platform,'--container','jiangchuan-qwen38-zh','--out-dir',str(out),'--probe-counters'],check=False)
  summaries=sorted((out/'gpu_profile').glob('*/summary.json'))
  if summaries:(out/'profiler_summary.json').write_text(summaries[-1].read_text())
 if a.contention:subprocess.run(bench+[prefix+'/run_contention_v4.py','--platform',a.platform,'--out-dir',runpath],cwd=root,check=True)
 cmd=bench+[prefix+'/analyze_trace_v4.py','--trace',runpath+'/trace.jsonl','--platform',a.platform,'--resources',runpath+'/resource_samples.jsonl','--metrics',runpath+'/metrics_samples.jsonl','--out',runpath+'/report_v4.md']
 for flag,file in [('--profile','profiler_summary.json'),('--contention','contention.json')]:
  if (out/file).exists():cmd += [flag,runpath+'/'+file]
 subprocess.run(cmd,cwd=root,check=True)
 subprocess.run([sys.executable,str(root/'render_trace_charts_v4.py'),'--run-dir',str(out)],check=True)
 append_code="from pathlib import Path; p=Path("+repr(runpath+'/report_v4.md')+"); p.write_text(p.read_text()+'\\n## 实测图表\\n\\n![KV and Mamba occupancy](kv_timeline.svg)\\n\\n![Latency](latency.svg)\\n')"
 subprocess.run(bench+['-c',append_code],cwd=root,check=True)
 print('REPORT='+str(out/'report_v4.md'))
if __name__=='__main__':main()
