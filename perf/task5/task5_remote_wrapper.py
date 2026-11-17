"""Task5 remote job wrapper; all writes scoped to new run directory."""
from pathlib import Path
import sys,json,subprocess,time
root=Path(sys.argv[1]);out=Path(sys.argv[2]);host=sys.argv[3];nonce=sys.argv[4];workload=sys.argv[5]
if (out/'trace.jsonl').exists():raise RuntimeError('Existing trace')
out.mkdir(parents=True,exist_ok=True)
log=(out/'sampler.log').open('w');sampler=subprocess.Popen([str(root/'.venv/bin/python'),str(root/'task5/task5_host.py'),'--out',str(out/'resource_samples.jsonl'),'--interval','.5'],stdout=log,stderr=subprocess.STDOUT)
cmd=([str(root/'.venv/bin/python'),str(root/'task5_run.py')] if host=='dgx' else ['docker','exec','-w','/workspace','mem0-benchmark-python','python3','/workspace/task5_run.py'])+['--platform',host,'--out-dir',str(out) if host=='dgx' else '/workspace/trace_results/'+out.name,'--nonce',nonce,'--workload',str(root/'task5'/workload) if host=='dgx' else '/workspace/task5/'+workload]
exitcode=None
try:
 with (out/'execution.log').open('w') as f:exitcode=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
finally:
 sampler.terminate()
 try:sampler.wait(timeout=10)
 except subprocess.TimeoutExpired:sampler.kill();sampler.wait()
 log.close();(out/'EXIT.json').write_text(json.dumps({'exit':exitcode,'done':(out/'DONE').exists(),'finished_wall':time.time()}))
