"""Task6 scoped remote launcher; runner owns exclusive creation of its out dir."""
from pathlib import Path
import subprocess,sys,json,time,threading
root=Path(sys.argv[1]);out=Path(sys.argv[2]);host=sys.argv[3];nonce=sys.argv[4]
assert not out.exists(), 'refuse existing output dir'
log=(Path(str(out)+'.sampler.log')).open('w');sampler_state={'process':None,'error':None}
def start_sampler_when_owned():
 try:
  deadline=time.monotonic()+60
  while not out.exists():
   if time.monotonic()>deadline:raise TimeoutError('runner did not create output directory')
   time.sleep(.05)
  sampler_state['process']=subprocess.Popen([str(root/'.venv/bin/python'),str(root/'task6/task6_host.py'),'--out',str(out/'resource_samples.jsonl'),'--interval','.2'],stdout=log,stderr=subprocess.STDOUT)
 except BaseException as exc:sampler_state['error']=exc
thread=threading.Thread(target=start_sampler_when_owned,daemon=True);thread.start()
cmd=([str(root/'.venv/bin/python'),str(root/'task6_probe.py')] if host=='dgx' else ['docker','exec','-w','/workspace','mem0-benchmark-python','python3','/workspace/task6_probe.py'])+['--platform',host,'--out-dir',str(out) if host=='dgx' else '/workspace/trace_results/'+out.name,'--nonce',nonce,'--workload',str(root/'task6/workload.json') if host=='dgx' else '/workspace/task6/workload.json','--repeats','3']
status=None;execution=Path(str(out)+'.execution.log')
try:
 with execution.open('w') as f:status=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
finally:
 thread.join(timeout=65)
 sampler=sampler_state['process']
 if sampler is not None:
  sampler.terminate()
  try:sampler.wait(timeout=10)
  except subprocess.TimeoutExpired:sampler.kill();sampler.wait()
 log.close()
 if out.exists():
  if execution.exists():execution.replace(out/'execution.log')
  if sampler_state['error']:(out/'SAMPLER_ERROR.json').write_text(json.dumps({'error':str(sampler_state['error'])}))
  (out/'EXIT.json').write_text(json.dumps({'exit':status,'done':(out/'DONE').exists(),'wall_end':time.time()}))
