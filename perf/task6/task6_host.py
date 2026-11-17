"""Task6 scoped CPU/IO/GPU sampler; GPU memory activity is percent, not GB/s."""
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import host_resources_v4 as host
base=host.base;original=base.discover_components

def discover(container='jiangchuan-qwen38-zh'):
 result=original(container)
 for p in base.psutil.process_iter(['pid','cmdline']):
  try:
   args=p.info.get('cmdline') or []
   if any(Path(a).name=='task6_probe.py' for a in args):
    result['pids']['benchmark'].add(p.pid)
    result['pids']['benchmark'].update(x.pid for x in p.children(recursive=True))
  except (base.psutil.Error,OSError):pass
 result['pids']['benchmark']-=result['pids']['sglang']|result['pids']['qdrant']
 if result['pids']['benchmark']:result['errors']['benchmark']=[]
 return result
base.discover_components=discover
if __name__=='__main__':
 host.enable_container_fallback()
 original_sample=base.HostSampler.sample
 def sample(self):
  result=original_sample(self)
  try:
   p=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=3)
   values={}
   for line in p.stdout.splitlines():
    key,val=line.split(',',1)
    try:values[int(key.strip())]=float(val.strip())
    except ValueError:values[int(key.strip())]=None
   for gpu in result['gpus']:gpu['memory_activity_pct']=values.get(gpu.get('gpu_index'))
  except (subprocess.SubprocessError,OSError,ValueError):
   for gpu in result['gpus']:gpu['memory_activity_pct']=None
  result['memory_activity_semantics']='nvidia-smi utilization.memory percent of sample period with device memory activity; not percent of peak bandwidth, not GB/s, not unified CPU+GPU memory-controller traffic'
  return result
 base.HostSampler.sample=sample;base.main()
