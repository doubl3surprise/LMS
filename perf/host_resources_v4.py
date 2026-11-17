import json,subprocess
from pathlib import Path
import host_sampler_v4 as base
# Host /proc IO is protected for root-owned container workers. Read it inside
# that same container, using existing Docker permission, not extra host privilege.
PROBE=r'''import os,json
from pathlib import Path
rows=[];skip={os.getpid(),os.getppid()};hz=os.sysconf('SC_CLK_TCK');pages=os.sysconf('SC_PAGE_SIZE')
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name) in skip:continue
 try:
  stat=(p/'stat').read_text();f=stat[stat.rindex(')')+2:].split();io=dict(line.split(':',1) for line in (p/'io').read_text().splitlines())
  rows.append({'pid':int(p.name),'create_time':float(f[19])/hz,'cpu_user_s':int(f[11])/hz,'cpu_system_s':int(f[12])/hz,'rss_bytes':int(f[21])*pages,'read_bytes':int(io['read_bytes']),'write_bytes':int(io['write_bytes']),'rchar':int(io['rchar']),'wchar':int(io['wchar']),'errors':[]})
 except (OSError,ValueError,KeyError):pass
print(json.dumps(rows))
'''
def enable_container_fallback():
 original=base.HostSampler.sample
 def sample(self):
  result=original(self)
  if not hasattr(self,'fallback_accumulator'):self.fallback_accumulator=base.ProcessAccumulator()
  records={};containers={'sglang':self.container,'qdrant':'memory-benchmarks-qdrant-1','benchmark':'mem0-benchmark-python'}
  for name,container in containers.items():
   if result['components'][name].get('read_bytes') is not None:continue
   # Qdrant image may lack Python; leave a genuine unsupported error in that case.
   try:
    p=subprocess.run(['docker','exec',container,'python3','-c',PROBE],capture_output=True,text=True,timeout=3,check=True)
    records[name]=json.loads(p.stdout)
   except (OSError,subprocess.SubprocessError,ValueError):
    if name!='qdrant':continue
    try:
     shell='for p in /proc/[0-9]*; do IFS= read -r comm < "$p/comm" 2>/dev/null || continue; case "$comm" in qdrant*) printf "PID %s\\n" "${p##*/}"; while IFS= read -r line; do printf "%s\\n" "$line"; done < "$p/io";; esac; done'
     text=subprocess.run(['docker','exec',container,'sh','-c',shell],capture_output=True,text=True,timeout=3,check=True).stdout
     io_rows=[];row=None
     for line in text.splitlines():
      if line.startswith('PID '):
       row={'pid':int(line[4:]),'create_time':0,'errors':[]};io_rows.append(row)
      elif row is not None and ':' in line:
       key,value=line.split(':',1)
       if key in ['read_bytes','write_bytes','rchar','wchar']:row[key]=int(value.strip())
     if io_rows:
      result['components'][name].update({key:sum(row[key] for row in io_rows) for key in ['read_bytes','write_bytes','rchar','wchar']})
      result['components'][name]['errors']=[]
      result['components'][name]['io_source']='container proc qdrant process counters'
    except (OSError,subprocess.SubprocessError,ValueError,KeyError):pass
    continue
  if records:
   fallback=self.fallback_accumulator.update(records)
   for name in records:
    # Use consistent fallback counter source for the entire component.
    result['components'][name]=dict(fallback[name],source='container_proc_root',scope='container process tree, sampler process excluded')
  return result
 base.HostSampler.sample=sample
if __name__=='__main__':
 enable_container_fallback();base.main()
