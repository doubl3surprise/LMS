#!/usr/bin/env python3
"""Create self-contained SVG line chart from real sample series; no plotting dependency."""
import argparse,html,json,math
from pathlib import Path

def render_series(series,title,ylabel,width=900,height=280):
 valid=[(float(x),float(y)) for _,points in series for x,y in points if math.isfinite(float(x)) and math.isfinite(float(y))]
 if not valid:return '<svg xmlns="http://www.w3.org/2000/svg"><text y="20">No samples</text></svg>'
 xmin=min(x for x,y in valid);xmax=max(x for x,y in valid);ymax=max(y for x,y in valid)*1.1 or 1;left=70;right=230;top=40;bottom=45;colors=['#2563eb','#dc2626','#059669','#7c3aed','#d97706','#0891b2']
 def xy(x,y):return left+(x-xmin)/max(xmax-xmin,1e-9)*(width-left-right),height-bottom-y/ymax*(height-top-bottom)
 s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="11" fill="#111827">',f'<text x="{left}" y="22" font-size="15">{html.escape(title)}</text>']
 for i in range(5):
  value=ymax*i/4;_,y=xy(xmin,value);s.append(f'<path d="M {left} {y} H {width-right}" stroke="#e5e7eb"/><text x="5" y="{y+4}">{value:.1f}</text>')
 for i,(name,points) in enumerate(series):
  color=colors[i%len(colors)];coords=' '.join(f'{x:.2f},{y:.2f}' for x,y in [xy(float(x),float(y)) for x,y in points]);s.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.5"/><text x="{width-right+12}" y="{top+18*i}" fill="{color}">{html.escape(name)}</text>')
 s.append(f'<text x="{left}" y="{height-10}">Elapsed seconds: {xmin:.1f} to {xmax:.1f}; {html.escape(ylabel)}</text></g></svg>');return ''.join(s)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);a=p.parse_args();d=Path(a.run_dir)
 traces=[json.loads(s) for s in (d/'trace.jsonl').read_text().splitlines() if s.strip()];ms=[json.loads(s) for s in (d/'metrics_samples.jsonl').read_text().splitlines() if s.strip()];t0=traces[0]['wall_start'];series={}
 for sample in ms:
  for metric in sample['metrics']:
   name=metric['name']
   if name not in ['sglang:kv_used_tokens','sglang:kv_evictable_tokens','sglang:kv_available_tokens','sglang:mamba_used_tokens']:continue
   if name.endswith('kv_available_tokens'):continue
   label=name.replace('sglang:','')+' rank='+str(metric['labels'].get('tp_rank','?'))
   series.setdefault(label,[]).append((sample['timestamp']-t0,metric['value']))
 (d/'kv_timeline.svg').write_text(render_series(list(series.items()),'KV / Mamba occupancy (independent units)','token/state slots'))
 points=[(i+1,t['timing']['server_prefill_elapsed_ms']) for i,t in enumerate(traces)];resp=[(i+1,t['timing']['response_ms']) for i,t in enumerate(traces)]
 (d/'latency.svg').write_text(render_series([('server Prefill ms',points),('response ms',resp)],'Request latency, Judge excluded','request index (x), milliseconds (y)'))
if __name__=='__main__':main()
