"""Task4 offline analysis. No model requests, no writes to memory stores."""
import argparse,json,hashlib,statistics,csv
from pathlib import Path
from collections import Counter,defaultdict
from trace_adapter import events,access_stats
from figures import svg

def select_rows(tag,rows):
 return [r for r in rows if r['condition']['id']=='native_n879_k10'] if tag=='native32' else rows

def replay_stats(rows):
 out={}
 for cond in sorted({r['condition'] for r in rows}):
  group=[r for r in rows if r['condition']==cond];first={r['pair']:r for r in group if r['phase']=='first'};repeat={r['pair']:r for r in group if r['phase']=='repeat'}
  assert set(first)==set(repeat)
  out[cond]={'pair_count':len(first),'repeat_hit_requests':sum(r['cached_tokens']>0 for r in repeat.values()),'prefill_first_mean_ms':statistics.mean(r['prefill_ms'] for r in first.values()),'prefill_repeat_mean_ms':statistics.mean(r['prefill_ms'] for r in repeat.values()),'repeat_cached_tokens_mean':statistics.mean(r['cached_tokens'] for r in repeat.values()),'first_cached_tokens':[r['cached_tokens'] for r in first.values()]}
 return out

def memory_coverage(spans,prefix,seen_text):
 all_tokens=set();repeat=set()
 for s in spans:
  tokens=set(range(s['start'],s['end']));all_tokens.update(tokens)
  if s['text_sha256'] in seen_text:repeat.update(tokens)
 return {'memory_tokens':len(all_tokens),'previously_seen_text_tokens':len(repeat),'compatible_memory_prefix_tokens':sum(t<prefix for t in all_tokens),'compatible_repeated_text_tokens':sum(t<prefix for t in repeat)}

def compare_sequences(left,right):
 return {'same_question_and_id_text_order':left==right,'same_question_and_text_order':[(q,[text for ident,text in ms]) for q,ms in left]==[(q,[text for ident,text in ms]) for q,ms in right]}

def load(path):return [json.loads(l) for l in path.read_text().splitlines()]

def main():
 from cache_core import reuse_distances,simulate
 from prefix_core import analyze_prefixes
 parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,default=Path(__file__).parent/'inputs');parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
 out=args.out;out.mkdir(parents=True,exist_ok=True);result={'cohorts':{},'replay':{},'hardware':{},'sources':{}};curves=[];access_export=[];frequency_export=[]
 labels={'native32':'Mem0 native / 32 independent QA','mmr152':'Custom MMR / 152 independent QA','sessions':'Custom MMR / 9 curated turns (3 sessions)'}
 expected={'native32':32,'mmr152':152,'sessions':9};signatures={}
 for host in ['dgx','5880ada']:
  result['cohorts'][host]={};prompt_file=args.inputs/host/'prompt_ids.json';prompt_data=json.loads(prompt_file.read_text()) if prompt_file.exists() else {}
  span_file=args.inputs/host/'memory_spans.json';span_data=json.loads(span_file.read_text()) if span_file.exists() else {}
  for tag in labels:
   path=args.inputs/host/(tag+'_trace.jsonl');rows=select_rows(tag,load(path));assert len(rows)==expected[tag],(host,tag,len(rows));assert len({r['request_id'] for r in rows})==len(rows)
   result['sources'][str(path.relative_to(args.inputs))]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected_requests':len(rows)}
   es=events(rows,namespace='conv-26');signatures[host,tag]=[(r.get('source_q_idx'),[(m.get('id',m.get('memory_id')),m['text']) for m in r['memories']]) for r in rows]
   if tag!='sessions':
    for e in es:e['session_id']=None
   s={'label':labels[tag],'requests':len(rows),'objects':access_stats(es,'object_key'),'texts':access_stats(es,'text_key'),'observed_cached_tokens':sum(r['tokens'].get('cached',r['tokens'].get('cached_tokens',0)) for r in rows),'prompt_tokens':sum(r['tokens'].get('prompt',r['tokens'].get('prompt_tokens',0)) for r in rows)}
   ids=prompt_data.get(tag,{})
   if ids:
    entry=[ids[r['request_id']] for r in rows]
    for r,e in zip(rows,entry):
     assert hashlib.sha256(json.dumps(e['token_ids']).encode()).hexdigest()==r.get('input_sha256',r.get('prompt_token_ids_sha256'))
     assert len(e['token_ids'])==r['native_meta']['prompt_tokens']
    seqs=[e['token_ids'] for e in entry];details=analyze_prefixes(seqs)
    s['prefix']={'requests':details,'logical_repeated_tokens':sum(x['prior_max_prefix_tokens'] for x in details),'prompt_tokens':sum(map(len,seqs)),'scope':'global within this cohort, block_size=1, no eviction','hypothetical_block_tokens':{str(b):sum(x['block_aligned_prefix_tokens'] for x in analyze_prefixes(seqs,block_size=b)) for b in [16,64,256]}}
    (out/f'{host}_{tag}_prefix.svg').write_text(svg(f'{host}/{tag}: logical prefix opportunity',{'eligible prefix':[(i+1,x['prior_max_prefix_tokens']) for i,x in enumerate(details)]},'request index','prefix tokens'))
    seen=set();coverage=[]
    for r,d in zip(rows,details):
     v=span_data.get(tag,{}).get(r['request_id'],{})
     if v.get('verified'):
      assert len(v['spans'])==len(r['memories'])
      for m,span in zip(r['memories'],v['spans']):
       assert span['memory_id']==str(m.get('id',m.get('memory_id')))
       assert span['text_sha256']==hashlib.sha256(m['text'].encode()).hexdigest()
       assert 0<=span['start']<span['end']<=r['native_meta']['prompt_tokens']
      metrics=memory_coverage(v['spans'],d['prior_max_prefix_tokens'],seen)
      measured=memory_coverage(v['spans'],r['tokens'].get('cached',r['tokens'].get('cached_tokens',0)),seen)
      coverage.append({'request_id':r['request_id'],**metrics,'cached_prefix_memory_tokens':measured['compatible_memory_prefix_tokens'],'cached_prefix_repeated_text_tokens':measured['compatible_repeated_text_tokens']})
     seen.update(hashlib.sha256(m['text'].encode()).hexdigest() for m in r['memories'])
    s['memory_prefix']={'verified_requests':len(coverage),'excluded_requests':len(rows)-len(coverage),'requests':coverage,'counts':{k:sum(x[k] for x in coverage) for k in ['memory_tokens','previously_seen_text_tokens','compatible_memory_prefix_tokens','compatible_repeated_text_tokens','cached_prefix_memory_tokens','cached_prefix_repeated_text_tokens']},'span_scope':'current user memory text; overlapping-token boundary convention; verified subset only'}
    if tag=='sessions':
     bysession=defaultdict(list)
     for r,seq in zip(rows,seqs):bysession[r['request_session_id']].append(seq)
     s['prefix']['within_session_repeated_tokens']=sum(sum(x['prior_max_prefix_tokens'] for x in analyze_prefixes(g)) for g in bysession.values())
   else:s['prefix']=None
   for field in ['object_key','text_key']:
    seq=[e[field] for e in es];rd=reuse_distances(seq);finite=[v for v in rd if v is not None]
    stat=s['objects' if field=='object_key' else 'texts'];stat['reuse_distance']={'cold':sum(v is None for v in rd),'histogram':dict(Counter(finite)),'median_non_cold':statistics.median(finite) if finite else None}
    for e,d in zip(es,rd):access_export.append({'host':host,'cohort':tag,'identity':field,'key':e[field],'request_id':e['request_id'],'request_index':e['request_index'],'rank':e['rank'],'utf8_bytes':e['size'],'reuse_distance':d,'session_id':e['session_id']})
    for key,n in stat['frequency']:frequency_export.append({'host':host,'cohort':tag,'identity':field,'key':key,'accesses':n})
    cev=[{'key':e[field],'size':e['size'],'request_id':e['request_id']} for e in es];unique=len(set(seq));totalbytes=sum({e[field]:e['size'] for e in es}.values())
    for unit,caps in [('objects',sorted(set([0,1,2,4,8,10,16,32,64,128,256,512,unique]))),('bytes',sorted(set([0,256,512,1024,2048,4096,8192,16384,32768,65536,totalbytes])))]:
     for policy in ['LRU','LFU']:
      values=[]
      for cap in caps:
       v=simulate(cev,cap,policy=policy,unit=unit);curves.append({'host':host,'cohort':tag,'identity':field,'policy':policy,'unit':unit,'capacity':cap,**v});values.append((cap,v['hit_rate']))
       if unit=='objects' and policy=='LRU':assert v['hits']==sum(d is not None and d<cap for d in rd)
      if field=='object_key':
       s.setdefault('curves',{}).setdefault(unit,{})[policy]=values
   s['utf8_unique_object_bytes']=sum({e['object_key']:e['size'] for e in es}.values());s['utf8_access_bytes']=sum(e['size'] for e in es)
   result['cohorts'][host][tag]=s
   if host=='dgx':
    for kind,title in [('objects','object'),('texts','exact text')]:
     stat=s[kind];freq=stat['frequency'];(out/f'{tag}_{kind}_frequency.svg').write_text(svg(f'{tag}: {title} access frequency',{'count':[(i+1,n) for i,(_,n) in enumerate(freq)]},'rank','accesses'))
    for unit in ['objects','bytes']:(out/f'{tag}_{unit}_capacity.svg').write_text(svg(f'{tag}: cold-start cache sensitivity',s['curves'][unit],unit,'hit ratio'))
    rd=s['objects']['reuse_distance']['histogram'];cum=0;pts=[]
    for d,n in sorted(rd.items()):cum+=n;pts.append((d,cum/max(1,s['objects']['accesses']-s['objects']['reuse_distance']['cold'])))
    (out/f'{tag}_reuse.svg').write_text(svg(f'{tag}: non-cold object reuse distance CDF',{'CDF':pts},'distinct intervening objects','fraction of non-cold reads'))
  result['replay'][host]=replay_stats(load(args.inputs/host/'cache_trace.jsonl'))
  hp=Path(__file__).parent/(host+'_hardware.json');result['hardware'][host]=json.loads(hp.read_text()) if hp.exists() else None
 result['cross_platform']={tag:compare_sequences(signatures['dgx',tag],signatures['5880ada',tag]) for tag in labels}
 both={h:json.loads((args.inputs/h/'prompt_ids.json').read_text()) for h in ['dgx','5880ada']}
 for tag in labels:
  traces={h:select_rows(tag,load(args.inputs/h/(tag+'_trace.jsonl'))) for h in both}
  hashes={h:[both[h][tag][r['request_id']]['sha256'] for r in traces[h]] for h in both}
  result['cross_platform'][tag]['same_prompt_token_hashes']=hashes['dgx']==hashes['5880ada']
  assert result['cross_platform'][tag]['same_prompt_token_hashes']
 result['input_files']={str(p.relative_to(args.inputs)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.inputs.rglob('*.json*')) if p.name!='manifest.json'}
 (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 for name,data in [('capacity_curves.csv',curves),('frequency.csv',frequency_export),('access_reuse.csv',access_export)]:
  with (out/name).open('w') as f:
   keys=list(dict.fromkeys(k for row in data for k in row));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
 from report import write_report
 write_report(result,curves,out)
 print(json.dumps({'out':str(out.resolve()),'cohorts':{h:{k:v['requests'] for k,v in s.items()} for h,s in result['cohorts'].items()},'cross_platform':result['cross_platform'],'curves':len(curves)},ensure_ascii=False))

if __name__=='__main__':main()
