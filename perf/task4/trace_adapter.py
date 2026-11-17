"""Normalize selected memory-object reads; never infer request sessions from evidence."""
import hashlib
from collections import Counter,defaultdict

def events(rows, namespace):
 out=[];versions={}
 for i,r in enumerate(rows):
  rid=r.get('request_id',str(i));user=r.get('user_id',namespace)
  session=[user,r['request_session_id']] if r.get('request_session_id') is not None else None
  for pos,m in enumerate(r.get('memories',[])):
   ident=m.get('id',m.get('memory_id'));text=m['text']
   if ident is None:raise ValueError('Missing memory ID')
   key=str(user)+':'+str(ident);th=hashlib.sha256(text.encode('utf-8')).hexdigest()
   if key in versions and versions[key]!=th:raise ValueError('Changed memory object requires version-aware keys: '+key)
   versions[key]=th
   out.append({'request_id':rid,'request_index':i,'session_id':session,'object_key':key,'memory_id':str(ident),'text_key':th,'size':len(text.encode('utf-8')),'rank':m.get('rank',pos+1),'text':text})
 return out

def access_stats(es,field='object_key'):
 counts=Counter(e[field] for e in es);sessions=defaultdict(set);prior=set();repeat=0
 groups=defaultdict(list)
 for e in es:groups[e['request_index']].append(e)
 gaps=[];last={}
 for index,group in groups.items():
  for e in group:
   key=e[field];repeat+=key in prior
   if e['session_id'] is not None:sessions[key].add(tuple(e['session_id']))
  for key in {e[field] for e in group}:
   if key in last:gaps.append(index-last[key])
   last[key]=index;prior.add(key)
 ordered=sorted(counts.items(),key=lambda t:(-t[1],t[0]));n=len(es)
 return {'accesses':n,'unique':len(counts),'repeated_accesses':n-len(counts),'prior_request_repeat_accesses':repeat,'top10_share':sum(v for k,v in ordered[:10])/n if n else 0,'cross_session_objects':sum(len(v)>1 for v in sessions.values()) if any(e['session_id'] is not None for e in es) else None,'session_count':len({tuple(e['session_id']) for e in es if e['session_id'] is not None}),'frequency':ordered,'request_gaps':gaps}
