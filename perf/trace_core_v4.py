"""Small, dependency-light trace transformations. All durations have explicit boundaries."""

def build_prompt(question, memories):
 """Offsets point to full memory text in the user message, excluding each label."""
 import copy
 items=copy.deepcopy(memories);parts=[];position=len('Memories:\n')
 for i,m in enumerate(items,1):
  label=f'[Memory {i}] ';m['prompt_char_offset']=position+len(label);m['text_len']=len(m['text'])
  line=label+m['text'];parts.append(line);position+=len(line)+1
 block='\n'.join(parts)
 user='Memories:\n'+block+f'\n\nQuestion: {question}\n\nAnswer in 1-2 sentences:'
 return {'messages':[{'role':'system','content':'You are a helpful assistant. Answer the question based on the provided memories. Be concise.'},{'role':'user','content':user}],'memory_block':block,'memories':items}

def slowdown(baseline, concurrent):
 """Ratio of mean observed elapsed times; undefined for missing/zero baseline."""
 if not baseline or not concurrent or sum(baseline)<=0:return None
 return (sum(concurrent)/len(concurrent))/(sum(baseline)/len(baseline))

def with_session_history(prompt, history):
 """Insert actual preceding turns; memory offsets are local to the last user message."""
 import copy
 result=copy.deepcopy(prompt)
 result['messages']=[result['messages'][0]]+copy.deepcopy(history)+[result['messages'][-1]]
 return result

def native_timing(meta, client_ttft_ms):
 start=meta.get('forward_entry_time');end=meta.get('prefill_finished_time')
 elapsed=(end-start)*1000 if isinstance(start,(int,float)) and isinstance(end,(int,float)) and end>=start else None
 queue=meta.get('queue_time')
 return {'server_prefill_elapsed_ms':elapsed,'server_queue_ms':queue*1000 if isinstance(queue,(int,float)) else None,'client_ttft_ms':client_ttft_ms}
