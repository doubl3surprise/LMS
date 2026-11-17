import hashlib

def subsets(records,sizes):
 ordered=sorted(records,key=lambda r:hashlib.sha256(str(r['id']).encode()).hexdigest())
 if len({r['id'] for r in ordered})!=len(ordered):raise ValueError('Duplicate IDs')
 if any(n>len(ordered) or n<1 for n in sizes):raise ValueError('Invalid size')
 return {n:ordered[:n] for n in sizes}

def fit_budget(text,budget,encode,decode):
 ids=encode(text)
 if budget is None or len(ids)<=budget:return text,ids
 cut=budget
 while cut>=0:
  result=decode(ids[:cut]);actual=encode(result)
  if len(actual)<=budget:return result,actual
  cut-=1
 raise ValueError('Cannot fit empty text')
