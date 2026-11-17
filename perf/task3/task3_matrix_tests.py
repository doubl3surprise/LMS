import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
class MatrixTest(unittest.TestCase):
 def test_nested_subset_order_independent(self):
  from task3_matrix_core import subsets
  records=[{'id':str(i),'text':str(i)} for i in range(879)]
  a=subsets(records,[128,512,879]);b=subsets(list(reversed(records)),[128,512,879])
  self.assertEqual(a,b);self.assertEqual(a[128],a[512][:128]);self.assertEqual(a[512],a[879][:512]);self.assertEqual(len({r['id'] for r in a[879]}),879)
 def test_budget_never_exceeds_limit(self):
  import task3_matrix_core as core
  self.assertTrue(hasattr(core,'fit_budget'),'token budget helper missing')
  def encode(t):return list(t.encode())
  def decode(ids):return bytes(ids).decode('utf-8',errors='ignore')
  text='记忆 text '*30
  clipped,ids=core.fit_budget(text,13,encode,decode)
  self.assertLessEqual(len(encode(clipped)),13);self.assertEqual(ids,encode(clipped))
  self.assertEqual(core.fit_budget('short',99,encode,decode)[0],'short')
if __name__=='__main__':unittest.main()
