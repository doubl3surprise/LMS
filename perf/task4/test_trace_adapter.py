import unittest
import trace_adapter as a

class AdapterTest(unittest.TestCase):
 def test_qualified_identity_and_exact_text(self):
  r={'request_id':'r','memories':[{'id':'x','text':'hello'},{'id':'y','text':'hello'}]}
  e=a.events([r],namespace='u')
  self.assertNotEqual(e[0]['object_key'],e[1]['object_key'])
  self.assertEqual(e[0]['text_key'],e[1]['text_key'])
  self.assertEqual(e[0]['size'],5)
  self.assertIsNone(e[0]['session_id'])
 def test_only_actual_session_not_evidence(self):
  r={'request_id':'r','user_id':'u','request_session_id':'s','evidence':['bad'], 'memories':[{'memory_id':'x','text':'北京','rank':1}]}
  x=a.events([r],namespace='n')[0]
  self.assertEqual(x['session_id'],['u','s'])
  self.assertEqual(x['size'],6)
 def test_overlap_duplicates_within_request_not_cross_request(self):
  es=a.events([{'request_id':'a','memories':[{'id':'1','text':'x'},{'id':'2','text':'x'}]}, {'request_id':'b','memories':[{'id':'1','text':'x'}]}],namespace='u')
  s=a.access_stats(es,'text_key')
  self.assertEqual(s['repeated_accesses'],2)
  self.assertEqual(s['prior_request_repeat_accesses'],1)
  self.assertEqual(s['cross_session_objects'],None)
 def test_updated_object_payload_rejected(self):
  with self.assertRaises(ValueError):a.events([{'request_id':'r','memories':[{'id':'x','text':'a'},{'id':'x','text':'b'}]}],namespace='u')

if __name__=='__main__':unittest.main()
