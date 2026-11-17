import unittest,xml.etree.ElementTree as E
import figures
class FigureTest(unittest.TestCase):
 def test_svg_is_valid_and_escapes_labels(self):
  s=figures.svg('A&B',{'LRU':[(0,0),(1,0.5),(2,1.0)]},'capacity','hit rate')
  root=E.fromstring(s)
  self.assertTrue(root.tag.endswith('svg'))
  self.assertIn('A&amp;B',s)
if __name__=='__main__':unittest.main()
