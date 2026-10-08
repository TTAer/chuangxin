import unittest
from pcbparse.competition import export_competition

class CompetitionCategoryFallbackTests(unittest.TestCase):
    def test_ambiguous_public_m_does_not_fail_whole_page(self):
        data={'components':{'TP1':{'type':'m','Name':None,'value':None,'bbox':[0,0,10,10]}},'pins':{'TP1':{}},'nets':{}}
        actual=export_competition(data,{'TP1':'M'})
        self.assertEqual(actual['components']['TP1']['type'],'other')
        self.assertEqual(actual['components']['TP1']['Name'],'TP1')
        self.assertEqual(data['components']['TP1']['type'],'m')

    def test_unrecognized_schema_type_still_rejected(self):
        data={'components':{'X':{'type':'unregistered','Name':None,'bbox':[0,0,10,10]}},'pins':{'X':{}},'nets':{}}
        with self.assertRaises(ValueError): export_competition(data)
