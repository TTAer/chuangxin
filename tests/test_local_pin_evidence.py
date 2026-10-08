import unittest
from experiments.local_pin_evidence import commit_additions

class EvidenceCommitTests(unittest.TestCase):
    def fixture(self):
        old={'components':{'U1':{'type':'ic','bbox':[0,0,10,10],'Name':'chip'}},'pins':{'U1':{'pin_1':{'pinname':'A','point':{'x':10,'y':0}}}},'nets':{'net_1':{'hyperGraph':'(U1.1)','edges':{'edge_1':[{'x':10,'y':0},{'x':20,'y':0}]}}}}
        obs=dict(old,wires=[[{'x':10,'y':0},{'x':20,'y':0}],[{'x':10,'y':10},{'x':20,'y':10}]],junctions=[],labels=[])
        return old,obs

    def test_existing_identity_cannot_be_overwritten(self):
        old,obs=self.fixture(); result,decisions=commit_additions(old,obs,[{'component':'U1','number':'1','name':'WRONG','point':{'x':10,'y':10}}])
        self.assertEqual(result,old); self.assertFalse(decisions[0]['committed'])

    def test_new_pin_does_not_change_old_edges_or_pins(self):
        old,obs=self.fixture(); result,decisions=commit_additions(old,obs,[{'component':'U1','number':'2','name':'B','point':{'x':10,'y':10}}])
        self.assertTrue(decisions[0]['committed']); self.assertEqual(result['nets']['net_1'],old['nets']['net_1']); self.assertEqual(result['pins']['U1']['pin_1'],old['pins']['U1']['pin_1'])

    def test_no_wire_contact_abstains(self):
        old,obs=self.fixture(); result,d=commit_additions(old,obs,[{'component':'U1','number':'2','name':'B','point':{'x':50,'y':50}}])
        self.assertEqual(result,old); self.assertFalse(d[0]['committed'])
