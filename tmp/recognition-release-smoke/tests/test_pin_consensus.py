import copy
import unittest
from pcbparse.pin_consensus import fuse_observations,electrical_signature,replay_electrical_associations
from pcbparse.graphschematic import parse_graph

def obs(x):
    return {'components':{'R1':{'type':'resistor','Name':None,'value':'10k','bbox':[0,0,5,5]}},'pins':{'R1':{'pin_1':{'pinname':'1','point':{'x':x,'y':10}}}},'wires':[[{'x':10,'y':10},{'x':40,'y':10}]],'image':{'width':100,'height':100},'labels':[]}

class ConsensusTests(unittest.TestCase):
    def test_agreement_fuses_while_preserving_wire_and_pin_identity(self):
        base,other=obs(15),obs(17); saved=copy.deepcopy(base)
        result,report=fuse_observations(base,other)
        self.assertEqual(result['pins']['R1']['pin_1']['point']['x'],16)
        self.assertEqual(base,saved); self.assertEqual(result['wires'],base['wires'])
        self.assertEqual(len(report['accepted']),1)
        self.assertEqual(electrical_signature(parse_graph(base)[0]),electrical_signature(parse_graph(result)[0]))

    def test_geometric_move_that_adds_connection_is_rejected(self):
        base,other=obs(6),obs(10)
        result,report=fuse_observations(base,other)
        self.assertEqual(result,base)
        self.assertEqual(report['rejected'][0]['reason'],'electrical_partition_change')

    def test_large_disagreement_and_semantic_disagreement_abstain(self):
        base,other=obs(15),obs(30)
        result,report=fuse_observations(base,other); self.assertEqual(result,base)
        self.assertEqual(report['rejected'][0]['reason'],'model_disagreement')
        other=obs(17); other['pins']['R1']['pin_1']['pinname']='different'
        self.assertEqual(fuse_observations(base,other)[0],base)

    def test_cannot_merge_different_image_geometry(self):
        base,other=obs(15),obs(17); other['wires']=[]
        with self.assertRaises(ValueError): fuse_observations(base,other)

    def test_refining_coordinates_does_not_create_new_global_associations(self):
        base=obs(4); _,report=parse_graph(base,optimize=False)
        result,new_report=replay_electrical_associations(base,report)
        self.assertEqual(new_report['association']['accepted'],[])
        self.assertTrue(new_report['association']['replayed_for_pin_consensus'])
        self.assertEqual(electrical_signature(result),electrical_signature(parse_graph(base,optimize=False)[0]))
