import copy
import unittest
import numpy as np
from pcbparse.graphschematic import DualGraph,parse_graph
from pcbparse.association_features import candidate_graph

def observation():
    return {'components':{'R1':{'type':'resistor','Name':None,'value':'10k','bbox':[0,0,10,10]}},'pins':{'R1':{'pin_1':{'pinname':'','point':{'x':12,'y':5}},'pin_2':{'pinname':'','point':{'x':-2,'y':5}}}},'wires':[[{'x':16,'y':5},{'x':35,'y':5}]],'image':{'width':100,'height':100}}

class LearnedAssociationTests(unittest.TestCase):
    def test_relations_row_normalized_and_no_truth_features(self):
        obs=observation(); graph=DualGraph(obs); candidates=graph.candidates(np.ones((100,100),np.uint8))
        tensors=candidate_graph(graph,candidates)
        self.assertEqual(tensors['nodes'].shape,(3,10))
        self.assertEqual(tensors['edge_features'].shape,(1,12))
        for key in ('component_relation','incidence_relation'): np.testing.assert_allclose(tensors[key].sum(1),1)
        # Changing raw IDs and textual values cannot teach membership by lookup.
        renamed=copy.deepcopy(obs); renamed['components']={'X':renamed['components']['R1']}; renamed['pins']={'X':renamed['pins']['R1']}; renamed['components']['X']['value']='22k'
        other=DualGraph(renamed); features=candidate_graph(other,other.candidates(np.ones((100,100),np.uint8)))
        for key in tensors: np.testing.assert_array_equal(tensors[key],features[key])
    def test_veto_cannot_add_wire_or_move_pin(self):
        class Veto:
            def filter(self,graph,candidates): return [],{'blocked':[c['id'] for c in candidates]}
        obs=observation(); mask=np.ones((100,100),np.uint8)
        base,_=parse_graph(obs,line_mask=mask); result,report=parse_graph(obs,line_mask=mask,association_model=Veto())
        self.assertTrue(report['learned_association']['blocked'])
        self.assertEqual(result['pins'],obs['pins'])
        self.assertEqual(result['components'],obs['components'])
        self.assertEqual(len(report['geometry_graph']['edges']),1)
        self.assertEqual(len(report['association']['accepted']),0)
    def test_neural_acceptance_cannot_override_no_pixels_or_topology(self):
        class Keep:
            def filter(self,graph,candidates): return candidates,{}
        obs=observation(); result,report=parse_graph(obs,association_model=Keep())
        self.assertEqual(report['association']['accepted'],[])
        obs['contact_decisions']=[{'point':{'x':16,'y':5},'state':'uncertain'}]
        result,report=parse_graph(obs,line_mask=np.ones((100,100),np.uint8),association_model=Keep())
        self.assertEqual(report['association']['accepted'],[])
    def test_forced_correction_cannot_override_learned_veto(self):
        class Veto:
            def filter(self,graph,candidates): return [],{}
        with self.assertRaisesRegex(ValueError,'learned association veto'):
            parse_graph(observation(),line_mask=np.ones((100,100),np.uint8),forced=['R1.1@wire_0'],association_model=Veto())
    def test_high_neural_gain_cannot_override_original_evidence_threshold(self):
        graph=DualGraph(observation()); options=graph.candidates(np.ones((100,100),np.uint8))
        options[0]['score']=.4; options[0]['association_gain']=.9
        _,report=graph.associate(options)
        self.assertEqual(report['accepted'],[])

if __name__=='__main__': unittest.main()
