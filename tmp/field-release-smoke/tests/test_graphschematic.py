import unittest
import numpy as np
from pcbparse.core import build,references
from pcbparse.graphschematic import parse_graph
from pcbparse.graphschematic import DualGraph
import random


def observation(pins,wires,labels=(),junctions=()):
    return {'components':{c:{'type':'ic','Name':None,'value':None,'bbox':[-10,-10,10,10]} for c in pins},'pins':pins,'wires':wires,'labels':list(labels),'junctions':list(junctions),'image':{'height':100,'width':100}}


def point(x,y): return {'x':x,'y':y}
def pin(x,y): return {'pin_1':{'pinname':'','point':point(x,y)}}


class DualGraphTests(unittest.TestCase):
    def test_global_beam_can_abandon_early_choice_for_better_consistent_assignment(self):
        obs=observation({'A':pin(10,20),'B':pin(34,20)},[[point(14,20),point(30,20)]],[{'name':'3.3V','pin':'A.1'},{'name':'GND','pin':'B.1'}])
        graph=DualGraph(obs)
        # Controlled evidence scores test joint optimization, independently of
        # the current hand-weighted visual feature scorer.
        options=[{'id':'A.1@wire_0','pin':'A.1','wire':0,'score':.70,'topology_allowed':True,'image_evidence_available':True,'image_support':1}, {'id':'B.1@wire_0','pin':'B.1','wire':0,'score':.90,'topology_allowed':True,'image_evidence_available':True,'image_support':1}]
        uf,report=graph.associate(options)
        self.assertEqual([o['pin'] for o in report['accepted']],['B.1'])
        result,_=graph.export(uf); self.assertEqual(len(result['nets']),2)

    def test_spatial_contacts_preserve_reference_semantics_on_random_geometries(self):
        rng=random.Random(20261004)
        for _ in range(20):
            wires=[]
            for j in range(15):
                x,y=rng.randrange(0,30,5),rng.randrange(0,30,5)
                wires.append([point(x,y),point(x+rng.randrange(5,25),y)] if j%2 else [point(x,y),point(x,y+rng.randrange(5,25))])
            obs=observation({'A':pin(5,5),'B':pin(20,20)},wires,junctions=[point(10,10)])
            result,_=parse_graph(obs,optimize=False)
            self.assertEqual(result,build(obs,tolerance=3))
    def test_spatial_build_matches_old_topology_and_keeps_flying_relations_separate(self):
        obs=observation({'A':pin(0,0),'B':pin(20,0),'C':pin(50,50)},[[point(0,0),point(20,0)]],[{'name':'X','pin':'A.1'},{'name':'X','pin':'C.1'}])
        result,report=parse_graph(obs,optimize=False)
        self.assertEqual(result,build(obs,tolerance=3))
        self.assertEqual(len(report['geometry_graph']['edges']),1)
        self.assertEqual(len(report['electrical_hypergraph']['semantic_links']),1)
        self.assertEqual(references(result['nets']['net_1']['hyperGraph']),{'A.1','B.1','C.1'})

    def test_interior_crossing_needs_topology_evidence(self):
        obs=observation({'A':pin(0,10),'B':pin(10,0)},[[point(0,10),point(20,10)],[point(10,0),point(10,20)]])
        result,_=parse_graph(obs,optimize=False); self.assertEqual(len(result['nets']),2)
        obs['junctions']=[point(10,10)]
        result,_=parse_graph(obs,optimize=False); self.assertEqual(len(result['nets']),1)

    def test_topology_abstention_overrides_geometric_T_contact(self):
        obs=observation({'A':pin(0,10),'B':pin(10,0)},[[point(0,10),point(20,10)],[point(10,0),point(10,10)]])
        obs['contact_decisions']=[{'point':point(10,10),'state':'uncertain'}]
        result,_=parse_graph(obs,optimize=False); self.assertEqual(len(result['nets']),2)

    def test_global_association_rejects_new_power_ground_short(self):
        obs=observation({'A':pin(10,20),'B':pin(30,20)},[[point(14,20),point(30,20)]],[{'name':'GND','pin':'A.1'},{'name':'3.3V','pin':'B.1'}])
        mask=np.zeros((100,100),np.uint8); mask[80,10:31]=255
        result,report=parse_graph(obs,line_mask=mask)
        self.assertEqual(len(result['nets']),2)
        self.assertTrue(report['association']['rejected'])

    def test_evidence_supported_attach_preserves_geometry_and_pin_coordinates(self):
        obs=observation({'A':pin(10,20),'B':pin(30,20)},[[point(14,20),point(30,20)]])
        obs['components']['A']['bbox']=[-10,10,10,30]
        mask=np.zeros((100,100),np.uint8); mask[80,10:31]=255
        result,report=parse_graph(obs,line_mask=mask)
        self.assertEqual(len(result['nets']),1)
        self.assertEqual(result['pins'],obs['pins'])
        self.assertEqual(list(result['nets']['net_1']['edges'].values()),obs['wires'])
        self.assertEqual(len(report['association']['accepted']),1)

    def test_missing_pixels_cannot_be_overridden_by_llm(self):
        obs=observation({'A':pin(10,20),'B':pin(30,20)},[[point(14,20),point(30,20)]])
        result,report=parse_graph(obs)
        self.assertEqual(len(result['nets']),2)
        option=report['geometry_graph']['hypotheses'][0]['id']
        with self.assertRaisesRegex(ValueError,'visual evidence'): parse_graph(obs,forced=[option])
