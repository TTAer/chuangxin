import unittest
import numpy as np
from pcbparse.identity_transaction import guard_new_components,wire_signature
from pcbparse.semantic_library import SemanticLibrary

class Library(SemanticLibrary):
    def __init__(self): self.entries={'ABC12345':{'pins':{'1':{'names':['VDD']}}}}

def fixture():
    obs={'components':{'U1':{'type':'ic','Name':'ABC12345','bbox':[20,20,40,80]},'R1':{'type':'resistor','Name':None,'value':'1k','bbox':[0,40,10,60]}},
         'pins':{'U1':{'pin_1':{'pinname':'VDD','point':{'x':20,'y':50}},'pin_200':{'pinname':'VDD','point':{'x':20,'y':60}}},'R1':{'pin_1':{'pinname':'','point':{'x':10,'y':50}}}},
         'wires':[[{'x':10,'y':50},{'x':20,'y':50}]],'labels':[],'junctions':[],
         'designator_association':{'records':[{'output_key':'U1','identity_new':True}]},'texts':[]}
    image=np.full((100,100,3),255,np.uint8); image[47:54,17:24]=0
    return obs,image

class IdentityTransactionTests(unittest.TestCase):
    def test_invalid_number_is_hypothesis_and_existing_geometry_preserved(self):
        obs,image=fixture(); actual,report=guard_new_components(obs,image,Library())
        self.assertEqual(set(actual['pins']['U1']),{'pin_1'})
        self.assertEqual(actual['pins']['R1'],obs['pins']['R1'])
        self.assertEqual(wire_signature(actual),wire_signature(obs))
        self.assertEqual(len(obs['pins']['U1']),2)
        self.assertTrue(any('number_not_in_confirmed_catalog' in d['reasons'] for d in report['records'][0]['decisions']))

    def test_no_wire_contact_keeps_component_without_electrical_pin(self):
        obs,image=fixture(); obs['wires']=[]
        actual,_=guard_new_components(obs,image,Library())
        self.assertIn('U1',actual['components']); self.assertEqual(set(actual['pins']['U1']),{'pin_1'})
        self.assertFalse(actual['electrical_pin_permissions']['U1.1'])
        from pcbparse.graphschematic import parse_graph
        pred,_=parse_graph(actual)
        self.assertTrue(all('U1.1' not in n['hyperGraph'] for n in pred['nets'].values()))

    def test_new_endpoint_cannot_bridge_separate_wires(self):
        obs,image=fixture(); obs['pins']['R1']={}; obs['wires']=[[{'x':0,'y':47},{'x':30,'y':47}],[{'x':0,'y':53},{'x':30,'y':53}]]
        actual,report=guard_new_components(obs,image,Library())
        self.assertFalse(actual['electrical_pin_permissions']['U1.1'])
        self.assertIn('would_bridge_existing_wire_groups',report['records'][0]['decisions'][0]['reasons'])

    def test_name_evidence_without_ink_cannot_create_net(self):
        obs,image=fixture(); image[:]=255
        actual,_=guard_new_components(obs,image,Library())
        self.assertEqual(actual['pins']['U1'],{})

    def test_catalog_scope_preserves_unknown_upstream_perception(self):
        obs,image=fixture(); obs['components']['U1']['Name']='UNKNOWN999'
        actual,report=guard_new_components(obs,image,Library(),mode='catalog')
        self.assertEqual(actual['pins'],obs['pins'])
        self.assertEqual(report['records'][0]['confidence_level'],'existing_perception')

    def test_quarantined_endpoint_stays_in_geometry_graph(self):
        obs,image=fixture(); obs['wires']=[]
        actual,_=guard_new_components(obs,image,Library())
        from pcbparse.graphschematic import parse_graph
        pred,report=parse_graph(actual)
        self.assertIn('U1.1',report['geometry_graph']['nodes'])
        self.assertFalse(report['geometry_graph']['nodes']['U1.1']['electrical_allowed'])
