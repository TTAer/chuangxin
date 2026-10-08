import unittest
import numpy as np
from pcbparse.topology_perception import TopologyPerception,dot_evidence,patch_tensor
from pcbparse.corrections import review_queue,validate_proposals,apply_semantic_choices
from pcbparse.graphschematic import parse_graph


class TopologyPerceptionTests(unittest.TestCase):
    def test_semantic_choice_uses_existing_ocr_and_cannot_change_geometry(self):
        obs={'components':{'R1':{'type':'resistor','Name':None,'value':'bad','bbox':[0,0,10,10]}},'pins':{'R1':{'pin_1':{'pinname':'','point':{'x':10,'y':5}}}},'wires':[],'labels':[],'junctions':[],'texts':[{'text':'10kΩ','bbox':[10,0,30,10]},{'text':'100kΩ','bbox':[10,15,30,25]}]}
        _,report=parse_graph(obs); queue=review_queue(obs,report)
        task=next(t for t in queue['tasks'] if t['kind']=='component_text_choice')
        payload={'observation_sha256':queue['observation_sha256'],'corrections':[{'candidate_id':task['id'],'decision':'choose_text','text_option_id':task['choices'][0]['id']}]}
        changed,edits=apply_semantic_choices(obs,payload,queue)
        self.assertEqual(changed['components']['R1']['value'],task['choices'][0]['text'])
        self.assertEqual(changed['pins'],obs['pins']); self.assertEqual(changed['wires'],obs['wires'])
        self.assertEqual(len(edits),1)
    def test_no_color_to_connection_rule_and_dot_normalizes_stroke_width(self):
        decisions=[]
        for color in ((0,0,0),(0,128,0),(0,0,180)):
            image=np.full((80,80,3),255,np.uint8); image[40,10:70]=color; image[10:70,40]=color
            _,j,_,r=TopologyPerception().perceive(image,[],[])
            self.assertFalse(j); decisions.append([c['state'] for c in r['crossings']])
        self.assertEqual(decisions[0],decisions[1]); self.assertEqual(decisions[1],decisions[2])
        mask=np.zeros((80,80),np.uint8); mask[38:42,10:70]=255; mask[10:70,38:42]=255
        self.assertFalse(dot_evidence(mask,40,40))
        mask=np.zeros((80,80),np.uint8); mask[40,10:70]=255; mask[10:70,40]=255; mask[38:43,38:43]=255
        self.assertTrue(dot_evidence(mask,40,40))

    def test_patch_feature_contract_and_T_topology(self):
        image=np.full((80,80,3),255,np.uint8); image[40,10:70]=0; image[10:41,40]=0
        self.assertEqual(patch_tensor(image,40,40).shape,(6,48,48))
        _,junctions,_,_=TopologyPerception().perceive(image,[],[])
        self.assertIn((40.,40.),junctions)

    def test_correction_cannot_edit_geometry_or_replay_on_other_graph(self):
        obs={'components':{'U1':{'type':'ic','Name':None,'value':None,'bbox':[0,0,10,10]}},'pins':{'U1':{'pin_1':{'pinname':'','point':{'x':10,'y':5}}}},'wires':[[{'x':14,'y':5},{'x':30,'y':5}]],'labels':[],'junctions':[]}
        _,report=parse_graph(obs); queue=review_queue(obs,report)
        with self.assertRaisesRegex(ValueError,'different observation'):
            validate_proposals({'observation_sha256':'wrong','corrections':[]},queue)
        cid=queue['tasks'][0]['id']
        with self.assertRaisesRegex(ValueError,'evidence gate'):
            validate_proposals({'observation_sha256':queue['observation_sha256'],'corrections':[{'candidate_id':cid,'decision':'accept'}]},queue)
        with self.assertRaisesRegex(ValueError,'unsupported correction'):
            validate_proposals({'observation_sha256':queue['observation_sha256'],'corrections':[{'candidate_id':cid,'decision':'reject','new_wire':[]} ]},queue)
