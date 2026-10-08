import copy
import unittest
import cv2
import numpy as np
from pcbparse.ic_structure import StructureAwareICParser,monotone_matches,side_of

class Library:
    entries={'ABC1234':{'pins':{'1':{'names':['VDD']},'2':{'names':['VDD']}}}}
    def identify(self,*args): return 'ABC1234',{}

def fixture():
    return {'components':{'U1':{'type':'ic','Name':'ABC1234','bbox':[30,10,110,90]}},
            'pins':{'U1':{'pin_1':{'pinname':'','point':{'x':23,'y':60}},'pin_2':{'pinname':'','point':{'x':23,'y':40}}}},
            'texts':[{'text':'1','bbox':[20,61,28,69],'confidence':.99}, {'text':'2','bbox':[20,41,28,49],'confidence':.99},
                     {'text':'VDD','bbox':[36,56,52,64],'confidence':.99}, {'text':'VDD','bbox':[36,36,52,44],'confidence':.99}],
            'wires':[], 'pin_hypotheses':[]}

class ICStructureTests(unittest.TestCase):
    def test_ordered_alignment_skips_and_does_not_cross(self):
        self.assertEqual(monotone_matches([[1,0,0],[0,0,1]]),[(0,0),(1,2)])
        result=monotone_matches([[0,10],[10,0]])
        self.assertEqual(len(result),1)
        self.assertEqual(monotone_matches([[],[]]),[])

    def test_side_assignment_four_sides(self):
        box=[20,20,80,80]
        for p,s in [((10,50),'left'),((90,50),'right'),((50,10),'bottom'),((50,90),'top')]:
            self.assertEqual(side_of(dict(zip(('x','y'),p)),box),s)

    def test_repeated_names_use_numbered_anchors_and_actual_stroke(self):
        obs=fixture(); original=copy.deepcopy(obs); image=np.full((100,140,3),255,np.uint8)
        cv2.line(image,(15,40),(35,40),(0,0,0),1); cv2.line(image,(15,60),(35,60),(0,0,0),1)
        result,report=StructureAwareICParser(Library()).refine_geometry(obs,image)
        self.assertEqual(report['geometry_edit_count'],2)
        self.assertEqual(result['pins']['U1']['pin_1'],{'pinname':'VDD','point':{'x':30.,'y':60.}})
        self.assertEqual(result['pins']['U1']['pin_2']['point']['y'],40.)
        self.assertEqual(obs,original); self.assertEqual(result['wires'],original['wires'])

    def test_blank_or_no_numeric_image_abstains(self):
        parser=StructureAwareICParser(Library()); obs=fixture(); blank=np.full((100,140,3),255,np.uint8)
        self.assertEqual(parser.refine_geometry(obs,blank)[0],obs)
        cv2.line(blank,(15,40),(35,40),(0,0,0),1)
        obs['texts']=[t for t in obs['texts'] if t['text']=='VDD']
        self.assertEqual(parser.refine_geometry(obs,blank)[0],obs)

    def test_foreign_side_names_and_unknown_number_do_not_create_pin(self):
        obs=fixture(); obs['texts']=[dict(t,bbox=[90,t['bbox'][1],105,t['bbox'][3]]) if t['text']=='VDD' else t for t in obs['texts']]
        image=np.full((100,140,3),255,np.uint8); cv2.line(image,(15,40),(35,40),(0,0,0),1)
        result,report=StructureAwareICParser(Library()).refine_geometry(obs,image)
        self.assertEqual(report['edits'],0); self.assertEqual(result,obs)
        self.assertEqual(report['fabricated_endpoints'],0)
