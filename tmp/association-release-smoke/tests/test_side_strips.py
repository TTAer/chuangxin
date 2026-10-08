import copy
import unittest
import numpy as np
from pcbparse.side_strips import band,world,tensor,number_side_points,refine_observations,recover_name_anchors,perpendicular_stroke

class SideStripTests(unittest.TestCase):
    def test_unique_name_recovers_number_without_numeric_ocr(self):
        points=[{'x':30,'y':40,'side':'left','score':.9}]
        texts=[{'text':'RESET','bbox':[35,36,56,44],'confidence':.99}]
        result=recover_name_anchors(points,[30,10,110,100],texts,{'7':{'names':['RESET']},'2':{'names':['GPIO']}},[])
        self.assertEqual(result[0]['number'],'7')
        self.assertEqual(result[0]['x'],30)
        self.assertEqual(recover_name_anchors(points,[30,10,110,100],texts,{'7':{'names':['RESET']}},[],['7']),[])

    def test_inferred_identity_requires_normal_stroke(self):
        image=np.full((80,80,3),255,np.uint8)
        image[40,26:35]=0
        self.assertTrue(perpendicular_stroke(image,{'x':30,'y':40,'side':'left'}))
        self.assertFalse(perpendicular_stroke(image,{'x':30,'y':40,'side':'top'}))
        self.assertFalse(perpendicular_stroke(image,{'x':-1,'y':40,'side':'left'}))

    def test_repeated_name_and_ambiguous_geometry_abstain(self):
        points=[{'x':30,'y':40,'side':'left','score':.9}]
        texts=[{'text':'VDD','bbox':[35,36,56,44],'confidence':.99}]
        catalog={'1':{'names':['VDD']},'2':{'names':['VDD']}}
        self.assertEqual(recover_name_anchors(points,[30,10,110,100],texts,catalog,[]),[])
        texts[0]['text']='RESET'; catalog={'7':{'names':['RESET']}}
        points.append({'x':30,'y':42,'side':'left','score':.9})
        self.assertEqual(recover_name_anchors(points,[30,10,110,100],texts,catalog,[]),[])
    def test_all_four_coordinate_transforms_preserve_source_pixels(self):
        image=np.arange(200*200,dtype=np.uint16).reshape(200,200)%251; image=image.astype(np.uint8); box=[50,50,150,150]
        for side in ('left','right','top','bottom'):
            patch=band(image,box,side,60)
            x,y=world(box,side,60,12,3)
            self.assertEqual(patch[35,12],image[int(y),int(x)])
            self.assertEqual(patch.shape,(64,256))
        self.assertEqual(tensor(patch).shape,(3,64,256))

    def test_outside_image_is_white(self):
        image=np.zeros((20,20),np.uint8); patch=band(image,[0,0,19,19],'left',-10)
        self.assertEqual(patch[0,0],255)

    def test_number_ocr_above_stub_and_monotone_order(self):
        points=[{'x':30,'y':40,'side':'left'},{'x':30,'y':60,'side':'left'}]
        texts=[{'text':'16','bbox':[20,26,28,38],'confidence':.99},{'text':'7','bbox':[20,46,28,58],'confidence':.99}]
        result=number_side_points(points,[30,10,100,100],texts)
        self.assertEqual([p['number'] for p in result],['16','7'])
        # Observed order is used; pin numbers are not sorted numerically.

    def test_unknown_catalog_keeps_existing_pins(self):
        class Locator:
            def __call__(self,*args): return [{'x':10,'y':20,'side':'left','score':.9}]
        obs={'components':{'U1':{'type':'ic','Name':'unknown','bbox':[20,0,80,90]}},
             'pins':{'U1':{'pin_1':{'point':{'x':10,'y':80},'pinname':'OLD'}}},'texts':[],'pin_hypotheses':[]}
        original=copy.deepcopy(obs)
        result,_=refine_observations(obs,np.full((100,100,3),255,np.uint8),Locator(),None)
        self.assertEqual(result,original); self.assertEqual(obs,original)
