import copy
import unittest
import cv2
import numpy as np
from pcbparse.component_boxes import body_box,apply_body_boxes

class ComponentBoxTests(unittest.TestCase):
    def test_parallel_plates_define_body_without_lead_extent(self):
        image=np.full((100,100,3),255,np.uint8)
        cv2.line(image,(30,45),(70,45),(0,0,0),2); cv2.line(image,(30,55),(70,55),(0,0,0),2)
        cv2.line(image,(50,15),(50,45),(0,0,0),1); cv2.line(image,(50,55),(50,85),(0,0,0),1)
        result=body_box(image,[20,10,80,90],'C')
        self.assertIsNotNone(result)
        self.assertGreater(result['bbox'][1],35); self.assertLess(result['bbox'][3],65)
        self.assertEqual(result['method'],'observed_parallel_capacitor_plates')
        remote=[{'bbox':[0,0,10,40],'text':'R8'},{'bbox':[0,0,60,5],'text':'C2'}]
        self.assertEqual(result,body_box(image,[20,10,80,90],'C',remote))

    def test_single_straight_stroke_and_other_types_abstain(self):
        image=np.full((100,100,3),255,np.uint8); cv2.line(image,(50,15),(50,85),(0,0,0),1)
        self.assertIsNone(body_box(image,[20,10,80,90],'C'))
        self.assertIsNone(body_box(image,[20,10,80,90],'Box'))

    def test_export_box_changes_preserve_all_pins_and_electrical_results(self):
        original={'components':{'C1':{'type':'capacitor','bbox':[0,0,80,90]}},'pins':{'C1':{'pin_1':{'point':{'x':40,'y':0},'pinname':'1'}}},'nets':{'n':{'hyperGraph':'(C1.1)','edges':{}}}}
        before=copy.deepcopy(original); after=apply_body_boxes(original,{'component_body_boxes':{'C1':[30,35,50,55]}})
        self.assertEqual(original,before); self.assertEqual(after['pins'],original['pins']); self.assertEqual(after['nets'],original['nets'])
        self.assertEqual(after['components']['C1']['bbox'],[30,35,50,55])
        with self.assertRaises(ValueError): apply_body_boxes(original,{'component_body_boxes':{'C2':[0,0,1,1]}})
