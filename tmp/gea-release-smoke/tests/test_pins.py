import unittest
import numpy as np
from pcbparse.pins import crop_tensor,peaks
from pcbparse.vision import refine_pin_locations,select_ic_name,correct_resistance_ocr


class PinTests(unittest.TestCase):
    def test_ic_identity_does_not_use_border_signal_or_power_value(self):
        texts=[{'text':'GPIO16','bbox':[0,20,20,30]}, {'text':'3.3V','bbox':[30,105,50,115]}, {'text':'STM32F103','bbox':[25,40,75,55]}]
        self.assertEqual(select_ic_name(texts,[0,0,100,100]),'STM32F103')
        self.assertEqual(correct_resistance_ocr('10kQ'),'10kΩ')
        self.assertEqual(correct_resistance_ocr('Q3'),'Q3')
    def test_crop_at_border_is_clipped_and_coordinates_are_consistent(self):
        image=np.full((30,40,3),255,np.uint8)
        tensor,box=crop_tensor(image,[0,0,10,10])
        self.assertEqual(tensor.shape,(6,96,96)); self.assertEqual(box[:2],[0,0])
        self.assertEqual(float(tensor[4,0,0]),-1); self.assertEqual(float(tensor[5,-1,-1]),1)

    def test_peak_plateau_is_not_counted_many_times(self):
        heat=np.zeros((20,20),np.float32); heat[4:6,4:6]=.8; heat[15,15]=.9
        self.assertEqual(len(peaks(heat)),2)

    def test_refinement_is_one_to_one_and_rejects_far_candidates(self):
        priors=[{'number':'1','name':'','x':0,'y':10},{'number':'2','name':'','x':20,'y':10}]
        result=refine_pin_locations(priors,[{'x':2,'y':10},{'x':100,'y':100}],[0,0,20,20])
        self.assertEqual(result[0]['x'],2); self.assertEqual(result[1]['x'],20)
        self.assertEqual([p['number'] for p in result],['1','2'])
