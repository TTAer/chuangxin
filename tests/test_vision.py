import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from pcbparse.core import references,validate
from pcbparse.dataset import convert_raw,index_dataset
from pcbparse.vision import masks,wire_lines,runs,read_image,write_image
from pcbparse.batch import run_batch


class VisionTests(unittest.TestCase):
    def test_color_masks_separate_wire_and_symbol(self):
        image=np.full((30,40,3),255,np.uint8)
        image[5:7,2:30]=(0,128,0)
        image[15:17,2:30]=(0,0,180)
        symbol,green,colored=masks(image)
        self.assertTrue(colored)
        self.assertEqual(int(symbol[15,5]),255); self.assertEqual(int(green[15,5]),0)
        self.assertEqual(int(symbol[5,5]),0); self.assertEqual(int(green[5,5]),255)

    def test_thick_line_is_single_segment(self):
        m=np.zeros((30,60),np.uint8); m[10:13,5:45]=255
        segments,_=wire_lines(m)
        self.assertEqual(len(segments),1)
        self.assertEqual(segments[0],((5.,11.),(44.,11.)))

    def test_wire_T_splits_and_cross_does_not(self):
        m=np.zeros((50,50),np.uint8); m[25,5:45]=255; m[5:26,25]=255
        segments,junctions=wire_lines(m)
        self.assertEqual(len(segments),3); self.assertTrue(junctions)
        m[5:45,25]=255
        segments,junctions=wire_lines(m)
        self.assertEqual(len(segments),2); self.assertFalse(junctions)

    def test_wire_dot_connects_interior_cross(self):
        m=np.zeros((50,50),np.uint8); m[25,5:45]=255; m[5:45,25]=255; m[23:28,23:28]=255
        segments,junctions=wire_lines(m)
        self.assertEqual(len(segments),4); self.assertTrue(junctions)

    def test_decimal_voltage_reference(self):
        self.assertEqual(references('(3.3V_1.1,R1.2)'),{'3.3V_1.1','R1.2'})

    def test_raw_adapter_splits_multiple_pin_numbers(self):
        raw={'canvas':{'height':100,'width':100},'texts':{'T_1':{'content':'U1'},'T_2':{'content':'5,6'}},'components':{'C_1':{'cpntId':'C_1','category':'Box','location':{'x1':10,'y1':10,'x2':20,'y2':20},'associatedTexts':{'T_1':{'content':'U1','details':'designator'}},'pins':{'P_1':{'pinNumberByOCR':'T_2','tip':[10,15]}}}},'edges':{'E_1':{'connect':{'P_1':{'cpntId':'C_1'}},'keyPointPaths':{'wire':[]}}}}
        result,issues=convert_raw(raw)
        self.assertFalse(validate(result)); self.assertEqual(set(result['pins']['U1']),{'pin_5','pin_6'})
        self.assertEqual(references(result['nets']['net_1']['hyperGraph']),{'U1.5','U1.6'})

    def test_unicode_image_path_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'原理图.png'; im=np.full((5,5,3),128,np.uint8)
            write_image(p,im); self.assertTrue(np.array_equal(im,read_image(p)))

    def test_batch_refuses_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'a').mkdir(); (root/'b').mkdir()
            (root/'a'/'same.png').write_bytes(b''); (root/'b'/'same.png').write_bytes(b'')
            with self.assertRaisesRegex(ValueError,'duplicate'): run_batch(root,root/'out','none','none')


if __name__=='__main__': unittest.main()
