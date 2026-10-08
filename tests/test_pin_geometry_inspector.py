import unittest
import numpy as np
from experiments.pin_geometry_inspector import error_vector,project,skeleton,skeleton_endpoints
from scripts.build_pin_geometry_inspector import match,xy

class PinGeometryInspectorTests(unittest.TestCase):
    def test_bottom_left_converted_once_and_bottom_error_inward(self):
        pred=xy({'x':1703.4322916667,'y':630.7291666667},1000)
        gt=xy({'x':1703,'y':642},1000)
        e=error_vector(pred,gt,'bottom')
        self.assertAlmostEqual(e['normal'],-11.2708333333)
        self.assertAlmostEqual(abs(e['tangent']),.4322916667)

    def test_normal_tangent_axes(self):
        for side in ('left','right','top','bottom'):
            e=error_vector([10,10],[13,14],side)
            self.assertEqual(e['distance'],5)
            self.assertAlmostEqual(e['normal']**2+e['tangent']**2,25)

    def test_recall_matching_not_double_counted(self):
        self.assertEqual(len(match([[0,0],[0,0]],[[0,0]])),1)
        self.assertEqual(len(match([[0,0]],[[5.01,0]])),0)

    def test_projection_clipped_to_segment(self):
        self.assertEqual(project([12,1],[0,0],[10,0])[0],[10,0])
        self.assertEqual(project([2,3],[1,1],[1,1])[0],[1,1])

    def test_skeleton_tips_and_border_exclusion(self):
        mask=np.zeros((20,20),bool);mask[10,4:16]=True
        sk=skeleton(mask)
        self.assertEqual(skeleton_endpoints(sk),[[4,10],[15,10]])
        mask[10,:]=True
        self.assertEqual(skeleton_endpoints(skeleton(mask)),[])

if __name__=='__main__':unittest.main()
