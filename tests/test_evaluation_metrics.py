import unittest
from scripts.evaluation_metrics import annotation_boxes,bbox_score

class EvaluationMetricsTests(unittest.TestCase):
    def test_mirror_and_supply_are_not_silently_omitted(self):
        boxes,excluded=annotation_boxes({'nmos-mirror_270':[[0,0,10,10]],'vdd_0':[[20,20,30,30]],'net-black_100':[[1,1,2,2]]})
        self.assertEqual([b['type'] for b in boxes],['mosfet','v'])
        self.assertEqual(excluded,{'net-black_100':1})

    def test_prediction_of_absent_shared_class_is_false_positive(self):
        pred={'components':{'R1':{'type':'resistor','bbox':[0,90,10,100]},'C1':{'type':'capacitor','bbox':[20,70,30,80]}}}
        metric=bbox_score(pred,[{'type':'r','bbox':[0,0,10,10]}],100,.5)
        self.assertEqual(metric['tp'],1); self.assertEqual(metric['predictions'],2)

    def test_duplicate_detection_cannot_match_same_gt_twice(self):
        pred={'components':{k:{'type':'resistor','bbox':[0,90,10,100]} for k in ('R1','R2')}}
        metric=bbox_score(pred,[{'type':'r','bbox':[0,0,10,10]}],100,.5)
        self.assertEqual(metric['tp'],1)

    def test_amsnet_uses_row_column_coordinates(self):
        boxes,_=annotation_boxes({'vdd_0':[[28,243,63,309]]})
        self.assertEqual(boxes[0]['bbox'],[243,28,309,63])
