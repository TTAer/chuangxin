import unittest
import numpy as np
from experiments.pin_token_v1 import K,candidates,text_features

class PinTokenV1Tests(unittest.TestCase):
    def test_short_candidate_list_masks_padding(self):
        order,features,valid=candidates([0,0,2,2],[0,0,20,20],[[1,1],[10,10]])
        self.assertEqual(order.tolist(),[0,1]); self.assertEqual(valid.sum(),2)
        self.assertEqual(features.shape,(K,12)); self.assertTrue(np.isfinite(features).all())

    def test_coordinates_have_same_translation_invariant_features(self):
        _,a,_=candidates([0,0,2,2],[0,0,20,20],[[1,1]])
        _,b,_=candidates([5,7,7,9],[5,7,25,27],[[6,8]])
        np.testing.assert_allclose(a,b)

    def test_features_do_not_contain_identity_labels(self):
        f=text_features('U12'); self.assertEqual(f.shape,(12,)); self.assertEqual(f[5],1)

    def test_empty_candidate_set_leaves_only_null(self):
        order,_,valid=candidates([0,0,2,2],[0,0,20,20],np.empty((0,2)))
        self.assertEqual(len(order),0); self.assertFalse(valid.any())
