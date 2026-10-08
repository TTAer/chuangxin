import unittest
from scripts.audit_rc0_box_errors import taxonomy

class BoxErrorTaxonomyTests(unittest.TestCase):
    def test_large_small_center_and_type(self):
        truth=[{'type':'r','bbox':[0,0,10,10]}]
        for pred, expected in [({'type':'r','bbox':[-5,-5,15,15]},'center_good_box_large'),
                               ({'type':'r','bbox':[3,3,7,7]},'center_good_box_small'),
                               ({'type':'r','bbox':[7,0,17,10]},'center_shift'),
                               ({'type':'c','bbox':[0,0,10,10]},'type_mismatch_candidate')]:
            self.assertEqual(taxonomy([pred],truth)['truth_rows'][0]['category'],expected)

    def test_one_proposal_cannot_explain_two_truth_objects(self):
        p={'type':'r','bbox':[0,0,10,10]}
        result=taxonomy([p],[p,p])
        self.assertEqual(result['counts']['strict_match'],1)
        self.assertEqual(result['counts']['unresolved_competition_or_type_geometry'],1)

    def test_missing_has_no_near_candidate(self):
        result=taxonomy([], [{'type':'r','bbox':[0,0,10,10]}])
        self.assertEqual(result['counts'],{'no_near_candidate':1})
