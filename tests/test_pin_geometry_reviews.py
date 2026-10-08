import copy
import unittest
from scripts.audit_pin_geometry_reviews import assess_reviews

class ReviewImportTests(unittest.TestCase):
    def fixture(self):
        c={'id':'x','domain':'gt10','source_image_sha256':'s','crop_sha256':'c','crop_origin_xy':[0,0],'crop_size':[20,20],'points':{'dataset_GT':[10,10]}}
        r={'schema':'pin-point-definition-review-v1','coordinate_system':'image top-left, integer pixel','reviewer_id':'A','reviewer_kind':'human-self-declared',
            'entries':[{'id':'x','source_image_sha256':'s','crop_sha256':'c','review_complete':True,'coordinate_convention':'stroke_outer_tip','manual_points':{'stroke_outer_tip':[10,10]}}]}
        return [c],r

    def test_changed_pixels_rejected(self):
        c,r=self.fixture();r['entries'][0]['source_image_sha256']='new'
        self.assertEqual(assess_reviews(c,[r])['accepted_review_entries'],0)

    def test_repeated_reviewer_cannot_supply_second_check(self):
        c,r=self.fixture();self.assertEqual(assess_reviews(c,[r,r])['two_reviewer_local_agreements'],[])

    def test_two_agreements_do_not_create_independent_design(self):
        c,r=self.fixture();s=copy.deepcopy(r);s['reviewer_id']='B';result=assess_reviews(c,[r,s])
        self.assertEqual(len(result['two_reviewer_local_agreements']),1);self.assertEqual(result['verified_designs'],0);self.assertFalse(result['first_inference_authorized'])

    def test_outside_crop_and_disagreement_rejected(self):
        c,r=self.fixture();s=copy.deepcopy(r);s['reviewer_id']='B';s['entries'][0]['manual_points']['stroke_outer_tip']=[18,10]
        self.assertEqual(len(assess_reviews(c,[r,s])['review_conflicts']),1)
        s['entries'][0]['manual_points']['stroke_outer_tip']=[20,10]
        self.assertEqual(assess_reviews(c,[s])['accepted_review_entries'],0)

if __name__=='__main__':unittest.main()
