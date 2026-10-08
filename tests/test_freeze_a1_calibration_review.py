import unittest
import tempfile
import json
from pathlib import Path
from scripts.prepare_freeze_a1_calibration_review import review_gate,whole_sheet
from scripts.freeze_a1_identity_gate import require_frozen_eligibility

class CalibrationReviewTests(unittest.TestCase):
    def fixture(self):
        r={'status':'complete','image_sha256':'i','source_cad_sha256':'c','source_pdf_sha256':'p',
            'whole_page_component_instances_reviewed':True,'whole_page_revision_verified':True,
            'point_convention':{'status':'verified','definition':'stroke_outer_tip','coordinate_system':'image_top_left_integer_pixel','evidence_records':['record.json']},
            'hidden_pin_rule':'exclude invisible','DNP_rule':'exclude populated scoring','multi_unit_rule':'display unit separately',
            'visible_key_rule':'pixel key','pin_number_name_human_checked':True,
            'review_passes':[{'status':'complete','reviewer_id':name,'reviewer_kind':'human','evidence_record':name+'.json'} for name in ('A','B')]}
        return r,{k:r[k] for k in ('image_sha256','source_cad_sha256','source_pdf_sha256')}

    def test_missing_point_definition_blocks(self):
        r,e=self.fixture();r['point_convention']['status']='unresolved'
        self.assertIn('point_definition_pending',review_gate(r,e)['reasons'])

    def test_same_reviewer_not_two_independent_reviews(self):
        r,e=self.fixture();r['review_passes'][1]['reviewer_id']='A'
        self.assertIn('independent_second_review_pending',review_gate(r,e)['reasons'])

    def test_agent_review_does_not_become_human_GT(self):
        r,e=self.fixture();r['review_passes'][1]['reviewer_kind']='agent'
        self.assertFalse(review_gate(r,e)['semantic_review_complete'])

    def test_semantic_review_never_grants_inference(self):
        r,e=self.fixture();s=review_gate(r,e)
        self.assertTrue(s['semantic_review_complete']);self.assertFalse(s['blind_ready'])

    def test_source_change_blocks(self):
        r,e=self.fixture();r['image_sha256']='changed'
        self.assertIn('image_sha256_mismatch',review_gate(r,e)['reasons'])

    def test_whole_sheet_keeps_non_ic_and_dnp(self):
        rows=whole_sheet(['root',['symbol',['lib_id','R'],['property','Reference','R1'],['property','Value','10k'],['at','0','0','90'],['dnp','yes']]])
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['dnp'],'yes');self.assertFalse(rows[0]['manual_instance_review_complete'])

    def test_old_eligible_flag_does_not_bypass_semantic_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'manifest.json';p.write_text(json.dumps({'eligible':True,'verified_designs':10}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Pin point definition'):require_frozen_eligibility(p)

    def test_calibration_flag_without_hashes_cannot_authorize_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'manifest.json';p.write_text(json.dumps({'eligible':True,'verified_designs':10,'pin_point_calibration_complete':True,'CoordinateConventionFrozen':True,'frozen_point_convention':['stroke_outer_tip','image_top_left_integer_pixel']}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'dependencies'):require_frozen_eligibility(p)

if __name__=='__main__':unittest.main()
