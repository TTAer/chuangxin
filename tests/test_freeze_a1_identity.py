import unittest
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from freeze_a1_identity_gate import assess, require_frozen_eligibility
from freeze_a1_identity_evaluator import evaluate
from audit_freeze_a1_identity import match_token

class FreezeA1IdentityTests(unittest.TestCase):
    def test_no_manifest_forbids_first_inference(self):
        with self.assertRaises(ValueError):
            require_frozen_eligibility(Path(__file__).parent/'nonexistent-freeze-a1-manifest.json')

    def test_pending_review_never_allows_inference(self):
        records=[{'id':i,'repository':str(i),'image_sha256':str(i),'observed_review':{'status':'agent_pixel_matched'}}for i in range(12)]
        assert not assess({'records':records},{'families':[]})['eligible']

    def test_repositories_cannot_substitute_independent_designs(self):
        records=[{'id':i,'repository':str(i),'image_sha256':str(i),'observed_review':{'status':'agent_pixel_matched'}}for i in range(12)]
        approvals={'families':[{'repository':str(i),'license_scope_approved':True,'cad_pdf_page_revision_verified':True,
            'historical_design_family_exclusion_verified':True,'historical_corpus_manifest':'history.json',
            'independent_review_complete':True,'verified_design_id':'one-family','license_scope_review_record':'license-review.json','page_revision_review_record':'page-review.json','family_exclusion_review_record':'family-review.json','independent_annotation_review_record':'annotation-review.json'}for i in range(12)]}
        result=assess({'records':records},approvals)
        assert result['verified_designs']==1 and not result['eligible']

    def test_exclusion_claim_requires_corpus_evidence(self):
        records=[{'id':0,'repository':'a','image_sha256':'a','observed_review':{'status':'agent_pixel_matched'}}]
        approval={'repository':'a','license_scope_approved':True,'cad_pdf_page_revision_verified':True,
            'historical_design_family_exclusion_verified':True,'independent_review_complete':True,'verified_design_id':'a'}
        assert assess({'records':records},{'families':[approval]})['eligible_ic_instances']==0

    def test_partial_evaluator_does_not_report_full_page_or_net(self):
        gt=[{'body_xyxy':[0,0,20,20],'key':'U2B','name':'IC','pins':[]}]
        pred=[{'body_xyxy':[0,0,20,20],'key':'U2','name':'IC','pins':[]},
              {'body_xyxy':[100,100,120,120],'key':'U9','name':'Other','pins':[]}]
        scores=evaluate(gt,pred)
        assert scores['key_correct']==0
        assert scores['full_page_component_recall'] is None and scores['net_score'] is None
        assert scores['out_of_scope_predictions_unscored']==1

    def test_native_token_cannot_be_assigned_twice(self):
        used=set();words=[{'text':'1','bbox_xyxy':[0,0,4,8]}]
        assert match_token(words,'1',[2,4],used,10)
        assert match_token(words,'1',[2,4],used,10) is None
