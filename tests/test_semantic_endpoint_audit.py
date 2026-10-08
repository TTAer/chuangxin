import copy
import unittest
from experiments.semantic_endpoint_audit import audited_semantics,case_fingerprint

class SemanticEndpointAuditTests(unittest.TestCase):
    def fixture(self,source_side='right',identity_reference=True):
        c={'id':'x','domain':'gt10','side':source_side,'pin_state_proxy':'wire',
            'crop_origin_xy':[0,0],'crop_size':[100,100],'points':{'predicted':[30,40]},'GT_pins_by_number':{'3':{'point_xy':[20,40],'identity_reference_comparable':identity_reference}}}
        c['case_evidence_sha256']=case_fingerprint(c)
        e={'id':'x','case_evidence_sha256':c['case_evidence_sha256'],'review_complete':True,'certainty':'clear','conclusion':'tip',
            'candidate_class':'same-physical-pin','observed_side':'right','observed_pin_number':'3','same_pin_evidence_note':'visible 3 and stem align',
            'observed_points':{'body':[20,40],'tip':[30,40]},'reference_seen_before_observations':False}
        r={'schema':'pin-geometry-review-v2','coordinate_system':'image top-left, integer pixel','reviewer_kind':'human-self-declared','reviewer_id':'A','entries':[e]}
        s=copy.deepcopy(r);s['reviewer_id']='B'
        return [c],[r,s]

    def test_no_reviews_no_semantic_p5(self):
        c,_=self.fixture();self.assertIsNone(audited_semantics(c,[])['semantic_statistics']['gt10']['ALL']['semantic_eligible_P5'])

    def test_independent_same_pin_uses_exact_number_and_observed_length(self):
        c,rs=self.fixture();r=audited_semantics(c,rs)['two_reviewer_agreements'][0]
        self.assertTrue(r['semantic_eligible']);self.assertEqual(r['normal_error'],-10);self.assertEqual(r['normalized_normal_error'],-1)

    def test_uncertainty_without_points_is_accepted_but_not_certified(self):
        c,rs=self.fixture()
        for r in rs:r['entries'][0].update(certainty='ambiguous',conclusion='continuous-no-boundary',observed_points={})
        result=audited_semantics(c,rs);self.assertEqual(result['accepted_observation_entries'],2);self.assertEqual(len(result['ambiguous_or_unknown']),1)
        self.assertIsNone(result['semantic_statistics']['gt10']['ALL']['semantic_eligible_P5'])

    def test_same_reviewer_does_not_supply_independence(self):
        c,rs=self.fixture();rs[1]['reviewer_id']='A';result=audited_semantics(c,rs)
        self.assertEqual(result['two_reviewer_agreements'],[])

    def test_wrong_pin_or_absent_identity_never_semantic_eligible(self):
        c,rs=self.fixture()
        for r in rs:r['entries'][0]['candidate_class']='wrong-pin'
        self.assertFalse(audited_semantics(c,rs)['two_reviewer_agreements'][0]['semantic_eligible'])
        for r in rs:r['entries'][0].update(candidate_class='same-physical-pin',observed_pin_number='99')
        self.assertFalse(audited_semantics(c,rs)['two_reviewer_agreements'][0]['semantic_eligible'])

    def test_points_disagree_or_evidence_changed(self):
        c,rs=self.fixture();rs[1]['entries'][0]['observed_points']['tip']=[39,40]
        self.assertEqual(len(audited_semantics(c,rs)['conflicts']),1)
        rs[1]['entries'][0]['case_evidence_sha256']='new';self.assertEqual(audited_semantics(c,rs)['accepted_observation_entries'],1)

    def test_revealing_GT_before_observation_not_eligible(self):
        c,rs=self.fixture();rs[0]['entries'][0]['reference_seen_before_observations']=True
        self.assertFalse(audited_semantics(c,rs)['two_reviewer_agreements'][0]['semantic_eligible'])

    def test_missing_manual_scale_keeps_normalization_null(self):
        c,rs=self.fixture()
        for r in rs:del r['entries'][0]['observed_points']['body']
        r=audited_semantics(c,rs)['two_reviewer_agreements'][0];self.assertTrue(r['semantic_eligible']);self.assertIsNone(r['normalized_normal_error'])

    def test_source_direction_not_reused_as_observed_side(self):
        c,rs=self.fixture()
        for r in rs:r['entries'][0]['observed_side']='unknown'
        self.assertFalse(audited_semantics(c,rs)['two_reviewer_agreements'][0]['semantic_eligible'])

    def test_manual_side_disagreement_diagnosed_without_runtime_change(self):
        c,rs=self.fixture(source_side='bottom');result=audited_semantics(c,rs)
        r=result['two_reviewer_agreements'][0];self.assertTrue(r['side_assignment_disagreement']);self.assertEqual(r['normal_error'],-10);self.assertFalse(result['runtime_changed'])
        self.assertEqual(result['definition_distance_statistics']['gt10']['by_side']['right']['tip']['reviewed_same_pin_cases'],1)

    def test_synthesized_GT_number_is_not_semantic_identity_evidence(self):
        c,rs=self.fixture(identity_reference=False)
        self.assertFalse(audited_semantics(c,rs)['two_reviewer_agreements'][0]['semantic_eligible'])

    def test_editing_reference_without_changing_fingerprint_is_rejected(self):
        c,rs=self.fixture();c[0]['GT_pins_by_number']['3']['point_xy']=[30,40]
        self.assertEqual(audited_semantics(c,rs)['accepted_observation_entries'],0)

if __name__=='__main__':unittest.main()
