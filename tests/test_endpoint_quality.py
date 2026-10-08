import unittest
import numpy as np
from experiments.endpoint_quality import pixel_evidence,bucket
from scripts.audit_unresolved_endpoint_quality import table,component_pairs


class EndpointQualityTests(unittest.TestCase):
    def test_wire_contact_cannot_authorize_or_upgrade_no_stroke(self):
        image=np.full((80,80),255,np.uint8)
        p={'x':20,'y':40}; wire=[p,{'x':5,'y':40}]
        r=pixel_evidence(image,p,[20,20,60,60],[wire])
        self.assertEqual(r['grade'],'G0')
        self.assertTrue(r['wire_segment_contact'])
        self.assertFalse(r['runtime_commit_authorized'])

    def test_boundary_and_normal_required_before_g3(self):
        image=np.full((80,80),255,np.uint8); image[40,16:31]=0
        p={'x':20,'y':40}; box=[20,20,60,60]
        self.assertEqual(pixel_evidence(image,p,box,[])['grade'],'G1')
        image[31:50,30]=0
        self.assertEqual(pixel_evidence(image,p,box,[])['grade'],'G2')
        r=pixel_evidence(image,p,box,[[p,{'x':5,'y':40}]])
        self.assertEqual(r['grade'],'G3')
        self.assertEqual(pixel_evidence(image,p,box,[[p,{'x':5,'y':40}]],False)['grade'],'G2')

    def test_unknowns_and_duplicates_not_hidden_in_ep5(self):
        rows=[{'grade':'G2','nearest_GT_distance':1,'one_to_one_EP5':True},
              {'grade':'G2','nearest_GT_distance':1,'one_to_one_EP5':False},
              {'grade':'G2','nearest_GT_distance':None,'one_to_one_EP5':False}]
        stats=table(rows)['G2']
        self.assertEqual(stats['nearest_EP5_known'],1)
        self.assertAlmostEqual(stats['nearest_EP5_all_candidates'],2/3)
        self.assertAlmostEqual(stats['one_to_one_EP5_all_candidates'],1/3)

    def test_component_mapping_one_to_one(self):
        obs={'components':{'U1':{'type':'ic','bbox':[0,0,10,10]},'U2':{'type':'ic','bbox':[0,0,10,10]}}}
        gt={'components':{'X':{'type':'ic','bbox':[0,0,10,10]}}}
        self.assertEqual(len(component_pairs(obs,gt)),1)

    def test_threshold_boundaries(self):
        self.assertEqual([bucket(v) for v in (3,5,8,12,12.01)],['<=3','<=5','<=8','<=12','>12'])

if __name__=='__main__': unittest.main()
