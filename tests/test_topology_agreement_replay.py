import unittest
from scripts.replay_topology_agreement import protected_merge,split_only_refinement,canonical


def graph(groups):
    return {'nets':{str(i):{'hyperGraph':'('+','.join(pins)+')','edges':{}} for i,pins in enumerate(groups)}}


class AgreementReplaySafetyTests(unittest.TestCase):
    def test_protected_bridge_detects_two_existing_nets(self):
        before=graph([['R1.1'],['R2.1']])
        self.assertTrue(protected_merge(before,graph([['R1.1','R2.1']])))
        self.assertFalse(protected_merge(before,before))

    def test_split_refinement_rejects_unrelated_merges_and_missing_pins(self):
        before=graph([['R1.1','R2.1'],['R3.1']])
        self.assertTrue(split_only_refinement(before,graph([['R1.1'],['R2.1'],['R3.1']])))
        self.assertFalse(split_only_refinement(before,graph([['R1.1'],['R2.1','R3.1']])))
        self.assertFalse(split_only_refinement(before,graph([['R1.1'],['R3.1']])))

    def test_canonical_comparison_ignores_net_id_and_pin_order(self):
        a=graph([['R1.1','R2.1']]);b=graph([['R2.1','R1.1']]);b['nets']['renamed']=b['nets'].pop('0')
        self.assertEqual(canonical(a),canonical(b))
