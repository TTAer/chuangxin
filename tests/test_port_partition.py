import unittest
import numpy as np
from experiments.port_partition import (RELATIONS, compatible_partitions,
                                        port_membership, transitivity_violations)


class PortPartitionTests(unittest.TestCase):
    def test_all_decoded_partitions_are_legal_and_impossible_triangle_rejected(self):
        self.assertEqual(RELATIONS.shape, (15, 6))
        self.assertTrue(all(transitivity_violations(r)==0 for r in RELATIONS))
        target=np.array([1,1,0,0,0,0]); valid=np.array([1,1,0,1,0,0])
        self.assertFalse(compatible_partitions(target,valid).any())

    def test_crossing_does_not_merge_distinct_wire_groups(self):
        segments=[('h',[-20,0],[20,0]),('v',[0,-20],[0,20])]
        target, valid, owners=port_membership(segments,[0,0])
        self.assertEqual(owners,['h','h','v','v'])
        np.testing.assert_array_equal(target,[1,0,0,0,0,1])
        self.assertTrue(valid.all())
        segments.append(('other',[-20,0],[20,0]))
        _, valid, owners=port_membership(segments,[0,0])
        self.assertIsNone(owners[0]);self.assertIsNone(owners[1])
        self.assertEqual(int(valid.sum()),1)
