import unittest
import numpy as np
from pcbparse.field_perception import FieldPerception, field_input, tile_starts
from pcbparse.topology_perception import TopologyPerception


class ConstantSession:
    def run(self, _, inputs):
        n, _, h, w = inputs['image'].shape
        return [np.full((n, 5, h, w), 20, np.float32)]


class FieldPerceptionTests(unittest.TestCase):
    def test_tile_coverage_at_edges_and_small_images(self):
        for length in (1, 80, 192, 193, 407):
            covered = np.zeros(length, bool)
            for start in tile_starts(length,192,96): covered[start:start+192] = True
            self.assertTrue(covered.all())

    def test_white_pixels_cannot_become_wires_even_with_high_probability(self):
        model = FieldPerception.__new__(FieldPerception)
        model.size=192; model.threshold=.7; model.path='test'; model.session=ConstantSession()
        image=np.full((203,311,3),255,np.uint8); image[100,20:290]=0
        mask,fields,report=model(image)
        self.assertEqual(int((mask>0).sum()),270)
        self.assertEqual(fields.shape,(5,203,311))
        self.assertTrue(np.isfinite(fields).all())
        self.assertTrue(report['actual_ink_gate'])

    def test_no_rgb_channel_and_external_mask_cannot_override_topology(self):
        image=np.full((80,80,3),255,np.uint8); image[40,10:70]=0; image[10:70,40]=0
        inputs=field_input(image)
        self.assertEqual(inputs.shape,(3,80,80))
        _,junctions,_,_=TopologyPerception().perceive(image,[],[],wire_mask=np.full((80,80),255,np.uint8))
        self.assertFalse(junctions)  # Interior crossover is not an electrical junction.
        partial=np.full((80,80),255,np.uint8); partial[41:,40]=0
        _,junctions,_,_=TopologyPerception().perceive(image,[],[],wire_mask=partial)
        self.assertFalse(junctions)  # Removing an arm cannot create T evidence.
        with self.assertRaises(ValueError):
            TopologyPerception().perceive(image,[],[],wire_mask=np.zeros((10,10),np.uint8))
