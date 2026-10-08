import unittest
from scripts.audit_local_pin_ocr import local_exact, restore_point


class LocalPinOCRTests(unittest.TestCase):
    def test_equal_number_elsewhere_is_not_recovery(self):
        tokens = [{'text': '2', 'bbox': [80, 80, 90, 90]}]
        self.assertFalse(local_exact(tokens, '2', [10, 10, 20, 20]))
        self.assertTrue(local_exact(tokens, '2', [80, 80, 90, 90]))

    def test_rotation_restores_original_pixel(self):
        # Original x=3,y=5 in width=20,height=10; CW becomes x=4,y=3.
        self.assertEqual(restore_point(8, 6, 'cw_2x', 20, 10, 2, (100, 50)), [103, 55])
        # CCW becomes x=5,y=16.
        self.assertEqual(restore_point(10, 32, 'ccw_2x', 20, 10, 2, (100, 50)), [103, 55])


if __name__ == '__main__':
    unittest.main()
