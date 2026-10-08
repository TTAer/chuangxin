import tempfile
import unittest
from pathlib import Path

from scripts.prepare_review_handoff import check_lock, revision_disposition, sha


class ReviewHandoffTests(unittest.TestCase):
    def test_review_lock_rejects_changed_pixels_and_missing_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p = root / 'pixels.png'
            p.write_bytes(b'original reviewed pixels')
            lock = {'sha256': {'pixels.png': sha(p)}}
            check_lock(root, lock)
            p.write_bytes(b'changed pixels')
            with self.assertRaises(ValueError):
                check_lock(root, lock)
            p.unlink()
            with self.assertRaises(ValueError):
                check_lock(root, lock)

    def test_revision_risk_boundaries_and_bluesmirf(self):
        self.assertEqual(revision_disposition(98, 100), 'human_priority')
        self.assertEqual(revision_disposition(97, 100), 'revision_review_required')
        self.assertEqual(revision_disposition(90, 100), 'revision_review_required')
        self.assertEqual(revision_disposition(89, 100), 'paused_revision_risk')
        self.assertEqual(revision_disposition(50, 89), 'paused_revision_risk')
        self.assertEqual(revision_disposition(0, 0), 'invalid_evidence')
