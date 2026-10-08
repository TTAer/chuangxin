import unittest
import numpy as np
from experiments.pin_token_residual import residual_predictions,preservation_metrics

class ResidualTests(unittest.TestCase):
    def test_safe_anchor_cannot_be_overridden(self):
        x=np.zeros((1,55)); x[0,-4:]=[.01,.2,.19,.9]
        self.assertEqual(residual_predictions(x,np.array([1.]),np.array([1]),np.array([1.]),.9).tolist(),[0])

    def test_uncertain_switch_requires_both_gates(self):
        x=np.zeros((3,55)); x[:,-4:]=[.1,.11,.01,.9]
        self.assertEqual(residual_predictions(x,np.array([.99,.8,.99]),np.array([1,1,2]),np.array([.95,.99,.8]),.95).tolist(),[1,0,0])

    def test_easy_and_hard_denominators_are_separate(self):
        m=preservation_metrics(np.array([0,1,1,0]),np.array([0,0,1,8]))
        self.assertEqual(m['EPR'],.5); self.assertEqual(m['HRR'],1); self.assertEqual(m['pin_tokens'],3)
