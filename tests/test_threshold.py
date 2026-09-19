import unittest

import numpy as np
import pandas as pd

from train_models import business_metrics, select_approval_threshold


class ApprovalThresholdTests(unittest.TestCase):
    def test_approves_lowest_risk_prefix(self):
        target = pd.Series([0, 0, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.3, 0.8])

        threshold = select_approval_threshold(target, scores, max_bad_rate=0.05)
        metrics = business_metrics(target, scores, threshold)

        self.assertAlmostEqual(threshold, 0.3)
        self.assertAlmostEqual(metrics["approval_rate"], 0.6)
        self.assertAlmostEqual(metrics["bad_rate_among_approved"], 0.0)

    def test_does_not_split_equal_score_group(self):
        target = pd.Series([0, 0, 1])
        scores = np.array([0.1, 0.2, 0.2])

        threshold = select_approval_threshold(target, scores, max_bad_rate=0.1)

        self.assertAlmostEqual(threshold, 0.1)

    def test_rejects_everyone_if_constraint_is_impossible(self):
        target = pd.Series([1, 1])
        scores = np.array([0.1, 0.2])

        threshold = select_approval_threshold(target, scores, max_bad_rate=0.05)
        metrics = business_metrics(target, scores, threshold)

        self.assertEqual(threshold, float("-inf"))
        self.assertAlmostEqual(metrics["approval_rate"], 0.0)
        self.assertAlmostEqual(metrics["bad_rate_among_approved"], 0.0)


if __name__ == "__main__":
    unittest.main()
