import unittest

import numpy as np

from ettin_guardrails.evaluation import fpr_fnr, pr_auc, tpr_at_fpr


class EvaluationTests(unittest.TestCase):
    def test_pr_auc_uses_trapezoids(self):
        self.assertAlmostEqual(pr_auc([0, 1], [0.9, 0.1]), 0.25)
        self.assertEqual(pr_auc([0, 1], [[0.9, 0.1], [0.1, 0.9]]), 1.0)


    def test_rejects_invalid_predictions(self):
        for labels, scores in [([], []), ([1], [0.5]), ([0, 2], [0.1, 0.9]),
                               ([0, 1], [0.1]), ([0, 1], [np.nan, 0.9]),
                               ([0, 1], [0.1, np.inf]), ([[0, 1]], [0.1, 0.9])]:
            for metric in [pr_auc, fpr_fnr, tpr_at_fpr]:
                with self.subTest(labels=labels, scores=scores, metric=metric):
                    with self.assertRaises(ValueError):
                        metric(labels, scores)


if __name__ == "__main__":
    unittest.main()
