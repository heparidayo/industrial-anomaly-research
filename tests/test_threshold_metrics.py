import math
import unittest
from src.evaluation.threshold_metrics import threshold_metrics


class FixedThresholdTests(unittest.TestCase):
    def test_boundary_uses_strict_greater_than(self):
        rows = [
            {"image_id": "a", "label": 0, "anomaly_score": 0.5, "defect_type": "good"},
            {"image_id": "b", "label": 0, "anomaly_score": 0.8, "defect_type": "good"},
            {"image_id": "c", "label": 1, "anomaly_score": 0.5, "defect_type": "defect"},
            {"image_id": "d", "label": 1, "anomaly_score": 0.9, "defect_type": "defect"},
        ]
        result = threshold_metrics(rows, 0.5)
        self.assertEqual((result["false_alarm"], result["missed"]), (1, 1))
        self.assertEqual((result["fpr"], result["fnr"], result["f1"]), (0.5, 0.5, 0.5))
        self.assertNotIn("reference", result)

    def test_normal_only_does_not_claim_fnr_or_auroc(self):
        result = threshold_metrics([{"label": 0, "anomaly_score": 1.0}], 2.0)
        self.assertEqual(result["fpr"], 0)
        self.assertIsNone(result["fnr"])
        self.assertIsNone(result["f1"])
        self.assertNotIn("auroc", result)

    def test_invalid_inputs_are_rejected(self):
        for rows, threshold in [
            ([], 1), ([{"label": 2, "anomaly_score": 0.1}], 1),
            ([{"label": 0, "anomaly_score": math.nan}], 1),
            ([{"label": 0, "anomaly_score": 0.1}], math.inf),
            ([{"image_id": "a", "label": 0, "anomaly_score": 0.1}] * 2, 1),
        ]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                threshold_metrics(rows, threshold)


if __name__ == "__main__":
    unittest.main()
