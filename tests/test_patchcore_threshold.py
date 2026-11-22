import unittest
import torch
from src.models.patchcore.threshold import effective_folds, held_out_scores


class GroupCalibrationTests(unittest.TestCase):
    def test_same_capture_views_are_held_out_together(self):
        features = torch.tensor([0.0, 0.1, 10.0, 10.1]).reshape(4, 1, 1)
        scores = held_out_scores(features, 10, 1.0, 1, 0, "cpu", log=None,
                                 groups=["a", "a", "b", "b"])
        self.assertTrue(all(score > 9 for score in scores))
        self.assertEqual(effective_folds(["a", "a", "b", "b"], 10), 2)

    def test_single_capture_and_invalid_folds_rejected(self):
        for groups, count in [(["a", "a"], 10), (["a", "b"], 1)]:
            with self.assertRaises(ValueError):
                effective_folds(groups, count)


if __name__ == "__main__":
    unittest.main()
