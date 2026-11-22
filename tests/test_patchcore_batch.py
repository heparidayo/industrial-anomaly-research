from pathlib import Path
import tempfile
import unittest
import torch

from src.models.patchcore.batch import predict_samples, write_predictions


class FakeModel:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def predict(self, path):
        self.calls.append(Path(path).name)
        return {"image_score": 0.25}


class BatchTests(unittest.TestCase):
    def test_batch_warmup_score_and_optional_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.png"
            path.write_bytes(b"placeholder")
            sample = {"path": path, "image_id": "image.png", "label": 0,
                      "defect_type": "good", "variation": "original", "level": 0,
                      "group_id": "capture"}
            model = FakeModel()
            records = predict_samples(model, [sample])
            self.assertEqual(len(model.calls), 2)
            self.assertEqual(records[0]["anomaly_score"], 0.25)
            self.assertEqual(records[0]["group_id"], "capture")
            self.assertGreaterEqual(records[0]["inference_ms"], 0)
            out = Path(temp) / "result.csv"
            write_predictions(records, out, {"model": "fake"})
            self.assertTrue(out.with_suffix(".csv.meta.json").is_file())
            with self.assertRaises(FileExistsError):
                write_predictions(records, out, {})

    def test_duplicate_ids_rejected_before_inference(self):
        model = FakeModel()
        with self.assertRaises(ValueError):
            predict_samples(model, [{"image_id": "a"}] * 2)
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
