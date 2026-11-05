import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.patchcore.compute_patchcore_threshold import main as compute_threshold
from scripts.evaluation.evaluate_threshold import main as evaluate_threshold
from scripts.evaluation.evaluate_predictions import main as evaluate_predictions
from scripts.patchcore.run_patchcore_experiment import artifact_paths, check_metadata, run_one
from src.models.patchcore.artifacts import file_sha256


class ExperimentIntegrationTests(unittest.TestCase):
    def test_artifact_names_match_effective_folds_and_keep_original_default(self):
        default = artifact_paths("bottle", 0.01, folds=5)
        alternate = artifact_paths("bottle", 0.01, seed=1, folds=5)
        self.assertEqual(default["prediction"].name, "bottle_original.csv")
        self.assertTrue(default["threshold"].name.endswith("_k5.json"))
        self.assertNotEqual(default["prediction"], alternate["prediction"])
        other_folds = artifact_paths("bottle", 0.01, folds=10)
        self.assertEqual(default["prediction"], other_folds["prediction"])
        self.assertNotEqual(default["threshold_metrics"], other_folds["threshold_metrics"])

    def test_runner_uses_selected_calibration_in_common_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(projection_dim=2, seed=0, split="test", folds=2)
            data_root = root / "data"
            inputs = {"folds": 2, "train_fingerprint": "train", "eval_fingerprint": "test"}

            def common_cli(script, *values):
                commands = {
                    "scripts/evaluation/evaluate_predictions.py": evaluate_predictions,
                    "scripts/evaluation/evaluate_threshold.py": evaluate_threshold,
                }
                self.assertIn(script, commands)
                with patch.object(sys, "argv", [script, *map(str, values)]):
                    commands[script]()

            with patch("scripts.patchcore.run_patchcore_experiment.ROOT", root), \
                 patch("scripts.patchcore.run_patchcore_experiment.run", side_effect=common_cli):
                paths = artifact_paths("synthetic", 1, 2, 0, 2)
                paths["raw"].parent.mkdir(parents=True)
                paths["raw"].write_bytes(b"raw")
                paths["raw"].with_suffix(".pt.meta.json").write_text(json.dumps({
                    "category": "synthetic", "data_root": str(data_root),
                    "input_fingerprint": "train", "image_names": ["a", "b", "c", "d"],
                }), encoding="utf-8")
                config = {"category": "synthetic", "percentage": 1, "projection_dim": 2,
                          "seed": 0, "raw_memory_sha256": file_sha256(paths["raw"])}
                paths["coreset"].write_bytes(b"bank")
                paths["coreset"].with_suffix(".pt.meta.json").write_text(
                    json.dumps(config), encoding="utf-8")
                paths["prediction"].parent.mkdir(parents=True)
                with paths["prediction"].open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=[
                        "image_id", "label", "defect_type", "anomaly_score", "inference_ms"])
                    writer.writeheader()
                    writer.writerows([
                        {"image_id": "normal", "label": 0, "defect_type": "good",
                         "anomaly_score": 0.5, "inference_ms": 1},
                        {"image_id": "defect", "label": 1, "defect_type": "defect",
                         "anomaly_score": 2, "inference_ms": 1},
                    ])
                paths["prediction"].with_suffix(".csv.meta.json").write_text(json.dumps({
                    "category": "synthetic", "split": "test", "input_fingerprint": "test",
                    "memory_sha256": file_sha256(paths["coreset"]),
                    "raw_memory_sha256": config["raw_memory_sha256"],
                    "coreset": {key: config[key] for key in ("percentage", "projection_dim", "seed")},
                }), encoding="utf-8")
                summaries = []
                for folds, threshold in ((2, 1.0), (3, 0.25)):
                    args.folds = inputs["folds"] = folds
                    selected = artifact_paths("synthetic", 1, 2, 0, folds)
                    selected["threshold"].write_text(
                        json.dumps({**config, "folds": folds, "threshold": threshold}), encoding="utf-8")
                    summaries.append(run_one("synthetic", 1, args, data_root, inputs))
                self.assertEqual([row["threshold"] for row in summaries], [1.0, 0.25])
                self.assertEqual([row["fpr"] for row in summaries], [0.0, 1.0])

    def test_stale_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result.csv"
            output.with_suffix(".csv.meta.json").write_text(
                json.dumps({"input_fingerprint": "old"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                check_metadata(output, {"input_fingerprint": "new"})

    def test_training_only_threshold_and_generic_evaluation_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, threshold, predictions, output = (root / name for name in
                ("raw.pt", "threshold.json", "predictions.csv", "metrics.json"))
            torch.save({
                "category": "synthetic", "stage": "raw_candidates",
                "embeddings": torch.tensor([[0., 0.], [1., 1.], [4., 4.], [5., 5.]]),
                "patches_per_image": 1, "image_names": ["a.png", "b.png", "c.png", "d.png"],
                "image_group_ids": ["a", "b", "c", "d"],
            }, raw)
            with patch.object(sys, "argv", ["compute", "--category", "synthetic",
                    "--input", str(raw), "--percentage", "1", "--output", str(threshold)]):
                compute_threshold()
            frozen = threshold.read_bytes()
            calibrated = json.loads(frozen)
            self.assertEqual(calibrated["folds"], 4)
            self.assertEqual(calibrated["calibration_split"], "train_normal_group_oof")
            for version in (0, 1):
                with predictions.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=[
                        "image_id", "label", "defect_type", "anomaly_score", "inference_ms"])
                    writer.writeheader()
                    writer.writerows([
                        {"image_id": "normal", "label": 0, "defect_type": "good",
                         "anomaly_score": 0.1, "inference_ms": 1},
                        {"image_id": "defect", "label": 1, "defect_type": "defect",
                         "anomaly_score": 100 if version else 0.2, "inference_ms": 1},
                    ])
                target = output.with_name(f"metrics-{version}.json")
                with patch.object(sys, "argv", ["evaluate", "--csv", str(predictions),
                        "--threshold", str(threshold), "--output", str(target)]):
                    evaluate_threshold()
                result = json.loads(target.read_text(encoding="utf-8"))
                self.assertNotIn("reference", result["groups"][0])
                self.assertEqual(threshold.read_bytes(), frozen)


if __name__ == "__main__":
    unittest.main()
