import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from src.preprocessing.company.dataset import write_image
from src.preprocessing.company.prepare import prepare_company_dataset
from src.preprocessing.company.part_rotation import rotate_part
from scripts.preprocessing.company.preprocess_part_rotation import main as rotate_cli
from scripts.preprocessing.validate_dataset import main as validate_cli
from scripts.patchcore.run_patchcore_variation import main as variation_cli
from scripts.evaluation.evaluate_threshold import main as threshold_cli


class FixturePredictor:
    device = torch.device("cpu")
    category = "company_cam1_roi"

    def __init__(self, memory):
        pass

    def predict(self, path):
        return {"image_score": 0.25}


class CompanyPipelineTests(unittest.TestCase):
    def test_rotation_keeps_background_and_zero_angle(self):
        image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        self.assertTrue(np.array_equal(rotate_part(image, (8, 8), 3, 0), image))
        rotated = rotate_part(image, (8, 8), 3, 90)
        self.assertTrue(np.array_equal(rotated[0, 0], image[0, 0]))
        self.assertTrue(np.any(rotated[6:10, 6:10] != image[6:10, 6:10]))

    def test_company_roi_variation_csv_and_fixed_evaluation_connect(self):
        with tempfile.TemporaryDirectory() as temp:
            root, src = Path(temp), Path(temp) / "원본"
            for cam in (1, 2):
                for index in range(6):
                    group = f"part-{cam}-{index}"
                    for view in (1, 2):
                        name = f"{group}_camera_{view}"
                        directory = src / "정상" if view == cam else src / "over/images/train"
                        write_image(directory / f"{name}.png", np.full((16, 16, 3), 100, np.uint8))
                        label = src / "over/labels/train" / f"{name}.txt"
                        label.parent.mkdir(parents=True, exist_ok=True)
                        label.write_text("1 0.3 0.5 0.5 0.3 0.7 0.5 0.5 0.7\n", encoding="utf-8")
            prepared = root / "prepared"
            prepare_company_dataset(src, prepared, n_train=2, n_validation=1,
                                    roi=True, roi_half=6, reference_image_size=16)
            config = {
                "experiment": {"name": "rotation_smoke"},
                "dataset": {"root": str(prepared), "category": "company_cam1_roi"},
                "labels": {"root": str(src)}, "output": {"root": str(root / "variations")},
                "part_rotation": {"margin": 0}, "variations": {"rotation_part": [-5, 5]},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch.object(sys, "argv", ["rotate", "--config", str(config_path)]):
                rotate_cli()
            manifest = root / "variations/company_cam1_roi/rotation_smoke/manifest.csv"
            with patch.object(sys, "argv", ["validate", str(manifest)]):
                validate_cli()
            with manifest.open(encoding="utf-8-sig", newline="") as handle:
                entries = list(csv.DictReader(handle))
            self.assertEqual(len(entries), 9)
            memory = root / "memory.pt"
            torch.save({"category": "company_cam1_roi", "embeddings": torch.zeros(1, 1),
                        "percentage": 1, "projection_dim": 1, "seed": 0}, memory)
            predictions = root / "predictions.csv"
            with patch("scripts.patchcore.run_patchcore_variation.PatchCorePredictor", FixturePredictor), \
                 patch.object(sys, "argv", ["predict", "--manifest", str(manifest),
                     "--memory", str(memory), "--output", str(predictions), "--image-root", str(root)]):
                variation_cli()
            with predictions.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(entries))
            self.assertTrue(all(row["camera"] == "camera_1" for row in rows))
            threshold = root / "threshold.json"
            threshold.write_text(json.dumps({"threshold": 0.3, "category": "company_cam1_roi"}), encoding="utf-8")
            result = root / "result.json"
            with patch.object(sys, "argv", ["evaluate", "--csv", str(predictions),
                    "--threshold", str(threshold), "--output", str(result)]):
                threshold_cli()
            groups = json.loads(result.read_text(encoding="utf-8"))["groups"]
            self.assertEqual(len(groups), 3)
            self.assertTrue(all(group["fpr"] == 0 and group["fnr"] is None for group in groups))


if __name__ == "__main__":
    unittest.main()
