import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.preprocessing.company.dataset import camera_of, crop, group_of, roi_boxes, split_by_camera, write_image
from src.preprocessing.company.prepare import prepare_company_dataset


def make_source(root, normal_groups=6):
    for camera in (1, 2):
        for index in range(normal_groups):
            group = f"part-{camera}-{index}"
            write_image(root / "정상" / f"{group}_camera_{camera}.png",
                        np.full((16, 16, 3), 100, np.uint8))
            write_image(root / "over/images/train" / f"{group}_camera_{3-camera}.png",
                        np.full((16, 16, 3), 200, np.uint8))


class CompanyPreprocessingTests(unittest.TestCase):
    def test_split_keeps_shot_groups_and_excludes_train_defect_views(self):
        with tempfile.TemporaryDirectory() as temp:
            make_source(Path(temp))
            split = split_by_camera(temp, 0, 2, 1)
            self.assertEqual(split, split_by_camera(temp, 0, 2, 1))
            roles = {}
            for rows in split.values():
                for role, cls, path in rows:
                    roles.setdefault(group_of(path), set()).add("train" if role == "unused" else role)
                    if role == "train":
                        self.assertEqual(cls, "good")
            self.assertTrue(all(len(value) == 1 for value in roles.values()))
            self.assertEqual(sum(role == "unused" for rows in split.values() for role, _, _ in rows), 4)
            self.assertTrue(any(role == "validation" for rows in split.values() for role, _, _ in rows))

    def test_roi_ignores_test_annotations(self):
        train = {"a_camera_1", "b_camera_2"}
        circles = {"a_camera_1": (10, 11, 4), "b_camera_2": (20, 21, 4),
                   "test_camera_1": (1000, 1000, 999)}
        boxes = roi_boxes(circles, 3, calibration_stems=train)
        circles["test_camera_1"] = (-1000, -1000, 4)
        self.assertEqual(boxes, roi_boxes(circles, 3, calibration_stems=train))
        self.assertEqual(boxes["camera_1"], [7, 8, 13, 14])

    def test_export_preserves_originals_and_camera_split(self):
        with tempfile.TemporaryDirectory() as temp:
            src, dst = Path(temp) / "회사 원본", Path(temp) / "prepared"
            make_source(src)
            original = {p.relative_to(src): p.read_bytes() for p in src.rglob("*.png")}
            categories = prepare_company_dataset(src, dst, n_train=2, n_validation=1)
            combined = categories["company_cam12"]
            with (combined / "split.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 24)
            self.assertEqual(len(list(combined.rglob("*.png"))), 20)
            self.assertTrue(all(not row["prepared_image"] for row in rows if row["split"] == "unused"))
            self.assertEqual(json.loads((combined / "dataset_info.json").read_text(encoding="utf-8"))["protocol_status"], "proposed")
            combined_roles = {row["file"]: row["split"] for row in rows}
            for category in ("company_cam1", "company_cam2"):
                with (categories[category] / "split.csv").open(encoding="utf-8-sig") as handle:
                    self.assertTrue(all(combined_roles[row["file"]] == row["split"] for row in csv.DictReader(handle)))
            self.assertEqual(original, {p.relative_to(src): p.read_bytes() for p in src.rglob("*.png")})
            with self.assertRaises(FileExistsError):
                prepare_company_dataset(src, dst, n_train=2, n_validation=1)

    def test_invalid_camera_quota_and_crop_fail(self):
        with self.assertRaises(ValueError):
            camera_of("part_camera_3.png")
        with self.assertRaises(ValueError):
            crop(np.zeros((8, 8, 3), np.uint8), [-1, 0, 4, 4])
        with tempfile.TemporaryDirectory() as temp:
            make_source(Path(temp), normal_groups=3)
            with self.assertRaises(ValueError):
                split_by_camera(temp, 0, 3, 1)


if __name__ == "__main__":
    unittest.main()
