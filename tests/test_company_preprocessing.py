import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.preprocessing.company.dataset import camera_of, crop, group_of, roi_boxes, split_by_camera, write_image
from src.preprocessing.company.prepare import prepare_company_dataset
from src.preprocessing.company.prepare_polar import prepare_polar_dataset
from scripts.preprocessing.company import label_center_tool


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

    def test_nested_normal_batches_are_opt_in_and_keep_source_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, dst = Path(temp) / "원본", Path(temp) / "준비"
            labels = raw / "대세 라벨링"
            make_source(labels)
            image = np.zeros((16, 16, 3), np.uint8)
            for cam in (1, 2):
                write_image(raw / "정상" / f"정상{cam}" / f"extra_camera_{cam}.png", image)
            write_image(raw / "정상/정상1/image_20261110_160305.png", image)
            v1 = prepare_company_dataset(raw, dst, n_train=2, n_validation=1)
            direct = split_by_camera(labels, 0, 2, 1)
            with (v1["company_cam12"] / "split.csv").open(encoding="utf-8-sig") as file:
                self.assertEqual({r["file"]: r["split"] for r in csv.DictReader(file)},
                                 {p.name: role for rows in direct.values() for role, _, p in rows})
            v2 = prepare_company_dataset(raw, dst, prefix="v2", n_train=2, n_validation=1,
                                         include_normal_batches=True)
            with (v2["v2_cam12"] / "split.csv").open(encoding="utf-8-sig") as file:
                rows = list(csv.DictReader(file))
            info = json.loads((v2["v2_cam12"] / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 26)
            self.assertEqual(Path(info["label_root"]), labels)
            self.assertTrue(all((Path(info["source_root"]) / r["source"]).is_file() for r in rows))
            extras = [r for r in rows if r["group_id"] == "extra"]
            self.assertEqual(len(extras), 2)
            self.assertEqual(len({r["split"] for r in extras}), 1)
            write_image(raw / "정상/정상2/extra_camera_1.png", image)
            with self.assertRaisesRegex(ValueError, "중복"):
                split_by_camera(labels, 0, 2, 1, normal_batches=raw / "정상")

    def test_label_tool_finds_samples_and_extra_images_in_batch_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, dst = Path(temp) / "정상", Path(temp) / "라벨"
            extra = label_center_tool.EXTRA_STEMS[0]
            other = "PLC-20261012-000001_camera_1"
            for name in (extra, other, "image_20261110_160305", "PLC-20260929-000002_camera_1"):
                write_image(raw / "정상1" / (name + ".png"), np.zeros((16, 16, 3), np.uint8))
            self.assertEqual({p.stem for p in label_center_tool.select_images(raw, 1, 0)}, {extra, other})
            with patch.object(sys, "argv", ["label", "--normal-dir", str(raw), "--out", str(dst), "--per-group", "0"]), \
                 patch.object(label_center_tool, "label_one", return_value="skip"), \
                 patch.object(label_center_tool.cv2, "destroyAllWindows"):
                label_center_tool.main()
            with (dst / "selection.csv").open(encoding="utf-8-sig") as file:
                self.assertEqual([(r["file"], r["reason"]) for r in csv.DictReader(file)],
                                 [(extra + ".png", "radius_suspect")])

    def test_common_train_exclusion_and_counts_across_layouts(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, dst = Path(temp) / "원본", Path(temp) / "준비"
            make_source(raw)
            split = split_by_camera(raw, 0, 2, 1)
            rejected = next(p for role, _, p in split["camera_1"] if role == "train")
            write_image(rejected, np.zeros((16, 16, 3), np.uint8))
            for group in split.values():
                kept = next(p for role, _, p in group if role == "train" and p != rejected)
                write_image(kept, np.full((16, 16, 3), 140, np.uint8))
            calibration = {cam: {"center": [8, 8], "radius": 5, "hole_to_outer": 2} for cam in split}
            with patch("src.preprocessing.company.prepare.calibrate_cameras", return_value=calibration), \
                 patch("src.preprocessing.company.prepare.find_part", return_value=((8, 8), 5, {})), \
                 patch("src.preprocessing.company.prepare.find_spring_start",
                       side_effect=lambda image, *_: (0, float(image[0, 0, 0]))) as measure:
                base = prepare_company_dataset(raw, dst, n_train=2, n_validation=1, exclude_train_rise_below=50)
            self.assertEqual(measure.call_count, 4)
            category = "company_cam12"
            source = base[category]
            info = json.loads((source / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["counts"]["train"], 3)
            self.assertEqual(info["train_filter"]["excluded_from_train"], [{"file": rejected.name, "start_rise": 0}])
            def unwrap(im, *args, **kwargs):
                return im, {"center_x": 8, "center_y": 8, "start_deg": 0, "start_rise": int(im[0, 0, 0])}

            with patch("src.preprocessing.company.prepare_polar.calibrate_prepared_cameras", return_value=calibration), \
                 patch("src.preprocessing.company.prepare_polar.unwrap_image", side_effect=unwrap):
                for suffix, options in [("_strip", {"rows": 1}), ("_stack", {"rows": 4}),
                                        ("_legacy", {"exclude_train_rise_below": 110})]:
                    target = prepare_polar_dataset(dst, [category], suffix=suffix, options=options)[category + suffix]
                    with (target / "split.csv").open(encoding="utf-8-sig") as file:
                        rows = list(csv.DictReader(file))
                    counts = json.loads((target / "dataset_info.json").read_text(encoding="utf-8"))["counts"]
                    self.assertEqual(counts, {k: sum(r["split"] == k for r in rows) for k in counts})
                    self.assertEqual(sum(bool(r["prepared_image"]) for r in rows), len(list(target.rglob("*.png"))))
                    if suffix == "_legacy":
                        self.assertEqual(counts["unused"], info["counts"]["unused"] + 1)
                    else:
                        self.assertEqual((source / "split.csv").read_bytes(), (target / "split.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
