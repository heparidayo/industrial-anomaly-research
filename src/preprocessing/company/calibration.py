"""카메라별 위치와 구멍 배율을 학습 정상의 기준 원 라벨로 보정한다."""
from pathlib import Path

import numpy as np

from .alignment import measure_hole_radius
from .dataset import camera_of, load_reference_circles, read_image


def calibrate_cameras(source_root, train_stems, image_size, *, train_images=None):
    """train_images를 함께 주면 해당 학습 영상에서 바깥 원/구멍 반지름 비율도 구한다."""
    train_stems = set(train_stems)
    labels = {p.stem for p in Path(source_root).glob("*/labels/train/*.txt")}
    circles = load_reference_circles(source_root, image_size, allowed_stems=train_stems & labels)
    result = {}
    for cam in sorted({camera_of(stem) for stem in train_stems}):
        stems = sorted(stem for stem in circles if camera_of(stem) == cam)
        if not stems:
            raise ValueError(f"{cam}: 기준 원 라벨이 있는 학습 정상 이미지가 없습니다. 분할 seed나 장수를 바꾸세요.")
        values = np.array([circles[stem] for stem in stems])
        params = {"center": values[:, :2].mean(0).tolist(), "radius": float(values[:, 2].mean()),
                  "n_labeled": len(stems), "calibration_stems": stems}
        if train_images is not None:
            ratios = []
            for stem in stems:
                x, y, outer = circles[stem]
                hole = measure_hole_radius(read_image(train_images[stem]), (x, y), outer)
                ratios.append(outer / hole)
            params["hole_to_outer"] = float(np.mean(ratios))
        result[cam] = params
    return result


def calibrate_prepared_cameras(source, info, rows, radius_from):
    train_images = {Path(row["file"]).stem: source / row["prepared_image"] for row in rows
                    if row["split"] == "train" and row["class"] == "good"}
    return calibrate_cameras(info.get("label_root", info["source_root"]), train_images,
                             info.get("reference_image_size", 1216),
                             train_images=train_images if radius_from == "hole" else None)
