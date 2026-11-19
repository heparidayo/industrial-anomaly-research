"""원본을 보존하며 공통 촬영번호 분할을 MVTec 모양의 입력으로 내보낸다."""

import csv
import json
import math
from pathlib import Path
import re
import tempfile

from .alignment import find_part, find_spring_start
from .calibration import calibrate_cameras
from .dataset import (
    NORMAL_FOLDER, camera_of, crop, group_of, load_reference_circles, read_image,
    resolve_source, roi_boxes, split_by_camera, split_counts, write_image,
)


def filter_train_by_rise(split, label_root, image_size, threshold):
    """학습 후보의 시작점 밝기 상승폭을 한 번 검사해 모든 입력 형태가 같은 분할을 쓰게 한다."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("학습 제외 기준은 0 이상의 유한한 값이어야 합니다.")
    train = {p.stem: p for rows in split.values() for role, cls, p in rows
             if role == "train" and cls == "good"}
    calibration = calibrate_cameras(label_root, train, image_size, train_images=train)
    excluded = {cam: [] for cam in split}
    for cam, rows in split.items():
        params = calibration[cam]
        for index, (role, cls, path) in enumerate(rows):
            if role != "train":
                continue
            image = read_image(path)
            center, radius, _ = find_part(
                image, params["center"], params["radius"], center_search="global",
                radius_from="hole", hole_to_outer=params["hole_to_outer"])
            _, rise = find_spring_start(image, center, radius)
            if rise < threshold:
                rows[index] = ("unused", cls, path)
                excluded[cam].append({"file": path.name, "start_rise": float(rise)})
        if not any(role == "train" for role, _, _ in rows):
            raise ValueError(f"{cam}: 밝기 상승폭 기준을 적용한 뒤 남은 학습 정상이 없습니다.")
    return calibration, excluded


def prepare_company_dataset(src, dst, *, prefix="company", seed=0,
                            n_train=10, n_validation=3, roi=False,
                            roi_half=420, reference_image_size=1216,
                            include_normal_batches=False, exclude_train_rise_below=None):
    raw_root, dst = Path(src).resolve(), Path(dst).resolve()
    src = resolve_source(raw_root)
    batch_dir = raw_root / NORMAL_FOLDER
    normal_batches = batch_dir if include_normal_batches and batch_dir.is_dir() else None
    source_root = raw_root if normal_batches is not None else src
    if dst.is_relative_to(raw_root):
        raise ValueError("원본 폴더 안에는 준비된 데이터를 저장할 수 없습니다.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
        raise ValueError("prefix는 영문·숫자·밑줄·하이픈만 사용하세요.")
    split = split_by_camera(src, seed, n_train, n_validation, normal_batches=normal_batches)
    if exclude_train_rise_below is not None:
        filter_calibration, excluded = filter_train_by_rise(
            split, src, reference_image_size, exclude_train_rise_below)
    calibration = {path.stem for rows in split.values()
                   for role, _, path in rows if role == "train"}
    boxes = None
    if roi:
        circles = load_reference_circles(
            src, reference_image_size, allowed_stems=calibration)
        boxes = roi_boxes(circles, roi_half, calibration_stems=calibration)
    suffix = ("_roi" if roi else "") + (f"_s{seed}" if seed else "")
    categories = {
        f"{prefix}_cam1{suffix}": ["camera_1"],
        f"{prefix}_cam2{suffix}": ["camera_2"],
        f"{prefix}_cam12{suffix}": ["camera_1", "camera_2"],
    }
    dst.mkdir(parents=True, exist_ok=True)
    if any((dst / category).exists() for category in categories):
        raise FileExistsError("결과 카테고리가 이미 있습니다. 새 prefix 또는 별도 출력 위치를 사용하세요.")

    # 중간 실패 결과를 완성된 데이터로 재사용하지 않도록 임시 폴더에서 준비한다.
    with tempfile.TemporaryDirectory(prefix=".company-prepare-", dir=dst) as temporary:
        staging = Path(temporary).resolve()
        if not staging.is_relative_to(dst):
            raise ValueError("임시 출력 위치가 데이터 출력 폴더 밖입니다.")
        for category, cameras in categories.items():
            out = staging / category
            out.mkdir()
            records = []
            for cam in cameras:
                for role, class_name, path in split[cam]:
                    prepared = ""
                    if role != "unused":
                        image = read_image(path)
                        if boxes:
                            image = crop(image, boxes[cam])
                        prepared = f"{role}/{class_name}/{path.name}"
                        write_image(out / prepared, image)
                    records.append({
                        "split": role, "class": class_name, "label": int(class_name != "good"),
                        "camera": cam, "group_id": group_of(path), "file": path.name,
                        "source": path.relative_to(source_root).as_posix(), "prepared_image": prepared,
                    })
            with (out / "split.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            info = {
                "protocol_status": "proposed",
                "split_unit": "capture_group",
                "seed": seed, "n_train_per_camera": n_train,
                "n_validation_per_camera": n_validation,
                "source_root": str(source_root), "roi_boxes_xyxy": boxes,
                "roi_calibration": "train_normal_only" if roi else None,
                "roi_calibration_stems": sorted(
                    stem for stem in calibration if camera_of(stem) in cameras) if roi else [],
                "reference_image_size": reference_image_size,
                "counts": split_counts(records),
            }
            if source_root != src:
                info["label_root"] = str(src)
            if exclude_train_rise_below is not None:
                info["train_filter"] = {
                    "exclude_train_rise_below": exclude_train_rise_below,
                    "camera_calibration": {cam: filter_calibration[cam] for cam in cameras},
                    "excluded_from_train": [entry for cam in cameras for entry in excluded[cam]],
                }
            (out / "dataset_info.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        for category in categories:
            target = (dst / category).resolve()
            if target.parent != dst:
                raise ValueError("잘못된 카테고리 출력 경로입니다.")
            (staging / category).rename(target)
    return {name: dst / name for name in categories}
