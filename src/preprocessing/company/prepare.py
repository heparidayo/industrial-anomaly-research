"""원본을 보존하며 공통 촬영번호 분할을 MVTec 모양의 입력으로 내보낸다."""

import csv
import json
from pathlib import Path
import re
import tempfile

from .dataset import (
    camera_of, crop, group_of, load_reference_circles, read_image,
    resolve_source, roi_boxes, split_by_camera, write_image,
)


def prepare_company_dataset(src, dst, *, prefix="company", seed=0,
                            n_train=10, n_validation=3, roi=False,
                            roi_half=420, reference_image_size=1216):
    src, dst = resolve_source(src), Path(dst).resolve()
    if dst.is_relative_to(src):
        raise ValueError("원본 폴더 안에는 준비된 데이터를 저장할 수 없습니다.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
        raise ValueError("prefix는 영문·숫자·밑줄·하이픈만 사용하세요.")
    split = split_by_camera(src, seed, n_train, n_validation)
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
                        "source": path.relative_to(src).as_posix(), "prepared_image": prepared,
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
                "source_root": str(src), "roi_boxes_xyxy": boxes,
                "roi_calibration": "train_normal_only" if roi else None,
                "roi_calibration_stems": sorted(
                    stem for stem in calibration if camera_of(stem) in cameras) if roi else [],
                "reference_image_size": reference_image_size,
                "counts": {name: sum(row["split"] == name for row in records)
                           for name in ("train", "validation", "test", "unused")},
            }
            (out / "dataset_info.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        for category in categories:
            target = (dst / category).resolve()
            if target.parent != dst:
                raise ValueError("잘못된 카테고리 출력 경로입니다.")
            (staging / category).rename(target)
    return {name: dst / name for name in categories}
