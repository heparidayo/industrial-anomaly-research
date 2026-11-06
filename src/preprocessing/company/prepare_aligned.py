"""prepare.py로 만든 카테고리를 '시작점 정렬 + 제외 영역 마스킹' 버전으로 다시 내보낸다.

입력 : data/processed/company/company_cam1 (원본 크기, ROI 아님)
출력 : data/processed/company/company_cam1_aligned
       - 폴더 구조(train/validation/test), split.csv는 입력과 똑같이 유지
         → 분할이 같으므로 정렬 전후 결과를 공정하게 비교할 수 있다
       - alignment.csv : 이미지별 검출 중심·시작점 각도·밝기 상승폭
       - dataset_info.json : 입력 정보 + 정렬 설정

라벨 사용: 카메라별 대략 중심·반지름을 "학습 정상 이미지 라벨"로만 정한다 (ROI와 같은 원칙).
각 이미지의 중심·시작점은 사진에서 찾는다.
"""

import csv
import json
from pathlib import Path
import tempfile

import numpy as np

from .alignment import align_image
from .dataset import camera_of, load_reference_circles, read_image, write_image

# 밝기 상승폭이 이보다 작으면 시작점 검출이 애매하다고 표시 (회사 데이터 97장의 최솟값은 약 148)
LOW_RISE_WARNING = 80.0


def calibrate_cameras(source_root, train_stems, image_size):
    """학습 정상 이미지의 기준 원 라벨 평균 → 카메라별 (대략 중심, 대략 반지름)."""
    circles = load_reference_circles(source_root, image_size, allowed_stems=set(train_stems))
    result = {}
    for cam in sorted({camera_of(stem) for stem in train_stems}):
        values = np.array([circles[stem] for stem in train_stems if camera_of(stem) == cam])
        # values 한 줄 = [중심 x, 중심 y, 반지름] → 이 카메라 이미지들의 평균을 쓴다
        result[cam] = {"center": values[:, :2].mean(0).tolist(), "radius": float(values[:, 2].mean())}
    return result


def prepare_aligned_dataset(data_root, categories, *, suffix="_aligned", options=None):
    """categories의 각 카테고리를 정렬 버전(<category><suffix>)으로 만든다.

    data_root : 준비된 카테고리들이 있는 폴더 (data/processed/company)
    options   : alignment.align_and_mask에 넘길 설정 (target_deg, exclusion_deg 등)
    반환      : {새 카테고리 이름: 경로}
    """
    data_root = Path(data_root).resolve()
    options = dict(options or {})
    targets = {category: f"{category}{suffix}" for category in categories}
    if any((data_root / name).exists() for name in targets.values()):
        raise FileExistsError("정렬 결과 카테고리가 이미 있습니다. 새 suffix를 사용하세요.")

    # 중간에 실패한 결과가 완성본처럼 남지 않도록 임시 폴더에 만든 뒤 마지막에 옮긴다 (prepare.py와 같은 방식)
    with tempfile.TemporaryDirectory(prefix=".company-align-", dir=data_root) as temporary:
        staging = Path(temporary)
        for source_name, target_name in targets.items():
            source = data_root / source_name
            info = json.loads((source / "dataset_info.json").read_text(encoding="utf-8"))
            if info.get("roi_boxes_xyxy"):
                # 라벨 좌표가 원본 이미지 기준이라, 이미 잘라 낸 ROI 카테고리는 받지 않는다
                raise ValueError(f"{source_name}: ROI가 아닌 원본 크기 카테고리를 입력하세요.")
            with (source / "split.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            # 카메라별 대략 중심·반지름 (학습 정상만 사용)
            train_stems = [Path(r["file"]).stem for r in rows if r["split"] == "train"]
            calibration = calibrate_cameras(
                info["source_root"], train_stems, info.get("reference_image_size", 1216))

            out = staging / target_name
            out.mkdir()
            records = []
            for row in rows:
                if not row["prepared_image"]:  # unused (학습 그룹의 불량 뷰)는 이미지가 없다
                    continue
                cam = calibration[row["camera"]]
                image = read_image(source / row["prepared_image"])
                aligned, detection = align_image(image, cam["center"], cam["radius"], **options)
                # 경로를 입력과 똑같이 써서 split.csv를 그대로 재사용할 수 있게 한다
                write_image(out / row["prepared_image"], aligned)
                records.append({
                    "prepared_image": row["prepared_image"], "split": row["split"],
                    "class": row["class"], "camera": row["camera"],
                    "center_x": round(detection["center_x"], 2),
                    "center_y": round(detection["center_y"], 2),
                    "start_deg": detection["start_deg"],
                    "start_rise": round(detection["start_rise"], 1),
                    # 밝기 절벽이 약하면 시작점을 잘못 찾았을 수 있으니 표시해 둔다
                    "low_confidence": detection["start_rise"] < LOW_RISE_WARNING,
                })

            # 분할 기록은 그대로 복사 (같은 분할 = 공정한 비교)
            (out / "split.csv").write_bytes((source / "split.csv").read_bytes())
            with (out / "alignment.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            info.update({
                "aligned_from": source_name,
                "alignment": {
                    "method": "image_center_and_spring_start",
                    "calibration": "train_normal_labels_only",
                    "camera_calibration": calibration,
                    "options": {k: list(v) if isinstance(v, tuple) else v for k, v in options.items()},
                    "low_confidence_count": sum(r["low_confidence"] for r in records),
                },
            })
            (out / "dataset_info.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

        # 모든 카테고리가 다 만들어졌을 때만 진짜 위치로 옮긴다
        for target_name in targets.values():
            (staging / target_name).rename(data_root / target_name)
    return {name: data_root / name for name in targets.values()}
