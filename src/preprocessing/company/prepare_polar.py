"""prepare.py로 만든 카테고리를 '면취 고리 펼치기(극좌표)' 버전으로 다시 내보낸다.

입력 : data/processed/company/company_cam1 (원본 크기, ROI 아님)
출력 : data/processed/company/company_cam1_polar
       - 폴더 구조(train/validation/test), split.csv는 입력과 똑같이 유지
         → 분할이 같으므로 원본·정렬·펼치기 결과를 공정하게 비교할 수 있다
       - polar.csv : 이미지별 검출 중심·시작점 각도·밝기 상승폭
       - dataset_info.json : 입력 정보 + 펼치기 설정

라벨 사용: prepare_aligned.py와 같다 (카메라별 대략 중심·반지름을 학습 정상 라벨로만 정함).
"""

import csv
import json
from pathlib import Path
import tempfile

from .dataset import read_image, write_image
from .polar import unwrap_image
from .prepare_aligned import LOW_RISE_WARNING, calibrate_cameras


def prepare_polar_dataset(data_root, categories, *, suffix="_polar", options=None):
    """categories의 각 카테고리를 펼치기 버전(<category><suffix>)으로 만든다.

    data_root : 준비된 카테고리들이 있는 폴더 (data/processed/company)
    options   : polar.unwrap_image에 넘길 설정 (rows, radial_px, inner, outer, angle_range)
    반환      : {새 카테고리 이름: 경로}
    """
    data_root = Path(data_root).resolve()                                    # 절대 경로로 통일
    options = dict(options or {})                                            # 원본 설정을 건드리지 않게 복사
    targets = {category: f"{category}{suffix}" for category in categories}  # 입력 → 출력 이름
    if any((data_root / name).exists() for name in targets.values()):       # 덮어쓰기 방지
        raise FileExistsError("펼치기 결과 카테고리가 이미 있습니다. 새 suffix를 사용하세요.")

    # 중간에 실패한 결과가 완성본처럼 남지 않도록 임시 폴더에 만든 뒤 마지막에 옮긴다
    with tempfile.TemporaryDirectory(prefix=".company-polar-", dir=data_root) as temporary:
        staging = Path(temporary)                                            # 임시 작업 폴더
        for source_name, target_name in targets.items():
            source = data_root / source_name                                 # 입력 카테고리 폴더
            info = json.loads((source / "dataset_info.json").read_text(encoding="utf-8"))
            if info.get("roi_boxes_xyxy"):                                   # ROI는 라벨 좌표와 안 맞음
                raise ValueError(f"{source_name}: ROI가 아닌 원본 크기 카테고리를 입력하세요.")
            if info.get("alignment") or info.get("polar"):                   # 이미 변환된 카테고리 방지
                raise ValueError(f"{source_name}: 정렬·펼치기 전의 원본 카테고리를 입력하세요.")
            with (source / "split.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))                          # 이미지별 분할 기록

            # 카메라별 대략 중심·반지름 (학습 정상 라벨만 사용)
            train_stems = [Path(r["file"]).stem for r in rows if r["split"] == "train"]
            calibration = calibrate_cameras(
                info["source_root"], train_stems, info.get("reference_image_size", 1216))

            out = staging / target_name                                      # 출력 카테고리 폴더
            out.mkdir()
            records = []                                                     # polar.csv에 쓸 내용
            for row in rows:
                if not row["prepared_image"]:                                # unused는 이미지가 없음
                    continue
                cam = calibration[row["camera"]]                             # 이 이미지 카메라의 대략값
                image = read_image(source / row["prepared_image"])           # 원본 크기 이미지 읽기
                stacked, detection = unwrap_image(image, cam["center"], cam["radius"], **options)
                write_image(out / row["prepared_image"], stacked)            # 입력과 같은 경로로 저장
                records.append({
                    "prepared_image": row["prepared_image"], "split": row["split"],
                    "class": row["class"], "camera": row["camera"],
                    "center_x": round(detection["center_x"], 2),
                    "center_y": round(detection["center_y"], 2),
                    "start_deg": detection["start_deg"],
                    "start_rise": round(detection["start_rise"], 1),
                    # 밝기 절벽이 약하면 시작점을 잘못 찾았을 수 있으니 표시
                    "low_confidence": detection["start_rise"] < LOW_RISE_WARNING,
                })

            (out / "split.csv").write_bytes((source / "split.csv").read_bytes())  # 분할 그대로 복사
            with (out / "polar.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)                                    # 이미지별 검출 기록
            info.update({                                                    # 어떻게 만들었는지 기록
                "polar_from": source_name,
                "polar": {
                    "method": "image_center_and_spring_start_then_ring_unwrap",
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
