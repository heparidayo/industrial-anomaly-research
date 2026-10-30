
# [이 스크립트의 역할] "부품만 회전" 정상편차 이미지 생성
# preprocess.py와 같은 방식으로 YAML을 읽어 test/good 정상 이미지에 회전을 적용한다.
# 차이점: 이미지 전체가 아니라 부품 원 안쪽만 회전 (src/preprocessing/part_rotation.py)
#
# 왜 필요한가:
#   preprocess.py의 회전은 받침대와 배경까지 같이 돌리고 빈 모서리를 반사로 채운다.
#   실제 공정에서는 받침대는 고정이고 부품만 돌아가므로, 그 상황을 따로 재현한다.
#
# 부품 원의 위치는 회사 라벨의 기준 원(class 1)에서 이미지별로 가져온다.
# 라벨은 "편차 이미지를 만들 때"만 쓰고(실제 회전 상황 재현용), 추론에는 쓰지 않는다.
#
# 실행 예:
#   python scripts/preprocessing/preprocess_part_rotation.py --config configs/experiments/company_cam1_roi_partrot_v1.yaml
#   python scripts/preprocessing/validate_dataset.py data/processed/company_cam1_roi/company_cam1_roi_partrot_v1/manifest.csv
#
# 결과: data/processed/<category>/<experiment.name>/
#   rotation_part_<p|m><각도>/*.png, original/*.png, manifest.csv, config_used.yaml
#   (preprocess.py와 같은 형식 → validate_dataset.py, run_patchcore_variation.py 그대로 사용)

from pathlib import Path
import argparse
import csv
import json
import shutil
import sys

import cv2
import yaml

# 레포 최상위 폴더를 찾아 import 경로에 추가
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# camera_of: 파일명으로 카메라 구분, load_reference_circles: 라벨에서 부품 원 위치 읽기
from src.preprocessing.company_dataset import camera_of, load_reference_circles
# 원 안쪽만 회전하는 변환 함수
from src.preprocessing.part_rotation import rotate_part


def main():
    parser = argparse.ArgumentParser(description="부품만 회전 정상편차 데이터 생성")
    # --config: 실험 조건 YAML (configs/experiments/company_cam*_roi_partrot_v1.yaml)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    # 상대경로면 레포 최상위 기준
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if not config_path.is_file():
        parser.error(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # YAML에서 필요한 값 꺼내기
    category = config["dataset"]["category"]          # 예) company_cam1_roi
    experiment_name = config["experiment"]["name"]    # 예) company_cam1_roi_partrot_v1
    angles = config["variations"]["rotation_part"]    # 회전 각도 목록 (도)
    margin = config["part_rotation"]["margin"]        # 기준 원 반지름에 더할 여유 픽셀

    category_dir = ROOT / config["dataset"]["root"] / category
    input_dir = category_dir / "test" / "good"        # 편차를 줄 원본 = 테스트용 정상 이미지
    output_dir = ROOT / config["output"]["root"] / category / experiment_name

    # preprocess.py와 같은 보호: 같은 실험 이름에 다른 설정이면 중단
    # (YAML 이름을 안 바꾸고 조건만 바꿔서 기존 결과를 덮어쓰는 실수 방지)
    saved_config = output_dir / "config_used.yaml"
    if saved_config.is_file():
        with open(saved_config, "r", encoding="utf-8") as file:
            if yaml.safe_load(file) != config:
                raise SystemExit(
                    "\n[오류] 동일한 실험 이름으로 서로 다른 설정이 사용되었습니다.\n"
                    "experiment.name을 새로운 이름으로 변경해주세요."
                )

    image_paths = sorted(input_dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"정상 테스트 이미지가 없습니다: {input_dir}")

    # ROI로 자른 카테고리면 라벨 좌표(원본 기준)를 박스만큼 옮겨야 한다
    # 예) 원본 기준 원 중심 (706, 648), ROI 박스 시작 (286, 228) → 잘린 이미지에서는 (420, 420)
    info_path = category_dir / "dataset_info.json"
    boxes = None
    if info_path.is_file():
        boxes = json.loads(info_path.read_text(encoding="utf-8"))["roi_boxes_xyxy"]

    # 모든 이미지의 기준 원 {파일명: (cx, cy, r)} (원본 1216px 좌표)
    circles = load_reference_circles(ROOT / config["labels"]["root"])

    records = []  # manifest.csv에 쓸 내용
    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        # 이 이미지의 기준 원 → 현재 이미지 좌표
        cx, cy, r = circles[image_path.stem]
        # ROI 박스의 왼쪽 위 좌표 (ROI가 아니면 0, 0 → 좌표 그대로)
        x1, y1 = (boxes[camera_of(image_path)][:2]) if boxes else (0, 0)
        center = (cx - x1, cy - y1)
        radius = r + margin  # 테두리의 어두운 링까지 같이 돌도록 여유를 더함

        # 원본(회전 0) + 각도별 회전 이미지
        variations = [("original", "original", 0, image)]
        for angle in angles:
            # 폴더 이름: +30 → rotation_part_p30, -30 → rotation_part_m30 (preprocess.py와 같은 규칙)
            folder = f"rotation_part_{'p' if angle >= 0 else 'm'}{abs(angle):g}"
            variations.append((folder, "rotation_part", angle,
                               rotate_part(image, center, radius, angle)))

        # 저장 + manifest 기록
        for folder, variation, level, result in variations:
            output_path = output_dir / folder / image_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), result):
                raise RuntimeError(f"이미지 저장에 실패했습니다: {output_path}")
            records.append({
                "image_id": output_path.relative_to(ROOT).as_posix(),  # 레포 기준 상대경로
                "source_image": image_path.name,  # 어느 원본에서 만들었는지
                "label": "normal",                # 편차 이미지는 모두 정상
                "variation": variation,
                "level": level,  # YAML 값 그대로 (validate_dataset.py가 문자열로 비교)
            })

        print(f"[완료] {image_path.name}")

    # manifest.csv 저장 (preprocess.py와 같은 컬럼)
    with open(output_dir / "manifest.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image_id", "source_image", "label", "variation", "level"]
        )
        writer.writeheader()
        writer.writerows(records)

    # 어떤 설정으로 만들었는지 사본 저장 (validate_dataset.py와 중복 실행 보호에 사용)
    shutil.copy2(config_path, output_dir / "config_used.yaml")

    print("\n부품만 회전 데이터 생성 완료!")
    print(f"실험 이름: {experiment_name}")
    print(f"원본 이미지: {len(image_paths)}장")
    print(f"생성 이미지: {len(records)}장")
    print(f"저장 경로: {output_dir}")


# 직접 실행했을 때만 main() 실행
if __name__ == "__main__":
    main()
