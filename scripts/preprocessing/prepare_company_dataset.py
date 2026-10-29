
# [이 스크립트의 역할] 회사 원본 데이터 → MVTec 형식 카테고리 만들기
# data/raw/company 의 원본을 카메라별로 나눠 train/good, test/<클래스> 구조로 복사한다.
# 이렇게 만들면 PatchCore 스크립트(build_patchcore_memory.py 등)를 코드 수정 없이 쓸 수 있다.
#
# 실행 예:
#   python scripts/preprocessing/prepare_company_dataset.py                  (seed 0, 원본 크기)
#   python scripts/preprocessing/prepare_company_dataset.py --roi            (부품 영역만 자르기)
#   python scripts/preprocessing/prepare_company_dataset.py --seed 1         (다른 분할)
#
# 결과 카테고리 (data/raw/mvtec_ad/ 아래):
#   company_cam1[_roi][_s<seed>]   카메라 1만
#   company_cam2[_roi][_s<seed>]   카메라 2만
#   company_cam12[_roi][_s<seed>]  카메라 1 + 2   (seed 0이면 _s0 생략)
# 카테고리마다:
#   split.csv          어떤 이미지가 학습/테스트인지 (다른 모델 담당과 같은 분할을 쓸 때 공유)
#   dataset_info.json  seed, 학습 장수, ROI 박스
#
# 결과 폴더 구조 예 (company_cam1):
#   train/good/*.png           학습용 정상 (카메라당 10장)
#   test/good/*.png            테스트용 정상 (나머지 6장)
#   test/over, test/step, test/미흡/*.png   불량 (전부 테스트)
#
# 원본(data/raw/company)은 수정하지 않는다.

from pathlib import Path
import argparse
import csv
import json
import sys

# 레포 최상위 폴더를 찾아 import 경로에 추가 (다른 스크립트와 같은 방식)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 실제 처리 함수는 src/preprocessing/company_dataset.py 에 있다
from src.preprocessing.company_dataset import (
    crop,                    # 박스로 자르기
    load_reference_circles,  # 라벨에서 부품 기준 원 위치 읽기
    read_image,              # 한글 경로도 되는 이미지 읽기
    roi_boxes,               # 카메라별 고정 ROI 박스 계산
    split_by_camera,         # 카메라별 학습/테스트 분할
    write_image,             # 한글 경로도 되는 이미지 저장
)


def resolve(path):
    # 상대경로면 레포 최상위 기준 절대경로로 바꾼다
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description="회사 데이터 MVTec 형식 변환")
    # --src: 회사 원본 폴더 (정상/, over/, step/, 미흡/ 이 들어 있는 곳)
    parser.add_argument("--src", type=Path, default=Path("data/raw/company"))
    # --dst: 결과 카테고리를 만들 폴더 (PatchCore 스크립트가 읽는 위치)
    parser.add_argument("--dst", type=Path, default=Path("data/raw/mvtec_ad"))
    # --prefix: 카테고리 이름 앞부분 (company → company_cam1 ...)
    parser.add_argument("--prefix", default="company")
    # 카메라별 정상 이미지 중 학습에 쓸 장수 (나머지는 test/good)
    parser.add_argument("--n-train", type=int, default=10)
    # --seed: 정상 이미지를 섞는 난수 시드 (같은 seed = 항상 같은 분할)
    parser.add_argument("--seed", type=int, default=0)
    # --roi: 카메라별 고정 박스로 부품 영역만 잘라서 저장
    parser.add_argument("--roi", action="store_true")
    # ROI 박스 반 크기 (기준 원 반지름 약 357px + 위치 흔들림 여유) → 840x840
    parser.add_argument("--roi-half", type=int, default=420)
    args = parser.parse_args()

    src = resolve(args.src)
    dst_root = resolve(args.dst)

    if not src.is_dir():
        parser.error(f"회사 원본 폴더가 없습니다: {src}")

    # 1. 카메라별 학습/테스트 분할
    #    split = {"camera_1": [("train", "good", 경로), ("test", "over", 경로), ...], "camera_2": [...]}
    split = split_by_camera(src, args.seed, args.n_train)
    # 2. ROI 박스 (--roi 일 때만)
    #    boxes = {"camera_1": [x1, y1, x2, y2], "camera_2": [...]}
    boxes = roi_boxes(load_reference_circles(src), args.roi_half) if args.roi else None

    # 3. 만들 카테고리 3개 (카메라 1 / 카메라 2 / 카메라 1+2)
    #    이름 뒤에 ROI 여부와 seed를 붙여 서로 다른 설정이 섞이지 않게 함
    suffix = ("_roi" if args.roi else "") + (f"_s{args.seed}" if args.seed else "")
    categories = {
        f"{args.prefix}_cam1{suffix}": ["camera_1"],
        f"{args.prefix}_cam2{suffix}": ["camera_2"],
        f"{args.prefix}_cam12{suffix}": ["camera_1", "camera_2"],  # 두 카메라 분할을 그대로 합침
    }

    for category, cameras in categories.items():
        out = dst_root / category

        # 이미 있으면 덮어쓰지 않고 건너뜀
        if out.exists():
            print("건너뜀 (이미 존재):", category)
            continue

        # 4. 이미지 복사 (ROI면 잘라서 저장)
        records = []  # split.csv에 쓸 내용
        for cam in cameras:
            for split_name, class_name, path in split[cam]:
                image = read_image(path)
                if boxes:
                    image = crop(image, boxes[cam])
                # 예) company_cam1/train/good/PLC-..._camera_1.png
                write_image(out / split_name / class_name / path.name, image)
                records.append({
                    "split": split_name,     # train / test
                    "class": class_name,     # good / over / step / 미흡
                    "camera": cam,
                    "file": path.name,
                    "source": path.relative_to(src).as_posix(),  # 원본 안의 위치
                })

        # 5. 분할 기록 저장 (utf-8-sig: 엑셀에서 한글이 안 깨짐)
        with open(out / "split.csv", "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

        # 6. 어떤 설정으로 만들었는지 기록 (부품 회전 스크립트가 ROI 박스를 여기서 읽음)
        info = {"seed": args.seed, "n_train_per_camera": args.n_train, "roi_boxes_xyxy": boxes}
        (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

        print(category, len(records), "ROI" if boxes else "")


# 직접 실행했을 때만 main() 실행
if __name__ == "__main__":
    main()
