
# [이 스크립트의 역할] PatchCore 3단계 - 테스트셋 전체 추론 → 결과 CSV 저장
# test 폴더의 모든 이미지(정상 + 불량)를 추론해서
# 이미지별 이상 점수와 추론시간을 팀 공통 형식의 CSV로 저장한다.
#
# 실행 예:
#   python scripts/patchcore/run_patchcore_batch.py --category bottle
#       --memory outputs/patchcore/bottle/coreset_raw_memory_full_1pct_d128_s0.pt
#
# 결과: outputs/predictions/patchcore/bottle_original.csv
# 다음 단계: scripts/evaluation/evaluate_predictions.py --csv <이 CSV> 로 AUROC 등 평가
#
# CSV 컬럼 (성능평가 담당과 공유하는 형식):
#   image_id      : 이미지 경로 (레포 기준 상대경로)
#   label         : 정답 (0 = 정상, 1 = 불량)
#   defect_type   : 폴더 이름 (good, broken_large, ...)
#   variation     : 정상편차 종류 (이 스크립트는 원본만 다루므로 항상 "original")
#   level         : 편차 강도 (원본이므로 항상 0)
#   anomaly_score : PatchCore 이미지 이상 점수 (클수록 이상)
#   inference_ms  : 이미지 1장 추론에 걸린 시간 (밀리초)

from pathlib import Path
import argparse
import csv
import sys
import time

import torch

# 레포 최상위 폴더를 찾아 import 경로에 추가
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 메모리 뱅크 로딩 + 추론을 묶어둔 클래스 (src/models/patchcore/model.py)
from src.models.patchcore.model import PatchCorePredictor


def main():
    parser = argparse.ArgumentParser(
        description="PatchCore 전체 테스트셋 추론"
    )

    # --category: 테스트할 카테고리 (메모리 뱅크의 카테고리와 같아야 함)
    parser.add_argument("--category", default="bottle")
    # --memory: coreset 메모리 뱅크 파일 (필수)
    parser.add_argument("--memory", type=Path, required=True)
    # --output: CSV 저장 경로 (안 주면 기본 경로 사용)
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    memory_path = args.memory
    if not memory_path.is_absolute():
        memory_path = ROOT / memory_path

    # 예) data/raw/mvtec_ad/bottle/test
    test_dir = (
        ROOT / "data/raw/mvtec_ad"
        / args.category / "test"
    )

    if not test_dir.is_dir():
        parser.error("테스트 데이터 폴더가 없습니다.")

    if not memory_path.is_file():
        parser.error("Memory Bank 파일이 없습니다.")

    # 출력 경로
    # --output을 안 줬으면 outputs/predictions/patchcore/<category>_original.csv
    # (or: 왼쪽 값이 None이면 오른쪽 값을 사용)
    output_path = args.output or (
        Path("outputs/predictions/patchcore")
        / f"{args.category}_original.csv"
    )

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    # 기존 결과를 덮어쓰지 않도록 보호
    # (coreset 비율 등을 바꿔 다시 돌릴 때는 --output으로 다른 파일명을 주면 된다)
    if output_path.exists():
        parser.error("이미 CSV 파일이 존재합니다.")

    # 1. 정상 및 불량 테스트 이미지 수집
    # "*/*.png": test 아래 모든 하위 폴더(good, broken_large, ...)의 png
    # 이름순 정렬 → 폴더 이름순, 그 안에서 파일 이름순
    image_paths = sorted(test_dir.glob("*/*.png"))

    if not image_paths:
        parser.error("테스트 이미지를 찾지 못했습니다.")

    # 2. PatchCore 모델을 한 번만 로드
    # CNN 로딩과 메모리 뱅크 GPU 업로드를 처음에 한 번만 하고 모든 이미지에 재사용
    model = PatchCorePredictor(memory_path)

    # bottle로 만든 메모리 뱅크로 다른 카테고리를 테스트하는 실수를 막는다
    if model.category != args.category:
        parser.error("Memory Bank의 카테고리가 일치하지 않습니다.")

    print("=== PatchCore 전체 테스트셋 추론 ===")
    print("사용 장치:", model.device)
    print("테스트 이미지:", len(image_paths))

    # 첫 이미지로 GPU 워밍업
    # GPU는 처음 한 번 실행할 때 준비 작업(커널 로딩, 메모리 할당 등) 때문에 유독 느리다.
    # 결과는 버리고 한 번 미리 돌려서, 추론시간 측정에 이 준비 시간이 섞이지 않게 한다.
    model.predict(image_paths[0])

    # 이미지별 결과(딕셔너리)를 모을 목록
    records = []

    # 3. 모든 이미지 추론
    for index, image_path in enumerate(image_paths):

        # GPU는 명령을 받아두고 나중에 실행하는(비동기) 방식이라,
        # 시간을 재기 전에 앞서 쌓인 GPU 작업이 모두 끝날 때까지 기다린다
        if model.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        # 이미지 1장 추론 (파일 읽기 + 전처리 + 특징 추출 + 거리 계산 포함)
        result = model.predict(image_path)

        # 이번 이미지의 GPU 작업이 실제로 다 끝난 시점까지 기다린 뒤 시간 측정 종료
        if model.device.type == "cuda":
            torch.cuda.synchronize()

        # 초 → 밀리초
        inference_ms = (time.perf_counter() - start) * 1000

        # 이미지가 들어있는 폴더 이름 = 결함 종류 (good, broken_large, ...)
        defect_type = image_path.parent.name

        # 정상=0, 불량=1
        label = 0 if defect_type == "good" else 1

        records.append({
            # 레포 기준 상대경로, 슬래시(/)로 통일 (Windows의 \ 대신)
            "image_id": image_path.relative_to(ROOT).as_posix(),
            "label": label,
            "defect_type": defect_type,
            "variation": "original",
            "level": 0,
            "anomaly_score": result["image_score"],
            # 소수점 셋째 자리까지 반올림
            "inference_ms": round(inference_ms, 3),
        })

        print(
            f"[{index + 1}/{len(image_paths)}] "
            f"{defect_type}/{image_path.name} "
            f"score={result['image_score']:.4f}"
        )

    # 4. CSV 저장
    # 저장 폴더가 없으면 만든다
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # newline="": Windows에서 CSV 줄 사이에 빈 줄이 생기는 문제 방지
    # utf-8-sig: 엑셀에서 열어도 한글이 깨지지 않는 인코딩
    with open(
        output_path, "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        # DictWriter: 딕셔너리 목록을 CSV로 쓰는 도구
        # fieldnames: 열(컬럼) 순서
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "label",
                "defect_type",
                "variation",
                "level",
                "anomaly_score",
                "inference_ms",
            ],
        )

        writer.writeheader()       # 첫 줄: 컬럼 이름
        writer.writerows(records)  # 나머지 줄: 이미지별 결과

    print("\n=== 추론 완료 ===")
    print("총 이미지:", len(records))
    print("CSV 저장:", output_path)


# 직접 실행했을 때만 main() 실행
if __name__ == "__main__":
    main()
