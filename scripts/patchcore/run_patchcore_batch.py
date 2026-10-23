
from pathlib import Path
import argparse
import csv
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.patchcore.model import PatchCorePredictor


def main():
    parser = argparse.ArgumentParser(
        description="PatchCore 전체 테스트셋 추론"
    )

    parser.add_argument("--category", default="bottle")
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    memory_path = args.memory
    if not memory_path.is_absolute():
        memory_path = ROOT / memory_path

    test_dir = (
        ROOT / "data/raw/mvtec_ad"
        / args.category / "test"
    )

    if not test_dir.is_dir():
        parser.error("테스트 데이터 폴더가 없습니다.")

    if not memory_path.is_file():
        parser.error("Memory Bank 파일이 없습니다.")

    # 출력 경로
    output_path = args.output or (
        Path("outputs/predictions/patchcore")
        / f"{args.category}_original.csv"
    )

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if output_path.exists():
        parser.error("이미 CSV 파일이 존재합니다.")

    # 1. 정상 및 불량 테스트 이미지 수집
    image_paths = sorted(test_dir.glob("*/*.png"))

    if not image_paths:
        parser.error("테스트 이미지를 찾지 못했습니다.")

    # 2. PatchCore 모델을 한 번만 로드
    model = PatchCorePredictor(memory_path)

    if model.category != args.category:
        parser.error("Memory Bank의 카테고리가 일치하지 않습니다.")

    print("=== PatchCore 전체 테스트셋 추론 ===")
    print("사용 장치:", model.device)
    print("테스트 이미지:", len(image_paths))

    # 첫 이미지로 GPU 워밍업
    model.predict(image_paths[0])

    records = []

    # 3. 모든 이미지 추론
    for index, image_path in enumerate(image_paths):

        if model.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        result = model.predict(image_path)

        if model.device.type == "cuda":
            torch.cuda.synchronize()

        inference_ms = (time.perf_counter() - start) * 1000

        defect_type = image_path.parent.name

        # 정상=0, 불량=1
        label = 0 if defect_type == "good" else 1

        records.append({
            "image_id": image_path.relative_to(ROOT).as_posix(),
            "label": label,
            "defect_type": defect_type,
            "variation": "original",
            "level": 0,
            "anomaly_score": result["image_score"],
            "inference_ms": round(inference_ms, 3),
        })

        print(
            f"[{index + 1}/{len(image_paths)}] "
            f"{defect_type}/{image_path.name} "
            f"score={result['image_score']:.4f}"
        )

    # 4. CSV 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(
        output_path, "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
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

        writer.writeheader()
        writer.writerows(records)

    print("\n=== 추론 완료 ===")
    print("총 이미지:", len(records))
    print("CSV 저장:", output_path)


if __name__ == "__main__":
    main()
