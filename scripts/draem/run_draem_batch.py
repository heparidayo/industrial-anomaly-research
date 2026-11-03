"""DRAEM 전체 테스트셋 추론 → 예측 CSV.

CSV 컬럼은 PatchCore(run_patchcore_batch.py)와 동일합니다.
    image_id, label, defect_type, variation, level, anomaly_score, inference_ms

예시:
    .\\.venv\\Scripts\\python.exe scripts/draem/run_draem_batch.py `
        --category bottle `
        --checkpoint outputs/draem/bottle/draem_e700_bs8_lr0p0001_s0.pt
"""

from pathlib import Path
import argparse
import csv
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.draem.model import DRAEMPredictor  # noqa: E402

FIELDNAMES = [
    "image_id",
    "label",
    "defect_type",
    "variation",
    "level",
    "anomaly_score",
    "inference_ms",
]


def main():
    parser = argparse.ArgumentParser(description="DRAEM 전체 테스트셋 추론")

    parser.add_argument("--category", default="bottle")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec_ad"),
                        help="<data-root>/<category>/test/ 를 읽음 (회사: data/processed/company_plc)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save-maps", action="store_true",
                        help="픽셀 단위 평가용 anomaly map을 .pt로 함께 저장")

    args = parser.parse_args()

    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    test_dir = data_root / args.category / "test"

    if not test_dir.is_dir():
        parser.error("테스트 데이터 폴더가 없습니다.")

    if not checkpoint.is_file():
        parser.error("체크포인트 파일이 없습니다.")

    # 출력 경로
    # MVTec은 기존 이름(bottle_original.csv), 그 외는 데이터 폴더 이름을 앞에 붙임
    prefix = "" if data_root.name == "mvtec_ad" else f"{data_root.name}_"
    output_path = args.output or (
        Path("outputs/predictions/draem") / f"{prefix}{args.category}_original.csv"
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if output_path.exists():
        parser.error("이미 CSV 파일이 존재합니다.")

    # 1. 정상 및 불량 테스트 이미지 수집
    image_paths = sorted(test_dir.glob("*/*.png"))
    if not image_paths:
        parser.error("테스트 이미지를 찾지 못했습니다.")

    # 2. 모델을 한 번만 로드
    model = DRAEMPredictor(checkpoint)

    if model.category != args.category:
        parser.error("체크포인트의 카테고리가 일치하지 않습니다.")

    print("=== DRAEM 전체 테스트셋 추론 ===")
    print("사용 장치:", model.device)
    print("테스트 이미지:", len(image_paths))

    # 첫 이미지로 GPU 워밍업
    model.predict(image_paths[0])

    records = []
    maps = {}

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
        label = 0 if defect_type == "good" else 1  # 정상=0, 불량=1
        image_id = image_path.relative_to(ROOT).as_posix()

        records.append({
            "image_id": image_id,
            "label": label,
            "defect_type": defect_type,
            "variation": "original",
            "level": 0,
            "anomaly_score": result["image_score"],
            "inference_ms": round(inference_ms, 3),
        })

        if args.save_maps:
            maps[image_id] = result["anomaly_map"].half()

        print(
            f"[{index + 1}/{len(image_paths)}] "
            f"{defect_type}/{image_path.name} "
            f"score={result['image_score']:.4f}"
        )

    # 4. CSV 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)

    if args.save_maps:
        maps_path = output_path.with_name(output_path.stem + "_maps.pt")
        torch.save({"maps": maps, "image_size": model.image_size}, maps_path)
        print("Anomaly map 저장:", maps_path)

    print("\n=== 추론 완료 ===")
    print("총 이미지:", len(records))
    print("CSV 저장:", output_path)


if __name__ == "__main__":
    main()
