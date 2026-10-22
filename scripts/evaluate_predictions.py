# csv 읽고 평가 결과 출력/저장


from pathlib import Path
import argparse
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_image_predictions


def main():
    parser = argparse.ArgumentParser(
        description="이미지 이상탐지 모델 성능 평가"
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="모델 추론 결과 CSV 경로",
    )

    args = parser.parse_args()

    csv_path = args.csv

    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path

    if not csv_path.is_file():
        parser.error(f"CSV 파일이 없습니다: {csv_path}")

    # 1. CSV 불러오기
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {
            "image_id", "label", "defect_type",
            "anomaly_score", "inference_ms"
        }

        if not required.issubset(reader.fieldnames or []):
            parser.error("CSV 필수 컬럼이 누락되었습니다.")

        records = list(reader)

    # 2. 성능 계산
    try:
        results = evaluate_image_predictions(records)
    except (ValueError, KeyError) as error:
        parser.error(str(error))

    # 3. 결과 출력
    print("\n=== 이미지 이상탐지 평가 ===")
    print("전체 이미지:", results["image_count"])
    print("정상 이미지:", results["normal_count"])
    print("불량 이미지:", results["anomaly_count"])

    print(f"\nImage-level AUROC: {results['image_auroc']:.4f}")
    print(f"평균 처리시간: {results['mean_inference_ms']:.2f} ms")
    print(f"중앙 처리시간: {results['median_inference_ms']:.2f} ms")

    print("\n[결함 종류별 점수]")

    for name, group in results["groups"].items():
        print(
            f"{name:15s} "
            f"n={group['count']:2d} "
            f"mean={group['mean']:.4f} "
            f"range=[{group['min']:.4f}, {group['max']:.4f}]"
        )

    # 4. JSON 파일 저장
    output_path = csv_path.with_name(
        csv_path.stem + "_metrics.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\n평가 결과 저장:", output_path)


if __name__ == "__main__":
    main()
