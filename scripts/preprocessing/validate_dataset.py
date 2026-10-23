# 전처리 후 누락된 데이터가 있는지 검사용 테스트 코드입니다.

from pathlib import Path
from collections import Counter, defaultdict
import csv
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    # 검사할 manifest를 실행 인자로 받기
    if len(sys.argv) != 2:
        raise SystemExit(
            "사용법: python scripts/preprocessing/validate_dataset.py "
            "data/processed/bottle/bottle_variation_v2/manifest.csv"
        )

    manifest_path = ROOT / sys.argv[1]
    output_dir = manifest_path.parent

    if not manifest_path.is_file():
        raise SystemExit("manifest.csv 파일이 없습니다.")

    # 1. CSV 불러오기
    with open(
        manifest_path, "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "image_id", "source_image", "label",
            "variation", "level"
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise SystemExit("manifest.csv 컬럼이 올바르지 않습니다.")

        records = list(reader)

    errors = []
    image_ids = set()
    listed_paths = set()
    conditions_by_source = defaultdict(list) # 원본기반 변형이미지들 묶어주기 

    # 2. 이미지 검사
    for record in records:
        image_id = record["image_id"]
        image_path = (ROOT / image_id).resolve()

        if image_id in image_ids:
            errors.append(f"중복 이미지 ID: {image_id}")

        image_ids.add(image_id)
        listed_paths.add(image_path)

        if not image_path.is_relative_to(output_dir.resolve()):
            errors.append(f"출력 폴더 밖의 이미지 경로: {image_id}")
        elif not image_path.is_file():
            errors.append(f"이미지 누락: {image_id}")

        if record["label"] != "normal":
            errors.append(f"잘못된 라벨: {image_id}")

        conditions_by_source[record["source_image"]].append(
            (record["variation"], record["level"])
        )

    # 3. 실제 이미지와 CSV 비교
    actual_paths = {
        path.resolve()
        for path in output_dir.rglob("*.png")
    }

    for path in actual_paths - listed_paths:
        errors.append(f"CSV에 없는 이미지: {path.name}")

    # 4. YAML 실험 조건 검사
    config_path = output_dir / "config_used.yaml"

    if not config_path.is_file():
        raise SystemExit("config_used.yaml 파일이 없습니다.")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    expected_conditions = {("original", "0")}

    for variation, levels in config["variations"].items():
        for level in levels:
            expected_conditions.add((variation, str(level)))

    for source, conditions in conditions_by_source.items():
        if set(conditions) != expected_conditions:
            errors.append(f"변형 조건 불일치: {source}")

        if len(conditions) != len(expected_conditions):
            errors.append(f"변형 개수 불일치: {source}")

    # 5. 결과 출력
    print("\n=== 전처리 데이터 검증 ===")
    print("Manifest:", manifest_path)
    print("CSV 데이터 수:", len(records))
    print("실제 PNG 수:", len(actual_paths))
    print("원본 이미지 수:", len(conditions_by_source))

    print("\n[변형 종류별 이미지 수]")

    # 종류별 개수 계산
    counts = Counter(
        record["variation"] for record in records
    )

    for name, count in sorted(counts.items()):
        print(f"{name}: {count}장")

    print("\n[검증 결과]")

    if errors:
        for error in errors[:20]:
            print("[오류]", error)

        print(f"\n총 {len(errors)}개 오류 발견")
        raise SystemExit(1)

    print("모든 검사를 통과했습니다!")


if __name__ == "__main__":
    main()
