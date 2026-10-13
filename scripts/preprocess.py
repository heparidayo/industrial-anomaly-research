
from pathlib import Path
import sys
import csv
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.variations import (
    load_image,
    save_image,
    adjust_brightness,
    rotate_image,
    translate_image,
)


def load_config():
    """YAML 실험 설정 파일을 읽는다."""

    config_path = ROOT / "configs/variations_bottle.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config, config_path


def make_variations(original, config):
    """설정 파일을 기준으로 변형 이미지를 생성한다."""

    variations = [
        ("original", "original", 0, original)
    ]

    # 밝기
    for value in config["variations"]["brightness"]:
        name = f"brightness_{'p' if value >= 0 else 'm'}{abs(value)}"

        variations.append((
            name,
            "brightness",
            value,
            adjust_brightness(original, value)
        ))

    # 회전
    for value in config["variations"]["rotation"]:
        name = f"rotation_{'p' if value >= 0 else 'm'}{abs(value)}"

        variations.append((
            name,
            "rotation",
            value,
            rotate_image(original, value)
        ))

    # 위치 이동 (X축)
    for value in config["variations"]["translation_x"]:
        name = f"translation_x{'p' if value >= 0 else 'm'}{abs(value)}"

        variations.append((
            name,
            "translation_x",
            value,
            translate_image(original, value, 0)
        ))

    return variations


def main():
    config, config_path = load_config()

    category = config["dataset"]["category"]
    experiment_name = config["experiment"]["name"]

    # 원본 정상 테스트 데이터
    input_dir = (
        ROOT
        / config["dataset"]["root"]
        / category
        / "test"
        / "good"
    )

    # 실험별 독립된 출력 폴더
    output_dir = (
        ROOT
        / config["output"]["root"]
        / category
        / experiment_name
    )

    if not input_dir.is_dir():
        raise SystemExit("정상 테스트 데이터 폴더가 없습니다.")

    image_paths = sorted(input_dir.glob("*.png"))

    if not image_paths:
        raise SystemExit("PNG 이미지를 찾을 수 없습니다.")

    records = []

    for image_path in image_paths:
        original = load_image(image_path)

        variations = make_variations(original, config)

        for folder, variation, level, image in variations:
            output_path = output_dir / folder / image_path.name

            save_image(output_path, image)

            records.append({
                "image_id": output_path.relative_to(ROOT).as_posix(),
                "source_image": image_path.name,
                "label": "normal",
                "variation": variation,
                "level": level
            })

        print(f"[완료] {image_path.name}")

    # Manifest 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "source_image",
                "label",
                "variation",
                "level"
            ]
        )
        writer.writeheader()
        writer.writerows(records)

    # 어떤 설정으로 생성했는지도 함께 보관
    shutil.copy2(config_path, output_dir / "config_used.yaml")

    print("\n정상편차 데이터 생성 완료!")
    print(f"실험 이름: {experiment_name}")
    print(f"원본 이미지: {len(image_paths)}장")
    print(f"생성 이미지: {len(records)}장")
    print(f"저장 경로: {output_dir}")


if __name__ == "__main__":
    main()
