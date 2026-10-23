
from pathlib import Path
import sys
import csv
import shutil
import yaml
import argparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.variations import (
    load_image,
    save_image,
    adjust_brightness,
    rotate_image,
    translate_image,
)



def load_config():
    """실행 인자로 전달받은 YAML 설정 파일을 읽는다."""

    parser = argparse.ArgumentParser(
        description="산업 이상탐지 정상편차 데이터 생성"
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="실험 YAML 파일 경로"
    )

    args = parser.parse_args()

    # 상대경로는 프로젝트 루트 기준으로 처리
    config_path = args.config

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config_path = config_path.resolve()

    # 파일 존재 여부 확인
    if not config_path.is_file():
        parser.error(
            f"설정 파일을 찾을 수 없습니다: {config_path}"
        )

    # YAML 파일 읽기
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        parser.error("YAML 설정 내용이 올바르지 않습니다.")

    print(f"[설정 파일] {config_path}")

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

    
    # 이전 실험 설정과 충돌하는지 검사
    # 실수로 YAML 이름 안바꾸고 데이터 덮어버리는거 대비
    saved_config = output_dir / "config_used.yaml"
    if saved_config.is_file():
        with open(saved_config, "r", encoding="utf-8") as file:
            previous_config = yaml.safe_load(file)

        if previous_config != config:
            raise SystemExit(
                "\n[오류] 동일한 실험 이름으로 서로 다른 설정이 "
                "사용되었습니다.\n"
                "experiment.name을 새로운 이름으로 변경해주세요."
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
