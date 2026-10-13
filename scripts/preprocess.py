
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.variations import (
    load_image,
    save_image,
    adjust_brightness,
    rotate_image,
    translate_image,
)


def main():
    input_path = (
        ROOT / "data/raw/mvtec_ad/bottle/test/good/000.png"
    )

    output_dir = ROOT / "data/processed/bottle"

    # 1. 원본 이미지
    original = load_image(input_path)

    # 2. 밝기 변형
    bright = adjust_brightness(original, +20)
    dark = adjust_brightness(original, -20)

    # 3. 회전 변형
    rotate_p5 = rotate_image(original, +5)
    rotate_m5 = rotate_image(original, -5)

    # 4. 위치 이동
    move_right = translate_image(original, 6, 0)
    move_left = translate_image(original, -6, 0)

    # 5. 이미지 저장
    images = {
        "original_000.png": original,
        "brightness_p20_000.png": bright,
        "brightness_m20_000.png": dark,
        "rotation_p5_000.png": rotate_p5,
        "rotation_m5_000.png": rotate_m5,
        "translation_xp6_000.png": move_right,
        "translation_xm6_000.png": move_left,
    }

    for filename, image in images.items():
        save_image(output_dir / filename, image)
        print(f"[저장 완료] {filename}")

    print(f"\n총 {len(images)}장 생성 완료!")


if __name__ == "__main__":
    main()
