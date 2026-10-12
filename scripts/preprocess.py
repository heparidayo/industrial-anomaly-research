from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.variations import (
    load_image, save_image,
    adjust_brightness,
)


def main():
    # 원본 이미지 경로
    input_path = (
        ROOT / "data/raw/mvtec_ad/bottle/test/good/000.png"
    )

    # 저장할 폴더 경로
    output_dir = ROOT / "data/processed/bottle"
    

    # 원본 이미지 로드
    original = load_image(input_path)

    # 밝기 변경
    bright = adjust_brightness(original, +20)
    dark = adjust_brightness(original, -20)

    # 결과 저장
    save_image(output_dir / "original_000.png", original)
    save_image(output_dir / "brightness_p20_000.png", bright)
    save_image(output_dir / "brightness_m20_000.png", dark)

    print("밝기 정상편차 이미지 생성 완료!")
    print("원본 평균 밝기:", original.mean()) # 모든 채널 평균 픽셀값 출력
    print("+20% 평균 밝기:", bright.mean())
    print("-20% 평균 밝기:", dark.mean())
    
    print("[경로]", output_dir)


if __name__ == "__main__":
    main()