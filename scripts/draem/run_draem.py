# [test] 이미지 몇 장 추론 + 시각화

"""예시:
    .\\.venv\\Scripts\\python.exe scripts/draem/run_draem.py `
        --checkpoint outputs/draem/bottle/draem_e700_bs8_lr0p0001_s0.pt `
        --image data/raw/mvtec_ad/bottle/test/broken_large/000.png `
        --image data/raw/mvtec_ad/bottle/test/good/000.png
"""

from pathlib import Path
import argparse
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.draem.model import DRAEMPredictor  # noqa: E402
from src.models.draem.visualization import save_draem_visualization  # noqa: E402


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description="DRAEM 이미지 이상 점수 추론")

    parser.add_argument("--checkpoint", type=Path, required=True, help="train_draem.py 결과 .pt")
    parser.add_argument("--image", type=Path, action="append", required=True,
                        help="테스트 이미지 경로 (여러 번 지정 가능)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/draem/visualizations"))
    parser.add_argument("--vis-max", type=float, default=1.0, help="맵 색상 상한 (DRAEM 맵은 0~1 확률)")

    args = parser.parse_args()

    if args.vis_max <= 0:
        parser.error("--vis-max는 0보다 커야 합니다.")

    checkpoint = resolve(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"체크포인트가 없습니다: {checkpoint}")

    output_dir = resolve(args.output_dir)

    # 1. 모델 로드
    model = DRAEMPredictor(checkpoint)
    print("사용 장치:", model.device)
    print("카테고리:", model.category)
    print("학습 epoch:", model.meta.get("trained_epochs"))

    # 2. 각 이미지 추론
    for image_path in args.image:
        image_path = resolve(image_path)
        if not image_path.is_file():
            parser.error(f"이미지가 없습니다: {image_path}")

        result = model.predict(image_path)

        print(f"\n[추론] {image_path}")
        print("Anomaly Map Shape:", tuple(result["anomaly_map"].shape))
        print(f"Image Anomaly Score: {result['image_score']:.6f}")

        # good/000.png -> good_000_anomaly.png
        output_path = output_dir / f"{image_path.parent.name}_{image_path.stem}_anomaly.png"

        save_draem_visualization(
            image=result["image"],
            reconstruction=result["reconstruction"],
            anomaly_map=result["anomaly_map"].numpy(),
            output_path=output_path,
            image_score=result["image_score"],
            vmax=args.vis_max,
        )
        print("시각화 저장:", output_path)

    print("\n추론 완료!")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
