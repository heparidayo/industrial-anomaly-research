# [test] 이미지 1장 테스트용


from pathlib import Path
import argparse
import sys

import torch
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.patchcore.backbone import PatchCoreBackbone
from src.models.patchcore.features import PatchFeatureExtractor
from src.models.patchcore.scoring import compute_patch_scores


def main():
    parser = argparse.ArgumentParser(
        description="PatchCore 이미지 이상 점수 추론"
    )

    parser.add_argument(
        "--memory",
        type=Path,
        required=True,
        help="Coreset Memory Bank 파일",
    )

    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        required=True,
        help="테스트 이미지 경로 (여러 번 지정 가능)",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("사용 장치:", device)

    # 1. Memory Bank 로드
    memory_path = args.memory
    if not memory_path.is_absolute():
        memory_path = ROOT / memory_path

    if not memory_path.is_file():
        parser.error(f"Memory Bank가 없습니다: {memory_path}")

    saved = torch.load(
        memory_path,
        map_location="cpu",
        weights_only=True,
    )

    memory_bank = saved["embeddings"].to(device)

    print("Memory Bank:", tuple(memory_bank.shape))

    # 2. Backbone 및 Feature Extractor 준비
    weights = Wide_ResNet50_2_Weights.DEFAULT
    transform = weights.transforms()

    backbone = PatchCoreBackbone(device)

    patch_extractor = PatchFeatureExtractor(
        patch_size=3,
        pretrain_embed_dim=1024,
        target_embed_dim=1024,
    ).to(device)

    patch_extractor.eval()

    # 3. 각 테스트 이미지 추론
    for image_path in args.image:
        if not image_path.is_absolute():
            image_path = ROOT / image_path

        if not image_path.is_file():
            parser.error(f"이미지가 없습니다: {image_path}")

        with Image.open(image_path) as image:
            tensor = transform(
                image.convert("RGB")
            ).unsqueeze(0).to(device)

        with torch.inference_mode():
            # CNN 특징 추출
            feature_maps = backbone.extract(tensor)

            # Patch Feature 생성
            patch_features = patch_extractor(feature_maps)

            # 최근접 이웃 거리 계산
            patch_scores = compute_patch_scores(
                patch_features,
                memory_bank,
            )

            # 이미지 단위 이상 점수
            image_score = patch_scores.max().item()

        print(f"\n[추론] {image_path}")
        print("Patch Score Shape:", tuple(patch_scores.shape))
        print(f"Image Anomaly Score: {image_score:.6f}")

    print("\n추론 완료!")


if __name__ == "__main__":
    main()
