
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.patchcore.backbone import PatchCoreBackbone


def main():
    print("=== PatchCore 특징 추출 ===")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"사용 장치: {device}") # GPU 확인용 추가

    image_path = (
        ROOT / "data/raw/mvtec_ad/bottle/test/good/000.png"
    )

    # 사전학습 가중치에 맞는 공식 전처리 사용 
    ## 공식 전처리가 정상 편차 실험에 영향을 줄수도있음!
    # todo: 전처리/이미지 입력 정책 통일하기~
    weights = Wide_ResNet50_2_Weights.DEFAULT
    transform = weights.transforms()

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    # 특징 추출기 생성
    backbone = PatchCoreBackbone(device)

    # Feature Map 추출
    features = backbone.extract(tensor)

    for name, feature in features.items():
        print(f"{name}: {tuple(feature.shape)}")

    print("\n특징 추출 완료!")


if __name__ == "__main__":
    main()
