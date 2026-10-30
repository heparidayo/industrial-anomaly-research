# [test] 이미지 1장 테스트용

# [이 스크립트의 역할] 지정한 이미지 몇 장을 추론하고 히트맵 이미지를 저장한다.
# 점수만 필요하면 run_patchcore_batch.py, "어디가 이상한지 눈으로 보기"가 필요하면 이 스크립트.
# 발표 자료의 Anomaly map 예시 그림을 만들 때 쓰면 된다.
#
# 실행 예 (--image는 여러 번 줄 수 있다):
#   python scripts/patchcore/run_patchcore.py
#       --memory outputs/patchcore/bottle/coreset_raw_memory_full_1pct_d128_s0.pt
#       --image data/raw/mvtec_ad/bottle/test/good/000.png
#       --image data/raw/mvtec_ad/bottle/test/broken_large/000.png
#
# 결과: outputs/patchcore/visualizations/good_000_anomaly.png 등
#       [입력 이미지 | 히트맵 | 겹친 이미지] 3장이 가로로 붙은 그림


from pathlib import Path
import argparse
import sys

import torch
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights

# 레포 최상위 폴더를 찾아 import 경로에 추가
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.patchcore.backbone import PatchCoreBackbone
from src.models.patchcore.features import PatchFeatureExtractor
from src.models.patchcore.scoring import compute_patch_scores
from src.models.patchcore.anomaly_map import create_anomaly_map
# 히트맵 그림 저장 함수 (평가 담당 쪽 코드)
from src.evaluation.visualization import save_anomaly_visualization

def main():
    parser = argparse.ArgumentParser(
        description="PatchCore 이미지 이상 점수 추론"
    )

    # --memory: 2단계에서 만든 coreset 메모리 뱅크 파일 (필수)
    parser.add_argument(
        "--memory",
        type=Path,
        required=True,
        help="Coreset Memory Bank 파일",
    )

    # --image: 추론할 이미지 경로 (필수)
    # action="append": --image를 여러 번 쓰면 전부 목록으로 모인다
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        required=True,
        help="테스트 이미지 경로 (여러 번 지정 가능)",
    )

    # --output-dir: 히트맵 그림을 저장할 폴더
    parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("outputs/patchcore/visualizations"),
)

    # --vis-max: 히트맵 색상의 최대 기준 점수
    # 점수 0 → 파란색, vis-max 이상 → 빨간색으로 칠한다.
    # 모든 그림에 같은 기준을 쓰기 때문에 이미지끼리 색을 공정하게 비교할 수 있다.
    # (bottle 기준 정상 약 10~12, 불량 약 15~24 → 기본값 25면 불량이 빨갛게 보임)
    parser.add_argument(
        "--vis-max",
        type=float,
        default=25.0,
    )

    args = parser.parse_args()

    # 0 이하로 나누면 안 되므로 확인
    if args.vis_max <= 0:
        parser.error("--vis-max는 0보다 커야 합니다.")

    output_dir = args.output_dir

    # 상대경로면 레포 최상위 기준으로 해석
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    # GPU가 있으면 GPU, 없으면 CPU
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

    # 파일을 CPU로 읽은 뒤 (weights_only=True: 보안상 텐서만 읽음)
    saved = torch.load(
        memory_path,
        map_location="cpu",
        weights_only=True,
    )

    # 정상 대표 벡터 [K, 1024]만 꺼내서 GPU로
    memory_bank = saved["embeddings"].to(device)

    print("Memory Bank:", tuple(memory_bank.shape))

    # 2. Backbone 및 Feature Extractor 준비
    # 메모리 뱅크를 만들 때와 똑같은 전처리·CNN·patch 설정을 써야
    # 정상 벡터와 같은 기준으로 비교할 수 있다
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

        # 이미지 읽기 → RGB 통일 → 전처리 → [1, 3, 224, 224] → GPU
        with Image.open(image_path) as image:
            tensor = transform(
                image.convert("RGB")
            ).unsqueeze(0).to(device)

        # 기울기 계산 없이 추론 (학습하지 않으므로)
        with torch.inference_mode():
            # CNN 특징 추출
            # {"layer2": [1, 512, 28, 28], "layer3": [1, 1024, 14, 14]}
            feature_maps = backbone.extract(tensor)

            # Patch Feature 생성
            # [1, 784, 1024]
            patch_features = patch_extractor(feature_maps)

            # 최근접 이웃 거리 계산
            # patch마다 가장 가까운 정상 벡터까지의 거리 [1, 784]
            patch_scores = compute_patch_scores(
                patch_features,
                memory_bank,
            )

            # 이미지 단위 이상 점수
            # 784개 patch 점수 중 최댓값 = 가장 이상한 부분의 점수
            image_score = patch_scores.max().item()

            # Patch Score -> Anomaly Map
            # 28x28 점수 → 224x224 이상 지도 (가우시안 블러 sigma=4)
            anomaly_map = create_anomaly_map(
                patch_scores=patch_scores,
                grid_size=feature_maps["layer2"].shape[-2:],
                image_size=tensor.shape[-2:],
                sigma=4.0,
            )


        # 이미지 점수 출력
        print(f"\n[추론] {image_path}")
        print("Patch Score Shape:", tuple(patch_scores.shape))
        print("Anomaly Map Shape:", tuple(anomaly_map.shape))
        print(f"Image Anomaly Score: {image_score:.6f}")

        # 파일명이 겹치지 않도록 클래스 이름 포함
        # good/000.png -> good_000_anomaly.png
        # (MVTec은 폴더마다 000.png가 있어서 폴더 이름을 붙이지 않으면 서로 덮어쓴다)
        filename = (
            f"{image_path.parent.name}_"
            f"{image_path.stem}_anomaly.png"
        )

        output_path = output_dir / filename

        # [입력 이미지 | 히트맵 | 겹친 이미지]를 가로로 이어 붙여 저장
        # tensor[0], anomaly_map[0]: 배치 차원을 떼고 이미지 1장 분량만 전달
        save_anomaly_visualization(
            image_tensor=tensor[0],
            anomaly_map=anomaly_map[0],
            output_path=output_path,
            vmax=args.vis_max,
        )

        print("시각화 저장:", output_path)


    print("\n추론 완료!")


# 직접 실행했을 때만 main() 실행
if __name__ == "__main__":
    main()
