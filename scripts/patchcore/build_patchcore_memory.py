
# [이 스크립트의 역할] PatchCore 1단계 - 정상 특징 수집
# 정상 학습 이미지(train/good) 전체에서 patch 벡터를 뽑아 하나의 .pt 파일로 저장한다.
#
# 실행 예:
#   python scripts/patchcore/build_patchcore_memory.py --category bottle
#   python scripts/patchcore/build_patchcore_memory.py --category bottle --limit 10   (빠른 테스트용)
#
# 결과: outputs/patchcore/<category>/raw_memory_full.pt  (bottle 기준 [163856, 1024], 약 640MB)
# 다음 단계: build_patchcore_coreset.py 로 이 파일을 줄인다.

from pathlib import Path
import argparse
import sys

import torch
from torchvision.models import Wide_ResNet50_2_Weights

# ROOT = 레포 최상위 폴더 (이 파일 위치에서 두 단계 위)
#   scripts/patchcore/build_patchcore_memory.py → parents[0]=patchcore, [1]=scripts, [2]=레포
ROOT = Path(__file__).resolve().parents[2]
# 어디서 실행하든 "from src. ..." import가 되도록 레포 최상위를 파이썬 검색 경로 맨 앞에 추가
sys.path.insert(0, str(ROOT))

from src.models.patchcore.backbone import PatchCoreBackbone
from src.models.patchcore.features import PatchFeatureExtractor
from src.models.patchcore.memory_bank import collect_normal_features


def main():
    # 명령줄 인자(--category 등)를 읽는 도구
    parser = argparse.ArgumentParser(
        description="PatchCore 정상 특징 후보 생성"
    )

    # --category: MVTec 카테고리 이름 (기본 bottle)
    parser.add_argument("--category", default="bottle")
    # --limit: 앞에서 몇 장만 쓸지 (디버깅용, 지정하지 않으면 전체 사용)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    # 1. 정상 학습 이미지
    # 예) C:\industrial-anomaly-research\data\raw\mvtec_ad\bottle\train\good
    train_dir = (
        ROOT / "data/raw/mvtec_ad"
        / args.category / "train/good"
    )

    # 폴더 안의 png 파일을 이름순으로 정렬 (실행할 때마다 순서가 같도록)
    image_paths = sorted(train_dir.glob("*.png"))

    if not image_paths:
        raise SystemExit("정상 학습 이미지를 찾을 수 없습니다.")

    # --limit을 줬다면 1 이상, 전체 장수 이하인지 확인한 뒤 앞에서부터 그만큼만 사용
    if args.limit is not None:
        if not 1 <= args.limit <= len(image_paths):
            parser.error("--limit 값이 올바르지 않습니다.")

        image_paths = image_paths[:args.limit]

    # 2. GPU 및 모델 준비
    # GPU가 있으면 GPU, 없으면 CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("사용 장치:", device)
    print("학습 정상 이미지:", len(image_paths))

    # 이미지 전처리 (크기 조정 → 가운데 224x224 자르기 → 정규화)
    # 테스트 때(model.py, run_patchcore.py)도 반드시 같은 전처리를 써야 한다
    weights = Wide_ResNet50_2_Weights.DEFAULT
    transform = weights.transforms()

    # CNN 특징 추출기 (WideResNet50의 layer2, layer3)
    backbone = PatchCoreBackbone(device)

    # 3x3 주변을 묶어 위치마다 1024차원 patch 벡터를 만드는 모듈
    patch_extractor = PatchFeatureExtractor(
        patch_size=3,
        pretrain_embed_dim=1024,
        target_embed_dim=1024,
    ).to(device)

    # 평가 모드 (학습 동작 끔)
    patch_extractor.eval()

    # 3. 정상 특징 수집
    # 모든 정상 이미지의 patch 벡터를 하나의 큰 표로 모은다
    # memory: [이미지 수 * 784, 1024], patches_per_image: 784
    memory, patches_per_image = collect_normal_features(
        image_paths=image_paths,
        backbone=backbone,
        patch_extractor=patch_extractor,
        transform=transform,
        device=device,
    )

    # 4. 결과 저장
    # 저장 폴더: outputs/patchcore/<category>/ (없으면 상위 폴더까지 만든다)
    output_dir = ROOT / "outputs/patchcore" / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    # 전체를 썼으면 "full", 일부만 썼으면 "debug_10"처럼 파일명에 표시
    suffix = (
        "full" if args.limit is None
        else f"debug_{args.limit}"
    )

    output_path = output_dir / f"raw_memory_{suffix}.pt"

    # 기존 결과를 실수로 덮어쓰지 않도록 보호
    # (다시 만들고 싶으면 기존 파일을 직접 지우거나 이름을 바꿔야 한다)
    if output_path.exists():
        raise SystemExit(
            f"이미 결과 파일이 존재합니다: {output_path}"
        )

    # 특징 벡터와 함께 "어떻게 만들었는지" 정보도 같이 저장
    # (나중에 파일만 보고도 실험 조건을 확인하고 재현할 수 있도록)
    torch.save({
        "embeddings": memory,                        # 정상 patch 벡터 전체 [N, 1024]
        "category": args.category,                   # 카테고리 이름
        "image_count": len(image_paths),             # 사용한 이미지 수
        "image_names": [p.name for p in image_paths],# 사용한 이미지 파일명 목록
        "patches_per_image": patches_per_image,      # 이미지당 patch 수 (784)
        "backbone": "wide_resnet50_2",               # 사용한 CNN
        "weights": weights.name,                     # 가중치 이름 (IMAGENET1K_V2)
        "transform": repr(transform),                # 전처리 설정을 글자로 기록
        "patch_size": 3,                             # 주변 묶음 크기
        "embedding_dim": memory.shape[1],            # 벡터 차원 (1024)
        "stage": "raw_candidates",                   # 아직 coreset 전 단계라는 표시
    }, output_path)

    print("\n=== 정상 특징 수집 완료 ===")
    print("Memory Shape:", tuple(memory.shape))
    # 메모리 크기 = 원소 개수 x 원소당 바이트(float32 = 4바이트) → MiB 단위로 변환
    print(
        "크기:",
        f"{memory.numel() * memory.element_size() / 1024**2:.1f} MiB"
    )
    print("저장 위치:", output_path)


# 이 파일을 직접 실행했을 때만 main()을 실행
# (다른 파일에서 import할 때는 실행되지 않음)
if __name__ == "__main__":
    main()
