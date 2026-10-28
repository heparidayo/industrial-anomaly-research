
# [이 파일의 역할]
# 지금까지의 부품(backbone, features, scoring, anomaly_map)을 하나로 묶은
# "PatchCore 추론기" 클래스.
#
# 사용 방법:
#   model = PatchCorePredictor("coreset_...pt")  # 메모리 뱅크와 CNN을 한 번만 준비
#   result = model.predict("이미지.png")         # 이미지마다 호출
#   result["image_score"]                        # 이미지 이상 점수
#
# 준비(모델 로딩)를 한 번만 하고 여러 이미지에 재사용하므로,
# 테스트셋 전체를 돌리는 run_patchcore_batch.py에서 이 클래스를 쓴다.

from pathlib import Path

import torch
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights

# 같은 폴더(patchcore 패키지)의 부품들을 불러온다
from .backbone import PatchCoreBackbone
from .features import PatchFeatureExtractor
from .scoring import compute_patch_scores
from .anomaly_map import create_anomaly_map


class PatchCorePredictor:
    """저장된 Memory Bank를 이용한 PatchCore 추론기"""

    def __init__(self, memory_path, device=None):
        # memory_path: build_patchcore_coreset.py가 만든 coreset .pt 파일 경로
        # device: None이면 GPU가 있으면 "cuda", 없으면 "cpu"를 자동 선택
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 저장된 메모리 뱅크 파일 읽기
        # map_location="cpu": 일단 CPU로 읽는다 (저장한 PC와 장치가 달라도 안전하게)
        # weights_only=True: 텐서와 기본 값만 읽고, 임의의 파이썬 코드는 실행하지 않는다 (보안)
        saved = torch.load(
            Path(memory_path),
            map_location="cpu",
            weights_only=True,
        )

        # 어떤 카테고리(예: "bottle")로 만든 메모리 뱅크인지 기록
        # (run_patchcore_batch.py에서 테스트 카테고리와 일치하는지 확인하는 데 쓴다)
        self.category = saved.get("category")

        # 정상 대표 벡터 [M, 1024]를 GPU로 올린다
        # 추론 때마다 옮기면 느리므로 처음에 한 번만 올려둔다
        self.memory_bank = saved["embeddings"].to(self.device)

        # 이미지 전처리: 메모리 뱅크를 만들 때와 반드시 똑같아야 한다
        # WideResNet50 기본 전처리 = 크기 조정(짧은 변 232) → 가운데 224x224 자르기
        #                          → 0~1 변환 → ImageNet 평균/표준편차로 정규화
        # (가운데 자르기 때문에 원본 가장자리 약 3%는 모델이 보지 못한다)
        weights = Wide_ResNet50_2_Weights.DEFAULT
        self.transform = weights.transforms()

        # CNN 특징 추출기 (layer2, layer3)
        self.backbone = PatchCoreBackbone(self.device)

        # patch 벡터 생성기
        # 설정값은 메모리 뱅크를 만들 때(build_patchcore_memory.py)와 같아야
        # 정상 벡터와 테스트 벡터를 같은 기준으로 비교할 수 있다
        self.patch_extractor = PatchFeatureExtractor(
            patch_size=3,
            pretrain_embed_dim=1024,
            target_embed_dim=1024,
        ).to(self.device)

        self.patch_extractor.eval()

    @torch.inference_mode()
    def predict(self, image_path, with_map=False):
        """한 장의 이미지를 추론한다."""
        # with_map=True면 히트맵용 이상 지도도 함께 만든다 (조금 더 느려짐)

        # 1. 이미지 읽기 → RGB 통일 → 전처리 → 배치 차원 추가 [1, 3, 224, 224] → GPU
        with Image.open(image_path) as image:
            tensor = self.transform(
                image.convert("RGB")
            ).unsqueeze(0).to(self.device)

        # 2. CNN 특징 추출 → patch 벡터 [1, 784, 1024]
        feature_maps = self.backbone.extract(tensor)
        patch_features = self.patch_extractor(feature_maps)

        # 3. patch마다 가장 가까운 정상 벡터까지의 거리 → [1, 784]
        patch_scores = compute_patch_scores(
            patch_features,
            self.memory_bank,
        )

        result = {
            # 이미지 이상 점수 = 784개 patch 점수 중 최댓값
            # "한 군데라도 많이 이상하면 이미지 전체가 이상"이라는 기준
            # (작은 결함 하나도 놓치지 않기 위함. 대신 한 곳만 튀어도 점수가 높아진다)
            # .item(): 텐서 1개짜리 값을 일반 파이썬 숫자로 변환
            "image_score": patch_scores.max().item(),
            # 이미지당 patch 수 (784) - 확인용
            "patch_count": patch_scores.shape[1],
        }

        if with_map:
            # 4. (선택) patch 점수 28x28 → 224x224 이상 지도
            # grid_size: layer2 특징 맵의 높이·너비 (28, 28)
            # image_size: 전처리된 입력 이미지의 높이·너비 (224, 224)
            anomaly_map = create_anomaly_map(
                patch_scores,
                grid_size=feature_maps["layer2"].shape[-2:],
                image_size=tensor.shape[-2:],
                sigma=4.0,
            )

            # 시각화용으로 CPU에 옮겨 저장 (배치 차원 [0]을 떼서 이미지 1장 분량만)
            result["anomaly_map"] = anomaly_map[0].cpu()
            # 정규화된 입력 이미지 (히트맵과 겹쳐 그릴 때 사용)
            result["image_tensor"] = tensor[0].cpu()

        return result
