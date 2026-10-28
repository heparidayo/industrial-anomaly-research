# CNN 불러오고 feature 추출용

# [이 파일의 역할]
# PatchCore는 신경망을 새로 학습하지 않는다.
# 대신 ImageNet으로 이미 학습된 CNN(WideResNet50)을 "눈"처럼 빌려 쓰고,
# 이미지를 넣었을 때 중간 계층에서 나오는 특징(feature map)만 꺼내 쓴다.
# 이 파일은 그 "특징을 꺼내는 장치"를 만드는 부분이다.


import torch

from torchvision.models import (
    wide_resnet50_2,          # 모델 구조 (WideResNet-50-2)
    Wide_ResNet50_2_Weights,  # 그 모델의 사전학습 가중치 정보
)

from torchvision.models.feature_extraction import (
    # 모델 중간 계층의 출력을 꺼낼 수 있게 모델을 감싸주는 도구
    create_feature_extractor,
)


class PatchCoreBackbone:
    """PatchCore에서 사용할 CNN 특징 추출기"""

    def __init__(self, device):
        # 연산할 장치 ("cuda" = GPU, "cpu" = CPU)
        self.device = device

        # DEFAULT = torchvision이 권장하는 최신 ImageNet 가중치 (IMAGENET1K_V2)
        # 처음 실행할 때 인터넷에서 자동으로 내려받고,
        # 이후에는 C:\Users\<사용자>\.cache\torch\hub\checkpoints 에 저장된 파일을 쓴다.
        weights = Wide_ResNet50_2_Weights.DEFAULT

        # 사전학습 가중치가 들어간 WideResNet50 전체 모델 생성
        backbone = wide_resnet50_2(weights=weights)

        # 모델 전체를 끝까지 돌리지 않고, 중간 계층 2개의 출력만 꺼내도록 설정
        #   - layer2: 비교적 얕은 층 → 질감, 모서리 같은 "세밀한" 정보
        #             224x224 입력 기준 출력 크기 [B, 512, 28, 28]
        #   - layer3: 더 깊은 층 → 모양, 부품 같은 "의미 있는" 정보
        #             224x224 입력 기준 출력 크기 [B, 1024, 14, 14]
        # layer1은 너무 단순하고, layer4는 ImageNet 분류에 치우쳐 있어서
        # PatchCore 논문에서는 이 중간 두 층을 섞어 쓰는 것이 가장 좋았다.
        # return_nodes의 왼쪽은 모델 안의 계층 이름, 오른쪽은 결과 딕셔너리에서 쓸 이름이다.
        self.extractor = create_feature_extractor(
            backbone,
            return_nodes={
                "layer2": "layer2",
                "layer3": "layer3",
            }
        )

        # 모델을 GPU(또는 CPU)로 옮긴다
        self.extractor = self.extractor.to(device)

        # 평가 모드로 전환
        # (BatchNorm 등이 학습 때처럼 값을 바꾸지 않고, 항상 같은 결과를 내도록 고정)
        self.extractor.eval()

    # inference_mode: 기울기(gradient) 계산을 완전히 끈다.
    # 학습을 하지 않으므로 필요 없고, 끄면 메모리를 덜 쓰고 더 빠르다.
    @torch.inference_mode()
    def extract(self, images):
        """이미지 배치에서 중간 계층 특징 추출"""
        # images: 전처리된 이미지 텐서 [B, 3, 224, 224]
        # 반환값: {"layer2": [B, 512, 28, 28], "layer3": [B, 1024, 14, 14]} 형태의 딕셔너리
        return self.extractor(images.to(self.device))
