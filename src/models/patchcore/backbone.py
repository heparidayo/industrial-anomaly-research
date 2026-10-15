# CNN 불러오고 feature 추출용


import torch

from torchvision.models import (
    wide_resnet50_2,
    Wide_ResNet50_2_Weights,
)

from torchvision.models.feature_extraction import (
    create_feature_extractor,
)


class PatchCoreBackbone:
    """PatchCore에서 사용할 CNN 특징 추출기"""

    def __init__(self, device):
        self.device = device

        weights = Wide_ResNet50_2_Weights.DEFAULT

        backbone = wide_resnet50_2(weights=weights)

        self.extractor = create_feature_extractor(
            backbone,
            return_nodes={
                "layer2": "layer2",
                "layer3": "layer3",
            }
        )

        self.extractor = self.extractor.to(device)
        self.extractor.eval()

    @torch.inference_mode()
    def extract(self, images):
        """이미지 배치에서 중간 계층 특징 추출"""
        return self.extractor(images.to(self.device))
