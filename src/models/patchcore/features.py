# patch fature 만들기

# https://github.com/amazon-science/patchcore-inspection/blob/main/sample_training.sh

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchFeatureExtractor(nn.Module):
    """PatchCore의 지역 Patch Feature 생성 및 결합"""

    def __init__(
        self,
        patch_size=3,
        pretrain_embed_dim=1024,
        target_embed_dim=1024,
    ):
        super().__init__()

        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size는 양의 홀수여야 합니다.")

        self.patch_size = patch_size
        self.pretrain_embed_dim = pretrain_embed_dim
        self.target_embed_dim = target_embed_dim

        # 각 위치에서 3x3 주변 특징 추출
        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=1,
            padding=patch_size // 2,
        )

    def _patchify(self, feature):
        """[B,C,H,W] -> [B,H,W,C,P,P]"""

        batch, channels, height, width = feature.shape

        patches = self.unfold(feature)

        return patches.transpose(1, 2).reshape(
            batch,
            height,
            width,
            channels,
            self.patch_size,
            self.patch_size,
        )

    def forward(self, feature_maps):
        # 1. 계층별 주변 Patch 특징 추출
        patch2 = self._patchify(feature_maps["layer2"])
        patch3 = self._patchify(feature_maps["layer3"])

        batch, h2, w2, c2, p, _ = patch2.shape
        batch3, h3, w3, c3, _, _ = patch3.shape

        if batch != batch3:
            raise ValueError("계층 간 배치 크기가 다릅니다.")

        # 2. Layer3 Patch 격자를 Layer2 크기로 맞춤
        patch3 = patch3.permute(
            0, 3, 4, 5, 1, 2
        ).reshape(-1, 1, h3, w3)

        patch3 = F.interpolate(
            patch3,
            size=(h2, w2),
            mode="bilinear",
            align_corners=False,
        )

        patch3 = patch3.reshape(
            batch, c3, p, p, h2, w2
        ).permute(0, 4, 5, 1, 2, 3)

        # 3. Patch별 벡터로 변환
        vec2 = patch2.reshape(-1, c2 * p * p)
        vec3 = patch3.reshape(-1, c3 * p * p)

        # 4. 각 계층의 특징을 1024차원으로 집계
        vec2 = F.adaptive_avg_pool1d(
            vec2.unsqueeze(1),
            self.pretrain_embed_dim,
        ).squeeze(1)

        vec3 = F.adaptive_avg_pool1d(
            vec3.unsqueeze(1),
            self.pretrain_embed_dim,
        ).squeeze(1)

        # 5. 두 계층의 특징 결합
        combined = torch.cat((vec2, vec3), dim=1)

        # 6. 최종 특징 차원 집계
        embeddings = F.adaptive_avg_pool1d(
            combined.unsqueeze(1),
            self.target_embed_dim,
        ).squeeze(1)

        return embeddings.reshape(
            batch, h2 * w2, self.target_embed_dim
        )
