# patch fature 만들기

# https://github.com/amazon-science/patchcore-inspection/blob/main/sample_training.sh

# [이 파일의 역할]
# backbone.py가 꺼낸 두 계층의 특징 맵(layer2, layer3)을 받아서
# 이미지의 각 위치(patch)마다 "이 부분은 이렇게 생겼다"를 나타내는
# 1024차원 벡터 하나씩을 만든다.
#
# 224x224 이미지 1장 → 28x28 = 784개 위치 → 784개의 1024차원 벡터
# 이 벡터들이 PatchCore에서 비교의 기본 단위가 된다.
#
# 전체 흐름 (이미지 1장, 224x224 입력 기준):
#   layer2 [512, 28, 28]  --3x3 주변 묶기-->  위치마다 512*9  = 4608차원
#   layer3 [1024, 14, 14] --3x3 주변 묶기-->  위치마다 1024*9 = 9216차원 --28x28로 늘리기
#   각각 1024차원으로 줄이기 → 이어붙여 2048차원 → 최종 1024차원으로 줄이기
#   결과: [1, 784, 1024]

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchFeatureExtractor(nn.Module):
    """PatchCore의 지역 Patch Feature 생성 및 결합"""

    def __init__(
        self,
        patch_size=3,            # 한 위치의 주변을 몇 x 몇 칸까지 묶을지 (3 → 3x3)
        pretrain_embed_dim=1024, # 계층별 특징을 1차로 줄일 차원
        target_embed_dim=1024,   # 최종 patch 벡터 차원
    ):
        super().__init__()

        # 주변을 가운데 기준으로 대칭이 되게 묶으려면 홀수여야 한다 (1, 3, 5, ...)
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size는 양의 홀수여야 합니다.")

        self.patch_size = patch_size
        self.pretrain_embed_dim = pretrain_embed_dim
        self.target_embed_dim = target_embed_dim

        # 각 위치에서 3x3 주변 특징 추출
        # nn.Unfold: 특징 맵 위를 3x3 창으로 한 칸씩(stride=1) 훑으며
        #            각 위치의 3x3 이웃 값을 한 줄로 펼쳐 모아준다.
        # padding=1(3//2): 가장자리 위치도 3x3 창이 들어가도록 바깥을 0으로 한 칸 채운다.
        #                  덕분에 출력 위치 개수가 입력과 같게 유지된다 (28x28 → 28x28).
        # 왜 주변까지 묶나? 한 칸만 보면 너무 좁아서, 이웃 정보를 함께 보면
        # 작은 위치 흔들림에도 덜 민감하고 결함의 맥락을 더 잘 잡는다.
        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=1,
            padding=patch_size // 2,
        )

    def _patchify(self, feature):
        """[B,C,H,W] -> [B,H,W,C,P,P]"""
        # B=배치(이미지 수), C=채널 수, H/W=특징 맵 높이/너비, P=patch_size(3)

        batch, channels, height, width = feature.shape

        # unfold 결과: [B, C*P*P, H*W]
        # (위치 H*W개 각각에 대해 C*3*3개의 값을 한 줄로 모은 것)
        patches = self.unfold(feature)

        # transpose(1, 2): [B, H*W, C*P*P] → "위치"가 앞으로 오게 순서 변경
        # reshape: 위치를 다시 (H, W) 격자로, 값들을 (C, P, P)로 풀어서
        #          [B, H, W, C, P, P] 모양으로 만든다
        return patches.transpose(1, 2).reshape(
            batch,
            height,
            width,
            channels,
            self.patch_size,
            self.patch_size,
        )

    # nn.Module에서는 model(x)처럼 호출하면 이 forward가 실행된다
    def forward(self, feature_maps):
        # feature_maps: backbone.extract()의 결과 딕셔너리

        # 1. 계층별 주변 Patch 특징 추출
        # patch2: [B, 28, 28, 512, 3, 3]
        # patch3: [B, 14, 14, 1024, 3, 3]
        patch2 = self._patchify(feature_maps["layer2"])
        patch3 = self._patchify(feature_maps["layer3"])

        # 모양 정보를 변수로 꺼내둔다 (h2=w2=28, c2=512, p=3 / h3=w3=14, c3=1024)
        batch, h2, w2, c2, p, _ = patch2.shape
        batch3, h3, w3, c3, _, _ = patch3.shape

        # 두 계층은 같은 이미지에서 나왔으므로 배치 크기가 같아야 정상
        if batch != batch3:
            raise ValueError("계층 간 배치 크기가 다릅니다.")

        # 2. Layer3 Patch 격자를 Layer2 크기로 맞춤
        # layer3는 14x14, layer2는 28x28이라 위치 개수가 다르다.
        # 위치끼리 1:1로 짝지으려면 layer3를 28x28로 늘려야 한다.

        # (a) 늘리기(interpolate)는 [N, 1, H, W] 모양의 2D 이미지에 적용해야 하므로
        #     [B, 14, 14, C, P, P] → [B, C, P, P, 14, 14] 로 순서를 바꾸고
        #     → [B*C*P*P, 1, 14, 14] 로 "작은 흑백 이미지 여러 장"처럼 펼친다
        patch3 = patch3.permute(
            0, 3, 4, 5, 1, 2
        ).reshape(-1, 1, h3, w3)

        # (b) 각 "작은 이미지"를 14x14 → 28x28로 부드럽게(bilinear) 확대
        patch3 = F.interpolate(
            patch3,
            size=(h2, w2),
            mode="bilinear",
            align_corners=False,
        )

        # (c) 원래 순서로 되돌림: [B, C, P, P, 28, 28] → [B, 28, 28, C, P, P]
        #     이제 patch2와 위치 격자가 똑같이 28x28이다
        patch3 = patch3.reshape(
            batch, c3, p, p, h2, w2
        ).permute(0, 4, 5, 1, 2, 3)

        # 3. Patch별 벡터로 변환
        # 모든 위치를 한 줄로 늘어놓고, 위치마다 값들을 1차원 벡터로 편다
        # vec2: [B*784, 512*9]  = [B*784, 4608]
        # vec3: [B*784, 1024*9] = [B*784, 9216]
        vec2 = patch2.reshape(-1, c2 * p * p)
        vec3 = patch3.reshape(-1, c3 * p * p)

        # 4. 각 계층의 특징을 1024차원으로 집계
        # adaptive_avg_pool1d: 긴 벡터를 1024개 구간으로 나눠 구간별 평균을 낸다.
        #   예) 4608차원 → 1024개 구간 → 1024차원
        # 학습 가능한 가중치가 없는 단순 평균이라, 학습 없이도 차원을 줄일 수 있다.
        # unsqueeze(1)/squeeze(1)는 이 함수가 요구하는 [N, 1, 길이] 모양을 맞추려고
        # 차원을 하나 잠깐 붙였다 떼는 것이다.
        vec2 = F.adaptive_avg_pool1d(
            vec2.unsqueeze(1),
            self.pretrain_embed_dim,
        ).squeeze(1)

        vec3 = F.adaptive_avg_pool1d(
            vec3.unsqueeze(1),
            self.pretrain_embed_dim,
        ).squeeze(1)

        # 5. 두 계층의 특징 결합
        # 같은 위치의 layer2 정보(세밀함)와 layer3 정보(의미)를 이어붙인다
        # [B*784, 1024] + [B*784, 1024] → [B*784, 2048]
        combined = torch.cat((vec2, vec3), dim=1)

        # 6. 최종 특징 차원 집계
        # 2048차원 → 1024차원으로 다시 구간 평균
        embeddings = F.adaptive_avg_pool1d(
            combined.unsqueeze(1),
            self.target_embed_dim,
        ).squeeze(1)

        # 이미지별로 다시 묶어서 반환: [B, 784, 1024]
        # = 이미지 B장, 각 이미지마다 784개 위치, 위치마다 1024차원 벡터
        return embeddings.reshape(
            batch, h2 * w2, self.target_embed_dim
        )
