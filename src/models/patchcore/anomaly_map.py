
# [이 파일의 역할]
# scoring.py가 계산한 patch별 점수(28x28 = 784개)를
# 원래 입력 이미지 크기(224x224)의 "이상 지도(Anomaly Map)"로 바꾼다.
#
# 이 지도를 색으로 칠하면 히트맵이 되고,
# "이미지의 어느 부분이 이상하다고 판단했는지"를 눈으로 확인할 수 있다.
# (이미지 1장의 최종 점수 자체는 이 지도가 아니라 patch 점수의 최댓값으로 정한다)

import math

import torch
import torch.nn.functional as F


# inference_mode: 기울기 계산을 끈다
@torch.inference_mode()
def create_anomaly_map(
    patch_scores,  # [B, N] patch별 점수 (예: [1, 784])
    grid_size,     # patch 격자 크기 (H_patch, W_patch) = (28, 28)
    image_size,    # 출력 지도 크기 (H_image, W_image) = (224, 224)
    sigma=4.0,     # 가우시안 블러 강도 (클수록 더 부드럽게 퍼진다, 논문 값 4)
):
    """
    Patch Score를 픽셀 단위 Anomaly Map으로 변환한다.

    patch_scores: [B, N]
    grid_size: (H_patch, W_patch)
    image_size: (H_image, W_image)

    반환: [B, H_image, W_image]
    """

    height, width = grid_size

    # ----- 입력값 검사 -----
    if patch_scores.ndim != 2:
        raise ValueError("patch_scores는 [B, N]이어야 합니다.")

    # 784개 점수를 28x28 격자로 정확히 다시 배치할 수 있어야 한다
    if patch_scores.shape[1] != height * width:
        raise ValueError("Patch 개수와 격자 크기가 일치하지 않습니다.")

    # 1. [B, 784] -> [B, 1, 28, 28]
    # 한 줄로 늘어선 점수를 다시 2D 격자로 되돌린다
    # 가운데 1은 채널 수 (흑백 이미지 1장처럼 취급하기 위함)
    batch = patch_scores.shape[0]

    anomaly_map = patch_scores.reshape(
        batch, 1, height, width
    )

    # 2. 원래 CNN 입력 해상도로 보간
    # 28x28 → 224x224로 부드럽게 확대 (bilinear = 이웃 값을 선형으로 섞어 채움)
    anomaly_map = F.interpolate(
        anomaly_map,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )

    # 3. Gaussian Smoothing -
    # 확대한 지도에는 격자 모양 경계가 남아 있어서, 가우시안 블러로 부드럽게 만든다.
    # 잡음 같은 튀는 값도 줄어들어 결함 영역이 더 자연스럽게 보인다.
    # (sigma=0이면 블러를 건너뛴다)
    if sigma > 0:
        # 블러 필터의 반경. 가우시안은 3*sigma 바깥은 거의 0이라 거기까지만 쓴다
        # sigma=4 → radius=12 → 필터 길이 25
        radius = math.ceil(3 * sigma)

        # -12, -11, ..., 0, ..., 11, 12 (필터 각 칸의 중심으로부터의 거리)
        coords = torch.arange(
            -radius, radius + 1,
            device=anomaly_map.device,
            dtype=anomaly_map.dtype,
        )

        # 가우시안(종 모양) 가중치: 가운데가 가장 크고 멀어질수록 작아진다
        kernel = torch.exp(
            -(coords ** 2) / (2 * sigma ** 2)
        )

        # 가중치 합이 1이 되게 정규화 (블러 후에도 점수 크기가 유지되도록)
        kernel = kernel / kernel.sum()

        # 2D 블러를 가로 1번 + 세로 1번으로 나눠서 한다.
        # 가우시안은 이렇게 나눠도 결과가 같고, 계산량은 훨씬 적다 (25x25 대신 25+25).

        # 가로 방향 필터링
        # [1, 1, 1, 25] 모양의 가로 필터
        kernel_x = kernel.view(1, 1, 1, -1)

        # 좌우를 12칸씩 반사(reflect) 패딩한 뒤 필터 적용
        # 반사 패딩: 가장자리 바깥을 안쪽 값을 거울처럼 비춰 채움
        # (0으로 채우면 가장자리 점수가 인위적으로 낮아지기 때문)
        anomaly_map = F.conv2d(
            F.pad(
                anomaly_map,
                (radius, radius, 0, 0),
                mode="reflect",
            ),
            kernel_x,
        )

        # 세로 방향 필터링
        # [1, 1, 25, 1] 모양의 세로 필터, 위아래를 12칸씩 반사 패딩
        kernel_y = kernel.view(1, 1, -1, 1)

        anomaly_map = F.conv2d(
            F.pad(
                anomaly_map,
                (0, 0, radius, radius),
                mode="reflect",
            ),
            kernel_y,
        )

    # 채널 차원(1)을 떼고 반환: [B, 224, 224]
    return anomaly_map.squeeze(1)
