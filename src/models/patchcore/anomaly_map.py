
import math

import torch
import torch.nn.functional as F


@torch.inference_mode()
def create_anomaly_map(
    patch_scores,
    grid_size,
    image_size,
    sigma=4.0,
):
    """
    Patch Score를 픽셀 단위 Anomaly Map으로 변환한다.

    patch_scores: [B, N]
    grid_size: (H_patch, W_patch)
    image_size: (H_image, W_image)

    반환: [B, H_image, W_image]
    """

    height, width = grid_size

    if patch_scores.ndim != 2:
        raise ValueError("patch_scores는 [B, N]이어야 합니다.")

    if patch_scores.shape[1] != height * width:
        raise ValueError("Patch 개수와 격자 크기가 일치하지 않습니다.")

    # 1. [B, 784] -> [B, 1, 28, 28]
    batch = patch_scores.shape[0]

    anomaly_map = patch_scores.reshape(
        batch, 1, height, width
    )

    # 2. 원래 CNN 입력 해상도로 보간
    anomaly_map = F.interpolate(
        anomaly_map,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )

    # 3. Gaussian Smoothing - 
    if sigma > 0:
        radius = math.ceil(3 * sigma)

        coords = torch.arange(
            -radius, radius + 1,
            device=anomaly_map.device,
            dtype=anomaly_map.dtype,
        )

        kernel = torch.exp(
            -(coords ** 2) / (2 * sigma ** 2)
        )

        kernel = kernel / kernel.sum()

        # 가로 방향 필터링
        kernel_x = kernel.view(1, 1, 1, -1)

        anomaly_map = F.conv2d(
            F.pad(
                anomaly_map,
                (radius, radius, 0, 0),
                mode="reflect",
            ),
            kernel_x,
        )

        # 세로 방향 필터링
        kernel_y = kernel.view(1, 1, -1, 1)

        anomaly_map = F.conv2d(
            F.pad(
                anomaly_map,
                (0, 0, radius, radius),
                mode="reflect",
            ),
            kernel_y,
        )

    return anomaly_map.squeeze(1)
