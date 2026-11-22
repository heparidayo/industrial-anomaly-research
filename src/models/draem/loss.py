"""DRAEM 손실 함수.

전체 손실 = L2(복원) + SSIM(복원) + Focal(마스크)   — 원본과 동일한 1:1:1 합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    window_2d = g @ g.t()
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    """1 - SSIM. 입력 범위는 [0, 1]로 가정 (원본도 이 범위에서 L=1로 계산됨)."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, img1, img2):
        channels = img1.shape[1]
        window = _gaussian_window(self.window_size, self.sigma, channels, img1.device, img1.dtype)
        pad = self.window_size // 2

        mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1.0 - ssim_map.mean()


class FocalLoss(nn.Module):
    """판별 네트워크용 Focal Loss.

    probs:  (N, 2, H, W) softmax 확률
    target: (N, 1, H, W) 결함 마스크 (0/1)
    """

    def __init__(self, gamma: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, probs, target):
        num_class = probs.shape[1]
        probs = probs.permute(0, 2, 3, 1).reshape(-1, num_class)
        target = target.reshape(-1).long()

        one_hot = F.one_hot(target, num_class).to(probs.dtype)
        one_hot = one_hot.clamp(self.smooth / (num_class - 1), 1.0 - self.smooth)

        pt = (one_hot * probs).sum(dim=1) + self.smooth
        loss = -torch.pow(1.0 - pt, self.gamma) * pt.log()
        return loss.mean()
