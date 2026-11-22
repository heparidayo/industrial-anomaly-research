"""
EfficientAD 오토인코더 — 논문 Table 8 (EfficientAD-S, M 공용).

인코더: stride 2의 4×4 합성곱 5개(256 → 8) + 8×8 합성곱으로 64차원 1×1 잠재 벡터
디코더: "bilinear 업샘플 → 4×4 합성곱(padding 2, 크기 +1) → ReLU → Dropout 0.2"
        를 6번 반복해 1 → 128까지 키운 뒤, out_size로 리사이즈하고
        3×3 합성곱 두 개로 Teacher와 같은 384채널을 출력

업샘플 목표 크기(3, 8, 15, 32, 63, 127)가 홀짝이 섞인 이유는, 뒤따르는
padding 2의 4×4 합성곱이 크기를 1씩 늘리기 때문입니다(3→4, 8→9, ..., 127→128).

입력 크기는 256이 기본이며, 다른 크기도 받을 수 있습니다(잠재 벡터 직전에 8×8로 맞춤).
out_size는 Teacher 출력 크기와 같아야 합니다.
  - Teacher padding=True  → 64 (논문 Table 8 그대로)
  - Teacher padding=False → 56

실행:  python autoencoder.py   (각 층 출력 크기와 파라미터 수 확인)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 64
DECODER_SIZES = (3, 8, 15, 32, 63, 127)


class AutoEncoder(nn.Module):
    def __init__(self, out_channels: int = 384, out_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.out_size = out_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        # 256 입력이면 여기까지 8×8. 다른 입력 크기(예: 432 → 13×13)도 8×8로 맞춘 뒤 압축해
        # 잠재 벡터를 항상 64차원 1×1로 유지 (256에서는 아무 변화 없음, 가중치 형식도 동일)
        self.to_latent = nn.Conv2d(64, LATENT_DIM, 8, stride=1, padding=0)  # 8×8 → 1×1
        self.dec_convs = nn.ModuleList(
            [nn.Conv2d(LATENT_DIM if i == 0 else 64, 64, 4, stride=1, padding=2) for i in range(6)])
        self.dropout = nn.Dropout(dropout)
        self.dec_final = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if z.shape[-1] != 8 or z.shape[-2] != 8:
            z = F.adaptive_avg_pool2d(z, 8)
        z = self.to_latent(z)                                          # (B, 64, 1, 1)
        for size, conv in zip(DECODER_SIZES, self.dec_convs):
            z = F.interpolate(z, size=(size, size), mode="bilinear", align_corners=False)
            z = self.dropout(F.relu(conv(z)))
        z = F.interpolate(z, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return self.dec_final(z)


def get_autoencoder(out_size: int = 64) -> AutoEncoder:
    return AutoEncoder(out_channels=384, out_size=out_size)


if __name__ == "__main__":
    ae = get_autoencoder().eval()
    x = torch.randn(1, 3, 256, 256)
    z = ae.to_latent(ae.encoder(x))
    print("잠재 벡터:", tuple(z.shape))
    for size, conv in zip(DECODER_SIZES, ae.dec_convs):
        z = conv(F.interpolate(z, size=(size, size), mode="bilinear", align_corners=False))
        print(f"  업샘플 {size:3d} → 합성곱 후 {tuple(z.shape[-2:])}")
    y = ae(x)
    n = sum(p.numel() for p in ae.parameters())
    print("출력:", tuple(y.shape), f"파라미터 {n / 1e6:.2f}M")
    assert y.shape == (1, 384, 64, 64)
    assert get_autoencoder(56)(x).shape == (1, 384, 56, 56)
    print("out_size=56 (Teacher 무패딩용): OK")
    big = get_autoencoder(108)(torch.randn(1, 3, 432, 432))
    assert big.shape == (1, 384, 108, 108)
    print("432 입력 → 108×108 출력 (4단 적층 원본 해상도용): OK")
