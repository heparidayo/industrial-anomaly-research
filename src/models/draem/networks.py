"""DRAEM 재구성 / 판별 서브네트워크.

원본 구현(VitjanZ/DRAEM, model_unet.py)과 같은 구조입니다.
- ReconstructiveSubNetwork: 결함이 합성된 이미지를 받아 정상처럼 복원 (3 → 3채널)
- DiscriminativeSubNetwork: [복원 이미지, 입력 이미지]를 받아 결함 마스크 예측 (6 → 2채널)
"""

import torch
import torch.nn as nn


def _double_conv(in_ch: int, out_ch: int, mid_ch: int | None = None) -> nn.Sequential:
    mid_ch = out_ch if mid_ch is None else mid_ch
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(mid_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _up_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def weights_init(module: nn.Module) -> None:
    """원본과 같은 초기화: Conv N(0, 0.02), BatchNorm N(1, 0.02)."""
    name = module.__class__.__name__
    if "Conv" in name:
        module.weight.data.normal_(0.0, 0.02)
    elif "BatchNorm" in name:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


# ── 재구성 서브네트워크 ─────────────────────────────────────

class EncoderReconstructive(nn.Module):
    def __init__(self, in_channels: int, base_width: int):
        super().__init__()
        bw = base_width
        self.block1 = _double_conv(in_channels, bw)
        self.block2 = _double_conv(bw, bw * 2)
        self.block3 = _double_conv(bw * 2, bw * 4)
        self.block4 = _double_conv(bw * 4, bw * 8)
        self.block5 = _double_conv(bw * 8, bw * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(self.pool(b1))
        b3 = self.block3(self.pool(b2))
        b4 = self.block4(self.pool(b3))
        b5 = self.block5(self.pool(b4))
        return b5


class DecoderReconstructive(nn.Module):
    def __init__(self, base_width: int, out_channels: int):
        super().__init__()
        bw = base_width
        self.up1 = _up_conv(bw * 8, bw * 8)
        self.db1 = _double_conv(bw * 8, bw * 4, mid_ch=bw * 8)
        self.up2 = _up_conv(bw * 4, bw * 4)
        self.db2 = _double_conv(bw * 4, bw * 2, mid_ch=bw * 4)
        self.up3 = _up_conv(bw * 2, bw * 2)
        self.db3 = _double_conv(bw * 2, bw, mid_ch=bw * 2)
        self.up4 = _up_conv(bw, bw)
        self.db4 = _double_conv(bw, bw)
        self.fin_out = nn.Conv2d(bw, out_channels, kernel_size=3, padding=1)

    def forward(self, b5):
        x = self.db1(self.up1(b5))
        x = self.db2(self.up2(x))
        x = self.db3(self.up3(x))
        x = self.db4(self.up4(x))
        return self.fin_out(x)


class ReconstructiveSubNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_width: int = 128):
        super().__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width)
        self.decoder = DecoderReconstructive(base_width, out_channels)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── 판별 서브네트워크 ───────────────────────────────────────

class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels: int, base_width: int):
        super().__init__()
        bw = base_width
        self.block1 = _double_conv(in_channels, bw)
        self.block2 = _double_conv(bw, bw * 2)
        self.block3 = _double_conv(bw * 2, bw * 4)
        self.block4 = _double_conv(bw * 4, bw * 8)
        self.block5 = _double_conv(bw * 8, bw * 8)
        self.block6 = _double_conv(bw * 8, bw * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(self.pool(b1))
        b3 = self.block3(self.pool(b2))
        b4 = self.block4(self.pool(b3))
        b5 = self.block5(self.pool(b4))
        b6 = self.block6(self.pool(b5))
        return b1, b2, b3, b4, b5, b6


class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width: int, out_channels: int):
        super().__init__()
        bw = base_width
        self.up_b = _up_conv(bw * 8, bw * 8)
        self.db_b = _double_conv(bw * (8 + 8), bw * 8)
        self.up1 = _up_conv(bw * 8, bw * 4)
        self.db1 = _double_conv(bw * (4 + 8), bw * 4)
        self.up2 = _up_conv(bw * 4, bw * 2)
        self.db2 = _double_conv(bw * (2 + 4), bw * 2)
        self.up3 = _up_conv(bw * 2, bw)
        self.db3 = _double_conv(bw * (1 + 2), bw)
        self.up4 = _up_conv(bw, bw)
        self.db4 = _double_conv(bw * 2, bw)
        self.fin_out = nn.Conv2d(bw, out_channels, kernel_size=3, padding=1)

    def forward(self, b1, b2, b3, b4, b5, b6):
        x = self.db_b(torch.cat((self.up_b(b6), b5), dim=1))
        x = self.db1(torch.cat((self.up1(x), b4), dim=1))
        x = self.db2(torch.cat((self.up2(x), b3), dim=1))
        x = self.db3(torch.cat((self.up3(x), b2), dim=1))
        x = self.db4(torch.cat((self.up4(x), b1), dim=1))
        return self.fin_out(x)


class DiscriminativeSubNetwork(nn.Module):
    def __init__(self, in_channels: int = 6, out_channels: int = 2, base_channels: int = 64):
        super().__init__()
        self.encoder_segment = EncoderDiscriminative(in_channels, base_channels)
        self.decoder_segment = DecoderDiscriminative(base_channels, out_channels)

    def forward(self, x):
        return self.decoder_segment(*self.encoder_segment(x))


def build_networks(rec_base_width: int = 128, seg_base_width: int = 64):
    """학습/추론에서 같은 구성을 쓰도록 생성 함수를 하나로 둡니다."""
    reconstructive = ReconstructiveSubNetwork(3, 3, rec_base_width)
    discriminative = DiscriminativeSubNetwork(6, 2, seg_base_width)
    return reconstructive, discriminative
