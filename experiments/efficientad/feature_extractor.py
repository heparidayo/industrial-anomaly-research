"""
증류 정답(target)을 만드는 특징 추출기 Ψ — EfficientAD 논문 Appendix A.2.

논문은 PatchCore가 coreset에 넣는 최종 특징을 그대로 증류 대상으로 사용합니다.
PatchCore 기본 설정을 따라 다음과 같이 구성합니다.
  1) ImageNet 사전학습 WideResNet-101-2의 layer2, layer3 출력을 꺼냄
     (512×512 입력 → layer2: 512×64×64, layer3: 1024×32×32)
  2) 각 위치에서 3×3 이웃 패치를 펼침(unfold)        → C·9 차원
  3) 층마다 채널 축으로 adaptive average pooling     → 1024 차원
  4) layer3를 64×64로 bilinear 보간
  5) 두 층을 이어 붙인 2048 차원을 다시 pooling      → 384 차원
  결과: Ψ: R^{3×512×512} → R^{384×64×64}

구현 메모:
  PatchCore 원본은 layer3를 먼저 64×64로 보간한 뒤 채널 pooling을 합니다.
  두 연산 모두 선형이고 서로 다른 축(공간 / 채널)에 작용하므로 순서를 바꿔도
  결과가 같습니다. 여기서는 메모리를 아끼기 위해 32×32에서 먼저 pooling합니다.
  PatchCore 공식 구현과 세부 수치까지 완전히 같다고 보장하지는 않으므로,
  정확한 재현이 필요하면 공식 코드의 출력과 한 번 비교해 보세요.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _channel_pool(x: torch.Tensor, out_dim: int) -> torch.Tensor:
    """(B, C, H, W)의 채널 축을 adaptive average pooling으로 out_dim까지 축소."""
    b, c, h, w = x.shape
    x = x.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
    x = F.adaptive_avg_pool1d(x, out_dim)
    return x.reshape(b, h, w, out_dim).permute(0, 3, 1, 2)


def _patchify(x: torch.Tensor, patch_size: int = 3) -> torch.Tensor:
    """각 위치의 patch_size×patch_size 이웃을 채널 축으로 펼침: (B, C, H, W) → (B, C·k², H, W)."""
    b, c, h, w = x.shape
    x = F.unfold(x, kernel_size=patch_size, padding=patch_size // 2, stride=1)
    return x.reshape(b, c * patch_size * patch_size, h, w)


class PatchCoreFeatureExtractor(nn.Module):
    def __init__(self, out_channels: int = 384, pretrain_embed_dim: int = 1024,
                 patch_size: int = 3, pretrained: bool = True):
        super().__init__()
        weights = (torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V1
                   if pretrained else None)
        wrn = torchvision.models.wide_resnet101_2(weights=weights)
        # layer3까지만 사용 (layer4, avgpool, fc는 필요 없음)
        self.stem = nn.Sequential(wrn.conv1, wrn.bn1, wrn.relu, wrn.maxpool, wrn.layer1)
        self.layer2 = wrn.layer2
        self.layer3 = wrn.layer3
        self.out_channels = out_channels
        self.pretrain_embed_dim = pretrain_embed_dim
        self.patch_size = patch_size
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        # BatchNorm 통계가 바뀌지 않도록 항상 eval 모드 유지
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.stem(x)
        f2 = self.layer2(f1)                      # (B, 512, 64, 64)  @ 512 입력
        f3 = self.layer3(f2)                      # (B, 1024, 32, 32)

        f2 = _channel_pool(_patchify(f2, self.patch_size), self.pretrain_embed_dim)
        f3 = _channel_pool(_patchify(f3, self.patch_size), self.pretrain_embed_dim)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)

        feats = torch.stack([f2, f3], dim=1)      # (B, 2, 1024, H, W)
        b, _, _, h, w = feats.shape
        feats = feats.permute(0, 3, 4, 1, 2).reshape(b * h * w, 1, -1)   # 층 순서대로 이어 붙임
        feats = F.adaptive_avg_pool1d(feats, self.out_channels)
        return feats.reshape(b, h, w, self.out_channels).permute(0, 3, 1, 2).contiguous()


if __name__ == "__main__":
    # 가중치 다운로드 없이 출력 형태만 확인
    psi = PatchCoreFeatureExtractor(pretrained=False)
    with torch.no_grad():
        y = psi(torch.randn(2, 3, 512, 512))
    print("Ψ 출력:", tuple(y.shape))   # (2, 384, 64, 64)
    assert y.shape == (2, 384, 64, 64)
