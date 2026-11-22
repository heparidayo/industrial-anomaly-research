"""
EfficientAD Student — 논문 Section 3.2, 3.3, Algorithm 1.

Student는 Teacher와 같은 PDN 구조에 출력 채널만 768개입니다.
  - 채널   0~383 : Teacher 모방  → 로컬 이상 맵 (구조적 이상)
  - 채널 384~767 : 오토인코더 모방 → 글로벌 이상 맵 (논리적 이상)

이 파일에는 Student 생성, Teacher 출력 정규화 통계, 두 가지 손실
(hard feature loss, pretraining penalty), 로컬 이상 맵 계산이 들어 있습니다.
오토인코더 관련 손실(L_AE, L_STAE)은 다음 단계에서 추가합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdn import get_pdn

TEACHER_CH = 384


def get_student(size: str = "s", padding: bool = True) -> nn.Sequential:
    return get_pdn(size, out_channels=2 * TEACHER_CH, padding=padding)


def split_student_output(y: torch.Tensor):
    """(B, 768, H, W) → (Teacher 모방 부분, 오토인코더 모방 부분)."""
    return y[:, :TEACHER_CH], y[:, TEACHER_CH:]


# ----------------------------------------------------------------------------
# Teacher 출력 정규화 (Algorithm 1의 3~10행)
# ----------------------------------------------------------------------------
@torch.no_grad()
def teacher_channel_stats(teacher: nn.Module, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """시나리오의 정상 학습 이미지에서 Teacher 출력의 채널별 평균·표준편차.
    모든 이미지·모든 위치의 값을 채널마다 모아 계산합니다."""
    s1 = s2 = None
    count = 0
    for x in loader:
        if isinstance(x, (list, tuple)):
            x = x[0]
        y = teacher(x.to(device)).double()
        if s1 is None:
            s1 = torch.zeros(y.shape[1], dtype=torch.float64, device=device)
            s2 = torch.zeros_like(s1)
        s1 += y.sum(dim=(0, 2, 3))
        s2 += (y * y).sum(dim=(0, 2, 3))
        count += y.shape[0] * y.shape[2] * y.shape[3]
    mean = s1 / count
    std = (s2 / count - mean * mean).clamp_min(1e-12).sqrt()
    return mean.float().view(1, -1, 1, 1), std.float().view(1, -1, 1, 1)


def normalize_teacher(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (t - mean) / std


# ----------------------------------------------------------------------------
# 손실 함수
# ----------------------------------------------------------------------------
def hard_feature_loss(t_norm: torch.Tensor, s_st: torch.Tensor, p_hard: float = 0.999) -> torch.Tensor:
    """오차 D = (T̂ − S)²의 p_hard-분위수 이상인 원소들의 평균 (Section 3.2).
    p_hard = 0이면 일반적인 S–T 손실(전체 평균)과 같습니다.
    논문은 배치 크기 1을 쓰므로 분위수는 배치 전체 원소에 대해 계산합니다."""
    d = (t_norm - s_st) ** 2
    if p_hard <= 0:
        return d.mean()
    flat = d.flatten()
    # torch.quantile은 원소 수 제한(2^24)이 있어 kthvalue로 같은 값을 구함
    k = max(1, int(round(p_hard * (flat.numel() - 1))) + 1)
    d_hard = flat.kthvalue(k).values.detach()
    return d[d >= d_hard].mean()


def pretraining_penalty(s_on_imagenet: torch.Tensor) -> torch.Tensor:
    """ImageNet 이미지에서 Student의 Teacher 모방 채널 출력을 0으로 끌어당기는 벌점.
    (384·64·64)^-1 Σ_c ||S(P)_c||²  — Algorithm 1의 22행."""
    s_st, _ = split_student_output(s_on_imagenet)
    return (s_st ** 2).mean()


# ----------------------------------------------------------------------------
# 이상 맵
# ----------------------------------------------------------------------------
def local_anomaly_map(t_norm: torch.Tensor, s_st: torch.Tensor, out_size: int = 256) -> torch.Tensor:
    """로컬 이상 맵: 채널 평균 제곱 차이 → out_size×out_size로 bilinear 리사이즈.
    반환: (B, 1, out_size, out_size)"""
    m = ((t_norm - s_st) ** 2).mean(dim=1, keepdim=True)
    if m.shape[-1] != out_size:
        m = F.interpolate(m, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return m


def quantile_normalize(m: torch.Tensor, q_a: float, q_b: float) -> torch.Tensor:
    """분위수 기반 맵 정규화: q_a → 0, q_b → 0.1 (Section 3.4)."""
    return 0.1 * (m - q_a) / (q_b - q_a)


if __name__ == "__main__":
    torch.manual_seed(0)
    s = get_student("s")
    y = s(torch.randn(1, 3, 256, 256))
    st, stae = split_student_output(y)
    print("Student 출력:", tuple(y.shape), "→", tuple(st.shape), "+", tuple(stae.shape))

    t = torch.randn_like(st)
    d = (t - st.detach()) ** 2
    loss = hard_feature_loss(t, st)
    n_sel = (d >= d.flatten().kthvalue(int(round(0.999 * (d.numel() - 1))) + 1).values).sum().item()
    print(f"hard feature loss: {loss.item():.4f} (선택 원소 {n_sel}/{d.numel()} = {n_sel / d.numel():.4%})")
    print("p_hard=0 일 때 전체 평균과 같은지:",
          torch.allclose(hard_feature_loss(t, st, 0.0), d.mean()))
