"""
Patch Description Network (PDN) — EfficientAD 논문 Table 6, Table 7.

Teacher와 Student는 같은 구조를 쓰고 마지막 출력 채널 수만 다릅니다.
  - Teacher : out_channels = 384
  - Student : out_channels = 768 (= Teacher 모방 384 + 오토인코더 모방 384)

padding 인자에 대해:
  패딩은 가중치 모양을 바꾸지 않습니다. 따라서 padding=True로 학습한 가중치를
  padding=False 모델에 그대로 불러올 수 있습니다(state_dict 키와 모양이 동일).
  논문은 지연 시간 측정 시 패딩을 끈 설정을 사용했습니다(Appendix E).
    - padding=True  : 256×256 입력 → 64×64 출력
    - padding=False : 256×256 입력 → 56×56 출력

실행:  python pdn.py   (출력 크기, 파라미터 수, 수용 영역 33×33 검증)
"""

import torch
import torch.nn as nn


def _p(value: int, padding: bool) -> int:
    return value if padding else 0


def get_pdn_small(out_channels: int = 384, padding: bool = True) -> nn.Sequential:
    """EfficientAD-S의 PDN (Table 6)."""
    return nn.Sequential(
        nn.Conv2d(3, 128, kernel_size=4, padding=_p(3, padding)),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=_p(1, padding)),
        nn.Conv2d(128, 256, kernel_size=4, padding=_p(3, padding)),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=_p(1, padding)),
        nn.Conv2d(256, 256, kernel_size=3, padding=_p(1, padding)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, out_channels, kernel_size=4, padding=0),
    )


def get_pdn_medium(out_channels: int = 384, padding: bool = True) -> nn.Sequential:
    """EfficientAD-M의 PDN (Table 7)."""
    return nn.Sequential(
        nn.Conv2d(3, 256, kernel_size=4, padding=_p(3, padding)),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=_p(1, padding)),
        nn.Conv2d(256, 512, kernel_size=4, padding=_p(3, padding)),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=_p(1, padding)),
        nn.Conv2d(512, 512, kernel_size=1, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=_p(1, padding)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, out_channels, kernel_size=4, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
    )


def get_pdn(size: str = "s", out_channels: int = 384, padding: bool = True) -> nn.Sequential:
    size = size.lower()
    if size in ("s", "small"):
        return get_pdn_small(out_channels, padding)
    if size in ("m", "medium"):
        return get_pdn_medium(out_channels, padding)
    raise ValueError(f"알 수 없는 PDN 크기: {size!r} ('s' 또는 'm')")


def load_teacher(path: str, size: str = "s", padding: bool = True,
                 device: str = "cpu") -> nn.Sequential:
    """distill_teacher.py가 저장한 Teacher 가중치를 불러와 고정된 상태로 반환."""
    ckpt = torch.load(path, map_location=device)
    state = ckpt["teacher"] if "teacher" in ckpt else ckpt
    teacher = get_pdn(size, out_channels=384, padding=padding)
    teacher.load_state_dict(state)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


# ----------------------------------------------------------------------------
# 자체 검증
# ----------------------------------------------------------------------------
def receptive_field(model: nn.Module, image_size: int = 256) -> tuple[int, int]:
    """출력 중앙 위치 하나에 대한 입력 기울기가 0이 아닌 영역의 크기(높이, 너비).
    논문 Figure 3 위쪽 행과 같은 방식의 측정입니다."""
    model = model.eval().double()
    x = torch.randn(1, 3, image_size, image_size, dtype=torch.float64, requires_grad=True)
    y = model(x)
    h, w = y.shape[-2] // 2, y.shape[-1] // 2
    y[0, :, h, w].abs().sum().backward()
    nz = x.grad.abs().sum(dim=(0, 1)) > 0
    rows = torch.nonzero(nz.any(dim=1)).flatten()
    cols = torch.nonzero(nz.any(dim=0)).flatten()
    model.float()
    return int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 3, 256, 256)
    for size in ("s", "m"):
        for role, ch in (("teacher", 384), ("student", 768)):
            for padding in (True, False):
                net = get_pdn(size, ch, padding)
                with torch.no_grad():
                    out = net(x)
                n_params = sum(p.numel() for p in net.parameters())
                print(f"PDN-{size.upper()} {role:7s} padding={str(padding):5s} "
                      f"→ 출력 {tuple(out.shape)}, 파라미터 {n_params / 1e6:.2f}M")

        rf = receptive_field(get_pdn(size, 384, padding=True))
        print(f"PDN-{size.upper()} 수용 영역: {rf[0]}×{rf[1]} 픽셀")
        assert rf == (33, 33), "수용 영역이 33×33이 아닙니다"

    # 패딩 유무에 따라 가중치가 호환되는지 확인
    a, b = get_pdn("s", 384, True), get_pdn("s", 384, False)
    b.load_state_dict(a.state_dict())
    print("padding=True 가중치를 padding=False 모델에 불러오기: OK")
