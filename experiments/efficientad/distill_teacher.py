"""
Teacher PDN 증류 학습 — EfficientAD 논문 Algorithm 3.

  1) Ψ(PatchCore WideResNet-101 특징)의 채널별 평균·표준편차를
     ImageNet 이미지 10,000장으로 계산
  2) 60,000회 반복, 배치 16, Adam(lr 1e-4, weight decay 1e-5)으로
     정규화된 Ψ 출력을 Teacher가 MSE로 따라 하도록 학습
     - Ψ  입력: 512×512
     - Teacher 입력: 256×256        → 둘 다 384×64×64 출력
     - 확률 0.1로 흑백 변환

사용 예:
  python distill_teacher.py --imagenet_dir /data/imagenet/train --size s --out teacher_s.pth
  python distill_teacher.py --imagenet_dir ... --size s --out teacher_s.pth --resume   # 이어서 학습
  python distill_teacher.py --imagenet_dir ... --iterations 200 --stat_images 160     # 빠른 동작 확인

--imagenet_dir 아래의 이미지를 하위 폴더까지 모두 찾아 사용합니다(클래스 라벨 불필요).
"""

import argparse
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms

from feature_extractor import PatchCoreFeatureExtractor
from pdn import get_pdn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------------
# 데이터
# ----------------------------------------------------------------------------
class DistillationDataset(Dataset):
    """같은 이미지를 Ψ용(512)과 Teacher용(256) 두 해상도로 반환."""

    def __init__(self, root: str, psi_size: int = 512, teacher_size: int = 256,
                 grayscale_p: float = 0.1):
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            raise FileNotFoundError(f"폴더가 없습니다: {root_path}")
        self.paths = sorted(p for p in root_path.rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not self.paths:
            found = sorted({p.suffix.lower() for p in root_path.rglob("*") if p.is_file()})
            raise FileNotFoundError(
                f"{root_path} 아래에서 이미지({', '.join(sorted(IMG_EXTS))})를 찾지 못했습니다. "
                f"발견된 파일 확장자: {found or '없음'}")
        self.gray = transforms.RandomGrayscale(p=grayscale_p)   # 3채널 유지
        norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        self.to_psi = transforms.Compose(
            [transforms.Resize((psi_size, psi_size),
                               interpolation=transforms.InterpolationMode.BILINEAR)] + norm)
        self.to_teacher = transforms.Compose(
            [transforms.Resize((teacher_size, teacher_size),
                               interpolation=transforms.InterpolationMode.BILINEAR)] + norm)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.gray(img)                       # 두 해상도에 같은 증강 적용
        return self.to_psi(img), self.to_teacher(img)


def make_loader(dataset, batch_size, num_samples, workers, seed):
    g = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples, generator=g)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=True, persistent_workers=workers > 0)


# ----------------------------------------------------------------------------
# 1단계: Ψ 채널 통계
# ----------------------------------------------------------------------------
@torch.no_grad()
def compute_psi_stats(psi, dataset, n_images, batch_size, workers, device, amp, seed):
    loader = make_loader(dataset, batch_size, n_images, workers, seed)
    s1 = torch.zeros(psi.out_channels, dtype=torch.float64, device=device)
    s2 = torch.zeros_like(s1)
    count = 0
    for i, (x_psi, _) in enumerate(loader):
        x_psi = x_psi.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            y = psi(x_psi)
        y = y.double()
        s1 += y.sum(dim=(0, 2, 3))
        s2 += (y * y).sum(dim=(0, 2, 3))
        count += y.shape[0] * y.shape[2] * y.shape[3]
        if (i + 1) % 50 == 0:
            print(f"  [통계] {(i + 1) * batch_size}/{n_images} 장")
    mean = s1 / count
    std = (s2 / count - mean * mean).clamp_min(1e-12).sqrt()
    return mean.float().view(1, -1, 1, 1), std.float().view(1, -1, 1, 1)


# ----------------------------------------------------------------------------
# 2단계: 증류 학습
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="EfficientAD Teacher PDN 증류")
    ap.add_argument("--imagenet_dir", required=True)
    ap.add_argument("--size", default="s", choices=["s", "m"])
    ap.add_argument("--out", default=None, help="저장 경로 (기본: teacher_{size}.pth)")
    ap.add_argument("--iterations", type=int, default=60_000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--stat_images", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true",
                    help="Ψ 순전파만 float16으로 실행 (속도·메모리 절약, 논문 설정은 아님)")
    ap.add_argument("--ckpt_every", type=int, default=5_000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out or f"teacher_{args.size}.pth")
    ckpt_path = out_path.with_suffix(".ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"장치: {device}, PDN-{args.size.upper()}, 반복 {args.iterations}, 배치 {args.batch_size}")

    dataset = DistillationDataset(args.imagenet_dir)
    print(f"이미지 {len(dataset)}장")
    psi = PatchCoreFeatureExtractor(out_channels=384, pretrained=True).to(device)

    # 증류는 padding=True로 학습해 출력 크기(64×64)를 Ψ와 맞춤.
    # 같은 가중치를 추론 때 padding=False 모델에 그대로 불러올 수 있음.
    teacher = get_pdn(args.size, out_channels=384, padding=True).to(device)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    start_it = 0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        teacher.load_state_dict(ckpt["teacher"])
        optimizer.load_state_dict(ckpt["optimizer"])
        mu, sigma = ckpt["psi_mean"].to(device), ckpt["psi_std"].to(device)
        start_it = ckpt["iteration"]
        print(f"{ckpt_path}에서 {start_it}회부터 이어서 학습")
    else:
        print("1단계: Ψ 채널 통계 계산")
        t0 = time.time()
        mu, sigma = compute_psi_stats(psi, dataset, args.stat_images, args.batch_size,
                                      args.workers, device, amp, args.seed)
        print(f"  완료 ({time.time() - t0:.0f}s)")

    def save(path, iteration, with_optimizer):
        state = {"teacher": teacher.state_dict(), "psi_mean": mu.cpu(), "psi_std": sigma.cpu(),
                 "size": args.size, "iteration": iteration}
        if with_optimizer:
            state["optimizer"] = optimizer.state_dict()
        tmp = str(path) + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)

    print("2단계: 증류 학습")
    remaining = args.iterations - start_it
    loader = make_loader(dataset, args.batch_size, max(remaining, 1) * args.batch_size,
                         args.workers, args.seed + start_it)
    teacher.train()
    running, t0 = 0.0, time.time()
    it = start_it
    for x_psi, x_t in loader:
        if it >= args.iterations:
            break
        x_psi = x_psi.to(device, non_blocking=True)
        x_t = x_t.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                target = psi(x_psi)
            target = (target.float() - mu) / sigma          # 채널별 정규화

        pred = teacher(x_t)
        loss = F.mse_loss(pred, target)                     # 배치 내 이미지별 MSE의 평균과 같음

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        it += 1
        running += loss.item()
        if it % args.log_every == 0:
            dt = time.time() - t0
            print(f"  it {it:6d}/{args.iterations}  loss {running / args.log_every:.4f}  "
                  f"({args.log_every / dt:.1f} it/s)")
            running, t0 = 0.0, time.time()
        if it % args.ckpt_every == 0:
            save(ckpt_path, it, with_optimizer=True)

    teacher.eval()
    save(out_path, it, with_optimizer=False)
    print(f"Teacher 저장: {out_path}")


if __name__ == "__main__":
    main()
