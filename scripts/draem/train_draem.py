"""DRAEM 학습 (PatchCore의 build_memory + build_coreset 자리).

예시 (Windows PowerShell, 프로젝트 루트):
    # 동작 확인용 짧은 학습
    .\\.venv\\Scripts\\python.exe scripts/draem/train_draem.py --category bottle --epochs 2 --limit 16 --num-workers 0

    # MVTec bottle: 원본 논문 설정 (700 epoch)
    .\\.venv\\Scripts\\python.exe scripts/draem/train_draem.py --category bottle

    # 회사 데이터: 정상 10장이라 epoch 대신 학습 횟수(step)로 지정
    .\\.venv\\Scripts\\python.exe scripts/draem/train_draem.py --data-root data/processed/company_plc --category camera_1 --steps 10000

    # 부품 홈 방향이 제각각이면 정상 회전 증강을 전 방향으로
    ... --rotate-normal-prob 0.5 --rotate-degrees 180

출력:
    outputs/draem/<category>/draem_<설정>.pt        최종 체크포인트 (덮어쓰기 금지)
    outputs/draem/<category>/draem_<설정>_last.pt   학습 중간 저장
    outputs/draem/<category>/logs/<설정>/           TensorBoard 로그
"""

from pathlib import Path
import argparse
import math
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.draem.imgaug_compat import ia  # noqa: E402
from src.models.draem.dataset import DRAEMTrainDataset, list_images, worker_init_fn  # noqa: E402
from src.models.draem.loss import FocalLoss, SSIMLoss  # noqa: E402
from src.models.draem.networks import build_networks, weights_init  # noqa: E402


def resolve(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ia.seed(seed)


def build_checkpoint(args, rec, seg, epoch, step, image_paths, elapsed):
    return {
        "reconstructive_state": rec.state_dict(),
        "discriminative_state": seg.state_dict(),
        "category": args.category,
        "data_root": args.data_root.as_posix(),
        "epochs": args.epochs,
        "trained_epochs": epoch,
        "steps": args.steps if args.steps is not None else -1,
        "trained_steps": step,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "image_size": args.image_size,
        "rec_base_width": 128,
        "seg_base_width": 64,
        "score_kernel": 21,
        "image_count": len(image_paths),
        "train_images": [p.name for p in image_paths],
        "anomaly_source": args.anomaly_source.as_posix(),
        "rotate_normal_prob": args.rotate_normal_prob,
        "rotate_degrees": args.rotate_degrees,
        "tag": args.tag,
        "train_seconds": round(elapsed, 1),
        "method": "draem",
    }


def main():
    parser = argparse.ArgumentParser(description="DRAEM 학습")

    parser.add_argument("--category", default="bottle")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument("--anomaly-source", type=Path, default=Path("data/raw/dtd/images"),
                        help="DTD images 폴더 (하위 폴더의 jpg 전체 사용)")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--steps", type=int, default=None,
                        help="총 학습 횟수(step). 지정하면 --epochs 대신 사용 (정상 이미지가 적을 때)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--rotate-normal-prob", type=float, default=0.3,
                        help="정상 학습 이미지 회전 확률 (원본 0.3, 끄려면 0)")
    parser.add_argument("--rotate-degrees", type=float, default=90,
                        help="정상 회전 범위 ±N도 (원본 90, 전 방향은 180)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="디버그용: 정상 이미지 개수 제한")
    parser.add_argument("--image-list", type=Path, default=None,
                        help="train/good 중 이 목록(파일 이름 한 줄에 하나)만 학습 (k-fold용)")
    parser.add_argument("--tag", default="", help="출력 파일 이름 끝에 붙일 표시 (예: fold0)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="체크포인트 폴더 (기본 outputs/draem/<category>)")
    parser.add_argument("--save-every", type=int, default=None,
                        help="N epoch마다 _last.pt 저장 (0이면 끔, 기본: 전체의 약 10%%마다)")

    args = parser.parse_args()

    if args.image_size % 32 != 0:
        parser.error("--image-size는 32의 배수여야 합니다. (U-Net 5단계 다운샘플링)")

    # 1. 정상 학습 이미지 / 이상 소스 이미지
    train_dir = resolve(args.data_root) / args.category / "train/good"
    image_paths = list_images(train_dir)
    if not image_paths:
        parser.error(f"정상 학습 이미지를 찾을 수 없습니다: {train_dir}")

    if args.image_list is not None:
        wanted = {ln.strip() for ln in resolve(args.image_list).read_text(encoding="utf-8").splitlines() if ln.strip()}
        image_paths = [p for p in image_paths if p.name in wanted]
        if len(image_paths) != len(wanted):
            parser.error(f"--image-list의 {len(wanted)}개 중 {len(image_paths)}개만 train/good에 있습니다.")

    if args.limit is not None:
        if not 1 <= args.limit <= len(image_paths):
            parser.error("--limit 값이 올바르지 않습니다.")
        image_paths = image_paths[:args.limit]

    source_dir = resolve(args.anomaly_source)
    source_paths = list_images(source_dir, recursive=True)
    if not source_paths:
        parser.error(f"이상 소스(DTD) 이미지를 찾을 수 없습니다: {source_dir}")

    # 2. 데이터 로더 (epoch당 step 수를 알아야 --steps를 epoch로 바꿀 수 있음)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DRAEMTrainDataset(
        image_paths, source_paths,
        image_size=args.image_size,
        rotate_normal_prob=args.rotate_normal_prob,
        rotate_degrees=args.rotate_degrees,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.seed),
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) > args.batch_size,
    )
    steps_per_epoch = len(loader)
    if args.steps is not None:
        args.epochs = math.ceil(args.steps / steps_per_epoch)
    save_every = args.save_every if args.save_every is not None else max(1, args.epochs // 10)
    log_every = max(1, args.epochs // 100)

    # 3. 출력 경로 (설정을 파일명에 기록, 기존 결과 보호)
    lr_name = f"{args.lr:g}".replace(".", "p").replace("-", "m")
    length = f"st{args.steps}" if args.steps is not None else f"e{args.epochs}"
    stem = f"draem_{length}_bs{args.batch_size}_lr{lr_name}_s{args.seed}"
    if args.rotate_normal_prob != 0.3 or args.rotate_degrees != 90:
        stem += f"_rot{args.rotate_normal_prob:g}x{args.rotate_degrees:g}".replace(".", "p")
    if args.limit is not None:
        stem += f"_debug{args.limit}"
    if args.tag:
        stem += f"_{args.tag}"

    output_dir = resolve(args.output_dir) if args.output_dir else ROOT / "outputs/draem" / args.category
    output_path = output_dir / f"{stem}.pt"
    last_path = output_dir / f"{stem}_last.pt"

    if output_path.exists():
        raise SystemExit(f"이미 결과 파일이 존재합니다: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. 모델, 손실, 최적화
    rec, seg = build_networks()
    rec.apply(weights_init)
    seg.apply(weights_init)
    rec.to(device)
    seg.to(device)

    optimizer = torch.optim.Adam([
        {"params": rec.parameters(), "lr": args.lr},
        {"params": seg.parameters(), "lr": args.lr},
    ])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [int(args.epochs * 0.8), int(args.epochs * 0.9)], gamma=0.2
    )

    l2_loss = torch.nn.MSELoss()
    ssim_loss = SSIMLoss()
    focal_loss = FocalLoss()

    writer = SummaryWriter(log_dir=str(output_dir / "logs" / stem))

    print("=== DRAEM 학습 ===")
    print("사용 장치:", device)
    print("데이터:", args.data_root.as_posix(), "/", args.category)
    print("정상 학습 이미지:", len(image_paths))
    print("이상 소스 이미지:", len(source_paths))
    print(f"학습량: {args.epochs} epoch × {steps_per_epoch} step = {args.epochs * steps_per_epoch} step")
    print(f"정상 회전 증강: 확률 {args.rotate_normal_prob}, ±{args.rotate_degrees:g}°")
    print("저장 위치:", output_path)

    # 5. 학습
    start_time = time.perf_counter()
    step = 0

    for epoch in range(1, args.epochs + 1):
        rec.train()
        seg.train()
        sums = {"l2": 0.0, "ssim": 0.0, "focal": 0.0, "total": 0.0}

        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            augmented = batch["augmented_image"].to(device, non_blocking=True)
            mask = batch["anomaly_mask"].to(device, non_blocking=True)

            reconstruction = rec(augmented)
            joined = torch.cat((reconstruction, augmented), dim=1)
            mask_prob = torch.softmax(seg(joined), dim=1)

            loss_l2 = l2_loss(reconstruction, image)
            loss_ssim = ssim_loss(reconstruction, image)
            loss_focal = focal_loss(mask_prob, mask)
            loss = loss_l2 + loss_ssim + loss_focal

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            for key, value in zip(sums, (loss_l2, loss_ssim, loss_focal, loss)):
                sums[key] += value.item()
            step += 1

        scheduler.step()

        n = max(steps_per_epoch, 1)
        if epoch % log_every == 0 or epoch == args.epochs:
            for key, value in sums.items():
                writer.add_scalar(f"loss/{key}", value / n, step)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], step)

            elapsed = time.perf_counter() - start_time
            print(
                f"[{epoch}/{args.epochs} · step {step}] total={sums['total'] / n:.4f} "
                f"l2={sums['l2'] / n:.4f} ssim={sums['ssim'] / n:.4f} "
                f"focal={sums['focal'] / n:.4f} ({elapsed / 60:.1f}분)"
            )

        if save_every and epoch % save_every == 0 and epoch != args.epochs:
            elapsed = time.perf_counter() - start_time
            torch.save(build_checkpoint(args, rec, seg, epoch, step, image_paths, elapsed), last_path)

    # 6. 최종 저장
    elapsed = time.perf_counter() - start_time
    torch.save(build_checkpoint(args, rec, seg, args.epochs, step, image_paths, elapsed), output_path)
    writer.close()

    if last_path.exists():
        last_path.unlink()

    print("\n=== DRAEM 학습 완료 ===")
    print("소요 시간:", f"{elapsed / 60:.1f}분")
    print("저장 위치:", output_path)


if __name__ == "__main__":
    main()
