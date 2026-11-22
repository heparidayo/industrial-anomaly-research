"""
Student 학습 (Student–Teacher 부분만) — 논문 Algorithm 1의 1~22행, 38~43행.

  1) 시나리오의 정상 학습 이미지로 Teacher 출력의 채널별 평균·표준편차 계산
  2) Student를 70,000회(배치 1) 학습
       L_ST = L_hard(p_hard=0.999) + pretraining penalty(ImageNet 이미지)
       Adam(lr 1e-4, weight decay 1e-5), 전체의 95%(66,500회) 이후 lr 1e-5
  3) 검증 이미지로 로컬 맵의 분위수 q_a(0.9), q_b(0.995) 계산
  4) 테스트 세트에서 이미지 단위 AU-ROC 평가 (점수 = 로컬 맵 최댓값)

이 단계는 오토인코더 없이 Student–Teacher 부분이 제대로 동작하는지 확인하는
용도입니다. 논문 Table 4의 "PDN + 맵 정규화 + hard feature loss + pretraining
penalty" 행(오토인코더 제외)에 해당합니다. 최종 EfficientAD는 다음 단계에서
오토인코더와 함께 Student를 처음부터 다시 학습합니다(손실을 한 번에 합산).

사용 예:
  python train_student.py --data_root /data/mvtec_ad --category bottle \
      --teacher teacher_s.pth --imagenet_dir /data/imagenet/train --size s
  # 빠른 동작 확인
  python train_student.py ... --iterations 500
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets import ImageNetPenalty, NormalImages, TestImages, split_train_val
from pdn import load_teacher
from student import (get_student, hard_feature_loss, local_anomaly_map, normalize_teacher,
                     pretraining_penalty, quantile_normalize, split_student_output,
                     teacher_channel_stats)


def auroc(labels, scores) -> float:
    """순위 기반 AU-ROC (동점은 평균 순위). sklearn의 roc_auc_score와 같은 값."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = avg_rank[inv]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def infinite_loader(dataset, num_samples, workers, seed, batch_size=1):
    g = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples, generator=g)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


@torch.no_grad()
def predict_local(teacher, student, x, mean, std):
    t = normalize_teacher(teacher(x), mean, std)
    s_st, _ = split_student_output(student(x))
    return local_anomaly_map(t, s_st)            # (B, 1, 256, 256)


def main():
    ap = argparse.ArgumentParser(description="EfficientAD Student–Teacher 학습")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--teacher", required=True, help="distill_teacher.py가 저장한 파일")
    ap.add_argument("--imagenet_dir", required=True)
    ap.add_argument("--size", default="s", choices=["s", "m"])
    ap.add_argument("--no_padding", action="store_true",
                    help="Teacher·Student를 패딩 없이 실행 (논문의 지연 시간 측정 설정)")
    ap.add_argument("--iterations", type=int, default=70_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--p_hard", type=float, default=0.999)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    padding = not args.no_padding
    cat_dir = Path(args.data_root) / args.category
    out_dir = Path(args.out_dir) / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 데이터
    train_paths, val_paths = split_train_val(cat_dir, args.val_ratio, args.seed)
    print(f"[{args.category}] 학습 {len(train_paths)}장, 검증 {len(val_paths)}장, 장치 {device}")
    train_set = NormalImages(train_paths)
    penalty_set = ImageNetPenalty(args.imagenet_dir)

    # ------------------------------------------------------------ 모델
    teacher = load_teacher(args.teacher, args.size, padding=padding, device=device)
    student = get_student(args.size, padding=padding).to(device)

    print("1) Teacher 출력 채널 통계 계산")
    stat_loader = DataLoader(train_set, batch_size=16, num_workers=args.workers)
    t_mean, t_std = teacher_channel_stats(teacher, stat_loader, device)

    # ------------------------------------------------------------ 학습
    print(f"2) Student 학습: {args.iterations}회, p_hard={args.p_hard}")
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    decay_at = int(args.iterations * 0.95)        # 70,000회 중 66,500회
    train_iter = iter(infinite_loader(train_set, args.iterations, args.workers, args.seed))
    pen_iter = iter(infinite_loader(penalty_set, args.iterations, args.workers, args.seed + 1))

    student.train()
    log = {"hard": 0.0, "penalty": 0.0}
    t0 = time.time()
    for it in range(1, args.iterations + 1):
        x = next(train_iter).to(device, non_blocking=True)
        p = next(pen_iter).to(device, non_blocking=True)

        with torch.no_grad():
            t_norm = normalize_teacher(teacher(x), t_mean, t_std)
        s_st, _ = split_student_output(student(x))
        loss_hard = hard_feature_loss(t_norm, s_st, args.p_hard)
        loss_pen = pretraining_penalty(student(p))
        loss = loss_hard + loss_pen

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it == decay_at + 1:
            for g in optimizer.param_groups:
                g["lr"] = args.lr * 0.1
        log["hard"] += loss_hard.item()
        log["penalty"] += loss_pen.item()
        if it % args.log_every == 0 or it == args.iterations:
            n = args.log_every if it % args.log_every == 0 else it % args.log_every
            print(f"  it {it:6d}  L_hard {log['hard'] / n:.4f}  penalty {log['penalty'] / n:.4f}"
                  f"  ({n / (time.time() - t0):.1f} it/s)")
            log = {"hard": 0.0, "penalty": 0.0}
            t0 = time.time()
    student.eval()

    # ------------------------------------------------------------ 맵 정규화
    print("3) 검증 이미지로 로컬 맵 분위수 계산")
    val_loader = DataLoader(NormalImages(val_paths), batch_size=8, num_workers=args.workers)
    vals = [predict_local(teacher, student, x.to(device), t_mean, t_std).cpu().numpy().ravel()
            for x in val_loader]
    vals = np.concatenate(vals)
    q_a, q_b = float(np.quantile(vals, 0.9)), float(np.quantile(vals, 0.995))
    print(f"  q_a={q_a:.6f}, q_b={q_b:.6f}")

    # ------------------------------------------------------------ 평가
    print("4) 테스트 평가")
    test_set = TestImages(cat_dir)
    labels, scores = [], []
    for i in range(len(test_set)):
        x, label, _, _, _ = test_set[i]
        m = quantile_normalize(predict_local(teacher, student, x[None].to(device), t_mean, t_std),
                               q_a, q_b)
        labels.append(label)
        scores.append(m.max().item())
    img_auroc = auroc(labels, scores)
    print(f"  이미지 단위 AU-ROC: {img_auroc * 100:.2f}%  "
          f"(정상 {labels.count(0)}장, 불량 {labels.count(1)}장)")

    # ------------------------------------------------------------ 저장
    ckpt = {"student": student.state_dict(), "teacher_mean": t_mean.cpu(), "teacher_std": t_std.cpu(),
            "q_st": (q_a, q_b), "size": args.size, "padding": padding, "args": vars(args)}
    torch.save(ckpt, out_dir / f"student_{args.size}.pth")
    with open(out_dir / "result_student_only.json", "w") as f:
        json.dump({"category": args.category, "image_auroc": img_auroc,
                   "q_st": [q_a, q_b], "iterations": args.iterations}, f, indent=2)
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
