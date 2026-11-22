"""
EfficientAD 전체 학습 — 논문 Algorithm 1.

  1) Teacher 출력의 채널별 평균·표준편차 (정상 학습 이미지)
  2) Student와 오토인코더를 함께 70,000회 학습 (배치 1)
       L_ST   = L_hard(원본 이미지) + pretraining penalty(ImageNet 이미지)
       L_AE   = MSE(T̂(증강), A(증강))
       L_STAE = MSE(A(증강), S_뒤384(증강))     ← 기울기가 오토인코더에도 흐름
       L      = L_ST + L_AE + L_STAE
     Adam(lr 1e-4, weight decay 1e-5), 95%(66,500회) 이후 lr 1e-5
  3) 검증 이미지로 로컬·글로벌 맵의 분위수 q_a(0.9), q_b(0.995)
  4) 테스트 평가: 이미지 AU-ROC(결합·로컬만·글로벌만), 픽셀 AU-ROC, AU-PRO

결과: <out_dir>/<category>/efficientad_<size>.pth (Teacher 포함, 이 파일 하나로 추론)
      <out_dir>/<category>/result.json

사용 예:
  python train.py --data_root ..\\..\\data\\raw\\dummy --category widget ^
      --teacher teacher_s.pth --imagenet_dir ..\\..\\data\\raw\\imagenet --size s
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from datasets import GeoAug, ImageNetPenalty, NormalImages, TestImages, split_train_val
from efficientad import EfficientAD
from metrics import auroc, pixel_metrics
from preprocess import Preprocess
from student import (hard_feature_loss, pretraining_penalty, split_student_output,
                     teacher_channel_stats)


def infinite_loader(dataset, num_samples, workers, seed, batch_size=1):
    g = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples, generator=g)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


@torch.no_grad()
def compute_quantiles(model, val_paths, device, workers, pre=None):
    loader = DataLoader(NormalImages(val_paths, preprocess=pre), batch_size=8, num_workers=workers)
    st, ae = [], []
    for x in loader:
        m_st, m_ae = model.raw_maps(x.to(device))
        st.append(m_st.float().cpu().numpy().ravel())
        ae.append(m_ae.float().cpu().numpy().ravel())
    st, ae = np.concatenate(st), np.concatenate(ae)
    return [float(np.quantile(st, 0.9)), float(np.quantile(st, 0.995)),
            float(np.quantile(ae, 0.9)), float(np.quantile(ae, 0.995))]


@torch.no_grad()
def evaluate(model, category_dir, device, pixel_eval=True, pre=None, split="test"):
    test_set = TestImages(category_dir, preprocess=pre, split=split)
    labels, s_comb, s_st, s_ae = [], [], [], []
    maps, masks, skipped = [], [], 0
    for i in range(len(test_set)):
        x, label, mask, (h, w), _ = test_set[i]
        comb, n_st, n_ae = model(x[None].to(device))
        labels.append(label)
        s_comb.append(model.image_score(comb).item())
        s_st.append(model.image_score(n_st).item())
        s_ae.append(model.image_score(n_ae).item())
        if pixel_eval:
            if label == 1 and test_set.items[i][2] is None:
                skipped += 1                     # 마스크 없는 불량 이미지는 픽셀 평가 제외
                continue
            full = F.interpolate(comb.float(), size=(h, w), mode="bilinear", align_corners=False)
            maps.append(full[0, 0].cpu().numpy().astype(np.float16))
            masks.append(mask.numpy().astype(bool))
    result = {
        "n_normal": labels.count(0), "n_anomalous": labels.count(1),
        "image_auroc": auroc(labels, s_comb),
        "image_auroc_local_only": auroc(labels, s_st),
        "image_auroc_global_only": auroc(labels, s_ae),
    }
    # 불량 유형(폴더)별: 정상 대 해당 유형
    types = [p.parent.name for p, _, _ in test_set.items]
    good = [i for i, t in enumerate(types) if t == "good"]
    for t in sorted(set(types) - {"good"}):
        idx = [i for i, tt in enumerate(types) if tt == t]
        lab = [0] * len(good) + [1] * len(idx)
        for name, sc in (("", s_comb), ("_local_only", s_st), ("_global_only", s_ae)):
            result[f"image_auroc{name}[{t}]"] = auroc(lab, [sc[i] for i in good + idx])
    if pixel_eval and maps:
        result.update(pixel_metrics(maps, masks))
        result["pixel_eval_skipped_no_mask"] = skipped
    return result


def main():
    ap = argparse.ArgumentParser(description="EfficientAD 전체 학습 (Algorithm 1)")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--teacher", required=True, help="distill_teacher.py가 저장한 파일")
    ap.add_argument("--imagenet_dir", required=True)
    ap.add_argument("--size", default="s", choices=["s", "m"])
    ap.add_argument("--image_size", type=int, default=256,
                    help="모델 입력 크기 (논문 256). 4단 적층은 432로 원본 해상도 그대로 사용 가능")
    ap.add_argument("--no_padding", action="store_true", help="Teacher·Student 무패딩 (출력 56×56)")
    ap.add_argument("--iterations", type=int, default=70_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--p_hard", type=float, default=0.999)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=1_000)
    ap.add_argument("--no_pixel_eval", action="store_true", help="픽셀 단위 평가 생략 (메모리 절약)")
    ap.add_argument("--skip_eval", action="store_true", help="테스트 평가 생략 (테스트 세트가 없을 때)")
    ap.add_argument("--eval_split", default="test", choices=["test", "validation"],
                    help="학습 후 평가할 폴더 (팀 공통 v2: 먼저 validation 59장, 설정 확정 후 test 231장)")
    ap.add_argument("--seed", type=int, default=0)
    g = ap.add_argument_group("정상 편차 증강 (논문에 없는 선택 기능, 기본 꺼짐)")
    g.add_argument("--rot_deg", type=float, default=0.0, help="최대 회전 각도 (180이면 전체 회전)")
    g.add_argument("--shift_px", type=float, default=0.0, help="최대 이동 픽셀 (256×256 기준)")
    g.add_argument("--geo_prob", type=float, default=1.0, help="회전·이동을 적용할 확률")
    g.add_argument("--geo_target", choices=["both", "st", "ae"], default="both",
                   help="적용 대상: both(둘 다), st(Student–Teacher만), ae(오토인코더만)")
    pg = ap.add_argument_group("입력 전처리 (논문에 없는 선택 기능, 기본 꺼짐) — 자세한 설명은 preprocess.py")
    pg.add_argument("--crop", default="none", help="none / center:0.7 / box:x0,y0,x1,y1 / auto:0.15")
    pg.add_argument("--bright_norm", default="none", choices=["none", "mean", "meanstd"],
                    help="이미지별 밝기 정규화")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cat_dir = Path(args.data_root) / args.category
    out_dir = Path(args.out_dir) / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 데이터
    train_paths, val_paths = split_train_val(cat_dir, args.val_ratio, args.seed)
    print(f"[{args.category}] 학습 {len(train_paths)}장, 검증 {len(val_paths)}장, 장치 {device}")
    if args.image_size < 64 or args.image_size % 4:
        ap.error("--image_size는 64 이상의 4의 배수여야 합니다")
    pre = Preprocess(args.crop, args.bright_norm, size=args.image_size)
    print(f"  입력 크기: {args.image_size}×{args.image_size}" + ("" if args.image_size == 256 else " (논문 256과 다름)"))
    print(f"  입력 전처리: {pre}" if pre.enabled else "  입력 전처리: 없음 (논문 설정)")
    geo = GeoAug(args.rot_deg, args.shift_px, args.geo_prob)
    train_set = NormalImages(train_paths, with_ae_aug=True,       # (Student–Teacher용, 오토인코더용)
                             geo_aug=geo, geo_target=args.geo_target, preprocess=pre)
    print(f"  정상 편차 증강: {geo} → 대상 {args.geo_target}" if geo.enabled
          else "  정상 편차 증강: 없음 (논문 설정)")
    penalty_set = ImageNetPenalty(args.imagenet_dir, args.image_size)

    # ------------------------------------------------------------ 모델
    model = EfficientAD(args.size, padding=not args.no_padding, image_size=args.image_size).to(device)
    t_ckpt = torch.load(args.teacher, map_location=device)
    model.teacher.load_state_dict(t_ckpt["teacher"] if "teacher" in t_ckpt else t_ckpt)
    model.freeze_teacher()

    print("1) Teacher 출력 채널 통계 계산")
    stat_loader = DataLoader(NormalImages(train_paths, preprocess=pre), batch_size=16, num_workers=args.workers)
    mean, std = teacher_channel_stats(model.teacher, stat_loader, device)
    model.t_mean.copy_(mean)
    model.t_std.copy_(std)

    # ------------------------------------------------------------ 학습
    print(f"2) Student + 오토인코더 학습: {args.iterations}회")
    params = list(model.student.parameters()) + list(model.autoencoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    decay_at = int(args.iterations * 0.95)
    train_iter = iter(infinite_loader(train_set, args.iterations, args.workers, args.seed))
    pen_iter = iter(infinite_loader(penalty_set, args.iterations, args.workers, args.seed + 1))

    model.student.train()
    model.autoencoder.train()
    keys = ("hard", "penalty", "ae", "stae")
    log = dict.fromkeys(keys, 0.0)
    t0 = time.time()
    for it in range(1, args.iterations + 1):
        x, x_aug = (v.to(device, non_blocking=True) for v in next(train_iter))
        p = next(pen_iter).to(device, non_blocking=True)

        # Student–Teacher (증강 없음)
        with torch.no_grad():
            t = model.normalize_teacher(model.teacher(x))
        s_st, _ = split_student_output(model.student(x))
        l_hard = hard_feature_loss(t, s_st, args.p_hard)
        l_pen = pretraining_penalty(model.student(p))

        # 오토인코더–Student (증강 이미지)
        with torch.no_grad():
            t_aug = model.normalize_teacher(model.teacher(x_aug))
        a = model.autoencoder(x_aug)
        _, s_ae = split_student_output(model.student(x_aug))
        l_ae = F.mse_loss(a, t_aug)
        l_stae = F.mse_loss(s_ae, a)                 # a를 detach하지 않음 (논문 A.1)

        loss = l_hard + l_pen + l_ae + l_stae
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it == decay_at + 1:
            for g in optimizer.param_groups:
                g["lr"] = args.lr * 0.1
        for k, v in zip(keys, (l_hard, l_pen, l_ae, l_stae)):
            log[k] += v.item()
        if it % args.log_every == 0 or it == args.iterations:
            n = args.log_every if it % args.log_every == 0 else it % args.log_every
            print(f"  it {it:6d}  " + "  ".join(f"{k} {log[k] / n:.4f}" for k in keys)
                  + f"  ({n / (time.time() - t0):.1f} it/s)")
            log = dict.fromkeys(keys, 0.0)
            t0 = time.time()
    model.eval()                                     # Dropout 끄기

    # ------------------------------------------------------------ 맵 정규화
    print("3) 검증 이미지로 분위수 계산")
    q = compute_quantiles(model, val_paths, device, args.workers, pre)
    model.quantiles.copy_(torch.tensor(q))
    print(f"  로컬  q_a={q[0]:.6f}, q_b={q[1]:.6f}\n  글로벌 q_a={q[2]:.6f}, q_b={q[3]:.6f}")

    meta = {"category": args.category, "iterations": args.iterations, "p_hard": args.p_hard,
            "val_ratio": args.val_ratio, "seed": args.seed, "teacher": str(args.teacher),
            "n_train": len(train_paths), "n_val": len(val_paths),
            "geo_aug": {"rot_deg": args.rot_deg, "shift_px": args.shift_px,
                        "prob": args.geo_prob, "target": args.geo_target},
            "preprocess": pre.to_dict(), "image_size": args.image_size}
    ckpt_path = out_dir / f"efficientad_{args.size}.pth"
    model.save(str(ckpt_path), meta)
    print(f"모델 저장: {ckpt_path}")

    # ------------------------------------------------------------ 평가
    result = {"meta": meta, "quantiles": q}
    if not args.skip_eval:
        print(f"4) 평가 ({args.eval_split})")
        result[args.eval_split] = evaluate(model, cat_dir, device, pixel_eval=not args.no_pixel_eval,
                                           pre=pre, split=args.eval_split)
        for k, v in result[args.eval_split].items():
            print(f"  {k}: {v * 100:.2f}%" if isinstance(v, float) else f"  {k}: {v}")
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
