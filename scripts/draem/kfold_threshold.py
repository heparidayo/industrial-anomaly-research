"""학습 정상 이미지로 threshold 정하기 (k-fold 교차 채점).

학습 정상을 k묶음으로 나눠, 한 묶음을 빼고 학습 → 뺀 이미지를 채점하는 과정을 k번 반복합니다.
그러면 모든 학습 정상이 "학습에 쓰지 않은 모델"로 한 번씩 채점되고,
그 점수의 최댓값을 threshold로 씁니다. 테스트 정답은 전혀 쓰지 않습니다.

    정상 10장, --folds 5  → 2장씩 빼고 8장으로 학습, 5번
    정상 10장, --folds 10 → leave-one-out (1장씩 빼고 9장으로 학습, 10번)

중요: fold 모델은 최종 모델과 **같은 학습 설정**(--steps, 회전 증강 등)으로 학습해야 합니다.
이 스크립트에 준 학습 옵션은 그대로 train_draem.py에 전달됩니다.

예시:
    .\\.venv\\Scripts\\python.exe scripts/draem/kfold_threshold.py `
        --data-root data/processed/company_plc --category camera_1 --folds 5 `
        --steps 10000 --rotate-normal-prob 0.5 --rotate-degrees 180

결과:
    outputs/draem/<category>/threshold/threshold_<설정>.json
        threshold (최댓값), 이미지별 점수, fold 구성, 학습 설정
    → evaluate_draem.py --threshold-json <이 파일> 로 과검률 · 미검률 · F1 계산
"""

from pathlib import Path
import argparse
import json
import random
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TRAIN_SCRIPT = ROOT / "scripts/draem/train_draem.py"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS)


def resolve(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser(description="k-fold 교차 채점으로 threshold 정하기")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/company_plc"))
    parser.add_argument("--category", required=True)
    parser.add_argument("--folds", type=int, default=5, help="묶음 수 (학습 정상 장수와 같으면 leave-one-out)")
    parser.add_argument("--split-seed", type=int, default=0, help="묶음 나누기 seed")
    parser.add_argument("--keep-checkpoints", action="store_true", help="fold 모델 파일을 지우지 않음")

    # train_draem.py로 그대로 넘길 학습 옵션 (최종 모델과 같게)
    train = parser.add_argument_group("학습 옵션 (최종 모델과 같게)")
    train.add_argument("--steps", type=int, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--rotate-normal-prob", type=float, default=0.3)
    train.add_argument("--rotate-degrees", type=float, default=90)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--anomaly-source", type=Path, default=Path("data/raw/dtd/images"))
    args = parser.parse_args()

    if args.steps is None and args.epochs is None:
        parser.error("--steps 또는 --epochs를 지정하세요. (최종 모델과 같은 값)")

    train_dir = resolve(args.data_root) / args.category / "train/good"
    images = sorted(p.name for p in list_images(train_dir))
    if len(images) < 2:
        parser.error(f"학습 정상 이미지가 부족합니다: {train_dir}")
    if not 2 <= args.folds <= len(images):
        parser.error(f"--folds는 2~{len(images)} 사이여야 합니다.")

    # 1. 묶음 나누기 (seed 고정)
    order = images[:]
    random.Random(args.split_seed).shuffle(order)
    folds = [order[i::args.folds] for i in range(args.folds)]

    length = f"st{args.steps}" if args.steps is not None else f"e{args.epochs}"
    setting = f"{length}_rot{args.rotate_normal_prob:g}x{args.rotate_degrees:g}_s{args.seed}".replace(".", "p")
    work_dir = ROOT / "outputs/draem" / args.category / "threshold" / f"k{args.folds}_{setting}"
    result_path = work_dir.parent / f"threshold_k{args.folds}_{setting}.json"
    if result_path.exists():
        raise SystemExit(f"이미 결과가 있습니다: {result_path}")
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== k-fold threshold: {args.category} ===")
    print(f"학습 정상 {len(images)}장 → {args.folds}묶음 ({' · '.join(str(len(f)) for f in folds)}장)")

    # 무거운 모듈은 학습이 끝난 뒤 채점할 때만 불러옴
    import torch
    from src.models.draem.model import DRAEMPredictor

    scores = {}
    for k, held_out in enumerate(folds):
        print(f"\n--- fold {k + 1}/{args.folds}: {len(held_out)}장 빼고 {len(images) - len(held_out)}장으로 학습 ---")

        list_path = work_dir / f"fold{k}_train.txt"
        list_path.write_text("\n".join(n for n in images if n not in held_out), encoding="utf-8")

        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--data-root", str(args.data_root), "--category", args.category,
            "--image-list", str(list_path), "--tag", f"fold{k}", "--output-dir", str(work_dir),
            "--batch-size", str(args.batch_size), "--lr", str(args.lr), "--seed", str(args.seed),
            "--rotate-normal-prob", str(args.rotate_normal_prob), "--rotate-degrees", str(args.rotate_degrees),
            "--num-workers", str(args.num_workers), "--anomaly-source", str(args.anomaly_source),
            "--save-every", "0",
        ]
        cmd += ["--steps", str(args.steps)] if args.steps is not None else ["--epochs", str(args.epochs)]
        subprocess.run(cmd, check=True, cwd=ROOT)

        ckpt = next(work_dir.glob(f"draem_*_fold{k}.pt"))
        model = DRAEMPredictor(ckpt)
        for name in held_out:
            scores[name] = model.predict(train_dir / name)["image_score"]
            print(f"  채점 {name}: {scores[name]:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not args.keep_checkpoints:
            ckpt.unlink()

    # 2. threshold = 학습에 쓰지 않은 정상 점수의 최댓값
    values = np.array(list(scores.values()))
    threshold = float(values.max())

    result = {
        "category": args.category,
        "data_root": args.data_root.as_posix(),
        "threshold": threshold,
        "rule": "max of out-of-fold normal scores",
        "score_stats": {"min": float(values.min()), "mean": float(values.mean()),
                        "max": threshold, "p95": float(np.percentile(values, 95))},
        "folds": folds,
        "scores": scores,
        "train": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 완료 ===")
    print(f"정상 점수 (학습에 안 쓴 모델로 채점): 최소 {values.min():.4f} · 평균 {values.mean():.4f} · 최대 {threshold:.4f}")
    print(f"threshold = {threshold:.4f}")
    print(f"저장: {result_path}")


if __name__ == "__main__":
    main()
