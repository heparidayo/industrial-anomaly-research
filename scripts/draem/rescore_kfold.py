"""남겨 둔 fold 모델로 기준값만 다시 계산 (재학습 없음).

kfold_threshold.py를 --keep-checkpoints로 돌렸다면, 나중에 마스크를 새로 만들어도
학습 없이 몇 분 만에 threshold_<마스크>를 다시 계산할 수 있습니다.

예시:
    .\\.venv\\Scripts\\python.exe scripts/draem/rescore_kfold.py --category camera_1
"""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="fold 모델로 마스크별 기준값 재계산")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/company_plc"))
    parser.add_argument("--category", required=True)
    parser.add_argument("--threshold-json", type=Path, default=None,
                        help="다시 계산할 threshold JSON (기본: 가장 최근 것)")
    args = parser.parse_args()

    import torch
    from src.models.draem.edge_mask import MASK_VARIANTS, load_edge_mask, masked_score
    from src.models.draem.model import DRAEMPredictor

    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    th_dir = ROOT / "outputs/draem" / args.category / "threshold"
    if args.threshold_json is not None:
        tj = args.threshold_json if args.threshold_json.is_absolute() else ROOT / args.threshold_json
    else:
        tj = max(th_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    print("대상:", tj.name)
    work_dir = th_dir / tj.stem.replace("threshold_", "")
    train_dir = data_root / args.category / "train" / "good"
    images = sorted(p.name for p in train_dir.glob("*.png"))
    variants = {k: data_root / args.category / d for k, (d, _) in MASK_VARIANTS.items()
                if (data_root / args.category / d).is_dir()}

    result = json.loads(tj.read_text(encoding="utf-8"))
    scores = {k: {} for k in variants}
    for list_path in sorted(work_dir.glob("fold*_train.txt")):
        k = list_path.stem.split("_")[0]
        ckpts = list(work_dir.glob(f"draem_*_{k}.pt"))
        if not ckpts:
            raise SystemExit(f"{k} 모델이 없습니다. kfold_threshold.py를 --keep-checkpoints로 다시 돌려야 합니다.")
        used = set(list_path.read_text(encoding="utf-8").split())
        model = DRAEMPredictor(ckpts[0])
        for name in (n for n in images if n not in used):
            r = model.predict(train_dir / name)
            for key, mdir in variants.items():
                mask = load_edge_mask(mdir / f"{Path(name).stem}.png", int(r["anomaly_map"].shape[-1]))
                scores[key][name], _ = masked_score(r["anomaly_map"].numpy(), mask, model.score_kernel)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for key, sc in scores.items():
        result[f"threshold_{key}"] = float(max(sc.values()))
        result[f"scores_{key}"] = sc
        print(f"threshold_{key} = {result[f'threshold_{key}']:.4f}  ({len(sc)}장)")
    tj.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", tj)


if __name__ == "__main__":
    main()
