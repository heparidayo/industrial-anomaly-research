"""기준값 = 학습에 안 쓴 val 정상을 최종 모델로 채점한 점수의 최댓값 (fold 모델 불필요).

정상이 충분할 때는 k-fold 대신 이 방법이 더 간단하고, 실제 판정하는 모델로 바로 재므로
'fold 모델이 덜 배워서 기준값이 부풀려지는' 문제도 없습니다.

    python scripts/draem/threshold_from_val.py --data-root data/processed/company_plc_v3 --category camera_2 \
        --checkpoint outputs/draem/camera_2/<최종 모델>.pt

결과: outputs/draem/<category>/threshold/threshold_val_<모델 이름>.json
      키: threshold (마스크 없음) · threshold_ring_ex (라벨 없는 링 − 시작점) · 점수 통계
"""

from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "draem"))


def main():
    parser = argparse.ArgumentParser(description="val 정상으로 기준값")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/company_plc_v3"))
    parser.add_argument("--category", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rule", choices=["max", "p99"], default="max")
    args = parser.parse_args()
    res = lambda p: p if p.is_absolute() else ROOT / p  # noqa: E731

    from src.models.draem.edge_mask import masked_score
    from src.models.draem.model import DRAEMPredictor
    from make_ring_masks import ring_mask
    from prepare_company_plc import imread

    ckpt = res(args.checkpoint)
    model = DRAEMPredictor(ckpt)
    paths = sorted((res(args.data_root) / args.category / "val" / "good").glob("*.png"))
    if not paths:
        raise SystemExit("val 정상이 없습니다. build_v3_split.py를 먼저 실행하세요.")
    full, ring = [], []
    for p in paths:
        r = model.predict(p)
        amap = r["anomaly_map"].numpy()
        full.append(r["image_score"])
        if getattr(model, "rect", None):   # 띠·적층 모델: 이미지 전체가 띠라 링 마스크 없음
            continue
        m = cv2.resize(ring_mask(imread(p)), amap.shape[::-1], interpolation=cv2.INTER_NEAREST)
        ring.append(masked_score(amap, (m > 127).astype(np.float32), model.score_kernel)[0])
    rule = (lambda v: float(np.max(v))) if args.rule == "max" else (lambda v: float(np.percentile(v, 99)))
    stats = lambda v: {"min": float(np.min(v)), "mean": float(np.mean(v)), "max": float(np.max(v)),  # noqa: E731
                       "p99": float(np.percentile(v, 99))}
    out = ckpt.parent / "threshold" / f"threshold_val_{ckpt.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"category": args.category, "checkpoint": ckpt.name, "rule": f"{args.rule} of val normals (최종 모델로 채점)",
              "n_val": len(paths), "threshold": rule(full), "score_stats": stats(full),
              "scores": {p.name: float(s) for p, s in zip(paths, full)}}
    if ring:
        result.update({"threshold_ring_ex": rule(ring), "score_stats_ring_ex": stats(ring),
                       "scores_ring_ex": {p.name: float(s) for p, s in zip(paths, ring)}})
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"val 정상 {len(paths)}장 · 기준값 (마스크 없음) {result['threshold']:.4f}"
          + (f" · (링) {result['threshold_ring_ex']:.4f}" if ring else ""))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
