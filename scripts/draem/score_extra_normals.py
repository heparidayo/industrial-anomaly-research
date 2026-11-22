"""추가 정상으로 과검률 재기 — 기존 모델 · 기존 기준값 그대로 (재학습 없음).

    python scripts/draem/score_extra_normals.py --category camera_2 --mask ring_ex \
        --checkpoint outputs/draem/camera_2/draem_st2000_bs8_lr0p0001_s0_rot0p5x180_amp.pt \
        --threshold-json outputs/draem/camera_2/threshold/threshold_k5_st2000_rot0p5x180_s0.json \
        --out outputs/report/extra_normals/v1_ring_ex

--mask ring_ex: 라벨 없는 고정 폭 링 − 시작점 (make_ring_masks.py와 같은 방식, 이미지마다 바로 만듦)
--mask none   : 결함 맵 전체 (띠 펴기 모델은 이것)
결과: <out>/<category>/scores.csv · summary.md · worst.png (점수 높은 정상 12장)
"""

from pathlib import Path
import argparse
import csv
import json
import re
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "draem"))


def main():
    parser = argparse.ArgumentParser(description="추가 정상 과검률")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/company_plc_extra"))
    parser.add_argument("--category", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold-json", type=Path, required=True)
    parser.add_argument("--mask", choices=["none", "ring_ex"], default="ring_ex")
    parser.add_argument("--out", type=Path, default=Path("outputs/report/extra_normals"))
    args = parser.parse_args()
    res = lambda p: p if p.is_absolute() else ROOT / p  # noqa: E731

    from src.models.draem.edge_mask import masked_score
    from src.models.draem.model import DRAEMPredictor
    from make_ring_masks import ring_mask
    from prepare_company_plc import imread, imwrite

    th = json.loads(res(args.threshold_json).read_text(encoding="utf-8"))
    key = "threshold" if args.mask == "none" else f"threshold_{args.mask}"
    if key not in th:
        raise SystemExit(f"{args.threshold_json.name}에 {key}가 없습니다. rescore_kfold.py를 먼저 실행하세요.")
    threshold = float(th[key])
    model = DRAEMPredictor(res(args.checkpoint))
    paths = sorted((res(args.data_root) / args.category / "good").glob("*.png"))
    if not paths:
        raise SystemExit("이미지가 없습니다. prepare_extra_normals.py를 먼저 실행하세요.")
    print(f"{args.category}: 정상 {len(paths)}장 · 기준값 {threshold:.4f} ({key}) · {res(args.checkpoint).name}")

    rows, cache = [], {}
    for i, p in enumerate(paths, 1):
        r = model.predict(p)
        amap = r["anomaly_map"].numpy()
        if args.mask == "ring_ex":
            m = cv2.resize(ring_mask(imread(p)), amap.shape[::-1], interpolation=cv2.INTER_NEAREST)
            score, amap = masked_score(amap, (m > 127).astype(np.float32), model.score_kernel)
        else:
            score = r["image_score"]
        date = re.search(r"PLC-(\d{8})", p.stem).group(1)
        rows.append({"file": p.name, "date": date, "score": round(score, 4), "false_alarm": int(score > threshold)})
        cache[p.name] = (r["image"], amap)
        if i % 50 == 0:
            print(f"  {i}/{len(paths)}")

    out = res(args.out) / args.category
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n, fa = len(rows), sum(r["false_alarm"] for r in rows)
    upper = 1 - 0.025 ** (1 / n) if fa == 0 else None
    lines = [f"# 추가 정상 과검률 — {args.category}", "",
             f"- 모델: {res(args.checkpoint).name}",
             f"- 기준값: {threshold:.4f} ({key}, {args.threshold_json.name}) · 마스크: {args.mask}",
             f"- **과검률: {fa}/{n} ({100 * fa / n:.1f}%)**" + (f" · 0장이라도 실제 과검률 95% 상한 {100 * upper:.1f}%" if upper else ""),
             f"- 점수: 최소 {min(r['score'] for r in rows):.3f} · 중앙값 {np.median([r['score'] for r in rows]):.3f} · 최대 {max(r['score'] for r in rows):.3f}",
             "", "| 촬영 날짜 | 장수 | 과검 | 과검률 | 점수 중앙값 |", "|---|---|---|---|---|"]
    for d in sorted({r["date"] for r in rows}):
        sub = [r for r in rows if r["date"] == d]
        k = sum(r["false_alarm"] for r in sub)
        lines.append(f"| {d[:4]}-{d[4:6]}-{d[6:]} | {len(sub)} | {k} | {100 * k / len(sub):.1f}% | {np.median([r['score'] for r in sub]):.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    worst = sorted(rows, key=lambda r: -r["score"])[:12]  # 점수 높은 정상 12장: 원본 | 겹침
    tiles = []
    for r in worst:
        img, amap = cache[r["file"]]
        img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        heat = cv2.applyColorMap((np.clip(amap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        pair = np.hstack([img8, cv2.addWeighted(img8, 0.5, cv2.resize(heat, img8.shape[1::-1]), 0.5, 0)])
        bar = np.full((32, pair.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"{r['date']}  {r['score']:.3f}{'  FA' if r['false_alarm'] else ''}", (6, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200) if r["false_alarm"] else (40, 40, 40), 2)
        tiles.append(np.vstack([bar, pair]))
    grid = np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles) - len(tiles) % 3 or 3, 3)])
    imwrite(out / "worst.png", grid)
    print(f"\n결과: {out}  (scores.csv · summary.md · worst.png)")


if __name__ == "__main__":
    main()
