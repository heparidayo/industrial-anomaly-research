"""라벨 윤곽선으로 '테두리 띠 폭'을 재서 불량 유형을 얼마나 가를 수 있는지 확인 (분석용).

라벨 txt: class 1 = 부품 바깥 원, class 0 = 부품 상면(가공면) 바깥 경계.
각 각도(1°)마다  띠 폭 = 바깥 원 반지름 − 상면 경계까지 거리(px, 원본 이미지 기준)

    띠 폭 평균  : 가공 범위가 넓은지 좁은지 (over ↑, 미흡 ↓ 예상)
    띠 폭 편차  : 경계가 고른지 (step ↑ 예상)

점수 = max(|평균 − 정상평균| / 정상표준편차, (편차 − 정상평균) / 정상표준편차)
정상 통계는 학습 정상(train)만으로 계산하고, 테스트셋에서 AUROC를 냅니다.

주의: 사람이 그린 라벨 윤곽선을 쓰므로 '측정값이 맞다면 얼마나 가를 수 있나'의 상한에 가깝습니다.
실제 검사에 쓰려면 상면 경계를 이미지에서 자동으로 찾아야 합니다.

예시:
    .\\.venv\\Scripts\\python.exe scripts/draem/measure_edge_band.py --src "C:\\대세 라벨링"

결과: outputs/report/edge_band/edge_band.csv · summary.md
"""

from pathlib import Path
import argparse
import csv

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TYPES = ["over", "step", "insufficient"]


def band_features(txt: Path, size=(1216, 1216)) -> dict | None:
    polys = {}
    for line in txt.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) >= 7:
            polys[v[0]] = np.array(v[1:], dtype=float).reshape(-1, 2) * size
    if "0" not in polys or "1" not in polys:
        return None
    c = polys["1"].mean(axis=0)
    R = np.hypot(*(polys["1"] - c).T).mean()
    p = polys["0"]
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    r = np.hypot(*(p - c).T)
    o = np.argsort(ang)
    a = np.r_[ang[o] - 2 * np.pi, ang[o], ang[o] + 2 * np.pi]
    rr = np.r_[r[o], r[o], r[o]]
    t = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    width = R - np.interp(t, a, rr)
    return {"band_mean": width.mean(), "band_std": width.std(), "band_min": width.min(), "band_max": width.max()}


def auroc(neg, pos) -> float:
    neg, pos = np.asarray(neg), np.asarray(pos)
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(neg) * len(pos)))


def main():
    parser = argparse.ArgumentParser(description="라벨 윤곽선으로 테두리 띠 폭 측정")
    parser.add_argument("--src", type=Path, required=True, help="'대세 라벨링' 폴더")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/company_plc"))
    parser.add_argument("--categories", nargs="+", default=["camera_1", "camera_2", "cam12"])
    parser.add_argument("--out", type=Path, default=Path("outputs/report/edge_band"))
    args = parser.parse_args()

    labels = {t.stem: t for t in args.src.resolve().rglob("labels/**/*.txt")}
    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    all_rows, lines = [], ["| 데이터셋 | 유형 | 장수 | 띠 폭 평균(px) | 띠 폭 편차(px) |", "|---|---|---|---|---|"]
    auc_lines = ["| 데이터셋 | 전체 AUROC | over | step | 미흡 |", "|---|---|---|---|---|"]

    for cat in args.categories:
        rows = list(csv.DictReader(open(data_root / cat / "manifest.csv", encoding="utf-8-sig")))
        items = []
        for row in rows:
            f = band_features(labels[Path(row["image_id"]).stem]) if Path(row["image_id"]).stem in labels else None
            if f:
                items.append({**f, "dataset": cat, "image": Path(row["image_id"]).name,
                              "split": row["split"], "defect_type": row["defect_type"]})

        train = [i for i in items if i["split"] == "train"]
        mu = {k: np.mean([i[k] for i in train]) for k in ("band_mean", "band_std")}
        sd = {k: np.std([i[k] for i in train]) for k in ("band_mean", "band_std")}
        for i in items:
            i["score"] = max(abs(i["band_mean"] - mu["band_mean"]) / sd["band_mean"],
                             (i["band_std"] - mu["band_std"]) / sd["band_std"])
        all_rows += items

        for t in ["good", *TYPES]:
            s = [i for i in items if i["defect_type"] == t]
            if s:
                lines.append(f"| {cat} | {t} | {len(s)} | {np.mean([i['band_mean'] for i in s]):.1f} ± {np.std([i['band_mean'] for i in s]):.1f} "
                             f"| {np.mean([i['band_std'] for i in s]):.1f} ± {np.std([i['band_std'] for i in s]):.1f} |")
        test = [i for i in items if i["split"] == "test"]
        good = [i["score"] for i in test if i["defect_type"] == "good"]
        aucs = [auroc(good, [i["score"] for i in test if i["defect_type"] != "good"])]
        aucs += [auroc(good, [i["score"] for i in test if i["defect_type"] == t]) for t in TYPES]
        auc_lines.append(f"| {cat} | " + " | ".join(f"{a:.3f}" for a in aucs) + " |")
        print(f"{cat}: 테두리 띠 점수 AUROC {aucs[0]:.3f} (over {aucs[1]:.3f} · step {aucs[2]:.3f} · 미흡 {aucs[3]:.3f})")

    with open(out / "edge_band.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    (out / "summary.md").write_text("\n".join(lines + [""] + auc_lines) + "\n", encoding="utf-8")
    print(f"\n완료: {out}  (summary.md · edge_band.csv)")


if __name__ == "__main__":
    main()
