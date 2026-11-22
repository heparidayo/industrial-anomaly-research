"""외면취 띠 두께 판정 — 최종 추천 설정으로 평가.

설정 (앞선 실험에서 가장 좋았던 조합):
  1) 날짜(로트)별 기준 두께 : 그 날짜의 train+validation 정상 중앙값 (test 는 쓰지 않음)
     - 그 날짜 정상이 MIN_REF 장 미만이면 전체 train+validation 정상 중앙값으로 대신
  2) 편차(스케일)는 모든 날짜 공통 : 각 정상에서 '자기 날짜 중앙값'을 뺀 값들의 MAD
     → 장수가 적은 날짜도 기준이 흔들리지 않음
  3) 종류별 점수 : over(둘레·30°구간 두꺼움), 미흡(둘레·30°구간 얇음), step(짧은 구간 급변)
  4) 기준값 : train+validation 정상(자기 자신 제외)에서 '셋 중 하나라도 넘는 비율'이 목표 과검률이 되게 종류별로
  5) 부품 단위 판정 : 같은 촬영번호의 camera_1·camera_2 중 하나라도 걸리면 불량
평가 : test 정상 과검률(날짜별), 실제 불량(validation+test) 검출, 가짜 불량(test 정상으로 생성) 검출
출력 : <out>/best_summary.md, best_scores.csv

실행 예:
  python scripts/draem/band_best_eval.py --category C:/company_v2_team_kit/data/processed/company/company_v2_cam12_polarstrip --out outputs/band_best
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_thickness as bt  # noqa: E402
import band_synth_eval as bs  # noqa: E402

TYPES = ("over", "미흡", "step")
ZK = {"over": "over_z", "미흡": "thin_z", "step": "step_z"}
MIN_REF = 5


def main():
    ap = argparse.ArgumentParser(description="띠 두께 판정 최종 설정 평가")
    ap.add_argument("--category", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/band_best"))
    ap.add_argument("--target-far", type=float, default=0.05)
    ap.add_argument("--no-synth", action="store_true", help="가짜 불량 시험 생략")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cat = args.category
    with (cat / "split.csv").open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["prepared_image"]]
    for r in rows:
        r["date"] = bt.date_of(r["file"])
    cams = sorted({r["camera"] for r in rows})
    key = lambda r: r["prepared_image"]

    print("1) 측정")
    F, imgs = {}, {}
    for i, r in enumerate(rows, 1):
        bgr = bt.read_bgr(cat / key(r))
        F[key(r)] = bt.measure(bgr)[0]
        if r["split"] == "test" and r["class"] == "good" and not args.no_synth:
            imgs[key(r)] = bgr
        if i % 100 == 0:
            print(f"   {i}/{len(rows)}")

    ref = [r for r in rows if r["class"] == "good" and r["split"] in ("train", "validation")]

    # --- 날짜별 중심 + 공통 편차
    def center(cam, date, feat, exclude=None):
        same = [F[key(p)][feat] for p in ref if p["camera"] == cam and p["date"] == date and key(p) != exclude]
        if len(same) >= MIN_REF:
            return float(np.median(same))
        pool = [F[key(p)][feat] for p in ref if p["camera"] == cam and key(p) != exclude]
        return float(np.median(pool))

    spread = {}
    for cam in cams:
        spread[cam] = {}
        for feat in bt.KEYS:
            resid = [F[key(p)][feat] - center(cam, p["date"], feat, exclude=key(p))
                     for p in ref if p["camera"] == cam]
            spread[cam][feat] = 1.4826 * float(np.median(np.abs(resid))) + 1e-6

    def zs(f, cam, date, exclude=None):
        base = {k: (center(cam, date, k, exclude), spread[cam][k]) for k in bt.KEYS}
        return bt.z_scores(f, base)

    Z = {}
    for r in rows:
        ref_date = r["date"] if r["class"] == "good" else "9/29"   # 실제 불량은 같은 날(9/29) 정상 기준
        Z[key(r)] = zs(F[key(r)], r["camera"], ref_date, exclude=key(r))

    # --- 기준값 (종류별, train+val 정상에서 합친 과검률 = 목표)
    thr = {}
    for cam in cams:
        fit = [Z[key(p)] for p in ref if p["camera"] == cam]
        lo, hi = 0.80, 1.0
        for _ in range(30):
            q = (lo + hi) / 2
            t = {k: np.quantile([z[k] for z in fit], q) for k in ZK.values()}
            far = np.mean([any(z[k] > t[k] for k in t) for z in fit])
            lo, hi = (q, hi) if far > args.target_far else (lo, q)
        thr[cam] = {k: float(np.quantile([z[k] for z in fit], hi)) for k in ZK.values()}

    def flags(z, cam):
        return {t: z[ZK[t]] > thr[cam][ZK[t]] for t in TYPES}

    pred = {key(r): any(flags(Z[key(r)], r["camera"]).values()) for r in rows}

    # --- 요약 (이미지 단위)
    test = [r for r in rows if r["split"] == "test"]
    evald = [r for r in rows if r["split"] in ("validation", "test") and r["class"] != "good"]
    L = ["# 띠 두께 판정 — 최종 설정 평가", "",
         "- 기준 두께: 날짜별 train+val 정상 중앙값 (정상 5장 미만인 날짜는 전체), 편차는 모든 날짜 공통",
         f"- 기준값: train+val 정상 과검률 {args.target_far:.0%} 가 되게 종류별 설정 → "
         + "; ".join(f"{c} over {thr[c]['over_z']:.1f} / 미흡 {thr[c]['thin_z']:.1f} / step {thr[c]['step_z']:.1f}" for c in cams),
         "- 평가에 쓴 test 정상은 기준 두께·기준값 계산에 쓰지 않음", "",
         "## 이미지 단위", "", "| 카메라 | test 정상 과검 | 실제 over | 실제 미흡 | 실제 step | 실제 불량 합계 |", "|---|---|---|---|---|---|"]
    for cam in cams + ["all"]:
        sel = lambda rr: [r for r in rr if cam == "all" or r["camera"] == cam]
        g = [r for r in sel(test) if r["class"] == "good"]
        fa = sum(pred[key(r)] for r in g)
        cells = []
        for t in TYPES:
            d = [r for r in sel(evald) if r["class"] == t]
            cells.append(f"{sum(pred[key(r)] for r in d)}/{len(d)}")
        d = sel(evald)
        L.append(f"| {cam} | {fa}/{len(g)} ({fa / max(len(g), 1):.1%}) | " + " | ".join(cells)
                 + f" | {sum(pred[key(r)] for r in d)}/{len(d)} ({sum(pred[key(r)] for r in d) / max(len(d), 1):.0%}) |")

    # --- 부품 단위 (촬영번호로 두 카메라 묶기)
    groups = collections.defaultdict(list)
    for r in rows:
        if r["split"] in ("validation", "test"):
            groups[r["group_id"]].append(r)
    part_g = [g for g in groups.values() if all(r["class"] == "good" for r in g) and g[0]["split"] == "test"]
    part_d = [g for g in groups.values() if any(r["class"] != "good" for r in g)]
    pg = sum(any(pred[key(r)] for r in g) for g in part_g)
    L += ["", "## 부품 단위 (두 카메라 중 하나라도 걸리면 불량)", "",
          f"- test 정상 부품 과검: {pg}/{len(part_g)} ({pg / max(len(part_g), 1):.1%})"]
    for t in TYPES:
        gg = [g for g in part_d if any(r["class"] == t for r in g)]
        L.append(f"- 실제 {t} 부품 검출: {sum(any(pred[key(r)] for r in g) for g in gg)}/{len(gg)}")
    tot = sum(any(pred[key(r)] for r in g) for g in part_d)
    L.append(f"- 실제 불량 부품 전체: {tot}/{len(part_d)} ({tot / max(len(part_d), 1):.0%})")

    # --- 날짜별 과검
    L += ["", "## 날짜별 test 정상 과검 (이미지)", "", "| 날짜 | 기준 장수 c1/c2 | camera_1 | camera_2 |", "|---|---|---|---|"]
    for dt in sorted({r["date"] for r in test if r["class"] == "good"}, key=bt.date_key):
        n_ref = [sum(1 for p in ref if p["camera"] == c and p["date"] == dt) for c in cams]
        cells = []
        for c in cams:
            g = [r for r in test if r["class"] == "good" and r["camera"] == c and r["date"] == dt]
            cells.append(f"{sum(pred[key(r)] for r in g)}/{len(g)}" if g else "—")
        L.append(f"| {dt} | {n_ref[0]}/{n_ref[1]}" + ("" if all(n >= MIN_REF for n in n_ref) else " (전체로 대신)")
                 + " | " + " | ".join(cells) + " |")

    # --- 가짜 불량
    out_rows = []
    if not args.no_synth:
        print("2) 가짜 불량")
        rng = np.random.default_rng(args.seed)
        amps = []
        for c in cams:
            g = [F[key(r)]["thick_med"] for r in rows if r["camera"] == c and r["date"] == "9/29" and r["class"] == "good"]
            o = [F[key(r)]["thick_med"] for r in rows if r["camera"] == c and r["class"] == "over"]
            amps.append(np.median(o) - np.median(g))
        over_amp = float(np.mean(amps))
        hits = collections.Counter()
        for i, (k, bgr) in enumerate(imgs.items(), 1):
            r = next(x for x in rows if key(x) == k)
            for t in TYPES:
                img2, amp = bs.make_defect(bgr, t, rng, over_amp)
                z = zs(bt.measure(img2)[0], r["camera"], r["date"])
                hit = any(flags(z, r["camera"]).values())
                hits[(t, r["date"])] += hit
                hits[(t, r["date"], "n")] += 1
                out_rows.append({"kind": t, "camera": r["camera"], "date": r["date"], "file": r["file"],
                                 "amp": round(amp, 1), **{kk: round(v, 2) for kk, v in z.items()}, "detected": int(hit)})
            if i % 50 == 0:
                print(f"   {i}/{len(imgs)}")
        L += ["", f"## 가짜 불량 검출 (test 정상 {len(imgs)}장에 종류별 1개씩, over 크기 ≈ +{over_amp:.1f}줄)", "",
              "| 날짜 | over | 미흡 | step |", "|---|---|---|---|"]
        dates = sorted({d for (_, d, *rest) in hits}, key=bt.date_key)
        for dt in dates:
            L.append(f"| {dt} | " + " | ".join(f"{hits[(t, dt)]}/{hits[(t, dt, 'n')]}" for t in TYPES) + " |")
        L.append("| **합계** | " + " | ".join(
            f"**{sum(hits[(t, d)] for d in dates)}/{sum(hits[(t, d, 'n')] for d in dates)} "
            f"({sum(hits[(t, d)] for d in dates) / max(sum(hits[(t, d, 'n')] for d in dates), 1):.0%})**" for t in TYPES) + " |")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "best_summary.md").write_text("\n".join(L), encoding="utf-8")
    with (args.out / "best_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["split", "class", "camera", "date", "group_id", "file", *bt.KEYS, *ZK.values(), "score", "pred_defect"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({"split": r["split"], "class": r["class"], "camera": r["camera"], "date": r["date"],
                        "group_id": r["group_id"], "file": r["file"],
                        **{k: round(F[key(r)][k], 2) for k in bt.KEYS},
                        **{k: round(v, 2) for k, v in Z[key(r)].items()},
                        "score": round(max(Z[key(r)][k] / thr[r["camera"]][k] for k in ZK.values()), 4),  # 1 넘으면 불량
                        "pred_defect": int(pred[key(r)])})
    if out_rows:
        with (args.out / "best_synth.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
    print("\n".join(L))
    print(f"\n저장: {args.out.resolve()}")


if __name__ == "__main__":
    main()
