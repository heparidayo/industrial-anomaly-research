"""9/29 말고 다른 날짜에서도 불량을 잡는지 '가짜 불량'으로 시험한다.

왜 : 실제 불량은 9/29 하루치뿐이라, 띠가 두꺼운 로트(9/15)·얇은 로트(10/29)에서도
     over·미흡·step을 잡는지 확인할 방법이 없다.
방법:
  1) 테스트 정상 띠 이미지마다 가짜 불량 3종을 만든다 (세로 방향으로 띠만 늘이거나 줄임, 표면 결은 유지)
     - over : 넓은 구간(90°~255°)에서 띠를 안쪽으로 +A줄 두껍게 (A는 실제 9/29 over 크기 기준)
     - 미흡 : 40°~120° 구간에서 띠를 60~90% 얇게
     - step : 6°~15° 짧은 구간만 띠가 +10~20줄 움푹 들어가게
  2) band_thickness.py 와 같은 측정·점수로 채점
     - global : 카메라별 학습 정상 전체 기준
     - lot    : 같은 날짜 정상 기준 (자기 자신 제외)
  3) 기준값: 학습+검증 정상(모든 날짜)으로 종류별 기준값을 정하되, 셋 중 하나라도 넘는 비율(과검률)이
     --target-far (기본 2%)가 되게 맞춘다 → 테스트 정상 과검률, 가짜 불량 검출률(날짜별), 실제 9/29 불량 검출률
출력: <out>/synth_summary.md, synth_scores.csv, synth_examples.png

실행 예:
  python scripts/draem/band_synth_eval.py --category C:/company_v2_team_kit/data/processed/company/company_v2_cam12_polarstrip --out outputs/band_synth
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_thickness as bt  # noqa: E402

TYPES = ("over", "미흡", "step")
ZKEY = {"over": "over_z", "미흡": "thin_z", "step": "step_z"}
DEG_PER_COL = (bt.ANGLES[1] - bt.ANGLES[0]) / 1727


# ---------------------------------------------------------------- 가짜 불량
def taper_profile(W, start, length, taper, amp, shape="flat"):
    """구간 [start, start+length) 에서 amp, 양 끝은 부드럽게."""
    x = np.arange(W, dtype=np.float32)
    p = np.zeros(W, np.float32)
    if shape == "bump":                                   # step: 가운데가 가장 깊은 종 모양
        c = start + length / 2
        u = np.clip(np.abs(x - c) / (length / 2), 0, 1)
        return amp * np.cos(u * np.pi / 2) ** 2
    inside = (x >= start) & (x < start + length)
    p[inside] = 1
    if taper > 0:
        k = int(taper)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
        s, e = int(start), int(start + length)
        lo = slice(max(s, 0), min(s + k, W))
        p[lo] = ramp[: lo.stop - lo.start]
        hi = slice(max(e - k, 0), min(e, W))
        p[hi] = np.minimum(p[hi], ramp[::-1][: hi.stop - hi.start])
    return amp * p


def warp_band(bgr, inner, outer, delta):
    """세로줄마다 안쪽 경계를 delta 줄 옮긴다 (띠·면 텍스처는 늘이거나 줄여서 유지)."""
    H, W = bgr.shape[:2]
    e = inner.copy()
    e[~np.isfinite(e)] = np.nanmedian(e)
    o = np.minimum(outer, e - 2)
    e2 = np.clip(e + delta, o + 1, H - 3)
    y = np.arange(H, dtype=np.float32)[:, None]
    src = np.where(y <= o, y,
          np.where(y <= e2, o + (y - o) * (e - o) / np.maximum(e2 - o, 1e-3),
                   e + (y - e2) * (H - 1 - e) / np.maximum(H - 1 - e2, 1e-3)))
    mx = np.repeat(np.arange(W, dtype=np.float32)[None, :], H, 0)
    return cv2.remap(bgr, mx, src.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def make_defect(bgr, kind, rng, over_amp):
    inner, outer = bt.detect_edges(bgr)
    fit, _ = bt.fit_outer(outer)
    W = bgr.shape[1]
    thick = np.nanmedian(inner - fit)
    if kind == "over":
        L = rng.uniform(90, 255) / DEG_PER_COL
        A = rng.uniform(0.7, 1.3) * over_amp
        d = taper_profile(W, rng.uniform(0, max(W - L, 1)), L, 15 / DEG_PER_COL, A)
    elif kind == "미흡":
        L = rng.uniform(40, 120) / DEG_PER_COL
        A = -rng.uniform(0.6, 0.9) * max(thick, 5)
        d = taper_profile(W, rng.uniform(0, W - L), L, 10 / DEG_PER_COL, A)
    else:
        L = rng.uniform(6, 15) / DEG_PER_COL
        A = rng.uniform(10, 20)
        d = taper_profile(W, rng.uniform(L, W - 2 * L), L, 0, A, shape="bump")
    return warp_band(bgr, inner, fit, d), float(A)


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser(description="가짜 불량으로 날짜별 검출 시험")
    ap.add_argument("--category", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/band_synth"))
    ap.add_argument("--target-far", type=float, default=0.02, help="학습+검증 정상 기준 목표 과검률")
    ap.add_argument("--exclude-z", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cat = args.category
    rng = np.random.default_rng(args.seed)
    with (cat / "split.csv").open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["prepared_image"]]
    for r in rows:
        r["date"] = bt.date_of(r["file"])
    cams = sorted({r["camera"] for r in rows})

    print("1) 원본 측정")
    feats, imgs = {}, {}
    for i, r in enumerate(rows, 1):
        bgr = bt.read_bgr(cat / r["prepared_image"])
        feats[r["prepared_image"]] = bt.measure(bgr)[0]
        if r["split"] == "test" and r["class"] == "good":
            imgs[r["prepared_image"]] = bgr
        if i % 100 == 0:
            print(f"   {i}/{len(rows)}")

    # 날짜 안에서 크게 벗어난 정상 제외 (기준 계산에서만)
    excluded = set()
    by_date = collections.defaultdict(list)
    for r in rows:
        if r["class"] == "good":
            by_date[(r["camera"], r["date"])].append(r)
    for L0 in by_date.values():
        if len(L0) >= 5:
            m, s = bt.robust([feats[r["prepared_image"]]["thick_med"] for r in L0])
            excluded |= {r["prepared_image"] for r in L0
                         if abs(feats[r["prepared_image"]]["thick_med"] - m) / s > args.exclude_z}

    # 실제 over 크기 = 9/29 over 두께 - 9/29 정상 두께 (카메라 평균)
    amps = []
    for cam in cams:
        g = [feats[r["prepared_image"]]["thick_med"] for r in by_date[(cam, "9/29")]]
        o = [feats[r["prepared_image"]]["thick_med"] for r in rows if r["camera"] == cam and r["class"] == "over"]
        amps.append(np.median(o) - np.median(g))
    over_amp = float(np.mean(amps))
    print(f"   실제 over 크기(9/29 over - 9/29 정상) ≈ +{over_amp:.1f}줄 → 가짜 over 크기 기준")

    print("2) 가짜 불량 생성·측정")
    synth = []
    examples = collections.defaultdict(list)
    for i, r in enumerate([r for r in rows if r["prepared_image"] in imgs], 1):
        for kind in TYPES:
            img2, amp = make_defect(imgs[r["prepared_image"]], kind, rng, over_amp)
            synth.append({"src": r, "kind": kind, "amp": amp, "f": bt.measure(img2)[0]})
            if len(examples[(kind, r["date"])]) < 1:
                examples[(kind, r["date"])].append((imgs[r["prepared_image"]], img2, r, amp))
        if i % 50 == 0:
            print(f"   {i}/{len(imgs)}")

    # 기준(baseline) 함수
    keys = bt.KEYS
    goods = collections.defaultdict(list)
    for r in rows:
        if r["class"] == "good" and r["prepared_image"] not in excluded:
            goods[(r["camera"], r["date"])].append(r)
    gbase = {cam: {k: bt.robust([feats[r["prepared_image"]][k] for r in rows if r["split"] == "train"
                                 and r["class"] == "good" and r["camera"] == cam
                                 and r["prepared_image"] not in excluded]) for k in keys} for cam in cams}

    def base_for(mode, cam, date, self_key):
        if mode == "global":
            return gbase[cam]
        peers = [p for p in goods[(cam, date)] if p["prepared_image"] != self_key]
        if len(peers) < 5:
            return gbase[cam]
        return {k: bt.robust([feats[p["prepared_image"]][k] for p in peers]) for k in keys}

    zs = bt.z_scores

    out_rows, L = [], ["# 가짜 불량으로 본 날짜별 검출 시험", "",
                       f"- 가짜 over 크기: 실제 9/29 over 두께 증가량 ≈ +{over_amp:.1f}줄 (×0.7~1.3, 90°~255° 구간)",
                       "- 가짜 미흡: 40°~120° 구간에서 띠 60~90% 감소 / 가짜 step: 6°~15° 구간 +10~20줄 움푹",
                       f"- 기준값: 학습+검증 정상(모든 날짜)에서 '셋 중 하나라도 넘는 비율' = {args.target_far:.0%} 가 되게 종류별로 설정",
                       f"- 기준 계산에서 뺀 정상(같은 날짜에서 z>{args.exclude_z:g}): {len(excluded)}장", ""]
    for mode in ("global", "lot"):
        # 정상·실제 불량 점수 (lot: 불량은 9/29 정상 기준)
        sc = {}
        for r in rows:
            d = r["date"] if r["class"] == "good" else "9/29"
            sc[r["prepared_image"]] = zs(feats[r["prepared_image"]], base_for(mode, r["camera"], d, r["prepared_image"]))
        for s in synth:
            r = s["src"]
            s[mode] = zs(s["f"], base_for(mode, r["camera"], r["date"], r["prepared_image"]))

        # 종류별 기준값: 같은 분위수 q 를 써서 합친 과검률이 목표가 되게 이분 탐색
        thr = {}
        for cam in cams:
            fit_set = [sc[r["prepared_image"]] for r in rows if r["camera"] == cam and r["class"] == "good"
                       and r["split"] in ("train", "validation") and r["prepared_image"] not in excluded]
            lo, hi = 0.80, 1.0
            for _ in range(30):
                q = (lo + hi) / 2
                t = {k: np.quantile([z[k] for z in fit_set], q) for k in ZKEY.values()}
                far = np.mean([any(z[k] > t[k] for k in t) for z in fit_set])
                lo, hi = (q, hi) if far > args.target_far else (lo, q)
            thr[cam] = {k: float(np.quantile([z[k] for z in fit_set], hi)) for k in ZKEY.values()}

        def hit(z, cam, kinds=ZKEY.values()):
            return any(z[k] > thr[cam][k] for k in kinds)

        L += [f"## 기준 = {mode} ({'카메라별 학습 정상 전체' if mode == 'global' else '같은 날짜 정상'})", "",
              "기준값 (z): " + "; ".join(f"{cam} over {thr[cam]['over_z']:.1f} / 미흡 {thr[cam]['thin_z']:.1f} / step {thr[cam]['step_z']:.1f}"
                                         for cam in cams), "",
              "| 날짜 | 카메라 | 테스트 정상 과검 | 가짜 over 검출 | 가짜 미흡 검출 | 가짜 step 검출 |", "|---|---|---|---|---|---|"]
        tot = collections.Counter()
        dates = sorted({s["src"]["date"] for s in synth}, key=bt.date_key)
        for dt in dates:
            for cam in cams:
                g = [r for r in rows if r["split"] == "test" and r["class"] == "good" and r["camera"] == cam and r["date"] == dt]
                if not g:
                    continue
                fa = sum(hit(sc[r["prepared_image"]], cam) for r in g)
                cells = []
                for kind in TYPES:
                    ss = [s for s in synth if s["kind"] == kind and s["src"]["camera"] == cam and s["src"]["date"] == dt]
                    h = sum(hit(s[mode], cam) for s in ss)
                    tot[(kind, "h")] += h
                    tot[(kind, "n")] += len(ss)
                    cells.append(f"{h}/{len(ss)}")
                tot["fa"] += fa
                tot["g"] += len(g)
                L.append(f"| {dt} | {cam[-1]} | {fa}/{len(g)} | " + " | ".join(cells) + " |")
        L.append(f"| **합계** | | **{tot['fa']}/{tot['g']} ({tot['fa'] / tot['g']:.1%})** | "
                 + " | ".join(f"**{tot[(k, 'h')]}/{tot[(k, 'n')]} ({tot[(k, 'h')] / max(tot[(k, 'n')], 1):.0%})**" for k in TYPES) + " |")
        real = [r for r in rows if r["class"] != "good" and r["split"] in ("validation", "test")]
        L += ["", "실제 9/29 불량 검출 (검증+테스트): " + ", ".join(
            f"{k} {sum(hit(sc[r['prepared_image']], r['camera']) for r in real if r['class'] == k)}"
            f"/{sum(1 for r in real if r['class'] == k)}" for k in TYPES), ""]
        for s in synth:
            out_rows.append({"mode": mode, "kind": s["kind"], "camera": s["src"]["camera"], "date": s["src"]["date"],
                             "file": s["src"]["file"], "amp": round(s["amp"], 1),
                             **{k: round(v, 2) for k, v in s[mode].items()}, "detected": int(hit(s[mode], s["src"]["camera"]))})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "synth_summary.md").write_text("\n".join(L), encoding="utf-8")
    with (args.out / "synth_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    tiles = []
    for dt in ("9/15", "10/12", "10/29"):
        for kind in TYPES:
            for a, b, r, amp in examples.get((kind, dt), [])[:1]:
                pair = np.vstack([a, np.full((4, a.shape[1], 3), 255, np.uint8), b])
                pair = cv2.resize(pair, (1152, pair.shape[0]), interpolation=cv2.INTER_AREA)
                lab = np.full((24, 1152, 3), 255, np.uint8)
                name = {"미흡": "insufficient"}.get(kind, kind)
                cv2.putText(lab, f"{dt} {r['camera']}  fake {name} (amp {amp:+.0f} rows)  top=original bottom=fake",
                            (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                tiles += [lab, pair]
    if tiles:
        cv2.imencode(".png", np.vstack(tiles))[1].tofile(str(args.out / "synth_examples.png"))
    print("\n".join(L))
    print(f"\n저장: {args.out.resolve()}")


if __name__ == "__main__":
    main()
