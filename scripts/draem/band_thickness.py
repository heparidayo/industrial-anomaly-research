"""외면취 띠 두께를 각도별로 직접 재서 over / step / 미흡을 판정한다 (학습 없음, 규칙 기반).

입력  : 팀 키트로 만든 띠 카테고리 (예: data/processed/company/company_v2_cam12_polarstrip)
        - 이미지 108×1728, 맨 윗줄 = 1.05R(바깥), 맨 아랫줄 = 0.75R(안쪽), 가로 = 시작점 기준 5°~260°
방법  :
  1) 안쪽 경계 : 세로줄마다 '어두운 띠 → 밝은 연마면'으로 처음 바뀌는 줄 (밝기 중간값 교차, 소수점)
  2) 바깥 경계 : 띠 위쪽에서 배경과 밝기가 갈라지는 줄을 찾고, 부품 외곽은 원이므로
                 a + b·cos(각도) + c·sin(각도) 로 맞춘다 → 중심이 조금 틀어져도(편심) 상쇄됨
  3) 두께(각도별) = 안쪽 경계 - 바깥 경계  [띠 줄 단위, 1줄 ≈ 원본 약 1px]
  4) 특징   : 두께 중앙값(over↑, 미흡↓), 국소 변화량(step: 주변 22° 중앙값 대비 최대 차이)
  5) 점수   : 카메라별 학습 정상(train/good) 대비 robust z (중앙값·MAD)
              over_z = (두께 - 기준)/편차, thin_z = (기준 - 두께)/편차, step_z = 국소 변화 z
              종합 점수 = 셋 중 최댓값
  6) 기준값 : 카메라별 검증 정상(validation/good) 종합 점수 최댓값 → 테스트 판정
  * --reference lot : 기준을 '같은 날짜(로트) 정상'으로 바꿔 보는 분석용 모드
                      (정상은 같은 날짜 다른 정상 대비, 불량은 9/29 정상 대비). 현장에서 로트 시작 때
                      정상 몇 장으로 기준을 다시 잡는 운영을 흉내 낸 것.
출력  : <out>/band_scores.csv, band_summary.md, band_overlay.png (경계 확인용)

실행 예:
  python band_thickness.py --category data/processed/company/company_v2_cam12_polarstrip --out outputs/band_thickness
"""
import argparse
import collections
import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np

INNER, OUTER, RADIAL = 0.75, 1.05, 108            # 키트 company_v2_polarstrip.yaml 과 같아야 함
ANGLES = (5.0, 260.0)
ROW_R = (OUTER - 1.0) / (OUTER - INNER) * (RADIAL - 1)   # 외곽선(1.0R) 예상 줄 ≈ 17.8
LOCAL_WIN = 401                                    # step 판단용 주변 구간 (약 60°)
WIN = 201                                          # 구간 두께 평균 창 (약 30°): 일부 구간만 두꺼운/얇은 불량용
STEP_MIN = 31                                      # step 으로 볼 최소 폭 (약 4.5°)
EDGE_LEVEL = 0.35                                  # 띠/면 경계 밝기 (0=띠, 1=면). 낮을수록 녹·얼룩을 띠로 착각 덜 함
KEYS = ("thick_med", "win_max", "win_min", "local_max")
SMOOTH = 15                                        # 두께 곡선 각도 방향 메디안 (약 2°)


# ---------------------------------------------------------------- 경계 검출
def read_bgr(path):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없음: {path}")
    return img


def detect_edges(bgr):
    """반환: inner(W,), outer(W,) 줄 위치. 못 찾은 칸은 nan.

    안쪽 경계는 '아래쪽 밝은 연마면'에서 위로 올라가다 처음 만나는 어두운 구간(3줄 이상)의 아랫단.
    → 띠 위쪽의 밝은 반사선(외곽 하이라이트)을 연마면으로 착각하지 않음.
    바깥 경계는 그 어두운 띠를 위로 거슬러 올라가 다시 밝아지는 줄(배경·반사선).
    """
    g = cv2.blur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (9, 3))
    H, W = g.shape
    r = np.arange(H)[:, None]
    lo = max(int(ROW_R) - 10, 1)
    bright = np.percentile(g[H // 2:], 75, axis=0)              # 연마면 밝기
    dark = np.percentile(g[lo:], 5, axis=0)                     # 띠 밝기
    thr = dark + EDGE_LEVEL * (bright - dark)
    below = g < thr
    run3 = below.copy()
    run3[2:] &= below[1:-1] & below[:-2]                        # r, r-1, r-2 모두 어두움
    run3[:lo + 2] = False
    has = run3.any(axis=0) & (bright - dark >= 25)
    last = H - 1 - run3[::-1].argmax(axis=0)                    # 가장 아래쪽 어두운 구간의 아랫단
    nxt = np.clip(last + 1, 0, H - 1)
    cols = np.arange(W)
    a, b = g[last, cols], g[nxt, cols]
    inner = last + np.clip((thr - a) / np.maximum(b - a, 1e-3), 0, 1)
    inner = np.where(has & (last < H - 2), inner, np.nan)

    li = np.where(np.isfinite(inner), last, H - 1)
    # 띠 안에서 가장 어두운 줄부터 위로 올라가다 배경·반사선으로 밝아지는 줄
    gv = cv2.blur(g, (1, 5))
    inband = (r >= lo) & (r <= li)
    dark_i = np.where(inband, gv, np.inf).argmin(axis=0)
    bandmask = (r >= dark_i - 2) & (r <= li - 1)
    band = np.nanmedian(np.where(bandmask, g, np.nan), axis=0)
    band = np.where(np.isfinite(band), band, g[dark_i, cols])
    k = np.maximum(6.0, 0.35 * band)
    up = (r <= dark_i) & (g > band + k)
    any_up = up.any(axis=0)
    top = H - 1 - up[::-1].argmax(axis=0)
    outer = np.where(any_up & np.isfinite(inner), top + 0.5, np.nan)
    return inner.astype(np.float32), outer.astype(np.float32)


def fit_outer(outer, iters=6):
    """외곽선은 원 → a + b·cos + c·sin 으로 맞춤 (반복 가중으로 튀는 값 무시)."""
    th = np.deg2rad(np.linspace(*ANGLES, outer.size))
    A = np.stack([np.ones_like(th), np.cos(th), np.sin(th)], 1)
    ok = np.isfinite(outer)
    if ok.mean() < 0.15:
        return np.full(outer.size, ROW_R, np.float32), float(ok.mean())
    w = np.ones(ok.sum())
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[ok] * w[:, None], outer[ok] * w, rcond=None)
        res = outer[ok] - A[ok] @ coef
        s = 1.4826 * np.median(np.abs(res)) + 0.3
        w = 1 / np.maximum(1, np.abs(res) / (2.5 * s))
    return (A @ coef).astype(np.float32), float(ok.mean())


def running_median(x, k):
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, k), axis=1)


def measure(bgr):
    inner, outer = detect_edges(bgr)
    fit, outer_valid = fit_outer(outer)
    t = inner - fit
    nan_frac = float(np.isnan(t).mean())
    t = np.where(np.isfinite(t), t, np.nanmedian(t) if np.isfinite(t).any() else 0)
    t = running_median(t, SMOOTH)
    local = t - running_median(t, LOCAL_WIN)
    local = np.convolve(local, np.ones(STEP_MIN) / STEP_MIN, mode="same")   # 짧은 튐(경계 오검출) 무시
    win = np.convolve(np.pad(t, WIN // 2, mode="edge"), np.ones(WIN) / WIN, mode="valid")
    return {
        "thick_med": float(np.median(t)), "thick_p2": float(np.percentile(t, 2)),
        "thick_p98": float(np.percentile(t, 98)), "local_max": float(np.abs(local).max()),
        "win_max": float(win.max()), "win_min": float(win.min()),
        "outer_valid": outer_valid, "inner_missing": nan_frac,
    }, (inner, fit, t)


def z_scores(x, base):
    """기준(base: 특징별 (중앙값, 편차)) 대비 벗어난 정도.
    over  : 둘레 전체가 두꺼움 또는 30° 구간이 두꺼움 중 큰 쪽
    미흡  : 둘레 전체가 얇음 또는 30° 구간이 얇음 중 큰 쪽
    step  : 주변보다 짧은 구간이 급변"""
    z = lambda k, sign=1: sign * (x[k] - base[k][0]) / base[k][1]
    return {"over_z": max(z("thick_med"), z("win_max")),
            "thin_z": max(z("thick_med", -1), z("win_min", -1)),
            "step_z": z("local_max")}


# ---------------------------------------------------------------- 평가 도구
def robust(v):
    v = np.asarray(v, float)
    m = np.median(v)
    return m, 1.4826 * np.median(np.abs(v - m)) + 1e-6


def auroc(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(neg) * len(pos)))


def date_of(name):
    m = re.search(r"PLC-(\d{8})-", name)
    return f"{int(m[1][4:6])}/{int(m[1][6:8])}" if m else "?"


def date_key(d):
    return tuple(int(x) for x in d.split("/")) if "/" in d else (99, 99)


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser(description="외면취 띠 두께 측정 판정")
    ap.add_argument("--category", type=Path, required=True, help="..._polarstrip 카테고리 폴더")
    ap.add_argument("--out", type=Path, default=Path("outputs/band_thickness"))
    ap.add_argument("--reference", choices=["global", "lot"], default="global",
                    help="global=학습 정상 전체 기준(기본), lot=같은 날짜 정상 기준")
    ap.add_argument("--exclude-z", type=float, default=0.0,
                    help="같은 날짜 정상 두께에서 이 z 이상 벗어난 정상을 기준·평가에서 제외 (0=안 함, 예: 3)")
    args = ap.parse_args()
    cat = args.category
    with (cat / "split.csv").open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["prepared_image"]]
    for r in rows:
        r["date"] = date_of(r["file"])

    feats, curves, times = {}, {}, []
    for i, r in enumerate(rows, 1):
        t0 = time.perf_counter()
        bgr = read_bgr(cat / r["prepared_image"])
        feats[r["prepared_image"]], curves[r["prepared_image"]] = measure(bgr)
        times.append(time.perf_counter() - t0)
        if i % 100 == 0:
            print(f"  측정 {i}/{len(rows)}")
    ms = 1000 * float(np.median(times))

    # --- 점수
    keys = KEYS
    cams = sorted({r["camera"] for r in rows})

    # --- 날짜(로트)별 정상 두께에서 크게 벗어난 정상 제외
    excluded = set()
    by_date = collections.defaultdict(list)
    for r in rows:
        if r["class"] == "good":
            by_date[(r["camera"], r["date"])].append(r)
    exc_rows = []
    if args.exclude_z > 0:
        for (cam, dt), L0 in by_date.items():
            if len(L0) < 5:
                continue
            m, s = robust([feats[r["prepared_image"]]["thick_med"] for r in L0])
            for r in L0:
                z = (feats[r["prepared_image"]]["thick_med"] - m) / s
                if abs(z) > args.exclude_z:
                    excluded.add(r["prepared_image"])
                    exc_rows.append({"camera": cam, "date": dt, "split": r["split"], "file": r["file"],
                                     "thick_med": round(feats[r["prepared_image"]]["thick_med"], 2),
                                     "date_median": round(m, 2), "z": round(z, 2)})

    gbase = {}
    for cam in cams:
        tr = [feats[r["prepared_image"]] for r in rows
              if r["split"] == "train" and r["class"] == "good" and r["camera"] == cam
              and r["prepared_image"] not in excluded]
        gbase[cam] = {k: robust([x[k] for x in tr]) for k in keys}

    goods = collections.defaultdict(list)
    for r in rows:
        if r["class"] == "good" and r["prepared_image"] not in excluded:
            goods[(r["camera"], r["date"])].append(r)

    recs = []
    for r in rows:
        x = feats[r["prepared_image"]]
        if args.reference == "global":
            base = gbase[r["camera"]]
        else:  # 같은 날짜(로트) 정상 기준, 자기 자신 제외. 불량은 9/29 정상 기준.
            ref_date = r["date"] if r["class"] == "good" else "9/29"
            peers = [p for p in goods[(r["camera"], ref_date)] if p is not r]
            if len(peers) < 5:
                peers = [p for p in rows if p["split"] == "train" and p["class"] == "good"
                         and p["camera"] == r["camera"] and p["prepared_image"] not in excluded]
            base = {k: robust([feats[p["prepared_image"]][k] for p in peers]) for k in keys}
        z = z_scores(x, base)
        recs.append({
            "split": r["split"], "class": r["class"], "camera": r["camera"], "date": r["date"],
            "file": r["file"], **{k: round(v, 3) for k, v in x.items()},
            **{k: round(v, 2) for k, v in z.items()}, "score": round(max(z.values()), 3),
            "excluded": int(r["prepared_image"] in excluded), "_key": r["prepared_image"],
        })

    thr = {}
    for cam in cams:
        v = [x["score"] for x in recs if x["split"] == "validation" and x["class"] == "good"
             and x["camera"] == cam and not x["excluded"]]
        thr[cam] = max(v)
    for x in recs:
        x["threshold"] = round(thr[x["camera"]], 3)
        x["pred_defect"] = int(x["score"] > thr[x["camera"]])

    args.out.mkdir(parents=True, exist_ok=True)
    fields = [k for k in recs[0] if not k.startswith("_")]
    with (args.out / "band_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    if exc_rows:
        with (args.out / "excluded_normals.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(exc_rows[0]))
            w.writeheader()
            w.writerows(sorted(exc_rows, key=lambda e: (e["camera"], date_key(e["date"]), -abs(e["z"]))))

    # --- 요약
    test_all = [x for x in recs if x["split"] == "test"]
    test = [x for x in test_all if not x["excluded"]]
    L = [f"# 외면취 띠 두께 판정 ({cat.name}, 기준={args.reference}, 제외 z>{args.exclude_z:g})", "",
         f"- 제외한 정상: {len(excluded)}장 (같은 날짜 정상 두께에서 z>{args.exclude_z:g}) → excluded_normals.csv"
         if args.exclude_z > 0 else "- 정상 제외 없음",
         f"  - 제외 정상을 과검으로 쳤을 때 테스트 과검: "
         + ", ".join(f"{cam} {sum(x['pred_defect'] or x['excluded'] for x in test_all if x['class'] == 'good' and x['camera'] == cam)}"
                     f"/{sum(1 for x in test_all if x['class'] == 'good' and x['camera'] == cam)}" for cam in cams)
         if args.exclude_z > 0 else "",
         f"- 판정 시간: 이미지당 약 {ms:.0f} ms (CPU, 읽기 포함, 중앙값)",
         "- 기준값(검증 정상 점수 최댓값): " + ", ".join(f"{c} {v:.2f}" for c, v in thr.items()), "",
         "## 테스트 종합", "",
         "| 카메라 | AUROC 전체 | AUROC 9/29 정상만 | 과검률 | 미검률 |", "|---|---|---|---|---|"]
    for cam in cams + ["all"]:
        s = [x for x in test if cam == "all" or x["camera"] == cam]
        g = [x["score"] for x in s if x["class"] == "good"]
        g9 = [x["score"] for x in s if x["class"] == "good" and x["date"] == "9/29"]
        d = [x["score"] for x in s if x["class"] != "good"]
        fa = sum(x["pred_defect"] for x in s if x["class"] == "good")
        mi = sum(1 - x["pred_defect"] for x in s if x["class"] != "good")
        L.append(f"| {cam} | {auroc(g, d):.3f} | {auroc(g9, d):.3f} (정상 {len(g9)}) | "
                 f"{fa}/{len(g)} ({fa / max(len(g), 1):.1%}) | {mi}/{len(d)} ({mi / max(len(d), 1):.1%}) |")

    L += ["", "## 불량 종류별 (해당 특징 하나만 썼을 때 AUROC, 테스트+검증 불량 vs 테스트 정상)", "",
          "| 카메라 | over (두께↑) | 미흡 (두께↓) | step (국소 변화) | 판정 검출 over / step / 미흡 |",
          "|---|---|---|---|---|"]
    for cam in cams:
        g = [x for x in test if x["class"] == "good" and x["camera"] == cam]
        dd = {c: [x for x in recs if x["class"] == c and x["camera"] == cam and x["split"] != "train"] for c in ("over", "미흡", "step")}
        a_over = auroc([x["over_z"] for x in g], [x["over_z"] for x in dd["over"]])
        a_thin = auroc([x["thin_z"] for x in g], [x["thin_z"] for x in dd["미흡"]])
        a_step = auroc([x["step_z"] for x in g], [x["step_z"] for x in dd["step"]])
        det = " / ".join(f"{sum(x['pred_defect'] for x in test if x['class'] == c and x['camera'] == cam)}"
                         f"/{sum(1 for x in test if x['class'] == c and x['camera'] == cam)}" for c in ("over", "step", "미흡"))
        L.append(f"| {cam} | {a_over:.3f} (n={len(dd['over'])}) | {a_thin:.3f} (n={len(dd['미흡'])}) | "
                 f"{a_step:.3f} (n={len(dd['step'])}) | {det} |")

    L += ["", "## 날짜별 정상 띠 두께 (전체 정상, 중앙값 ± 편차, 띠 줄 단위)", "",
          "| 날짜 | camera_1 | camera_2 |", "|---|---|---|"]
    dates = sorted({x["date"] for x in recs if x["class"] == "good"}, key=date_key)
    for dt in dates:
        cells = []
        for cam in cams:
            v = [x["thick_med"] for x in recs if x["class"] == "good" and x["date"] == dt and x["camera"] == cam]
            cells.append(f"{np.median(v):.1f} ± {np.std(v):.1f} ({len(v)})" if v else "—")
        L.append(f"| {dt} | " + " | ".join(cells) + " |")
    for cls in ("over", "step", "미흡"):
        cells = []
        for cam in cams:
            v = [x["thick_med"] for x in recs if x["class"] == cls and x["camera"] == cam]
            cells.append(f"{np.median(v):.1f} ± {np.std(v):.1f} ({len(v)})" if v else "—")
        L.append(f"| 9/29 {cls} | " + " | ".join(cells) + " |")

    L += ["", "## 날짜별 과검률 (테스트 정상)", "", "| 날짜 | camera_1 | camera_2 |", "|---|---|---|"]
    for dt in sorted({x["date"] for x in test if x["class"] == "good"}, key=date_key):
        cells = []
        for cam in cams:
            g = [x for x in test if x["class"] == "good" and x["date"] == dt and x["camera"] == cam]
            cells.append(f"{sum(x['pred_defect'] for x in g)}/{len(g)}" if g else "—")
        L.append(f"| {dt} | " + " | ".join(cells) + " |")
    (args.out / "band_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    # --- 경계 확인 그림: 카메라·종류별 테스트 1장씩 + 점수 높은 정상 2장
    pick = []
    for cam in cams:
        for c in ("over", "step", "미흡"):
            pick += [x for x in test if x["class"] == c and x["camera"] == cam][:1]
        pick += sorted([x for x in test if x["class"] == "good" and x["camera"] == cam], key=lambda x: -x["score"])[:2]
    tiles = []
    for x in pick:
        img = read_bgr(cat / x["_key"])
        inner, fit, _ = curves[x["_key"]]
        cols = np.arange(img.shape[1])
        ok = np.isfinite(inner)
        img[np.clip(np.round(inner[ok]).astype(int), 0, RADIAL - 1), cols[ok]] = (0, 0, 255)
        img[np.clip(np.round(fit).astype(int), 0, RADIAL - 1), cols] = (0, 255, 0)
        img = cv2.resize(img, (1728, 162), interpolation=cv2.INTER_NEAREST)
        lab = np.full((24, 1728, 3), 255, np.uint8)
        name = {"미흡": "insufficient"}.get(x["class"], x["class"])
        cv2.putText(lab, f"{x['camera']} {name} {x['date']}  score {x['score']:.1f} (thr {thr[x['camera']]:.1f})"
                         f"  thick {x['thick_med']:.1f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        tiles += [lab, img]
    if tiles:
        cv2.imencode(".png", np.vstack(tiles))[1].tofile(str(args.out / "band_overlay.png"))
    print(f"\n저장: {args.out.resolve()}")


if __name__ == "__main__":
    main()
