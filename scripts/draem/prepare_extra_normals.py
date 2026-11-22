"""추가로 받은 정상 원본(라벨 없음) → 기존 데이터와 같은 방식으로 부품만 잘라 저장.

기존 데이터는 라벨의 바깥 원 좌표로 잘랐지만, 추가 정상은 라벨이 없으므로 원을 직접 찾습니다.
  1) 흰 연마면이 밝고 바깥 외면취 띠가 어두운 원을 넓게 찾고 (축소 이미지)
  2) 각도마다 '밝음 → 어두움'으로 바뀌는 가장 바깥 경계(연마면 끝)에 원을 맞춘 뒤
  3) 바깥 원 반지름 R = 연마면 반지름 / 0.915 (기존 정상 라벨에서 잰 비율)
  4) 기존과 같이 반폭 R × 1.1 정사각형으로 자름
기존 정상 크롭으로 확인한 오차: 중심 약 6px 이내 · 반지름 약 1% 이내.

    python scripts/draem/prepare_extra_normals.py --src C:/정상1 C:/정상2

결과: data/processed/company_plc_extra/<camera_1|camera_2>/good/*.png
     data/processed/company_plc_extra/circles.csv (이미지별 원 · 날짜)
     data/processed/company_plc_extra/_check/check_XX.png (잘린 결과 확인용, 초록 = 링 마스크 범위)
PLC-…_camera_N.png 형식이 아닌 파일은 건너뜁니다.
"""

from pathlib import Path
import argparse
import csv
import re
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "draem"))
from prepare_company_plc import crop_square, imread, imwrite  # noqa: E402

NAME_RE = re.compile(r"^PLC-(\d{8})-\d+-\d+-\d+_camera_([12])$")
FACE_RATIO = 0.915  # 연마면 바깥 경계 반지름 / 바깥 원 반지름 (기존 정상 라벨 기준)


def _ring(g, cx, cy, r, a, b, n_ang=120, n_rad=4):
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    rs = np.linspace(a * r, b * r, n_rad)
    xs = (cx + np.outer(rs, np.cos(ang))).astype(np.float32)
    ys = (cy + np.outer(rs, np.sin(ang))).astype(np.float32)
    return np.median(cv2.remap(g, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), axis=0)


def _score(g, cx, cy, r):
    face, band, out = _ring(g, cx, cy, r, 0.65, 0.86), _ring(g, cx, cy, r, 0.95, 0.99), _ring(g, cx, cy, r, 1.03, 1.08)
    bright = face >= np.percentile(face, 40)
    v = float(np.mean((face - band)[bright])) + 0.5 * float(np.mean(out - band))
    return v if np.isfinite(v) else -1e9


def _fit_ls(pts):
    x, y = pts[:, 0], pts[:, 1]
    sol, *_ = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones_like(x)], x ** 2 + y ** 2, rcond=None)
    return sol[0], sol[1], np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2)


def find_part_circle(image, r_min=280, r_max=420):
    """(cx, cy, R) — R = 바깥 원 반지름."""
    g = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    small = cv2.resize(g, None, fx=0.25, fy=0.25)
    h, w = small.shape
    best, cx, cy, r = -1e9, w / 2, h / 2, (r_min + r_max) / 8
    for y in range(int(h * 0.28), int(h * 0.72), 4):
        for x in range(int(w * 0.28), int(w * 0.72), 4):
            for rr in range(r_min // 4, r_max // 4 + 1, 4):
                s = _score(small, x, y, rr)
                if s > best:
                    best, cx, cy, r = s, x, y, rr
    cx, cy, r = cx * 4.0, cy * 4.0, r * 4.0
    ang = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    for _ in range(3):
        rs = np.arange(0.7 * r, 1.15 * r, 1.0)
        xs = (cx + np.outer(rs, np.cos(ang))).astype(np.float32)
        ys = (cy + np.outer(rs, np.sin(ang))).astype(np.float32)
        prof = cv2.remap(g, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        d = prof[:-4] - prof[4:]                       # 밝음 → 어두움
        mag = d.max(axis=0)
        k = len(d) - 1 - np.argmax((d >= 0.5 * mag[None, :])[::-1], axis=0)  # 가장 바깥 강한 경계
        inner = prof[: max(1, len(rs) // 4)].mean(axis=0)
        ok = (mag >= np.percentile(mag, 40)) & (inner >= np.percentile(inner, 35))
        if ok.sum() < 30:
            break
        pr = rs[k + 2]
        pts = np.c_[cx + pr * np.cos(ang), cy + pr * np.sin(ang)][ok]
        for _ in range(3):
            c = _fit_ls(pts)
            res = np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - c[2])
            keep = res < max(3.0, 2.5 * np.median(res))
            if keep.sum() < 30:
                break
            pts = pts[keep]
        cx, cy, rw = _fit_ls(pts)
        r = rw / FACE_RATIO
    return float(cx), float(cy), float(r)


def main():
    parser = argparse.ArgumentParser(description="추가 정상 원본 → 부품 크롭")
    parser.add_argument("--src", type=Path, nargs="+", required=True)
    parser.add_argument("--dst", type=Path, default=Path("data/processed/company_plc_extra"))
    parser.add_argument("--margin", type=float, default=1.1)
    args = parser.parse_args()
    dst = args.dst if args.dst.is_absolute() else ROOT / args.dst

    paths, skipped = [], []
    for s in args.src:
        for p in sorted(Path(s).glob("*.png")):
            (paths if NAME_RE.match(p.stem) else skipped).append(p)
    for p in skipped:
        print(f"  [건너뜀: 이름 형식 다름] {p}")
    print(f"처리할 이미지: {len(paths)}장")

    rows, thumbs, t0 = [], [], time.time()
    for i, p in enumerate(paths, 1):
        date, cam = NAME_RE.match(p.stem).groups()
        image = imread(p)
        cx, cy, R = find_part_circle(image)
        crop = crop_square(image, cx, cy, int(round(R * args.margin)))
        imwrite(dst / f"camera_{cam}" / "good" / f"{p.stem}.png", crop)
        rows.append({"file": p.name, "camera": f"camera_{cam}", "date": date, "source": p.parent.name,
                     "cx": round(cx, 1), "cy": round(cy, 1), "radius": round(R, 1)})
        v = cv2.resize(crop, (200, 200), interpolation=cv2.INTER_AREA)
        Rv = 100 / args.margin
        for rr in (0.80, 1.03):
            cv2.circle(v, (100, 100), int(Rv * rr), (0, 220, 0), 1, cv2.LINE_AA)
        cv2.putText(v, f"{date[4:]} c{cam} R{R:.0f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        thumbs.append(v)
        if i % 25 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}  ({(time.time() - t0) / i:.1f}초/장)")

    with open(dst / "circles.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    radii = np.array([r["radius"] for r in rows])
    print(f"바깥 원 반지름: 최소 {radii.min():.0f} · 중앙값 {np.median(radii):.0f} · 최대 {radii.max():.0f} px")
    for j in range(0, len(thumbs), 50):  # 확인용: 50장씩 10×5
        chunk = thumbs[j:j + 50] + [np.full((200, 200, 3), 255, np.uint8)] * (-len(thumbs[j:j + 50]) % 10)
        grid = np.vstack([np.hstack(chunk[k:k + 10]) for k in range(0, len(chunk), 10)])
        imwrite(dst / "_check" / f"check_{j // 50 + 1:02d}.png", grid)
    for cam in ("camera_1", "camera_2"):
        print(f"{cam}: {sum(r['camera'] == cam for r in rows)}장 → {dst / cam / 'good'}")
    print(f"확인용 그림: {dst / '_check'}  (초록 원 사이 = 평가에 쓰는 링 범위)")


if __name__ == "__main__":
    main()
