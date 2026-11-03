"""회사 데이터 결과 정리: 테스트셋 추론 → 지표 표 + 원본 | 히트맵 | 겹침 사례 그림.

학습(train_draem.py)과 threshold(kfold_threshold.py)가 끝난 뒤 한 번 실행하면
발표 자료에 넣을 표와 그림이 한 폴더에 모입니다.

예시:
    .\\.venv\\Scripts\\python.exe scripts/draem/make_report.py --categories camera_1 camera_2 cam12

    # MVTec bottle (결과는 따로 모음)
    .\\.venv\\Scripts\\python.exe scripts/draem/make_report.py --data-root data/raw/mvtec_ad --categories bottle --out outputs/report/mvtec

    # 체크포인트를 직접 지정하려면 (categories 순서대로)
    ... --checkpoints outputs/draem/camera_1/a.pt outputs/draem/camera_2/b.pt outputs/draem/cam12/c.pt

    체크포인트를 안 주면 outputs/draem/<category>/ 의 가장 최근 .pt를,
    threshold JSON을 안 주면 outputs/draem/<category>/threshold/ 의 가장 최근 .json을 씁니다.

결과 (outputs/report/company/):
    summary.md                 발표 표 그대로 (데이터셋 × AUROC · 과검률 · 미검률 · 추론시간) + 유형별
    summary.csv                같은 내용 CSV
    <category>/predictions.csv 이미지별 점수 (run_draem_batch.py와 같은 컬럼)
    <category>/cases_typical.png   정상 · over · step · insufficient 대표 사례 (행) × 원본 | 히트맵 | 겹침 (열)
    <category>/cases_hardest.png   가장 헷갈린 사례 (점수가 가장 높은 정상, 가장 낮은 불량)
    <category>/cases/*.png         사례 한 장씩
"""

from pathlib import Path
import argparse
import csv
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_draem import evaluate  # noqa: E402

CLASS_ORDER = ["good", "over", "step", "insufficient"]
CSV_FIELDS = ["image_id", "label", "defect_type", "variation", "level", "anomaly_score", "inference_ms"]
PANEL = 320   # 그림 한 칸 크기(px)
GAP = 8


def resolve(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def newest(folder: Path, pattern: str, exclude=("_last", "timing", "debug")):
    files = [p for p in folder.glob(pattern) if not any(x in p.stem for x in exclude)]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


# ── 그림 ────────────────────────────────────────────────────

def panels(image: np.ndarray, amap: np.ndarray) -> np.ndarray:
    """image: (H, W, 3) BGR 0~1, amap: (H, W) 0~1 → 원본 | 히트맵 | 겹침 한 줄."""
    orig = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap((np.clip(amap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.55, heat, 0.45, 0)
    tiles = [cv2.resize(t, (PANEL, PANEL), interpolation=cv2.INTER_LINEAR) for t in (orig, heat, overlay)]
    gap = np.full((PANEL, GAP, 3), 255, np.uint8)
    return np.hstack([tiles[0], gap, tiles[1], gap, tiles[2]])


def with_header(row: np.ndarray, text: str, color=(40, 40, 40)) -> np.ndarray:
    header = np.full((40, row.shape[1], 3), 255, np.uint8)
    cv2.putText(header, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return np.vstack([header, row])


def column_titles(width: int) -> np.ndarray:
    bar = np.full((36, width, 3), 255, np.uint8)
    for i, t in enumerate(["original", "heatmap", "overlay"]):
        x = i * (PANEL + GAP) + PANEL // 2 - 50
        cv2.putText(bar, t, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 2, cv2.LINE_AA)
    return bar


def grid(rows: list[np.ndarray]) -> np.ndarray:
    width = rows[0].shape[1]
    space = np.full((16, width, 3), 255, np.uint8)
    parts = [column_titles(width)]
    for r in rows:
        parts += [r, space]
    return np.vstack(parts[:-1])


# ── 카테고리 하나 처리 ──────────────────────────────────────

def run_category(model, test_dir: Path, out_dir: Path, threshold: float | None):
    """model.predict(path) → {image_score, anomaly_map, image} 를 쓰는 어떤 모델이든 동작."""
    paths = sorted(test_dir.glob("*/*.png"))
    if not paths:
        raise SystemExit(f"테스트 이미지가 없습니다: {test_dir}")

    model.predict(paths[0])  # 워밍업
    records, cache = [], {}
    for p in paths:
        start = time.perf_counter()
        r = model.predict(p)
        ms = (time.perf_counter() - start) * 1000
        dtype = p.parent.name
        records.append({
            "image_id": p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix(),
            "label": 0 if dtype == "good" else 1, "defect_type": dtype,
            "variation": "original", "level": 0,
            "anomaly_score": r["image_score"], "inference_ms": round(ms, 3),
        })
        amap = r["anomaly_map"]
        amap = amap.numpy() if hasattr(amap, "numpy") else np.asarray(amap)
        cache[p] = (r["image"], amap, r["image_score"], dtype)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(records)

    # 사례 고르기: 대표(점수 중앙값) / 가장 헷갈린 것(정상은 최고점, 불량은 최저점)
    def verdict(score, dtype):
        if threshold is None:
            return ""
        flagged = score > threshold
        if dtype == "good":
            return "  FALSE ALARM" if flagged else "  OK"
        return "  OK" if flagged else "  MISS"

    case_dir = out_dir / "cases"
    case_dir.mkdir(exist_ok=True)
    typical_rows, hardest_rows = [], []
    present = {v[3] for v in cache.values()}
    classes = [c for c in CLASS_ORDER if c in present] + sorted(present - set(CLASS_ORDER))
    for cls in classes:  # 회사: good · over · step · insufficient / MVTec: good · broken_large · …
        items = sorted([(v[2], p) for p, v in cache.items() if v[3] == cls])
        median = items[len(items) // 2][1]
        hardest = items[-1][1] if cls == "good" else items[0][1]
        for kind, p, bucket in (("typical", median, typical_rows), ("hardest", hardest, hardest_rows)):
            image, amap, score, _ = cache[p]
            row = with_header(panels(image, amap), f"{cls}  score {score:.3f}{verdict(score, cls)}  ({p.stem[-20:]})")
            bucket.append(row)
            cv2.imwrite(str(case_dir / f"{kind}_{cls}_{p.stem}.png"), row)

    cv2.imwrite(str(out_dir / "cases_typical.png"), grid(typical_rows))
    cv2.imwrite(str(out_dir / "cases_hardest.png"), grid(hardest_rows))
    return csv_path


# ── 표 ──────────────────────────────────────────────────────

def write_summary(results: list[dict], out_root: Path):
    keys = []
    for r in results:
        keys += [k for k in r if k not in keys]
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    def short(v):
        return v.split(" ")[0] if isinstance(v, str) else "-"

    lines = ["| 데이터셋 | 이미지 AUROC | 과검률 | 미검률 | 추론 ms/장 |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['dataset']} | {r['image_auroc']:.3f} | {short(r.get('false_alarm'))} | "
                     f"{short(r.get('miss'))} | {r['inference_ms']:.1f} |")
    lines += ["", "| 데이터셋 | 유형 | AUROC | 미검률 |", "|---|---|---|---|"]
    for r in results:
        for t in [k[6:] for k in r if k.startswith("auroc_")]:
            if f"auroc_{t}" in r:
                lines.append(f"| {r['dataset']} | {t} | {r[f'auroc_{t}']:.3f} | {short(r.get(f'miss_{t}'))} |")
    lines += ["", "| 데이터셋 | threshold | 체크포인트 |", "|---|---|---|"]
    for r in results:
        th = f"{r['threshold']:.4f}" if "threshold" in r else "없음"
        lines.append(f"| {r['dataset']} | {th} | {r.get('checkpoint', '')} |")
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── 메인 ────────────────────────────────────────────────────

def main():
    import json

    parser = argparse.ArgumentParser(description="회사 데이터 결과 표 + 원본 | 히트맵 | 겹침 사례 그림")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/company_plc"))
    parser.add_argument("--categories", nargs="+", default=["camera_1", "camera_2", "cam12"])
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=None)
    parser.add_argument("--threshold-jsons", type=Path, nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/report/company"))
    args = parser.parse_args()

    from src.models.draem.model import DRAEMPredictor  # torch는 여기서만 필요

    out_root = resolve(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    results = []

    for i, cat in enumerate(args.categories):
        ckpt = resolve(args.checkpoints[i]) if args.checkpoints else newest(ROOT / "outputs/draem" / cat, "*.pt")
        if ckpt is None or not ckpt.is_file():
            print(f"[건너뜀] {cat}: 체크포인트가 없습니다.")
            continue
        tj = resolve(args.threshold_jsons[i]) if args.threshold_jsons else newest(ROOT / "outputs/draem" / cat / "threshold", "*.json")
        threshold = float(json.loads(tj.read_text(encoding="utf-8"))["threshold"]) if tj and tj.is_file() else None

        print(f"\n=== {cat} ===")
        print("체크포인트:", ckpt.name)
        print("threshold:", f"{threshold:.4f} ({tj.name})" if threshold is not None else "없음 (과검률 · 미검률 생략)")

        model = DRAEMPredictor(ckpt)
        csv_path = run_category(model, resolve(args.data_root) / cat / "test", out_root / cat, threshold)
        r = evaluate(csv_path, threshold)
        r["dataset"] = cat
        r["checkpoint"] = ckpt.name
        results.append(r)
        print(f"AUROC {r['image_auroc']:.3f} · 과검률 {r.get('false_alarm', '-')} · "
              f"미검률 {r.get('miss', '-')} · {r['inference_ms']:.1f} ms/장")
        del model

    if results:
        write_summary(results, out_root)
        print(f"\n완료: {out_root}")
        print("  summary.md · summary.csv · <category>/cases_typical.png · cases_hardest.png")


if __name__ == "__main__":
    main()
