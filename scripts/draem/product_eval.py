"""제품 단위 판정: camera_1 과 camera_2 결과를 부품 번호로 묶어 과검률 · 미검률을 냅니다.

현장 규칙: 두 카메라(스프링 양쪽 면)가 모두 정상이어야 정상 제품,
           어느 한쪽이라도 불량이면 불량 제품.

  제품 정답  = 두 이미지 중 하나라도 불량이면 불량
  제품 판정  = 두 이미지 중 하나라도 점수 > 그 카메라의 기준값이면 불량

이미지는 테스트셋(predictions.csv)에 있는 것만 씁니다. 학습에 쓴 정상 이미지는
모델이 이미 봐서 점수가 낮게 나오므로 판정에 넣지 않습니다(그 면은 '확인 안 됨'으로 표시).

예시:
    .\\.venv\\Scripts\\python.exe scripts/draem/product_eval.py
    .\\.venv\\Scripts\\python.exe scripts/draem/product_eval.py --mask edge_ex

결과: <report 폴더>/summary_product.md · products.csv
"""

from pathlib import Path
import argparse
import csv
import json
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]


def resolve(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def newest(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main():
    parser = argparse.ArgumentParser(description="camera_1 + camera_2 제품 단위 과검률 · 미검률")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/company_plc"))
    parser.add_argument("--cameras", nargs=2, default=["camera_1", "camera_2"])
    parser.add_argument("--mask", choices=["edge", "edge_ex"], default=None, help="make_report.py --mask 와 같은 값")
    parser.add_argument("--report-dir", type=Path, default=None, help="기본: outputs/report/company[_<mask>]")
    args = parser.parse_args()

    report = resolve(args.report_dir or Path("outputs/report/company" + (f"_{args.mask}" if args.mask else "")))
    th_key = f"threshold_{args.mask}" if args.mask else "threshold"

    parts = defaultdict(dict)   # part_id → camera → dict
    thresholds = {}
    for cam in args.cameras:
        manifest = {Path(r["image_id"]).stem: r for r in
                    csv.DictReader(open(resolve(args.data_root) / cam / "manifest.csv", encoding="utf-8-sig"))}
        tj = newest(ROOT / "outputs/draem" / cam / "threshold", "*.json")
        if tj is None or th_key not in json.loads(tj.read_text(encoding="utf-8")):
            raise SystemExit(f"{cam}: {th_key}가 있는 threshold JSON이 없습니다 (kfold_threshold.py 먼저)")
        thresholds[cam] = float(json.loads(tj.read_text(encoding="utf-8"))[th_key])
        pred_csv = report / cam / "predictions.csv"
        if not pred_csv.is_file():
            raise SystemExit(f"{pred_csv} 가 없습니다 (make_report.py 먼저)")
        tested = set()
        for p in csv.DictReader(open(pred_csv, encoding="utf-8-sig")):
            stem = Path(p["image_id"]).stem
            m = manifest[stem]
            score = float(p["anomaly_score"])
            parts[m["part_id"]][cam] = {"type": m["defect_type"], "score": score,
                                         "flag": score > thresholds[cam], "tested": True}
            tested.add(stem)
        for stem, m in manifest.items():  # 학습에 쓴 이미지: 정답만 기록, 판정에는 안 씀
            if stem not in tested:
                parts[m["part_id"]].setdefault(cam, {"type": m["defect_type"], "score": None,
                                                     "flag": False, "tested": False})

    rows, by_type = [], defaultdict(lambda: [0, 0])  # type → [제품 수, 미검 수]
    normal_total = normal_fa = ng_total = ng_miss = lucky = 0
    for pid, cams in sorted(parts.items()):
        types = {c: v["type"] for c, v in cams.items()}
        defect_types = sorted({t for t in types.values() if t != "good"})
        is_ng = bool(defect_types)
        judged_ng = any(v["flag"] for v in cams.values() if v["tested"])
        caught_on_defect = any(v["flag"] for v in cams.values() if v["tested"] and v["type"] != "good")
        ptype = "+".join(defect_types) if is_ng else "good"
        if is_ng:
            ng_total += 1
            by_type[ptype][0] += 1
            if not judged_ng:
                ng_miss += 1
                by_type[ptype][1] += 1
            elif not caught_on_defect:
                lucky += 1
        else:
            normal_total += 1
            normal_fa += judged_ng
        rows.append({"part_id": pid, "product_type": ptype, "judged": "NG" if judged_ng else "OK",
                     **{f"{c}_type": cams.get(c, {}).get("type", "") for c in args.cameras},
                     **{f"{c}_score": ("" if cams.get(c, {}).get("score") is None else f"{cams[c]['score']:.4f}")
                        for c in args.cameras}})

    with open(report / "products.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    pct = lambda a, b: f"{a}/{b} ({a / b * 100:.1f}%)" if b else "계산 불가 (0개)"  # noqa: E731
    lines = [
        f"# 제품 단위 판정 ({'마스크 ' + args.mask if args.mask else '마스크 없음'})", "",
        "| 항목 | 값 |", "|---|---|",
        f"| 제품 수 | {len(parts)} (정상 제품 {normal_total} · 불량 제품 {ng_total}) |",
        f"| 기준값 | " + " · ".join(f"{c} {t:.4f}" for c, t in thresholds.items()) + " |",
        f"| 제품 과검률 | {pct(normal_fa, normal_total)} |",
        f"| 제품 미검률 | {pct(ng_miss, ng_total)} |",
        f"| 불량 제품인데 정상 면 점수로 NG가 된 경우 | {lucky} |",
        "", "| 불량 제품 유형 | 제품 수 | 미검 |", "|---|---|---|",
    ] + [f"| {t} | {n} | {m} |" for t, (n, m) in sorted(by_type.items())]
    (report / "summary_product.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n완료: {report / 'summary_product.md'} · products.csv")
    if normal_total == 0:
        print("※ 정상 제품(두 면 모두 정상)이 데이터에 없어 제품 과검률은 계산할 수 없습니다.")


if __name__ == "__main__":
    main()
