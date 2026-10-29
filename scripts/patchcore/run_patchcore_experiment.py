
# [이 스크립트의 역할] 여러 카테고리 × coreset 비율로 PatchCore 전체 과정을 한 번에 실행하고 요약
# 카테고리마다 아래 단계를 순서대로 실행한다. 결과 파일이 이미 있는 단계는 건너뛴다.
#   1) build_patchcore_memory.py       정상 특징 수집
#   2) build_patchcore_coreset.py      coreset (비율별)
#   3) run_patchcore_batch.py          테스트셋 추론
#   4) evaluate_predictions.py         AUROC, 추론시간
#   5) compute_patchcore_threshold.py  threshold (비율별)
#   6) evaluate_patchcore_threshold.py 과검률, 미검률, F1
#
# 실행 예:
#   python scripts/patchcore/run_patchcore_experiment.py --name company_coreset
#       --categories company_cam1 company_cam2 company_cam12 --pcts 0.01 0.1 0.25
#   (분할 반복) --categories company_cam1_s1 company_cam1_s2 ...  ← prepare_company_dataset.py --seed N 으로 먼저 생성
#
#   (표만 다시 만들기) python scripts/patchcore/run_patchcore_experiment.py --name company_coreset --render-only
#
# 예측 CSV 이름: 1% → <category>_original.csv, 그 외 → <category>_original_<N>pct.csv
# 요약 결과 (outputs/patchcore/summary/):
#   <name>.csv  계산용 (카테고리·비율별 한 줄, 그래프 스크립트가 읽음)
#   <name>.md   보기용 표 (과검률·미검률을 %와 장수로, 불량 종류별 AUROC 표 포함) + 터미널에도 출력

from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import sys

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PATCHCORE = ROOT / "scripts" / "patchcore"
# 하위 스크립트도 같은 Python(.venv)으로, 한글 출력이 깨지지 않게 실행
PYTHON = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run(args):
    """하위 스크립트 실행. 실패하면 그 출력과 함께 멈춘다."""
    # 터미널에서 "python <스크립트> <인자들>"을 친 것과 같은 동작
    # cwd=ROOT: 레포 최상위에서 실행, capture_output: 출력을 화면 대신 변수로 받음
    result = subprocess.run(
        [PYTHON, *map(str, args)], cwd=ROOT, env=ENV,
        capture_output=True, text=True, encoding="utf-8",
    )
    # returncode 0 = 성공. 실패하면 마지막 출력 2000자를 보여주고 멈춘다
    if result.returncode != 0:
        raise RuntimeError(
            f"실패: {' '.join(map(str, args))}\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return result.stdout


def pct_name(percentage):
    # 0.01 → "1pct", 0.005 → "0p5pct" (build_patchcore_coreset.py와 같은 규칙)
    return f"{percentage * 100:g}pct".replace(".", "p")


def run_one(category, percentage):
    """한 카테고리·비율의 전체 과정을 실행하고 요약 한 줄을 반환한다."""
    # 각 단계는 "결과 파일이 없을 때만" 실행한다
    # → 중간에 멈춰도 다시 실행하면 이어서 진행되고, 이미 만든 결과는 덮어쓰지 않는다
    out_dir = ROOT / "outputs/patchcore" / category

    # 1) 정상 특징 수집 (비율과 무관해서 카테고리당 한 번)
    raw = out_dir / "raw_memory_full.pt"
    if not raw.exists():
        run([PATCHCORE / "build_patchcore_memory.py", "--category", category])

    # 2) coreset (비율마다 파일이 따로 생김)
    coreset = out_dir / f"coreset_raw_memory_full_{pct_name(percentage)}_d128_s0.pt"
    if not coreset.exists():
        run([PATCHCORE / "build_patchcore_coreset.py", "--input", raw,
             "--percentage", percentage])

    # 3) 테스트셋 추론 → 예측 CSV
    # 1%는 기본 이름(<category>_original.csv)을 써서 run_patchcore_batch.py 단독 실행 결과와 같게 맞춤
    pred_dir = ROOT / "outputs/predictions/patchcore"
    csv_path = pred_dir / (
        f"{category}_original.csv" if percentage == 0.01
        else f"{category}_original_{pct_name(percentage)}.csv"
    )
    if not csv_path.exists():
        run([PATCHCORE / "run_patchcore_batch.py", "--category", category,
             "--memory", coreset, "--output", csv_path])

    # 4) AUROC, 추론시간 (evaluate_predictions.py가 <CSV이름>_metrics.json 으로 저장)
    metrics_path = csv_path.with_name(csv_path.stem + "_metrics.json")
    if not metrics_path.exists():
        run([ROOT / "scripts/evaluation/evaluate_predictions.py", "--csv", csv_path])

    # 5) threshold (추론에 쓴 coreset과 같은 비율로 계산해야 기준이 맞음)
    threshold_path = out_dir / f"threshold_{pct_name(percentage)}_d128_s0_k10.json"
    if not threshold_path.exists():
        run([PATCHCORE / "compute_patchcore_threshold.py", "--category", category,
             "--percentage", percentage])

    # 6) 과검률, 미검률, F1 — 빠르므로 매번 다시 계산
    run([PATCHCORE / "evaluate_patchcore_threshold.py", "--csv", csv_path,
         "--threshold", threshold_path])
    threshold_metrics_path = csv_path.with_name(csv_path.stem + "_threshold_metrics.json")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = json.loads(threshold_metrics_path.read_text(encoding="utf-8"))
    group = threshold["groups"][0]  # 원본 테스트셋 = ("original", 0) 그룹 하나

    # 요약 한 줄 (요약 CSV의 한 행)
    row = {
        "category": category,
        "pct": percentage,                                   # coreset 비율
        "auroc": round(metrics["image_auroc"], 4),
        "mean_ms": round(metrics["mean_inference_ms"], 2),   # 평균 추론시간 (ms/장)
        "threshold": round(threshold["threshold"], 3),
        "fpr": group["fpr"],                                 # 과검률 (0~1)
        "false_alarm": f"{group['false_alarm']}/{group['normal_count']}",  # 과검 장수 예) 1/6
        "fnr": group["fnr"],                                 # 미검률 (0~1)
        "missed": f"{group['missed']}/{group['defect_count']}",            # 미검 장수 예) 32/33
        "f1": round(group["f1"], 4),
        # 참고: 불량을 하나도 놓치지 않으려면 과검률이 얼마인가 (테스트 정답으로 정한 값)
        "ref_zero_miss_fpr": group["reference"]["zero_miss"]["fpr"],
    }

    # 불량 종류별 AUROC (정상 + 해당 종류만 놓고 계산)
    with open(csv_path, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for defect_type in sorted({r["defect_type"] for r in rows} - {"good"}):
        subset = [r for r in rows if r["defect_type"] in ("good", defect_type)]
        row[f"auroc_{defect_type}"] = round(roc_auc_score(
            [int(r["label"]) for r in subset],
            [float(r["anomaly_score"]) for r in subset],
        ), 4)
    return row


def summary_markdown(name, rows):
    """
    요약 결과를 사람이 읽기 쉬운 Markdown 표로 만든다.

    rows는 run_one()이 만든 딕셔너리(숫자) 또는 요약 CSV에서 읽은 딕셔너리(글자) 둘 다 가능.
    표 1: 카테고리·비율별 주요 지표 / 표 2: 불량 종류별 AUROC
    """
    def num(value):
        # 빈칸(해당 없음)은 None, 나머지는 숫자로
        return None if value in ("", None) else float(value)

    def pct(value, count):
        # 0.1666, "1/6" → "16.7% (1/6)"
        value = num(value)
        return "-" if value is None else f"{value * 100:.1f}% ({count})"

    lines = [
        f"# PatchCore 실험 요약: {name}",
        "",
        "판정 규칙: 이상 점수 > threshold → 불량. "
        "threshold = 학습 정상 이미지를 돌아가며 빼고 채점한 점수의 최댓값 (테스트 정답 미사용)",
        "",
        "- 과검률: 정상인데 불량으로 판정한 비율 (괄호 = 장수)",
        "- 미검률: 불량인데 놓친 비율 (괄호 = 장수)",
        "- 추론시간: 이미지 1장 평균 (실행 시점의 GPU 상태에 따라 달라질 수 있음)",
        "",
        "## 주요 지표",
        "",
        "| 카테고리 | coreset | AUROC | 과검률 | 미검률 | F1 | 추론시간 | threshold |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['category']} | {num(r['pct']) * 100:g}% | {num(r['auroc']):.3f} "
            f"| {pct(r['fpr'], r['false_alarm'])} | {pct(r['fnr'], r['missed'])} "
            f"| {num(r['f1']):.3f} | {num(r['mean_ms']):.1f}ms | {num(r['threshold']):.2f} |"
        )

    # 불량 종류별 AUROC (auroc_over, auroc_step ... 컬럼을 모아서 표로)
    # 카테고리마다 불량 종류가 다를 수 있어서 (bottle vs 회사) 있는 것만 모은다
    defect_keys = []
    for r in rows:
        for key in r:
            if key.startswith("auroc_") and key not in defect_keys:
                defect_keys.append(key)
    if defect_keys:
        names = [key[len("auroc_"):] for key in defect_keys]
        lines += [
            "",
            "## 불량 종류별 AUROC (정상 + 해당 불량만 놓고 계산)",
            "",
            "| 카테고리 | coreset | " + " | ".join(names) + " |",
            "|---|---|" + "---|" * len(names),
        ]
        for r in rows:
            values = [num(r.get(key)) for key in defect_keys]
            cells = ["-" if v is None else f"{v:.3f}" for v in values]
            lines.append(f"| {r['category']} | {num(r['pct']) * 100:g}% | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="PatchCore 실험 일괄 실행 + 요약")
    parser.add_argument("--name", required=True, help="요약 CSV 이름")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--pcts", nargs="+", type=float, default=[0.01])
    # --render-only: 실험은 돌리지 않고, 이미 있는 요약 CSV(<name>.csv)로 보기용 표만 다시 만든다
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / "outputs/patchcore/summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{args.name}.csv"

    if args.render_only:
        if not output_path.is_file():
            parser.error(f"요약 CSV가 없습니다: {output_path}")
        with open(output_path, encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    else:
        if not args.categories:
            parser.error("--categories 가 필요합니다 (--render-only 가 아닐 때).")

        # 카테고리 × 비율 조합을 모두 실행 (예: 카테고리 3개 × 비율 3개 = 9줄)
        rows = []
        for category in args.categories:
            for percentage in args.pcts:
                row = run_one(category, percentage)
                rows.append(row)
                print(f"[완료] {category} coreset {percentage * 100:g}%", flush=True)

        # 컬럼 순서: 첫 줄의 순서를 따르고, 다른 줄에만 있는 컬럼은 뒤에
        first = list(rows[0])
        keys = sorted({k for r in rows for k in r},
                      key=lambda k: first.index(k) if k in first else len(first))

        # 계산용 CSV (그래프 스크립트가 이 컬럼 이름으로 읽으므로 형식 유지)
        with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    # 보기용 Markdown 표 (VS Code 미리보기 Ctrl+Shift+V 로 표 형태로 보임)
    markdown = summary_markdown(args.name, rows)
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(markdown, encoding="utf-8")

    # 터미널에도 같은 표 출력
    print()
    print(markdown)
    print("요약 CSV (계산용):", output_path)
    print("요약 표 (보기용):", markdown_path)


# 직접 실행했을 때만 main() 실행
if __name__ == "__main__":
    main()
