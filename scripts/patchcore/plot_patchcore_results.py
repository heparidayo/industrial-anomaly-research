
# [이 스크립트의 역할] PatchCore 실험 결과 그래프 3개 생성 (발표/보고서용)
#   1_coreset_comparison.png      coreset 비율별 AUROC (bottle + 회사 cam1/cam2/cam1+2) + bottle 추론시간
#   2_split_repeat_auroc.png      학습/테스트 분할 5가지 × 원본 vs ROI crop AUROC
#   3_rotation_overdetection.png  부품만 회전 각도별 과검률 (+ 이미지 전체 회전 비교)
#
# 필요한 결과 (먼저 실행):
#   run_patchcore_experiment.py 로 만든 요약 CSV (outputs/patchcore/summary/)
#     company_coreset.csv, company_roi_coreset.csv, company_seeds.csv, company_roi_seeds.csv
#   evaluate_patchcore_threshold.py 로 만든 정상편차 지표 JSON (outputs/predictions/patchcore/)
#
# 의존성: matplotlib (requirements-patchcore.txt), 한글 글꼴: 맑은 고딕 (Windows 기본)
#
# 실행 예:
#   python scripts/patchcore/plot_patchcore_results.py
# 결과: outputs/patchcore/presentation_figures/charts/*.png

import csv
import json
from pathlib import Path

import matplotlib
# "Agg" = 화면 창을 띄우지 않고 파일로만 그리는 방식 (터미널에서 실행해도 에러 없음)
# 반드시 pyplot을 불러오기 전에 정해야 한다
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 레포 최상위 폴더 (이 파일에서 두 단계 위)
ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "outputs/predictions/patchcore"   # 예측 CSV, 지표 JSON 위치
SUM = ROOT / "outputs/patchcore/summary"        # run_patchcore_experiment.py 요약 CSV 위치
OUT = ROOT / "outputs/patchcore/presentation_figures/charts"  # 그래프 저장 위치
OUT.mkdir(parents=True, exist_ok=True)

# 색상 (그래프 3개 모두 같은 색 규칙을 써서 카메라별 색이 바뀌지 않게 함)
#   INK  : 진한 글자/선 (bottle 선, 강조 글자)
#   INK2 : 보조 글자 (축 이름, 눈금)
#   MUTED: 흐린 회색 (기준선, 보조 설명)
#   GRID : 격자선
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
# 데이터 색: 파랑 / 주황 / 초록 (색약인 사람도 구분 가능한 조합) + 범례용 회색
S1, S2, S3, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
# 카메라 설정별 고정 색 → 1번, 3번 그래프에서 cam1은 항상 파랑
CAM_COLOR = {"cam1": S1, "cam2": S2, "cam1+2": S3}

# matplotlib 전체 기본 스타일 (모든 그래프에 공통 적용)
plt.rcParams.update({
    # 한글 글꼴 + 마이너스 기호 깨짐 방지
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    # 위/오른쪽 테두리 선은 없애서 그래프를 깔끔하게
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    # 흰 배경 (PPT에 그대로 붙이기 좋게)
    "figure.facecolor": "white", "axes.facecolor": "white",
    # 범례 테두리 상자 없음
    "legend.frameon": False,
})


def load(name):
    # 요약 CSV(outputs/patchcore/summary/<name>)를 딕셔너리 리스트로 읽는다
    return list(csv.DictReader(open(SUM / name, encoding="utf-8-sig")))


def jload(p):
    # JSON 파일을 읽는다
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save(fig, name):
    # dpi=200: PPT에서 크게 띄워도 흐리지 않은 해상도
    # bbox_inches="tight": 바깥 여백을 자동으로 잘라냄
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)  # 메모리 정리 (그래프를 여러 개 그릴 때 필요)
    print(OUT / name)


# ---------------------------------------------------------------- 1. coreset
# 왼쪽: coreset 비율별 AUROC (bottle + 회사 3설정)
# 오른쪽: bottle 비율별 평균 추론시간
# ※ 두 값은 단위가 달라서 한 그래프의 y축 두 개로 그리지 않고 옆에 나란히 그린다
def chart_coreset():
    pcts = [1, 10, 25]  # coreset 비율(%)
    bottle_auc, bottle_ms = [], []
    for p in pcts:
        # bottle 1%는 기본 이름, 나머지는 _<N>pct 가 붙은 CSV
        name = "bottle_original.csv" if p == 1 else f"bottle_original_{p}pct.csv"
        # evaluate_predictions.py 결과 JSON에서 AUROC와 평균 추론시간을 가져온다
        m = jload(PRED / name.replace(".csv", "_metrics.json"))
        bottle_auc.append(m["image_auroc"])
        bottle_ms.append(m["mean_inference_ms"])
    # 회사 데이터 비율별 AUROC (요약 CSV에서 카테고리별로 모음)
    # 결과 예: {"cam1": [1% AUROC, 10% AUROC, 25% AUROC], ...}
    rows = load("company_coreset.csv")
    comp = {cam: [float(r["auroc"]) for r in rows if r["category"] == cat]
            for cam, cat in [("cam1", "company_cam1"), ("cam2", "company_cam2"), ("cam1+2", "company_cam12")]}

    # 그래프 2칸 (왼쪽 a를 1.5배 넓게)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    x = range(len(pcts))  # x 위치 0, 1, 2 (눈금 글자만 1% / 10% / 25%로 표시)

    # --- 왼쪽: AUROC 꺾은선
    a.plot(x, bottle_auc, color=INK, lw=2, marker="o", ms=8, label="bottle")
    for cam, v in comp.items():
        a.plot(x, v, color=CAM_COLOR[cam], lw=2, marker="o", ms=8, label=f"회사 {cam}")
    # 0.5 = 무작위 수준 기준선 (점선)
    a.axhline(0.5, color=MUTED, lw=1, ls="--")
    a.text(0.5, 0.507, "무작위(0.5)", ha="center", va="bottom", color=MUTED, fontsize=10)
    # 선 끝에 이름 직접 표시 (범례를 안 봐도 어느 선인지 알 수 있게)
    a.text(2.05, bottle_auc[-1], "bottle", va="center", color=INK, fontsize=11)
    for cam, v in comp.items():
        a.text(2.05, v[-1], cam, va="center", color=INK2, fontsize=11)
    a.set_xticks(list(x), [f"{p}%" for p in pcts])
    a.set_xlim(-0.2, 2.55)  # 오른쪽 끝 이름표가 들어갈 여유
    a.set_ylim(0.3, 1.05)
    a.set_xlabel("coreset 비율 (메모리 뱅크 크기)")
    a.set_ylabel("Image AUROC")
    a.set_title("메모리를 늘려도 나아지지 않는다", loc="left")
    a.legend(loc="lower left", ncol=2, fontsize=10)

    # --- 오른쪽: bottle 추론시간 막대
    # (회사 데이터 추론시간은 측정 시점의 GPU 상태가 달라 비교가 불공정해서 bottle만 표시)
    bars = b.bar(x, bottle_ms, color=S1, width=0.55)
    # 막대 위에 값 표시
    for rect, v in zip(bars, bottle_ms):
        b.text(rect.get_x() + rect.get_width() / 2, v + 0.6, f"{v:.1f}ms",
               ha="center", color=INK, fontsize=11)
    b.set_xticks(list(x), [f"{p}%" for p in pcts])
    b.set_ylim(0, max(bottle_ms) * 1.25)  # 막대 위 글자가 들어갈 여유
    b.set_xlabel("coreset 비율")
    b.set_ylabel("평균 추론시간 (ms/장)")
    b.set_title("bottle 추론시간", loc="left")
    b.grid(axis="x", visible=False)  # 막대그래프는 세로 격자 불필요
    save(fig, "1_coreset_comparison.png")


# ---------------------------------------------------------------- 2. seeds
# 카메라 설정(cam1 / cam2 / cam1+2)마다 점 5개 = 학습/테스트 분할 5가지(seed 0~4)의 AUROC
# 원본(파랑)과 ROI crop(주황)을 나란히 놓고, 가로 막대 = 5회 평균
def chart_seeds():
    # seed 0 결과는 coreset 요약 CSV의 1% 줄에 있다 (category 이름에 _s 가 없음)
    base = {r["category"]: r for r in load("company_coreset.csv") + load("company_roi_coreset.csv")
            if r["pct"] == "0.01"}
    # seed 1~4 결과 (category 이름이 ..._s1 ~ _s4)
    seeds = {"원본": load("company_seeds.csv"), "ROI crop": load("company_roi_seeds.csv")}
    cams = [("cam1", "company_cam1"), ("cam2", "company_cam2"), ("cam1+2", "company_cam12")]
    color = {"원본": S1, "ROI crop": S2}
    # 같은 카메라 위치에서 원본은 왼쪽, ROI는 오른쪽으로 살짝 비켜 그림
    offset = {"원본": -0.15, "ROI crop": 0.15}

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for kind, rows in seeds.items():
        suf = "" if kind == "원본" else "_roi"
        for i, (cam, cat) in enumerate(cams):
            c = cat + suf
            # seed 0 + seed 1~4 = AUROC 5개
            vals = [float(base[c]["auroc"])] + [float(r["auroc"]) for r in rows
                                                if r["category"].startswith(c + "_s")]
            xs = [i + offset[kind]] * len(vals)
            # 분할 1회 = 점 하나 (반투명 + 흰 테두리 → 겹친 점도 구분)
            # 범례는 첫 카메라에서만 한 번 등록 (i == 0)
            ax.scatter(xs, vals, s=60, color=color[kind], alpha=0.55,
                       edgecolor="white", linewidth=1.5, zorder=3,
                       label=f"{kind} (분할 1회)" if i == 0 else None)
            # 5회 평균 = 짧은 가로 막대
            mean = sum(vals) / len(vals)
            ax.plot([i + offset[kind] - 0.1, i + offset[kind] + 0.1], [mean, mean],
                    color=color[kind], lw=3, zorder=4,
                    label=f"{kind} 평균" if i == 0 else None)
            # 아래쪽에 평균 숫자 표시
            ax.text(i + offset[kind], 0.27, f"평균 {mean:.2f}", ha="center",
                    color=INK2, fontsize=10)
    # 기준선: 0.5(무작위), 1.0(bottle 성능)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(2.45, 0.505, "무작위(0.5)", va="bottom", color=MUTED, fontsize=10)
    ax.axhline(1.0, color=INK, lw=1, ls=":")
    ax.text(2.45, 1.005, "bottle(1.0)", va="bottom", color=INK, fontsize=10)
    ax.set_xticks(range(3), [c for c, _ in cams])
    ax.set_xlim(-0.5, 2.85)
    ax.set_ylim(0.22, 1.05)  # 아래쪽 "평균" 글자가 들어갈 여유
    ax.set_ylabel("Image AUROC")
    ax.set_title("학습/테스트 분할 5가지 × 원본 vs ROI crop", loc="left")
    ax.grid(axis="x", visible=False)
    # 범례를 bottle(1.0) 선 바로 아래에 배치 (선과 겹치지 않게)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93), ncol=2, fontsize=10)
    save(fig, "2_split_repeat_auroc.png")


# ---------------------------------------------------------------- 3. rotation
def fpr_by_abs_angle(path, variation):
    # 정상편차 지표 JSON → {|각도|: 과검률(%)}
    # +각도와 -각도 결과를 합쳐서 하나의 점으로 만든다
    #   예) +30° 과검 5/6, -30° 과검 6/6 → ±30° = 11/12 = 91.7%
    # variation: 볼 편차 이름 ("rotation_part" = 부품만 회전, "rotation" = 이미지 전체 회전)
    #   같은 JSON에 밝기/이동 결과도 있어서, 이름으로 걸러야 한다
    g = jload(path)["groups"]
    agg = {}  # {|각도|: (과검 수 합, 정상 수 합)}
    for e in g:
        if e["variation"] not in ("original", variation):
            continue
        key = 0 if e["variation"] == "original" else abs(e["level"])
        fa, n = agg.get(key, (0, 0))
        agg[key] = (fa + e["false_alarm"], n + e["normal_count"])
    return {k: fa / n * 100 for k, (fa, n) in agg.items()}


# x축: 회전 각도 (원본, ±2, ±5, ±10, ±30, ±90, 180)
# 꺾은선: 부품만 회전했을 때 과검률 (카메라 설정별)
# X 표시: 이미지 전체를 ±10° 돌렸을 때 과검률 (조장 preprocess.py 방식, 비교용)
def chart_rotation():
    angles = [0, 2, 5, 10, 30, 90, 180]
    # ROI 데이터 기준 (부품만 회전 실험은 ROI 카테고리에서 진행)
    cams = [("cam1", "company_cam1_roi"), ("cam2", "company_cam2_roi"), ("cam1+2", "company_cam12_roi")]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = range(len(angles))  # 각도 간격이 고르지 않아서 x 위치는 0~6 순번으로 균등 배치
    for cam, cat in cams:
        # 부품만 회전 결과 (preprocess_part_rotation.py → run_patchcore_variation.py → 평가)
        part = fpr_by_abs_angle(PRED / f"{cat}_{cat}_partrot_v1_threshold_metrics.json", "rotation_part")
        ax.plot(x, [part[a] for a in angles], color=CAM_COLOR[cam], lw=2,
                marker="o", ms=8, label=f"{cam} 부품만 회전", zorder=3)
        # 선 오른쪽 끝에 카메라 이름 직접 표시
        ax.text(len(angles) - 1 + 0.12, part[180], cam, va="center", color=INK2, fontsize=11)
        # 이미지 전체 회전 결과 (preprocess.py 방식), ±10° 위치에만 X로 표시
        whole = fpr_by_abs_angle(PRED / f"{cat}_{cat}_variation_v1_threshold_metrics.json", "rotation")
        ax.scatter([angles.index(10)], [whole[10]], s=90, marker="X",
                   color=CAM_COLOR[cam], edgecolor="white", linewidth=1, zorder=4)
    # X 표시 범례는 카메라마다 따로 넣지 않고 회색 하나로 (빈 점을 그려 범례만 등록)
    ax.scatter([], [], s=90, marker="X", color=GRAY, label="이미지 전체 회전 ±10° (조장 전처리)")
    # X 표시 설명 (세 설정 모두 100%라 점이 한 곳에 겹쳐 보임)
    ax.annotate("전체 회전 ±10°: 세 설정 모두 100%\n(회전으로 생긴 모서리 인공물 때문 → 과장)", xy=(3, 100), xytext=(0.1, 80),
                color=INK2, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    # ±10° 이내 구간을 옅은 배경으로 강조 (이 구간은 과검 변화가 작음)
    ax.axvspan(-0.3, 3.3, color="#f3f2ef", zorder=0)
    ax.text(1.5, -9, "±10° 이내", ha="center", color=MUTED, fontsize=10)
    ax.set_xticks(list(x), ["원본"] + [f"±{a}°" if a != 180 else "180°" for a in angles[1:]])
    ax.set_xlim(-0.3, len(angles) - 0.4)
    ax.set_ylim(-14, 108)  # 아래쪽 "±10° 이내" 글자가 들어갈 여유
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("과검률 (%)  ※ 정상을 불량으로 판정")
    ax.set_xlabel("부품 회전 각도 (ROI 데이터, coreset 1%)")
    ax.set_title("부품이 30° 이상 돌면 정상도 불량으로 판정", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", fontsize=10)
    save(fig, "3_rotation_overdetection.png")


# 직접 실행했을 때만 그래프 3개 생성
if __name__ == "__main__":
    chart_coreset()
    chart_seeds()
    chart_rotation()
