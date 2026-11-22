"""제공한 실험 결과만 시각화한다. 특정 성능 수치나 원인 해석을 고정하지 않는다."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def load_summary(paths):
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                for key in ("pct", "auroc", "mean_ms"):
                    row[key] = float(row[key])
                    if not math.isfinite(row[key]):
                        raise ValueError(f"유효하지 않은 그래프 수치: {path}")
                rows.append(row)
    if not rows:
        raise ValueError("요약 CSV에 결과가 없습니다.")
    return rows


def plot_summary(paths, output):
    rows = load_summary(paths)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["category"], row.get("split", "unknown"))].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for (category, split), points in sorted(groups.items()):
        points.sort(key=lambda row: row["pct"])
        label = f"{category} ({split})"
        axes[0].plot([r["pct"] * 100 for r in points],
                     [r["auroc"] for r in points], "o-", label=label)
        axes[1].plot([r["pct"] * 100 for r in points],
                     [r["mean_ms"] for r in points], "o-", label=label)
    axes[0].set(ylabel="Image AUROC", title="Detection scores")
    axes[1].set(ylabel="Mean latency (ms/image)", title="Recorded inference latency")
    for axis in axes:
        axis.set_xlabel("Coreset (%)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    target = output / "summary.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def plot_variations(path, output):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    original = next((g for g in result["groups"] if g["variation"] == "original"), None)
    groups = defaultdict(list)
    for group in result["groups"]:
        if group["variation"] != "original":
            groups[group["variation"]].append(group)
    outputs = []
    for variation, points in sorted(groups.items()):
        points.sort(key=lambda point: point["level"])
        if any(p["fpr"] is None for p in points):
            raise ValueError("정상 이미지가 없는 그룹의 과검률은 그릴 수 없습니다.")
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot([p["level"] for p in points], [p["fpr"] * 100 for p in points], "o-")
        if original and original["fpr"] is not None:
            axis.axhline(original["fpr"] * 100, color="gray", linestyle="--", label="Original FPR")
            axis.legend()
        axis.set(xlabel="Variation level", ylabel="False positive rate (%)",
                 title=f"{Path(path).stem}: {variation}", ylim=(-2, 102))
        axis.grid(alpha=0.25)
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", variation)
        target = output / f"{Path(path).stem}_{safe}.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        outputs.append(target)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="실제 PatchCore 결과 CSV·JSON 그래프")
    parser.add_argument("--summary", nargs="+", type=Path)
    parser.add_argument("--threshold-metrics", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/patchcore/presentation_figures/charts"))
    args = parser.parse_args()
    if not args.summary and not args.threshold_metrics:
        parser.error("--summary 또는 --threshold-metrics 입력이 필요합니다.")
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    try:
        output = resolve(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if args.summary:
            print(plot_summary([resolve(path) for path in args.summary], output))
        for path in args.threshold_metrics or []:
            for result in plot_variations(resolve(path), output):
                print(result)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
