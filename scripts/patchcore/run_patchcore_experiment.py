"""PatchCore 실행을 묶고 공통 평가기를 호출한다. 모델 재구현은 포함하지 않는다."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.patchcore.batch import make_samples
from src.evaluation.metrics import image_auroc_by_defect
from src.models.patchcore.artifacts import file_sha256, image_group_id, input_fingerprint, train_samples
from src.models.patchcore.threshold import effective_folds


def pct_name(value):
    return f"{value * 100:g}pct".replace(".", "p")


def artifact_paths(category, percentage, projection_dim=128, seed=0, folds=10, split="test"):
    bank_dir = ROOT / "outputs/patchcore" / category
    tag = f"{pct_name(percentage)}_d{projection_dim}_s{seed}"
    suffix = "original" if split == "test" else split
    # 기존 기본 bottle 출력명은 유지하고, 다른 설정은 이름으로 구분한다.
    default = percentage == 0.01 and projection_dim == 128 and seed == 0
    prediction = ROOT / "outputs/predictions/patchcore" / (
        f"{category}_{suffix}{'' if default else '_' + tag}.csv")
    return {
        "raw": bank_dir / "raw_memory_full.pt",
        "coreset": bank_dir / f"coreset_raw_memory_full_{tag}.pt",
        "prediction": prediction,
        "metrics": prediction.with_name(prediction.stem + "_metrics.json"),
        "threshold": bank_dir / f"threshold_{tag}_k{folds}.json",
        "threshold_metrics": prediction.with_name(prediction.stem + f"_threshold_k{folds}_metrics.json"),
    }


def check_metadata(path, expected, *, sidecar=True):
    source = path.with_suffix(path.suffix + ".meta.json") if sidecar else path
    if not source.is_file():
        raise ValueError(f"출처 메타데이터가 없습니다. 기존 결과를 확인하고 새 실험으로 분리하세요: {path}")
    actual = json.loads(source.read_text(encoding="utf-8"))
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError(f"결과와 현재 입력·설정이 다릅니다: {path}")


def run(script, *args):
    command = [sys.executable, str(ROOT / script), *map(str, args)]
    print("실행:", script, flush=True)
    result = subprocess.run(command, cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8",
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if result.returncode:
        raise RuntimeError(f"하위 실행 실패: {script}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def request_manifest(args, data_root):
    if args.projection_dim < 1:
        raise ValueError("투영 차원은 양수여야 합니다.")
    if not args.categories or len(set(args.categories)) != len(args.categories):
        raise ValueError("중복되지 않는 --categories 목록이 필요합니다.")
    if len(set(args.pcts)) != len(args.pcts) or not all(0 < value <= 1 for value in args.pcts):
        raise ValueError("Coreset 비율은 중복되지 않는 (0, 1] 값이어야 합니다.")
    inputs = {}
    for category in args.categories:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", category):
            raise ValueError("카테고리 이름에 경로를 넣을 수 없습니다.")
        paths = sorted((data_root / category / "train/good").glob("*.png"))
        samples = make_samples(data_root, category, args.split, project_root=ROOT)
        if not paths or not samples:
            raise ValueError(f"{category}: 학습·{args.split} 입력이 없습니다.")
        groups = [image_group_id(path.name) for path in paths]
        folds = effective_folds(groups, args.folds)
        inputs[category] = {
            "train_fingerprint": input_fingerprint(train_samples(paths)),
            "eval_fingerprint": input_fingerprint(samples), "folds": folds,
            "split_manifest_sha256": file_sha256(data_root / category / "split.csv")
                if (data_root / category / "split.csv").is_file() else None,
            "dataset_info_sha256": file_sha256(data_root / category / "dataset_info.json")
                if (data_root / category / "dataset_info.json").is_file() else None,
        }
    code_files = list((ROOT / "src/models/patchcore").glob("*.py"))
    code_files += list((ROOT / "src/evaluation").glob("*.py"))
    code_files += list((ROOT / "scripts/patchcore").glob("*.py"))
    code_files += list((ROOT / "scripts/evaluation").glob("*.py"))
    code_hash = hashlib.sha256("".join(file_sha256(p) for p in sorted(code_files)).encode()).hexdigest()
    return {
        "data_root": str(data_root), "split": args.split, "categories": args.categories,
        "percentages": args.pcts, "projection_dim": args.projection_dim, "seed": args.seed,
        "requested_folds": args.folds, "inputs": inputs, "code_sha256": code_hash,
        "python": sys.version, "torch": importlib.metadata.version("torch"),
    }


def run_one(category, percentage, args, data_root, inputs):
    paths = artifact_paths(category, percentage, args.projection_dim,
                           args.seed, inputs["folds"], args.split)
    if not paths["raw"].exists():
        run("scripts/patchcore/build_patchcore_memory.py",
            "--category", category, "--data-root", data_root)
    check_metadata(paths["raw"], {"category": category,
                                  "data_root": str(data_root),
                                  "input_fingerprint": inputs["train_fingerprint"]})
    config = {"category": category, "percentage": percentage,
              "projection_dim": args.projection_dim, "seed": args.seed,
              "raw_memory_sha256": file_sha256(paths["raw"])}
    if not paths["coreset"].exists():
        run("scripts/patchcore/build_patchcore_coreset.py", "--input", paths["raw"],
            "--percentage", percentage, "--projection-dim", args.projection_dim, "--seed", args.seed)
    check_metadata(paths["coreset"], config)
    if not paths["prediction"].exists():
        run("scripts/patchcore/run_patchcore_batch.py", "--category", category,
            "--data-root", data_root, "--split", args.split, "--memory", paths["coreset"],
            "--output", paths["prediction"])
    check_metadata(paths["prediction"], {
        "category": category, "split": args.split,
        "input_fingerprint": inputs["eval_fingerprint"],
        "memory_sha256": file_sha256(paths["coreset"]),
    })
    if not paths["metrics"].exists():
        run("scripts/evaluation/evaluate_predictions.py", "--csv", paths["prediction"])
    if not paths["threshold"].exists():
        run("scripts/patchcore/compute_patchcore_threshold.py", "--category", category,
            "--percentage", percentage, "--projection-dim", args.projection_dim, "--seed", args.seed,
            "--folds", args.folds, "--output", paths["threshold"])
    check_metadata(paths["threshold"], {**config, "folds": inputs["folds"]}, sidecar=False)
    if not paths["threshold_metrics"].exists():
        run("scripts/evaluation/evaluate_threshold.py", "--csv", paths["prediction"],
            "--threshold", paths["threshold"], "--category", category,
            "--output", paths["threshold_metrics"])
    threshold_info = json.loads(paths["threshold"].read_text(encoding="utf-8"))
    check_metadata(paths["threshold_metrics"], {
        "csv": str(paths["prediction"]), "threshold_file": str(paths["threshold"]),
        "threshold": threshold_info["threshold"],
    }, sidecar=False)
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    fixed = json.loads(paths["threshold_metrics"].read_text(encoding="utf-8"))
    group = fixed["groups"][0]
    with paths["prediction"].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    raw_metadata = json.loads(paths["raw"].with_suffix(".pt.meta.json").read_text(encoding="utf-8"))
    dataset_info_path = data_root / category / "dataset_info.json"
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8")) if dataset_info_path.is_file() else {}
    return {
        "category": category, "split": args.split, "pct": percentage,
        "train_normal_count": len(raw_metadata["image_names"]),
        "model_seed": args.seed, "split_seed": dataset_info.get("seed"),
        "protocol_status": dataset_info.get("protocol_status", "dataset_predefined"),
        "auroc": metrics["image_auroc"], "mean_ms": metrics["mean_inference_ms"],
        "threshold": fixed["threshold"], "fpr": group["fpr"], "fnr": group["fnr"],
        "false_alarm": f"{group['false_alarm']}/{group['normal_count']}",
        "missed": f"{group['missed']}/{group['defect_count']}", "f1": group["f1"],
        **{f"auroc_{name}": value for name, value in image_auroc_by_defect(rows).items()},
    }


def summary_markdown(name, rows):
    lines = [f"# PatchCore 실험: {name}", "",
             "고정 임계값 평가입니다. 테스트 정답을 사용한 최적 임계값은 계산하지 않습니다.", ""]
    for row in rows:
        lines.append(f"- {row['category']} ({row.get('split', 'unknown')}), "
                     f"Coreset {float(row['pct']) * 100:g}%, AUROC {float(row['auroc']):.4f}, "
                     f"FPR {float(row['fpr']):.4f}, FNR {float(row['fnr']):.4f}, "
                     f"평균 {float(row['mean_ms']):.2f} ms")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="PatchCore 실행·공통 평가 연결·요약")
    parser.add_argument("--name", required=True)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--pcts", nargs="+", type=float, default=[0.01])
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("name은 영문·숫자·밑줄·하이픈만 사용하세요.")
    output = ROOT / "outputs/patchcore/summary" / f"{args.name}.csv"
    try:
        if args.render_only:
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        else:
            data_root = (args.data_root if args.data_root.is_absolute() else ROOT / args.data_root).resolve()
            request = request_manifest(args, data_root)
            if args.plan_only:
                print(json.dumps(request, ensure_ascii=False, indent=2))
                return
            manifest = output.with_suffix(".run.json")
            if manifest.exists():
                if not args.resume or json.loads(manifest.read_text(encoding="utf-8")) != request:
                    raise ValueError("같은 이름의 실험이 있습니다. 동일 조건은 --resume, 다른 조건은 새 --name을 사용하세요.")
            elif args.resume:
                raise ValueError("재개할 실험 기록이 없습니다.")
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with manifest.open("x", encoding="utf-8") as handle:
                    json.dump(request, handle, ensure_ascii=False, indent=2)
            rows = [run_one(category, pct, args, data_root, request["inputs"][category])
                    for category in args.categories for pct in args.pcts]
            keys = list(dict.fromkeys(key for row in rows for key in row))
            with output.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        markdown = summary_markdown(args.name, rows)
        output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    print(markdown)
    print("요약 CSV:", output)


if __name__ == "__main__":
    main()
