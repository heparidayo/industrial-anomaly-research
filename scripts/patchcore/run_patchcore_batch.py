"""MVTec 모양의 원본 입력을 추론한다. 기존 bottle 명령과 CSV 컬럼을 유지한다."""
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.patchcore.model import PatchCorePredictor
from src.models.patchcore.batch import make_samples, predict_samples, prediction_metadata, write_predictions


def main():
    parser = argparse.ArgumentParser(description="PatchCore 원본 이미지 배치 추론")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    output = resolve(args.output or Path("outputs/predictions/patchcore") /
                     f"{args.category}_{'original' if args.split == 'test' else 'validation'}.csv")
    if output.exists() or output.with_suffix(output.suffix + ".meta.json").exists():
        parser.error("예측 결과가 이미 존재합니다.")
    try:
        samples = make_samples(resolve(args.data_root), args.category, args.split, project_root=ROOT)
        if not samples:
            raise ValueError("추론할 PNG 이미지가 없습니다.")
        memory = resolve(args.memory)
        model = PatchCorePredictor(memory)
        if model.category != args.category:
            raise ValueError("Memory Bank 카테고리가 일치하지 않습니다.")
        metadata = prediction_metadata(memory, samples, category=args.category, split=args.split, device=model.device)
        records = predict_samples(model, samples,
                                  progress=lambda i, n: print(f"[{i}/{n}]") if i % 20 == 0 or i == n else None)
        write_predictions(records, output, metadata)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print("사용 장치:", model.device)
    print("예측 CSV:", output)


if __name__ == "__main__":
    main()
