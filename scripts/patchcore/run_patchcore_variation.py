"""정상편차 manifest를 공통 PatchCore 배치 경로로 추론한다."""
import argparse
import csv
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.patchcore.model import PatchCorePredictor
from src.models.patchcore.batch import predict_samples, prediction_metadata, write_predictions
from src.models.patchcore.artifacts import image_group_id


def main():
    parser = argparse.ArgumentParser(description="PatchCore 정상편차 manifest 추론")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-root", type=Path, default=ROOT)
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    try:
        manifest = resolve(args.manifest)
        category, experiment = manifest.parent.parent.name, manifest.parent.name
        output = resolve(args.output or Path("outputs/predictions/patchcore") / f"{category}_{experiment}.csv")
        if output.exists() or output.with_suffix(output.suffix + ".meta.json").exists():
            raise FileExistsError("정상편차 예측 결과가 이미 있습니다.")
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"image_id", "source_image", "label", "variation", "level"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("manifest 필수 컬럼이 없습니다.")
            entries = list(reader)
        if not entries or any(entry["label"] != "normal" for entry in entries):
            raise ValueError("정상 이미지 manifest만 사용할 수 있습니다.")
        samples = []
        for entry in entries:
            sample = {
                "path": resolve(args.image_root) / entry["image_id"],
                "image_id": entry["image_id"], "label": 0, "defect_type": "good",
                "variation": entry["variation"], "level": entry["level"],
                "source_image": entry["source_image"],
            }
            match = re.search(r"_(camera_[12])$", Path(entry["source_image"]).stem)
            if match:
                sample.update(camera=match[1], group_id=image_group_id(entry["source_image"]))
            samples.append(sample)
        memory = resolve(args.memory)
        model = PatchCorePredictor(memory)
        if model.category != category:
            raise ValueError("Memory Bank와 manifest의 카테고리가 다릅니다.")
        metadata = prediction_metadata(memory, samples, category=category, split="test_variation", device=model.device)
        records = predict_samples(model, samples)
        write_predictions(records, output, metadata)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print("예측 CSV:", output)


if __name__ == "__main__":
    main()
