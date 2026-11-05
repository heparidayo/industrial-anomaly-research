


from pathlib import Path
import argparse
import csv
import json
import sys

import torch
from torchvision.models import Wide_ResNet50_2_Weights


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))

from src.models.patchcore.backbone import PatchCoreBackbone
from src.models.patchcore.features import PatchFeatureExtractor
from src.models.patchcore.memory_bank import collect_normal_features
from src.models.patchcore.artifacts import image_group_id, input_fingerprint, train_samples


def main():

    parser = argparse.ArgumentParser(
        description="PatchCore 정상 특징 후보 생성"
    )


    parser.add_argument("--category", default="bottle")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec_ad"))

    args = parser.parse_args()

    # 1. 정상 학습 이미지

    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    train_dir = data_root / args.category / "train/good"


    image_paths = sorted(train_dir.glob("*.png"))

    if not image_paths:
        raise SystemExit("정상 학습 이미지를 찾을 수 없습니다.")


    if args.limit is not None:
        if not 1 <= args.limit <= len(image_paths):
            parser.error("--limit 값이 올바르지 않습니다.")

        image_paths = image_paths[:args.limit]

    # 2. GPU 및 모델 준비

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("사용 장치:", device)
    print("학습 정상 이미지:", len(image_paths))


    weights = Wide_ResNet50_2_Weights.DEFAULT
    transform = weights.transforms()


    backbone = PatchCoreBackbone(device)


    patch_extractor = PatchFeatureExtractor(
        patch_size=3,
        pretrain_embed_dim=1024,
        target_embed_dim=1024,
    ).to(device)


    patch_extractor.eval()

    # 3. 정상 특징 수집


    memory, patches_per_image = collect_normal_features(
        image_paths=image_paths,
        backbone=backbone,
        patch_extractor=patch_extractor,
        transform=transform,
        device=device,
    )

    # 4. 결과 저장

    output_dir = ROOT / "outputs/patchcore" / args.category
    output_dir.mkdir(parents=True, exist_ok=True)


    suffix = (
        "full" if args.limit is None
        else f"debug_{args.limit}"
    )

    output_path = output_dir / f"raw_memory_{suffix}.pt"
    sidecar = output_path.with_suffix(output_path.suffix + ".meta.json")

    # 기존 결과를 실수로 덮어쓰지 않도록 보호

    if output_path.exists() or sidecar.exists():
        raise SystemExit(
            f"이미 결과 파일이 존재합니다: {output_path}"
        )

    group_ids = {p.name: image_group_id(p.name) for p in image_paths}
    split_manifest = data_root / args.category / "split.csv"
    if split_manifest.is_file():
        with split_manifest.open(encoding="utf-8-sig", newline="") as handle:
            rows = {row["file"]: row for row in csv.DictReader(handle)}
        if any(rows[p.name]["split"] != "train" or rows[p.name]["class"] != "good" for p in image_paths):
            parser.error("학습 입력과 split.csv의 정상 학습 목록이 일치하지 않습니다.")
        group_ids = {p.name: rows[p.name]["group_id"] for p in image_paths}
    metadata = {
        "category": args.category,
        "image_names": [p.name for p in image_paths],
        "image_group_ids": [group_ids[p.name] for p in image_paths],
        "input_fingerprint": input_fingerprint(train_samples(image_paths)),
        "data_root": str(data_root.resolve()),
    }


    torch.save({
        "embeddings": memory,
        "category": args.category,
        "image_count": len(image_paths),
        "image_names": [p.name for p in image_paths],
        "patches_per_image": patches_per_image,
        "backbone": "wide_resnet50_2",
        "weights": weights.name,
        "transform": repr(transform),
        "patch_size": 3,
        "embedding_dim": memory.shape[1],
        "stage": "raw_candidates",
        **metadata,
    }, output_path)
    sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 정상 특징 수집 완료 ===")
    print("Memory Shape:", tuple(memory.shape))

    print(
        "크기:",
        f"{memory.numel() * memory.element_size() / 1024**2:.1f} MiB"
    )
    print("저장 위치:", output_path)


if __name__ == "__main__":
    main()
