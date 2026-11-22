"""원본·정상편차 입력이 공유하는 PatchCore 추론과 CSV 저장."""
import csv
import json
from pathlib import Path
import time
import torch

from .artifacts import file_sha256, input_fingerprint

COLUMNS = ["image_id", "label", "defect_type", "variation", "level", "anomaly_score", "inference_ms"]
EXTRA_COLUMNS = ["source_image", "camera", "group_id"]


def predict_samples(model, samples, *, progress=None):
    if not samples:
        raise ValueError("추론할 이미지가 없습니다.")
    ids = [sample["image_id"] for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("중복 image_id가 있습니다.")
    for sample in samples:
        if not Path(sample["path"]).is_file() or sample["label"] not in (0, 1):
            raise ValueError("입력 이미지 또는 라벨을 확인하세요.")
    model.predict(samples[0]["path"])
    records = []
    for index, sample in enumerate(samples):
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.predict(sample["path"])
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        record = {key: sample[key] for key in COLUMNS[:5]}
        record.update(anomaly_score=result["image_score"],
                      inference_ms=round((time.perf_counter() - start) * 1000, 3))
        record.update({key: sample[key] for key in EXTRA_COLUMNS if key in sample})
        records.append(record)
        if progress:
            progress(index + 1, len(samples))
    return records


def prediction_metadata(memory_path, samples, *, category, split, device=None):
    saved = torch.load(memory_path, map_location="cpu", weights_only=True)
    return {
        "model": "patchcore_prototype", "category": category, "split": split,
        "memory_sha256": file_sha256(memory_path),
        "input_fingerprint": input_fingerprint(samples),
        "coreset": {key: saved.get(key) for key in ("percentage", "projection_dim", "seed")},
        "raw_memory_sha256": saved.get("raw_memory_sha256"),
        "device": str(device) if device is not None else None,
        "gpu_name": torch.cuda.get_device_name(device) if device is not None and device.type == "cuda" else None,
        "torch_version": str(torch.__version__), "cuda_build": torch.version.cuda,
        "timing": "image_read_transform_features_score; initialization,warmup,csv excluded",
    }


def write_predictions(records, path, metadata):
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if path.exists() or sidecar.exists():
        raise FileExistsError("예측 CSV 또는 메타데이터가 이미 있습니다.")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = COLUMNS + [key for key in EXTRA_COLUMNS if any(key in row for row in records)]
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def make_samples(data_root, category, split="test", *, project_root):
    category_dir = Path(data_root) / category
    paths = sorted((category_dir / split).glob("*/*.png"))
    metadata = {}
    manifest = category_dir / "split.csv"
    if manifest.is_file():
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            metadata = {row["file"]: row for row in csv.DictReader(handle)}
    samples = []
    for path in paths:
        try:
            image_id = path.relative_to(project_root).as_posix()
        except ValueError:
            image_id = path.as_posix()
        sample = {
            "path": path, "image_id": image_id,
            "label": int(path.parent.name != "good"), "defect_type": path.parent.name,
            "variation": "original", "level": 0,
        }
        if metadata:
            row = metadata[path.name]
            if row["split"] != split:
                raise ValueError("준비된 이미지와 split.csv가 일치하지 않습니다.")
            sample.update(camera=row["camera"], group_id=row["group_id"])
        samples.append(sample)
    return samples
