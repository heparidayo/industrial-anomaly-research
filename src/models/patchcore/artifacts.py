"""실험 입력과 산출물의 출처 확인."""
import hashlib
import json
from pathlib import Path
import re


def image_group_id(name):
    return re.sub(r"_camera_[12]$", "", Path(name).stem)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(samples):
    """입력 목록·라벨·편차·파일 변경을 기록한다. 결과 CSV의 캐시 확인에 사용."""
    rows = []
    for sample in samples:
        path = Path(sample["path"])
        stat = path.stat()
        row = {key: value for key, value in sample.items() if key != "path"}
        row.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        rows.append(row)
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def train_samples(paths):
    return [{"path": path, "image_id": path.name} for path in paths]
