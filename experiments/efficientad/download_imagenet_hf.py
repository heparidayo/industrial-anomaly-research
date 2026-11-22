"""
Hugging Face ImageNet-1k(ILSVRC/imagenet-1k) → JPEG 파일로 받기.

Hugging Face에는 이미지가 parquet 파일(표 형식) 안에 묶여 있어서, 우리 코드가
바로 읽을 수 없습니다. 이 스크립트는 parquet 조각을 하나씩 받아 그 안의 이미지를
JPEG 파일로 꺼낸 뒤, 받은 parquet 조각은 지워서 디스크를 아낍니다.

출력:
  <out>/val/*.JPEG                    검증 세트 (조각 14개, 약 5만 장)
  <out>/train/<클래스번호>/*.JPEG      학습 세트 일부 (--train_shards 개수만큼)

사전 준비:
  1) huggingface.co에서 ILSVRC/imagenet-1k 이용 조건 동의
  2) 로그인:  python -c "from huggingface_hub import login; login()"
  3) parquet 읽기 도구 설치:  python -m pip install pyarrow

사용 예:
  python download_imagenet_hf.py --out ..\\..\\data\\raw\\imagenet --val
  python download_imagenet_hf.py --out ..\\..\\data\\raw\\imagenet --train_shards 20

중간에 멈춰도 같은 명령을 다시 실행하면 끝난 조각은 건너뜁니다.
"""

import argparse
from collections import Counter
from pathlib import Path

REPO = "ILSVRC/imagenet-1k"


def list_shards(split_prefix: str):
    from huggingface_hub import list_repo_files
    files = list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if f.startswith(f"data/{split_prefix}-") and f.endswith(".parquet"))


def pick_evenly(items, k):
    """목록 전체에서 고르게 k개 선택 (앞쪽 조각에만 쏠리지 않게)."""
    if k >= len(items):
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def extract_parquet(parquet_path: Path, out_dir: Path, by_class: bool, prefix: str) -> Counter:
    """parquet 안의 이미지를 JPEG로 저장. 반환: 클래스별 장수."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    names = set(pf.schema_arrow.names)
    if "image" not in names:
        raise SystemExit(f"'image' 열이 없습니다. 열 목록: {sorted(names)}")
    cols = ["image"] + (["label"] if "label" in names else [])
    counts, idx = Counter(), 0
    for batch in pf.iter_batches(columns=cols, batch_size=256):
        img_col = batch.column("image")
        datas = img_col.field("bytes").to_pylist()
        paths = img_col.field("path").to_pylist() if "path" in [f.name for f in img_col.type] else [None] * len(datas)
        labels = batch.column("label").to_pylist() if "label" in cols else [None] * len(datas)
        for data, p, lab in zip(datas, paths, labels):
            idx += 1
            if not data:
                continue
            name = Path(p).name if p else f"{prefix}_{idx:06d}.JPEG"
            if not name.lower().endswith((".jpeg", ".jpg", ".png")):
                name += ".JPEG"
            sub = f"{lab:04d}" if (by_class and lab is not None and lab >= 0) else ""
            dst = out_dir / sub / name if sub else out_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.write_bytes(data)
            counts[sub or "-"] += 1
    return counts


def process(shards, out_dir: Path, cache: Path, by_class: bool, keep: bool, local: bool):
    total = Counter()
    for i, shard in enumerate(shards, 1):
        done = out_dir / f".done_{Path(shard).stem}"
        if done.exists():
            print(f"  [{i}/{len(shards)}] {shard} 이미 완료 → 건너뜀")
            continue
        if local:
            path = Path(shard)
        else:
            from huggingface_hub import hf_hub_download
            print(f"  [{i}/{len(shards)}] 다운로드: {shard}")
            path = Path(hf_hub_download(REPO, shard, repo_type="dataset", local_dir=str(cache)))
        counts = extract_parquet(path, out_dir, by_class, Path(shard).stem)
        total.update(counts)
        out_dir.mkdir(parents=True, exist_ok=True)
        done.write_text(str(sum(counts.values())))
        print(f"      → {sum(counts.values())}장 추출")
        if not keep and not local:
            path.unlink(missing_ok=True)
    return total


def main():
    ap = argparse.ArgumentParser(description="Hugging Face ImageNet-1k parquet → JPEG")
    ap.add_argument("--out", required=True, help="예: ..\\..\\data\\raw\\imagenet")
    ap.add_argument("--val", action="store_true", help="검증 세트 전체 (조각 14개)")
    ap.add_argument("--train_shards", type=int, default=0, help="학습 세트 조각 수 (전체에서 고르게 선택)")
    ap.add_argument("--keep_parquet", action="store_true", help="추출 후 parquet 조각을 지우지 않음")
    ap.add_argument("--local_parquet", help="(테스트용) 내려받지 않고 이 폴더의 parquet를 사용")
    args = ap.parse_args()
    if not args.val and args.train_shards <= 0:
        ap.error("--val 또는 --train_shards를 지정하세요")

    out = Path(args.out)
    cache = out / "_hf_cache"
    local = args.local_parquet is not None

    def shards_for(prefix):
        if local:
            return sorted(str(p) for p in Path(args.local_parquet).glob(f"{prefix}-*.parquet"))
        return list_shards(prefix)

    if args.val:
        shards = shards_for("validation")
        print(f"검증 세트: 조각 {len(shards)}개")
        c = process(shards, out / "val", cache, by_class=False, keep=args.keep_parquet, local=local)
        print(f"검증 세트 완료: 이번 실행에서 {sum(c.values())}장 추출 → {(out / 'val').resolve()}")

    if args.train_shards > 0:
        all_train = shards_for("train")
        shards = pick_evenly(all_train, args.train_shards)
        print(f"학습 세트: 전체 조각 {len(all_train)}개 중 {len(shards)}개 선택")
        c = process(shards, out / "train", cache, by_class=True, keep=args.keep_parquet, local=local)
        print(f"학습 세트 완료: 이번 실행에서 {sum(c.values())}장, 클래스 {len(c)}개 → {(out / 'train').resolve()}")

    if cache.exists() and not args.keep_parquet:
        try:
            for p in sorted(cache.rglob("*"), reverse=True):
                p.unlink() if p.is_file() else p.rmdir()
            cache.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
