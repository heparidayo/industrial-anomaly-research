"""
산업체 데이터 구조 점검 — 이미지 내용은 출력하지 않고 "구조 정보"만 요약합니다.

출력 내용:
  - 클래스 폴더별 이미지 수, 확장자, 이미지 크기 분포
  - label 폴더의 파일 수, 확장자, 형식 추정 (JSON 키 / TXT 열 수 / 마스크 픽셀 값)
  - 이미지와 라벨 파일 이름 짝 맞춤 여부
  - 파일 이름 패턴 (카메라 구분이 이름에 들어 있는지 확인용)

출력에는 좌표값이나 이미지 픽셀이 포함되지 않으므로 공유해도 되는 수준이지만,
팀 기준에 따라 판단해 주세요. 결과는 화면과 inspect_report.txt에 저장됩니다.

사용 예 (PowerShell):
  python inspect_company_data.py --root "D:\\data\\대세 라벨링"
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LINES = []


def out(s=""):
    print(s)
    LINES.append(s)


def name_pattern(stem: str) -> str:
    """숫자를 #으로 바꿔 이름 패턴만 남김 (예: CAM1_0023 → CAM#_#)."""
    return re.sub(r"\d+", "#", stem)


def describe_json(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        data = json.loads(path.read_text(encoding="cp949"))
    if isinstance(data, dict):
        keys = sorted(data.keys())
        hint = ""
        if "shapes" in data:
            types = Counter(s.get("shape_type", "?") for s in data["shapes"])
            labels = Counter(s.get("label", "?") for s in data["shapes"])
            hint = f" → Labelme 형식 추정, 도형 {dict(types)}, 라벨 {dict(labels)}"
        elif {"images", "annotations"} <= set(keys):
            hint = " → COCO 형식 추정"
        return f"JSON 최상위 키 {keys[:12]}{hint}"
    if isinstance(data, list):
        return f"JSON 리스트 (길이 {len(data)})"
    return f"JSON ({type(data).__name__})"


def describe_txt(path: Path) -> str:
    rows = [l.split() for l in path.read_text(errors="ignore").splitlines() if l.strip()]
    if not rows:
        return "빈 TXT"
    cols = Counter(len(r) for r in rows)
    hint = ""
    if all(len(r) == 5 for r in rows):
        hint = " → YOLO 박스 형식 추정 (클래스 x y w h)"
    elif all(len(r) >= 7 and len(r) % 2 == 1 for r in rows):
        hint = " → YOLO 다각형(세그멘테이션) 형식 추정"
    classes = Counter(r[0] for r in rows)
    return f"TXT {len(rows)}줄, 줄당 값 개수 {dict(cols)}, 첫 열 값 {dict(classes)}{hint}"


def describe_mask(path: Path) -> str:
    im = Image.open(path)
    vals = np.unique(np.asarray(im.convert("L"))).tolist()
    return f"이미지형 라벨 {im.size[0]}×{im.size[1]}, 모드 {im.mode}, 픽셀 값 {vals[:10]}{' …' if len(vals) > 10 else ''}"


def describe_label(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".json":
            return describe_json(path)
        if ext == ".txt":
            return describe_txt(path)
        if ext in IMG_EXTS:
            return describe_mask(path)
        if ext == ".xml":
            return "XML → Pascal VOC 등 추정"
        return f"알 수 없는 형식 ({ext})"
    except Exception as e:  # 형식 판별 실패도 정보이므로 계속 진행
        return f"읽기 실패: {type(e).__name__}"


def summarize_images(paths, max_open=30):
    sizes = Counter()
    for p in paths[:max_open]:
        try:
            with Image.open(p) as im:
                sizes[f"{im.size[0]}×{im.size[1]} {im.mode}"] += 1
        except Exception:
            sizes["열기 실패"] += 1
    return dict(sizes)


def main():
    ap = argparse.ArgumentParser(description="산업체 데이터 구조 점검")
    ap.add_argument("--root", required=True)
    ap.add_argument("--report", default="inspect_report.txt")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"폴더가 없습니다: {root.resolve()}")

    out(f"루트: {root.resolve()}")
    out(f"하위 폴더: {[p.name for p in sorted(root.iterdir()) if p.is_dir()]}")
    for cls in sorted(p for p in root.iterdir() if p.is_dir()):
        out("")
        out(f"■ {cls.name}")
        img_dir = cls / "images" if (cls / "images").is_dir() else cls
        lab_dir = next((cls / n for n in ("label", "labels") if (cls / n).is_dir()), None)
        imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS
                      and (lab_dir is None or lab_dir not in p.parents))
        out(f"  이미지 위치: {img_dir.relative_to(root)}  ({len(imgs)}장)")
        out(f"  확장자: {dict(Counter(p.suffix.lower() for p in imgs))}")
        out(f"  크기 (최대 30장 확인): {summarize_images(imgs)}")
        pats = Counter(name_pattern(p.stem) for p in imgs)
        out(f"  이름 패턴 상위: {pats.most_common(5)}")
        subdirs = Counter(p.parent.relative_to(img_dir).as_posix() for p in imgs)
        if len(subdirs) > 1:
            out(f"  이미지 하위 폴더: {dict(subdirs)}")

        if lab_dir is None:
            out("  라벨 폴더: 없음")
            continue
        labs = sorted(p for p in lab_dir.rglob("*") if p.is_file())
        out(f"  라벨 위치: {lab_dir.relative_to(root)}  ({len(labs)}개)")
        out(f"  라벨 확장자: {dict(Counter(p.suffix.lower() for p in labs))}")
        for p in labs[:3]:
            out(f"  라벨 예시 형식: {describe_label(p)}")
        img_stems, lab_stems = {p.stem for p in imgs}, {p.stem for p in labs}
        both = img_stems & lab_stems
        out(f"  이름 짝 맞춤: 일치 {len(both)}, 라벨 없는 이미지 {len(img_stems - lab_stems)}, "
            f"이미지 없는 라벨 {len(lab_stems - img_stems)}")
        if not both and labs and imgs:
            out(f"  (이름이 바로 일치하지 않음: 이미지 패턴 {name_pattern(imgs[0].stem)}, "
                f"라벨 패턴 {name_pattern(labs[0].stem)})")

    Path(args.report).write_text("\n".join(LINES), encoding="utf-8")
    print(f"\n요약 저장: {Path(args.report).resolve()}")


if __name__ == "__main__":
    main()
