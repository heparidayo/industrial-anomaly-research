"""
EfficientAD 추론 — 학습된 체크포인트로 이미지 폴더를 검사하고 예측 CSV를 저장.

입력 방식 두 가지:
  --input <폴더>                하위 폴더까지 모든 이미지 (라벨 없음)
  --data_root <루트> --category <이름>   test/ 폴더 (폴더 이름으로 라벨: good=0, 그 외=1)

CSV 열: image_path, label, score, score_local, score_global
  score는 결합 맵 최댓값(논문의 이미지 점수), 나머지는 분석용.
  팀 공통 예측 CSV 형식이 정해지면 write_csv()만 맞춰 고치면 됩니다.

선택 기능:
  --save_maps <폴더>  결합 맵(.npy, 원본 크기)과 히트맵 겹침 이미지(.png) 저장
  --benchmark 1000    배치 1 지연 시간 측정 (CPU→GPU 전송 ~ 결과가 CPU에 도착까지)
  --fp16              float16 추론 (논문 속도 측정 설정)

사용 예:
  python predict.py --checkpoint runs/widget/efficientad_s.pth ^
      --data_root ..\\..\\data\\raw\\dummy --category widget --out_csv pred.csv --save_maps maps
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from datasets import IMG_EXTS, IMAGENET_MEAN, IMAGENET_STD
from efficientad import EfficientAD
from metrics import auroc
from preprocess import Preprocess


def load_image(path, pre: Preprocess):
    """학습 때와 같은 전처리(자르기·밝기 정규화)를 적용.
    반환: 모델 입력 텐서, 자른 영역 크기 (h, w), 히트맵을 겹칠 이미지(자른 영역, 원본 해상도)."""
    orig = Image.open(path).convert("RGB")
    img, box = pre(orig)
    view = orig.crop(box) if box != (0, 0, orig.size[0], orig.size[1]) else orig
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1)), (box[3] - box[1], box[2] - box[0]), view


def collect_items(args):
    """(경로, 라벨 또는 None) 목록."""
    if args.input:
        root = Path(args.input)
        return [(p, None) for p in sorted(root.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    test_dir = Path(args.data_root) / args.category / args.split
    if not test_dir.is_dir():
        raise SystemExit(f"평가 폴더가 없습니다: {test_dir}")
    items = []
    for d in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        label = 0 if d.name == "good" else 1
        items += [(p, label) for p in sorted(d.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    return items


def load_split_info(cat_dir: Path):
    """팀 공통 split.csv에서 prepared_image → (camera, group_id, class) 대응표. 없으면 빈 dict."""
    path = cat_dir / "split.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["prepared_image"]: r for r in csv.DictReader(f) if r.get("prepared_image")}


def write_template_csv(path, rows):
    """팀 공통 형식 (predictions_template.csv)."""
    cols = ["image_id", "label", "defect_type", "anomaly_score", "inference_ms", "camera", "group_id"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def save_heatmap(out_dir: Path, rel_name: str, comb_full: np.ndarray, pil_img):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{rel_name}.npy", comb_full.astype(np.float16))
    # 정규화된 점수는 0~1 컬러 스케일에 맞게 설계됨 (논문 Section 3.4)
    heat = (np.clip(comb_full, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    base = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / f"{rel_name}.png"), cv2.addWeighted(base, 0.5, heat, 0.5, 0))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label", "score", "score_local", "score_global"])
        w.writerows(rows)


@torch.no_grad()
def benchmark(model, device, n, dtype):
    x_cpu = torch.randn(1, 3, model.image_size, model.image_size)
    def step():
        x = x_cpu.to(device, dtype=dtype)
        comb, _, _ = model(x)
        return comb.float().cpu()
    for _ in range(n):                               # 워밍업
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser(description="EfficientAD 추론 (Algorithm 2)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", help="이미지 폴더 (라벨 없음)")
    ap.add_argument("--data_root")
    ap.add_argument("--category")
    ap.add_argument("--split", default="test", choices=["test", "validation"],
                    help="--data_root 사용 시 평가할 폴더")
    ap.add_argument("--out_csv", default="predictions.csv")
    ap.add_argument("--template_csv", help="팀 공통 형식(predictions_template.csv)으로도 저장할 경로")
    ap.add_argument("--save_maps", help="히트맵 저장 폴더")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--benchmark", type=int, default=0, help="지연 시간 측정 반복 수 (0이면 생략)")
    args = ap.parse_args()
    if not args.input and not (args.data_root and args.category) and not args.benchmark:
        ap.error("--input 또는 --data_root와 --category를 지정하세요")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    model = EfficientAD.load(args.checkpoint, map_location=device).to(device, dtype=dtype)
    pre = Preprocess.from_dict(getattr(model, "meta", {}).get("preprocess"))
    print(f"모델: PDN-{model.size.upper()}, padding={model.padding}, 장치 {device}, {dtype}, {pre}")

    if args.benchmark:
        ms = benchmark(model, device, args.benchmark, dtype)
        print(f"지연 시간 (배치 1, {args.benchmark}회 평균): {ms:.2f} ms")
        if not (args.input or args.data_root):
            return

    items = collect_items(args)
    cat_dir = Path(args.data_root) / args.category if args.data_root else None
    info = load_split_info(cat_dir) if cat_dir else {}
    print(f"이미지 {len(items)}장" + (f" ({args.split})" if cat_dir else "")
          + f", 입력: 원본 → {pre} → {model.image_size}×{model.image_size} bilinear")
    rows, trows, scores, labels, types = [], [], [], [], []
    maps_dir = Path(args.save_maps) if args.save_maps else None
    if items:                                        # 첫 이미지 시간이 GPU 준비 시간으로 부풀지 않게 워밍업
        x0, _, _ = load_image(items[0][0], pre)
        for _ in range(10):
            model(x0[None].to(device, dtype=dtype))
    for i, (path, label) in enumerate(items):
        x, (h, w), pil = load_image(path, pre)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()                     # 추론 시간: GPU 전송 ~ 점수가 CPU에 도착 (이미지 읽기·전처리 제외)
        comb, n_st, n_ae = model(x[None].to(device, dtype=dtype))
        score = model.image_score(comb).item()
        ms = (time.perf_counter() - t0) * 1000
        if cat_dir:
            rel = path.relative_to(cat_dir).as_posix()
            r = info.get(rel, {})
            dtype_name = r.get("class") or path.parent.name
            trows.append({"image_id": path.name, "label": "" if label is None else label,
                          "defect_type": dtype_name, "anomaly_score": f"{score:.6f}",
                          "inference_ms": f"{ms:.2f}", "camera": r.get("camera", ""),
                          "group_id": r.get("group_id", "")})
            if label is not None:
                scores.append(score); labels.append(label); types.append(dtype_name)
        rows.append([str(path), "" if label is None else label, f"{score:.6f}",
                     f"{model.image_score(n_st).item():.6f}", f"{model.image_score(n_ae).item():.6f}"])
        if maps_dir:
            full = F.interpolate(comb.float(), size=(h, w), mode="bilinear", align_corners=False)
            rel = f"{i:05d}_{path.parent.name}_{path.stem}"
            save_heatmap(maps_dir, rel, full[0, 0].cpu().numpy(), pil)
    write_csv(args.out_csv, rows)
    print(f"예측 CSV 저장: {args.out_csv}")
    if args.template_csv:
        if not trows:
            print("  [주의] --template_csv는 --data_root·--category로 실행할 때만 저장됩니다")
        else:
            write_template_csv(args.template_csv, trows)
            missing = sum(1 for r in trows if not r["group_id"])
            print(f"팀 공통 형식 저장: {args.template_csv}"
                  + (f"  [주의] split.csv에서 찾지 못한 이미지 {missing}장" if missing else ""))
    if labels and 0 < sum(labels) < len(labels):
        print(f"AU-ROC: {auroc(labels, scores) * 100:.2f}%  (정상 {labels.count(0)}장, 불량 {sum(labels)}장)")
        good = [s for s, t in zip(scores, types) if t == "good"]
        for t in sorted(set(types) - {"good"}):
            bad = [s for s, tt in zip(scores, types) if tt == t]
            print(f"  {t}: {auroc([0] * len(good) + [1] * len(bad), good + bad) * 100:.2f}% ({len(bad)}장)")
        print(f"평균 추론 시간: {np.mean([float(r['inference_ms']) for r in trows]):.2f} ms/장")
    if maps_dir:
        print(f"히트맵 저장: {maps_dir}")


if __name__ == "__main__":
    main()
