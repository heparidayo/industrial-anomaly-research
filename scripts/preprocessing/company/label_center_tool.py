"""부품 바깥 원(class 1) 정답 라벨 클릭 도구.

목적: 자동 중심 검출을 채점할 정답지를 만든다 (보정에는 쓰지 않음, PATCHCORE_PROGRESS.md I-4 참고).
대상: 원본 정상 중 09-29가 아닌 날짜에서 (날짜, 카메라)마다 n장 (고정 seed)
저장: data/labels_extra/company_center/<파일이름>.txt  (기존 라벨과 같은 형식: "1 x1 y1 ... x96 y96", 0~1 좌표)

조작
  왼쪽 클릭 : 부품 끝면의 가장 바깥 테두리 위에 점 찍기 (4~6개, 둘레에 고르게)
  오른쪽 클릭 / z : 마지막 점 지우기
  Enter / Space : 저장하고 다음 (점 3개 이상 필요)
  s : 이 이미지 건너뛰기      q / Esc : 끝내기 (다시 실행하면 이어서 함)

실행: .\\.venv\\Scripts\\python.exe scripts/preprocessing/company/label_center_tool.py
"""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import random
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]                        # 레포 최상위 폴더
sys.path.insert(0, str(ROOT))
from src.preprocessing.company.alignment import _fit_circle        # 점들에 맞는 원 (최소제곱)
from src.preprocessing.company.dataset import camera_of, read_image  # 파일명→카메라, 한글 경로 읽기

N_POINTS = 96                                                       # 저장할 원 둘레 점 수 (기존 라벨과 같음)
SKIP_DATES = {"20260929"}                                           # 이미 라벨 있는 날짜는 제외


def date_of(path):
    """PLC-20261012-... → '20261012'."""
    return Path(path).name.split("-")[1]


def select_images(normal_dir, per_group, seed):
    """(날짜, 카메라)마다 per_group장씩 고정 seed로 뽑는다."""
    groups = defaultdict(list)                                      # (날짜, 카메라) → 파일 목록
    for path in sorted(Path(normal_dir).glob("*.png")):
        if date_of(path) not in SKIP_DATES:
            groups[(date_of(path), camera_of(path))].append(path)
    rng = random.Random(seed)                                       # 다시 실행해도 같은 선택
    chosen = []
    for key in sorted(groups):
        chosen += rng.sample(groups[key], min(per_group, len(groups[key])))  # 장수가 모자라면 있는 만큼
    return chosen


def circle_polygon(center, radius, width, height):
    """원 → 0~1 좌표 점 96개 문자열 (class 1)."""
    angles = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)    # 둘레를 고르게 나눈 각도
    xs = (center[0] + radius * np.cos(angles)) / width              # x를 이미지 폭으로 나눠 0~1로
    ys = (center[1] + radius * np.sin(angles)) / height             # y를 이미지 높이로 나눠 0~1로
    return "1 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))


def label_one(image, title, scale):
    """한 장을 클릭해서 (중심, 반지름)을 돌려준다. 건너뛰면 'skip', 끝내면 'quit'."""
    points = []                                                     # 원본 좌표로 저장한 클릭 점

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x / scale, y / scale))                   # 화면 좌표 → 원본 좌표
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()                                            # 마지막 점 취소

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)  # 화면용 축소본
    while True:
        canvas = shown.copy()
        for px, py in points:                                       # 찍은 점 표시 (빨강)
            cv2.circle(canvas, (round(px * scale), round(py * scale)), 4, (0, 0, 255), -1)
        fit = _fit_circle(np.array(points)) if len(points) >= 3 else None  # 3점 이상이면 원 맞춤
        if fit is not None:                                         # 맞춘 원과 중심 표시 (초록)
            (cx, cy), r = fit
            cv2.circle(canvas, (round(cx * scale), round(cy * scale)), round(r * scale), (0, 255, 0), 2)
            cv2.drawMarker(canvas, (round(cx * scale), round(cy * scale)), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(canvas, f"{title}  pts={len(points)}  Enter=save s=skip z=undo q=quit",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        cv2.imshow(title, canvas)
        key = cv2.waitKey(30) & 0xFF                                # 30ms마다 다시 그림
        if key in (13, 32) and fit is not None:                     # Enter·Space = 저장
            cv2.destroyWindow(title)
            return fit
        if key == ord("z") and points:
            points.pop()                                            # 마지막 점 취소
        if key == ord("s"):
            cv2.destroyWindow(title)
            return "skip"
        if key in (ord("q"), 27):                                   # q·Esc = 종료
            cv2.destroyWindow(title)
            return "quit"
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:  # 창 X 버튼 = 종료
            return "quit"


def main():
    parser = argparse.ArgumentParser(description="부품 바깥 원 정답 라벨 클릭 도구")
    parser.add_argument("--normal-dir", type=Path, default=ROOT / "data/raw/company/정상")
    parser.add_argument("--out", type=Path, default=ROOT / "data/labels_extra/company_center")
    parser.add_argument("--per-group", type=int, default=2)        # (날짜, 카메라)마다 장수
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--display", type=int, default=900)        # 화면에 보일 크기(px)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    chosen = select_images(args.normal_dir, args.per_group, args.seed)
    with (args.out / "selection.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "date", "camera"])                # 어떤 이미지를 골랐는지 기록
        writer.writerows([p.name, date_of(p), camera_of(p)] for p in chosen)

    todo = [p for p in chosen if not (args.out / f"{p.stem}.txt").exists()]  # 이미 단 것은 건너뜀
    print(f"선택 {len(chosen)}장, 남은 {len(todo)}장")
    for i, path in enumerate(todo, 1):
        image = read_image(path)                                    # 한글 경로도 읽기
        height, width = image.shape[:2]
        result = label_one(image, f"[{i}/{len(todo)}] {path.stem}", args.display / max(height, width))
        if result == "quit":
            break
        if result == "skip":
            continue
        center, radius = result
        (args.out / f"{path.stem}.txt").write_text(
            circle_polygon(center, radius, width, height) + "\n", encoding="utf-8")  # 기존 라벨 형식으로 저장
        print(f"저장: {path.stem}  중심=({center[0]:.1f}, {center[1]:.1f})  반지름={radius:.1f}")
    cv2.destroyAllWindows()
    done = sum((args.out / f"{p.stem}.txt").exists() for p in chosen)
    print(f"완료 {done}/{len(chosen)}장 → {args.out}")


if __name__ == "__main__":
    main()
