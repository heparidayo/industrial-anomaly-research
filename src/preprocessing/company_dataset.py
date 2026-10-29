# 산업체(회사) 데이터 준비용 함수
#
# 회사 원본 데이터(data/raw/company) 구조:
#   정상/*.png                          정상 이미지 (camera_1, camera_2 섞여 있음)
#   <불량>/images/train/*.png            불량 이미지 (over, step, 미흡)
#   <불량>/labels/train/*.txt            YOLO segmentation 라벨
#     - class 0: 실제 부품 테두리 윤곽
#     - class 1: 기준 원 (점 96개) → 부품 위치와 크기를 알 수 있다
#   파일명 끝이 _camera_1 / _camera_2 로 카메라를 구분한다.
#
# 이 파일의 함수로 MVTec 형식(train/good, test/<불량>)을 만들면
# 기존 PatchCore / 평가 스크립트를 그대로 쓸 수 있다.

import random
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ("camera_1", "camera_2")
NORMAL_FOLDER = "정상"
DEFECT_FOLDERS = ("over", "step", "미흡")
IMAGE_SIZE = 1216  # 원본 이미지 한 변 픽셀 (라벨 좌표 0~1 → 픽셀 변환용)


def camera_of(path):
    """파일명으로 카메라 이름을 알아낸다."""
    return "camera_1" if Path(path).stem.endswith("camera_1") else "camera_2"


def list_images(src, folder):
    """폴더 안의 PNG 목록 (불량 폴더는 images/train 아래에 있다)"""
    d = Path(src) / folder
    sub = d / "images" / "train"
    # 정상 폴더는 PNG가 바로 들어 있고, 불량 폴더는 images/train 아래에 있다
    return sorted((sub if sub.is_dir() else d).glob("*.png"))


def load_reference_circles(src, image_size=IMAGE_SIZE):
    """
    라벨의 기준 원(class 1)에서 이미지별 중심과 반지름을 구한다.

    반환: {파일명(stem): (cx, cy, r)}  (원본 이미지 픽셀 좌표)
    정상 이미지의 라벨은 같은 부품의 불량 폴더 labels에 함께 들어 있다.
    """
    circles = {}
    # 모든 불량 폴더의 라벨 파일 (*/labels/train/*.txt)
    for path in Path(src).glob("*/labels/train/*.txt"):
        # 라벨 한 줄 = "클래스 x1 y1 x2 y2 ..." (좌표는 0~1 비율)
        for line in path.read_text().splitlines():
            values = line.split()
            if values and values[0] == "1":
                # 점 좌표를 [점 개수, 2] 모양으로 바꾸고 픽셀 단위로 변환
                xy = np.array(values[1:], float).reshape(-1, 2) * image_size
                # 원 위의 점 96개의 평균 = 중심
                center = xy.mean(0)
                # 중심에서 각 점까지 거리의 평균 = 반지름
                radius = np.linalg.norm(xy - center, axis=1).mean()
                circles[path.stem] = (float(center[0]), float(center[1]), float(radius))
    return circles


def roi_boxes(circles, half):
    """
    카메라별 고정 ROI 박스 [x1, y1, x2, y2]

    이미지마다 라벨로 자르면 실제 공정에서는 쓸 수 없는 정답 정보를 쓰는 셈이라,
    카메라별 기준 원 중심의 평균에서 ±half 픽셀인 박스 하나로 고정한다.
    """
    boxes = {}
    for cam in CAMERAS:
        # 이 카메라 이미지들의 기준 원 중심만 모으기
        centers = [(cx, cy) for stem, (cx, cy, _) in circles.items() if stem.endswith(cam)]
        # 평균 중심을 정수 픽셀로 반올림
        cx, cy = np.mean(centers, axis=0).round().astype(int)
        # 중심에서 상하좌우 half 픽셀 → 정사각형 박스 (half=420이면 840x840)
        boxes[cam] = [int(cx - half), int(cy - half), int(cx + half), int(cy + half)]
    return boxes


def split_by_camera(src, seed, n_train):
    """
    카메라별로 정상 이미지를 학습 n_train장 / 나머지 테스트로 나눈다.
    불량 이미지는 전부 테스트.

    반환: {카메라: [(split, 클래스, 경로), ...]}
      split  : "train" / "test"
      클래스 : "good" / "over" / "step" / "미흡"
    """
    normals = list_images(src, NORMAL_FOLDER)                            # 정상 32장 (두 카메라 섞임)
    defects = {name: list_images(src, name) for name in DEFECT_FOLDERS}  # {"over": [...], "step": [...], "미흡": [...]}

    # 하나의 난수 생성기로 camera_1 → camera_2 순서로 섞는다 (같은 seed = 같은 분할)
    rng = random.Random(seed)
    split = {}
    for cam in CAMERAS:
        # 이 카메라의 정상 이미지만 골라 무작위로 섞기
        shuffled = [p for p in normals if p.stem.endswith(cam)]
        rng.shuffle(shuffled)
        # 앞 n_train장 = 학습, 나머지 = 테스트 (목록은 다시 파일명순으로 정렬해서 보기 좋게)
        rows = [("train", "good", p) for p in sorted(shuffled[:n_train])]
        rows += [("test", "good", p) for p in sorted(shuffled[n_train:])]
        # 이 카메라의 불량 이미지는 전부 테스트
        for name, paths in defects.items():
            rows += [("test", name, p) for p in paths if p.stem.endswith(cam)]
        split[cam] = rows
    return split


def read_image(path):
    """한글 경로도 읽을 수 있게 바이트로 읽어 디코딩 (cv2.imread는 한글 경로 불가)"""
    # np.fromfile: 파일을 바이트 배열로 읽기 → imdecode: PNG 바이트를 이미지(BGR)로 변환
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def write_image(path, image):
    """한글 경로도 쓸 수 있게 인코딩 후 바이트로 저장"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # 저장 폴더가 없으면 생성
    # imencode: 이미지를 PNG 바이트로 변환 → tofile: 바이트를 파일로 저장
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"이미지 인코딩 실패: {path}")
    buffer.tofile(str(path))


def crop(image, box):
    """[x1, y1, x2, y2] 박스로 자르기"""
    x1, y1, x2, y2 = box
    # 이미지 배열은 [세로(y), 가로(x)] 순서
    return image[y1:y2, x1:x2]
