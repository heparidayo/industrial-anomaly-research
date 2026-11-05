"""산업체 이미지의 식별, 촬영번호 단위 분할, 학습 데이터 기반 ROI 보정."""

from collections import defaultdict
from pathlib import Path
import random
import re

import cv2
import numpy as np

CAMERAS = ("camera_1", "camera_2")
NORMAL_FOLDER = "정상"
DEFECT_FOLDERS = ("over", "step", "미흡")
IMAGE_SIZE = 1216
CAMERA_PATTERN = re.compile(r"^(?P<group>.+)_(?P<camera>camera_[12])$")


def camera_of(path):
    match = CAMERA_PATTERN.fullmatch(Path(path).stem)
    if not match:
        raise ValueError(f"촬영번호·카메라를 읽을 수 없는 파일명: {Path(path).name}")
    return match["camera"]


def group_of(path):
    """카메라 접미사를 뺀 촬영번호. 같은 부품의 두 뷰를 같은 분할에 둔다."""
    camera_of(path)
    return CAMERA_PATTERN.fullmatch(Path(path).stem)["group"]


def resolve_source(src):
    """정상/가 있는 원본 폴더 또는 그 폴더 하나를 포함한 상위 폴더."""
    src = Path(src).resolve()
    if (src / NORMAL_FOLDER).is_dir():
        return src
    candidates = sorted(p for p in src.iterdir() if p.is_dir() and (p / NORMAL_FOLDER).is_dir()) if src.is_dir() else []
    if len(candidates) != 1:
        raise ValueError("--src에 정상/ 및 불량 종류 폴더가 있는 원본 경로를 지정하세요.")
    return candidates[0]


def list_images(src, folder):
    directory = Path(src) / folder
    nested = directory / "images" / "train"
    return sorted((nested if nested.is_dir() else directory).glob("*.png"))


def split_by_camera(src, seed, n_train, n_validation=3):
    """두 카메라에 공통인 촬영번호 분할을 만든다.

    정상은 카메라당 n_train장 학습, n_validation장 검증, 나머지 테스트.
    학습 그룹의 불량 뷰는 unused로 기록해 검증·테스트에서 제외한다.
    불량만 있는 촬영 그룹은 검증·테스트에 배분한다.
    """
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in (n_train, n_validation)):
        raise ValueError("학습·검증 정상 장수는 양의 정수여야 합니다.")
    src = resolve_source(src)
    groups = defaultdict(list)
    seen = set()
    folders = [p for p in sorted(src.iterdir()) if p.is_dir()]
    for folder in folders:
        class_name = "good" if folder.name == NORMAL_FOLDER else folder.name
        for path in list_images(src, folder.name):
            key = (group_of(path), camera_of(path))
            if key in seen:
                raise ValueError(f"중복 촬영번호·카메라: {path.name}")
            seen.add(key)
            groups[key[0]].append((class_name, path))
    if not groups:
        raise ValueError("원본 PNG 이미지가 없습니다.")

    pools = defaultdict(list)
    for group_id, rows in sorted(groups.items()):
        signature = tuple(sorted(camera_of(path) for cls, path in rows if cls == "good"))
        pools[signature].append(group_id)
    rng = random.Random(seed)
    for group_ids in pools.values():
        rng.shuffle(group_ids)

    assignments = {}

    def allocate(name, count):
        both = pools[CAMERAS]
        common_count = min(count, len(both))
        required_single = count - common_count
        if any(len(pools[(cam,)]) < required_single for cam in CAMERAS):
            raise ValueError(f"{name}: 카메라별 정상 촬영그룹이 부족합니다.")
        chosen = both[:common_count]
        del both[:common_count]
        for cam in CAMERAS:
            chosen += pools[(cam,)][:required_single]
            del pools[(cam,)][:required_single]
        assignments.update((group_id, name) for group_id in chosen)

    allocate("train", n_train)
    allocate("validation", n_validation)
    for signature, group_ids in pools.items():
        if signature:
            assignments.update((group_id, "test") for group_id in group_ids)
    defects_only = pools[()]
    cut = len(defects_only) // 2
    assignments.update((group_id, "validation" if i < cut else "test")
                       for i, group_id in enumerate(defects_only))

    result = {cam: [] for cam in CAMERAS}
    for group_id, rows in sorted(groups.items()):
        split = assignments[group_id]
        for class_name, path in rows:
            role = "unused" if split == "train" and class_name != "good" else split
            result[camera_of(path)].append((role, class_name, path))
    for cam, rows in result.items():
        for name in ("train", "validation", "test"):
            classes = {cls for split, cls, _ in rows if split == name}
            if "good" not in classes:
                raise ValueError(f"{cam}/{name}: 정상 이미지가 없습니다. 장수를 조정하세요.")
            if name != "train" and not (classes - {"good"}):
                raise ValueError(f"{cam}/{name}: 불량 촬영그룹이 부족합니다.")
    return result


def load_reference_circles(src, image_size=IMAGE_SIZE, *, allowed_stems=None):
    """기준 원 class 1을 읽는다. ROI 보정에는 학습 정상 stem만 전달한다."""
    circles = {}
    for path in sorted(Path(src).glob("*/labels/train/*.txt")):
        if allowed_stems is not None and path.stem not in allowed_stems:
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            values = line.split()
            if not values or values[0] != "1":
                continue
            xy = np.asarray(values[1:], dtype=float)
            if xy.size < 6 or xy.size % 2 or not np.isfinite(xy).all():
                raise ValueError(f"잘못된 기준 원 좌표: {path.name}")
            xy = xy.reshape(-1, 2) * image_size
            center = xy.mean(0)
            radius = np.linalg.norm(xy - center, axis=1).mean()
            if radius <= 0:
                raise ValueError(f"기준 원 반지름이 잘못되었습니다: {path.name}")
            if path.stem in circles:
                raise ValueError(f"기준 원 라벨 중복: {path.name}")
            circles[path.stem] = (float(center[0]), float(center[1]), float(radius))
    if allowed_stems is not None and set(allowed_stems) - circles.keys():
        raise ValueError("필요한 학습·변형 이미지의 기준 원 라벨이 누락되었습니다.")
    return circles


def roi_boxes(circles, half, *, calibration_stems):
    """학습 정상 이미지로만 카메라별 고정 ROI를 정한다."""
    if not isinstance(half, int) or half <= 0:
        raise ValueError("ROI 반 크기는 양의 정수여야 합니다.")
    if not calibration_stems or set(calibration_stems) - circles.keys():
        raise ValueError("학습 정상 ROI 보정 라벨이 부족합니다.")
    boxes = {}
    for cam in CAMERAS:
        centers = [(circles[stem][0], circles[stem][1])
                   for stem in sorted(calibration_stems) if camera_of(stem) == cam]
        if not centers:
            raise ValueError(f"{cam}: 학습 정상 ROI 보정 이미지가 없습니다.")
        cx, cy = np.asarray(centers).mean(axis=0).round().astype(int)
        boxes[cam] = [int(cx - half), int(cy - half), int(cx + half), int(cy + half)]
    return boxes


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"이미지 인코딩 실패: {path}")
    buffer.tofile(str(path))


def crop(image, box):
    """박스가 이미지 밖으로 나가면 조용히 잘못 자르지 않고 중단한다."""
    x1, y1, x2, y2 = box
    height, width = image.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"ROI가 이미지 범위를 벗어납니다: {box}, image={width}x{height}")
    return image[y1:y2, x1:x2]
