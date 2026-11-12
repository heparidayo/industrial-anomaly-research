# 산업체 데이터 공통 전처리

모델과 무관하게 재사용하는 회사 데이터 준비·합성 편차 기능입니다.

- 실행: scripts/preprocessing/company/prepare_dataset.py
- 설정: configs/datasets/company_v1.yaml
- 원본: data/raw/company/ 아래 정상/와 불량 종류 폴더
- 준비 결과: data/processed/company/<category>/
- 정상편차 설정: configs/experiments/company/
- 정상편차 결과: data/processed/company_variations/<category>/<experiment>/

현재 설정의 status는 proposed입니다. 팀 공통 분할·ROI·허용 편차는 전처리·모델·평가 담당자가 검토 후 확정합니다.
같은 촬영번호의 두 카메라는 같은 분할에 둡니다. 학습 그룹의 불량 뷰는 unused로 기록합니다.
validation은 모델·조건 검토에, test는 설정을 고정한 뒤 최종 평가에 사용합니다.
split.csv는 카메라 1·2·1+2에 동일한 그룹 배정을 사용합니다.

ROI 보정은 학습 정상 라벨만 사용하고 테스트 이미지에는 고정 박스를 적용합니다.
이미지 크기·기준 원 라벨이 다른 데이터에는 설정과 읽기 함수를 검토해야 합니다.
부품 회전은 원 마스크 안쪽을 회전해 합성 편차를 만드는 기능이며 실제 회전의 물리적 재현을 보장하지 않습니다.
회전 원 라벨은 합성 이미지 생성에만 사용합니다.

실행 예시는 [PatchCore 실험 안내](../../models/patchcore/README.md)의 공통 데이터 준비 절을 참고하세요.

## 스프링 시작점 정렬 (제안)

스프링 끝면은 선재 시작점(연마되지 않은 어두운 부채꼴)의 위치가 부품마다 달라, 모델이 이 위치 차이를 불량으로 오해할 수 있습니다.
기업 검사 기준(시작점 0° 기준 -5°~100° 검사 제외)에 맞춰, 시작점을 같은 방향으로 돌리고 제외 영역과 부품 바깥을 가린 카테고리를 만듭니다.

- 구현: alignment.py (중심·시작점 검출, 정렬·마스킹), prepare_aligned.py (카테고리 단위 내보내기)
- 실행: scripts/preprocessing/company/align_dataset.py
- 설정: configs/datasets/company_align_v1.yaml (status: proposed)
- 입력: prepare_dataset.py로 만든 원본 크기 카테고리 (ROI 카테고리는 받지 않음)
- 출력: data/processed/company/<category>_aligned/ (split.csv는 입력과 동일, alignment.csv에 이미지별 검출 결과)

```powershell
.\.venv\Scripts\python.exe scripts/preprocessing/company/align_dataset.py --config configs/datasets/company_align_v1.yaml
```

라벨은 카메라별 대략 중심·반지름을 학습 정상 이미지로만 정할 때 사용하고, 각 이미지의 중심과 시작점은 사진에서 찾습니다.
중심은 가운데 구멍 경계 → 하얀 연마면 바깥 경계 순서로 원을 맞추고, 시작점은 연마면 띠의 밝기가 "어두움 → 밝음"으로 가장 크게 바뀌는 각도입니다.
회사 데이터 97장에서 라벨 외곽선 기준과 비교한 시작점 각도 차이는 중앙값 약 0.5°, 최대 약 4.5°입니다.
조명이나 부품이 바뀌면 alignment.csv의 start_rise(밝기 상승폭)와 low_confidence를 먼저 확인하세요.

## 면취 고리 펼치기 (제안)

기업 불량 정의는 면취 폭 w'(반지름 방향 두께)와 각도 범위로 되어 있습니다(과다 = 폭 넓음, 미흡 = 폭 좁음, 다단 = 일부 각도만 폭이 다름).
면취 고리를 "가로 = 각도, 세로 = 반지름"인 띠로 펼치면 폭이 띠의 세로 두께로 바로 보이고, 구멍·배경을 버려 224px로 줄일 때 폭 정보가 덜 사라집니다.

- 구현: polar.py (띠 펼치기·토막 쌓기, 중심·시작점 검출은 alignment.py 재사용), prepare_polar.py (카테고리 단위 내보내기)
- 실행: scripts/preprocessing/company/unwrap_dataset.py
- 설정: configs/datasets/company_polar_v1.yaml (status: proposed)
- 입력: prepare_dataset.py로 만든 원본 크기 카테고리 (ROI·정렬 카테고리는 받지 않음)
- 출력: data/processed/company/<category>_polar/ (split.csv는 입력과 동일, polar.csv에 이미지별 검출 결과)

```powershell
.\.venv\Scripts\python.exe scripts/preprocessing/company/unwrap_dataset.py --config configs/datasets/company_polar_v1.yaml
```

시작점 기준 5°~260°(기업 검사 제외 -5°~100°를 뺀 구간), 외곽선 반지름의 0.75~1.05배를 펼칩니다.
반지름 범위는 학습 정상 라벨의 면취 안쪽 경계(0.84~0.97배)에 중심 오차 여유를 더해 정했습니다.
긴 띠는 4토막으로 잘라 위아래로 쌓아 432×432 정사각형으로 저장합니다. PatchCore 입력 변환(짧은 변 232 → 가운데 224 자르기)에서 띠가 잘리지 않게 하기 위해서입니다.
중심이 틀리면 펼친 띠가 물결처럼 휘므로, 결과가 이상하면 polar.csv의 중심 값과 샘플 이미지를 먼저 확인하세요.

쌓지 않은 긴 띠(1728×108)가 필요하면 `configs/datasets/company_polarstrip_v1.yaml`(rows 1, columns 1728)로 `<category>_polarstrip`을 만듭니다.
펼치는 범위·해상도는 4토막 버전과 같아서, 4토막으로 다시 쌓으면 `_polar` 이미지와 픽셀까지 같습니다.
이 카테고리는 PatchCore를 `--input-mode full`로 실행해야 띠가 잘리지 않습니다([PatchCore 실험 안내](../../models/patchcore/README.md) 참고).

```powershell
.\.venv\Scripts\python.exe scripts/preprocessing/company/unwrap_dataset.py --config configs/datasets/company_polarstrip_v1.yaml
```
원래 bottle용 preprocess.py·validate_dataset.py와 src/preprocessing/variations.py는 유지합니다.
