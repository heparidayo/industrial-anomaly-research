# 2026 학석박공동R&D 프로젝트

정상 편차를 고려한 산업 이미지 이상탐지 연구를 위한 협업 저장소입니다.

- [0. 서론](#0-서론)
- [1. 초기 설정](#1-초기-설정)
- [2. 프로젝트 구조와 아키텍처](#2-프로젝트-구조와-아키텍처)
- [3. 협업 규칙 및 주의사항](#3-협업-규칙-및-주의사항)
- [4. 문의](#4-문의)

## 0. 서론

안녕하세요. 이번 학석박 R&D 프로젝트에서 로즈마리팀의 팀장을 맡은 게임학과 3학년 김경건입니다.

저를 포함해 머신러닝/딥러닝 프로젝트가 익숙치 않은 팀원분도 함께 진행할 수 있도록 이 리드미를 작성했습니다. 초기 설정부터 프로젝트 전체 흐름, 역할별 코드 구조, 협업 등을 정리했습니다.

해당 레포는 각자의 코드를 같은 흐름으로 연결하기 위한 기본적인 틀 입니다. 각 담당자가 전처리, 모델구현, 평가를 추가하고 다른 팀원도 같은 조건과 명령으로 실험을 다시 실행할 수 있도록 만드는 것이 목적입니다.

이미 본인 환경에서 모델 구현을 진행하신 분은 그 코드를 이어서 사용하시면 됩니다. 저장소에 맞추려고 처음부터 다시 구현하지 말고 실행 방법과 필요한 설정만 정리해서 저희의 입출력에만 연결해주세요.

직접 리드미와 코드를 읽고 진행하셔도 되고, 본인이 사용하는 AI나 학교에서 지원하는 클로트 team에 리드미와 레포를 먹여 활용하셔도 됩니다. 단, 산업체 데이터를 다룰때는 조심해주세요.

모르는 내용이나 막히는 부분은 부담 없이 질문해 주세요. 이번 학기 잘 부탁드립니다!

## 1. 초기 설정

아래는 `Windows PowerShell`과 `Python 3.11` 기준입니다.

### 1-1. 준비 프로그램과 저장소 내려받기

[Git](https://git-scm.com/downloads)과 Python 3.11을 먼저 설치해 주세요.

프로젝트를 보관할 위치에서 PowerShell을 열고 실행합니다.

```powershell
git clone https://github.com/heparidayo/industrial-anomaly-research.git
Set-Location industrial-anomaly-research
py -3.11 --version
```

이후 명령은 모두 프로젝트 루트(requirements.txt가 있는 폴더)에서 실행합니다.

### 1-2. 가상환경 만들기

가상환경은 이 프로젝트에서 사용할 Python 패키지를 별도로 설치하는 공간입니다.

```powershell
py -3.11 -m venv .venv
```
만약, `py` 명령이 없다면 `python -m venv .venv`로 생성할 수 있습니다.

### 1-3. 패키지 설치

현재 공용 PatchCore 예시에서 사용하는 조합은 `torch 2.9.0`, `torchvision 0.24.0`입니다. 아래 GPU 혹은 CPU 중 하나만 선택하면됩니다.

- CUDA 12.8 빌드 (GPU)

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

- CPU 빌드 (GPU가 없다면..)

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
출처는 [PyTorch 공식 이전 버전 안내](https://pytorch.org/get-started/previous-versions/#v290)입니다. 

> 특히 모델 담당 분들은, 모델 구현에서 별도 버전이 필요하면 담당 모델의 의존성 파일과 설치 방법을 추가해주세요. 예를 들어 `requirements-draem.txt`처럼 분리할 수 있습니다. 혹시나 공용 패키지 버전을 바꿔야되는 경우 저 혹은 다른 모델 담당자 분들에게 미리 말씀주시면 좋겠습니다.

- DRAEM 담당 추가 설치 (위 공용 설치 후 실행)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-draem.txt
.\.venv\Scripts\python.exe scripts/draem/check_draem_env.py
```

공용 패키지 버전(torch, torchvision, numpy, opencv-python)은 바꾸지 않고 `imgaug`, `tensorboard`만 추가합니다. imgaug 0.4.0은 NumPy 2.x에서 그대로 동작하지 않아 `src/models/draem/imgaug_compat.py`에서 보완했습니다. DRAEM 이외 모델 담당자는 설치하지 않아도 됩니다.

### 1-4. 로컬 폴더와 데이터 준비

데이터와 결과 폴더(`./data`, `./outputs`)은 github로 전달되지 않으므로 미리 준비해둔 코드로 생성해줍니다.

```powershell
New-Item -ItemType Directory -Force -Path data, outputs, experiments | Out-Null
```

공개 데이터는 [MVTec AD 공식 페이지](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)의 다운로드 안내에 따라 받습니다. 사용 조건도 공식 페이지에서 확인해 주세요. 현재 공용 실행 예시는 `bottle`을 사용합니다.

```text
data/
└── raw/
    ├── mvtec_ad/
    │   └── bottle/
    │       ├── train/good/              # 정상 학습 이미지
    │       ├── test/good/               # 정상 테스트 이미지
    │       ├── test/<defect_type>/      # 불량 테스트 이미지
    │       └── ground_truth/<defect_type>/ # 결함 마스크
    └── company/                        # 산업체 제공 데이터
```

산업체 데이터는 저 혹은 저희 단톡으로 전달받고, 각 모델 담당자는 전달받은 데이터를 본인 모델의 입력 방식에 맞게 읽어 실험을 진행해 주세요.    
아래 실행 명령은 MVTec bottle 기준 예시입니다. 원본 파일은 유지하고, 학습/검증/테스트 분할은 팀 공통 기준(해당 레포 구조)를 사용합니다.


### 1-5. 설치 확인

폴더와 패키지 설정을 `check_setup.py` 코드로 확인합니다.

```powershell
.\.venv\Scripts\python.exe scripts/setup/check_setup.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch, torchvision, cv2, yaml, sklearn; print('패키지 확인 완료')"
```

bottle을 배치했다면 `inspect_dataset.py` 코드로 이미지 개수도 확인합니다.

```powershell
.\.venv\Scripts\python.exe scripts/preprocessing/inspect_dataset.py
```

NVIDIA GPU를 사용하는 분은 `check_torch.py` 코드로 다음 점검도 진행합니다. 이 코드는 GPU에서 실제 텐서 연산을 진행하고, CPU 환경에서는 CUDA를 사용할 수 없다고 오류가 잡히게 되어있습니다.

```powershell
.\.venv\Scripts\python.exe scripts/setup/check_torch.py
```

## 2. 프로젝트 구조와 아키텍처

먼저 폴더 역할 구분입니다.

- **`src/`**: 여러 실행 코드에서 재사용하는 함수와 모델을 구현합니다.
- **`scripts/`**: 명령을 받아 `src/`의 기능을 호출하는 진입점입니다.
- **`configs/`**: 밝기·회전 등 실험 조건을 YAML로 관리합니다.
- **`data/`**: 원본과 전처리 데이터를 로컬에 보관합니다.
- **`outputs/`**: 가중치, (patchCore 기준) Memory Bank, 예측 CSV, 평가 결과를 로컬에 저장합니다.

새로운 변환 함수나 모델 로직은 `src/`에 넣고, 실행 인자와 호출 순서는 `scripts/`, 바꿔 가며 비교할 조건은 `configs/`로 나눕니다.

```text
configs/experiments/      # 공용 실험 조건
scripts/
├── setup/               # 설치 및 환경 점검
├── preprocessing/       # 데이터 점검, 정상편차 생성/검증
├── patchcore/           # 현재 공용 PatchCore 예시 실행
└── evaluation/          # 예측 CSV 평가
src/
├── preprocessing/       # 이미지 변환 함수
├── models/patchcore/    # 현재 공용 PatchCore 예시 구현
└── evaluation/          # 지표 계산, 시각화
```

DRAEM과 EfficientAD 담당자는 구현을 연결하면서 `src/models/draem/`, `scripts/draem/` 및 `src/models/efficientad/`, `scripts/efficientad/`처럼 담당 폴더를 추가하면 됩니다. 위 모델의 폴더와 실행 코드는 아직 공용 저장소에 없습니다.

### 2-1. 전체 아키텍처

(제공된 자동차 정밀 부품 이미지에서 어떤 모델 + 전처리 조합이 정상과 불량을 잘 구분하며, 밝기/회전/위치 변화에도 안정적으로 동작하는가?)

### 2-2. 전처리 아키텍처

현재 전처리는 `test/good`의 정상 이미지에 밝기/회전/X축 이동을 각각 적용합니다. 예를 들어 밝기 `20`은 20% 증가, 회전 `5`는 5도, 이동 `6`은 원본 이미지에서 X축 6픽셀 이동입니다. 각 변형은 원본에서 개별 생성합니다.

```mermaid
flowchart TB
    CFG["YAML 설정"] --> LOAD["preprocess.py: load_config"]
    INPUT["test/good의 PNG"] --> READ["variations.py: load_image"]
    LOAD --> MAKE["preprocess.py: make_variations"]
    READ --> MAKE
    MAKE --> ORIGINAL["원본 유지"]
    MAKE --> BRIGHT["adjust_brightness"]
    MAKE --> ROTATE["rotate_image"]
    MAKE --> MOVE["translate_image"]
    ORIGINAL --> SAVE["variations.py: save_image"]
    BRIGHT --> SAVE
    ROTATE --> SAVE
    MOVE --> SAVE
    SAVE --> OUTPUT["조건별 PNG + manifest.csv + config_used.yaml"]
    OUTPUT --> CHECK["별도 실행: validate_dataset.py"]
    CHECK --> REPORT["파일 누락·중복·편차 조건 검사 결과"]
```

파일 기준으로 읽으면 다음 순서입니다.

```text
configs/experiments/bottle_variation_v2.yaml
  → scripts/preprocessing/preprocess.py
  → src/preprocessing/variations.py의 이미지 읽기·변형·저장 함수
  → data/processed/bottle/bottle_variation_v2/
  → scripts/preprocessing/validate_dataset.py로 결과 검사
```

기존 전처리 실행 명령입니다.

```powershell
.\.venv\Scripts\python.exe scripts/preprocessing/preprocess.py --config configs/experiments/bottle_variation_v2.yaml
.\.venv\Scripts\python.exe scripts/preprocessing/validate_dataset.py data/processed/bottle/bottle_variation_v2/manifest.csv
```

생성되는 `manifest.csv`는 “어느 원본에서 어떤 변형을 만들었는지” 적어 둔 목록입니다. 현재 컬럼은 `image_id, source_image, label, variation, level`이며 라벨은 `normal` 문자열입니다. `config_used.yaml`은 생성에 사용한 설정의 사본입니다.

같은 실험 이름으로 다른 조건을 사용하면 충돌을 막기 위해 실행이 중단됩니다. 새로운 조건은 YAML의 `experiment.name`도 바꿔 주세요.

*TODO 아래 작성 필요*

### 2-3. 모델 아키텍처

모델 내부 구조와 학습 방식은 담당자가 사용하는 구현에 따라 달라집니다. 각 모델의 상세 아키텍처는 담당자가 자신의 실제 코드 기준으로 설명하고 이 문서에 추가해 주세요.

현재 공용 저장소의 PatchCore는 제가 실행 흐름을 확인하려고 만든 예제입니다. 별도로 진행 중인 PatchCore 담당자의 구현은 담당자 설명과 설정을 기준으로 연결합니다.

#### 2-3-1. 예시 모델 아키텍처
#### 2-3-2. PatchCore 아키텍처
#### 2-3-3. AfficientAD 아키텍처
#### 2-3-4. DRAEM 아키텍처


### 2-4. 평가 아키텍처

(작성 예정)

## 3. 협업 규칙 및 주의사항

(작성 예정)

### 3-1. 데이터 

(작성 예정)

- 카메라 1
- 카메라 2
- 카메라 1 + 2
- 
### 3-2. Git에 올리는 것 vs 올리면 안되는 것

(작성 예정)

- 공유할거 
- 공유안할거

## 4. 문의

설정 오류, 코드 연결 등등 궁금한 점은 단톡방이나 따로 알려드린 개인 연락처로 편하게 연락해 주세요. 카톡, 전화, 디코 어느 쪽이든 괜찮습니다.
