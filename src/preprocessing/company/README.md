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
원래 bottle용 preprocess.py·validate_dataset.py와 src/preprocessing/variations.py는 유지합니다.
