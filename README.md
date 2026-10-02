# CN7 · RG3 공정 데이터·초기 모델 파이프라인

[7단계 시안](03.modeling/common/전체파이프라인_7단계.png) · [구조·구현 범위](03.modeling/common/CN7_RG3_CI_CT_CD_전체구조.md) · [이전 시안](03.modeling/common/전체파이프라인_찐막.png)

CN7·RG3는 데이터·모델·버전을 각각 분리합니다. 별도 검사·라벨 연결 단계 없이 라벨 포함 배치에 기존 전처리 정책을 적용합니다.

| 파일 | 현재 구현 |
|---|---|
| [EDA](01.EDA/01_eda.ipynb) | 분포·중복·상충 라벨·변수 조합 분석 |
| [전처리](02.preprocessing/02_preprocess.ipynb) | 입력 검증·원본 연결·제품 빈도·위험 이력 집계 |
| [분할](02.preprocessing/03_split.ipynb) | 고정 Test·개발 4-fold·누수 점검 |
| [IF](03.modeling/models/04.1_isolation_forest.ipynb) | 정상 학습·점수·순위·고정 모델 저장 |
| [OCSVM](03.modeling/models/ocsvm.ipynb) | 독립 초기 학습·설정 탐색·평가·초기 후보 등록 |
| [공통](03.modeling/common/00_pipeline_core.ipynb) | 입력 검증·모델 저장/로드·해시·버전 |
| [LR 매뉴얼](03.modeling/models/LOGISTIC_REGRESSION_MODELING_MANUAL.md) | 설계 지침·예제, 실행 모델 미구현 |
| [RF 매뉴얼](03.modeling/models/RANDOM_FOREST_MODELING_MANUAL.md) | 설계 지침·예제, 실행 모델 미구현 |

현재 배치 운영·드리프트·운영 CT·승격/롤백 노트북은 없습니다. 초기 모델의 최종 적합은 신규 배치 기반 CT와 구분합니다. 드리프트·재학습은 보류된 설계입니다.

## 현재 실행 순서

1. EDA로 데이터 구조를 확인합니다.
2. 전처리를 CN7·RG3 각각 실행한 뒤 분할을 실행합니다. 유효한 산출물이 있으면 재사용합니다.
3. IF·OCSVM을 데이터셋별로 실행합니다. OCSVM 기본 원본 순서는 IF와 독립이고 선택적 IF 정렬 모드만 IF 산출물이 필요합니다.

데이터셋 변경 시 커널을 재시작합니다. IF 고정 버전이 있으면 저장된 최종 모델을 유지합니다. IF 입력 내보내기는 fold 정렬 자료 재생성을 위해 fold 모델을 다시 적합하지만 최종 IF를 교체하지 않습니다.

## 저장과 제한

- `data/`: 원본·스키마·전처리·분할.
- `runtime/{cn7|rg3}/models/`: 고정 IF·OCSVM 초기 후보.
- `runtime/{cn7|rg3}/registry.json`: 고정 IF와 운영 포인터. 현재 운영 OCSVM은 미지정.
- `runtime/{cn7|rg3}/if_inputs/`: IF 점수·순위·선택적 정렬 비교 자료.
- `output/`: 실행 시 생성되는 보고서. 영속 모델 저장소와 구분.

IF 점수는 불량 확률이 아니며 조기 탐지 효과는 검증 전입니다. 비라벨 스케일 정합은 미확인이라 자동 투입하지 않습니다.

목표 7단계: 데이터 준비 → EDA·전처리 → 초기 모델·기준선 → 배치 유입·추론 → 드리프트 감지 → 모델별 CT → 검증·교체. 신규 배치에 기존 전처리 정책을 재사용하되 추론 시 저장된 모델 변환을 사용합니다. 비라벨은 정합 확인 후 추론·감지만 수행하며 현 설계의 재학습에서는 제외합니다.
