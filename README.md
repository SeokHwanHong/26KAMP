# CN7 · RG3 공정 데이터·모델·로컬 운영 파이프라인

2026-10-06 권장안 1~5 확정: CLI와 운영 노트북은 `governance_runtime.Operations`를 사용합니다. 독립 제품 검사 후 명시적 모델/검사 역할 전환, RG3 우선·무작위 검사 계획, 승인 참조 갱신, 영구 위험 이력과 학습 범위 분리, CT/CD 승인 기록을 추가했습니다. [현재 7단계 인수 기준](03.modeling/common/WORKFLOW_ACCEPTANCE.md)과 [확정 운영 정책·설정](03.modeling/common/GOVERNANCE_OPERATIONS.md)을 참고하세요. 검증은 `python run_checks_only.py`로 실행합니다.

2026-10-03: 배치 추론·드리프트·모델별 CT·평가·승격/롤백의 로컬 실행 경로를 추가했습니다. [운영 가이드](03.modeling/common/OPERATIONS_GUIDE.txt), [운영 노트북](03.modeling/04_operations.ipynb), [로지스틱 실험 노트북](03.modeling/models/05_logistic_regression.ipynb)을 진입점으로 사용합니다. 실제 운영 모델은 이번 작업에서 교체하지 않았습니다.

팀 공유 검증: `python 03.modeling/tests/run_checks.py`. [테스트 플로우](03.modeling/common/TEST_FLOW.txt)와 [Claude 전달 프롬프트](exports/Claude_KAMP_프로젝트학습_테스트_검토_프롬프트.txt)를 함께 참고하세요. 임시 파일에 의존하지 않는 정식 검증 진입점입니다.

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
| [LR](03.modeling/models/logistic_regression.py) | 8개 입력 구성·240개 설정·1001개 임계값 탐색, CN7/RG3 실행·독립 재검산 |
| [운영 공통](03.modeling/common/pipeline_runtime.py) | 배치 전처리·추론·분포 감지·4종 모델 재학습·구간 안내·검증·승격/롤백 |
| [RF 매뉴얼](03.modeling/models/RANDOM_FOREST_MODELING_MANUAL.md) | 운영 공통에 고정 기준 모델과 전체 재학습 구현. 매뉴얼 전체 탐색은 미실행 |

초기 모델의 최종 적합과 신규 배치 기반 CT는 구분합니다. 운영 코드는 합성 자료를 사용한 격리 통합 테스트를 통과했으며 현장 운영 검증을 의미하지 않습니다. 경보선·배포 기준은 기록 가능한 시험 기본값입니다. 실제 좌표 정합·원인 확인·독립 평가 자료가 있어야 운영 교체를 검토할 수 있습니다.

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
- `runtime/{cn7|rg3}/{batches,baselines,drift,evaluations,assessments}/`: 신규 운영 경로의 추적 이력.
- `output/logistic/{cn7|rg3}/<run_id>/`: LR 탐색·OOF·Test·계수·분포 안내·감사 결과.

OCSVM 결과도 실행별 고유 run_id를 사용합니다. 초기 후보 등록 시 감사한 파일 해시·선택·코드·데이터·분할이 현재 산출물과 같은지 확인합니다.

IF 점수는 불량 확률이 아니며 조기 탐지 효과는 검증 전입니다. 비라벨 스케일 정합은 미확인이라 자동 투입하지 않습니다.

목표 7단계: 데이터 준비 → EDA·전처리 → 초기 모델·기준선 → 배치 유입·추론 → 드리프트 감지 → 모델별 CT → 검증·교체. 신규 배치에 기존 전처리 정책을 재사용하되 추론 시 저장된 모델 변환을 사용합니다. 비라벨은 정합 확인 후 추론·감지만 수행하며 현 설계의 재학습에서는 제외합니다.
