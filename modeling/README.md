# OCSVM 분석 노트북

현재 분석의 진입점은 데이터별 통합본입니다.

랜덤 포레스트 구현은 [전용 모델링 매뉴얼](RANDOM_FOREST_MODELING_MANUAL.md)을 참고하세요. 트리 구조·클래스 가중치·임계값의 F1 탐색, OOF 예제와 공정군 중요도 검증을 정리했습니다.

로지스틱 회귀 구현은 [전용 모델링 매뉴얼](LOGISTIC_REGRESSION_MODELING_MANUAL.md)을 참고하세요. L2 기준 모델, C·클래스 가중치·임계값의 공동 탐색, 코드 예제와 결과 검증 절차를 포함합니다.

다른 모델을 적용할 때는 [CN7·RG3 모델링·학습·검증 매뉴얼](MODELING_TRAINING_VALIDATION_MANUAL.md)을 참고하세요. 원본과 전처리·분할 코드를 전달해 산출물을 재현하는 순서, 현재 코드의 의존 파일, 정상 기반·지도학습의 학습 대상, 임계값을 포함한 F1 탐색 및 최종 검증을 정리했습니다.

| 데이터 | 통합본 |
|---|---|
| CN7 | [ocsvm_cn7_integrated.ipynb](ocsvm_cn7_integrated.ipynb) |
| RG3 | [ocsvm_rg3_integrated.ipynb](ocsvm_rg3_integrated.ipynb) |

CN7·RG3 통합본은 도메인 가설 정의 → 합산 validation F1 기반 개발 탐색 → 공정군 중요도 진단 → 고정 test 후속 평가 순서입니다. 새 커널에서 위에서 아래로 실행하며, 전처리와 고정 분할 파일은 기존 자료를 사용합니다.

CN7 최신 결과는 `output/ocsvm_cn7_integrated/validation_f1_20260929/`에 저장됩니다. `selection_manifest.json`은 F1 기반 개발 선택, `top_joint_candidates.csv`는 전체 조합 상위 20개, `top_model_candidates.csv`는 모델 설정별 대표 후보 상위 20개, `test_metrics.csv`는 선택 모델의 후속 평가입니다. RG3 최신 결과는 `output/ocsvm_rg3_integrated/validation_f1_20260929/`에 같은 구성으로 저장합니다. 이전 RG3 `baseline`·`exploration`·`feature_scenarios` 결과는 보존합니다.

두 데이터 모두 7개 시나리오별 616개 모델 설정과 임계값 93개를 탐색합니다. 각 fold의 정상 train 전체로 학습하고 validation의 TP·FP·FN을 합산한 OOF F1을 최대화합니다. 동률은 fold F1 표준편차와 입력 수로 비교합니다. FPR 상한 필터는 없으며 FPR·AP는 참고용입니다. 전체 임계값별 성능은 `threshold_search.csv`에 저장합니다. 선택 후 개발 정상 전체로 재학습하고 같은 숫자 임계값을 적용합니다. 추가 보정 분할은 없습니다. 공정군 중요도는 F1 감소량을 중심으로 보고하되 변수를 자동 삭제하지 않습니다.

각 통합본의 구현·회귀 테스트 6개·결과 감사는 모두 통합 노트북의 코드 셀에 포함되어 있습니다. 외부 프로젝트 `.py` import나 실행 스크립트 없이 새 커널에서 전체 실행하세요. A~G는 분석, H는 회귀 테스트, I는 전체 탐색 결과·상위 후보·저장 모델의 독립 재검증입니다. 데이터와 고정 분할 파일, numpy/pandas/scikit-learn/matplotlib/joblib/nbformat/IPython 설치는 필요합니다. 기존 AP 기반 통합본은 [보존본](archive/ocsvm_cn7_integrated_ap_20260929.ipynb), 기존 확대 탐색 결과는 `output/ocsvm_cn7_integrated/expanded_search_20260929/`에 남아 있습니다.

기존 `conservative`, `exploration`, `feature_scenarios` 노트북은 통합 이전 실험 기록으로 보존합니다. 기존 노트북을 먼저 실행할 필요 없이 통합본에서 전체 흐름을 실행할 수 있습니다.

이전 65:35 보정 분할 분석은 `archive/ocsvm_cn7_integrated_calibration_20260929.ipynb`, 그 결과는 `output/ocsvm_cn7_integrated/domain_fpr_20260929/`에 보존합니다.

이전 FPR 상한 기반 validation 탐색은 `archive/ocsvm_cn7_integrated_fpr_validation_20260929.ipynb`, 그 결과는 `output/ocsvm_cn7_integrated/validation_threshold_20260929/`에 보존합니다. 외부 모듈을 사용하던 마지막 F1 버전은 `archive/ocsvm_cn7_integrated_f1_external_20260929.ipynb`에 있습니다. 기존 `cn7_domain_fpr.py`, `test_cn7_domain_fpr.py`와 `tmp`의 생성·수정·감사 스크립트는 이전 작업 기록이며 현재 노트북 실행에는 필요하지 않습니다. 이후 변경은 통합 노트북 셀에서 수행하세요.

RG3는 CN7의 셀 구성을 적용하되 RG3 고정 분할과 데이터로 독립 학습·선택합니다. 개발 472개(불량 20개), test 119개(불량 5개)이며 각 validation fold에는 불량 5개가 있습니다. 공정 변수 24개가 같아 동일한 도메인 가설을 사용합니다.
