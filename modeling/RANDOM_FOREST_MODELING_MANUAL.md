# CN7·RG3 랜덤 포레스트 모델링·학습·검증 매뉴얼

작성일: 2026-09-29 · 버전: 1.0 · 상위 기준: [공통 매뉴얼 v1.1](MODELING_TRAINING_VALIDATION_MANUAL.md)

이 문서는 CN7과 RG3에 `RandomForestClassifier`를 독립 적용하기 위한 구현 지침이다. 원본 → 전처리 → 고정 분할 → 개발 탐색 → 최종 재학습 → test 후속 평가 순서를 따른다. 탐색 범위는 재현 가능한 초기 설계이며 검증된 최적값이 아니다. 문서의 예제는 구현을 돕는 부분 코드이고, 전체 모델링 결과를 뜻하지 않는다.

## 1. 목표와 모델별 차이

목표는 공정 입력 패턴에 불량 이력이 있는지(`PassOrFail=1`) 분류하는 것이다. 동일 입력 패턴의 원본 라벨을 최댓값으로 집계하므로 개별 제품 불량률 예측과 구분한다.

| 항목 | 랜덤 포레스트 적용 기준 |
|---|---|
| 학습 유형 | 지도학습 이진 분류, `training_mode=supervised` |
| Fold 학습 자료 | 해당 train의 정상·위험 전체. OCSVM의 정상 필터를 복사하지 않음 |
| 기본 입력 변환 | Train 전체에서 상수 열 제거. 원변수 기준 모델에는 scaler를 추가하지 않음 |
| 위험 점수 | `predict_proba`에서 라벨 1에 해당하는 열 |
| 판정 | `p > t`이면 위험. 기본 `predict()`로 선택 임계값을 대체하지 않음 |
| 탐색 대상 | 입력 구성·트리 구조·클래스 가중치·숫자 임계값 |
| 선택 지표 | 네 validation fold의 혼동행렬을 합산한 OOF F1 |
| 최종 학습 | CN7 개발 484개, RG3 개발 472개 전체 |

트리는 train에서 분할 기준을 최적화하고, 여러 트리의 출력을 모아 위험 점수를 만든다. **트리 학습 기준과 후보 선택 지표는 다르다.** 초기 실험은 트리의 `criterion="gini"`를 고정하고, 완성된 후보의 validation F1로 파라미터와 t를 선택한다.

클래스 가중치는 학습 시 클래스별 영향력을 바꾸며, 임계값은 학습 후 확률을 이진 판정으로 바꾼다. 두 요소를 별도로 정의하고 공동 비교한다. 다수 클래스가 많다는 이유로 accuracy를 선택 지표로 사용하지 않는다.

## 2. 전달물·전처리·분할

원본 CSV 4개와 전처리·split 코드를 전달한다. 상세 파일 목록과 현재 전처리의 스키마 JSON·기준 DOCX 의존성은 공통 매뉴얼 2절을 따른다. 분할 CSV는 실행 산출물이며 원본과 코드가 재현 기준이다.

1. [CN7 전처리](../preprocessing/preprocess_cn7_conservative.ipynb), [RG3 전처리](../preprocessing/preprocess_rg3_conservative.ipynb)를 각각 전체 실행한다.
2. [공통 split 코드](../preprocessing/split_conservative_data.ipynb)를 실행한다. Seed 42, 계층화 test 20%, 개발 계층화 4-fold와 행 정렬 규칙을 유지한다.
3. 기준 배정표·해시와 대조하고 공통 매뉴얼 3절의 loader로 입력을 검증한다.
4. 탐색 함수에는 development만 전달한다. 모델별로 유리한 split을 다시 생성하지 않는다.

| 데이터 | 전체 | 개발 정상 / 위험 | Test 정상 / 위험 | Fold train 정상 / 위험 |
|---|---:|---:|---:|---|
| CN7 | 606 | 473 / 11 | 119 / 3 | fold 0: 354 / 9, 나머지: 355 / 8 |
| RG3 | 591 | 452 / 20 | 114 / 5 | 모든 fold: 339 / 15 |

`pattern_row`는 전체 `X_labeled.csv`의 0부터 시작하는 위치다. 축소된 development 테이블의 위치와 혼동하지 않는다. 라벨·관측 횟수·pattern_type·ID·partition·fold는 모델 입력이 아니다. 비라벨 자료는 학습·검증에서 제외한다.

## 3. 입력 구성과 변환

### 3.1. 필수 기준 모델

`raw_no_scaler`를 기준 구성으로 사용한다.

```text
제공 원변수 24개
  → train 전체로 상수 열 제거 적합
  → train 정상·위험 전체로 RandomForestClassifier 적합
```

원변수의 순서에 따라 분할하는 트리 모델이므로 로지스틱 회귀처럼 규제 척도를 맞추기 위한 StandardScaler를 기본 추가하지 않는다. 제공값을 물리 단위로 복원한다는 의미는 아니다. 상수 제거는 train에서만 정하고 validation/test는 같은 열을 사용한다. 입력은 유한값 검사를 통과해야 하며, 결측을 발견하면 새 대체 정책을 조용히 추가하지 말고 데이터 버전을 확인한다.

### 3.2. 도메인 가설 비교

공통 매뉴얼의 형체·주기 제외, 압력, 가소화, 열 상태, 도메인 전체, 공정별 PCA를 선택적으로 비교한다. 실행할 구성 목록은 test 평가 전에 고정한다.

- 정상 기준 z로 정의된 파생변수는 해당 train 정상으로 기준 통계를 적합하고 train 정상·위험 모두에 적용한다.
- 기존 정상 기준 공정별 PCA를 재사용하면 PCA 역시 해당 train 정상으로 적합한다. 이 변환은 선택한 입력 구성의 일부이지 모든 랜덤 포레스트의 필수 단계가 아니다.
- 변환 후 상수 제거는 train 전체로 적합한다. 분류기는 항상 변환된 train 전체와 라벨로 학습한다.
- 최종 재학습도 같은 적합 범위를 유지한다. 전체 데이터로 미리 z·PCA·특징 선택을 적합하지 않는다.

트리는 변수 간 비선형 관계를 학습할 수 있지만, 도메인 차이·집계 변수가 더 간단한 분할을 제공할 수 있는지는 validation으로 확인해야 한다. 파생변수 추가는 변수 무작위 선택의 구성도 바꾸므로 반드시 개선된다고 가정하지 않는다. PCA는 축 방향도 바꾸므로 성능 개선을 전제하지 않는다.

OCSVM·로지스틱 회귀와 입력 및 변환이 다르면 전체 파이프라인 비교로 보고한다. 모델 구조만 비교하려면 동일 표현을 사용하는 별도 결과가 필요하다.

## 4. 초기 탐색 설계

탐색 크기를 제한하기 위해 트리 수는 첫 실험에서 **300개로 고정**하고 깊이·리프 크기·분할 후보 변수 수·가중치를 비교한다. 300은 계산 예산을 정한 시작값이며 충분성이나 최적성이 검증된 수치가 아니다. 트리 수의 추가 비교는 아래 후속 실험 규칙을 따른다.

| 항목 | 초기 설정 | 이유와 취급 |
|---|---|---|
| n_estimators | 300 고정 | 첫 탐색의 계산량 제한 |
| max_depth | `[3, 6, None]` | 얕은 구조부터 깊이 제한 없는 구조 비교 |
| min_samples_leaf | `[1, 3, 5, 10]` | 작은 리프와 완화된 분할 비교. 큰 리프가 희소 위험 패턴을 놓칠 가능성도 함께 평가 |
| max_features | `["sqrt", 0.5, 1.0]` | 각 분할에서 고려하는 변수 수 비교. 1.0은 전체 변수 |
| class_weight | `[None, "balanced", "balanced_subsample"]` | 무가중·train 빈도 보정·트리별 bootstrap 빈도 보정 비교 |
| t | `np.linspace(0, 1, 1001)` | 0.001 간격, 0.5 포함. 최적 해상도라는 보장은 없음 |
| criterion | `"gini"` | 첫 비교의 고정 학습 기준 |
| bootstrap / max_samples | `True` / `None` | 각 트리에 train 행 수만큼 복원 추출 |
| min_samples_split | 2 | 리프 크기를 주 구조 제어 대상으로 두고 고정 |
| ccp_alpha | 0.0 | 첫 실험에서 추가 가지치기 탐색 제외 |
| oob_score / warm_start | `False` / `False` | 고정 4-fold 평가 사용, 후보 간 학습 상태 재사용 방지 |
| random_state / n_jobs | 42 / 1 | Seed 고정 및 예제 실행 자원 제한 |

`balanced`는 해당 fit의 클래스 빈도를, `balanced_subsample`은 각 bootstrap 표본의 빈도를 사용한다. 이는 균형 있는 표본 수를 뽑는다는 뜻이 아니다. `predict_proba`는 각 트리의 클래스 확률을 평균한다. [RandomForestClassifier 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

입력 구성 1개당 3 × 4 × 3 × 3 = **108개 모델 설정**, 432회 forest 학습, 108 × 1001 = **108,108개 설정·t 평가**다. 생성 트리는 129,600개이며 최종 재학습은 별도다. 구성 S개면 각각 S배다. 임계값마다 forest를 재학습하지 않는다.

후속 실험에서 트리 수를 `[100, 300, 600]`으로 비교하려면 탐색 전에 선언한다. 전체 구조 후보와 공동 비교하면 324개 모델 설정이며, 일부 상위 구조만 재평가하면 단계적 탐색으로 명시한다. 트리 수를 바꾸고 기존 t와 성능을 그대로 재사용하지 않는다. 새로운 OOF 점수를 생성해 t도 다시 선택한다. Test를 보고 이 실험을 설계하지 않는다.

0.001 간격은 근접 확률 사이의 최적 판정 구간을 놓칠 수 있다. 간격을 줄이는 후속 실험도 development에서만 수행하고 변경 이력을 남긴다. 최적값은 선언한 탐색 범위 내 최적값이다. Seed를 바꿔 가장 잘 나온 값만 선택하지 않는다. 추가 seed의 안정성을 평가하면 목록·집계 정책을 미리 정하고 분포를 함께 보고한다.

## 5. 기준 입력의 구현 예제

현재 확인한 scikit-learn 1.9.1 기준이다. 아래 코드는 후보 생성과 한 설정의 4-fold OOF 평가를 구현한다. 전체 탐색·저장·도메인 변환 구현은 별도 셀에 추가한다.

```python
from itertools import product
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

THRESHOLDS = np.linspace(0.0, 1.0, 1001)
CANDIDATES = [
    dict(max_depth=d, min_samples_leaf=leaf,
         max_features=features, class_weight=weight)
    for d, leaf, features, weight in product(
        [3, 6, None], [1, 3, 5, 10],
        ["sqrt", 0.5, 1.0], [None, "balanced", "balanced_subsample"]
    )
]

def make_model(params):
    return Pipeline([
        ("constant", VarianceThreshold(threshold=0.0)),
        ("rf", RandomForestClassifier(
            n_estimators=300, criterion="gini",
            bootstrap=True, max_samples=None, min_samples_split=2,
            ccp_alpha=0.0, oob_score=False, warm_start=False,
            random_state=42, n_jobs=1, **params,
        )),
    ])

def probability_of_risk(model, X):
    classes = model.named_steps["rf"].classes_
    col = np.flatnonzero(classes == 1)
    assert len(col) == 1
    p = model.predict_proba(X)[:, col[0]]
    assert p.shape == (len(X),) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    return p

def oof_one_setting(X_dev, y_dev, folds, params):
    y, folds = np.asarray(y_dev), np.asarray(folds)
    assert len(X_dev) == len(y) == len(folds)
    assert set(folds) == {0, 1, 2, 3} and set(y) == {0, 1}
    assert np.isfinite(X_dev.to_numpy()).all()
    p, seen = np.full(len(y), np.nan), np.zeros(len(y), dtype=int)
    audit = []
    for k in range(4):
        tr = np.flatnonzero(folds != k)
        va = np.flatnonzero(folds == k)
        assert set(y[tr]) == {0, 1} and set(y[va]) == {0, 1}
        model = make_model(params)
        model.fit(X_dev.iloc[tr], y[tr])
        p[va] = probability_of_risk(model, X_dev.iloc[va])
        seen[va] += 1
        forest = model.named_steps["rf"]
        audit.append(dict(
            fold=k, n_train=len(tr), n_train_positive=int(y[tr].sum()),
            n_features=int(model.named_steps["constant"].get_support().sum()),
            n_trees=len(forest.estimators_),
            max_fitted_depth=max(tree.get_depth() for tree in forest.estimators_),
        ))
    assert (seen == 1).all() and np.isfinite(p).all()
    return p, audit

def binary_metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    tp = int(((y == 1) & (prediction == 1)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    tn = int(((y == 0) & (prediction == 0)).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
```

공통 loader로 읽은 `X.iloc[dev]`, `y.iloc[dev]`, `assignment.iloc[dev]["cv_fold"]`를 같은 순서로 전달한다. `enumerate(CANDIDATES)` 순서를 후보 ID로 저장한다. 입력 구성이 여러 개면 구성 순서도 사전 고정한다. 모든 t에 대해 `(p > t)`로 예측하고 전체·fold별 지표를 저장한다.

## 6. 선택·실패·계산 관리

```text
OOF F1 = 2ΣTP / (2ΣTP + ΣFP + ΣFN)
정렬: OOF F1 내림차순
   → fold F1 표본 표준편차(ddof=1) 오름차순
   → fold 간 최대 입력 수 오름차순
   → 사전 고정한 모델 후보 ID 오름차순
   → abs(t) 오름차순 → t 내림차순
```

같은 후보의 네 fold에 같은 숫자 t를 적용한다. Fold마다 별도 최적 t를 선택해 성능을 합치거나 t를 평균하지 않는다. 입력 수는 분류기에 전달한 열 수이며 중요도가 0보다 큰 변수 수가 아니다. F1의 0 분모는 0으로 처리한다. 정밀도·재현율도 정의되지 않는 분모는 0으로 통일한다.

AP·FPR·accuracy·OOB 점수를 추가 선택 조건으로 사용하지 않는다. 특히 `oob_score_`를 고정 validation F1 대신 사용하지 않는다. OOB는 bootstrap 과정에서 생기는 별도 진단이며 선언한 4-fold 검증을 대체하지 않는다. `GridSearchCV(scoring="f1")`의 기본 fold 평균·기본 판정만으로 본 선택 규칙이 구현되지는 않는다.

실패한 fold만 제외해 후보를 평가하지 않는다. 입력 오류·메모리 부족·학습 실패를 후보 단위로 기록하고 실행을 중단하거나 후보 전체를 실패 처리한다. 랜덤 포레스트에는 로지스틱 회귀의 `max_iter` 수렴 검사를 복사하지 않는다. 실제 트리 수·깊이·유효 점수·학습 시간·메모리 사용을 확인한다.

병렬화를 켤 때는 후보와 forest 양쪽을 동시에 무제한 병렬 실행하지 않는다. 실행 환경과 병렬 수준을 기록한다. `warm_start`로 다른 후보나 fold에서 학습한 트리를 이어 쓰지 않는다. Bootstrap은 해당 train 내부 복원 추출이며 별도 65:35 보정 분할이 아니다.

## 7. 최종 재학습과 평가

1. 전체 탐색 결과·상위 20개·선택 manifest를 test 점수 계산 전에 저장한다.
2. 선택 설정으로 CN7 개발 484개, RG3 개발 472개 전체를 새로 학습한다. 상수 제거와 선택한 도메인 변환도 정해진 범위에서 다시 적합한다.
3. 클래스 가중치는 최종 fit의 개발 라벨 및 선택한 bootstrap 정책으로 계산한다. Fold 가중치를 평균하지 않는다.
4. 선택한 숫자 t를 그대로 test 확률에 적용한다. 재학습 후 점수 분포가 달라져도 test로 t를 조정하지 않는다.
5. TP/FP/FN/TN, F1, precision, recall, FPR과 참고 지표를 저장한다.
6. 열 순서·변환·forest·t·라벨 의미·환경·선택 manifest 식별자를 함께 저장한다. 재로딩 후 동일 점수와 판정을 검증한다.

모델 종류 간 선택도 development F1로 확정한다. 현재 test는 기존 OCSVM 분석에서 관찰했으므로 완전히 미사용인 독립 test로 표현하지 않는다. 선택에 사용한 OOF F1도 튜닝 성적이며 독립 성능 추정으로 과장하지 않는다. 클래스 가중치를 포함한 forest 확률을 실제 제품 불량률로 보정된 값이라고 해석하지 않는다.

## 8. 변수 중요도와 결과 해석

`feature_importances_`는 학습 중 불순도 감소에 따른 중요도다. 고유값이 많은 변수에 유리할 수 있고 검증 성능 기여나 원인 효과를 뜻하지 않는다. 단독으로 변수를 자동 삭제하지 않는다. Validation에서 고정 t의 F1 감소를 보는 permutation 진단을 함께 사용한다. [공식 permutation importance 안내](https://scikit-learn.org/stable/modules/permutation_importance.html)

공정군 진단은 선택한 후보의 fold 모델과 해당 validation 자료로 수행한다.

1. 선택 모델과 t를 고정하고 원래 OOF F1을 계산한다.
2. 각 validation fold 안에서 공정군 원변수들을 같은 행 순열로 함께 섞는다. 반복 수 20, seed 42를 초기 진단 설정으로 기록한다.
3. 파생변수를 사용하는 경우 섞인 원변수에서 다시 계산하고, 적합 완료된 변환만 적용한다. 모델이나 PCA를 다시 학습하지 않는다.
4. 각 반복의 네 fold 예측을 합쳐 OOF F1을 계산한다. 원래 F1에서 뺀 감소량의 평균·표준편차, 재현율 감소·FPR 증가를 보고한다.
5. 음의 중요도도 그대로 남긴다. 진단 결과를 보고 변수를 바꾸면 별도 후보 탐색으로 기록하고 test로 재선택하지 않는다.

기본 permutation 함수에 `scoring="f1"`만 전달하면 기본 `predict()`의 경계를 사용할 수 있다. 선택 t를 사용하는 scorer를 명시하거나 직접 예측을 집계한다. 상관된 변수는 서로 대체할 수 있고 순열은 비현실적 조합을 만들 수 있으므로 중요도가 낮아도 공정상 무의미하다는 결론을 내리지 않는다. [상관 변수에 대한 공식 예제](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html)

## 9. 보고서와 산출물

구현 경로는 `modeling/random_forest_cn7_integrated.ipynb`, `modeling/random_forest_rg3_integrated.ipynb`, 출력은 `output/random_forest_<dataset>/<experiment_id>/`를 권장한다. 이 경로는 향후 구현 명명 규칙이다. 프로젝트 함수·검증 코드는 노트북 셀에 포함한다.

| 셀 구간 | 내용 |
|---|---|
| A | 목적·환경·원본·전처리·분할 검증 |
| B | 입력 가설·변환·모델 후보·임계값 목록 |
| C | 학습·확률 추출·혼동행렬·후보 순위 함수 |
| D | 4-fold 공동 탐색·상위 후보·선택 확정 |
| E | 개발 전체 재학습·저장·test 후속 평가 |
| F | 공정군 중요도·상수 판정 기준선·모델 비교 |
| G | 내부 검증·저장물 독립 재집계 |

공통 매뉴얼 9절의 산출물에 `forest_audit.csv`, `impurity_importance.csv`, `group_permutation_importance.csv`를 추가한다. 후보별 실제 트리 수·깊이·학습 시간·실패 이력을 남긴다. OOF 예측에는 `pattern_row`, `fold`, `label`, `probability`, `threshold`, `prediction`을 포함한다.

| 데이터 | 입력 구성 | 트리 수/깊이/리프/변수 수/가중치 | t | OOF F1 | Fold F1 표준편차 | OOF TP/FP/FN | Test F1 | Test TP/FP/FN/TN |
|---|---|---|---:|---:|---:|---|---:|---|
| CN7 | 실행 후 작성 | — | — | — | — | — | — | — |
| RG3 | 실행 후 작성 | — | — | — | — | — | — | — |

전부 정상·전부 위험 판정의 F1도 함께 계산한다. OCSVM 기준값은 공통 매뉴얼 10절, 로지스틱 회귀 절차는 [전용 매뉴얼](LOGISTIC_REGRESSION_MODELING_MANUAL.md)을 참조한다. 비교 모델이 아직 실행되지 않았다면 결과 칸을 채우지 않는다. CN7 test 위험 3개·RG3 5개이므로 소수 판정 차이만으로 우위를 단정하지 않는다.

## 10. 완료 확인표

- [ ] 원본·전처리·split 코드로 재현하고 기준 배정과 대조했다.
- [ ] Fold 학습에 정상·위험 모두 포함했고 메타데이터를 입력에서 제외했다.
- [ ] 상수 제거·도메인 변환·가중치 계산의 적합 범위를 확인했다.
- [ ] 원변수 기준 모델은 scaler 없이 평가하고 추가 표현을 구분했다.
- [ ] 확률의 클래스 순서·유한값·범위·엄격한 `>` 판정을 확인했다.
- [ ] 개발 행마다 OOF에 한 번 등장하고 후보별 같은 t를 적용했다.
- [ ] 합산 F1·동률 규칙·후보 수·상위 후보를 독립 재계산했다.
- [ ] OOB·AP·FPR·test를 숨은 선택 기준으로 쓰지 않았다.
- [ ] 후보·fold 사이에 학습된 트리를 재사용하지 않았다.
- [ ] 최종 개발 전체 재학습과 저장 모델 재로딩을 검증했다.
- [ ] 중요도는 선택 t와 validation에서 진단하고 인과 효과로 해석하지 않았다.
- [ ] 새 커널에서 셀 전체가 실행되며 반복 관찰한 test의 한계를 기록했다.

범위·seed·입력·임계값 간격을 바꾸면 실험 ID와 후보 목록을 갱신한다. 이 문서의 초기 설정을 검증 없이 최적 파라미터로 전달하지 않는다.
