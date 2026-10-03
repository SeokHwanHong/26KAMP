# CN7·RG3 로지스틱 회귀 모델링·학습·검증 매뉴얼

2026-10-03 구현·실행 완료: [실행 노트북](05_logistic_regression.ipynb), [학습 소스](logistic_regression.py). 8개 입력 구성 × 30개 설정 × 1001개 임계값을 비교했고 결과는 `output/logistic/<dataset>/<run_id>/`에 보존합니다. 아래 초기 명명 권고 대신 현재 경로를 사용합니다. 공통 코드는 `../common/pipeline_runtime.py`로 재사용하며 전달 시 함께 포함합니다. 실제 실행 환경은 scikit-learn 1.9.0입니다. 현재 결과: CN7 OOF F1 0.625 / Test 0, RG3 OOF F1 0.089286 / Test 0.083333. 운영 승격 근거가 아닌 후속 탐색 결과입니다.

모듈화 경로 갱신: 2026-10-02. CN7/RG3 전처리·OCSVM은 각각 공통 노트북의 `DATASET`으로 선택합니다. 이 문서의 과거 산출물 경로는 당시 기록이며 새 실행 결과는 `output/modular/`에 저장됩니다.

작성일: 2026-09-29 · 버전: 1.0 · 상위 기준: [공통 매뉴얼 v1.1](../common/CN7_RG3_CI_CT_CD_전체구조.md)

이 문서는 CN7과 RG3에 로지스틱 회귀를 독립 적용하기 위한 구현 지침이다. 원본 → 전처리 → 분할 → 개발 탐색 → 최종 재학습 → test 후속 평가 순서를 따른다. 아래 탐색 범위는 **최적값이 아니라 재현 가능한 초기 실험 설계**다. 이 문서에는 새 모델의 성능 결과를 주장하지 않는다.

## 1. 목표와 OCSVM에서 바뀌는 부분

목표는 입력 패턴에 불량 이력이 있는지(`PassOrFail=1`) 분류하는 것이다. 원본 중복 패턴의 라벨을 최댓값으로 집계했으므로 개별 생산품의 불량 확률 추정과는 다르다.

| 항목 | 로지스틱 회귀 적용 기준 |
|---|---|
| 학습 유형 | 지도학습 이진 분류, `training_mode=supervised` |
| Fold 학습 자료 | 해당 train의 정상·위험 전체. `y == 0` 필터 금지 |
| 모델 입력 | 공정 변수와 사전에 정의한 도메인 파생변수. 라벨 요약·ID·fold 정보 제외 |
| 위험 점수 | `predict_proba`에서 라벨 1에 해당하는 열 |
| 판정 | `p > t`이면 1. 기본 `predict()`로 선택한 t를 대체하지 않음 |
| 공동 탐색 | 입력 구성, 규제 강도 C, 클래스 가중치, 숫자 임계값 t |
| 선택 | 네 validation fold의 혼동행렬을 합산한 OOF F1 최대화 |
| 최종 학습 | CN7 개발 484개, RG3 개발 472개 전체 |

학습은 규제된 로지스틱 손실을 최소화한다. **학습 손실과 모델 선택 지표는 다르다.** 계수는 train에서 손실을 줄이도록 학습하고, 그 계수로 얻은 validation 점수에 여러 t를 적용하여 F1이 가장 높은 설정을 고른다. FPR 상한이나 AP를 선택 조건으로 넣지 않는다.

클래스 가중치는 학습 시 각 클래스 오류가 손실에 미치는 비중을 바꾸고, 임계값은 학습된 점수를 이진 판정으로 바꾼다. 두 설정은 역할이 다르므로 함께 비교한다. 가중치를 쓰면 반드시 좋아진다고 가정하지 않는다.

## 2. 전달물과 데이터 준비

전달 패키지는 공통 매뉴얼 2절을 따른다. 필수 원본은 `data/origin/moldset_{labeled|unlabeled}_{cn7|rg3}.csv` 4개다. 현재 전처리 코드가 읽는 `data/schema/input_features.json`도 포함한다.

1. [CN7 전처리](../../02.preprocessing/02_preprocess.ipynb), [RG3 전처리](../../02.preprocessing/02_preprocess.ipynb)를 각각 전체 실행한다.
2. [공통 split 코드](../../02.preprocessing/03_split.ipynb)를 실행한다. Seed 42, 계층화 test 20%, 개발 계층화 4-fold와 개발 행 정렬 규칙을 유지한다.
3. 생성한 배정표를 보존한 기준 배정과 대조한다. 모델별로 분할을 바꾸지 않는다.
4. 공통 매뉴얼 3절의 `load_modeling_data`를 노트북 셀에 복사해 입력·해시·정렬을 검증한다. 탐색 함수에는 development만 전달한다.

| 데이터 | 전체 | 개발 정상 / 위험 | Test 정상 / 위험 | 각 fold train 정상 / 위험 |
|---|---:|---:|---:|---|
| CN7 | 606 | 473 / 11 | 119 / 3 | fold 0: 354 / 9, 나머지: 355 / 8 |
| RG3 | 591 | 452 / 20 | 114 / 5 | 모든 fold: 339 / 15 |

`pattern_row`는 전체 `X_labeled.csv`의 위치다. 축소된 development 테이블의 위치 인덱스로 사용하지 않는다. 비라벨 자료는 학습·검증에 추가하지 않는다. 제공 원본에도 사전 표준화가 있으므로 물리 단위가 복원되었다고 해석하지 않는다.

## 3. 입력 구성과 변환

### 3.1. 필수 기준 실험

`raw_alltrain_scaled`를 기준 입력으로 사용한다.

```text
원변수 24개
  → 해당 train 전체에서 상수 열 제거
  → 해당 train 전체에서 StandardScaler 적합
  → 해당 train 전체 X, y로 LogisticRegression 적합
```

Validation/test에는 적합된 열 선택과 scaler를 그대로 적용한다. Train에 존재하는 이상치를 validation 분포에 맞춰 제거하지 않는다. 현재 입력은 유한값 검증을 통과해야 하며, 결측을 발견하면 자동 대체 정책을 새로 추가하지 말고 입력 버전을 먼저 확인한다.

스케일링은 변수마다 규제가 적용되는 척도를 정리하기 위한 모델 내부 변환이다. Train 평균·표준편차를 저장해 이후 입력에 재사용한다. [StandardScaler 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

### 3.2. 도메인 가설 비교 실험

공통 매뉴얼의 원변수, 형체·주기 제외, 압력, 가소화, 열 상태, 도메인 전체, 공정별 PCA를 선택적으로 비교한다. 어떤 구성까지 실행할지 test를 보기 전에 선언한다.

- 정상 대비 편차를 정의하는 도메인 z와 공정별 PCA는 기존 정상 기준 변환을 재사용할 경우 해당 train의 정상만으로 적합한다.
- 그 변환을 **train의 정상·위험 모두**에 적용한다. 최종 상수 제거·StandardScaler는 변환된 train 전체로 적합하고 분류기를 학습한다.
- 기존 `fit_features`가 수행하는 변환과 추가 scaler의 순서를 manifest에 기록한다. `raw_alltrain_scaled`와 기존 정상 기준 입력을 같은 표현이라고 기록하지 않는다.
- OCSVM과 모델 자체만 비교하려면 동일한 변환 표현을 사용하는 별도 비교를 제시한다. 입력·스케일링까지 다르면 전체 파이프라인 비교다.

선형 차이·평균 파생변수는 원변수의 선형 결합이다. 원변수를 모두 유지한 로지스틱 회귀에서 새로운 비선형 경계를 추가하지 않지만, 중복 표현이 규제 효과와 계수 분배를 바꿀 수 있다. 배럴 표준편차나 PCA 재구성 오차는 비선형 정보를 추가할 수 있다. 따라서 파생변수 수가 많다는 이유만으로 개선을 기대하거나 계수 크기를 독립적인 원인 효과로 해석하지 않는다.

## 4. 초기 탐색 범위와 설정 근거

첫 실험은 **L2 규제**로 고정한다. 상관된 공정 변수의 계수를 완화하는 기준 모델을 만들고, 적은 위험 표본에서 불필요한 탐색 확대를 피하기 위한 설계다. L1·Elastic Net은 필요하면 별도 확장 실험으로 선언한다.

| 항목 | 초기 설정 | 선택 근거 / 취급 |
|---|---|---|
| 규제 | L2 | 기본 비교에서 고정 |
| C | `np.logspace(-4, 3, 15)` | 여러 규모의 규제 강도를 비교하는 반 로그 간격. 이 범위의 통계적 최적성을 가정하지 않음 |
| class_weight | `None`, `"balanced"` | 원래 손실과 클래스 빈도 보정 손실 비교 |
| t | `np.linspace(0, 1, 1001)` | 확률 범위에서 0.001 간격의 고정 탐색. 0.5 포함 |
| solver | `"lbfgs"` | L2 실험에서 고정 |
| fit_intercept | `True` | 절편 포함 |
| max_iter / tol | 5000 / 1e-6 | 수렴 관리용 고정값. 성능 최적화 대상으로 취급하지 않음 |
| random_state | 42 | 설정 기록용. Seed만으로 환경 간 완전 동일성을 보장하지 않음 |
| 재표본화 / 별도 확률 보정 | 사용하지 않음 | 첫 비교의 변경 요인 제한 |

`C`가 작을수록 규제가 강하다. `balanced` 가중치는 각 fit의 y에서 `n / (2 × 클래스별 개수)`로 계산된다. Scikit-learn 1.9.1에서는 L2를 `l1_ratio=0.0`으로 지정하며 `penalty`는 deprecated 상태다. 예제는 현재 확인한 1.9.1을 기준으로 한다. 다른 버전에서는 API를 확인하고 환경을 기록한다. [LogisticRegression 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

입력 구성 1개당 30개 모델 설정 × 4 folds = 120회 학습, 30 × 1001 = 30,030개 설정·t 평가다. 최종 재학습은 별도다. 구성 수가 S이면 각각 S배다. 임계값마다 모델을 다시 학습하지 않는다.

0.001 간격은 계산량과 해상도를 정한 초기 선택이다. 근접 점수가 같은 구간에 몰리면 최상의 F1 구간을 놓칠 수 있다. 필요하면 development에서만 간격을 세분화하는 후속 실험을 선언하고 전체 탐색 이력을 보존한다. C의 경계값이 선택되어도 test를 보고 범위를 확장하지 않는다. 각 실험의 최적값은 **그 실험의 탐색 범위 내 최적값**이다.

## 5. 기준 입력의 구현 예제

아래 코드는 `raw_alltrain_scaled`의 모델 생성과 **한 설정의 4-fold 점수 생성** 예제다. 완성된 전체 탐색·저장 프로그램은 아니다. 개발 행만 받은 함수로 작성하여 test가 학습에 들어갈 경로를 줄인다.

```python
import warnings
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

C_GRID = np.logspace(-4, 3, 15)
WEIGHT_GRID = [None, "balanced"]
THRESHOLDS = np.linspace(0.0, 1.0, 1001)

def make_model(C, class_weight):
    return Pipeline([
        ("constant", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            C=float(C), l1_ratio=0.0, solver="lbfgs",
            class_weight=class_weight, fit_intercept=True,
            max_iter=5000, tol=1e-6, random_state=42,
        )),
    ])

def probability_of_risk(model, X):
    classes = model.named_steps["lr"].classes_
    col = np.flatnonzero(classes == 1)
    assert len(col) == 1
    p = model.predict_proba(X)[:, col[0]]
    assert p.shape == (len(X),) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    return p

def oof_one_setting(X_dev, y_dev, folds, C, class_weight):
    # X_dev, y_dev, folds는 같은 순서. folds는 개발 행의 cv_fold만 포함.
    y = np.asarray(y_dev)
    folds = np.asarray(folds)
    assert len(X_dev) == len(y) == len(folds)
    assert set(folds) == {0, 1, 2, 3} and set(y) == {0, 1}
    p = np.full(len(y), np.nan)
    seen = np.zeros(len(y), dtype=int)
    audit = []
    for k in range(4):
        tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        assert set(y[tr]) == {0, 1} and set(y[va]) == {0, 1}
        model = make_model(C, class_weight)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X_dev.iloc[tr], y[tr])
        p[va] = probability_of_risk(model, X_dev.iloc[va])
        seen[va] += 1
        audit.append({
            "fold": k,
            "n_train": len(tr), "n_train_positive": int(y[tr].sum()),
            "n_features": int(model.named_steps["constant"].get_support().sum()),
            "n_iter": model.named_steps["lr"].n_iter_.tolist(),
        })
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

노트북에서는 다음 순서로 연결한다.

```text
공통 loader로 X, y, assignment, dev, test 읽기
X_dev = X.iloc[dev], y_dev = y.iloc[dev]
folds = assignment.iloc[dev]["cv_fold"]
입력 구성 → C 오름차순 → class_weight(None, balanced) 순서로 후보 ID 고정
각 설정에서 oof_one_setting에 해당하는 처리 실행
각 t에 대해 pred = (p > t) 계산
전체 OOF의 binary_metrics와 fold별 binary_metrics 저장
```

## 6. 후보 선택과 수렴 처리

### 6.1. 선택 규칙

같은 후보의 네 fold에 같은 숫자 t를 적용한다. Fold마다 최적 t를 따로 고른 뒤 성능을 합치지 않는다.

```text
pooled OOF F1 = 2ΣTP / (2ΣTP + ΣFP + ΣFN)
정렬: pooled OOF F1 내림차순
   → fold F1 표본 표준편차(ddof=1) 오름차순
   → fold 간 최대 입력 수 오름차순
   → 사전 고정한 모델 후보 ID 오름차순
   → abs(t) 오름차순 → t 내림차순
```

F1의 분모가 0이면 0으로 처리한다. 입력 수는 실제 분류기에 전달한 열 수이며 비영 계수 수가 아니다. AP·FPR·accuracy를 숨은 정렬 조건으로 추가하지 않는다. 평균 fold F1과 합산 F1을 구분한다. 후보 ID는 입력 구성·C·class_weight를 식별하고 t는 별도 열로 저장한다.

`GridSearchCV(scoring="f1")` 또는 `LogisticRegressionCV`의 기본 사용만으로 이 공동 탐색·합산 F1 선택이 구현되지는 않는다. OOF 예측을 명시적으로 수집한다. 선택에 사용한 OOF F1은 튜닝 성적이며 독립적인 일반화 성능 추정치로 표현하지 않는다.

### 6.2. 수렴과 실패 후보

예제 코드는 수렴 경고를 오류로 처리해 중단한다. 실패 후보·fold·설정을 기록하고 실패한 fold를 제외한 성능으로 후보를 평가하지 않는다.

반복 수를 늘리면 모든 후보에 동일한 재시도 정책을 적용하고 계산 설정을 기록한다. 수렴하지 않은 모델을 검증 완료 모델로 전달하지 않는다. LBFGS의 수치 최적화에는 validation 감시용 추가 분할을 기본 도입하지 않는다. 정상 train의 65:35 분할도 사용하지 않는다.

## 7. 선택 확정·최종 재학습·test

1. 전체 후보와 상위 20개를 저장하고 입력 구성·C·class_weight·t·변환 정책·버전을 test 평가 전에 확정한다.
2. CN7은 개발 484행, RG3는 개발 472행 전체로 최종 파이프라인을 재학습한다. 일반 scaler는 개발 전체, 정상 기준 도메인 변환을 쓰면 그 부분만 개발 정상으로 적합한다.
3. `balanced`는 최종 fit에서도 개발 y로 계산한다. Fold별 가중치를 평균해 재사용하지 않는다.
4. Test를 transform하고 라벨 1 확률에 선택한 동일 숫자 t를 적용한다. Test로 t나 계수를 조정하지 않는다.
5. F1·precision·recall·FPR·TP/FP/FN/TN을 저장한다. AP·ROC-AUC는 참고 지표로 추가할 수 있다.
6. 입력 열 순서·파이프라인·t·라벨 정의·환경·manifest 식별자를 bundle에 저장하고 재로딩 후 점수·판정 일치를 확인한다.

개발에서 선택한 t가 재학습 후에도 최적이라는 보장은 없지만 test로 재조정하지 않는다. 현재 test는 OCSVM 분석에서 관찰했으므로 미사용 독립 test로 표현하지 않는다. 모델 종류 간 선택도 개발 F1로 확정하고 test는 후속 평가로 보고한다.

## 8. 결과 해석과 비교

계수표에는 변환 후 입력 이름·계수·절편·스케일링 통계를 저장한다. 개별 표준화 원변수의 계수는 그 입력을 train 표준편차 한 단위 바꿀 때 모델상 log-odds 변화에 대응한다. 다만 상관 변수·파생변수가 있으면 다른 열을 고정하는 변화가 물리적으로 가능하지 않을 수 있다. 규제된 계수를 원인 효과나 p값과 혼동하지 않는다.

`predict_proba`는 해당 라벨·학습 조건에 대한 모델 출력이다. 특히 클래스 가중치를 사용한 출력이 실제 제품 불량률로 보정되었다고 가정하지 않는다. 초기 실험에서는 확률 보정을 추가하지 않고 판정 성능을 보고한다.

| 데이터 | 입력 구성 | C | class_weight | t | OOF F1 | Fold F1 표준편차 | OOF TP/FP/FN | Test F1 | Test TP/FP/FN/TN |
|---|---|---:|---|---:|---:|---:|---|---:|---|
| CN7 | 실행 후 작성 | — | — | — | — | — | — | — | — |
| RG3 | 실행 후 작성 | — | — | — | — | — | — | — | — |

전부 정상·전부 위험 판정의 F1도 같은 라벨 분포에서 계산한다. OCSVM 비교값은 공통 매뉴얼 10절을 참조하고 분할·입력 표현·탐색 수의 차이를 기록한다. CN7 개발 위험 11개·test 3개, RG3 개발 위험 20개·test 5개이므로 소수 판정 차이를 큰 우위로 단정하지 않는다.

## 9. 노트북 구성과 전달 확인표

구현 경로는 `modeling/logistic_cn7_integrated.ipynb`, `modeling/logistic_rg3_integrated.ipynb`, 출력은 `output/logistic_<dataset>/<experiment_id>/`를 권장한다. 이는 향후 구현 명명 규칙이며 이 매뉴얼 작성으로 모델 구현까지 완료했다는 뜻은 아니다.

| 셀 구간 | 내용 |
|---|---|
| A | 목적·환경·원본/전처리/분할 검증 |
| B | 입력 구성·변환 함수·후보 목록 |
| C | 학습·확률 추출·혼동행렬·후보 정렬 함수 |
| D | 4-fold 공동 탐색, 상위 후보, 선택 manifest |
| E | 개발 전체 재학습, 저장, test 후속 평가 |
| F | 계수 해석·상수 판정 기준선·OCSVM 비교 |
| G | 내부 검증·저장 결과의 독립 재집계 |

필요한 프로젝트 코드는 노트북 셀에 포함한다. 공통 매뉴얼 9절 산출물에 더해 `coefficients.csv`와 수렴 기록을 저장한다. OOF 예측에는 `pattern_row`, `fold`, `label`, `probability`, `threshold`, `prediction`을 남긴다.

- [ ] 원본과 전처리·split 코드로 재현하고 기준 분할과 대조했다.
- [ ] 정상·위험 모두 학습에 포함되고 fold별 개수가 표와 일치한다.
- [ ] 상수 제거·scaler·파생변수·가중치 계산이 올바른 train 범위에서 수행됐다.
- [ ] 라벨 1의 확률 열, 유한값, 0~1 범위, 엄격한 `>` 판정을 확인했다.
- [ ] 개발 행마다 OOF에 한 번 포함되고 같은 후보의 t가 모든 fold에서 같다.
- [ ] 합산 F1을 예측에서 재계산하고 순위·동률 처리·상위 후보가 재현됐다.
- [ ] 수렴 경고·실패 후보를 기록하고 실패 fold만 제외하지 않았다.
- [ ] 최종 학습 행 수·변환 범위·t를 검증하고 저장 모델이 재현됐다.
- [ ] 새 커널에서 모든 셀이 순서대로 실행되며 test를 선택에 쓰지 않았다.
- [ ] 보정된 확률·인과 효과·독립 test라는 근거 없는 해석을 피했다.

설계값을 변경하면 실험 ID와 후보 목록을 갱신한다. 성능을 확인하지 않은 C 범위·가중치·t를 CN7/RG3의 최적값으로 전달하지 않는다.
