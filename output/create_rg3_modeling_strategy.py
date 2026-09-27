from pathlib import Path
import ast, zipfile, json, hashlib
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/rg3_modeling_strategy'; OUT.mkdir(exist_ok=True)
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
parts=[]
# Reuse only document layout helpers; never execute the CN7 report generator.
tree=ast.parse((ROOT/'output/create_cn7_modeling_strategy.py').read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'layout_helpers','exec'))

title(1,'RG3 모델링 설계 및 학습 전략')
p('랜덤포레스트 · One-Class SVM · 로지스틱 회귀','Subtitle')
p('기준: EDA/rg3eda.ipynb 및 전처리·군집화 산출물 | 2026-09-27','Small')
p('세 모델의 선정 근거, 학습과 검증, 손실함수, 실패 분석을 정리한 실행 전 설계안이다. 아래 파라미터는 초기 후보이며 성능이 검증된 최적값이 아니다. EDA 노트북은 수정하거나 모델링 코드를 추가하지 않는다.')
h('EDA에서 확인된 사실과 설계 결정 [1]')
table(['관측 결과','모델링에 반영할 결정'],[
('1,182행: 정상 1,157 / 불량 25(2.12%). 591개 패턴 모두 2행','원본 행과 라벨을 보존하고 동일 패턴을 같은 검증 폴드에 배치.'),
('566개 정상 전용, 25개 충돌, 불량 전용 0개','개별 정상/불량의 완전 분리보다 새로운 조건에서의 상대적 위험 순위를 평가.'),
('G2 불량 10/482, G3 불량 15/684: 2.07% 대 2.19%','조건군별 모델을 먼저 만들지 않는다. 공통 모델 결과를 조건군별로 진단.'),
('3개 군집화 방법에서 공정 군집은 형성되지만 불량 전용 군집은 없음','군집 번호를 불량 라벨로 해석하지 않고 기본 입력에도 추가하지 않음.'),
('상수열 제외 시 비라벨 18,416/35,941행(51.24%)이 관측 범위 밖','개발 검증과 적용 범위 점검을 분리. 범위 이탈을 불량으로 확정하지 않음.')],[4050,5250])
h('관측 자료의 구조적 식별 한계')
p('동일 입력에 동일 점수·동일 임계값을 적용하는 모델에서는 TP 하나마다 같은 입력의 정상 FP가 최소 하나 발생한다. 따라서 이 관측 자료에서 TP>0이면 precision=TP/(TP+FP)≤50%이다. 이는 미래 데이터의 상한이 아니라 현재 반복·라벨 구조에서 도출한 한계다.')
p('이 한계를 넘는 결과가 나오면 ID·행 순서·라벨 파생 정보의 유입, 짝의 분할, 점수 동률을 행 순서로 깨는 top-k 평가부터 검사한다. 모델이 충돌 자체를 해결했다고 해석하지 않는다.')
h('세 모델의 역할')
p('로지스틱 회귀: 약한 선형 위험도 신호의 기준선 → 랜덤포레스트: 조건 결합의 추가 정보 검증 → One-Class SVM: 정상 영역의 희귀도가 위험 순위와 연결되는지 검증. 세 모델 모두 동일 외부 폴드에서 평가한다.')

title(2,'그룹 분할과 전처리 파이프라인')
table(['외부 폴드','학습 / 검증 행','검증 불량','검증 고유 패턴'],[('0','788 / 394','8','197'),('1','788 / 394','8','197'),('2','788 / 394','9','197')],[1600,2900,2200,2600])
p('기존 labeled_metadata.csv의 fold를 고정한다. 분할 방식은 고유 패턴을 정상 전용·충돌 유형으로 층화한 StratifiedKFold이다. 일반 행 층화나 새 StratifiedGroupKFold 결과로 바꾸지 않는다. pattern_id는 24개 원본 값의 완전 일치로 정의한다. [1]')
h('외부 3겹 × 내부 2겹 개발 검증')
p('① 외부 검증 폴드를 잠금 → ② 외부 학습의 패턴 유형을 그 학습 라벨만으로 계산 → ③ 유형별 내부 2겹 분할(seed=42) → ④ 내부 평균 AP로 설정 선택 → ⑤ 내부 예측으로 경보 정책 선택 → ⑥ 외부 학습 전체로 재적합 후 검증.')
p('외부 학습에는 충돌 16~17개가 남아 내부 검증에 8~9개씩 배치할 수 있다. CN7과 비교 절차를 맞추고 탐색 변동을 제한하기 위해 내부 2겹을 권장한다. 유형은 분할 균형에만 사용한다. 내부·외부 모두 패턴 겹침 0, 두 클래스 존재를 확인한다.')
table(['처리','적용 기준'],[
('입력과 타깃','X_labeled.csv 24개 공정 변수; y_labeled.csv의 0=정상·1=불량. ID, 홀짝, 기록 순서, 빈도, 유형, fold 제외.'),
('상수열','원본 파일은 유지. 각 학습 구간에서 VarianceThreshold(0) 적합 후 같은 마스크로 검증 변환.'),
('스케일링','RF는 없음. LR은 전체 학습 행, OCSVM은 정상 학습 행에 StandardScaler 적합; 검증·비라벨에는 transform만 적용.'),
('원본 보존','불량 우선 축약, 충돌 삭제, SMOTE, 클리핑 없이 시작. 비라벨을 정상으로 간주해 학습에 합치지 않음.'),
('기존 EDA 변환','전체 데이터로 만든 순위 변환·군집·PCA 좌표를 모델 입력으로 재사용하지 않음.')],[2400,6900])
p('전체 라벨을 이미 EDA에 사용했으므로 중첩 검증도 독립적인 최종 시험은 아니다. ID를 시간으로 간주하지 않으며, 제공 좌표의 추가 스케일링이 파일별 표준화 차이나 물리 단위를 복원한다고 보지 않는다.','Small')

title(3,'랜덤포레스트: 비선형 위험도 비교')
h('선정 근거와 학습 가설')
p('온도·속도 결합 구간에서 불량 9/218행(4.13%)이 관측되었지만 정상 209행이 포함되고 불량 16행은 밖에 있다. 이 탐색 결과는 약한 조건 결합을 검증할 이유이며, 유효한 불량 규칙이 확정되었다는 근거는 아니다. RF로 추가 비선형 정보가 검증 폴드에도 남는지 확인한다. [1]')
table(['항목','초기 설정과 탐색'],[
('고정','n_estimators=500, criterion="gini", max_features="sqrt", bootstrap=True, random_state=42, n_jobs=-1, oob_score=False'),
('기본값','max_depth=3, min_samples_leaf=5, class_weight=None'),
('후보 8개','max_depth ∈ {3, 6}; min_samples_leaf ∈ {5, 15}; class_weight ∈ {None, "balanced"}'),
('학습 / 출력','정상·불량 모두 학습. classes_에서 클래스 1의 predict_proba 열을 찾아 불량 점수로 사용.')],[2300,7000])
h('학습 기준과 불균형 처리 [2]')
p('가중 노드 비율 pₖ에 대한 Gini 불순도 G=1−Σₖpₖ²를 사용해 분할 전후 불순도 감소를 최대화한다. 포레스트 전체를 하나의 교차엔트로피로 경사하강하는 구조는 아니다. 트리들의 확률 출력을 평균한다.')
p('balanced는 적합 데이터에서 클래스 k에 n/(2nₖ) 가중치를 준다. 불량의 영향을 높이지만 충돌 패턴에서 정상과 불량의 학습 비용 균형도 바꾼다. None과 비교해 순위 개선과 오탐 증가를 구분한다. 가중치가 동일 입력의 라벨 차이를 복원하지는 않는다.')
h('RG3에 맞춘 제한과 실패 진단')
p('CN7보다 큰 말단 크기 후보를 사용해 25개 충돌 조건의 암기를 억제한다. 이는 보수적 시작점이며 효과는 내부 검증으로 판단한다. 기본적인 행 부트스트랩 OOB는 같은 패턴의 짝이 학습에 남을 수 있어 평가 기준으로 사용하지 않는다.')
p('G2·G3를 나누는 변수만 중요하고 각 조건군 안에서 성능이 약하면, 공정 조건 차이를 학습했을 뿐 불량 위험을 설명하지 못했을 가능성을 점검한다. 조건군별 AP·TP·FP를 반드시 함께 기록한다.')
p('훈련 점수만 좋으면 패턴 누수 검사 후 깊이 제한과 말단 크기를 검토한다. 외부 검증 중요도는 설명용으로만 사용하고, 그 결과로 변수를 고른 뒤 같은 검증 점수를 독립 성능으로 재보고하지 않는다.','Small')

title(4,'로지스틱 회귀: 약한 신호의 기준선')
h('선정 근거')
p('개별 분포 차이가 작고 군집 내 불량 집중 근거도 제한적이다. 따라서 복잡한 모델에 앞서, 규제된 선형 위험도만으로 재현되는 정보가 있는지 확인한다. 금형 온도 3·4의 상관 0.987과 적은 불량 표본을 고려해 L2 규제를 사용한다. [1]')
table(['항목','초기 설정과 탐색'],[
('파이프라인','학습 상수열 제거 → StandardScaler → LogisticRegression'),
('기본값','penalty="l2", C=0.1, solver="lbfgs", max_iter=5000, class_weight=None'),
('후보 6개','C ∈ {0.01, 0.1, 1}; class_weight ∈ {None, "balanced"}'),
('학습 / 출력','정상·불량 모두 학습. 불량(클래스 1) 확률을 순위 점수로 사용; 임계값은 별도 결정.')],[2300,7000])
h('손실함수와 규제 [3]')
p('pᵢ=sigmoid(b+βᵀxᵢ). 목적함수의 형태는 가중 이진 교차엔트로피와 L2 규제의 합이다: L=−Σᵢaᵢ[yᵢ log pᵢ+(1−yᵢ)log(1−pᵢ)] + λ‖β‖²/2. 절편은 규제 항에서 제외한다. C가 작을수록 규제가 강하며, 정확한 계수 정규화는 구현 관례를 따른다.')
p('aᵢ는 클래스 가중치다. balanced는 학습 자료의 n/(2nₖ)를 사용하며 실제 공정 비용으로 측정된 값은 아니다. 가중 학습 확률을 곧바로 실제 불량 확률로 해석하지 않는다.')
h('해석과 개선 판단')
p('계수 부호와 폴드별 변동, 수렴 경고와 n_iter_를 기록한다. 높은 상관이 있는 변수의 부호 변화는 단일 변수의 제조 효과 반전으로 해석하지 않는다. 규제·반복 행·소수 불량이 있으므로 단순 회귀의 유의확률 검정을 중심으로 보고하지 않는다.')
p('RF가 LR보다 안정적으로 개선되면 조건 결합의 추가 정보 가능성을 확인한다. 둘 다 기준선 부근이면 “비선형 모델이 더 필요하다”로 바로 결론 내리지 않고 라벨 충돌·적은 표본·적용 범위·미측정 변수의 한계를 먼저 점검한다.')
p('KS 상위 변수만 고르거나 EDA에서 발견한 고온·고속 규칙을 기본 특징으로 추가하지 않는다. 초기 비교는 공통 입력으로 수행하고, 변수 축소는 별도의 학습 폴드 내부 민감도 분석으로 제한한다.')
p('설정 문법은 현재 환경의 scikit-learn 1.2.1을 기준으로 한다. 버전 변경 시 penalty 등 API 변경을 확인하고 실행 환경을 기록한다.','Small')

title(5,'One-Class SVM: 희귀 정상과 불량의 관계')
h('선정 근거의 강도와 학습 대상')
p('RG3에는 정상과 분리된 불량 전용 패턴이 없다. 따라서 CN7의 범위 밖 불량 가설을 그대로 적용할 근거가 없다. OCSVM은 정상 영역에서 상대적으로 희귀한 조건에 충돌 불량이 더 자주 나타나는지 확인하는 비교 모델이다. 우월한 탐지력을 전제하지 않는다. [1]')
p('각 학습 구간의 정상 행만으로 전처리와 모델을 적합한다. 충돌 패턴의 정상 행도 유지한다. 검증은 정상·불량 전체에서 수행한다. 정상 학습에 사용되지 않은 새 패턴에서 평가하되, 검증의 정상 짝과 불량 짝은 같은 점수를 받는다.')
table(['항목','초기 설정과 탐색'],[
('파이프라인','학습 정상 선택 → 정상에서 상수열 제거·StandardScaler 적합 → OneClassSVM'),
('기본값','kernel="rbf", nu=0.02, gamma="scale", tol=0.001'),
('후보 9개','nu ∈ {0.01, 0.02, 0.05}; gamma ∈ {0.25/d, 1/d, 4/d}; d=해당 정상 학습의 활성 변수 수'),
('출력','s=−decision_function. 높을수록 이상. predict의 −1=이상·+1=정상은 원본 라벨 1/0과 다름.')],[2300,7000])
h('목적함수와 매개변수 [4, 5]')
p('min ½‖w‖² + (1/(νn))Σᵢξᵢ − ρ; 제약 wᵀφ(xᵢ)≥ρ−ξᵢ, ξᵢ≥0. n은 정상 학습 행 수, ξ는 경계 위반 허용량이다. RBF는 exp(−γ‖x−x′‖²)로 거리에 반응한다.')
p('nu는 학습 오류 비율의 상한과 서포트 벡터 비율의 하한에 관련된다. 관측 불량률 2.12%나 실제 검증 오탐률과 같지 않다. gamma가 커지면 더 좁고 복잡한 경계가 가능하다. 내부 라벨로 후보를 선택하므로 전체 평가 절차는 순수 무라벨 학습·선택이 아니다.')
h('RG3에서 특히 확인할 실패')
p('G1·G4의 희귀 정상이나 주요 조건군의 경계에 경보가 몰리는지 확인한다. 공정의 희귀도가 불량 위험도와 다르면 OCSVM의 AP가 낮아도 구현 오류라고 단정하지 않는다.')
p('정상 학습에서는 정상 전용 패턴이 2행, 충돌 패턴이 1행으로 남는다. 이 밀도 가중 구조도 결과에 영향을 줄 수 있다. 기본 비교는 원본 빈도를 유지하고, 정상 패턴 균등 가중은 필요할 때만 별도 민감도 분석으로 시행한다.','Small')

title(6,'평가와 임계값 선택')
table(['항목','보고 기준'],[
('주 지표','내부 폴드 평균 AP(average precision)로 후보 선택. 외부 AP는 폴드별 값과 평균·범위를 기록. 사다리꼴 PR-AUC와 구분. [6]'),
('보조 지표','ROC-AUC, precision, recall, F2, 정상 오탐률, 경보율 및 TP/FP/FN/TN 실제 건수.'),
('기준선','항상 정상: 정확도 97.88%, recall=0. 전체 불량률 2.12%를 AP 해석 참조로 사용; 무작위 순위의 실현 AP와 동일한 수치는 아님.'),
('분해 결과','G2·G3 각각의 AP와 혼동행렬. G1·G4는 불량 0개이므로 AP·recall 해석을 피하고 정상 경보 건수를 보고.'),
('동률 / 집계','동일 입력의 짝에 같은 점수·판정을 보장. top-k에서 짝을 임의로 쪼개지 않음. 모든 패턴이 2행이라 패턴 균등 가중은 행 가중과 비례.')],[2300,7000])
h('내부에서 경보 규칙을 고르고 외부에서 평가')
p('1. 내부 평균 AP로 설정 선택. 동률이면 LR은 더 작은 C, RF는 더 얕은 깊이·큰 말단, OCSVM은 더 작은 gamma를 우선한다. 나머지 동률은 사전 고정한 후보 순서로 결정한다.')
p('2. 선택 설정에서 α∈{1%, 2%, 5%, 10%, 20%} 비교. 각 내부 학습 모델의 정상 학습 점수에서 tα=quantile(s정상,1−α)를 구해 내부 검증에 s>tα 적용. 평균 F2 최대의 α를 선택하고 동률이면 작은 α를 선택한다.')
p('3. 외부 학습 전체로 재적합 후, 해당 모델의 정상 학습 점수에서 선택된 α의 tα를 다시 구한다. 이를 외부 검증에 고정 적용한다. RF/LR은 클래스 1 확률, OCSVM은 음의 decision_function을 공통 방향의 점수로 사용한다.')
p('분위수 방식은 linear, 비교 연산은 >로 고정한다. α는 경보 규칙이며 외부 오탐률 보장이 아니다. 학습 점수의 낙관성과 동률 때문에 실제 경보율이 달라지므로 α·임계값·검증 오탐률을 함께 기록한다.')
h('해석 시 주의할 점')
p('대회 공식 지표와 오탐/미탐 비용은 제공되지 않았다. F2는 누락을 더 중시하는 잠정 지표이며 실행 전 공식 기준과 정합성을 확인한다. 임계값 결과와 순위 성능을 분리해 읽는다.')
p('불량 8~9개인 폴드는 한 건으로 recall이 11.1~12.5%p 변한다. 3개 폴드 표준편차를 신뢰구간으로 부르지 않는다. OCSVM의 폴드별 점수 척도가 달라 원점수를 합친 OOF AP로 순위를 결정하지 않는다. 고정 규칙 적용 후 혼동행렬은 합산할 수 있다.','Small')

title(7,'군집 결과 활용과 실패 원인 개선')
h('군집 안정성과 불량 예측력을 구분 [1]')
p('기존 80% 부분표본 반복의 평균 ARI는 K-Means 0.984, Ward 0.912, K-Medoids 0.660이다. 이는 관측 공정 구조의 재현성 지표이며 불량 분리 성능이 아니다. 군집별 충돌 비율은 조건군을 유지한 라벨 무작위화 참조 범위 안이었다.')
p('기본 모델에 기존 전체 데이터 군집 번호를 추가하지 않는다. 필요 시 학습 폴드 안에서만 K-Means를 재적합하고 검증에는 predict로 군집을 배정한 보조 실험을 시행한다. 군집화 전처리 역시 학습 내 적합한다. Ward 결과는 새 데이터 predict를 가진 모델처럼 취급하지 않는다.')
table(['실패 관측','원인 점검과 개선'],[
('precision이 50%를 넘음','관측 짝 구조의 상한과 충돌. ID·라벨 파생 정보·동률 처리·집계 오류부터 검사.'),
('정상 짝과 불량을 함께 경보','현재 입력의 식별 한계. 모델 복잡도 확대보다 생성·검수·측정 해상도와 추가 기록 확인.'),
('RF만 훈련 성능이 높음','충돌 조건 암기 또는 누수. 그룹 겹침 검사 후 깊이·말단 제한 점검.'),
('G2/G3 구분은 잘 되나 AP는 낮음','공정 조건 구분과 불량 구분의 차이. 조건군별 위험 순위를 확인.'),
('OCSVM이 G1/G4에 경보 집중','희귀 정상 탐지 가능성. 표본 부족을 불량 근거로 바꾸지 않음.'),
('비라벨 경보 급증','범위·고유값·표준화 차이 점검. 비라벨을 별도 재표준화하거나 정상 범위로 클리핑하지 않음.')],[3200,6100])
h('실행·저장·최종 재학습')
p('입력 행 정렬·그룹 중복 검사 → LR/RF/OCSVM 중첩 검증 → 실패 유형 비교 → 변경 하나씩 추가 실험 → 모델·정책 확정 → 전체 라벨 데이터에서 그룹 3겹 선택 후 최종 적합 순서로 진행한다.')
p('최종 RF/LR은 모든 라벨 행, OCSVM은 모든 정상 행으로 학습한다. 점수·판정·source_id·pattern_id·fold·선택 설정·α·임계값과 모델/스케일러/상수 마스크, 파일 해시·버전을 저장한다. 모델 비교에 쓴 개발 점수는 독립 시험 성능이 아니다.')
p('초기 후보 8+6+9=23개: 외부 3겹 × 내부 2겹 후보 적합 138회, 외부 재적합 9회. EDA와 원본 파일은 유지하고 모델링 산출물을 별도 디렉터리에 저장한다.','Small')

title(8,'근거와 적용 범위')
p('데이터 사실은 RG3 노트북 및 저장된 산출물을 근거로 했다. 후보 범위·내부 분할·α·F2는 제안한 설계다. 세 모델의 성능, 최적 설정, 대회 개선 효과는 아직 검증하지 않았다.')
refs=[('[1] RG3 원자료와 분석 산출물','EDA/rg3eda.ipynb; data/processed/rg3/preprocessing_manifest.json; fold_summary.csv; output/rg3_clustering_eda/; output/rg3_structure_eda/'),('[2] RandomForestClassifier','https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'),('[3] LogisticRegression 및 목적함수','https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html\nhttps://scikit-learn.org/stable/modules/linear_model.html#logistic-regression'),('[4] OneClassSVM','https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html'),('[5] SVM 목적함수','https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation'),('[6] Average precision','https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html')]
for name,urls in refs:
    h(name)
    for u in urls.splitlines(): p(u,'Small')
p('온라인 API 문서 확인일: 2026-09-27. stable 문서는 변경될 수 있으므로 실행 환경의 버전을 고정한다. 노트북의 EDA 결과는 연관과 관측 구조를 보여주며 제조 실패의 인과관계나 개선 효과를 증명하지 않는다.','Small')

template=ROOT/'output/cn7_modeling_strategy/CN7_modeling_strategy.docx'
with zipfile.ZipFile(template) as z: entries={n:z.read(n) for n in z.namelist()}
old=ET.fromstring(entries['word/document.xml'])
sect=ET.tostring(old.find(f'{{{W}}}body/{{{W}}}sectPr'),encoding='unicode')
document=f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{W}"><w:body>{"".join(parts)}{sect}</w:body></w:document>'
entries['word/document.xml']=document.encode('utf-8')
entries['word/footer1.xml']=entries['word/footer1.xml'].replace(b'CN7',b'RG3')
for name,b in entries.items():
    if name.endswith(('.xml','.rels')): ET.fromstring(b)
target=OUT/'RG3_modeling_strategy.docx'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name,b in entries.items(): z.writestr(name,b)
with zipfile.ZipFile(target) as z: assert z.testzip() is None
X=pd.read_csv(ROOT/'data/processed/rg3/X_labeled.csv')
y=pd.read_csv(ROOT/'data/processed/rg3/y_labeled.csv').iloc[:,0]
m=pd.read_csv(ROOT/'data/processed/rg3/labeled_metadata.csv')
assert len(X)==1182 and y.sum()==25 and len(X.drop_duplicates())==591
g=pd.util.hash_pandas_object(X,index=False)
a=pd.DataFrame({'g':g,'y':y,'fold':m.fold}).groupby('g').agg(n=('y','size'),bad=('y','sum'),folds=('fold','nunique'))
assert (a.n==2).all() and (a.bad==1).sum()==25 and a.bad.max()==1 and a.folds.max()==1
verification={'source_notebook':'EDA/rg3eda.ipynb','source_sha256':hashlib.sha256((ROOT/'EDA/rg3eda.ipynb').read_bytes()).hexdigest(),'rows':1182,'patterns':591,'conflicting_patterns':25,'defect_only_patterns':0,'document':target.name,'verification':'XML and ZIP parsed; actual input equality, paired labels and fold isolation checked. Office rendering not performed.','models_trained':False}
(OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(target); print('Verified:',target.stat().st_size,'bytes')
