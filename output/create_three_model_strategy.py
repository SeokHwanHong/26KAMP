from pathlib import Path

# Reuse only the document formatting helpers; do not execute the prior report.
helper = Path(__file__).with_name('create_logistic_strategy.py')
exec(compile(helper.read_text(encoding='utf-8').split('stats=[]')[0], str(helper), 'exec'))
OUT = ROOT / 'output/three_model_strategy'
OUT.mkdir(exist_ok=True)
stats=[]
sources={}
for name in ['cn7','rg3']:
    path=ROOT/f'data/origin/moldset_labeled_{name}.csv'
    df=pd.read_csv(path)
    features=json.loads((ROOT/f'data/processed/{name}/preprocessing_manifest.json').read_text(encoding='utf-8'))['feature_columns']
    label=[c for c in df if c not in features and not c.startswith('Unnamed')][0]
    n0=int((df[label]==0).sum()); n1=int((df[label]==1).sum())
    groups=df.groupby(features,dropna=False)[label].agg(['min','max','size'])
    conflict=int(((groups['min']==0)&(groups['max']==1)).sum())
    only=int((groups['min']==1).sum())
    stats.append([name.upper(),len(df),n0,n1,len(groups),conflict,only])
    sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()

doc.add_heading('사출성형 불량 탐지 모델 학습 전략',0)
p('Logistic Regression · Random Forest · One Class SVM')
p('CN7 및 RG3 | 2026년 9월 28일')
p('공정 입력을 이용해 정상과 불량을 구분하고, 어떤 공정 조건에서 불량 탐지와 누락이 발생하는지 분석한다. LR은 선형 관계의 기준 모델, RF는 변수 간 조건 조합을 학습하는 모델, OCSVM은 정상 영역에서 벗어나는 불량을 탐지하는 모델로 비교한다. CN7과 RG3는 각각 학습한다.')
h('1 공통 설계와 모델별 역할')
table(['구분','LR','RF','OCSVM'],[
('학습 관측','정상과 불량','정상과 불량','정상만'),
('학습 목적','가중 로그 손실과 L2 규제','가중 불순도 감소','정상 영역의 경계 추정'),
('입력 스케일','학습 구간 재표준화','제공값 그대로','학습 정상으로 재표준화'),
('불균형 대응','클래스 가중치 후보 비교','균형 가중치 확정','nu와 gamma 및 임계값'),
('주요 질문','선형 관계로 구분 가능한가','조건 조합이 도움이 되는가','불량이 정상 영역 밖인가')],[1.05,1.7,1.7,1.7])
p('공통 평가에는 불량을 양성 클래스 1로 둔 F1과 F2를 사용한다. 동일 입력 패턴을 묶은 외부 4-fold에서 평가하고, 내부 검증에서 설정과 판정 임계값을 선택한다. 외부 검증 라벨로 설정이나 임계값을 조정하지 않는다.')
p('확정 사항은 외부 4-fold, 공통 F1·F2 보고, RF의 정상 대비 불량 가중치 n₀/n₁이다. LR 가중치 후보와 모델별 탐색 범위, 내부 AP 기반 설정 선택은 실행을 위한 제안이다. 임계값은 공통으로 내부 F2 기준을 사용한다. 본 문서는 설계이며 성능 결과는 아직 없다.')

page('2 데이터 구조와 교차검증')
table(['데이터','전체','정상','불량','패턴','상충','불량 전용'],stats,[.75,.8,.8,.7,.8,.8,1.5])
p('패턴은 원본 24개 입력값이 모두 같은 집단이다. 상충은 같은 입력에 정상과 불량 라벨이 함께 있는 패턴이다. CN7의 불량 17행 중 11행은 상충 패턴, 6행은 불량 전용 3개 패턴에 속한다. RG3의 불량 25행은 모두 상충 패턴에 속한다. 원본 행과 라벨을 유지하고 불량 우선 중복 제거는 하지 않는다.')
h('외부 4 fold와 내부 선택')
p('학습에 약 75%를 사용하여 학습 표본 확보와 검증 표본 수 사이에서 절충한다. 4가 최적이라는 주장은 하지 않는다. StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)를 초기 분할안으로 사용하고, group에는 pattern_id를 지정한다. 그룹 보존으로 인해 클래스 비율은 완전히 같지 않을 수 있다.')
p('각 fold의 정상·불량 수, 고유 패턴 수와 유형, 학습·검증 패턴 교집합을 확인한다. CN7에는 불량 전용 패턴이 3개뿐이므로 모든 검증 fold에 해당 유형을 포함시킬 수 없다. 현재 저장된 processed 메타데이터는 3-fold이며, 4-fold 재생성은 모델 구현 단계에서 수행한다.')
p('외부 학습 안에서 내부 2-fold를 구성한다. 내부에서도 그룹을 보존하고 두 클래스가 존재하는지 확인한다. 같은 내부 분할을 모델 간 재사용한다. 전처리와 클래스 수 계산은 매번 실제 적합에 사용하는 구간에서만 수행한다.')
h('입력과 스케일')
p('입력은 24개 공정 변수이며 ID, fold, pattern_id, 라벨에서 계산한 패턴 유형은 제외한다. 학습 구간에서 상수 변수를 제거한다. 제공 가이드에서 확인한 CSV의 기존 표준화와 별개로, LR은 학습 전체, OCSVM은 학습 정상에만 StandardScaler를 적합한다. RF는 추가 표준화 없이 사용한다.')
p('검증 데이터를 별도로 표준화하지 않는다. 기존 변환이 고정된 컬럼별 양의 선형 표준화라면 학습 fold 재표준화는 원자료의 학습 fold 표준화와 동등하다. 비라벨 파일은 별도 표준화된 자료이므로 향후 추론 전에 동일 좌표계인지 확인한다. 기본 실험에서는 비라벨 데이터, PCA, SMOTE를 학습에 추가하지 않는다.')

page('3 LR 원리와 가중 로그 손실')
p('LR은 공정 변수의 선형 결합을 sigmoid 함수에 통과시켜 0~1의 예측값을 만든다. 변수별 영향이 로그 오즈에 선형으로 더해지는 모델이며, 상호작용을 표현하려면 추가 특징 설계가 필요하다.')
p('zᵢ = β₀ + βᵀxᵢ     pᵢ = 1 / (1 + exp(−zᵢ))')
p('ℓᵢ = −[yᵢ log(pᵢ) + (1−yᵢ) log(1−pᵢ)]')
p('J = (1/n) Σᵢ aᵢℓᵢ + (λ/2) ||β||²')
table(['기호','의미'],[
('xᵢ 및 yᵢ','전처리된 행 i의 입력 벡터와 라벨 0 또는 1'),
('β₀ 및 β','절편과 변수별 회귀 계수'),
('pᵢ 및 ℓᵢ','불량 예측값과 가중치 적용 전 행별 로그 손실'),
('aᵢ 및 n','행별 가중치와 현재 학습 행 수'),
('λ','L2 규제 강도이며 절편은 규제에서 제외')],[1.2,4.95])
p('목적함수는 개념식이다. 구현에서는 C가 작을수록 규제가 강하다. 정규화 관례를 확인하지 않고 λ=1/C로 단정하지 않는다. L2 규제는 소표본과 상관 변수로 인해 계수가 과도하게 커지는 것을 억제한다.')
h('불량 비율이 낮아도 학습되는 이유')
p('불량 비율은 초기값이 아니다. 계수와 절편이 0이면 초기 예측은 0.5이다. 로그 손실의 zᵢ에 대한 기울기는 pᵢ−yᵢ이고, 불량에 0.02를 예측하면 −0.98이다. 가중 손실에서는 aᵢ가 곱해진다. 불량을 낮게 예측했다고 학습 신호가 사라지는 것은 아니다.')
h('초기 비교안')
p('불량 대 정상 가중치 비율 r은 1, √(n₀/n₁), n₀/n₁을 비교한다. 평균 행 가중치를 1로 맞추려면 a₀=n/(n₀+r n₁), a₁=r a₀로 둔다. n₀와 n₁은 해당 학습 구간의 클래스 수이다. 균형 설정은 class_weight="balanced"와 같다.')
p('L2와 lbfgs를 사용하고 C는 0.01, 0.1, 1, 10을 초기 후보로 둔다. max_iter=3000에서 수렴을 확인한다. 가중 예측값은 실제 불량 발생률로 바로 해석하지 않는다. focal loss 등 추가 손실 변경은 첫 실험에서 제외한다.')

page('4 RF 원리와 분할 기준')
p('RF는 여러 의사결정나무의 예측을 평균한다. 각 트리는 학습 행을 복원추출하고, 각 노드에서 무작위로 선택한 일부 컬럼을 대상으로 분할을 탐색한다. 온도가 높고 시간이 짧은 경우처럼 조건의 결합을 분기 경로로 표현한다. 무작위성은 후보를 다양하게 만들며, 후보 안에서는 불순도 감소가 큰 분할을 선택한다.')
p('p₁,w(t) = [Σᵢ∈Iₜ aᵢ 1(yᵢ=1)] / [Σᵢ∈Iₜ aᵢ]')
p('p₀,w(t) = 1 − p₁,w(t)     G_w(t) = 1 − p₀,w(t)² − p₁,w(t)²')
p('ΔG_w = G_w(t) − [(W_L/W_t)G_w(L) + (W_R/W_t)G_w(R)]')
table(['기호','의미'],[
('t 및 Iₜ','현재 노드와 그 노드에 속한 행 인덱스 집합'),
('aᵢ 및 1(yᵢ=1)','행 가중치와 불량이면 1인 지시함수'),
('p₁,w 및 p₀,w','가중치를 반영한 불량 비율과 정상 비율'),
('G_w 및 ΔG_w','가중 Gini 불순도와 분할 후 감소량'),
('W_t 및 W_L 및 W_R','부모 노드와 좌우 자식 노드의 가중치 총합')],[1.7,4.45])
p('이진 Gini는 한 클래스만 존재할 때 0, 가중 클래스 비율이 50%씩일 때 0.5이다. 클래스 가중치와 부트스트랩 중복 횟수가 유효 기여도에 반영된다. RF는 LR처럼 전체 로그 손실을 미분해서 최적화하는 모델이 아니다.')
h('균형 가중치의 의미')
p('정상 가중치 1, 불량 가중치 n₀/n₁을 기준으로 확정한다. 그러면 전체 학습에서 정상 총 가중치와 불량 총 가중치가 같아진다. class_weight="balanced"는 같은 상대 비율을 구현한다. 불량을 더 자주 추출하는 옵션은 아니며, 각 노드나 트리의 클래스 비중까지 50%로 만드는 것도 아니다.')
p('정상 98개와 불량 2개라면 불량 한 행 가중치는 49이다. 반면 n₀²/n₁은 불량 전체를 정상 전체보다 n₀배 강조하고 데이터 복제만으로도 상대 가중치가 바뀌므로 채택하지 않는다. 가중치는 각 학습 구간에서 계산하며 전체 검증 라벨을 사용하지 않는다.')

page('5 RF 예측과 복잡도 제어')
p('새 입력은 각 트리의 조건을 따라 리프 노드에 도착한다. 해당 리프의 가중 불량 비율을 트리 예측값으로 사용하며, scikit-learn RF는 트리별 클래스 확률을 평균한다.')
p('p_RF(x) = (1/B) Σ_b p_b(x)')
p('B는 트리 수, p_b(x)는 b번째 트리의 예측값이다. 단순한 라벨 다수결과는 구분한다. 가중 학습의 출력은 우선 불량 예측 점수로 해석하고, 내부 검증에서 정한 임계값으로 정상 0과 불량 1을 판정한다.')
table(['설정','역할과 검토 방향'],[
('n_estimators','트리 수를 늘려 무작위 예측 변동을 줄인다.'),
('max_depth','분기 깊이를 제한하여 세부 조건 암기를 줄인다.'),
('min_samples_leaf','리프의 최소 행 수로 소집단 과적합을 제어한다.'),
('max_features','분할 후보 컬럼 수로 트리 다양성을 조절한다.'),
('class_weight','balanced로 고정한다.'),
('criterion 및 bootstrap','gini와 복원추출을 기본으로 사용한다.')],[1.65,4.5])
h('실행을 위한 초기 후보')
p('n_estimators=500, max_features="sqrt", random_state=42를 초기 고정안으로 둔다. max_depth는 3, 5, None, min_samples_leaf는 1, 3, 5를 비교하는 9개 조합을 제안한다. 이 값들은 검증된 최적값이 아니다. 가중치 효과를 확인하기 위한 무가중 기준 모델은 별도로 보고한다.')
p('리프 최소 행 수가 너무 크면 소수 불량을 따로 분리하기 어렵고, 너무 작으면 불량 몇 개를 외울 수 있다. 동일 입력 중복 행 수는 독립적인 공정 패턴 수와 다르므로, 리프 행 수만으로 근거가 충분하다고 해석하지 않는다.')
h('해석과 평가상의 주의점')
p('변수 중요도는 모델이 활용한 정보이며 공정의 인과관계가 아니다. 상관된 변수 사이에서는 중요도가 나뉘거나 한쪽에 집중될 수 있다. 대표 분기와 검증 기반 중요도를 함께 확인하고, 전체 데이터에서 얻은 해석을 다시 특징 선택에 쓰면 별도 검증한다.')
p('OOB 평가는 트리별 미추출 행을 평가하지만 동일 입력의 다른 행이 학습에 남을 수 있다. 따라서 본 데이터에서는 그룹을 보존한 외부 검증을 대신하지 않는다. 동일 입력의 상충 라벨에는 동일 점수가 나오므로 RF도 그 라벨들을 완전히 분리할 수 없다.')

page('6 OCSVM 원리와 목적함수')
p('OCSVM은 정상 관측의 분포를 학습하고 정상 영역 밖의 새 관측을 이상 후보로 판단한다. 일반 분류 SVM인 SVC와 달리 정상과 불량 사이의 경계를 직접 학습하지 않는다. 본 설계에서는 정상만으로 적합하고, 불량 라벨은 내부 검증의 설정 및 임계값 선택에 사용한다.')
p('min J = ½||w||² + [1/(νn)] Σᵢ ξᵢ − ρ')
p('제약조건   wᵀφ(xᵢ) ≥ ρ − ξᵢ,   ξᵢ ≥ 0')
table(['기호','의미'],[
('n 및 xᵢ','학습 정상 관측 수와 전처리된 입력 벡터'),
('φ(xᵢ)','커널이 암묵적으로 표현하는 특징 공간의 입력'),
('w 및 ρ','특징 공간의 경계 벡터와 경계 위치'),
('ξᵢ','정상 관측 i의 경계 조건 위반 정도'),
('ν 또는 nu','경계 위반 벌점과 해의 성질을 조절하는 값')],[1.3,4.85])
p('첫 항은 경계를 규제하고, 두 번째 항은 경계 조건 위반에 벌점을 준다. 마지막 항은 경계 위치를 조정한다. 일부 학습 관측의 위반을 허용하면서 정상 영역을 표현한다. 불량에 n₀/n₁ 가중치를 주는 분류 손실과는 다르다.')
h('RBF 커널과 gamma')
p('K(xᵢ,xⱼ) = exp(−γ ||xᵢ−xⱼ||²)')
p('K는 두 관측의 유사도, 제곱 거리 ||xᵢ−xⱼ||²는 입력 차이, γ는 거리 증가에 따라 유사도가 감소하는 속도이다. gamma가 크면 가까운 관측만 유사하게 보고 정상 영역이 세밀해질 수 있다. 작으면 넓은 범위의 관측을 유사하게 본다. 학습 정상 기준의 스케일링이 중요하다.')
h('nu의 해석')
p('nu는 이론적으로 학습 경계 위반 비율의 상한이자 서포트 벡터 비율의 하한이다. 서포트 벡터는 경계를 결정하는 학습 관측이다. 실제 계산에서는 경계 위의 점과 수치 오차를 고려해야 한다. nu=0.03이 새 데이터의 3%를 불량으로 판정한다는 뜻은 아니며, 정상만 학습하므로 전체 불량률을 그대로 nu로 정할 근거도 없다.')

page('7 OCSVM 점수와 학습 전략')
p('f(x) = wᵀφ(x) − ρ     s(x) = −f(x)')
p('f는 decision_function의 결정 점수이다. 기본적으로 양수이면 정상 영역 안, 음수이면 밖이다. predict는 각각 +1과 −1을 반환한다. 이 −1은 확정 불량이 아니라 이상 탐지 결과이다. 우리 공통 라벨은 정상 0, 불량 1로 유지한다.')
p('s는 부호를 뒤집은 이상 점수로 클수록 이상 가능성이 높다. s(x)>τ이면 불량 1, 아니면 정상 0으로 판정한다. τ는 내부 검증에서 선택한다. s는 0~1 확률이 아니므로 LR의 로그 손실을 그대로 계산하지 않는다.')
h('적합과 내부 검증')
p('먼저 정상·불량 전체를 대상으로 그룹 분할한다. 이후 내부 학습의 정상만 선택하여 상수 제거, 스케일러, OCSVM을 적합한다. 내부 검증의 정상·불량 모두에 이상 점수를 산출하고 평균 AP로 nu와 gamma를 선택한다. 선택한 설정의 내부 OOF 점수에서 F2 임계값을 정한다.')
p('RBF 커널, nu 후보 0.01, 0.05, 0.10과 gamma 기준값의 0.1배, 1배, 10배를 초기안으로 제안한다. gamma 기준값은 각 정상 학습의 표준화 후 남은 변수 수 d에 대해 1/d로 둔다. 총 9개 후보이며 최적값으로 확정한 수치는 아니다.')
p('선택 후 외부 학습의 정상 전체로 다시 적합하고 외부 검증 전체에 평가한다. 설정과 임계값 선택에 불량 라벨을 사용하므로 전체 실험을 완전한 무라벨 비지도 실험이라고 설명하지 않는다.')
h('정상만 학습할 때의 한계')
p('CN7의 불량 전용 패턴은 정상 영역 밖에 놓이는지 검토할 가치가 있다. RG3의 모든 불량은 정상과 같은 입력을 가지므로, 같은 검증 그룹의 정상·불량 쌍에는 동일 점수가 나온다. 탐지율 상승과 오탐 증가를 함께 해석해야 한다.')
p('OCSVM 점수 크기는 재적합 모델마다 달라질 수 있다. 내부 점수에서 고른 수치 임계값이 외부 재적합 모델에서 얼마나 안정적인지 기록한다. 초기안에서는 이 절차를 일관되게 적용하고, 안정성이 부족하면 정상 기준 점수 정규화 등 별도의 사전 정의된 비교안을 검토한다. 외부 결과에 맞춰 임계값을 다시 조정하지 않는다.')

page('8 공통 평가와 결과 분석')
p('모든 모델에서 실제 불량을 양성으로 둔다. TP는 탐지한 불량, FP는 불량으로 오인한 정상, FN은 놓친 불량, TN은 정상으로 판정한 정상이다.')
p('Precision = TP/(TP+FP)     Recall = TP/(TP+FN)')
p('Fβ = (1+β²) × Precision × Recall / (β² × Precision + Recall)')
p('여기서 β는 F 지표의 Recall 강조 정도이며 LR의 계수 기호와는 별개이다. β=1이면 F1, β=2이면 F2이다. 분모가 0인 경우 점수 0과 관련 관측 수를 기록하는 규칙을 고정한다.')
table(['항목','분석 목적'],[
('AP','불량에 높은 점수를 부여하는 순위 성능'),
('F1','선택한 임계값에서 Precision과 Recall의 균형'),
('F2','불량 누락에 더 큰 비중을 둔 평가'),
('TP 및 FP 및 FN 및 TN','점수 차이를 실제 탐지·오탐 건수로 설명'),
('fold별 결과','분할에 따른 성능 변동과 유형 차이 확인'),
('전체 OOF 결과','각 행의 외부 검증 예측을 모아 전체 성능 계산')],[1.5,4.65])
p('F1·F2는 binary 기준으로 계산한다. 정상 다수의 영향을 크게 받는 weighted F1으로 대체하지 않는다. fold별 평균·표준편차와 전체 OOF 지표를 모두 보고한다. 평균 F1과 전체 TP·FP·FN으로 계산한 F1은 서로 다를 수 있다.')
h('선택 기준과 평가 기준의 구분')
p('내부 평균 AP로 모델 설정을 선택하고, 선택 설정의 내부 OOF 예측으로 F2 최대 임계값을 선택한다. AP 동률이면 더 단순한 모델을 우선하는 규칙을 실행 전에 고정한다. F2 동률이면 Precision이 높은 임계값, 이후 더 높은 임계값을 우선한다. 외부 F1은 F2로 정한 임계값에서의 균형 성능이다.')
p('외부 fold마다 임계값이 다를 수 있으므로 전체 OOF 분류 지표는 각 fold의 판정을 합쳐 계산한다. 특히 OCSVM의 전체 OOF AP는 모델별 점수 척도 차이에 영향을 받을 수 있어 fold별 AP와 함께 해석한다. 소수 불량 한두 건의 차이가 점수를 크게 바꿀 수 있으므로 건수를 생략하지 않는다.')

page('9 시각화와 실행 결과 저장')
h('LR과 RF의 행별 손실 그림')
p('X축은 외부 OOF 불량 예측값 pᵢ, Y축은 가중치 적용 전 로그 손실 ℓᵢ로 둔다. 실제 정상은 파랑, 불량은 빨강으로 표시한다. RF에서도 로그 손실을 사후 평가값으로 계산할 수 있지만 트리가 직접 최적화하는 분할 기준은 아니다. 확률 0 또는 1은 작은 ε를 이용해 제한하여 계산한다.')
p('왼쪽 위 빨간 점은 불량에 낮은 점수를 준 사례, 오른쪽 위 파란 점은 정상에 높은 점수를 준 사례이다. 두 클래스 곡선 위에서 점이 겹칠 수 있으므로 투명도나 관측 개수를 표시한다. 가중 손실은 별도 그림으로 제공하고 서로 다른 가중치의 손실 크기를 직접 성능 순위로 해석하지 않는다.')
h('OCSVM과 공통 그림')
p('OCSVM은 확률이 없으므로 로그 손실 대신 정상·불량별 이상 점수 분포와 임계값을 표시한다. 공통으로 PR 곡선, fold별 F1·F2, 혼동행렬과 누락 사례 표를 제공한다. 임계값은 fold별 패널에 표시하며 통합 그림에 단일 임계값을 그리지 않는다. 행 ID를 시간으로 해석하지 않는다.')
h('실행 및 저장 순서')
for s in [
'1. 원본 해시와 클래스 수를 확인하고 그룹을 보존한 외부 4-fold를 생성한다.',
'2. 동일 내부 분할에서 모델별 전처리, 설정 선택과 임계값 선택을 수행한다.',
'3. 외부 검증 예측을 저장하고 F1·F2 및 오탐·누락을 계산한다.',
'4. CN7 불량 전용 패턴과 상충 패턴, RG3 상충 쌍의 결과를 구분해서 해석한다.',
'5. 평가 완료 후 전체 라벨 데이터 내부 검증으로 설정과 임계값을 정하고 최종 모델을 적합한다.'
]: p(s)
p('행별 저장 항목은 source_id, pattern_id, model, outer_fold, y_true, score, threshold, y_pred이다. LR·RF는 log_loss, class_weight, weighted_log_loss도 저장한다. 라이브러리 버전, seed, 설정 후보, 선택값, 수렴 여부와 데이터 해시를 기록한다. 전체 EDA를 이미 수행한 만큼 결과는 개발 평가로 해석하고 독립 데이터에서 재확인한다.')

page('10 참고 자료와 적용 상태')
p('데이터 수치와 입력 패턴 유형은 문서 생성 시 원본 CSV에서 재확인하였다. 본 문서 생성은 기존 데이터, 분할 파일, 노트북 또는 학습 코드를 변경하지 않는다. 실제 모델 학습과 성능 비교는 이후 구현 작업이다.')
for s in [
'프로젝트 원본: data/origin/moldset_labeled_cn7.csv 및 moldset_labeled_rg3.csv.',
'프로젝트 메타데이터: data/processed/cn7 및 rg3의 preprocessing_manifest.json과 fold_summary.csv.',
'제공 가이드: 04. Guidebook_Molding-압축됨.pdf. 앞서 확인한 파일별 표준화 설명을 반영하였다.',
'LogisticRegression 공식 문서\nhttps://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html',
'RandomForestClassifier 공식 문서\nhttps://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html',
'OneClassSVM 공식 문서\nhttps://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html',
'이상 탐지 가이드\nhttps://scikit-learn.org/stable/modules/outlier_detection.html',
'StratifiedGroupKFold 공식 문서\nhttps://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html',
'F beta score 공식 문서\nhttps://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html'
]: p(s)
file=OUT/'LR_RF_OCSVM_학습전략_통합.docx'
doc.save(file)
check=Document(file)
assert len(check.tables)==7
assert len([p for p in check.paragraphs if p.style.name=='Heading 1'])>=10
with __import__('zipfile').ZipFile(file) as z:
    assert z.testzip() is None
verification={'document':file.name,'source_hashes':sources,'data_summary':stats,'models_trained':False,'existing_folds_modified':False,'tables':len(check.tables),'visual_review':False,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
(OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print('Document created and structural checks passed.')
