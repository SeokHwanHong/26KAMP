from pathlib import Path
helper=Path(__file__).with_name('create_logistic_strategy.py')
exec(compile(helper.read_text(encoding='utf-8').split('stats=[]')[0],str(helper),'exec'))
OUT=ROOT/'output/ocsvm_modeling_report';OUT.mkdir(exist_ok=True)
sources={}
def readcsv(relative):
    path=ROOT/relative;sources[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)
def readjson(relative):
    path=ROOT/relative;sources[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8'))
def figure(relative,width=6.1):
    path=ROOT/relative;sources[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    doc.add_picture(str(path),width=Inches(width))
def f(v):return f'{float(v):.3f}'
prep={n:readjson(f'data/processed/{n}/conservative/preprocessing_manifest.json') for n in ['cn7','rg3']}
baseline={n:readcsv(f'output/ocsvm_conservative/{n}/evaluation.csv') for n in prep}
search={n:readcsv(f'output/ocsvm_development_exploration/{n}/parameter_search.csv') for n in prep}
importance={n:readcsv(f'output/ocsvm_development_exploration/{n}/permutation_importance_summary.csv') for n in prep}
scenarios=readcsv('output/ocsvm_cn7_feature_scenarios/scenario_comparison.csv')
final=readjson('output/ocsvm_cn7_feature_scenarios/final_test/final_manifest.json')
test=readcsv('output/ocsvm_cn7_feature_scenarios/final_test/test_metrics.csv').iloc[0]
types=readcsv('output/ocsvm_cn7_feature_scenarios/final_test/failure_analysis/pattern_type_performance.csv')
ranks=readcsv('output/ocsvm_cn7_feature_scenarios/final_test/failure_analysis/missed_risk_ranks.csv')
verification=readjson('output/ocsvm_cn7_feature_scenarios/final_test/failure_analysis/verification.json')

doc.add_heading('OCSVM 공정 위험 탐지 모델링 결과',0)
p('CN7 및 RG3 전처리부터 CN7 최종 평가와 실패 분석까지')
p('작성 기준 2026년 9월 28일')
p('동일 공정 입력에서 불량이 한 번이라도 관측되면 위험 조건으로 관리한다는 목적에 따라 OCSVM을 학습했다. 정상 패턴만으로 정상 영역을 추정하고, 그 영역을 벗어나는 정도를 위험 점수로 사용했다. CN7·RG3의 초기 모델과 확장 탐색을 수행한 뒤 CN7에서 파생변수 시나리오, 최종 test 평가 및 실패 원인 분석까지 진행했다.')
h('주요 결과')
p(f'CN7에서 개발 평균 AP가 가장 높았던 구성은 형체·전체 주기 변수를 제외한 21개 입력이었다. 평균 AP는 {f(scenarios.iloc[0].mean_AP)}였지만, 개발 OOF 임계값을 고정 test에 적용하면 위험 3개를 모두 놓쳤다. ROC-AUC {f(test.ROC_AUC)}와 AP {f(test.AP)}는 순위 정보와 최종 판정 성능을 구분해서 해석해야 함을 보여준다.')
p('실패 진단에서 개발 데이터의 불량 전용 패턴은 3개 모두 탐지했지만 정상·불량 공존 패턴은 8개 중 1개만 탐지했다. Test 위험 3개는 모두 공존 패턴이었다. 정상 영역 안에 있는 위험 조건을 탐지하려는 목표와 OCSVM의 이상 탐지 가정 사이의 차이가 주요 한계로 나타났다.')
table(['범위','완료 내용'],[
('CN7 및 RG3','패턴 통합, 고정 test와 개발 4-fold, 초기 9개 설정 탐색, 300개 확장 탐색, 7개 입력 시나리오 및 변수 중요도'),
('CN7 추가 분석','6개 파생변수 시나리오별 재최적화, 최종 test 평가, 2차원 시각화와 실패 원인 분석'),
('미수행','RG3 파생변수 최종 모델, 비라벨 운영 추론, LR·RF 비교')],[1.35,4.8])
p('본 문서의 개발 결과는 반복적인 모델·변수 선택에 사용되었다. 초기 test 결과를 확인한 뒤 후속 실험을 진행했으므로 CN7 최종 결과는 같은 holdout의 후속 평가이며 새 독립 검증이 아니다.')

page('1 데이터 전처리와 고정 분할')
p('ID를 제외한 24개 공정 입력이 정확히 같은 행을 하나의 패턴으로 통합했다. 대표 라벨은 y_g=max(y_i)로 정했다. 정상은 0, 불량 이력이 있는 위험 조건은 1이다. 정상·불량 관측 횟수, 원본 ID와 패턴 연결표는 보존하고 모델 입력에서는 제외했다.')
table(['데이터','라벨 원본','통합 정상','통합 위험','비라벨 원본','비라벨 패턴'],[
(n.upper(),prep[n]['labeled_rows_before'],prep[n]['normal_patterns'],prep[n]['risk_patterns'],prep[n]['unlabeled_rows_before'],prep[n]['unlabeled_unique_patterns']) for n in prep],[.75,.95,1,1,1.2,1.25])
p('비라벨도 동일 입력을 통합하고 원본 행으로 예측을 복원할 연결표를 저장했다. 비라벨 라벨은 생성하지 않았다. 입력 누락·무한대·중복과 원본 복원, CSV 저장 후 재읽기 일치를 검증했다.')
h('고정 test 20퍼센트와 개발 80퍼센트')
p('대표 라벨을 기준으로 stratified test 20%를 먼저 고정한 뒤, 개발 80%에 StratifiedKFold 4분할을 적용했다. seed는 42이다. 개발 패턴은 학습에 3회, 검증에 1회 참여하며 test는 선택 과정에 참여하지 않는다. 입력 인덱스는 전체 X_labeled.csv의 pattern_row 기준이다.')
table(['데이터','학습 수','검증 수','검증 위험 수','고정 test','test 위험'],[
('CN7','363','121','2 / 3 / 3 / 3','122','3'),('RG3','354','118','5 / 5 / 5 / 5','119','5')],[.75,.85,.85,1.55,1.1,1.05])
p('CN7 개발 전체는 484개 중 위험 11개, RG3는 472개 중 위험 20개이다. 같은 입력 패턴이 학습·검증·test 사이에 겹치지 않도록 검증했다. OCSVM의 정상 선택은 분할 후 학습 구간 안에서만 수행했다.')
h('입력 스케일과 처리 범위')
p('제공 CSV는 이미 파일별로 표준화되었다. 모델에서는 정상 학습 구간으로 상수 제거와 StandardScaler를 적합한 뒤 검증·test에 동일 변환을 적용했다. 비라벨과 라벨 자료의 기존 표준화 좌표계 정합은 아직 확인되지 않았으므로 비라벨 추론은 수행하지 않았다.')

page('2 모델 원리와 학습 및 평가 기준')
p('OCSVM은 정상 데이터가 주로 존재하는 영역의 경계를 추정한다. 일반적인 분류 SVM과 달리 불량 라벨을 경계 적합에 직접 사용하지 않는다. 다만 개발 검증에서 위험 라벨로 설정과 임계값을 선택하므로 전체 절차가 완전한 무라벨 실험은 아니다.')
p('목적함수   min ½||w||² + (1/(νn)) Σᵢ ξᵢ − ρ')
p('제약조건   wᵀφ(xᵢ) ≥ ρ − ξᵢ,   ξᵢ ≥ 0')
table(['기호','의미'],[
('n 및 xᵢ','정상 학습 관측 수와 입력 벡터'),('φ 및 w','커널 특징 공간 변환과 경계 벡터'),('ρ 및 ξᵢ','경계 위치와 관측별 경계 위반 정도'),('ν 또는 nu','이론적으로 학습 위반 비율 상한과 서포트 벡터 비율 하한에 관련된 값'),('γ 또는 gamma','RBF 유사도 exp(−γ||xᵢ−xⱼ||²)의 거리 민감도')],[1.1,5.05])
p('nu는 실제 불량률이나 새 데이터의 위험 예측 비율이 아니다. 위험 점수 s(x)=−decision_function(x)로 방향을 통일하고 s(x)>τ일 때 위험으로 판정했다. 기본 정상 경계는 s=0이며 점수는 확률이 아니다.')
h('선택과 보고 기준')
p('설정은 네 검증 fold의 평균 Average Precision(AP)으로 선택했다. 동률은 사전 정의한 후보 순서로 결정했다. 선택 모델의 OOF 점수를 모아 F2 최대 임계값을 정하고, 동률이면 Precision, 이후 높은 임계값을 우선했다. 개발 정상 전체로 재학습한 뒤 임계값을 바꾸지 않고 test에 적용했다.')
p('F1·F2, Precision·Recall, TP·FP·FN·TN, AP와 ROC-AUC를 보고했다. Fβ=(1+β²)PR/(β²P+R)이며 P는 Precision, R은 Recall이다. OOF로 선택한 임계값의 OOF 성능은 개발 성능이다. Fold별 AP 평균과 OOF 통합 AP는 계산이 달라 일치하지 않는다.')
p('모든 적합의 수렴 여부를 확인했다. 목적함수 크기 대신 서포트 벡터 수, 정상 학습 중 경계 밖 비율과 점수 분위수를 진단 기록으로 남겼다. 다른 적합 모델 사이 점수 척도 차이가 있을 수 있어 임계값 전이는 별도로 점검했다.')

page('3 기준 모델과 확장 파라미터 탐색')
p('첫 실험에서는 nu 0.01·0.05·0.10과 gamma 배수 0.1·1·10을 비교했다. gamma=배수/d이며 d는 정상 학습에서 상수 제거 후 변수 수이다. 데이터셋당 9개 설정×4-fold를 실행했다.')
rows=[]
for n,df in baseline.items():
    r=df.loc[df.partition.eq('test')].iloc[0]
    rows.append((n.upper(),f(r.AP),f(r.ROC_AUC),f(r.F1),f(r.F2),f'{int(r.TP)} / {int(r.FP)} / {int(r.FN)}'))
table(['초기 test','AP','ROC-AUC','F1','F2','TP / FP / FN'],rows,[.85,.8,1,.8,.8,1.8])
h('300개 설정으로 확대')
p('초기 탐색이 성능의 가능성을 충분히 확인하지 못할 수 있어 nu 0.005~0.10을 0.005 간격으로 20개, gamma 배수 0.03~1을 로그 간격으로 15개 설정했다. 각 데이터셋에서 1,200회 적합했다. 이 범위는 개발 결과에 근거한 탐색 범위이며 이론적 최적 범위는 아니다.')
rows=[]
for n,df in search.items():
    r=df.sort_values(['mean_AP','candidate_id'],ascending=[False,True]).iloc[0]
    rows.append((n.upper(),f(r.nu),f(r.gamma_multiplier),f(r.mean_AP),f(r.std_AP)))
table(['데이터','선택 nu','gamma 배수','평균 AP','fold 표준편차'],rows,[.9,1,1.35,1.3,1.5])
p('CN7은 gamma 배수가 하한 0.03에서 선택되었다. 범위 밖의 더 좋은 설정 가능성을 배제할 수 없다. RG3는 범위 내부에서 선택됐지만 최고 평균 AP 약 0.049로 구분 성능이 제한적이었다. 후보 수 증가에 따른 개발 검증 과적합 가능성 때문에 점수 하나만으로 최적성을 주장하지 않았다.')
h('입력 처리 시나리오')
p('전체 변수·정상 재표준화, 제공 스케일 유지, 충전·전환 제외, 계량·가소화 제외, 실린더·호퍼 온도 제외, 금형 온도 제외, 형체·전체 주기 제외의 7개를 비교했다. 이 단계는 전체 변수에서 선택한 nu와 gamma 배수를 고정한 민감도 실험이다. 변수 수 변화에 따라 실제 gamma도 변하므로 순수한 단일 변수 효과와 구분한다.')

page('4 변수와 공정 변수군 중요도')
p('선택된 전체 변수 모델을 고정하고 검증 컬럼을 섞었을 때의 AP 감소량을 permutation importance로 계산했다. 24개 컬럼과 5개 공정 변수군에 대해 fold마다 20회 반복하여 데이터셋당 2,320개 평가를 저장했다. 변수군은 같은 행 순열로 함께 섞어 그룹 내부 관계를 보존했다.')
p('평균 AP 감소량이 양수이면 해당 모델이 그 정보에 의존한 것이다. 음수는 섞은 뒤 점수가 좋아진 경우이다. 상관 변수는 서로 대체할 수 있으며, 정상 학습에서 제거된 상수 컬럼의 중요도는 0이다. 반복·fold 표준편차를 신뢰구간이나 인과관계로 해석하지 않았다.')
rows=[]
for n,df in importance.items():
    for _,r in df.loc[df.kind.eq('column')].sort_values('mean_AP_decrease',ascending=False).head(3).iterrows():
        rows.append((n.upper(),r.feature,f(r.mean_AP_decrease),f'{int(r.positive_folds)}/4'))
table(['데이터','컬럼','평균 AP 감소','양수 fold'],rows,[.7,3.25,1.2,.9])
p('CN7에서는 Max_Injection_Speed가 가장 큰 중요도를 보였고 네 fold 모두 양수였다. 충전·전환 변수군을 함께 섞은 AP 감소량은 약 0.255였다. 반면 RG3 상위 변수들은 평균 중요도가 작고 fold별 부호도 불안정했다. 성능이 낮은 모델에서 낮은 중요도가 나왔다고 공정 정보가 무의미하다고 결론 내릴 수는 없다.')
figure('output/ocsvm_development_exploration/cn7/group_importance.png')
p('CN7 공정 변수군 중요도. 오차막대는 fold 평균 중요도의 표준편차이다.')

page('5 CN7 파생변수 생성과 재최적화')
p('공정상 관련된 변수의 상대 편차, 중요도가 높았던 사출 속도와의 상호작용, 공정별 상관 구조의 압축을 가설로 삼았다. z는 각 fold의 정상 학습에서 표준화한 값이다. z 차이·곱은 물리 단위의 차이·동력·에너지가 아니다.')
table(['구성','정의'],[
('상대 편차','사출−충전 시간, 최대−평균 RPM, 최대−평균 배압, 사출−전환 압력, 금형온도3−4'),
('온도 분포','배럴 z 평균·표준편차·범위, 금형 z 평균'),
('상호작용','속도×충전시간, 속도×사출압력, 속도×쿠션, 계량시간×평균배압'),
('공정별 PCA','각 변수군 최대 2개 PC 점수와 재구성 RMSE. 모든 학습은 정상 구간에만 적합')],[1.3,4.85])
p('6개 시나리오 각각에서 300개 설정을 재탐색하여 총 7,200회 적합했다. 파생변수는 새 측정 정보를 만드는 것이 아니라 RBF 거리와 변수별 비중을 바꾸는 효과도 포함한다. 모든 특징 변환을 fold 안에서 적합하고 test는 탐색에 사용하지 않았다.')
table(['시나리오','평균 AP','F2','TP / FP / FN'],[(r.description,f(r.mean_AP),f(r.F2),f'{int(r.TP)} / {int(r.FP)} / {int(r.FN)}') for _,r in scenarios.iterrows()],[3.1,.9,.8,1.35])
p('최고 구성은 파생변수 추가가 아닌 형체·전체 주기 제외(S1)였다. Cycle_Time, Clamp_Close_Time, Clamp_Open_Position을 제외하여 입력 21개를 사용했다. 네 fold 모두 전체 원변수 기준보다 AP가 높았다. 파생변수 후보의 효과가 검증되었다고 일괄 해석하지 않는다.')

page('6 CN7 최종 모델과 고정 test 평가')
p(f'개발 평균 AP로 S1을 선택했다. nu={final["nu"]}, gamma 배수={final["gamma_multiplier"]:.6f}, 실제 gamma={final["gamma"]:.8f}이다. 개발 OOF에서 선택한 임계값은 {final["threshold"]:.8f}이다. 개발 정상 473개에서 전처리와 OCSVM을 다시 적합했다.')
table(['지표','고정 test 결과'],[
('평가 패턴','122개 중 위험 3개'),('TP / FP / FN / TN',f'{int(test.TP)} / {int(test.FP)} / {int(test.FN)} / {int(test.TN)}'),
('Precision / Recall','0 / 0'),('F1 / F2','0 / 0'),('ROC-AUC / AP',f'{f(test.ROC_AUC)} / {f(test.AP)}'),
('정확도 / 균형 정확도','97.54% / 50.00%')],[2.9,3.25])
p('모두 정상으로 판정했기 때문에 정확도는 높지만 위험 탐지는 실패했다. 위험 예측이 0건인 Precision은 정의되지 않으며 구현 규칙에 따라 0으로 보고했다. 모델 저장 후 다시 읽어 같은 점수가 산출되는지도 확인했다.')
figure('output/ocsvm_cn7_feature_scenarios/final_test/test_evaluation.png')
p('최종 모델의 PR·ROC 곡선과 혼동행렬. 순위 지표가 양호해 보여도 선택 임계값에서 탐지가 되지 않을 수 있다.')
h('평가의 한계')
p('초기 모델의 test 결과가 이미 관찰되었고, 이후 개발 분석과 이번 최종 평가가 같은 holdout을 사용했다. 알고리즘 선택에는 개발 지표만 사용했지만, 연구 과정 전체에서 완전히 새로운 독립 test라고 설명하지 않는다. 추가 모델 개선에는 별도 검증 자료가 필요하다.')

page('7 CN7 점수와 입력 분포 시각화')
figure('output/ocsvm_cn7_feature_scenarios/final_test/test_distribution.png')
p('PCA는 최종 모델 입력으로 변환한 개발 정상에서만 적합했다. 두 축 설명분산 합계는 약 52.6%이며, 회색은 개발 정상 배경이다. 실제 위험은 빨간 X, 예측은 모든 test가 정상으로 표시된다. OCSVM은 군집 모델이 아니므로 이 그림은 군집 번호가 아닌 입력 분포와 판정 결과를 나타낸다.')
p('2차원 위치가 겹친다는 사실만으로 고차원에서 구분 불가능하다고 단정하지 않는다. 오른쪽 아래 점수 그림은 세 위험이 모두 고정 임계값 아래에 놓임을 직접 보여준다. ID는 시간 정보로 사용하지 않았다.')

page('8 실패 원인 분석')
table(['평가 구간','위험 유형','위험 수','탐지','누락'],[(r.partition,r.pattern_type,int(r.risk_patterns),int(r.detected_risk),int(r.missed_risk)) for _,r in types.loc[types.risk_patterns.gt(0)].iterrows()],[1.65,1.65,.95,.95,.95])
p('공존(conflicting)은 원본에서 정상·불량이 함께 관측된 패턴, 불량 전용(defect_only)은 불량만 관측된 패턴이다. 개발에서는 불량 전용 3개를 모두 찾았지만 공존 패턴은 8개 중 1개만 찾았다. Test의 위험 3개는 모두 공존 패턴이었다. 따라서 개발 탐지 성과가 모든 위험 유형에서 균등한 것은 아니었다.')
h('임계값만의 문제인가')
p('개발 임계값에서 TP=0, FP=0이었다. 기본 경계 0에서도 TP=0, FP=2였고 네 fold 모델에서도 세 test 위험 점수는 모두 음수였다. 재학습의 점수 이동만으로 실패를 설명하기 어렵다. 최종 모델에서 점수가 더 낮아진 변화도 있어 전이 영향을 완전히 배제하지는 않는다.')
table(['위험 패턴','위험 순위','점수','이 점수까지 탐지 시 FP'],[(r.pattern_id,int(r.rank_from_most_risky),f(r.risk_score),int(r.FP_if_including_this_score)) for _,r in ranks.iterrows()],[1.75,1,1,2.4])
p('세 위험을 모두 포함하려면 정상 40개도 경보해야 했다. 이는 test 라벨을 사용한 사후 비용 설명이며 새 운영 임계값 추천이 아니다. 세 위험 모두 개별 21개 변수에서 정상 학습 범위 안에 있었고 최근접 정상 거리 백분위도 약 76%, 25%, 68%였다. 이는 정상 영역과의 유사성을 보조하는 증거이며 인과 증명은 아니다.')
h('현재 근거가 지지하는 해석')
p('보수적으로 정의한 위험 조건은 정상 영역 밖에만 존재하지 않는다. OCSVM은 불량 전용의 뚜렷한 이상 패턴에는 반응했지만 정상과 유사한 공존 위험에 취약했다. 이는 불량 우선 집계가 잘못되었다는 뜻이 아니다. 동일 공존 패턴의 정상 행이 학습에 남아 발생한 누수도 아니며, 통합 후 서로 다른 패턴 사이의 겹침 문제이다.')

page('9 재현 기록과 후속 검증')
h('검증한 항목')
p('원본 해시, 전처리·분할 해시, 행 정렬, 학습·검증·test의 패턴 분리, 정상 전용 변환 적합, 모든 후보 수렴, OOF 전체 행 포함, 모델 저장·복원 예측 일치를 확인했다. 실패 진단 과정에서 최종 모델·임계값·test 예측은 변경하지 않았다.')
p(f'실패 진단의 저장 모델은 scikit-learn 1.9.1, 진단 실행 환경은 {verification.get("runtime_sklearn","1.9.0")}으로 버전 차이 경고가 있었다. 저장 예측과 재계산 점수의 최대 절대 차이는 {verification.get("recomputed_score_max_absolute_difference",0)}였다. 동일 점수가 확인되었더라도 재현 실행은 저장 버전에 맞춘 커널을 사용하는 것이 적절하다.')
h('다음 단계')
p('동일 개발 분할에서 LR·RF로 위험 라벨을 직접 학습하고 공존 패턴 Recall을 별도로 비교한다. 전체 위험 Recall과 함께 위험 유형별 결과, 오탐 수, 점검 후보 비중을 보고한다. 현장 점검량 또는 오탐 허용량이 정해지면 개발 데이터에서 그 제약 아래 임계값을 정하는 별도 실험을 설계한다. 향후 새로운 로트·시간대 자료로 검증한다.')
h('주요 실행 파일과 결과 위치')
for s in [
'전처리: preprocessing/preprocess_cn7_conservative.ipynb 및 preprocess_rg3_conservative.ipynb',
'분할: preprocessing/split_conservative_data.ipynb',
'기준 모델: modeling/ocsvm_cn7_conservative.ipynb 및 ocsvm_rg3_conservative.ipynb',
'확장 탐색과 중요도: modeling/ocsvm_cn7_exploration.ipynb 및 ocsvm_rg3_exploration.ipynb',
'CN7 파생변수·최종 평가·실패 분석: modeling/ocsvm_cn7_feature_scenarios.ipynb',
'결과: output/ocsvm_conservative, output/ocsvm_development_exploration, output/ocsvm_cn7_feature_scenarios',
'최종 진단: output/ocsvm_cn7_feature_scenarios/final_test/failure_analysis'
]:p(s)
h('기술 참고')
p('scikit-learn OneClassSVM 공식 문서: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html')
p('scikit-learn 모델 저장과 버전 호환: https://scikit-learn.org/stable/model_persistence.html')
file=OUT/'OCSVM_모델링_전체과정_및_실패분석.docx'
doc.save(file)
check=Document(file)
assert len(check.inline_shapes)==3
assert len(check.tables)==12
with __import__('zipfile').ZipFile(file) as z:assert z.testzip() is None
(OUT/'verification.json').write_text(json.dumps({'source_hashes':sources,'document_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'tables':len(check.tables),'figures':len(check.inline_shapes),'visual_review':False,'models_retrained':False},ensure_ascii=False,indent=2),encoding='utf-8')
print('Document generated and structural checks passed.')
