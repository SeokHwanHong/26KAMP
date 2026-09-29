from pathlib import Path
helper=Path(__file__).with_name('create_logistic_strategy.py')
exec(compile(helper.read_text(encoding='utf-8').split('stats=[]')[0],str(helper),'exec'))
OUT=ROOT/'output/conservative_process_strategy'
OUT.mkdir(exist_ok=True)
counts=[]
hashes={}
for name in ['cn7','rg3']:
    source=ROOT/f'data/origin/moldset_labeled_{name}.csv'
    df=pd.read_csv(source)
    cols=json.loads((ROOT/f'data/processed/{name}/preprocessing_manifest.json').read_text(encoding='utf-8'))['feature_columns']
    label=[c for c in df if c not in cols and not c.startswith('Unnamed')][0]
    grouped=df.groupby(cols,dropna=False)[label].agg(['min','max','size'])
    counts.append([name.upper(),len(df),int((df[label]==1).sum()),len(grouped),int((grouped['max']==0).sum()),int((grouped['max']==1).sum())])
    hashes[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()

doc.add_heading('보수적 공정 관리를 위한 위험 조건 탐지 절차',0)
p('CN7 및 RG3 사출성형 데이터 | LR · RF · OCSVM | 2026년 9월 28일')
p('동일한 공정 입력에서 불량이 한 번이라도 관측되었다면 해당 조건을 위험 조건으로 관리한다. 정상 관측이 함께 존재하더라도 불량 이력을 우선하여 대표 라벨을 부여하고, 이러한 공정 조건을 탐지하는 모델을 구축한다. 목적은 개별 제품의 불량 여부를 단정하는 것이 아니라 보수적 공정 점검 대상을 선별하는 것이다.')
h('1 분석 목적과 전체 흐름')
p('원본 보존 및 패턴 통합 → test 분리 → 학습 구간 KS 분석과 파생변수 검토 → 모델 학습 및 검증 → 임계값 확정 → test 최종 평가 → 비라벨 입력 정합 확인 → 위험 후보 시각화 및 분석 순서로 진행한다.')
table(['단계','주요 결과물'],[
('전처리','입력 패턴별 대표 라벨과 원본 행 연결표'),
('데이터 분리','고정 seed와 train·validation·test 매핑'),
('EDA','KS 통계량, ECDF 및 분포 비교 그림'),
('모델링','LR·RF·OCSVM의 전처리와 학습 설정'),
('평가','F1·F2, Precision·Recall, AP·ROC-AUC, 혼동행렬'),
('비라벨 적용','모델별 위험 점수와 점검 후보 및 일치·불일치 분석')],[1.2,4.95])
p('위험 라벨 1은 관측 기간에 불량 이력이 있는 조건을 뜻한다. 라벨 0은 관측된 불량 이력이 없는 조건이며, 앞으로도 불량이 발생하지 않는다는 보장은 아니다. 평가 단위는 제품 행이 아니라 통합된 공정 입력 패턴이다.')
p('확정된 설계는 불량 우선 패턴 통합, 고정 test 20%, 개발 데이터 80% 내부 Stratified 4-fold, 세 모델 비교와 공통 F1·F2 평가이다. 각 교차검증 반복에서 전체 기준 train 약 60%, validation 약 20%, 고정 test 약 20%가 된다. 실제 분할과 모델 학습은 아직 수행하지 않았다.')

page('2 전처리와 불량 우선 패턴 통합')
p('원본 CSV는 보존한다. ID를 제외한 원본 24개 공정 입력값이 모두 같은 행을 하나의 패턴 g로 정의하고, 정상 0·불량 1 라벨에 대해 다음 대표값을 사용한다.')
p('y_g = max { y_i : i ∈ g }')
p('g는 동일 입력 행들의 집합, i는 원본 행, y_i는 원본 라벨, y_g는 통합 패턴의 대표 라벨이다. 모두 정상이면 0, 불량이 하나라도 있으면 1이다. 동일 라벨끼리 반복된 경우도 한 행으로 통합한다. 근사적으로 비슷한 입력을 합치는 작업은 포함하지 않는다.')
table(['데이터','원본 행','원본 불량','통합 패턴','라벨 0','라벨 1'],counts,[.8,.9,.95,1.1,1.1,1.1])
p('CN7은 원본 불량 17행이 14개 위험 패턴으로 통합된다. RG3는 25개 위험 패턴이 유지된다. 통합 이후 클래스 비율과 가중치는 이 패턴 기준으로 다시 계산한다. 원본 행 기준 불량 수나 비율을 그대로 사용하지 않는다.')
h('보존할 메타데이터')
p('pattern_id, source_id 목록, 정상 관측 수, 불량 관측 수, 총 관측 수, 상충 여부, 대표 라벨을 보존한다. 이 정보는 추적과 사후 분석에 사용하며, 라벨에서 계산한 불량 횟수·상충 여부 등을 모델 입력으로 넣지 않는다.')
p('관측 횟수가 다른 조건은 불량이 한 번이라도 발견될 기회도 다르다. 대표 라벨 정책은 유지하되, 결과 해석에서는 총 관측 수와 불량 반복 여부를 함께 확인한다. 이를 정상 라벨의 오류 수정이라고 설명하지 않고 보수적 위험 조건 정의라고 설명한다.')
h('입력 품질 확인')
p('결측, 자료형, 상수 컬럼, 컬럼 순서와 단위를 점검한다. 결측 대체, 상수 제거, 스케일러 등 데이터에서 값을 추정하는 전처리는 분할 후 학습 구간에만 적합한다. 라벨과 무관한 정확한 중복 패턴 정의 및 대표 라벨 집계는 분할 전에 고정한다.')

page('3 데이터 분리와 KS 분석')
h('고정 test와 개발 데이터 내부 4 fold')
p('통합 패턴에서 대표 라벨을 기준으로 stratified test 20%를 먼저 분리하고 고정한다. 나머지 개발 데이터 80%를 StratifiedKFold 4분할로 나눈다. test 분리와 fold 생성의 seed를 고정하며 초기값 42를 사용한다. 세 모델은 동일 test 및 fold 매핑을 재사용한다. 동일 패턴은 통합되었으므로 분할 사이에 반복되지 않아야 한다.')
table(['반복','Train 전체 약 60%','Validation 전체 약 20%','Test 전체 약 20%'],[
('1','B · C · D','A','고정 test'),('2','A · C · D','B','고정 test'),('3','A · B · D','C','고정 test'),('4','A · B · C','D','고정 test')],[.55,1.85,1.9,1.65])
table(['데이터','Train 위험 패턴','Validation 위험 패턴','Test 위험 패턴'],[
('CN7','약 8~9개','약 2~3개','약 3개'),('RG3','약 15개','약 5개','약 5개')],[.85,1.75,1.75,1.6])
p('위 수치는 예상값이며 실제 할당은 분할 후 확인한다. CN7 test에 위험 패턴이 3개라면 한 개의 탐지 차이가 Recall 약 33.3%p 차이를 만든다. 분할 자체가 잘못된 것은 아니지만 점수와 함께 TP·FP·FN을 반드시 보고한다.')
p('A~D는 개발 데이터 안의 네 fold이다. 개발 패턴은 검증에 한 번, 학습에 세 번 사용되며 test는 네 번의 학습·검증에 전혀 참여하지 않는다. 표본 수의 반올림 때문에 비율은 정확히 일치하지 않을 수 있다. 분할별 라벨 수와 패턴 교집합을 기록한다.')
h('KS 분석은 학습 구간에서 수행')
h('KS 통계량과 그래프')
p('KS로 특징을 선택한다면 각 반복의 학습 3개 fold에서 계산하고 선택 규칙을 해당 validation에 적용한다. 개발 전체의 분포 그림은 탐색용으로 구분하고 test를 선택에 사용하지 않는다. D_j = sup_x |F_0,j(x) − F_1,j(x)|는 변수 j의 두 클래스 누적분포 차이 중 최댓값이다.')
p('변수별 D와 표본 수를 기록하고 ECDF, 히스토그램 또는 상자그림을 함께 확인한다. ECDF에서 가장 큰 수직 차이가 D이다. KS가 크다는 이유만으로 해당 변수를 인과적 원인이나 충분한 분류 기준으로 단정하지 않는다. 기존 전체 데이터 EDA의 영향은 새 분할로 완전히 사라지지 않으므로 결과는 개발 단계의 근거로 설명한다.')

page('4 모델별 학습과 전처리')
table(['모델','적합 데이터','전처리와 주요 설정'],[
('LR','학습 라벨 0과 1','학습 기준 스케일링, L2 규제, 클래스 가중치'),
('RF','학습 라벨 0과 1','추가 스케일링 없이 balanced 가중치'),
('OCSVM','학습 라벨 0만','학습 정상 기준 스케일링, RBF 및 nu·gamma')],[.8,1.55,3.8])
p('LR은 선형 관계의 기준 모델, RF는 공정 조건의 비선형 결합, OCSVM은 불량 이력이 없는 조건의 영역에서 벗어나는 입력을 탐지한다. OCSVM의 학습 라벨 0은 패턴 통합 후 정상 집합이다. validation과 test에는 세 모델 모두 라벨 0과 1을 포함한다.')
h('손실 및 목적함수의 역할')
p('LR은 가중 로그 손실과 L2 규제를 최소화한다. RF는 노드별 가중 Gini 불순도가 감소하도록 분할한다. OCSVM은 정상 영역 경계와 위반 벌점의 균형을 학습한다. 이 값들은 척도가 달라 모델 간 직접 우열 비교에 사용하지 않고 학습 진단과 원리 설명에 사용한다.')
h('가중치와 스케일')
p('RF는 정상 대비 위험 클래스의 가중치를 n₀/n₁로 설정한다. n₀와 n₁은 매번 적합에 사용하는 통합 학습 패턴 수이다. class_weight="balanced"로 구현하며, 전체 데이터나 test 라벨 수로 계산하지 않는다. LR은 무가중·제곱근 가중·균형 가중 비교를 초기 제안으로 유지한다. OCSVM에는 이 클래스 가중치를 적용하지 않는다.')
p('제공 CSV는 이미 파일별로 표준화되어 있다. LR은 학습 패턴 전체, OCSVM은 학습 라벨 0 패턴에서 스케일러를 적합하고 같은 변환을 validation과 test에 적용한다. RF는 제공값을 그대로 사용한다. 상수 제거도 각 모델의 적합 구간에서 결정한다.')
h('특징 구성')
p('원래 공정 변수 모델을 기준선으로 유지하고, 공정 지식을 반영한 파생변수 모델을 비교한다. 기존 표준화 값에서 계산한 차이·비율은 물리 단위의 차이·비율과 다를 수 있다. 단위에 근거한 파생변수는 원자료 또는 변환 파라미터를 확인한 후 정의한다. PCA 등 학습이 필요한 변환은 학습 구간에서만 적합한다.')

page('5 검증을 통한 설정과 임계값 선택')
p('학습 데이터가 LR 계수, RF 분할, OCSVM 경계를 결정한다. validation은 이 내부 파라미터를 직접 학습시키는 데이터가 아니라 하이퍼파라미터·특징 구성·판정 임계값을 선택하는 데이터이다. 표준 구현의 세 모델을 모두 epoch별 validation 조기 종료 방식으로 설명하지 않는다.')
table(['모델','주요 조정 대상'],[
('LR','C, 클래스 가중치, 입력 특징 구성'),
('RF','max_depth, min_samples_leaf, max_features, 트리 수'),
('OCSVM','nu, gamma, 입력 특징 구성')],[1.0,5.15])
p('설정 후보는 실행 전에 좁게 정의하고, 개발 4-fold의 평균 검증 AP를 주 선택 지표로 사용하는 것을 제안한다. 각 후보를 네 번 적합하여 비교한다. 모든 후보와 결과를 기록하며, 선택에 사용한 교차검증 성능은 개발 성능으로 구분한다. 최종 성능은 고정 test에서 평가한다.')
h('공통 임계값 선택 규칙')
p('선택한 설정으로 개발 데이터 각 행이 validation일 때 얻은 OOF 점수를 모아 F2가 최대인 임계값을 고른다. 동률이면 Precision이 높은 값, 이후 더 높은 임계값을 선택한다. LR·RF는 위험 클래스 점수, OCSVM은 −decision_function을 사용하여 클수록 위험하게 방향을 맞춘다. 이 임계값을 고정 test에 맞춰 조정하지 않는다.')
p('교차검증 후 개발 전체로 재적합할 경우 점수 척도가 달라질 수 있다. 특히 OCSVM의 수치 임계값 전이 안정성을 기록한다. test 결과를 보고 임계값을 다시 맞추지 않는다. F2로 선택한 임계값의 F1은 균형 성능 보고값이며 F1 최적화 결과라고 부르지 않는다.')
h('파생변수 중요도와 수정')
p('검증 기반 permutation importance를 모델 공통 분석법으로 검토하고 RF 분할 중요도 및 LR 계수를 보조로 본다. 상관된 변수는 중요도가 분산될 수 있다. 중요도가 낮다고 즉시 제거하지 않고 원래 변수 기준선과 검증 성능을 비교한다. test를 보고 변수나 파생식을 수정하지 않는다.')
p('특징 구성·하이퍼파라미터·임계값을 확정한 뒤 개발 데이터 80% 전체로 최종 모델을 재학습한다. LR·RF는 개발 전체, OCSVM은 개발 중 라벨 0만 사용한다. 전처리도 해당 최종 적합 구간에서 다시 적합한다. 고정 test 20%에 확정된 파이프라인과 임계값을 적용해 최종 평가하고, 이 모델을 비라벨 추론에 사용한다.')

page('6 공통 평가와 시각화')
p('양성 클래스 1은 불량 이력이 있는 위험 공정 조건이다. TP는 탐지한 위험 조건, FP는 위험으로 분류한 라벨 0 조건, FN은 놓친 위험 조건, TN은 라벨 0으로 분류한 라벨 0 조건이다. 이는 관측된 대표 라벨 기준의 판정이다.')
p('Precision = TP/(TP+FP)     Recall = TP/(TP+FN)')
p('Fβ = (1+β²) Precision Recall / (β² Precision + Recall)')
p('β=1이면 F1, β=2이면 F2이다. 분모가 0인 경우 점수 0과 관련 건수를 함께 기록한다. weighted 평균 대신 위험 클래스를 양성으로 하는 binary F1·F2를 사용한다.')
table(['평가 항목','해석'],[
('F1 및 F2','탐지와 오탐 균형 및 누락을 강조한 평가'),
('Precision 및 Recall','후보 정확도와 위험 조건 검출률'),
('TP·FP·FN·TN','소표본에서 점수 차이를 실제 건수로 확인'),
('PR 곡선 및 AP','위험 점수 순위와 후보 정밀도·재현율 관계'),
('ROC 곡선 및 ROC-AUC','임계값 전반의 클래스 구분 능력'),
('모델별 목적함수','모델 내부 학습 진단이며 공통 순위 지표는 아님')],[1.7,4.45])
p('ROC와 PR은 0·1 판정이 아니라 연속 점수로 계산한다. 개발 4-fold의 fold별 값과 평균·표준편차, OOF 결과를 함께 보고한다. OOF로 임계값을 선택한 뒤 같은 OOF에서 계산한 F1·F2는 선택 과정이 반영된 개발 성능이다. OCSVM의 서로 다른 적합 모델 점수는 척도가 다를 수 있으므로 통합 순위 지표를 fold별 지표와 함께 해석한다. 최종 test 결과는 별도 표에 기록한다.')
h('그림 구성')
p('LR·RF는 X축 위험 예측값, Y축 비가중 행별 로그 손실로 그리며 실제 라벨을 색상으로 구분한다. RF의 로그 손실은 사후 진단값이다. OCSVM은 확률이 없으므로 점수 분포를 사용한다. 공통으로 PR·ROC 곡선, 혼동행렬, 공정 변수별 위험 후보 분포를 제공한다. 실제 시간 정보가 없는 ID는 시간축으로 해석하지 않는다.')

page('7 비라벨 데이터 적용과 공정 해석')
h('적용 전에 동일 좌표계 확인')
p('제공 자료의 labeled와 unlabeled가 각각 표준화되었다면 같은 물리 값도 다른 숫자로 표현될 수 있다. 두 파일 모두 평균 0·표준편차 1이라는 사실만으로 동일 입력 좌표계가 보장되지는 않는다.')
p('z_L = (x−μ_L)/σ_L     z_U = (x−μ_U)/σ_U')
p('x는 원래 물리 값, μ와 σ는 각 파일의 표준화 평균과 표준편차이다. 원자료 또는 표준화 파라미터를 확보하여 동일 기준으로 변환한 후 학습 파이프라인을 적용한다. labeled 스케일러를 별도로 표준화된 unlabeled 값에 다시 적용하는 것만으로는 이 문제가 해결되지 않는다.')
p('정합을 확인하지 못하면 비라벨 예측을 운영 판정으로 사용하지 않고 제약이 있는 탐색 결과로 한정한다. 입력 범위와 분포 차이는 별도로 기록한다. 이 확인은 모델 성능과 별개의 입력 유효성 점검이다.')
h('추론과 결과 저장')
p('최종 모델과 동일한 컬럼·순서·파생변수 계산·상수 제거·스케일링·임계값을 적용한다. 전체 변수 기준 모델과 특징 조정 모델을 혼동하지 않는다. 비라벨 원본 행에 결과를 연결하여 반복적으로 나타나는 위험 후보 조건도 추적한다.')
p('source_id, pattern_id, 모델명, 위험 점수, 임계값, 위험 후보 여부, 모델 간 일치 여부를 저장한다. 공정 변수 분포, 후보 비중, 점수 분포, 모델 간 불일치 사례를 시각화한다. 반복 출현 횟수와 위험 조건의 특성을 함께 검토하되 확인되지 않은 불량 건수로 세지 않는다.')
h('최종 해석')
p('비라벨 결과의 1은 학습된 불량 이력 조건과 유사하거나 정상 영역에서 벗어난 점검 후보이다. 실제 불량 확정이나 실제 불량률을 의미하지 않는다. 정답이 없는 비라벨 자료에서는 F1·F2를 산출할 수 없으며, 현장 확인이나 추후 라벨로 평가를 보완한다.')

page('8 실행 기록과 참고 자료')
p('본 문서는 목적과 절차를 정리한 설계서이다. 원본 라벨 변경, 패턴 통합 파일 생성, 분할 갱신, 모델 학습 또는 비라벨 예측은 수행하지 않았다. 원본 CSV의 행 수와 통합 후 클래스 수는 문서 생성 시 재계산하였다.')
h('실행 시 남길 기록')
p('원본 해시, 패턴 집계 규칙, 대표 라벨 및 원본 연결표, 분할 방식·seed·클래스 수, 전처리 적합 범위, 특징 목록과 파생식, 하이퍼파라미터 후보·선택값, 임계값 선택 기준, 검증·test 지표, 비라벨 스케일 확인 결과를 저장한다.')
h('참고 자료')
for s in [
'프로젝트 데이터: data/origin/moldset_labeled_cn7.csv 및 moldset_labeled_rg3.csv.',
'프로젝트 컬럼 정의: data/processed/cn7 및 rg3의 preprocessing_manifest.json.',
'제공 표준화 설명: 04. Guidebook_Molding-압축됨.pdf의 사전 검토 내용.',
'scikit-learn 전처리와 누수 방지\nhttps://scikit-learn.org/stable/common_pitfalls.html',
'scikit-learn Precision Recall\nhttps://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html',
'scikit-learn Permutation importance\nhttps://scikit-learn.org/stable/modules/permutation_importance.html',
'scikit-learn 모델 공식 문서\nhttps://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html\nhttps://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html\nhttps://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html'
]: p(s)
file=OUT/'보수적_공정관리_전체_분석_프로세스.docx'
doc.save(file)
check=Document(file)
assert len(check.tables)==7
all_text='\n'.join(p.text for p in check.paragraphs)
assert '고정 6:2:2' not in all_text and '권장 대안' not in all_text
assert counts[0][3:]==[606,592,14] and counts[1][3:]==[591,566,25]
with __import__('zipfile').ZipFile(file) as z: assert z.testzip() is None
(OUT/'verification.json').write_text(json.dumps({'source_hashes':hashes,'counts':counts,'models_trained':False,'data_modified':False,'visual_review':False,'tables':len(check.tables),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()},ensure_ascii=False,indent=2),encoding='utf-8')
print('Document generated; source counts and document structure verified.')
