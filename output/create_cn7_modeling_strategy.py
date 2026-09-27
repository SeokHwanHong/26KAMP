from pathlib import Path
import zipfile, xml.etree.ElementTree as ET, hashlib, json
from xml.sax.saxutils import escape
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/cn7_modeling_strategy'; OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'output/cn7_eda_summary/CN7_EDA_summary.docx'
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
parts=[]
def p(text,style='Normal'):
    parts.append(f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr><w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>')
def title(n,text):
    if parts: parts.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    p(f'{n:02d}  {text}','Title')
def h(text): p(text,'Heading1')
def table(headers,rows,widths=None):
    widths=widths or [9300//len(headers)]*len(headers)
    parts.append('<w:tbl><w:tblPr><w:tblW w:w="9300" w:type="dxa"/><w:tblLayout w:type="fixed"/><w:tblBorders><w:bottom w:val="single" w:sz="4" w:color="D7E2EC"/><w:insideH w:val="single" w:sz="4" w:color="D7E2EC"/></w:tblBorders><w:tblCellMar><w:top w:w="95" w:type="dxa"/><w:left w:w="110" w:type="dxa"/><w:bottom w:w="95" w:type="dxa"/><w:right w:w="110" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>'+''.join(f'<w:gridCol w:w="{v}"/>' for v in widths)+'</w:tblGrid>')
    for i,row in enumerate([headers]+rows):
        parts.append('<w:tr><w:trPr><w:cantSplit/>'+('<w:tblHeader/>' if i==0 else '')+'</w:trPr>')
        for val,width in zip(row,widths):
            parts.append(f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/><w:shd w:fill="'+('183B56' if i==0 else ('F0F5F8' if i%2 else 'FFFFFF'))+'"/></w:tcPr><w:p><w:pPr><w:pStyle w:val="TableText"/></w:pPr><w:r><w:rPr>'+('<w:b/><w:color w:val="FFFFFF"/>' if i==0 else '')+f'</w:rPr><w:t>{escape(str(val))}</w:t></w:r></w:p></w:tc>')
        parts.append('</w:tr>')
    parts.append('</w:tbl>'); p('')

title(1,'CN7 모델링 설계 및 학습 전략')
p('랜덤포레스트 · One-Class SVM · 로지스틱 회귀','Subtitle')
p('기준: CN7_EDA_summary.docx, 전처리 산출물 및 scikit-learn 공식 문서 | 작성일: 2026-09-27','Small')
p('이 문서는 모델 실행 전의 설계안이다. 제안한 설정은 검증된 최적값이 아니며, 성능 수치는 아직 산출하지 않았다. 목표는 재현 가능한 비교와 실패 원인 구분이다.')
h('EDA에서 모델링으로 이어지는 근거')
table(['확인된 사실 [1]','설계에 반영할 결정'],[
('1,211행 / 불량 17행(1.40%) / 606개 입력 패턴','정확도를 주 지표로 사용하지 않는다. 동일 입력은 같은 폴드에 묶는다.'),
('불량 11행은 정상 11행과 입력이 완전히 동일','원본 라벨을 보존한다. 세 모델 모두 이 두 라벨에 서로 다른 점수를 줄 수 없다.'),
('불량 전용 6행은 3개 패턴; 사출 속도·쿠션 위치가 관측 정상 범위 밖','One-Class SVM의 정상 영역 이탈 탐지 가설을 검증하되, 3개 패턴에 대한 결과로 제한한다.'),
('온도·계량 관련 분포 차이; 온도 3·4 상관 0.988','선형 위험도와 비선형 조건 결합을 비교한다. 상관된 변수의 개별 중요도는 신중히 해석한다.'),
('고온·짧은 계량 시간 구간 375행에 불량 17행과 정상 358행','그 구간 자체를 불량 규칙으로 쓰지 않는다. 탐지율과 오탐을 함께 확인한다.')],[3400,5900])
h('세 모델이 답할 서로 다른 질문')
table(['모델','선정 근거와 검증 가설'],[
('로지스틱 회귀','적은 불량과 상관 변수에 L2 규제를 적용한 기준선. 전반적인 선형 위험도 변화로 충분한가?'),
('랜덤포레스트','온도·시간·압력의 비선형 관계와 조건 결합을 표현. 선형 기준선보다 추가 정보가 있는가?'),
('One-Class SVM','정상 데이터의 경계를 학습. 불량 전용 패턴은 정상 영역 밖에 있는가? 충돌 불량도 탐지 가능한가?')],[2300,7000])
p('권장 실행 순서: 로지스틱 회귀 → 랜덤포레스트 → One-Class SVM. 사용자 지정 세 모델은 모두 같은 외부 검증 폴드에서 비교한다.','Small')

title(2,'데이터 분리와 공통 파이프라인')
h('기존 3개 폴드를 외부 개발 검증으로 고정')
table(['폴드','학습 / 검증 행','검증 불량','검증 불량 전용 패턴'],[('0','807 / 404','5','1'),('1','807 / 404','6','1'),('2','808 / 403','6','1')],[1200,2900,2200,3000])
p('labeled_metadata.csv의 fold를 그대로 사용한다. 기존 분할은 고유 패턴의 유형을 층화한 StratifiedKFold이며, 일반 행 단위 분할이나 StratifiedGroupKFold를 새로 적용한 결과가 아니다. 24개 원본 입력의 완전 일치로 정의한 pattern_id는 전처리 전에 고정한다. [1]')
h('외부 3겹 × 내부 2겹의 선택 절차')
p('① 외부 검증 1개 폴드를 잠근다 → ② 나머지 학습 데이터의 고유 패턴을 정상 전용·충돌·불량 전용으로 나눈 뒤 내부 2겹으로 층화한다 → ③ 내부 검증 평균 AP로 설정을 고른다 → ④ 내부 예측으로 임계값 규칙을 선택한다 → ⑤ 외부 학습 전체로 재적합하고 외부 검증을 한 번 평가한다.')
p('외부 학습에는 불량 전용 패턴이 2개만 남으므로 내부는 2겹으로 제한한다. 내부 유형은 해당 외부 학습 라벨만으로 계산한다. random_state=42를 고정하고 모든 모델에 동일한 내부 분할을 준다. 내부 폴드에도 두 클래스가 존재하고 패턴 중복이 없는지 검사한다.')
h('전처리는 각 학습 구간 안에서만 적합')
table(['대상','적용 원칙'],[
('입력 / 타깃','X_labeled.csv의 24개 공정 변수, y_labeled.csv의 0=정상·1=불량. ID·패턴 유형·빈도·fold는 입력에서 제외.'),
('상수열 / 스케일','학습에서 상수인 열을 VarianceThreshold(0)로 제외. RF는 스케일링 없음; LR은 전체 학습 행, OCSVM은 정상 학습 행에만 StandardScaler 적합.'),
('행과 라벨','충돌 행·특이값 보존. 기본 실험은 중복 축약·SMOTE·언더샘플링·클리핑 없이 수행.'),
('비라벨 35,239행','학습·스케일 추정·파라미터 선택에서 제외. 모델 확정 후 점수 산출 대상으로 사용; 정상이라고 가정하지 않음.')],[2400,6900])
p('전체 라벨이 이미 EDA에 사용되었다. 중첩 검증은 추가 튜닝 누수를 줄이지만, EDA 선택 편향까지 제거하지는 않는다. 결과는 개발 성능이며 새 로트·시간대의 독립 시험이 필요하다. 제공된 표준화의 원래 적합 범위도 확인되지 않았다.','Small')

title(3,'랜덤포레스트: 조건 결합의 추가 정보')
h('학습 방식과 설정 근거')
p('정상과 불량을 함께 학습한다. 여러 트리의 결과를 평균하여 비선형 경계를 표현하되, 불량 표본이 적고 입력이 반복되므로 트리 깊이와 말단 표본 수를 제한한다. 복잡한 트리가 라벨 충돌을 해소할 수 있다고 기대하지 않는다. [2]')
table(['항목','제안 설정'],[
('기본 설정','n_estimators=500, criterion="gini", max_depth=3, min_samples_leaf=3'),
('고정 설정','max_features="sqrt", bootstrap=True, random_state=42, n_jobs=-1, oob_score=False'),
('내부 탐색: 8개','max_depth ∈ {3, 6}; min_samples_leaf ∈ {3, 10}; class_weight ∈ {None, "balanced"}'),
('예측 점수','predict_proba의 불량(클래스 1) 열. 클래스 순서를 classes_로 확인한다.')],[2700,6600])
h('손실함수에 해당하는 학습 기준')
p('각 노드에서 가중 클래스 비율 p를 사용한 Gini 불순도 G=1−Σₖpₖ²를 계산하고, 자식 노드의 가중 불순도를 가장 크게 줄이는 분할을 선택한다. 포레스트 전체에 하나의 교차엔트로피를 두고 경사하강하는 방식은 아니다. [2]')
p('balanced 가중치는 해당 적합 데이터의 n/(2nₖ)이다. 소수 불량의 영향력을 높이는 동시에 충돌 불량에도 큰 가중치를 주므로, 정상 동반 행의 오탐 증가를 확인해야 한다. class_weight=None도 함께 비교한다. [2]')
h('RF에서 특히 점검할 실패')
p('훈련 점수만 높고 외부 AP가 낮다면 패턴 암기·표본 부족을 의심한다. min_samples_leaf=10은 불량 전용의 작은 영역까지 평활화할 수 있으므로, 깊이를 늘리는 것보다 먼저 해당 3개 패턴의 누락을 확인한다.')
p('행 부트스트랩 기반 OOB는 동일 입력의 다른 복제 행이 트리 학습에 들어갈 수 있어 주 검증으로 사용하지 않는다. 기존 그룹 폴드가 일반화 평가의 기준이다.')
p('중요도는 외부 검증의 permutation importance를 보조로 확인한다. 온도 3·4처럼 상관이 높은 변수는 개별 중요도가 분산될 수 있으므로 함께 섞는 분석도 병기한다. 중요도는 제조 원인이나 개입 효과가 아니다.')
p('선정 판단: 로지스틱 회귀 대비 외부 폴드별 AP와 같은 경보 정책의 탐지/오탐이 개선되는지 확인한다. 특정 불량 전용 패턴 하나에서만 이득이 나면 보편적 개선으로 결론 내리지 않는다.','Small')

title(4,'로지스틱 회귀: 규제된 선형 기준선')
h('학습 방식과 설정 근거')
p('정상과 불량을 함께 학습하며 p(y=1|x)=sigmoid(b+βᵀx)를 추정한다. 변수별 분포 차이가 선형 점수로 연결되는지 검증하는 기준선이다. 불량 17행과 높은 변수 상관을 고려해 무규제 추정보다 L2 규제를 우선한다. [3, 4]')
table(['항목','제안 설정'],[
('파이프라인','학습 상수열 제거 → StandardScaler → LogisticRegression'),
('기본 설정','penalty="l2", C=0.1, solver="lbfgs", max_iter=5000, class_weight="balanced"'),
('내부 탐색: 6개','C ∈ {0.01, 0.1, 1}; class_weight ∈ {None, "balanced"}'),
('예측 점수','predict_proba의 클래스 1 열. 임계값은 0.5로 고정하지 않고 내부에서 선택한다.')],[2700,6600])
h('손실함수와 규제')
p('가중 이진 교차엔트로피 + L2 규제: L=−Σᵢaᵢ[yᵢ log pᵢ+(1−yᵢ)log(1−pᵢ)] + λ‖β‖²/2. aᵢ는 클래스 가중치이며, 절편은 이 규제 항에서 제외한다. 이 식은 목적의 형태를 나타낸다. 구현의 정규화 관례와 별개로 C가 작을수록 규제가 강하다. [3, 4]')
p('balanced는 불량 누락에 더 큰 학습 비용을 준다. 그러나 실제 공정 비용을 측정한 값은 아니며, 가중 학습의 출력 확률은 실제 불량률로 그대로 해석하지 않는다. 불량이 적어 별도 확률 보정도 기본 단계에서는 보류한다.')
h('해석 및 실패 점검')
p('계수의 부호와 폴드 간 변동을 기록한다. 표준화 후 계수는 학습 표준편차 단위의 조건부 연관이며, 다른 변수와의 상관 때문에 부호가 바뀔 수 있다. 작은 표본·반복 행·L2 규제가 있으므로 일반 회귀의 p값이나 인과적 효과로 보고하지 않는다.')
p('수렴 경고와 n_iter_를 확인한다. 학습과 검증이 모두 낮으면 선형 경계의 한계를 검토하고 RF와 비교한다. 훈련만 좋다면 C를 줄이는 방향을 우선 검토한다. 상호작용항 추가나 KS 상위 변수 선택은 첫 비교 후 별도 실험으로 분리한다.')
p('현재 설치 환경은 scikit-learn 1.2.1이다. 위 penalty="l2" 설정은 이 환경 기준이며, 실행 환경을 갱신할 경우 공식 문서의 API 변경을 확인하고 버전을 고정한다.','Small')

title(5,'One-Class SVM: 정상 영역의 경계')
h('학습과 검증의 라벨 사용을 구분')
p('모델과 전처리의 적합에는 해당 학습 구간의 정상(y=0)만 사용한다. 내부·외부 검증에는 정상과 불량을 모두 포함한다. 내부 검증 라벨로 설정을 선택하므로 전체 절차는 라벨을 전혀 쓰지 않는 순수 비지도 평가가 아니라, 정상 전용 학습에 라벨 기반 선택을 결합한 방식이다. [5, 6]')
p('충돌 패턴의 정상 행도 정상 학습에 유지한다. 충돌이라는 이유로 정상 행을 빼면 관측 정상의 정의가 바뀐다. 외부 검증에서 확인한 패턴 유형을 학습 행 선택에 사용하지 않는다.')
table(['항목','제안 설정'],[
('파이프라인','학습 정상 행 선택 → 정상에서 상수열 제거·StandardScaler 적합 → OneClassSVM'),
('기본 설정','kernel="rbf", nu=0.02, gamma="scale", tol=0.001'),
('내부 탐색: 9개','nu ∈ {0.01, 0.02, 0.05}; gamma ∈ {0.25/d, 1/d, 4/d}; d는 해당 학습 정상의 활성 변수 수'),
('점수 방향','s(x)=−decision_function(x). 클수록 이상. 기본 predict의 −1=이상, +1=정상; 원본 라벨 1/0과 다르다.')],[2700,6600])
h('목적함수와 nu의 의미')
p('min ½‖w‖² + (1/(νn))Σᵢξᵢ − ρ, 제약: wᵀφ(xᵢ) ≥ ρ−ξᵢ, ξᵢ≥0. n은 학습 정상 행 수, ξ는 경계 위반 허용량, φ는 커널의 특징 표현이다. RBF 커널은 exp(−γ‖x−x′‖²)이다. [5, 7]')
p('nu는 학습 오류 비율의 상한과 서포트 벡터 비율의 하한에 관련된 매개변수다. 불량률 1.40%를 뜻하지 않으며, nu=0.014로 자동 결정하지 않는다. gamma가 크면 정상 영역이 좁고 복잡해질 수 있다. [5]')
h('CN7에서 기대할 수 있는 것과 한계')
p('불량 전용 3개 패턴의 영역 이탈을 검증할 가치가 있다. 그러나 동일 입력을 가진 정상·불량은 동일 점수를 받는다. 충돌 불량을 검출하면 같은 입력의 정상도 함께 경보가 발생한다. 이상 점수는 불량 확률이 아니다.')
p('정상 분포가 복잡하면 경계의 희귀 정상에 오탐이 집중될 수 있다. 정상의 점수 분포·서포트 벡터 비율·nu/gamma 민감도를 기록한다. 비라벨 데이터를 정상 학습에 합치거나 이상 점수만으로 불량 라벨을 부여하지 않는다.','Small')

title(6,'평가 지표와 판정 임계값')
h('주 지표와 보조 지표')
table(['구분','기록할 결과'],[
('주 선택 지표','내부 폴드 AP의 단순 평균. 외부에서는 폴드별 AP와 평균·범위를 함께 보고한다. AP는 사다리꼴 PR-AUC와 구분해 표기. [8]'),
('보조 지표','ROC-AUC, precision, recall, F2, 정상 오탐률 FP/(FP+TN), 경보율, TP/FP/FN/TN 실제 행 수. [9]'),
('기준선','항상 정상인 분류기는 정확도 98.60%, 불량 recall=0. 전체 유병률 1.40%는 AP 해석의 참조값이며 실제 무작위 순위 AP와 정확히 같다는 뜻은 아님.'),
('집계 단위','원본 행 기준이 주 결과. 추가로 각 행에 1/해당 패턴 행 수 가중치를 주어 패턴 균등 지표를 계산; 충돌 라벨은 유지.')],[2600,6700])
h('임계값은 학습 안에서 결정하고 외부에는 고정 적용')
p('1. 내부 평균 AP로 모델 설정을 선택한다. 동률이면 더 단순하거나 더 강하게 규제된 설정을 우선한다. 외부 점수로 후보를 추가하지 않는다.')
p('2. 선택된 설정에서 경보 정책 α∈{1%, 2%, 5%, 10%, 20%}를 비교한다. 각 내부 학습 모델의 정상 학습 점수에서 tα=quantile(s정상, 1−α)를 구하고, 내부 검증에 s>tα를 적용한다. 내부 폴드 평균 F2가 가장 큰 α를 선택하고 동률이면 작은 α를 택한다.')
p('3. 외부 학습 전체로 재적합한 뒤 그 모델의 정상 학습 점수로 선택된 α의 tα를 다시 구한다. 외부 검증에는 이 임계값을 그대로 적용한다. RF/LR은 불량 확률, OCSVM은 음의 decision_function을 s로 사용한다.')
p('이 α는 임계값 생성 규칙이지 외부 오탐률 보장이 아니다. 학습 점수의 낙관성과 RF 점수 동률 때문에 실제 경보율이 달라질 수 있다. quantile 방식은 linear, 비교는 엄격한 >로 고정하고 α·t·검증 오탐률을 모두 저장한다.')
h('작은 표본에서 결과를 읽는 방법')
p('F2는 누락을 더 중시하는 잠정 운영 지표다. 대회 공식 지표와 오탐/미탐 비용이 아직 제공되지 않았으므로 확인 후 사전에 고정한다. 순위 성능(AP)과 임계값 성능을 분리하면 모델 실패인지 경보 정책 실패인지 구별할 수 있다.')
p('불량 5~6개인 폴드는 한 건으로 recall이 16.7~20%p 변한다. 3개 폴드의 표준편차를 신뢰구간으로 부르지 않는다. 특히 OCSVM은 폴드별 점수 척도가 달라 원점수를 합친 OOF AP로 순위를 결정하지 않는다. 임계값 적용 후의 혼동행렬은 합산 가능하다.','Small')

title(7,'실패 원인 분석과 개선 순서')
table(['관측할 현상','먼저 점검할 원인','다음 조치'],[
('충돌 불량 미탐 / 동반 정상 오탐','같은 입력에는 같은 점수라는 구조적 제약','라벨 변경보다 로트·시간·검수 및 측정 해상도 기록 확인'),
('불량 전용 3개 패턴만 잘 탐지','특이 사례에 의존하는 성능','충돌 11행과 불량 전용 6행의 TP/FN을 별도 보고'),
('AP는 양호하지만 경보가 과다','가중치·임계값·정상 경계의 문제','학습 설정과 경보 정책을 분리해 비교'),
('RF 훈련만 우수','깊은 트리의 암기 또는 누수','패턴 중복 검사 후 깊이/말단 제한 강화'),
('LR 계수 부호가 폴드마다 변동','상관 변수와 소수 불량의 영향','규제 강화, 상관 변수 묶음 해석'),
('OCSVM 정상 오탐 집중','희귀 정상·스케일·커널 경계','정상 하위 구간과 nu/gamma 민감도 점검'),
('비라벨에서 경보 급증','입력 범위 또는 표준화 차이 가능','변수별 범위·분위수 이탈 검사; 곧바로 불량 증가로 해석하지 않음')],[2400,3300,3600])
h('분석 결과를 남길 최소 산출물')
p('행별 외부 예측: source_id, pattern_id, fold, 실제 라벨, 모델, 점수, 임계값, 예측 라벨. 평가 후에만 패턴 유형을 결합한다. 충돌 구간은 정상/불량 양쪽을 포함해 보고한다.')
p('실험 명세: 원본·전처리 파일 해시, 라이브러리 버전, 입력 열, 모든 내부 후보 점수, 선택 설정, seed, α와 임계값, 상수열 마스크·스케일러·모델. 기존 전처리 파일은 덮어쓰지 않는다.')
h('실행 및 최종 학습 순서')
p('① 입력/라벨/메타데이터 행 정렬 및 그룹 중복 검사 → ② LR·RF·OCSVM 중첩 개발 검증 → ③ 공통 지표와 실패 유형 비교 → ④ 필요한 변경 한 가지만 추가 실험 → ⑤ 설계 확정 후 전체 라벨 데이터에서 그룹 3겹 내부 선택을 다시 수행하고 재학습한다.')
p('최종 RF/LR은 전체 라벨 행, 최종 OCSVM은 전체 정상 행으로 적합한다. 최종 설정과 경보 규칙을 확정한 후 비라벨 35,239행에 점수를 부여한다. 모델 선택에 사용한 개발 점수를 독립 시험 성능으로 다시 제시하지 않는다.')
p('초기 비교에는 8+6+9=23개 후보를 사용한다. 외부 3겹·내부 2겹 기준 후보 적합 138회와 외부 최종 적합 9회가 기본 규모다. 병렬화는 검색과 모델 양쪽을 동시에 최대화하지 않아 과도한 CPU 점유를 피한다.','Small')

title(8,'근거 자료와 적용 범위')
p('데이터 관련 사실은 기존 CN7 EDA 문서와 전처리 명세를 근거로 했다. 폴드 수, 후보 범위, α 정책, F2 활용 및 개선 순서는 CN7에 맞춰 제안한 설계이며 공식 문서가 보장하는 최적 설정이 아니다.')
refs=[
('[1] CN7 EDA 및 전처리','output/cn7_eda_summary/CN7_EDA_summary.docx; data/processed/cn7/preprocessing_manifest.json; fold_summary.csv'),
('[2] RandomForestClassifier','https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'),
('[3] LogisticRegression','https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'),
('[4] Logistic regression: 목적함수','https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression'),
('[5] OneClassSVM','https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html'),
('[6] Novelty and Outlier Detection','https://scikit-learn.org/1.7/modules/outlier_detection.html'),
('[7] SVM: Mathematical formulation','https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation'),
('[8] Average precision','https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html'),
('[9] Classification metrics','https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics')]
for label,url in refs:
    h(label); p(url,'Small')
p('온라인 문서 확인일: 2026-09-27. stable 문서는 갱신될 수 있으므로 실제 실행에서는 설치 버전과 API를 함께 기록한다. 이 문서에는 모델 학습 결과나 대회 성능 개선을 입증하는 주장이 포함되어 있지 않다.','Small')

styles=f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="{W}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Malgun Gothic" w:hAnsi="Malgun Gothic" w:eastAsia="맑은 고딕"/><w:sz w:val="21"/><w:color w:val="243746"/><w:lang w:val="ko-KR" w:eastAsia="ko-KR"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="100" w:line="280" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:keepNext/><w:spacing w:before="0" w:after="220"/></w:pPr><w:rPr><w:b/><w:sz w:val="34"/><w:color w:val="183B56"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:rPr><w:sz w:val="25"/><w:color w:val="187E86"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:keepNext/><w:spacing w:before="160" w:after="85"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="187E86"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Small"><w:name w:val="Small"/><w:rPr><w:sz w:val="18"/><w:color w:val="526777"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:pPr><w:spacing w:after="0" w:line="250" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="19"/></w:rPr></w:style></w:styles>'''
doc=f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{W}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:body>{''.join(parts)}<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter"/><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="950" w:right="1300" w:bottom="950" w:left="1300" w:header="400" w:footer="400"/></w:sectPr></w:body></w:document>'''
footer=f'''<?xml version="1.0" encoding="UTF-8"?><w:ftr xmlns:w="{W}"><w:p><w:pPr><w:jc w:val="right"/></w:pPr><w:r><w:rPr><w:sz w:val="16"/><w:color w:val="526777"/></w:rPr><w:t>CN7 · 모델링 설계  |  </w:t></w:r><w:fldSimple w:instr="PAGE"/></w:p></w:ftr>'''
rels='''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/><Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/></Relationships>'''
ct='''<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>'''
rootrels='''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
target=OUT/'CN7_modeling_strategy.docx'
entries={'[Content_Types].xml':ct,'_rels/.rels':rootrels,'word/document.xml':doc,'word/styles.xml':styles,'word/footer1.xml':footer,'word/_rels/document.xml.rels':rels}
for content in entries.values(): ET.fromstring(content)
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name,content in entries.items(): z.writestr(name,content.encode('utf-8'))
with zipfile.ZipFile(target) as z: assert z.testzip() is None
y=pd.read_csv(ROOT/'data/processed/cn7/y_labeled.csv').iloc[:,0]
assert len(y)==1211 and y.sum()==17
meta=pd.read_csv(ROOT/'data/processed/cn7/labeled_metadata.csv')
assert meta.groupby('pattern_id')['fold'].nunique().max()==1
assert meta.pattern_id.nunique()==606
verification={'source':str(SOURCE.relative_to(ROOT)),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'document':target.name,'labeled_rows':len(y),'defect_rows':int(y.sum()),'pattern_count':606,'validation':'DOCX ZIP and XML parsed; source counts and group isolation checked. Office rendering not performed.','models_trained':False}
(OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(target); print('Verified DOCX XML, ZIP, counts and pattern isolation. Bytes:',target.stat().st_size)
