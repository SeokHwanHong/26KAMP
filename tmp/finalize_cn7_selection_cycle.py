from pathlib import Path
import json,uuid,base64
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'modeling/ocsvm_cn7_feature_scenarios.ipynb'
n=json.loads(p.read_text(encoding='utf-8'))
text='''### 이번 실행의 해석

평균 검증 AP 기준으로 선택된 `stable_all`은 원변수 8개와 파생변수 3개(배압 상대편차, 사출·전환압력 상대편차, 속도·사출압력 상호작용), 총 11개 입력입니다. 명칭의 stable은 중요도가 4-fold 중 3개 이상에서 양수라는 후보 생성 규칙을 뜻하며 **모델 성능이 안정적이라는 의미는 아닙니다.**

평균 AP는 기존 S1의 0.37997에서 0.38942로 소폭 높아졌지만 fold 표준편차는 0.08272에서 0.28304로 커졌습니다. 개발 OOF F2는 0.400에서 0.320으로 낮아졌고 FP는 2개에서 73개로 늘었습니다. 평균 fold AP와 통합 OOF AP는 서로 다른 집계이며, fold마다 적합한 모델의 원시 점수 척도가 같다고 보장되지 않습니다.

고정 test 후속 평가에서는 TP=0, FP=12, FN=3, TN=107, AP=0.03989, ROC-AUC=0.53221입니다. 기존 S1 후속 평가의 TP=0, FP=0보다 오탐이 늘었으며, **변수 선택이 실제 탐지 성능을 개선했다는 근거는 없습니다.** 선택 규칙이 평균 AP 하나였기 때문에 이 후보가 선택된 사실과, 운영상 적절한 모델이라는 판단은 구분해야 합니다.

Test 결과로 기존 모델에 다시 되돌려 선택하거나 임계값을 조정하지 않았습니다. 다음 실험에서 안정성·오탐량을 선택 제약에 넣으려면 개발 데이터에서 규칙을 먼저 정하고 검증해야 합니다. 이미 사용한 test에 대한 반복 개선 결과는 독립 검증으로 보고하지 않습니다.
'''
n['cells'].append({'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{'selection_cycle':True},'source':text.splitlines(keepends=True)})
images=0;codes=0
for c in n['cells']:
 if not c['metadata'].get('selection_cycle'):continue
 if c['cell_type']=='code':
  codes+=1;assert c['execution_count'] is not None
  compile(''.join(c['source']),'selection_cell','exec')
  for o in c['outputs']:
   assert o.get('output_type')!='error' and o.get('name')!='stderr'
   if 'image/png' in o.get('data',{}):
    assert base64.b64decode(o['data']['image/png']).startswith(b'\x89PNG\r\n\x1a\n');images+=1
assert codes==5 and images==3
p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
summary=ROOT/'output/ocsvm_cn7_feature_scenarios/selection_cycle/analysis_summary.md'
summary.write_text(summary.read_text(encoding='utf-8')+'\n\n'+text,encoding='utf-8')
print('Verified',codes,'executed cells and',images,'embedded PNG plots.')
