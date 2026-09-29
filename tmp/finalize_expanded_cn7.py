from pathlib import Path
import json,hashlib,base64
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'modeling/ocsvm_cn7_integrated.ipynb';n=json.loads(p.read_text(encoding='utf-8'))
for c in n['cells']:
 if c['cell_type']!='markdown':continue
 s=''.join(c['source'])
 s=s.replace('9개 설정의 평균 검증 AP','616개 설정의 평균 검증 AP')
 s=s.replace('nu 0.005~0.10의 20개 값, gamma 배수 0.03~1의 로그 간격 15개 값','nu 0.001~0.40의 28개 값, gamma 배수 0.003~10의 22개 값')
 s=s.replace('nu 20개, gamma 배수 15개','nu 28개, gamma 배수 22개')
 s=s.replace('이번 모델은 모든 test를 정상으로 판정하므로 예측상 위험 집단은 비어 있습니다. ','')
 s=s.replace('현재 test 위험 3개가 모두 누락된 원인을 진단합니다.','이번 test의 위험 탐지 및 누락 원인을 진단합니다.')
 s=s.replace('최종 모델의 21차원 변환 공간','최종 모델이 선택한 변환 공간')
 c['source']=s.splitlines(keepends=True)
out=ROOT/'output/ocsvm_cn7_integrated/expanded_search_20260929'
scenario=pd.read_csv(out/'feature_scenarios/scenario_comparison.csv').iloc[0]
selection=json.loads((out/'feature_scenarios/selection_cycle/selection_manifest.json').read_text(encoding='utf-8'))
test=pd.read_csv(out/'feature_scenarios/selection_cycle/test_metrics.csv').iloc[0]
summary=f'''## 확대 탐색 실행 결과

모든 단계에서 nu 28개 × gamma 배수 22개 = 616개 조합을 사용했습니다. C부 6개 시나리오 비교는 3,696개 후보와 14,784회 fold 적합을 수행했습니다.

- 시나리오 비교 최고: {scenario['scenario']}, 평균 검증 AP {scenario['mean_AP']:.6f}.
- 중요도 기반 후속 선택: {selection['candidate']}, 입력 {len(selection['names'])}개, 평균 검증 AP {selection['mean_AP']:.6f}.
- 선택 파라미터: nu={selection['nu']}, gamma 배수={selection['gamma_multiplier']}; 실제 gamma는 입력 수로 나눕니다.
- 후속 test: TP={int(test.TP)}, FP={int(test.FP)}, FN={int(test.FN)}, TN={int(test.TN)}, AP={test.AP:.6f}, F2={test.F2:.6f}.

결과는 확대 전의 해석과 구분해야 합니다. 선택은 개발 결과로만 수행했으며, 반복 관찰한 test에 대한 후속 평가입니다. 탐색 범위 확대 자체가 탐지 성능 향상을 보장하지 않습니다.
'''
n['cells'].append({'cell_type':'markdown','id':'expanded-search-summary','metadata':{},'source':summary.splitlines(keepends=True)})
codes=[c for c in n['cells'] if c['cell_type']=='code']
for c in codes:
 assert c['execution_count'] is not None
 compile(''.join(c['source']),c['id'],'exec')
 for o in c['outputs']:
  assert o.get('output_type')!='error' and o.get('name')!='stderr'
  if 'image/png' in o.get('data',{}):assert base64.b64decode(o['data']['image/png']).startswith(b'\x89PNG')
assert len(pd.read_csv(out/'feature_scenarios/all_parameter_search.csv'))==3696
assert len(pd.read_csv(out/'feature_scenarios/all_fold_metrics.csv'))==14784
for f in (out/'feature_scenarios/selection_cycle').glob('*_search.csv'):
 if 'fold_search' not in f.name:assert len(pd.read_csv(f))==616
p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
(out/'expanded_summary.md').write_text(summary,encoding='utf-8')
v=out/'integration_verification.json';record=json.loads(v.read_text());record['notebook_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();record['grid_size']=616;record['scenario_fold_fit_count']=14784
v.write_text(json.dumps(record,indent=2),encoding='utf-8')
print(summary)
