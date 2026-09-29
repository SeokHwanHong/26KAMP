from pathlib import Path
import json,copy
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'modeling/ocsvm_cn7_integrated.ipynb'
n=json.loads(p.read_text(encoding='utf-8'))
backup=ROOT/'output/ocsvm_cn7_integrated/before_expanded_search.ipynb'
if not backup.exists():backup.write_bytes(p.read_bytes())
config="""# 모든 분석 단계에서 같은 확대 탐색 범위를 사용합니다.
NU_VALUES = sorted(set([0.001, 0.0025] + np.round(np.arange(1, 21) * 0.005, 3).tolist() + [0.125, 0.15, 0.20, 0.25, 0.30, 0.40]))
GAMMA_MULTIPLIERS = sorted(set(np.geomspace(0.03, 1.0, 15).tolist() + [0.003, 0.005, 0.01, 0.02, 2.0, 3.0, 10.0]))
GRID_SIZE = len(NU_VALUES) * len(GAMMA_MULTIPLIERS)
print(f'확대 탐색: nu {len(NU_VALUES)}개 × gamma 배수 {len(GAMMA_MULTIPLIERS)}개 = {GRID_SIZE}개 조합')
"""
for c in n['cells']:
 s=''.join(c['source'])
 if c['cell_type']=='code':
  lines=s.splitlines(keepends=True)
  if any(line.startswith('NU_VALUES =') for line in lines):
   s=''.join(config if line.startswith('NU_VALUES =') else '' if line.startswith('GAMMA_MULTIPLIERS =') else line for line in lines)
  s=s.replace('output/ocsvm_cn7_integrated/', 'output/ocsvm_cn7_integrated/expanded_search_20260929/')
  s=s.replace("{candidate}/300개 설정", "{candidate}/{GRID_SIZE}개 설정")
  s=s.replace("'300設定'.replace('設定','개 설정')", "f'{GRID_SIZE}개 설정'")
  s=s.replace('assert len(search) == 1800 and len(fold_search) == 7200','assert len(search) == len(SCENARIOS)*GRID_SIZE and len(fold_search) == len(SCENARIOS)*GRID_SIZE*4')
  s=s.replace('ticks=[0,7,14]','ticks=sorted(set([0,len(part.columns)//2,len(part.columns)-1]))')
  s=s.replace("ax.set_yticks([0,9,19],[f'{part.index[i]:.3f}' for i in [0,9,19]])", "ax.set_yticks([0,len(part.index)//2,len(part.index)-1],[f'{part.index[i]:.4g}' for i in [0,len(part.index)//2,len(part.index)-1]])")
  s=s.replace("'fit_count':7200", "'fit_count':len(SCENARIOS)*GRID_SIZE*4")
  c['outputs']=[];c['execution_count']=None
  compile(s,c['id'],'exec')
 else:
  s=s.replace('300개','616개').replace('300個','616개').replace('7,200회','14,784회').replace('300개×4-fold×6개','616개×4-fold×6개')
  if c['id'] in ['cn7-b-15','cn7-c-31','cn7-c-42']:
   s='> **확대 전 실험의 해석 기록입니다. 아래 숫자는 새 탐색 결과가 아닙니다. 최신 결과는 실행 표와 마지막 요약을 확인하세요.**\n\n'+s
  if c['id']=='cn7-a-1':
   s='### A.1. 설정과 라이브러리\n\n이제 A·B·C 모두 동일한 확대 탐색을 사용합니다. nu 28개 × gamma 배수 22개 = 616개 조합입니다. 실제 gamma는 배수를 상수 제거 후 입력 수 d로 나눕니다.\n'
 c['source']=s.splitlines(keepends=True)
n['cells'][0]['source']=('''# CN7 OCSVM 통합 분석 — 확대 탐색

**현재 탐색 범위는 모든 단계에서 616개 조합입니다.**

- nu: 0.001~0.40, 총 28개. 기존 0.005~0.10 구간의 0.005 간격을 유지하고 양쪽을 확장했습니다.
- gamma 배수: 0.003~10, 총 22개. 기존 로그 간격 15개를 유지하고 작은 배수와 큰 배수를 추가했습니다. 실제 gamma=배수/d입니다.
- nu는 불량 비율을 뜻하지 않습니다. 큰 nu도 정상 영역 경계에 대한 민감도 비교 후보로 검토합니다.
- 각 단계·변수 후보마다 같은 범위로 탐색합니다. 탐색 횟수와 그래프 눈금은 후보 수에 맞춰 자동 계산합니다.

A 기준 모델 → B 확장 탐색·중요도 → C 파생변수·최종 평가 → C.16~20 중요도 기반 재검증 순서입니다. 새 커널에서 위에서 아래로 실행하세요.

결과는 `output/ocsvm_cn7_integrated/expanded_search_20260929/`에 별도로 저장합니다. 기존 실험의 해석은 과거 기록으로 표시했습니다. 최신 수치는 실행 표와 마지막 요약을 기준으로 확인하세요. 반복 관찰한 test의 후속 평가이며, test로 파라미터나 변수를 선택하지 않습니다.
''').splitlines(keepends=True)
p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
runner=(ROOT/'tmp/run_ocsvm_integrated.py').read_text(encoding='utf-8')
runner=runner.replace("if isinstance(raw,str):raw=raw.encode()", "encoded=raw if isinstance(raw,str) else base64.b64encode(raw).decode()")
runner=runner.replace("base64.b64encode(raw).decode()},'metadata'", "encoded},'metadata'")
runner=runner.replace("out=ROOT/f'output/ocsvm_{dataset}_integrated'", "out=ROOT/f'output/ocsvm_{dataset}_integrated/expanded_search_20260929'")
(ROOT/'tmp/run_expanded_cn7.py').write_text(runner,encoding='utf-8')
print('Updated grid and dynamic search counts.')
