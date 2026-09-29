from pathlib import Path
import copy,json,re,hashlib
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'modeling/ocsvm_cn7_feature_scenarios.ipynb'
target=ROOT/'modeling/ocsvm_cn7_integrated.ipynb'
s=json.loads(source.read_text(encoding='utf-8'));n=json.loads(target.read_text(encoding='utf-8'))
assert not any(c.get('metadata',{}).get('selection_cycle') for c in n['cells'])
for index,cell in enumerate(s['cells']):
 if not cell.get('metadata',{}).get('selection_cycle'):continue
 c=copy.deepcopy(cell);c['id']=f'cn7-c-{index}'
 c['metadata']['integration_source']={'notebook':source.name,'cell_index':index}
 text=''.join(c['source'])
 if c['cell_type']=='markdown':
  text=re.sub(r'^### ', '#### ',text,flags=re.M)
  text=re.sub(r'^## (\d+)\. ',r'### C.\1. ',text,flags=re.M)
 c['source']=text.splitlines(keepends=True);n['cells'].append(c)
n['cells'][0]['source'].append('\n\n**현재 최종 작업 단계: C.16~C.20.** 중요도 기반 후보 구성 → 후보별 4-fold 재검증 → 개발 정상 전체 재학습 → test 후속 평가를 포함합니다. C.1~C.15는 이전 실험 기록입니다. 새 산출물은 `output/ocsvm_cn7_integrated/feature_scenarios/selection_cycle/`에 저장됩니다.\n')
n['metadata']['integration']['source_sha256'][str(source.relative_to(ROOT)).replace('\\','/')]=hashlib.sha256(source.read_bytes()).hexdigest()
assert len({c['id'] for c in n['cells']})==len(n['cells'])
target.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
# 검증 실행기는 통합본의 C부 설정과 새 단계만 실행합니다.
runner=(ROOT/'tmp/run_cn7_selection_cycle.py').read_text(encoding='utf-8')
runner=runner.replace("modeling/ocsvm_cn7_feature_scenarios.ipynb","modeling/ocsvm_cn7_integrated.ipynb")
runner=runner.replace("if index not in [2,3,4,6] and not cell['metadata'].get('selection_cycle'):continue", "origin=cell['metadata'].get('integration_source',{})\n if not (origin.get('notebook')=='ocsvm_cn7_feature_scenarios.ipynb' and origin.get('cell_index') in [2,3,4,6]) and not cell['metadata'].get('selection_cycle'):continue")
(ROOT/'tmp/run_cn7_integrated_selection.py').write_text(runner,encoding='utf-8')
print('Added 11 cells to CN7 integrated notebook.')
