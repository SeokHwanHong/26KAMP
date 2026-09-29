from pathlib import Path
import copy,json,re,hashlib
ROOT=Path(__file__).resolve().parents[1]
for dataset in ['cn7','rg3']:
    cells=[];sources={}
    def markdown(text):
        cells.append({'cell_type':'markdown','id':f'{dataset}-intro-{len(cells)}','metadata':{},'source':text.splitlines(keepends=True)})
    markdown(f'''# {dataset.upper()} OCSVM 통합 분석

기준 모델부터 확장 탐색, 공정 변수 중요도, 파생변수 비교, 최종 test 평가와 실패 분석까지 순서대로 실행하는 통합본입니다.

## 분석 순서

1. **A. 기준 모델:** 9개 설정 비교와 최초 평가
2. **B. 확장 탐색:** 300개 설정, 공정 변수군 비교, 컬럼·변수군 중요도
3. **C. 최종 분석:** 6개 변수 시나리오 재최적화, 모델 선택, test 평가·분포 시각화, 실패 원인 분석

각 부의 설정 셀은 해당 단계의 탐색 범위와 함수를 다시 정의합니다. 새 커널에서 위에서 아래로 실행하세요. 각 부의 해석은 그 단계의 결과이며 **최종 결과는 C부**를 기준으로 확인합니다. 저장된 그래프와 표도 포함되어 있습니다.

전처리와 고정 분할은 기존 `data/processed/{dataset}/conservative`를 사용합니다. 모델은 정상 학습 데이터만 사용하고, 선택은 개발 검증 데이터로 수행합니다. 이미 관찰했던 test의 후속 평가이므로 새로운 독립 검증으로 해석하지 않습니다. 비라벨 추론은 수행하지 않습니다.

통합본 실행 결과는 `output/ocsvm_{dataset}_integrated/` 아래 `baseline`, `exploration`, `feature_scenarios`에 저장됩니다. 기존 노트북과 산출물은 이전 실험 기록으로 보존합니다.
''')
    for prefix,stage,title in [('A','conservative','기준 모델'),('B','exploration','확장 탐색과 변수 중요도'),('C','feature_scenarios','파생변수·최종 평가·실패 분석')]:
        src=ROOT/f'modeling/ocsvm_{dataset}_{stage}.ipynb'
        raw=src.read_bytes();n=json.loads(raw);sources[str(src.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
        markdown(f'## {prefix}. {title}\n\n원본: `{src.name}`. 아래 셀은 해당 분석 단계의 코드와 해석입니다.\n')
        for index,original in enumerate(n['cells']):
            c=copy.deepcopy(original);c['id']=f'{dataset}-{prefix.lower()}-{index}'
            c.setdefault('metadata',{})['integration_source']={'notebook':src.name,'cell_index':index}
            s=''.join(c['source'])
            if c['cell_type']=='markdown':
                s=re.sub(r'^# (.+)$',rf'### {prefix}부 개요 — \1',s,flags=re.M)
                s=re.sub(r'^## (\d+)\. ',rf'### {prefix}.\1. ',s,flags=re.M)
                s=re.sub(r'^## (?!#)(.+)$',rf'### {prefix}부 \1',s,flags=re.M)
            else:
                replacements={
                    'output/ocsvm_conservative':f'output/ocsvm_{dataset}_integrated/baseline',
                    'output/ocsvm_development_exploration':f'output/ocsvm_{dataset}_integrated/exploration',
                    f'output/ocsvm_{dataset}_feature_scenarios':f'output/ocsvm_{dataset}_integrated/feature_scenarios'}
                for old,new in replacements.items():s=s.replace(old,new)
                compile(s,f'{src.name}:{index}','exec')
            c['source']=s.splitlines(keepends=True);cells.append(c)
    dest=ROOT/f'modeling/ocsvm_{dataset}_integrated.ipynb'
    notebook={'cells':cells,'metadata':copy.deepcopy(n['metadata']),'nbformat':4,'nbformat_minor':5}
    notebook['metadata']['integration']={'source_sha256':sources,'execution_order':['A','B','C']}
    assert len({c['id'] for c in cells})==len(cells)
    dest.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    print(dest.name,len(cells),'cells')
