from pathlib import Path
import json,uuid
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
interpretations={
'cn7': '''전체 변수 모델은 nu=0.010, gamma=0.03/d에서 평균 검증 AP 0.332를 기록했습니다. gamma가 탐색 하한에 있으므로 더 작은 방향은 후속 검토 후보이며, 현재 범위 밖의 최적성을 주장하지 않습니다.

동일한 nu와 gamma 배수에서 형체·전체 주기 변수를 제외하면 평균 AP가 0.372로 높아졌습니다. 개발 OOF의 TP는 4개로 같고 FP는 4개에서 2개로 줄었습니다. 반면 위험 11개 중 7개는 여전히 누락합니다. 입력 제외와 d 변화의 효과가 함께 포함된 민감도 결과이며, 변수를 제거한 뒤 재최적화한 최종 성능은 아닙니다.

Max_Injection_Speed의 평균 AP 감소량은 0.180으로 가장 크며 네 fold 모두 양수입니다. 충전·전환 변수군의 감소량은 0.255입니다. 이 모델의 탐지가 해당 정보에 의존한다는 뜻이며 불량 원인이라는 결론은 아닙니다. 충전·전환을 제외한 평균 AP는 0.171로 낮아졌습니다.

후속 실험은 전체 변수 기준과 형체·전체 주기 제외 모델을 비교하는 방향이 유력합니다. 개발 결과를 근거로 선정된 후보이므로 독립적으로 검증된 개선이라고 부르지 않습니다.''',
'rg3': '''전체 변수 모델은 nu=0.075, gamma≈0.173205/d에서 평균 검증 AP 0.0485를 기록했습니다. 개발 위험 비율은 20/472≈0.0424입니다. 탐색 확대로도 위험 조건을 구분하는 성능은 제한적입니다.

형체·전체 주기 제외 시 평균 AP는 0.0499로 가장 높았지만 개선 폭이 작습니다. 전체 변수의 OOF F2 최적 임계값은 위험 20개를 모두 찾는 대신 정상 452개 중 450개를 위험으로 분류합니다. 높은 Recall만으로 좋은 탐지 모델이라고 해석할 수 없습니다.

컬럼별 평균 AP 감소량 상위는 Max_Back_Pressure 약 0.0062, Barrel_Temperature_6 약 0.0052, Average_Back_Pressure 약 0.0051입니다. 각 컬럼은 네 fold 중 두 fold에서만 양의 중요도를 보여 안정적인 핵심 변수로 단정하기 어렵습니다. 모델 성능이 낮은 상태의 낮은 중요도가 공정 정보 자체의 부재를 뜻하지는 않습니다.

후속 작업에서는 OCSVM의 추가 변수 제거만 반복하기보다 같은 개발 분할에서 LR·RF를 비교하여 위험 라벨을 직접 사용하는 학습이 도움이 되는지 확인하는 편이 유용합니다.'''
}
for name,body in interpretations.items():
    out=ROOT/'output/ocsvm_development_exploration'/name
    p=ROOT/f'modeling/ocsvm_{name}_exploration.ipynb'
    n=json.loads(p.read_text(encoding='utf-8'))
    text='## 7. 실행 결과 해석\n\n'+body+'\n\n이번 분석은 개발 데이터만 사용했습니다. 기존 test 예측과 모델 파일은 변경하지 않았습니다. 300개 설정×4-fold, 7개 시나리오×4-fold를 실행했고, 24개 컬럼과 5개 변수군의 중요도를 fold마다 20회 반복했습니다. 총 2,320개 permutation 평가를 기록했습니다.\n'
    n['cells'].append({'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{},'source':text.splitlines(keepends=True)})
    assert all(not any(o.get('output_type')=='error' for o in c.get('outputs',[])) for c in n['cells'])
    assert all(c['execution_count'] is not None for c in n['cells'] if c['cell_type']=='code')
    p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
    (out/'analysis_summary.md').write_text('# '+name.upper()+' OCSVM 개발 분석\n\n'+text,encoding='utf-8')
    print(name,'notebook summary and executed cells verified')
