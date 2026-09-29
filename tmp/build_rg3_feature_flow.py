from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
for filename in ['create_cn7_feature_scenarios.py','add_cn7_final_test.py','add_cn7_test_visuals.py','add_cn7_failure_analysis.py']:
    source=(ROOT/'tmp'/filename).read_text(encoding='utf-8').replace('cn7','rg3').replace('CN7','RG3')
    source=source.replace('현재 test 위험 3개가 모두 누락된 원인을 진단합니다.', '고정 test의 미탐지와 오탐을 진단합니다.')
    source=source.replace('최종 모델의 21차원 변환 공간', '최종 모델의 선택된 변환 공간')
    source=source.replace('np.full(3,2)', "np.full(int(test_pred.y_true.eq(1).sum()),2)")
    exec(compile(source,filename,'exec'),{'__file__':str(ROOT/'tmp'/filename)})
p=ROOT/'modeling/ocsvm_rg3_feature_scenarios.ipynb'
n=json.loads(p.read_text(encoding='utf-8'))
for cell in n['cells']:
    s=''.join(cell['source'])
    s=s.replace('기존 개발 분석의 제외 효과를 재최적화로 확인', '공정 주기 변수 제외의 효과를 재최적화로 확인')
    s=s.replace('이번 결과만으로 최종 모델을 교체하지 않습니다.', '아래 최종 평가 단계에서 개발 결과만으로 모델을 확정합니다.')
    cell['source']=s.splitlines(keepends=True)
p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
print(p)
