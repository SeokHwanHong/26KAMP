"""Audit both RG3 conflict-label hypotheses without changing source labels.

Uses the Python standard library. Reuses an existing experiment only after
checking exact feature/label/split identity. No assumed label is ground truth.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/conflicting_label_scenarios/rg3_20261001'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    source=ROOT/'kamp_data/moldset_labeled_rg3.csv'
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    base=ROOT/'data/processed/rg3/conservative'
    features=read_csv(base/'X_labeled.csv')
    columns=list(features[0])
    rows=read_csv(source)
    groups={}
    for index,row in enumerate(rows):
        key=tuple(float(row[c]) for c in columns)
        groups.setdefault(key,[]).append((index+1,int(float(row['PassOrFail']))))
    assert len(columns)==24 and len(groups)==len(features)
    labels=read_csv(base/'y_labeled.csv')
    split=read_csv(base/'splits/split_assignments.csv')
    result=[]
    for i,feature in enumerate(features):
        key=tuple(float(feature[c]) for c in columns)
        members=groups[key]; ys=[y for _,y in members]
        conflict=len(set(ys))>1
        a=0 if conflict else ys[0]
        b=1 if conflict else ys[0]
        assert b==int(float(labels[i]['PassOrFail']))
        assert i==int(split[i]['pattern_row']) and b==int(split[i]['label'])
        result.append(dict(pattern_row=i,source_rows_1_based=[j for j,_ in members],
                           original_labels=ys,conflict=conflict,
                           assume_normal=a,assume_defect=b,partition=split[i]['partition']))
    counts=[]
    for scenario in ['assume_normal','assume_defect']:
        for partition in ['all','development','test']:
            part=[r for r in result if partition=='all' or r['partition']==partition]
            positives=sum(r[scenario] for r in part)
            counts.append(dict(scenario=scenario,partition=partition,patterns=len(part),
                               normal=len(part)-positives,defect=positives))
    assert sum(r['assume_normal'] for r in result)==0
    prior=ROOT/'output/stacking_complementarity/20261001_nested_v1/rg3'
    manifest=json.loads((prior/'selection_manifest.json').read_text(encoding='utf-8'))
    assert before==manifest['info']['source_sha256']
    predictions=read_csv(prior/'test_predictions.csv')
    test_ids={r['pattern_row'] for r in result if r['partition']=='test'}
    assert {int(r['row']) for r in predictions}==test_ids
    assert all(int(r['label'])==result[int(r['row'])]['assume_defect'] for r in predictions)
    prior_metrics=read_csv(prior/'test_metrics.csv')
    OUT.mkdir(parents=True,exist_ok=True)
    report=dict(source_sha256=before,raw_rows=len(rows),unique_patterns=len(groups),
                conflicting_patterns=sum(r['conflict'] for r in result),counts=counts,
                normal_scenario=dict(status='no_positive_labels',
                    note='Two-class defect learning and defect recall evaluation unavailable. All-normal output is tautological.'),
                defect_scenario=dict(status='identical_to_existing_pattern_experiment',
                    reused_results=str(prior.relative_to(ROOT)),test_metrics=prior_metrics),
                warning='These are assumed labels. Comparing their scores cannot determine which labels are genuine.')
    (OUT/'scenario_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'pattern_label_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['RG3 충돌 라벨 가정 비교',
           '원본 라벨을 변경하지 않고 동일한 24개 입력 패턴으로 비교함.',
           f'원본 {len(rows)}행 / 고유 입력 {len(groups)}개 / 충돌 {sum(r["conflict"] for r in result)}개',
           '',*[str(r) for r in counts],'',
           'A. 충돌을 모두 정상: 591개 패턴 전부 정상. 불량 정답이 0개이므로 불량 분류 학습·탐지 검증 불가.',
           '모두 정상으로 출력하면 가정된 라벨에 정확도 100%이지만 불량 탐지 성능의 증거가 아님.',
           'B. 충돌을 모두 불량: 정상 566개, 불량 가정 25개. 기존 위험 패턴 라벨과 완전히 일치.',
           '입력·라벨·테스트 행·원본 해시 일치를 확인했으므로 동일한 재학습은 반복하지 않고 기존 결과를 재사용.',
           'B의 테스트 상위 12개 검사: RF 0/5, 관계잔차 1/5, IF 0/5, 단순결합 1/5, 학습결합 0/5.',
           '결론: 가정 비교는 가능하지만 이 결과로 어느 라벨이 진짜인지 판정할 수 없음.',
           '테스트 정답까지 가정으로 바꾸면 그 가정에 대한 적합도만 측정하게 됨.',
           '진위를 비교하려면 라벨이 확인된 별도 평가 자료가 필요함.']
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    print(json.dumps({k:v for k,v in report.items() if k not in ['defect_scenario']},ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
