"""Read-only real-data examples; Test remains historical, no model selection.

Also records the dataset-type evidence: does a simple range-history score rank risk
patterns better than chance on development OOF (permutation test)? Test is not used.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from pipeline_runtime import ROOT,data,fingerprints,read,write,write_guidance,new_id,feature_columns,bin_spec,bins,topk,fit_supervised,score,library
from decision_runtime import inspection_advice,usability


def signal_check(dataset,repeats=1000,seed=0):
    """Development-only OOF ranking evidence used to justify CN7형/RG3형 routing."""
    x,y,a,dev,_,_=data(dataset);fold=a.loc[dev,'cv_fold'].to_numpy();s=np.zeros(len(dev))
    for k in range(4):
        tr,va=dev[fold!=k],dev[fold==k];yt=y.loc[tr].to_numpy();base=yt.mean();acc=np.zeros(len(va))
        for c in feature_columns():
            spec=bin_spec(x.loc[tr,c]);t=pd.DataFrame(dict(b=bins(x.loc[tr,c],spec),y=yt)).groupby('b').y.agg(['sum','count'])
            rate=(t['sum']+base*10)/(t['count']+10)   # shrink sparse bins to the base rate
            acc+=pd.Series(bins(x.loc[va,c],spec)).map(rate).fillna(base).to_numpy()
        s[fold==k]=acc/len(feature_columns())
    yd=y.loc[dev].to_numpy();auc=roc_auc_score(yd,s);rng=np.random.default_rng(seed)
    perm=np.array([roc_auc_score(rng.permutation(yd),s) for _ in range(repeats)])
    p=float((1+np.sum(perm>=auc))/(1+repeats))
    return dict(dataset=dataset,development_patterns=len(dev),risks=int(yd.sum()),oof_auc=float(auc),
                permutation_p=p,top10=topk(yd,s,.1),
                route='model_selection' if p<.05 else 'distribution_monitoring',
                note='개발 OOF만 사용. 단순 구간 점수 기준이며 모델 성능 추정 아님')


def cn7_budget_comparison(fraction=.1):
    """Development OOF only: risks found within the same inspection budget for RF/LR/OCSVM."""
    x,y,a,dev,_,_=data('cn7');fold=a.loc[dev,'cv_fold'].to_numpy();yd=y.loc[dev].to_numpy()
    ns=library('cn7');rf=np.zeros(len(dev));oc=np.zeros(len(dev))
    choice=read(sorted(ROOT.glob('runtime/cn7/models/ocsvm-*/manifest.json'))[0])['choice']
    for k in range(4):
        tr,va=dev[fold!=k],dev[fold==k]
        rf[fold==k]=score(fit_supervised('cn7',x.loc[tr],y.loc[tr],kind='rf'),x.loc[va])
        normal=x.loc[tr][y.loc[tr].eq(0)];tf,vals=ns['fit_features'](normal,choice['scenario'])
        model=ns['fit_model'](vals,choice['nu'],choice['gamma_multiplier'])
        oc[fold==k]=-model.decision_function(ns['transform_features'](tf,x.loc[va]))
    lr_run=sorted((ROOT/'output/logistic/cn7').iterdir())[-1]
    lr=pd.read_csv(lr_run/'selected_oof_predictions.csv').set_index('pattern_row').loc[dev,'probability'].to_numpy()
    rows={}
    for name,s in [('rf_fixed_runtime',rf),('lr_selected_run',lr),('ocsvm_selected',oc)]:
        rows[name]=dict(oof_auc=float(roc_auc_score(yd,s)),**{f'top{int(f*100)}':topk(yd,s,f) for f in (.05,fraction,.2)})
    return dict(risks=int(yd.sum()),development_patterns=len(dev),models=rows,
        note='개발 OOF만 사용(Test 미사용). LR은 튜닝된 설정의 OOF라 낙관적일 수 있고 RF는 고정 설정이다.')


out=ROOT/'output/decision_review'/new_id('review');out.mkdir(parents=True)
budget=cn7_budget_comparison();write(out/'cn7_budget_comparison.json',budget)
signals=[signal_check(d) for d in ['cn7','rg3']]
write(out/'dataset_signal_check.json',signals)
x,y,_,dev,test,_=data('rg3')
products=x.loc[test].reset_index(drop=True)
products['record_id']=['historical-pattern-'+str(i) for i in test]
products['fingerprint']=fingerprints(products.iloc[:,:24])
guidance,rows=inspection_advice(x.loc[dev],y.loc[dev],products)
write_guidance(out,guidance)
levels=pd.Series([r['evidence_level'] for r in rows]).value_counts().to_dict()
write(out/'rg3_reinspection_examples.json',dict(scope='historical Test patterns, not new production or product defect probabilities',
      level_counts=levels,records=rows))
folder=sorted((ROOT/'output/logistic/cn7').iterdir())[-1]
m=read(folder/'test_metrics.json');policy=dict(min_recall=.5,min_precision=.2,max_FPR=.2)
reasons=usability(m,policy)
write(out/'cn7_current_lr_usability.json',dict(metrics=m,policy=policy,reasons=reasons,
    selected=None,note='historical followup only; no new three-model experiment or independent performance claim'))
lines=['데이터 유형 판정(개발 OOF 순열 검정, Test 미사용)']
for s in signals:
    lines.append(f"{s['dataset'].upper()}: OOF AUC {s['oof_auc']:.3f}, 순열 p {s['permutation_p']:.3f}, "
                 f"상위10% 검사 위험 {s['top10']['expected_TP']:.1f}/{s['risks']} → {s['route']}")
lines+=['',f"CN7 검사 예산 비교(개발 OOF, 위험 {budget['risks']}개)"]
for n,m in budget['models'].items():
    lines.append(f"{n}: AUC {m['oof_auc']:.3f}, 상위10% {m['top10']['expected_TP']:.1f}개, 상위5% {m['top5']['expected_TP']:.1f}개, 상위20% {m['top20']['expected_TP']:.1f}개")
lines+=['',
    'RG3 재검사 안내 (기존 Test 고유 패턴에 개발 참조 적용)',
    f'고유 패턴 {len(rows)}개 중 재검사 권고 {sum(r["reinspection_recommended"] for r in rows)}개, 근거 단계 {levels}.',
    '재검사 권고는 학습 범위 밖·정보 부족 근거. 구간 위험 이력은 참고 표시(예측력 미검증).',
    '이 수치는 검사 정책 동작 확인용이며 실제 불량 발견률이 아닙니다.',
    f'CN7 현재 LR: 선정 보류. 기준 미달 사유 {reasons}.',
    'LR·RF·OCSVM 최종 선정에는 동일한 별도 평가 자료가 필요하며, 선정에 쓴 평가 자료로는 승격할 수 없습니다.']
(out/'결과안내.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(out);print('\n'.join(lines))
