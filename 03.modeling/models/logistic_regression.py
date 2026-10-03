"""CN7/RG3 LR manual experiment, immutable runs, no deployment.
python 03.modeling/models/logistic_regression.py --dataset cn7
"""
import argparse
import json
from pathlib import Path
import sys
import warnings
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score,roc_auc_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from pipeline_runtime import (ROOT,Runtime,library,data,write,read,digest,new_id,fit_supervised,
                              score,metrics,range_guidance,write_guidance,topk)

SCENARIOS=['raw_alltrain_scaled','raw','legacy_without_cycle','pressure','plasticizing','thermal','domain_all','group_pca']
C_GRID=np.logspace(-4,3,15)
WEIGHTS=[None,'balanced']
THRESHOLDS=np.linspace(0.,1.,1001)


def run(dataset,run_id=None,state_root=None,scenarios=None):
    scenarios=scenarios or SCENARIOS
    ns=library(dataset);x,y,assignment,dev,test,source=data(dataset)
    out=ROOT/'output/logistic'/dataset/(run_id or new_id('lr'))
    out.mkdir(parents=True,exist_ok=False)
    config=dict(dataset=dataset,scenarios=scenarios,C_values=C_GRID.tolist(),weights=WEIGHTS,
                thresholds=THRESHOLDS.tolist(),source=source,code_sha256=digest(__file__),
                python=platform.python_version(),sklearn=sklearn.__version__,
                runtime_code_sha256=digest(ROOT/'03.modeling/common/pipeline_runtime.py'),
                test_role='previously inspected followup; not independent',label_target='risk history pattern')
    write(out/'run_config.json',config)
    rows=[];all_oof={};audits=[];candidate_id=0;failures=[]
    for scenario in scenarios:
        # Cache fitted transformations per fold, then vary only classifier parameters.
        cache=[]
        for fold in range(4):
            tr=dev[assignment.loc[dev,'cv_fold'].to_numpy()!=fold]
            va=dev[assignment.loc[dev,'cv_fold'].to_numpy()==fold]
            assert not(set(tr)&set(va) or set(tr)&set(test))
            seed=fit_supervised(dataset,x.loc[tr],y.loc[tr],scenario=scenario,C=1.)
            domain=seed['domain'];prep=seed['model'][:-1]
            rawtr=ns['transform_features'](domain,x.loc[tr]) if domain else x.loc[tr]
            rawva=ns['transform_features'](domain,x.loc[va]) if domain else x.loc[va]
            cache.append((fold,tr,va,prep.transform(rawtr),prep.transform(rawva)))
            audits.append(dict(scenario=scenario,fold=fold,train=tr.tolist(),validation=va.tolist(),
                  train_normal=int(y.loc[tr].eq(0).sum()),train_risk=int(y.loc[tr].sum()),
                  scaler_rows=len(tr),domain_reference_rows=int(y.loc[tr].eq(0).sum()) if domain else None))
        for C in C_GRID:
            for weight in WEIGHTS:
                from sklearn.linear_model import LogisticRegression
                folds=[];predictions=[];convergence=[]
                try:
                    for fold,tr,va,tx,vx in cache:
                        model=LogisticRegression(C=float(C),class_weight=weight,solver='lbfgs',max_iter=5000,tol=1e-6,random_state=42)
                        with warnings.catch_warnings():
                            warnings.simplefilter('error',ConvergenceWarning);model.fit(tx,y.loc[tr])
                        s=model.predict_proba(vx)[:,int(np.flatnonzero(model.classes_==1)[0])]
                        folds.append(dict(truth=y.loc[va].to_numpy(),scores=s,
                             AP_reference=float(average_precision_score(y.loc[va],s)),
                             ROC_AUC_reference=float(roc_auc_score(y.loc[va],s))))
                        predictions.append(pd.DataFrame(dict(pattern_row=va,fold=fold,label=y.loc[va].to_numpy(),probability=s)))
                        convergence.append(int(model.n_iter_[0]))
                except ConvergenceWarning as exc:
                    failures.append(dict(candidate_id=candidate_id,scenario=scenario,C=C,weight=weight,error=str(exc)))
                    candidate_id+=1;continue
                table=ns['threshold_trials'](folds,THRESHOLDS).assign(candidate_id=candidate_id,
                     scenario=scenario,C=float(C),class_weight='none' if weight is None else weight,
                     feature_count=max(c[3].shape[1] for c in cache),max_n_iter=max(convergence))
                table.to_csv(out/'threshold_search.csv',mode='w' if not rows else 'a',header=not rows,index=False)
                best=ns['rank_candidates'](table).iloc[0].to_dict();rows.append(best)
                all_oof[candidate_id]=pd.concat(predictions).sort_values('pattern_row')
                candidate_id+=1
        print(dataset,scenario,'completed',flush=True)
    write(out/'failed_candidates.json',failures);write(out/'fold_audit.json',audits)
    if not rows:raise RuntimeError('No converged candidates')
    comparison=pd.DataFrame(rows);ranked=ns['rank_candidates'](comparison)
    ranked.to_csv(out/'candidate_search.csv',index=False)
    choice=ranked.iloc[0].to_dict()
    write(out/'selection_manifest.json',dict(choice=choice,rule='pooled OOF F1, fold F1 std, feature count, candidate ID, threshold',
              test_used=False,config_sha256=digest(out/'run_config.json')))
    frozen=digest(out/'selection_manifest.json')
    oof=all_oof[int(choice['candidate_id'])].copy();oof['threshold']=choice['threshold']
    oof['prediction']=(oof.probability>choice['threshold']).astype(int)
    assert set(oof.pattern_row)==set(dev) and oof.pattern_row.is_unique
    assert np.isclose(metrics(oof.label,oof.prediction)['F1'],choice['F1'])
    oof.to_csv(out/'selected_oof_predictions.csv',index=False)
    bundle=fit_supervised(dataset,x.loc[dev],y.loc[dev],scenario=choice['scenario'],C=choice['C'],
                          class_weight=None if choice['class_weight']=='none' else choice['class_weight'])
    bundle.update(threshold=float(choice['threshold']),choice=choice,selection_sha256=frozen)
    joblib.dump(bundle,out/'model.joblib');scores=score(bundle,x.loc[test])
    np.testing.assert_array_equal(scores,score(joblib.load(out/'model.joblib'),x.loc[test]))
    predictions=scores>bundle['threshold']
    result=dict(**metrics(y.loc[test],predictions),AP=float(average_precision_score(y.loc[test],scores)),
                ROC_AUC=float(roc_auc_score(y.loc[test],scores)),top10=topk(y.loc[test],scores),
                always_normal=metrics(y.loc[test],np.zeros(len(test))),always_risk=metrics(y.loc[test],np.ones(len(test))))
    write(out/'test_metrics.json',result)
    pd.DataFrame(dict(pattern_row=test,label=y.loc[test].to_numpy(),probability=scores,
                     threshold=bundle['threshold'],prediction=predictions.astype(int))).to_csv(out/'test_predictions.csv',index=False)
    model=bundle['model'];domain=bundle['domain']
    names=np.array(domain['feature_names'] if domain else list(x))
    names=names[model[0].get_support()]
    pd.DataFrame(dict(feature=names,coefficient=model[-1].coef_[0],center=model[1].mean_,scale=model[1].scale_)).to_csv(out/'coefficients.csv',index=False)
    write(out/'coefficient_notes.json',dict(intercept=float(model[-1].intercept_[0]),
            meaning='regularized model log-odds; not causal effect or calibrated product defect probability'))
    guidance=range_guidance(x.loc[dev],y.loc[dev],x.loc[test],y.loc[test])
    write_guidance(out,guidance)
    assert digest(out/'selection_manifest.json')==frozen
    evidence={p.name:digest(p) for p in out.iterdir() if p.is_file()}
    write(out/'audit.json',dict(passed=True,artifact_sha256=evidence,oof_exactly_once=True,
               model_reload_identical=True,threshold_frozen=True,train_count=len(dev),
               searched_candidates=candidate_id,completed_candidates=len(rows),failures=len(failures)))
    # Verify all audit-bound artifacts immediately before registration.
    assert all(digest(out/name)==h for name,h in evidence.items())
    version=Runtime(dataset,state_root).save_candidate(bundle,x.loc[dev],y.loc[dev],
                source=dict(run=str(out),evidence_sha256=digest(out/'audit.json'),files=evidence))
    write(out/'registered_candidate.json',dict(version=version,active=False))
    summary=f'''{dataset.upper()} 로지스틱 회귀 결과
입력 구성: {choice['scenario']}, C={choice['C']}, class_weight={choice['class_weight']}, t={choice['threshold']}
개발 OOF F1={choice['F1']:.6f} (튜닝 점수)
Test F1={result['F1']:.6f}, TP={result['TP']}, FP={result['FP']}, FN={result['FN']}, TN={result['TN']}
원변수/도메인 {len(scenarios)}구성, 후보 {candidate_id}개, 성공 {len(rows)}개, 임계값 1001개.
Test는 이미 관찰한 후속 평가이며 독립 성능 아님. 후보 저장만 수행, 운영 승격 없음.
계수는 인과 효과 아님. 구간 안내는 위험 이력 패턴의 관측 통계이며 표본 부족을 표시함.
'''
    (out/'summary.txt').write_text(summary,encoding='utf-8');print(summary,flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['cn7','rg3','both'],default='both')
    args=parser.parse_args()
    for dataset in ['cn7','rg3'] if args.dataset=='both' else [args.dataset]:run(dataset)
