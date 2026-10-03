# Supplementary re-audit (scratch, not project code): same checks as audit_logistic_runs.py,
# but score comparison uses atol=1e-12 and requires identical binary predictions.
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,joblib
ROOT=Path('/home/claude/work/KAMP');sys.path.insert(0,str(ROOT/'03.modeling/common'))
from pipeline_runtime import data,digest,read,library,score,metrics
out=[]
for ds in ['cn7','rg3']:
    folder=sorted((ROOT/'output/logistic'/ds).iterdir())[-1]
    config=read(folder/'run_config.json');audit=read(folder/'audit.json');sel=read(folder/'selection_manifest.json');choice=sel['choice']
    r=dict(dataset=ds,run=folder.name)
    r['code_hash_current']=config['code_sha256']==digest(ROOT/'03.modeling/models/logistic_regression.py')
    r['runtime_hash_current']=config['runtime_code_sha256']==digest(ROOT/'03.modeling/common/pipeline_runtime.py')
    bad=[n for n,h in audit['artifact_sha256'].items() if digest(folder/n)!=h];r['audit_artifacts_mismatch']=bad
    ns=library(ds);top=[];rows=0;maxerr=0
    for ch in pd.read_csv(folder/'threshold_search.csv',float_precision='round_trip',chunksize=30000):
        rows+=len(ch);maxerr=max(maxerr,float(np.abs(ch.F1-2*ch.TP/(2*ch.TP+ch.FP+ch.FN)).max()))
        top.append(ns['rank_candidates'](ch).head(1))
    w=ns['rank_candidates'](pd.concat(top)).iloc[0]
    r.update(rows=rows,f1_formula_maxerr=maxerr,winner_reproduced=bool(w.candidate_id==choice['candidate_id'] and w.threshold==choice['threshold']))
    x,y,assign,dev,test,_=data(ds);ok=True
    for a in read(folder/'fold_audit.json'):
        tr,va=set(a['train']),set(a['validation']);ok&=tr.isdisjoint(va) and (tr|va)==set(dev) and tr.isdisjoint(test) and y.loc[list(tr)].sum()>0
    r['fold_roles_ok']=bool(ok)
    p=pd.read_csv(folder/'selected_oof_predictions.csv',float_precision='round_trip')
    r['oof_once']=bool(p.pattern_row.is_unique and set(p.pattern_row)==set(dev))
    r['oof_metrics']=metrics(p.label,p.prediction);r['oof_F1_matches']=r['oof_metrics']['F1']==choice['F1']
    b=joblib.load(folder/'model.joblib');saved=pd.read_csv(folder/'test_predictions.csv',float_precision='round_trip')
    s=score(b,x.loc[test]);r['test_score_max_abs_diff']=float(np.abs(s-saved.probability).max())
    r['test_pred_identical']=bool(((s>b['threshold']).astype(int)==saved.prediction).all())
    r['min_margin_to_threshold']=float(np.abs(s-b['threshold']).min())
    r['test_metrics_recomputed']=metrics(y.loc[test],saved.prediction);r['test_metrics_saved']={k:v for k,v in read(folder/'test_metrics.json').items() if k in('TP','FP','FN','TN','F1','AP','ROC_AUC')}
    r['choice']={k:choice[k] for k in ['scenario','C','class_weight','threshold','F1','TP','FP','FN','TN','F1_std']}
    out.append(r)
print(json.dumps(out,indent=1,ensure_ascii=False,default=str))
