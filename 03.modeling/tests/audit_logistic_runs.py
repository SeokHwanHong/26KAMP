from pathlib import Path
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'03.modeling/common'))
from pipeline_runtime import data,digest,read,write,library,score,metrics

results=[]
for ds in ['cn7','rg3']:
    # Latest COMPLETED run only: an aborted run leaves a folder without audit/registration files.
    folder=sorted(p for p in (ROOT/'output/logistic'/ds).iterdir()
                  if (p/'audit.json').exists() and (p/'registered_candidate.json').exists())[-1]
    config=read(folder/'run_config.json');audit=read(folder/'audit.json');choice=read(folder/'selection_manifest.json')['choice']
    assert config['code_sha256']==digest(ROOT/'03.modeling/models/logistic_regression.py')
    assert config['runtime_code_sha256']==digest(ROOT/'03.modeling/common/pipeline_runtime.py')
    assert all(digest(folder/n)==h for n,h in audit['artifact_sha256'].items())
    ns=library(ds);top=[];rows=0
    for chunk in pd.read_csv(folder/'threshold_search.csv',float_precision='round_trip',chunksize=30000):
        rows+=len(chunk)
        np.testing.assert_allclose(chunk.F1,2*chunk.TP/(2*chunk.TP+chunk.FP+chunk.FN),atol=1e-15,rtol=0)
        top.append(ns['rank_candidates'](chunk).head(1))
    winner=ns['rank_candidates'](pd.concat(top)).iloc[0]
    assert winner.candidate_id==choice['candidate_id'] and winner.threshold==choice['threshold']
    assert rows==240*1001
    x,y,assign,dev,test,_=data(ds)
    for a in read(folder/'fold_audit.json'):
        tr,va=set(a['train']),set(a['validation'])
        assert tr.isdisjoint(va) and (tr|va)==set(dev) and tr.isdisjoint(test)
        assert y.loc[list(tr)].sum()>0
    p=pd.read_csv(folder/'selected_oof_predictions.csv',float_precision='round_trip')
    assert p.pattern_row.is_unique and set(p.pattern_row)==set(dev)
    assert metrics(p.label,p.prediction)['F1']==choice['F1']
    b=joblib.load(folder/'model.joblib');saved=pd.read_csv(folder/'test_predictions.csv',float_precision='round_trip')
    # Probabilities may differ in the last bits across OS/BLAS/sklearn builds; decisions must not.
    current=score(b,x.loc[test]);np.testing.assert_allclose(current,saved.probability,rtol=0,atol=1e-12)
    np.testing.assert_array_equal((current>b['threshold']).astype(int),saved.prediction)
    assert b['threshold']==choice['threshold']
    result=dict(dataset=ds,run=str(folder.relative_to(ROOT)),passed=True,joint_trials_verified=rows,
                candidate_selection_reproduced=True,source_hashes_current=True,oof_exactly_once=True,
                fold_roles_verified=True,test_predictions_reproduced=True)
    results.append(result)
# Do not add files to the immutable LR run folder; store the audit with this test run.
out=ROOT/'output/operations_tests'/os.environ.get('KAMP_TEST_RUN_ID','manual-'+__import__('datetime').datetime.now().strftime('%Y%m%dT%H%M%S'));out.mkdir(parents=True,exist_ok=True)
write(out/'logistic_independent_audit.json',results)
print(json.dumps(results,indent=2))
