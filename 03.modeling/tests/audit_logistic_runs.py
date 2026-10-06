from pathlib import Path
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'03.modeling/common'))
from pipeline_runtime import data,digest,read,write,library,score,metrics,fit_supervised
from source_provenance import verify_source_digest
import hashlib
import ast

RUNTIME=ROOT/'03.modeling/common/pipeline_runtime.py'
PROVENANCE=ROOT/'03.modeling/common/provenance'


def training_definitions(path):
    """Every module statement except the operational class 'Operations' (batch/drift/CT/CD commands).
    LR training/evaluation uses only these definitions (Runtime, fit_supervised, data, score, metrics, ...)."""
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    return [ast.dump(n) for n in tree.body if not (isinstance(n,ast.ClassDef) and n.name=='Operations')]


def notebook_library_status():
    """The LR runs did not record the notebook library hash. Compare with the 2026-10-05 snapshot only to detect
    later changes; equality with run time is shown by the selected-config refit below, not by this hash."""
    snap=read(PROVENANCE/'notebook_library_snapshot.json')
    nb=read(ROOT/snap['notebook'])
    src=''.join(''.join(c['source'])+'\n' for c in nb['cells']
                if c['cell_type']=='code' and 'pipeline_library' in c.get('metadata',{}).get('tags',[]))
    return 'unchanged_since_20261005_snapshot' if hashlib.sha256(src.encode('utf-8')).hexdigest()==snap['library_code_sha256'] \
        else 'changed_since_20261005_snapshot'


def runtime_provenance(recorded):
    """The LR run recorded the runtime source hash at run time. The source may have changed later
    (2026-10-05: drift recovery in Operations). Keep the old record unchanged and verify it against the
    archived copy; separately prove the training definitions are identical to the current file."""
    try:
        match=verify_source_digest(RUNTIME,recorded)
        return 'current:'+match
    except AssertionError:pass
    archived=PROVENANCE/f'pipeline_runtime_{recorded[:12]}.py'
    assert archived.exists(),f'기록된 runtime 해시의 보관본 없음: {archived.name}'
    match=verify_source_digest(archived,recorded)
    assert training_definitions(archived)==training_definitions(RUNTIME),'학습 경로 정의가 보관본과 다름: LR 재실행 필요'
    return 'archived_source_verified:'+match+'+training_definitions_unchanged'

results=[]
for ds in ['cn7','rg3']:
    # Latest COMPLETED run only: an aborted run leaves a folder without audit/registration files.
    folder=sorted(p for p in (ROOT/'output/logistic'/ds).iterdir()
                  if (p/'audit.json').exists() and (p/'registered_candidate.json').exists())[-1]
    config=read(folder/'run_config.json');audit=read(folder/'audit.json');choice=read(folder/'selection_manifest.json')['choice']
    lr_source=verify_source_digest(ROOT/'03.modeling/models/logistic_regression.py',config['code_sha256'])
    runtime_source=runtime_provenance(config['runtime_code_sha256'])
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
    # Training path (2026-10-05): re-fit the selected configuration on the development patterns with the CURRENT
    # code (pipeline_runtime + notebook fit_features/transform_features) and compare with the saved model.
    # This checks the whole training path for the selected configuration, not every searched configuration.
    refit=fit_supervised(ds,x.loc[dev],y.loc[dev],scenario=choice['scenario'],C=choice['C'],
                         class_weight=None if choice['class_weight']=='none' else choice['class_weight'])
    rows_all=list(dev)+list(test)
    np.testing.assert_allclose(score(refit,x.loc[rows_all]),score(b,x.loc[rows_all]),rtol=0,atol=1e-8)
    np.testing.assert_array_equal((score(refit,x.loc[test])>b['threshold']).astype(int),saved.prediction)
    result=dict(dataset=ds,run=str(folder.relative_to(ROOT)),passed=True,joint_trials_verified=rows,
                candidate_selection_reproduced=True,lr_code_hash_current=True,lr_source_match=lr_source,runtime_source=runtime_source,
                selected_config_refit_reproduced=True,notebook_library_hash_recorded_at_run=False,
                notebook_library=notebook_library_status(),
                scope_note='저장 모델 예측 재현 + 선택 설정 재학습 재현. 탐색한 240개 설정 전부의 재학습 재현은 아님',
                oof_exactly_once=True,
                fold_roles_verified=True,test_predictions_reproduced=True)
    results.append(result)
# Do not add files to the immutable LR run folder; store the audit with this test run.
out=ROOT/'output/operations_tests'/os.environ.get('KAMP_TEST_RUN_ID','manual-'+__import__('datetime').datetime.now().strftime('%Y%m%dT%H%M%S'));out.mkdir(parents=True,exist_ok=True)
write(out/'logistic_independent_audit.json',results)
print(json.dumps(results,indent=2))
