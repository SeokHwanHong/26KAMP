"""Validate figure coverage, notebook preservation and frozen result consistency."""
from pathlib import Path
import json,subprocess,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/report_visualizations/20261006_v4'
manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));audit=json.loads((OUT/'notebook_execution_audit.json').read_text(encoding='utf-8'))
assert len(manifest['figures'])==40 and len(manifest['notebooks'])==14
assert sum(r['figure_outputs'] for r in audit['notebooks'])==40
for key in manifest['notebooks']:
    current=json.loads((ROOT/key).read_text(encoding='utf-8'))
    original=json.loads(subprocess.check_output(['git','show','HEAD:'+key],cwd=ROOT).decode('utf-8'))
    preserved=[c for c in current['cells'] if 'report_v4_visualization' not in c.get('metadata',{}).get('tags',[])]
    assert preserved==original['cells'],key+' original cells changed'
    for c in current['cells']:
        if 'report_v4_visualization' in c.get('metadata',{}).get('tags',[]):
            assert 'pipeline_library' not in c.get('metadata',{}).get('tags',[])
            assert not any(o['output_type']=='error' for o in c.get('outputs',[]))
for f in manifest['figures']:assert (ROOT/f['path']).is_file()
expected={('cn7','rf'):(3,8,0,111),('cn7','lr'):(0,0,3,119),('cn7','ocsvm'):(0,0,3,119),('rg3','lr'):(5,110,0,4),('rg3','ocsvm'):(5,89,0,25)}
metrics=pd.read_csv(OUT/'frozen_model_metrics.csv')
for (d,m),counts in expected.items():
    r=metrics[(metrics.dataset==d)&(metrics.model==m)].iloc[0]
    assert tuple(int(r[k]) for k in ['TP','FP','FN','TN'])==counts
for d,run in [('cn7','lr-20261003T040927Z-66e8d685'),('rg3','lr-20261003T040950Z-8c8b16ec')]:
    historical=pd.read_csv(ROOT/f'output/logistic/{d}/{run}/test_predictions.csv').set_index('pattern_row').sort_index()
    fresh=pd.read_csv(OUT/f'{d}_lr_frozen_test.csv').set_index('pattern_row').sort_index()
    np.testing.assert_allclose(fresh.score,historical.probability,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(fresh.prediction,historical.prediction)
    version=manifest['model_versions'][d]['if'];historic_if=pd.read_csv(ROOT/f'runtime/{d}/if_inputs/{version}/test_scores.csv').set_index('pattern_row').sort_index()
    fresh_if=pd.read_csv(OUT/f'{d}_if_frozen_test.csv').set_index('pattern_row').sort_index()
    np.testing.assert_allclose(fresh_if.score,historic_if.if_score,rtol=1e-12,atol=1e-12)
    for name in ['error_interval_counts','error_joint_counts']:
        groups=pd.read_csv(OUT/(name+'.csv')).query('dataset==@d')
        assert groups.N.sum()==len(fresh) and groups.risk.sum()==int(fresh.label.sum())
policy=pd.read_csv(OUT/'rg3_actual_policy_comparison.csv')
for plan in json.loads((OUT/'rg3_actual_policy_plans.json').read_text(encoding='utf-8')):
    assert plan['status']=='planned' and len(plan['selected_ids'])==plan['budget_k']
    assert len(set(plan['selected_ids']))==plan['budget_k']
    assert not set(plan['priority_ids'])&set(plan['random_ids'])
assert np.allclose(policy.random_mean_TP,policy.k*5/238)
assert policy.iloc[1].IF_TP==2 and policy.iloc[1].policy_fixed_TP==0
report=json.loads((OUT/'report_audit.json').read_text(encoding='utf-8'))
assert report['figures_embedded']==40 and set(report['figure_order'])=={f['id'] for f in manifest['figures']}
result=dict(status='passed',notebooks=14,figures=40,original_cells_match_git=True,
            frozen_scores_match_historical=True,error_group_counts_reconcile=True,policy_budget_exact=True)
(OUT/'validation_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
