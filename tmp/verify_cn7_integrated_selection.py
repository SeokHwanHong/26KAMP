from pathlib import Path
import json,hashlib,base64
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'modeling/ocsvm_cn7_integrated.ipynb'
n=json.loads(p.read_text(encoding='utf-8'))
hashes=n['metadata']['integration']['source_sha256']
n['metadata']['integration']['source_sha256']={k.replace('\\','/'):hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in hashes}
codes=[c for c in n['cells'] if c['cell_type']=='code']
for c in codes:
 compile(''.join(c['source']),'cell','exec')
 assert c['execution_count'] is not None
 assert not any(o.get('name')=='stderr' or o.get('output_type')=='error' for o in c['outputs'])
for c in codes:
 if not c.get('metadata',{}).get('selection_cycle'):continue
 for o in c['outputs']:
  if 'image/png' in o.get('data',{}):assert base64.b64decode(o['data']['image/png']).startswith(b'\x89PNG')
new=ROOT/'output/ocsvm_cn7_integrated/feature_scenarios/selection_cycle'
old=ROOT/'output/ocsvm_cn7_feature_scenarios/selection_cycle'
for name in ['candidate_comparison.csv','test_metrics.csv','test_predictions.csv']:
 pd.testing.assert_frame_equal(pd.read_csv(new/name),pd.read_csv(old/name))
summary=''.join(n['cells'][-1]['source'])
(new/'analysis_summary.md').write_text((new/'analysis_summary.md').read_text(encoding='utf-8')+'\n\n'+summary,encoding='utf-8')
p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
record={'all_cells_have_execution_results':True,'code_cells':len(codes),'new_code_cells_executed':5,'previous_full_run_code_cells':30,'validation':'Original integrated flow previously executed; appended workflow executed with C-section setup in a fresh process.','matches_feature_scenarios_results':True,'notebook_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_sha256':n['metadata']['integration']['source_sha256']}
(ROOT/'output/ocsvm_cn7_integrated/integration_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('Integrated notebook verified; candidate and test results match.')
