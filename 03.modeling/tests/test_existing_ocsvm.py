"""Run notebook regression cases without full hyperparameter search or deployment."""
from pathlib import Path
import io
import json
import os
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
(ROOT/'tmp').mkdir(exist_ok=True)
tempfile.tempdir=str(ROOT/'tmp')
sys.stdout.reconfigure(encoding='utf-8')
results=[]
notebook=ROOT/'03.modeling/models/ocsvm.ipynb'
nb=json.loads(notebook.read_text(encoding='utf-8'))
out=ROOT/'output/operations_tests'/os.environ.get('KAMP_TEST_RUN_ID','manual-'+__import__('datetime').datetime.now().strftime('%Y%m%dT%H%M%S'));out.mkdir(parents=True,exist_ok=True)
for ds in ['cn7','rg3']:
    ns={'__name__':'notebook_regression'}
    for c in nb['cells']:
        if c['cell_type']=='code' and 'pipeline_library' in c.get('metadata',{}).get('tags',[]):
            exec(compile(''.join(c['source']),str(notebook),'exec'),ns)
    ns['DATASET']=ds
    cases=[c for c in nb['cells'] if c['cell_type']=='code' and 'class WorkflowTests(' in ''.join(c['source'])]
    assert len(cases)==1
    exec(''.join(cases[0]['source']),ns)
    for order in ['original','if_desc']:
        ns['TRAIN_ORDER']=order;stream=io.StringIO()
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ns['WorkflowTests']))
        (out/f'ocsvm_{ds}_{order}.txt').write_text(stream.getvalue(),encoding='utf-8')
        results.append(dict(dataset=ds,order=order,tests=result.testsRun,passed=result.wasSuccessful()))
        print(ds,order,result.testsRun,result.wasSuccessful(),flush=True)
(out/'existing_ocsvm_regression.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
sys.exit(0 if all(r['passed'] for r in results) else 1)
