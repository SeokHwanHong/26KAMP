"""Policy continuity, legacy evaluation recovery and portable source provenance."""
from pathlib import Path
import contextlib
import hashlib
import json
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import test_workflow_runtime as fixture
import pipeline_runtime as pr
import workflow_runtime as wf
from pipeline_runtime import ROOT,read,write


class ReviewTests(unittest.TestCase):
    ops=fixture.WorkflowTests.ops
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='review_20261006_')
        self.root=Path(self.tmp.name)
    def tearDown(self):
        assert self.root.resolve().is_relative_to((ROOT/'tmp').resolve())
        self.tmp.cleanup()

    def window(self,ops,seed,threshold):
        original=pr.read
        def calibration(path,default=None):
            if Path(path).name.startswith('calibration_'):
                return dict(threshold=threshold,n_eff=45,repeats=20,quantile=ops.policy['drift_quantile'])
            return original(path,default)
        with patch.object(pr,'read',side_effect=calibration):
            return ops.ingest(fixture.sample(45,seed,shift=9)[0],batch_id=f'W{seed}',coordinates_confirmed=True)['drift']

    def test_policy_return_restarts_consecutive_alarm(self):
        ops,_,_,_=self.ops();runs=[]
        for seed,q,threshold in [(801,.5,0),(802,.999,1),(803,.5,0),(804,.5,0)]:
            ops.policy['drift_quantile']=q;d=self.window(ops,seed,threshold)
            runs.append((d['status'],d['streak']))
        self.assertEqual(runs,[('watch',1),('normal',0),('watch',1),('review',2)])

    def test_legacy_does_not_block_new_registration_or_leave_partial(self):
        ops,_,_,_=self.ops();x,y=fixture.sample(80,811)
        old=ops.register_evaluation(x.assign(PassOrFail=y),'original labels')
        folder=ops.state/'evaluations'/old;(folder/'manifest_integrity.json').unlink()
        original=(folder/'manifest.json').read_bytes();x,y=fixture.sample(80,812)
        new=ops.register_evaluation(x.assign(PassOrFail=y),'new labels')
        self.assertEqual(read(ops.state/'evaluations'/new/'manifest.json')['purpose'],'independent')
        self.assertEqual((folder/'manifest.json').read_bytes(),original)
        self.assertFalse((folder/'manifest_integrity.json').exists())

    def test_registration_failure_is_staged_and_same_id_retry_recovers(self):
        ops,_,_,_=self.ops();x,y=fixture.sample(80,813);real=wf.write
        def fail_manifest(path,value):
            if Path(path).name=='manifest.json':raise OSError('injected registration failure')
            return real(path,value)
        with patch.object(wf,'write',side_effect=fail_manifest):
            with self.assertRaises(OSError):ops.register_evaluation(x.assign(PassOrFail=y),'labels',evaluation_id='RECOVER')
        self.assertFalse((ops.state/'evaluations/RECOVER').exists())
        self.assertEqual(ops.register_evaluation(x.assign(PassOrFail=y),'labels',evaluation_id='RECOVER'),'RECOVER')
        ops._verify_evaluation_manifest('RECOVER')

    def test_line_endings_are_equivalent_but_code_edits_are_not(self):
        from source_provenance import verify_source_digest
        p=self.root/'code.py';lf=b'def value():\n    return 1\n';expected=hashlib.sha256(lf).hexdigest()
        p.write_bytes(lf.replace(b'\n',b'\r\n'))
        self.assertEqual(verify_source_digest(p,expected),'line_endings_equivalent')
        p.write_bytes(b'def value():\r\n    return 2\r\n')
        with self.assertRaises(AssertionError):verify_source_digest(p,expected)

    def test_legacy_overlap_and_migration_never_claim_independence(self):
        ops,_,_,_=self.ops();x,y=fixture.sample(80,821)
        old=ops.register_evaluation(x.assign(PassOrFail=y),'original labels')
        folder=ops.state/'evaluations'/old;(folder/'manifest_integrity.json').unlink()
        original={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
        subset=ops.register_evaluation(x.iloc[:40].assign(PassOrFail=y.iloc[:40]),'subset labels')
        m=read(ops.state/'evaluations'/subset/'manifest.json')
        self.assertEqual(m['purpose'],'historical_followup');self.assertIn(old,m['legacy_evaluation_ids'])
        migrated=ops.migrate_evaluation(old,'confirmed original source','legacy evidence recovery','MIGRATED')
        m=read(ops.state/'evaluations'/migrated/'manifest.json')
        self.assertEqual(m['purpose'],'historical_followup')
        self.assertEqual(m['migration']['source_evaluation_id'],old)
        self.assertEqual({p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()},original)
        ops._verify_evaluation_manifest(migrated)

    def test_staged_retry_cannot_replace_original_content(self):
        ops,_,_,_=self.ops();x,y=fixture.sample(80,822);real=wf.write
        def fail(path,value):
            if Path(path).name=='manifest.json':raise OSError('injected')
            return real(path,value)
        with patch.object(wf,'write',side_effect=fail):
            with self.assertRaises(OSError):ops.register_evaluation(x.assign(PassOrFail=y),'labels',evaluation_id='STAGED')
        x2,y2=fixture.sample(80,823)
        with self.assertRaisesRegex(ValueError,'최초 요청과 다름'):
            ops.register_evaluation(x2.assign(PassOrFail=y2),'labels',evaluation_id='STAGED')
        self.assertEqual(ops.evaluation_registration_status()[0]['evaluation_id'],'STAGED')

    def test_missing_older_monitor_entry_does_not_replace_latest_policy(self):
        ops,_,_,_=self.ops();ops.policy['drift_quantile']=.5
        first=self.window(ops,831,0)
        ops.policy['drift_quantile']=.999;self.window(ops,832,1)
        m=read(ops.state/'monitor.json');m['history']=m['history'][1:]
        write(ops.state/'monitor.json',m);ops.reconcile_monitor()
        self.assertEqual(read(ops.state/'monitor.json')['history'][0]['id'],first['id'])
        ops.policy['drift_quantile']=.5;last=self.window(ops,833,0)
        self.assertEqual((last['status'],last['streak']),('watch',1))

    def test_saved_active_models_load_and_score_in_fresh_process(self):
        deps=(ROOT/'03.modeling/requirements_operations.txt').read_text(encoding='utf-8')
        self.assertIn('pyarrow==23.0.1',deps)
        code="""import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from pipeline_runtime import read,ROOT,Runtime,data,score
import pyarrow,numpy as np
assert pyarrow.__version__=='23.0.1'
for ds in ['cn7','rg3']:
 x,*_=data(ds)
 # Read artifacts directly through Runtime without constructing/changing operating state.
 obj=object.__new__(Runtime);obj.state=ROOT/'runtime'/ds;obj.dataset=ds
 from pipeline_runtime import digest
 obj.schema_hash=digest(ROOT/'data/schema/input_features.json')
 for version in obj.registry()['active_models'].values():
  b,_=obj.load_model(version);assert np.isfinite(score(b,x.iloc[:2])).all()
print('model_load_and_score_passed')
"""
        result=subprocess.run([sys.executable,'-I','-B','-c',code,str(ROOT/'03.modeling/common')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_baseline_return_restarts_alarm(self):
        ops,_,_,_=self.ops();original=read(ops.state/'baseline_pointer.json')['id']
        self.window(ops,841,0);ops.create_baseline();self.window(ops,842,0)
        write(ops.state/'baseline_pointer.json',dict(id=original))
        last=self.window(ops,843,0)
        self.assertEqual((last['status'],last['streak']),('watch',1))

    def test_atomic_registration_rename_failure_can_retry(self):
        ops,_,_,_=self.ops();x,y=fixture.sample(80,851);real=wf.os.replace
        def fail_final(source,target):
            if Path(source).is_dir():raise OSError('injected commit failure')
            return real(source,target)
        with patch.object(wf.os,'replace',side_effect=fail_final):
            with self.assertRaises(OSError):ops.register_evaluation(x.assign(PassOrFail=y),'labels',evaluation_id='RENAME')
        self.assertFalse((ops.state/'evaluations/RENAME').exists())
        self.assertEqual(ops.register_evaluation(x.assign(PassOrFail=y),'labels',evaluation_id='RENAME'),'RENAME')
        ops._verify_evaluation_manifest('RENAME')


if __name__=='__main__':unittest.main(verbosity=2)
