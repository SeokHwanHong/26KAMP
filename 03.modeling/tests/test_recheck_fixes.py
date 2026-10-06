"""Regression probes for the three 2026-10-05 review findings; temporary state only."""
from pathlib import Path
import contextlib
import copy
import io
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import run_checks_only as runner
import test_workflow_runtime as fixture
sample=fixture.sample
from pipeline_runtime import read,write
import decision_runtime as decision


class RunnerTests(unittest.TestCase):
    def run_case(self,regression_code,baseline='valid',pending=None):
        class FakeOperations:
            def __init__(self,*args):pass
            def active(self):return {'lr':'fixture'}
            def baseline_status(self):
                if baseline=='exception':raise OSError('injected status failure')
                return dict(valid=baseline=='valid',reasons=[] if baseline=='valid' else ['injected invalid baseline'])
            def pending_batches(self):return pending or []
        (runner.ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'tmp',prefix='runner_test_') as td,patch.object(runner,'OUT',Path(td)), \
             patch.object(runner,'versions',return_value={}), \
             patch.object(runner.subprocess,'run',return_value=types.SimpleNamespace(returncode=regression_code,stdout='fixture')), \
             patch('governance_runtime.Operations',FakeOperations),contextlib.redirect_stdout(io.StringIO()):
            assert Path(td).resolve().is_relative_to((runner.ROOT/'tmp').resolve())
            code=runner.main()
            return code,json.loads((Path(td)/'checks_only_latest.json').read_text(encoding='utf-8'))

    def test_regression_and_runtime_exit_matrix(self):
        for regression in (0,1):
            for baseline in ('valid','invalid','exception'):
                with self.subTest(regression=regression,baseline=baseline):
                    code,result=self.run_case(regression,baseline)
                    expected=0 if regression==0 and baseline=='valid' else 1
                    self.assertEqual(code,expected)
                    self.assertEqual(result['run_checks']['passed'],regression==0)
                    self.assertEqual(result['runtime_validation']['passed'],baseline=='valid')

    def test_normal_waiting_is_not_failure(self):
        code,result=self.run_case(0,pending=[dict(batch_id='W',stage='waiting',error=None)])
        self.assertEqual(code,0)
        self.assertTrue(result['runtime_validation']['passed'])

    def test_failed_and_unknown_pending_are_not_success(self):
        for stage in ('advice_failed','latest_advice_failed','processing_evidence_missing','reservation_pending','unknown'):
            with self.subTest(stage=stage):
                code,result=self.run_case(0,pending=[dict(batch_id='B',stage=stage,error=None)])
                self.assertEqual(code,1)


class RecoveryTests(unittest.TestCase):
    setUp=fixture.WorkflowTests.setUp
    tearDown=fixture.WorkflowTests.tearDown
    ops=fixture.WorkflowTests.ops
    candidate_assessment=fixture.WorkflowTests.candidate_assessment

    def test_completed_resume_evidence_matrix(self):
        original_root=self.root
        for explicit in (False,True):
            for damage in ('intact','missing','directory','unreadable','changed','manifest_missing','manifest_json','manifest_type','manifest_files','patterns_changed'):
                with self.subTest(explicit=explicit,damage=damage):
                    bid=f"B{int(explicit)}-{damage}"
                    self.root=original_root/bid
                    ops,_,_,_=self.ops()
                    frame=sample(8,100+int(explicit))[0]
                    if explicit:frame['record_id']=[f'{bid}:{i}' for i in range(len(frame))]
                    ops.ingest(frame,batch_id=bid,coordinates_confirmed=True)
                    folder=ops.state/'batches'/bid;product=folder/'products.csv';manifest=folder/'manifest.json'
                    if damage=='missing':product.unlink()
                    elif damage=='directory':product.unlink();product.mkdir()
                    elif damage=='changed':product.write_text('changed',encoding='utf-8')
                    elif damage=='manifest_missing':manifest.unlink()
                    elif damage=='manifest_json':manifest.write_text('{broken',encoding='utf-8')
                    elif damage=='manifest_type':write(manifest,[])
                    elif damage=='manifest_files':m=read(manifest);m.pop('files');write(manifest,m)
                    elif damage=='patterns_changed':(folder/'patterns.csv').write_text('changed',encoding='utf-8')
                    ledger=read(ops.state/'batch_ledger.json');monitor=read(ops.state/'monitor.json');index=read(ops.state/'dedup_index.json')
                    real=decision.digest
                    def failing_digest(path):
                        if Path(path)==product:raise PermissionError('injected read failure')
                        return real(path)
                    with patch.object(decision,'digest',side_effect=failing_digest if damage=='unreadable' else real):
                        result=ops.resume(bid)
                    if damage=='intact':self.assertEqual(result['processing']['retry'],'already_completed')
                    else:self.assertEqual(result['status'],'rejected')
                    self.assertEqual(read(ops.state/'batch_ledger.json'),ledger)
                    self.assertEqual(read(ops.state/'monitor.json'),monitor)
                    self.assertEqual(read(ops.state/'dedup_index.json'),index)

    def test_initial_context_survives_completion_and_waiting_refresh(self):
        ops,x,y,old=self.ops()
        ops.ingest(sample(20,201)[0],batch_id='W',coordinates_confirmed=True)
        initial=read(ops._processing_path('W'))['advice_context']
        meta=read(ops.state/'batches/W/advice_latest.json')
        self.assertEqual(initial,meta['context'])
        _,assessment,_=self.candidate_assessment(ops,x,y,old)
        ops.promote(assessment['id'])
        ops.policy['rg3_history_triggers_recheck']=True
        ops.ingest(sample(25,202)[0],batch_id='N',coordinates_confirmed=True)
        proc=read(ops._processing_path('W'));latest=read(ops.state/'batches/W/advice_latest.json')
        self.assertEqual(proc['advice_context'],initial)
        self.assertNotEqual(latest['revision'],meta['revision'])
        self.assertEqual(latest['context']['versions'],initial['versions'])
        self.assertNotEqual(latest['context']['baseline'],initial['baseline'])
        self.assertTrue(latest['context']['policy']['rg3_history_triggers_recheck'])
        self.assertEqual(read(ops.state/'batches/W/advice_history'/meta['revision']/'context.json')['context'],initial)

    def test_failure_context_survives_transition_and_resume(self):
        ops,x,y,old=self.ops()
        with patch.object(ops,'_rg3_advice',side_effect=OSError('injected advice failure')):
            ops.ingest(sample(45,203)[0],batch_id='F',coordinates_confirmed=True)
        before=read(ops._processing_path('F'))
        self.assertEqual(before['stage'],'advice_failed')
        self.assertIn('injected advice failure',before['error'])
        self.assertEqual(before['advice_context']['versions']['lr'],old)
        candidate,assessment,_=self.candidate_assessment(ops,x,y,old)
        ops.promote(assessment['id']);ops.policy['rg3_history_triggers_recheck']=True
        result=ops.resume('F');after=read(ops._processing_path('F'))
        self.assertEqual(result['processing']['stage'],'completed')
        self.assertEqual(after['advice_context'],before['advice_context'])
        self.assertEqual(after['attempts'][:-1],before['attempts'])
        self.assertTrue(after['attempts'][-1]['resumed'])
        self.assertIsNone(after['error'])
        self.assertEqual(read(ops.state/'batches/F/advice_latest.json')['context'],before['advice_context'])

    def test_failed_waiting_refresh_restores_initial_context(self):
        ops,_,_,_=self.ops()
        ops.ingest(sample(20,301)[0],batch_id='W',coordinates_confirmed=True)
        initial=copy.deepcopy(read(ops._processing_path('W')))
        original=ops._advise
        def fail_waiting(result):
            if result['id']=='W':raise OSError('injected followup failure')
            return original(result)
        with patch.object(ops,'_advise',side_effect=fail_waiting):
            ops.ingest(sample(25,302)[0],batch_id='N',coordinates_confirmed=True)
        failed=read(ops._processing_path('W'))
        self.assertEqual(failed['advice_context'],initial['advice_context'])
        self.assertEqual(failed['stage'],'completed')
        self.assertIn('injected followup failure',failed['latest_advice_error'])
        monitor=read(ops.state/'monitor.json')
        ops.refresh_advice('W')
        restored=read(ops._processing_path('W'))
        self.assertEqual(restored['advice_context'],initial['advice_context'])
        self.assertIsNone(restored['latest_advice_error'])
        self.assertEqual(read(ops.state/'monitor.json'),monitor)


if __name__=='__main__':unittest.main(verbosity=2)
