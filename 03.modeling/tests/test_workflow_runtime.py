"""Isolated P1/P2 failure recovery and lineage acceptance tests."""
from pathlib import Path
import sys,tempfile,unittest,copy,subprocess,json
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from workflow_runtime import Operations,LockHeld
import workflow_runtime as w
from pipeline_runtime import ROOT,read,write,fit_supervised,feature_columns,score
import numpy as np
import pandas as pd

def sample(n=120,seed=1,shift=0):
    x=pd.DataFrame(np.random.default_rng(seed).normal(size=(n,24))+shift,columns=feature_columns())
    return x,(x.iloc[:,0]>shift).astype(int)

class WorkflowTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def ops(self,ds='rg3'):
        ops=Operations(ds,self.root,dict(min_window_rows=40,min_window_patterns=30,bootstrap_repeats=20,cn7_decision_mode='threshold'))
        x,y=sample();model=fit_supervised(ds,x,y,C=100);model['threshold']=1
        old=ops.save_candidate(model,x,y,source={'synthetic':True});ops.initialize(old,'isolated synthetic reference')
        ops.create_baseline();return ops,x,y,old
    def candidate_assessment(self,ops,x,y,old):
        v=ops.save_candidate(fit_supervised(ops.dataset,x,y,C=100),x,y,source={'synthetic':True},parent=old)
        ex,ey=sample(180,22);eid=ops.register_evaluation(ex.assign(PassOrFail=ey),'synthetic independent')
        assessment=ops.evaluate(v,eid)
        self.assertTrue(assessment['passed'],assessment['reasons']);return v,assessment,eid

    def test_advice_failure_resume_uses_original_versions(self):
        for ds in ('cn7','rg3'):
            ops,x,y,old=self.ops(ds);ex,ey=sample(45,7)
            with patch.object(ops,'_cn7_priority' if ds=='cn7' else '_rg3_advice',side_effect=RuntimeError('injected')):
                r=ops.ingest(ex.assign(PassOrFail=ey),batch_id='B',label_source='synthetic',coordinates_confirmed=True)
            self.assertEqual(r['processing']['stage'],'advice_failed')
            v,a,_=self.candidate_assessment(ops,x,y,old);ops.promote(a['id'])
            self.assertEqual(ops.resume('B')['processing']['stage'],'completed')
            meta=read(ops.state/'batches/B/advice_latest.json')
            self.assertEqual(meta['context']['versions']['lr'],old)
            self.assertEqual(ops.active()['lr'],v)

    def test_waiting_followup_preserves_history_and_no_resend(self):
        ops,_,_,_=self.ops();ex,_=sample(20,8)
        first=ops.ingest(ex,batch_id='W',coordinates_confirmed=True)
        self.assertEqual(first['drift']['status'],'waiting')
        initial=read(ops.state/'batches/W/advice_latest.json')['revision']
        ex,_=sample(25,9);r=ops.ingest(ex,batch_id='NEXT',coordinates_confirmed=True)
        status=ops.batch_status('W');self.assertTrue(status['distribution_final'])
        self.assertEqual(status['latest_drift']['drift_id'],r['drift']['id'])
        self.assertNotEqual(status['advice']['revision'],initial)
        self.assertTrue((ops.state/'batches/W/advice_history'/initial/'context.json').exists())
        self.assertEqual(read(ops.state/'batch_ledger.json')['batches'],['W','NEXT'])

    def test_transition_preparation_failure_and_resume(self):
        ops,x,y,old=self.ops('cn7');v,a,_=self.candidate_assessment(ops,x,y,old)
        with patch.object(ops,'_build_baseline',side_effect=OSError('injected baseline failure')):
            with self.assertRaises(OSError):ops.promote(a['id'])
        self.assertEqual(ops.active()['lr'],old);self.assertFalse(ops.baseline_status()['valid'])
        with self.assertRaisesRegex(ValueError,'전환 미완료'):ops.ingest(sample(8,10)[0],coordinates_confirmed=True)
        self.assertEqual(ops.resume_transition(),v);self.assertTrue(ops.baseline_status()['valid'])
        self.assertEqual(ops.rollback('lr','synthetic rollback'),old)

    def test_pointer_failure_and_rollback_failure_are_recoverable(self):
        ops,x,y,old=self.ops();v,a,_=self.candidate_assessment(ops,x,y,old)
        real=w.write
        def failed(path,value):
            if Path(path).name=='baseline_pointer.json':raise OSError('injected pointer failure')
            return real(path,value)
        with patch.object(w,'write',side_effect=failed):
            with self.assertRaises(OSError):ops.promote(a['id'])
        with self.assertRaisesRegex(ValueError,'전환 미완료'):ops.detect()
        self.assertEqual(ops.resume_transition(),v)
        with patch.object(ops,'_build_baseline',side_effect=OSError('rollback baseline failure')):
            with self.assertRaises(OSError):ops.rollback('lr','test')
        self.assertEqual(ops.resume_transition(),old);self.assertTrue(ops.baseline_status()['valid'])
        self.assertEqual(len([e for e in ops.registry()['history'] if e['action']=='rollback']),1)

    def test_actual_second_process_cannot_enter_operation(self):
        ops,_,_,_=self.ops()
        code="import sys;sys.path.insert(0,sys.argv[1]);from workflow_runtime import Operations,LockHeld;from pathlib import Path\no=Operations('rg3',Path(sys.argv[2]))\ntry:\n with o.operation():pass\nexcept LockHeld:sys.exit(7)"
        with ops.operation():
            r=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'03.modeling/common'),str(self.root)],capture_output=True)
        self.assertEqual(r.returncode,7,r.stderr.decode(errors='replace'))

    def test_same_object_second_thread_cannot_reenter(self):
        from concurrent.futures import ThreadPoolExecutor
        ops,_,_,_=self.ops()
        def enter():
            try:
                with ops.operation():return 'entered'
            except LockHeld:return 'blocked'
        with ops.operation(),ThreadPoolExecutor(max_workers=1) as pool:
            self.assertEqual(pool.submit(enter).result(),'blocked')

    def test_cn7_rf_advice_reuses_original_rf_after_transition(self):
        ops,x,y,_=self.ops('cn7');ops.policy['cn7_decision_mode']='budget'
        rf=ops.save_candidate(fit_supervised('cn7',x,y,kind='rf'),x,y,source={'synthetic':True})
        ops.initialize(rf,'synthetic rf');ops.create_baseline()
        ex,_=sample(45,55)
        with patch.object(ops,'_cn7_priority',side_effect=RuntimeError('injected')):
            ops.ingest(ex,batch_id='RF-B',coordinates_confirmed=True)
        changed=ops.save_candidate(fit_supervised('cn7',x,1-y,kind='rf'),x,y,source={'synthetic':True})
        registry=copy.deepcopy(ops.registry());registry['active_models']['rf']=changed
        with ops.operation():ops._transition(registry,changed)
        r=ops.resume('RF-B');self.assertEqual(r['inspection_priority']['model_version'],rf)
        self.assertEqual(ops.active()['rf'],changed)

    def test_waiting_followup_advice_failure_is_repairable(self):
        ops,_,_,_=self.ops();ops.ingest(sample(20,61)[0],batch_id='W',coordinates_confirmed=True)
        real=ops._advise
        def fail_earlier(result):
            if result['id']=='W':raise OSError('injected refresh failure')
            return real(result)
        with patch.object(ops,'_advise',side_effect=fail_earlier):
            ops.ingest(sample(25,62)[0],batch_id='N',coordinates_confirmed=True)
        self.assertTrue(ops.batch_status('W')['processing']['latest_advice_error'])
        old_monitor=read(ops.state/'monitor.json')
        fixed=ops.refresh_advice('W');self.assertIsNone(fixed['processing']['latest_advice_error'])
        self.assertEqual(read(ops.state/'monitor.json'),old_monitor)

    def test_missing_or_corrupt_processing_is_rejected_without_resend(self):
        ops,_,_,_=self.ops();x,_=sample(8,63)
        for i,damage in enumerate(('missing','invalid_json','wrong_type')):
            bid=f'PROC{i}';ops.ingest(x+10*i,batch_id=bid,coordinates_confirmed=True)
            path=ops._processing_path(bid)
            if damage=='missing':path.unlink()
            elif damage=='invalid_json':path.write_text('{broken')
            else:write(path,[])
            ledger=read(ops.state/'batch_ledger.json')
            self.assertEqual(ops.ingest(x+10*i,batch_id=bid,coordinates_confirmed=True)['status'],'rejected')
            self.assertEqual(read(ops.state/'batch_ledger.json'),ledger)

    def test_overlap_subset_cannot_be_final_evaluation(self):
        ops,x,y,old=self.ops();v,a,eid=self.candidate_assessment(ops,x,y,old)
        write(ops.state/'selections/synthetic.json',dict(evaluation_id=eid))
        ex,ey=sample(180,22)
        subset=ops.register_evaluation(ex.iloc[:100].assign(PassOrFail=ey.iloc[:100]),'synthetic subset')
        self.assertTrue(ops._selection_overlap(subset))

    def test_evaluation_fingerprint_is_recomputed(self):
        ops,_,_,_=self.ops();ex,ey=sample(80,25);eid=ops.register_evaluation(ex.assign(PassOrFail=ey),'synthetic')
        folder=ops.state/'evaluations'/eid;p=pd.read_csv(folder/'patterns.csv',float_precision='round_trip')
        p.loc[0,'fingerprint']='fake';p.to_csv(folder/'patterns.csv',index=False)
        from pipeline_runtime import digest
        m=read(folder/'manifest.json');m['sha256']=digest(folder/'patterns.csv');m['fingerprints']=p.fingerprint.tolist();write(folder/'manifest.json',m)
        write(folder/'manifest_integrity.json',dict(manifest_sha256=digest(folder/'manifest.json')))
        with self.assertRaisesRegex(ValueError,'fingerprint'):ops._verify_evaluation_manifest(eid)

    def test_selection_re_registration_overlap_blocks_promotion(self):
        ops,x,y,old=self.ops();v,a,eid=self.candidate_assessment(ops,x,y,old)
        write(ops.state/'selections/synthetic.json',dict(evaluation_id=eid))
        ex,ey=sample(180,22);duplicate=ops.register_evaluation(ex.assign(PassOrFail=ey),'synthetic copy')
        self.assertIn(eid,read(ops.state/'evaluations'/duplicate/'manifest.json')['previous_evaluation_ids'])
        a2=ops.evaluate(v,duplicate)
        with self.assertRaisesRegex(ValueError,'선정 자료'):ops.promote(a2['id'])

    def test_full_labeled_unlabeled_ct_hold_promote_rollback(self):
        for ds in ('cn7','rg3'):
            ops,x,y,old=self.ops(ds);bids=[]
            for seed in (31,32):
                ex,ey=sample(45,seed,shift=9)
                r=ops.ingest(ex.assign(PassOrFail=ey),label_source='synthetic process change',coordinates_confirmed=True)
                bids.append(r['id'])
            self.assertEqual(r['drift']['status'],'review')
            candidate=ops.retrain('lr',bids,r['drift']['id'],'synthetic injected process shift')
            self.assertNotEqual(candidate,old)
            # Refit keeps parent threshold; it must be held, not silently deployed.
            ex,ey=sample(180,42);eid=ops.register_evaluation(ex.assign(PassOrFail=ey),'synthetic holdout')
            held=ops.evaluate(candidate,eid);self.assertFalse(held['passed'])
            self.assertEqual(ops.active()['lr'],old)
            v,a,_=self.candidate_assessment(ops,x,y,old);ops.promote(a['id']);ops.rollback('lr','synthetic exercise')
            self.assertEqual(ops.active()['lr'],old)
            unlabeled=ops.ingest(sample(45,99)[0],coordinates_confirmed=True)
            self.assertFalse(unlabeled['labeled'])

if __name__=='__main__':unittest.main(verbosity=2)
