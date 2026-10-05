"""Reinspection and absolute acceptance checks; synthetic cases are technical tests.
2026-10-05: F01-F07 team review regression tests added (ReviewFixTests), incl. commit-stage failure and tie/evaluation rule."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from decision_runtime import Operations,inspection_advice,usability
from pipeline_runtime import ROOT,feature_columns,fit_supervised,fingerprints,read
import numpy as np
import pandas as pd

class DecisionTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
        self.ops=Operations('cn7',Path(self.tmp.name))
    def tearDown(self):self.tmp.cleanup()
    def frame(self,n=80,seed=1):
        x=pd.DataFrame(np.random.default_rng(seed).normal(size=(n,24)),columns=feature_columns())
        return x,(x.iloc[:,0]>0).astype(int)
    def test_bad_absolute_performance_rejected(self):
        for m in [dict(recall=0,precision=0,FPR=0),dict(recall=1,precision=.04,FPR=.96)]:
            self.assertTrue(usability(m,self.ops.policy))
    def test_information_shortage_not_called_safe(self):
        x,_=self.frame(4);p=x.assign(record_id=range(4),fingerprint=fingerprints(x))
        _,rows=inspection_advice(x,np.zeros(4),p)
        self.assertTrue(all(r['reinspection_recommended'] and r['decision']=='확정 불가' for r in rows))
        self.assertTrue(all(r['evidence_level']=='insufficient_information' for r in rows))
    def test_incoming_labels_do_not_drive_advice(self):
        x,y=self.frame();p=x.assign(record_id=range(len(x)),fingerprint=fingerprints(x))
        _,a=inspection_advice(x,y,p.assign(label=0));_,b=inspection_advice(x,y,p.assign(label=1))
        self.assertEqual(a,b)
        self.assertTrue(any(r['risk_history_ranges'] for r in a))
    def test_batch_writes_user_reinspection_results(self):
        ops=Operations('rg3',Path(self.tmp.name));x,y=self.frame()
        version=ops.save_candidate(fit_supervised('rg3',x,y),x,y,source={'fixture':True})
        ops.initialize(version,'synthetic test')
        ex,_=self.frame(8,7)
        report=ops.ingest(ex,coordinates_confirmed=True)
        self.assertEqual(report['status'],'accepted')
        advice=read(report['inspection_advice']['json'])
        self.assertEqual(len(advice['records']),8)
        self.assertTrue(Path(report['inspection_advice']['ranges']).exists())
    def test_select_holds_all_unusable_and_selects_qualified_only(self):
        candidates=['lr','rf','ocsvm']
        def report(v,e):
            return dict(candidate=v,passed=v=='rf',reasons=[] if v=='rf' else ['기준 미달'],
                candidate_metrics=dict(F1=.6,recall=.7,FPR=.1))
        with patch.object(self.ops,'load_model',side_effect=lambda v:({'kind':v},{})),patch.object(self.ops,'evaluate',side_effect=report):
            result=self.ops.select_cn7(candidates,'same-eval')
            self.assertEqual(result['selected'],'rf')
        with patch.object(self.ops,'load_model',side_effect=lambda v:({'kind':v},{})),patch.object(self.ops,'evaluate',side_effect=lambda v,e:dict(report('lr',e),candidate=v)):
            self.assertIsNone(self.ops.select_cn7(candidates,'same-eval')['selected'])
    def test_actual_evaluate_blocks_historical_even_if_good(self):
        x,y=self.frame(100);b=fit_supervised('cn7',x,y,C=100)
        v=self.ops.save_candidate(b,x,y,source={'fixture':True})
        ex,ey=self.frame(100,2)
        eid=self.ops.register_evaluation(ex.assign(PassOrFail=ey),'synthetic',purpose='historical_followup')
        report=self.ops.evaluate(v,eid)
        self.assertFalse(report['passed'])
        self.assertTrue(any('후속 평가' in r for r in report['reasons']))

    def test_history_alone_is_reference_not_recheck_by_default(self):
        x,y=self.frame(400,3);p=x.assign(record_id=range(len(x)),fingerprint=fingerprints(x))
        _,rows=inspection_advice(x,y,p)
        elevated=[r for r in rows if r['evidence_level']=='elevated_risk_history']
        self.assertTrue(elevated)
        self.assertFalse(any(r['reinspection_recommended'] for r in elevated))
        self.assertLess(sum(r['reinspection_recommended'] for r in rows),len(rows))
        _,on=inspection_advice(x,y,p,history_triggers=True)
        self.assertTrue(all(r['reinspection_recommended'] for r in on if r['evidence_level']=='elevated_risk_history'))
    def test_out_of_reference_value_recommends_check(self):
        x,y=self.frame(400,3);q=x.iloc[:2].copy();q.iloc[0,0]=x.iloc[:,0].max()+10
        p=q.assign(record_id=range(2),fingerprint=fingerprints(q))
        _,rows=inspection_advice(x,y,p)
        self.assertEqual(rows[0]['evidence_level'],'out_of_reference')
        self.assertTrue(rows[0]['reinspection_recommended'] and rows[0]['outside_reference'])
    def test_resent_batch_held(self):
        ops=Operations('rg3',Path(self.tmp.name)/'dup',dict(min_window_rows=40,min_window_patterns=30,bootstrap_repeats=40))
        x,y=self.frame();v=ops.save_candidate(fit_supervised('rg3',x,y),x,y,source={'fixture':True});ops.initialize(v,'synthetic')
        ex,ey=self.frame(45,9);frame=ex.assign(PassOrFail=ey,product_id=[f'L-{i}' for i in range(45)])
        first=ops.ingest(frame,label_source='synthetic',coordinates_confirmed=True)
        self.assertEqual(first['status'],'accepted')
        again=ops.ingest(frame,label_source='synthetic',coordinates_confirmed=True)
        self.assertEqual(again['status'],'held');self.assertIn('동일한 내용',again['reason'])
        part=ops.ingest(frame.iloc[:10],label_source='synthetic',coordinates_confirmed=True)
        self.assertEqual(part['status'],'held');self.assertIn('제품 ID 중복',part['reason'])
        self.assertEqual(read(ops.state/'batch_ledger.json')['batches'],[first['id']])
    def test_selection_evaluation_cannot_justify_promotion(self):
        from pipeline_runtime import write
        write(self.ops.state/'selections'/'selection-x.json',dict(evaluation_id='eval-used'))
        write(self.ops.state/'assessments'/'assess-x'/'assessment.json',
              dict(kind='lr',evaluation_id='eval-used',candidate_metrics=dict(recall=1.,precision=1.,FPR=0.)))
        with self.assertRaisesRegex(ValueError,'선정에 사용한 평가'):self.ops.promote('assess-x')

    def _cn7_with_rf(self,root):
        ops=Operations('cn7',Path(self.tmp.name)/root,dict(min_window_rows=40,min_window_patterns=30,bootstrap_repeats=40))
        x,y=self.frame(160,11);b=fit_supervised('cn7',x,y,kind='rf')
        v=ops.save_candidate(b,x,y,source={'fixture':True});ops.initialize(v,'synthetic rf reference')
        return ops,v,x,y
    def test_cn7_batch_flags_inspection_budget(self):
        ops,v,_,_=self._cn7_with_rf('budget')
        ex,_=self.frame(40,12);r=ops.ingest(ex,coordinates_confirmed=True)
        info=r['inspection_priority'];self.assertEqual(info['status'],'ranked')
        self.assertEqual(info['budget_k'],4);self.assertEqual(info['flagged'],4);self.assertEqual(info['tie_rule'],'exact_k')
        table=pd.read_csv(info['csv']);self.assertEqual(len(table),40)
        top=table[table.inspect.eq(1)];self.assertGreaterEqual(top.risk_score.min(),table[table.inspect.eq(0)].risk_score.max())
    def test_cn7_without_active_rf_reports_status(self):
        ops=Operations('cn7',Path(self.tmp.name)/'norf',dict(min_window_rows=40,min_window_patterns=30,bootstrap_repeats=40))
        x,y=self.frame();v=ops.save_candidate(fit_supervised('cn7',x,y),x,y,source={});ops.initialize(v,'lr only')
        r=ops.ingest(self.frame(40,13)[0],coordinates_confirmed=True)
        self.assertEqual(r['inspection_priority']['status'],'no_active_model')
    def test_cn7_budget_gate_compares_same_budget(self):
        ops,v0,x,y=self._cn7_with_rf('gate')
        rng=np.random.default_rng(0);noise=fit_supervised('cn7',x,pd.Series(rng.permutation(y.to_numpy())),kind='rf')
        weak=ops.save_candidate(noise,x,y,source={},parent=v0)
        good=ops.save_candidate(fit_supervised('cn7',x,y,kind='rf'),x,y,source={},parent=v0)
        ex=pd.DataFrame(np.random.default_rng(14).normal(size=(200,24)),columns=feature_columns())
        ey=(ex.iloc[:,0]>1.3).astype(int)
        eid=ops.register_evaluation(ex.assign(PassOrFail=ey),'synthetic independent')
        bad=ops.evaluate(weak,eid);self.assertFalse(bad['passed']);self.assertEqual(bad['decision_mode'],'budget')
        self.assertIn('candidate_budget_metrics',bad)
        same=ops.evaluate(good,eid)
        self.assertTrue(any('개선 없음' in r for r in same['reasons']))
    def test_rg3_uncertainty_levels_and_zones(self):
        from decision_runtime import uncertainty_level,drift_zones
        self.assertEqual(uncertainty_level('monitor','review'),'high')
        self.assertEqual(uncertainty_level('out_of_reference','normal'),'high')
        self.assertEqual(uncertainty_level('elevated_risk_history','watch'),'medium')
        self.assertEqual(uncertainty_level('insufficient_information','normal'),'medium')
        self.assertEqual(uncertainty_level('elevated_risk_history','normal'),'low')
        z=drift_zones(dict(status='review',threshold=.1,changes={'a':.3,'b':.05},outside_reference={'a':4}))
        self.assertEqual(z['zones'][0]['representation'],'a');self.assertTrue(z['zones'][0]['over_threshold'])
        self.assertFalse(z['zones'][1]['over_threshold'])
    def test_rg3_review_drift_marks_batch_high(self):
        ops=Operations('rg3',Path(self.tmp.name)/'zones',dict(min_window_rows=40,min_window_patterns=30,bootstrap_repeats=40))
        x,y=self.frame(120,15);v=ops.save_candidate(fit_supervised('rg3',x,y),x,y,source={});ops.initialize(v,'synthetic')
        for seed in [16,17]:
            ex=pd.DataFrame(np.random.default_rng(seed).normal(size=(45,24))+9,columns=feature_columns())
            r=ops.ingest(ex,coordinates_confirmed=True)
        self.assertEqual(r['drift']['status'],'review')
        advice=read(r['inspection_advice']['json'])
        self.assertTrue(all(rec['uncertainty_level']=='high' and rec['reinspection_recommended'] for rec in advice['records']))
        self.assertTrue(advice['uncertainty_zones']['zones'])
    def test_stale_lock_reports_owner_and_can_be_cleared(self):
        from decision_runtime import LockHeld
        with self.ops.lock():
            with self.assertRaises(LockHeld) as ctx:
                with self.ops.lock():pass
            self.assertIn('unlock',str(ctx.exception))
            with self.assertRaisesRegex(ValueError,'진행 중'):self.ops.clear_lock('test')
        self.assertEqual(self.ops.clear_lock('nothing to clear')['status'],'no_lock')
        (self.ops.state/'.writer.lock').write_text('crashed job')
        self.assertEqual(self.ops.clear_lock('crashed run',force=True)['status'],'cleared')
        self.assertFalse((self.ops.state/'.writer.lock').exists())
    def test_pandas_row_index_is_not_cross_batch_product_id(self):
        ops=Operations('rg3',Path(self.tmp.name)/'idx',dict(min_window_rows=10**6))
        x,y=self.frame();v=ops.save_candidate(fit_supervised('rg3',x,y),x,y,source={});ops.initialize(v,'synthetic')
        for seed in [21,22]:
            ex,_=self.frame(30,seed);r=ops.ingest(ex.assign(**{'Unnamed: 0':range(30)}),coordinates_confirmed=True)
            self.assertEqual(r['status'],'accepted')


class ReviewFixTests(unittest.TestCase):
    """2026-10-05 team review F01-F07. Synthetic fixtures; technical behaviour only."""
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
    def tearDown(self):self.tmp.cleanup()
    def ops(self,name,dataset='rg3',n_ref=120,**policy):
        base=dict(min_window_rows=40,min_window_patterns=30,bootstrap_repeats=40);base.update(policy)
        ops=Operations(dataset,Path(self.tmp.name)/name,base)
        x=pd.DataFrame(np.random.default_rng(31).normal(size=(n_ref,24)),columns=feature_columns());y=(x.iloc[:,0]>0).astype(int)
        self.reference=x
        kind='rf' if dataset=='cn7' else 'lr'
        v=ops.save_candidate(fit_supervised(dataset,x,y,kind=kind),x,y,source={'fixture':True});ops.initialize(v,'synthetic')
        return ops
    def batch(self,n=45,seed=40):
        x=pd.DataFrame(np.random.default_rng(seed).normal(size=(n,24)),columns=feature_columns())
        x.iloc[1]=x.iloc[0].to_numpy().copy()  # same input pattern twice, conflicting labels
        y=(x.iloc[:,0]>0).astype(int).to_numpy().copy();y[0],y[1]=0,1
        return x.assign(PassOrFail=y)

    def test_f01_reordered_resend_without_ids_is_held(self):
        ops=self.ops('f01');frame=self.batch()
        first=ops.ingest(frame,label_source='synthetic',coordinates_confirmed=True)
        self.assertEqual(first['status'],'accepted')
        swapped=frame.iloc[[1,0]+list(range(2,len(frame)))].reset_index(drop=True)
        again=ops.ingest(swapped,label_source='synthetic',coordinates_confirmed=True)
        self.assertEqual(again['status'],'held');self.assertIn('중복 의심',again['reason'])
        self.assertEqual(read(ops.state/'batch_ledger.json')['batches'],[first['id']])
    def test_f01_same_content_new_ids_is_separate_production(self):
        ops=self.ops('f01b');frame=self.batch()
        first=ops.ingest(frame.assign(product_id=[f'A-{i}' for i in range(45)]),label_source='s',coordinates_confirmed=True)
        second=ops.ingest(frame.assign(product_id=[f'B-{i}' for i in range(45)]),label_source='s',coordinates_confirmed=True)
        self.assertEqual(second['status'],'accepted');self.assertIn(first['id'],second['duplicate_note'])
    def test_f02_drift_failure_keeps_commit_and_resumes_same_batch_id(self):
        ops=self.ops('f02');frame=self.batch()
        with patch.object(ops,'_run_detect',side_effect=RuntimeError('injected')):
            r=ops.ingest(frame,batch_id='LOT-1',label_source='s',coordinates_confirmed=True)
        self.assertEqual(r['status'],'accepted');self.assertEqual(r['processing']['stage'],'drift_failed')
        self.assertIn('LOT-1',read(ops.state/'dedup_index.json')['content'].values())
        self.assertEqual(ops.pending_batches()[0]['stage'],'drift_failed')
        other=ops.ingest(frame,batch_id='LOT-2',label_source='s',coordinates_confirmed=True)
        self.assertEqual(other['status'],'held')                # different id = duplicate, not a second lot
        again=ops.ingest(frame,batch_id='LOT-1',label_source='s',coordinates_confirmed=True)
        self.assertEqual(again['processing'],dict(stage='completed',resumed=True));self.assertIn('inspection_advice',again)
        self.assertEqual(read(ops.state/'batch_ledger.json')['batches'],['LOT-1'])
        done=ops.ingest(frame,batch_id='LOT-1',label_source='s',coordinates_confirmed=True)
        self.assertEqual(done['processing']['retry'],'already_completed');self.assertEqual(ops.pending_batches(),[])
    def test_f02_advice_failure_resumes_without_new_drift_window(self):
        ops=self.ops('f02b');frame=self.batch()
        with patch.object(ops,'_rg3_advice',side_effect=RuntimeError('injected')):
            r=ops.ingest(frame,batch_id='LOT-9',label_source='s',coordinates_confirmed=True)
        self.assertEqual(r['processing']['stage'],'advice_failed');drift_id=r['drift']['id']
        again=ops.resume('LOT-9')
        self.assertEqual(again['processing']['stage'],'completed');self.assertEqual(again['drift']['id'],drift_id)
        self.assertEqual(len(list((ops.state/'drift').glob('*.json'))),1)
    def test_f02_same_batch_id_with_other_content_is_conflict(self):
        ops=self.ops('f02c')
        ops.ingest(self.batch(),batch_id='LOT-3',label_source='s',coordinates_confirmed=True)
        r=ops.ingest(self.batch(seed=41),batch_id='LOT-3',label_source='s',coordinates_confirmed=True)
        self.assertEqual(r['status'],'conflict')
    def test_f02_held_batch_id_can_be_resent_after_fix(self):
        ops=self.ops('f02d');frame=self.batch()
        self.assertEqual(ops.ingest(frame,batch_id='LOT-4',label_source='s')['status'],'held')
        self.assertEqual(ops.ingest(frame,batch_id='LOT-4',label_source='s',coordinates_confirmed=True)['status'],'accepted')
        self.assertTrue(list((ops.state/'held_archive').glob('LOT-4-*')))
    def test_f03_ingest_lock_blocks_concurrent_registration(self):
        from decision_runtime import LockHeld
        ops=self.ops('f03')
        with ops._exclusive('ingest'):
            with self.assertRaises(LockHeld) as ctx:ops.ingest(self.batch(),label_source='s',coordinates_confirmed=True)
            self.assertIn('--lock ingest',str(ctx.exception))
        self.assertFalse((ops.state/'batch_ledger.json').exists())
    def test_f04_waiting_is_pending_not_low(self):
        from decision_runtime import uncertainty_level
        self.assertEqual(uncertainty_level('monitor','waiting'),'pending')
        self.assertEqual(uncertainty_level('out_of_reference','waiting'),'high')
        from pipeline_runtime import data
        ops=self.ops('f04');xx,_,_,dev,_,_=data('rg3');x=xx.loc[dev].iloc[:8].reset_index(drop=True)  # baseline rows: in range
        r=ops.ingest(x,coordinates_confirmed=True);self.assertEqual(r['drift']['status'],'waiting')
        recs=read(r['inspection_advice']['json'])['records']
        self.assertFalse(any(rec['uncertainty_level']=='low' for rec in recs))
        self.assertTrue(any(rec['uncertainty_label']=='판단 대기' for rec in recs))
    def test_f05_ties_follow_one_rule(self):
        from decision_runtime import budget_flags,budget_metrics
        fps=[f'f{9-i}' for i in range(10)]                     # f9..f0: the tie winner is f0 (index 9)
        flag,_,k,_,ties=budget_flags([.5]*10,.1,[f'r{i}' for i in range(10)],fingerprints=fps)
        self.assertEqual((k,int(flag.sum())),(1,1));self.assertTrue(flag[9]);self.assertTrue(ties['ties_cut'])
        flag,_,_,_,ties=budget_flags([.5]*10,.1,rule='all_ties')
        self.assertEqual(int(flag.sum()),10);self.assertEqual(ties['over_budget'],9)
    def test_f05_evaluation_uses_the_same_selection_as_operation(self):
        from decision_runtime import budget_flags,budget_metrics
        fps=[f'f{9-i}' for i in range(10)];s=[.5]*10
        picked=budget_flags(s,.1,fingerprints=fps)[0]
        for y in [np.array([0]*9+[1]),np.array([1]+[0]*9)]:
            m=budget_metrics(y,s,.1,fingerprints=fps)
            self.assertEqual(m['found_TP'],int(y[picked].sum()))     # same item decides both
            self.assertAlmostEqual(m['random_tie_expected_TP'],.1)  # random tie value kept as reference only
        self.assertEqual(budget_metrics(np.array([1]+[0]*9),s,.1,'all_ties',fps)['found_TP'],1)
    def test_zero_inspection_fraction_is_rejected(self):
        from decision_runtime import budget_flags
        with self.assertRaisesRegex(ValueError,'0 초과'):budget_flags([.1,.2],0)
        with self.assertRaisesRegex(ValueError,'0 초과'):Operations('cn7',Path(self.tmp.name)/'z',dict(inspection_fraction=0))
    def _fail_on(self,module,filename):
        import importlib
        mod=importlib.import_module(module);real=mod.write;state={'n':0}
        def bad(path,value):
            if Path(path).name==filename and state['n']==0:
                state['n']+=1;raise OSError('injected '+filename)
            return real(path,value)
        return patch.object(mod,'write',side_effect=bad)
    def test_commit_interrupted_after_ledger_is_recovered_and_blocks_duplicates(self):
        ops=self.ops('c1');frame=self.batch()
        with self._fail_on('decision_runtime','processing.json'):
            with self.assertRaises(OSError):ops.ingest(frame,batch_id='LOT-C',label_source='s',coordinates_confirmed=True)
        self.assertEqual(read(ops.state/'batch_ledger.json')['batches'],['LOT-C'])
        self.assertEqual([p['stage'] for p in ops.pending_batches()],['commit_incomplete'])
        self.assertEqual(ops.ingest(frame,batch_id='LOT-D',label_source='s',coordinates_confirmed=True)['status'],'held')
        again=ops.ingest(frame,batch_id='LOT-C',label_source='s',coordinates_confirmed=True)
        self.assertEqual(again['processing']['stage'],'completed')
        self.assertEqual(read(ops.state/'batch_ledger.json')['batches'],['LOT-C']);self.assertEqual(ops.pending_batches(),[])
        self.assertEqual(read(ops.state/'dedup_index.json')['reservations'],{})
    def test_commit_interrupted_at_base_ledger_write_is_recovered(self):
        ops=self.ops('c2');frame=self.batch()
        with self._fail_on('pipeline_runtime','batch_ledger.json'):
            with self.assertRaises(OSError):ops.ingest(frame,batch_id='LOT-E',label_source='s',coordinates_confirmed=True)
        self.assertFalse((ops.state/'batch_ledger.json').exists())
        self.assertEqual([p['stage'] for p in ops.pending_batches()],['commit_incomplete'])
        self.assertEqual(ops.ingest(frame,batch_id='LOT-F',label_source='s',coordinates_confirmed=True)['status'],'held')
        self.assertEqual(ops.resume('LOT-E')['processing']['stage'],'completed')
        self.assertEqual(read(ops.state/'batch_ledger.json')['batches'],['LOT-E'])
    def test_reservation_write_failure_stores_nothing(self):
        ops=self.ops('c3');frame=self.batch()
        with self._fail_on('decision_runtime','dedup_index.json'):
            with self.assertRaises(OSError):ops.ingest(frame,batch_id='LOT-G',label_source='s',coordinates_confirmed=True)
        self.assertFalse((ops.state/'batches'/'LOT-G').exists());self.assertEqual(ops.pending_batches(),[])
        self.assertEqual(ops.ingest(frame,batch_id='LOT-G',label_source='s',coordinates_confirmed=True)['processing']['stage'],'completed')
    def test_index_failure_after_ledger_is_listed_and_blocks_duplicates(self):
        ops=self.ops('c5');frame=self.batch()
        import decision_runtime;real=decision_runtime.write;n={'c':0}
        def bad(path,value):
            if Path(path).name=='dedup_index.json':
                n['c']+=1
                if n['c']==2:raise OSError('injected')        # 2nd index write = after files+ledger
            return real(path,value)
        with patch.object(decision_runtime,'write',side_effect=bad):
            with self.assertRaises(OSError):ops.ingest(frame,batch_id='LOT-J',label_source='s',coordinates_confirmed=True)
        self.assertEqual([p['batch_id'] for p in ops.pending_batches()],['LOT-J'])
        self.assertEqual(ops.ingest(frame,batch_id='LOT-K',label_source='s',coordinates_confirmed=True)['status'],'held')
        self.assertEqual(ops.ingest(frame,batch_id='LOT-J',label_source='s',coordinates_confirmed=True)['processing']['stage'],'completed')
        self.assertEqual(read(ops.state/'dedup_index.json')['reservations'],{})
        self.assertEqual(read(ops.state/'batch_ledger.json')['batches'],['LOT-J'])
    def test_interrupted_reservation_rejects_other_content_with_same_batch_id(self):
        ops=self.ops('c6');abc=self.batch(seed=80).assign(product_id=[f'A-{i}' for i in range(45)])
        deff=self.batch(seed=81).assign(product_id=[f'D-{i}' for i in range(45)])
        with self._fail_on('pipeline_runtime','manifest.json'):         # stop before the batch is stored
            with self.assertRaises(OSError):ops.ingest(abc,batch_id='LOT1',label_source='s',coordinates_confirmed=True)
        self.assertEqual([p['stage'] for p in ops.pending_batches()],['reserved_not_stored'])
        r=ops.ingest(deff,batch_id='LOT1',label_source='s',coordinates_confirmed=True)
        self.assertEqual(r['status'],'conflict');self.assertTrue(r['reservation_kept'])
        idx=read(ops.state/'dedup_index.json')
        self.assertEqual(idx['reservations']['LOT1']['ids'],sorted(f'A-{i}' for i in range(45)))
        self.assertFalse(any(i.startswith('D-') for i in idx['ids']))
        # same IDs, different content is also refused
        self.assertEqual(ops.ingest(self.batch(seed=82).assign(product_id=[f'A-{i}' for i in range(45)]),
                                    batch_id='LOT1',label_source='s',coordinates_confirmed=True)['status'],'conflict')
        self.assertEqual(ops.ingest(deff,batch_id='LOT2',label_source='s',coordinates_confirmed=True)['status'],'accepted')
        done=ops.ingest(abc,batch_id='LOT1',label_source='s',coordinates_confirmed=True)   # original file finishes LOT1
        self.assertEqual(done['processing']['stage'],'completed')
        self.assertEqual(sorted(read(ops.state/'batch_ledger.json')['batches']),['LOT1','LOT2'])
        self.assertEqual(read(ops.state/'dedup_index.json')['reservations'],{});self.assertEqual(ops.pending_batches(),[])
        self.assertEqual(ops.ingest(abc,batch_id='LOT3',label_source='s',coordinates_confirmed=True)['status'],'held')
    def test_interrupted_commit_rejects_other_ids_with_same_batch_id(self):
        ops=self.ops('c7');abc=self.batch(seed=83).assign(product_id=[f'A-{i}' for i in range(45)])
        with self._fail_on('decision_runtime','processing.json'):        # stored + ledger, commit unfinished
            with self.assertRaises(OSError):ops.ingest(abc,batch_id='LOT5',label_source='s',coordinates_confirmed=True)
        other=abc.assign(product_id=[f'X-{i}' for i in range(45)])       # same content, other IDs
        self.assertEqual(ops.ingest(other,batch_id='LOT5',label_source='s',coordinates_confirmed=True)['status'],'conflict')
        self.assertEqual(ops.ingest(abc,batch_id='LOT5',label_source='s',coordinates_confirmed=True)['processing']['stage'],'completed')
    def test_held_after_reservation_releases_it(self):
        ops=self.ops('c4');frame=self.batch().assign(product_id=[f'P-{i}' for i in range(45)])
        with patch('pipeline_runtime.Operations.active',return_value={}):   # base holds: no reference model
            self.assertEqual(ops.ingest(frame,batch_id='LOT-H',label_source='s',coordinates_confirmed=True)['status'],'held')
        idx=read(ops.state/'dedup_index.json')
        self.assertEqual((idx['reservations'],idx['ids']),({},{}))
        self.assertEqual(ops.ingest(frame,batch_id='LOT-I',label_source='s',coordinates_confirmed=True)['status'],'accepted')
    def test_f06_quantile_change_recalibrates(self):
        ops=self.ops('f06')
        r1=ops.ingest(self.batch(seed=60),label_source='s',coordinates_confirmed=True)
        self.assertEqual(r1['drift']['calibration']['quantile'],.99)
        ops2=Operations('rg3',ops.state.parent,dict(min_window_rows=40,min_window_patterns=30,bootstrap_repeats=40,drift_quantile=.5))
        r2=ops2.ingest(self.batch(seed=61),label_source='s',coordinates_confirmed=True)
        self.assertEqual(r2['drift']['calibration']['quantile'],.5)
        self.assertEqual(r2['drift']['calibration_signature']['quantile'],.5)
        self.assertLess(r2['drift']['threshold'],r1['drift']['threshold'])
        self.assertTrue(list(ops.state.glob('baselines/*/calibration_archive/*.json')))
    def test_f07_changed_purpose_blocks_evaluate_and_promote(self):
        from pipeline_runtime import write
        ops=self.ops('f07','cn7');ex=pd.DataFrame(np.random.default_rng(70).normal(size=(100,24)),columns=feature_columns())
        eid=ops.register_evaluation(ex.assign(PassOrFail=(ex.iloc[:,0]>1).astype(int)),'s',purpose='historical_followup')
        m=read(ops.state/'evaluations'/eid/'manifest.json');m['purpose']='independent'
        write(ops.state/'evaluations'/eid/'manifest.json',m)
        version=ops.active()['rf']
        with self.assertRaisesRegex(ValueError,'변경'):ops.evaluate(version,eid)


class Phase1Tests(unittest.TestCase):
    """2026-10-05 team review 2 / phase 1 (P0): R01 drift ledger recovery, R02 completed-batch link protection,
    R03 exact_k-only selection/promotion, product ID contract, read-only baseline status. Synthetic fixtures."""
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
    def tearDown(self):self.tmp.cleanup()
    def ops(self,name,dataset='rg3',**policy):
        base=dict(min_window_rows=40,min_window_patterns=30,bootstrap_repeats=40);base.update(policy)
        ops=Operations(dataset,Path(self.tmp.name)/name,base)
        x=pd.DataFrame(np.random.default_rng(31).normal(size=(120,24)),columns=feature_columns());y=(x.iloc[:,0]>0).astype(int)
        v=ops.save_candidate(fit_supervised(dataset,x,y,kind='rf' if dataset=='cn7' else 'lr'),x,y,source={'fixture':True})
        ops.initialize(v,'synthetic');return ops
    def batch(self,seed,n=45,ids=None,shift=0.):
        x=pd.DataFrame(np.random.default_rng(seed).normal(size=(n,24))+shift,columns=feature_columns())
        f=x.assign(PassOrFail=(x.iloc[:,0]>shift).astype(int))
        return f.assign(product_id=ids) if ids is not None else f
    def fail_write(self,module,match,nth=1):
        import importlib
        mod=importlib.import_module(module);real=mod.write;state={'n':0}
        def bad(path,value):
            if match(Path(path).name):
                state['n']+=1
                if state['n']==nth:raise OSError('injected '+Path(path).name)
            return real(path,value)
        return patch.object(mod,'write',side_effect=bad)
    def monitor(self,ops):return read(ops.state/'monitor.json',{'consumed':[],'history':[]})

    # ---- R01 ----
    def test_r01_monitor_failure_then_resume_does_not_reconsume(self):
        ops=self.ops('r01a')
        with self.fail_write('pipeline_runtime',lambda n:n=='monitor.json'):
            r=ops.ingest(self.batch(1),batch_id='M1',label_source='s',coordinates_confirmed=True)
        self.assertEqual(r['processing']['stage'],'drift_failed')
        self.assertEqual(len(list((ops.state/'drift').glob('drift-*.json'))),1)   # record saved, ledger not
        for _ in range(3):self.assertEqual(ops.resume('M1')['processing']['stage'] if _==0 else ops.resume('M1')['processing']['retry'],
                                           'completed' if _==0 else 'already_completed')
        m=self.monitor(ops);self.assertEqual(m['consumed'],['M1']);self.assertEqual(len(m['history']),1)
        r2=ops.ingest(self.batch(2),batch_id='M2',label_source='s',coordinates_confirmed=True)
        self.assertEqual(r2['drift']['batch_ids'],['M2'])
        m=self.monitor(ops);self.assertEqual(m['consumed'],['M1','M2']);self.assertEqual(len(m['history']),2)
        self.assertEqual(len({h['id'] for h in m['history']}),2)
    def test_r01_next_detect_repairs_ledger_even_without_resume(self):
        ops=self.ops('r01b')
        with self.fail_write('pipeline_runtime',lambda n:n=='monitor.json'):
            ops.ingest(self.batch(3),batch_id='M1',label_source='s',coordinates_confirmed=True)
        r2=ops.ingest(self.batch(4),batch_id='M2',label_source='s',coordinates_confirmed=True)
        self.assertEqual(r2['drift']['batch_ids'],['M2'])
        h=self.monitor(ops)['history'];self.assertTrue(h[0].get('recovered'));self.assertEqual(len(h),2)
        self.assertEqual(ops.resume('M1')['processing']['stage'],'completed');self.assertEqual(len(self.monitor(ops)['history']),2)
    def test_r01_drift_record_failure_reruns_detection_once(self):
        ops=self.ops('r01c')
        with self.fail_write('pipeline_runtime',lambda n:n.startswith('drift-')):
            self.assertEqual(ops.ingest(self.batch(5),batch_id='M1',label_source='s',coordinates_confirmed=True)['processing']['stage'],'drift_failed')
        self.assertFalse(list((ops.state/'drift').glob('drift-*.json')));self.assertEqual(self.monitor(ops)['consumed'],[])
        self.assertEqual(ops.resume('M1')['processing']['stage'],'completed')
        m=self.monitor(ops);self.assertEqual((m['consumed'],len(m['history'])),(['M1'],1))
    def test_r01_processing_failure_after_drift_reuses_record(self):
        ops=self.ops('r01d')
        with self.fail_write('decision_runtime',lambda n:n=='processing.json',nth=2):    # 1st = commit, 2nd = after drift
            with self.assertRaises(OSError):ops.ingest(self.batch(6),batch_id='M1',label_source='s',coordinates_confirmed=True)
        self.assertEqual(ops.pending_batches()[0]['stage'],'drift_pending')
        self.assertEqual(ops.resume('M1')['processing']['stage'],'completed')
        self.assertEqual(len(list((ops.state/'drift').glob('drift-*.json'))),1);self.assertEqual(len(self.monitor(ops)['history']),1)
    def test_r01_streak_does_not_cross_alarm_policy(self):
        ops=self.ops('r01e')
        r1=ops.ingest(self.batch(7,shift=9),batch_id='S1',coordinates_confirmed=True,label_source='s')
        self.assertEqual(r1['drift']['status'],'watch')
        ops2=Operations('rg3',ops.state.parent,dict(min_window_rows=40,min_window_patterns=30,bootstrap_repeats=40,drift_quantile=.95))
        r2=ops2.ingest(self.batch(8,shift=9),batch_id='S2',coordinates_confirmed=True,label_source='s')
        self.assertEqual((r2['drift']['status'],r2['drift']['streak']),('watch',1))        # policy changed: streak restarts
        r3=ops2.ingest(self.batch(9,shift=9),batch_id='S3',coordinates_confirmed=True,label_source='s')
        self.assertEqual((r3['drift']['status'],r3['drift']['streak']),('review',2))

    # ---- R02 / product ID contract ----
    def test_r02_completed_batch_link_is_protected(self):
        ops=self.ops('r02a');ids=[f'P-{i}' for i in range(45)];f=self.batch(10,ids=ids)
        first=ops.ingest(f,batch_id='B1',label_source='s',coordinates_confirmed=True)
        self.assertEqual(first['processing']['stage'],'completed')
        proc=read(ops.state/'batches'/'B1'/'processing.json');self.assertEqual(proc['id_mode'],'explicit_product_id')
        same=ops.ingest(f.sample(frac=1,random_state=0).reset_index(drop=True),batch_id='B1',label_source='s',coordinates_confirmed=True)
        self.assertEqual(same['processing']['retry'],'already_completed')                 # reorder only
        cases={'ids replaced':f.assign(product_id=[f'X-{i}' for i in range(45)])}
        swap=f.copy();swap.loc[[0,1],'product_id']=[ids[1],ids[0]];cases['inputs swapped between IDs']=swap
        relabel=f.copy();relabel.loc[0,'PassOrFail']=1-relabel.loc[0,'PassOrFail'];cases['label changed']=relabel
        for name,frame in cases.items():
            r=ops.ingest(frame,batch_id='B1',label_source='s',coordinates_confirmed=True)
            self.assertEqual(r['status'],'conflict',name)
        bad=ops.ingest(f,batch_id='B1',label_source='s',coordinates_confirmed=False)        # cannot compare -> no success
        self.assertEqual(bad['status'],'rejected')
        stored=pd.read_csv(ops.state/'batches'/'B1'/'products.csv');self.assertEqual(sorted(stored.record_id),sorted(ids))
        self.assertEqual(read(ops.state/'batch_ledger.json')['batches'],['B1'])
    def test_r02_interrupted_batch_rejects_id_wise_swap(self):
        ops=self.ops('r02b');ids=[f'P-{i}' for i in range(45)];f=self.batch(11,ids=ids)
        with self.fail_write('pipeline_runtime',lambda n:n=='manifest.json'):
            with self.assertRaises(OSError):ops.ingest(f,batch_id='B2',label_source='s',coordinates_confirmed=True)
        swap=f.copy();swap.loc[[0,1],'product_id']=[ids[1],ids[0]]
        self.assertEqual(ops.ingest(swap,batch_id='B2',label_source='s',coordinates_confirmed=True)['status'],'conflict')
        self.assertEqual(ops.ingest(f,batch_id='B2',label_source='s',coordinates_confirmed=True)['processing']['stage'],'completed')
    def test_r02_no_id_batch_uses_compat_path(self):
        ops=self.ops('r02c');f=self.batch(12)
        ops.ingest(f,batch_id='N1',label_source='s',coordinates_confirmed=True)
        self.assertEqual(read(ops.state/'batches'/'N1'/'processing.json')['id_mode'],'no_product_id_compat')
        r=ops.ingest(f.iloc[::-1].reset_index(drop=True),batch_id='N1',label_source='s',coordinates_confirmed=True)
        self.assertEqual(r['processing']['retry'],'already_completed')
        self.assertEqual(ops.ingest(f,batch_id='N2',label_source='s',coordinates_confirmed=True)['status'],'held')
    def test_same_input_new_production_is_collected_and_patterns_merge(self):
        ops=self.ops('r02d');f=self.batch(13)
        ops.ingest(f.assign(product_id=[f'A-{i}' for i in range(45)]),batch_id='P1',label_source='s',coordinates_confirmed=True)
        r=ops.ingest(f.assign(product_id=[f'B-{i}' for i in range(45)]),batch_id='P2',label_source='s',coordinates_confirmed=True)
        self.assertEqual(r['status'],'accepted')
        prods=pd.read_csv(ops.state/'batches'/'P2'/'products.csv');pats=pd.read_csv(ops.state/'batches'/'P2'/'patterns.csv')
        self.assertEqual(len(prods),45);self.assertEqual(int(pats.product_count.sum()),45)
        self.assertEqual(pats.fingerprint.nunique(),len(pats))

    def test_r02_legacy_completed_batch_is_checked_against_stored_products(self):
        ops=self.ops('r02e');ids=[f'P-{i}' for i in range(45)];f=self.batch(14,ids=ids)
        ops.ingest(f,batch_id='L1',label_source='s',coordinates_confirmed=True)
        from pipeline_runtime import write
        path=ops.state/'batches'/'L1'/'processing.json';proc=read(path)
        for k in ('link_sha256','id_mode'):proc.pop(k)                       # record from before link hashes existed
        write(path,proc)
        self.assertEqual(ops.ingest(f.assign(product_id=[f'X-{i}' for i in range(45)]),batch_id='L1',label_source='s',
                                    coordinates_confirmed=True)['status'],'conflict')
        swap=f.copy();swap.loc[[0,1],'product_id']=[ids[1],ids[0]]
        self.assertEqual(ops.ingest(swap,batch_id='L1',label_source='s',coordinates_confirmed=True)['status'],'conflict')
        self.assertEqual(ops.ingest(f.drop(columns='product_id'),batch_id='L1',label_source='s',coordinates_confirmed=True)['status'],'conflict')
        self.assertEqual(ops.ingest(f.iloc[::-1].reset_index(drop=True),batch_id='L1',label_source='s',
                                    coordinates_confirmed=True)['processing']['retry'],'already_completed')
        (ops.state/'batches'/'L1'/'products.csv').write_text('tampered',encoding='utf-8')   # evidence gone
        self.assertEqual(ops.ingest(f,batch_id='L1',label_source='s',coordinates_confirmed=True)['status'],'rejected')
    def test_contract_a_rejects_invalid_product_ids(self):
        ops=self.ops('ida');f=self.batch(15)
        cases={'empty':['']+[f'P-{i}' for i in range(1,45)],'blank':['  ']+[f'P-{i}' for i in range(1,45)],
               'missing mixed':[None]+[f'P-{i}' for i in range(1,45)],'nan mixed':[np.nan]+[f'P-{i}' for i in range(1,45)],
               'float':[float(i)+.5 for i in range(45)],'padded':[' P-0']+[f'P-{i}' for i in range(1,45)]}
        for n,(name,ids) in enumerate(cases.items()):
            r=ops.ingest(f.assign(product_id=pd.Series(ids,dtype=object)),batch_id=f'BAD{n}',label_source='s',coordinates_confirmed=True)
            self.assertEqual(r['status'],'held',name);self.assertIn('유효한 생산 제품 ID가 아님',r['reason'],name)
        self.assertEqual(read(ops.state/'dedup_index.json',{'reservations':{}})['reservations'],{})
        self.assertFalse((ops.state/'batch_ledger.json').exists())
        ok=ops.ingest(f.assign(product_id=list(range(1000,1045))),batch_id='GOOD',label_source='s',coordinates_confirmed=True)
        self.assertEqual(ok['status'],'accepted')                                        # integer IDs are valid
        bad=ops.ingest(f.assign(product_id=['']*45),batch_id='GOOD',label_source='s',coordinates_confirmed=True)
        self.assertEqual(bad['status'],'rejected')                                       # existing batch untouched

    # ---- R03 ----
    def test_r03_all_ties_is_reference_only(self):
        from decision_runtime import budget_metrics
        m=budget_metrics(np.array([1]*5+[0]*5),[.5]*10,.1,'all_ties',[f'f{i}' for i in range(10)])
        self.assertEqual((m['k'],m['inspected'],m['over_budget'],m['found_TP']),(1,10,9,5));self.assertTrue(m['reference_only'])
        ops=self.ops('r03',dataset='cn7',cn7_tie_rule='all_ties')
        with self.assertRaisesRegex(ValueError,'exact_k'):ops.select_cn7(['a','b','c'],'e')
        ex=pd.DataFrame(np.random.default_rng(90).normal(size=(200,24)),columns=feature_columns())
        eid=ops.register_evaluation(ex.assign(PassOrFail=(ex.iloc[:,0]>1.3).astype(int)),'s')
        x=pd.DataFrame(np.random.default_rng(91).normal(size=(160,24)),columns=feature_columns());y=(x.iloc[:,0]>1.3).astype(int)
        cand=ops.save_candidate(fit_supervised('cn7',x,y,kind='rf'),x,y,source={},parent=ops.active()['rf'])
        a=ops.evaluate(cand,eid);self.assertFalse(a['passed']);self.assertTrue(any('all_ties' in r for r in a['reasons']))
        cm=a['candidate_budget_metrics'];self.assertLessEqual(cm['capture_of_reachable'] or 0,1.0)
        from pipeline_runtime import write
        a['passed']=True;a['reasons']=[];folder=ops.state/'assessments'/a['id'];write(folder/'assessment.json',a)
        with self.assertRaisesRegex(ValueError,'exact_k'):ops.promote(a['id'])
    def test_exact_k_small_batch_and_untied(self):
        from decision_runtime import budget_flags
        flag,_,k,_,ties=budget_flags([.9,.1,.5],.1,['a','b','c'],fingerprints=['x','y','z'])
        self.assertEqual((k,list(flag),ties['ties_cut']),(1,[True,False,False],False))

    # ---- baseline status (read-only) ----
    def test_baseline_status_is_read_only_and_detects_mismatch(self):
        ops=self.ops('bs');st=ops.baseline_status();self.assertFalse(st['valid'])
        self.assertFalse((ops.state/'baseline_pointer.json').exists())                       # did not create one
        ops.create_baseline();self.assertTrue(ops.baseline_status()['valid'])
        from pipeline_runtime import write
        reg=read(ops.state/'registry.json');reg['active_models']['lr']='lr-other';write(ops.state/'registry.json',reg)
        self.assertIn('기준선 모델 버전이 현재 운영 모델과 다름',ops.baseline_status()['reasons'])


class ResendMatrixTests(unittest.TestCase):
    """Re-send comparison over every combination instead of one reported case (2026-10-05, after Codex re-review).
    ID source x resend variant x processing stage (+ legacy records, + evidence file state)."""
    SOURCES=['record_id','product_id','Unnamed: 0','none']
    # variant -> expected outcome per contract (A = record_id/product_id, C = Unnamed: 0 / none). None = not applicable.
    EXPECT={'identical':('ok','ok'),'reordered':('ok','ok'),'ids_replaced':('conflict','ok'),'ids_swapped':('conflict','ok'),
            'label_changed':('conflict','conflict'),'id_column_removed':('conflict','ok'),'id_column_added':(None,'conflict')}
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True);self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tmp');self.n=0
    def tearDown(self):self.tmp.cleanup()
    def fresh(self):
        self.n+=1
        ops=Operations('rg3',Path(self.tmp.name)/f's{self.n}',dict(min_window_rows=10**6))   # drift waits: fast
        x=pd.DataFrame(np.random.default_rng(31).normal(size=(120,24)),columns=feature_columns());y=(x.iloc[:,0]>0).astype(int)
        ops.initialize(ops.save_candidate(fit_supervised('rg3',x,y),x,y,source={}),'synthetic');return ops
    def frame(self,source):
        x=pd.DataFrame(np.random.default_rng(55).normal(size=(30,24)),columns=feature_columns())
        x.iloc[1]=x.iloc[0].to_numpy().copy()
        y=(x.iloc[:,0]>0).astype(int).to_numpy().copy();y[0],y[1]=0,1
        f=x.assign(PassOrFail=y)
        if source=='Unnamed: 0':f.insert(0,'Unnamed: 0',range(30))
        elif source!='none':f[source]=[f'{source[0].upper()}-{i}' for i in range(30)]
        return f
    def variant(self,f,source,name):
        idc=None if source=='none' else source;g=f.copy()
        if name=='identical':return g
        if name=='reordered':return g.sample(frac=1,random_state=1).reset_index(drop=True)
        if name=='ids_replaced':
            if idc is None:return None
            g[idc]=[f'Z-{i}' for i in range(30)] if idc!='Unnamed: 0' else list(range(100,130));return g
        if name=='ids_swapped':
            if idc is None:return None
            v=g[idc].tolist();v[2],v[3]=v[3],v[2];g[idc]=v;return g
        if name=='label_changed':g.loc[2,'PassOrFail']=1-g.loc[2,'PassOrFail'];return g
        if name=='id_column_removed':return None if idc is None else g.drop(columns=idc)
        if name=='id_column_added':return g.assign(product_id=[f'N-{i}' for i in range(30)])
    def expected(self,source,name):
        return self.EXPECT[name][0 if source in ('record_id','product_id') else 1]
    def stage_batch(self,ops,f,stage):
        """Bring batch B into the given stage."""
        import importlib
        if stage=='completed':
            self.assertEqual(ops.ingest(f,batch_id='B',label_source='s',coordinates_confirmed=True)['status'],'accepted');return
        if stage in ('drift_failed','advice_failed'):
            target='_run_detect' if stage=='drift_failed' else '_advise'
            with patch.object(ops,target,side_effect=RuntimeError('injected')):
                r=ops.ingest(f,batch_id='B',label_source='s',coordinates_confirmed=True)
            self.assertEqual(r['processing']['stage'],stage);return
        mod,name=('decision_runtime','processing.json') if stage=='commit_incomplete' else ('pipeline_runtime','manifest.json')
        mod=importlib.import_module(mod);real=mod.write;state={'n':0}
        def bad(path,value):
            if Path(path).name==name and state['n']==0:state['n']+=1;raise OSError('injected')
            return real(path,value)
        with patch.object(mod,'write',side_effect=bad):
            with self.assertRaises(OSError):ops.ingest(f,batch_id='B',label_source='s',coordinates_confirmed=True)
        self.assertEqual([p['stage'] for p in ops.pending_batches()],[stage])
    def outcome(self,r):
        return 'ok' if r['status']=='accepted' else r['status']
    def test_matrix_by_source_variant_stage(self):
        checked=0
        for stage in ['completed','commit_incomplete','reserved_not_stored','drift_failed','advice_failed']:
            for source in self.SOURCES:
                for name in self.EXPECT:
                    exp=self.expected(source,name);f=self.frame(source);g=self.variant(f,source,name)
                    if exp is None or g is None:continue
                    with self.subTest(stage=stage,source=source,variant=name):
                        ops=self.fresh();self.stage_batch(ops,f,stage)
                        before={str(p):p.read_bytes() for p in (ops.state/'batches'/'B').glob('*') if p.is_file()}
                        r=ops.ingest(g,batch_id='B',label_source='s',coordinates_confirmed=True)
                        self.assertEqual(self.outcome(r),exp,r.get('reason'))
                        if stage=='completed' and exp=='ok':self.assertEqual(r['processing']['retry'],'already_completed')
                        if exp!='ok' or stage=='completed':
                            self.assertEqual(before,{str(p):p.read_bytes() for p in (ops.state/'batches'/'B').glob('*') if p.is_file()})
                        self.assertEqual(read(ops.state/'batch_ledger.json',{'batches':[]})['batches'].count('B'),
                                         0 if (stage=='reserved_not_stored' and exp!='ok') else 1)
                        checked+=1
        self.assertEqual(checked,115)
    def test_matrix_legacy_records_without_link_or_mode(self):
        from pipeline_runtime import write
        for source in self.SOURCES:
            for name in self.EXPECT:
                exp=self.expected(source,name);f=self.frame(source);g=self.variant(f,source,name)
                if exp is None or g is None:continue
                with self.subTest(source=source,variant=name):
                    ops=self.fresh();ops.ingest(f,batch_id='B',label_source='s',coordinates_confirmed=True)
                    path=ops.state/'batches'/'B'/'processing.json';proc=read(path)
                    for k in ('link_sha256','id_mode'):proc.pop(k,None)
                    write(path,proc)
                    self.assertEqual(self.outcome(ops.ingest(g,batch_id='B',label_source='s',coordinates_confirmed=True)),exp)
    def test_evidence_file_states_are_explicit_rejections(self):
        import shutil
        for state in ['missing','corrupted','directory','manifest_missing','read_denied']:
            for source in self.SOURCES:
                with self.subTest(state=state,source=source):
                    ops=self.fresh();f=self.frame(source);ops.ingest(f,batch_id='B',label_source='s',coordinates_confirmed=True)
                    folder=ops.state/'batches'/'B';pc=folder/'products.csv'
                    if state=='missing':pc.unlink()
                    elif state=='corrupted':pc.write_text('x',encoding='utf-8')
                    elif state=='directory':pc.unlink();pc.mkdir()
                    elif state=='manifest_missing':
                        m=read(folder/'manifest.json');m['files'].pop('products.csv')
                        from pipeline_runtime import write;write(folder/'manifest.json',m)
                    if state=='read_denied':
                        import decision_runtime
                        real=decision_runtime.digest
                        def denied(path):
                            if Path(path).name=='products.csv':raise PermissionError('injected read denial')
                            return real(path)
                        with patch.object(decision_runtime,'digest',side_effect=denied):
                            r=ops.ingest(f,batch_id='B',label_source='s',coordinates_confirmed=True)
                    else:r=ops.ingest(f,batch_id='B',label_source='s',coordinates_confirmed=True)
                    self.assertEqual(r['status'],'rejected');self.assertIn('비교하지 못했습니다',r['reason'])

    def test_manifest_damage_never_reregisters_committed_batch(self):
        from pipeline_runtime import write
        for stage in ('completed','commit_incomplete'):
            for source in self.SOURCES:
                for damage in ('missing','invalid_json','wrong_type','directory'):
                    with self.subTest(stage=stage,source=source,damage=damage):
                        ops=self.fresh();f=self.frame(source);self.stage_batch(ops,f,stage)
                        path=ops.state/'batches/B/manifest.json'
                        if damage=='missing':path.unlink()
                        elif damage=='directory':path.unlink();path.mkdir()
                        elif damage=='wrong_type':write(path,[])
                        else:path.write_text('{broken',encoding='utf-8')
                        ledger=read(ops.state/'batch_ledger.json');index=read(ops.state/'dedup_index.json')
                        r=ops.ingest(f,batch_id='B',label_source='s',coordinates_confirmed=True)
                        self.assertEqual(r['status'],'rejected')
                        self.assertEqual(read(ops.state/'batch_ledger.json'),ledger)
                        self.assertEqual(read(ops.state/'dedup_index.json'),index)

    def test_legacy_contract_requires_positive_evidence(self):
        from pipeline_runtime import write
        for source in self.SOURCES:
            for damage in ('missing_index','empty_ids','partial_ids','invalid_json','wrong_type'):
                with self.subTest(source=source,damage=damage):
                    ops=self.fresh();f=self.frame(source);ops.ingest(f,'B','s',True)
                    proc_path=ops._processing_path('B');proc=read(proc_path)
                    for k in ('id_mode','link_sha256'):proc.pop(k,None)
                    write(proc_path,proc)
                    manifest_path=ops.state/'batches/B/manifest.json';manifest=read(manifest_path)
                    manifest.pop('id_mode',None);write(manifest_path,manifest)
                    index_path=ops.state/'dedup_index.json';index=read(index_path)
                    if damage=='missing_index':index_path.unlink()
                    elif damage=='invalid_json':index_path.write_text('{broken',encoding='utf-8')
                    elif damage=='wrong_type':write(index_path,dict(ids=[]))
                    else:
                        index['ids']=dict(list(index['ids'].items())[:1]) if damage=='partial_ids' else {}
                        write(index_path,index)
                    for g in (f,f.drop(columns=source) if source!='none' else f):
                        self.assertEqual(ops.ingest(g,'B','s',True)['status'],'rejected')

    def test_contract_source_survives_proc_and_index_loss(self):
        from pipeline_runtime import write
        for source in self.SOURCES:
            with self.subTest(source=source):
                ops=self.fresh();f=self.frame(source);ops.ingest(f,'B','s',True)
                proc=read(ops._processing_path('B'))
                for k in ('id_mode','link_sha256'):proc.pop(k,None)
                write(ops._processing_path('B'),proc)
                (ops.state/'dedup_index.json').unlink()
                r=ops.ingest(f.iloc[::-1].reset_index(drop=True),'B','s',True)
                self.assertEqual(r['processing']['retry'],'already_completed')

    def test_label_presence_and_same_pattern_label_swap(self):
        for source in self.SOURCES:
            for variant in ('add_label','remove_label','swap_labels'):
                with self.subTest(source=source,variant=variant):
                    ops=self.fresh();f=self.frame(source)
                    if variant=='add_label':original=f.drop(columns='PassOrFail');changed=f
                    elif variant=='remove_label':original=f;changed=f.drop(columns='PassOrFail')
                    else:
                        original=f;changed=f.copy();changed.loc[[0,1],'PassOrFail']=[1,0]
                    ops.ingest(original,'B','s',True)
                    r=ops.ingest(changed,'B','s',True)
                    expected='ok' if variant=='swap_labels' and source in ('none','Unnamed: 0') else 'conflict'
                    self.assertEqual(self.outcome(r),expected)

if __name__=='__main__':unittest.main(verbosity=2)
