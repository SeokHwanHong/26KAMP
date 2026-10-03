"""Reinspection and absolute acceptance checks; synthetic cases are technical tests."""
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
        self.assertEqual(info['budget_k'],4);self.assertGreaterEqual(info['flagged'],4)
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

if __name__=='__main__':unittest.main(verbosity=2)
