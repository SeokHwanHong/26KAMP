"""All five approved recommendations, isolated fixtures and real code execution."""
from pathlib import Path
import sys,tempfile,unittest,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from governance_runtime import Operations
from pipeline_runtime import ROOT,read,write,fit_supervised,feature_columns,score,fingerprints
from test_workflow_runtime import sample
import pandas as pd
import numpy as np


class GovernanceTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True);self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='governance_test_');self.root=Path(self.tmp.name)
    def tearDown(self):
        assert self.root.resolve().is_relative_to((ROOT/'tmp').resolve());self.tmp.cleanup()
    def ops(self,ds='rg3',configured=True):
        o=Operations(ds,self.root,dict(min_window_rows=40,min_window_patterns=30,bootstrap_repeats=20,min_train_normal=10,min_train_risk=3,min_eval_risk=3))
        x,y=sample();b=fit_supervised(ds,x,y,C=100);b['threshold']=1
        old=o.save_candidate(b,x,y,source={'synthetic':True});o.initialize(old,'synthetic initial');o.create_baseline()
        if configured:self.configure(o)
        return o,x,y,old
    def configure(self,o,selection=None,sampling=True):
        settings=dict(approvers={a:['test-reviewer'] for a in ['ct','promote','reference','rollback']},training_selection=selection or {'mode':'cumulative'})
        if sampling:settings['rg3_sampling']={a:dict(fraction=.2,random_share=.4) for a in ['waiting','normal','watch','review']}
        return o.configure_governance(settings,'test-owner','synthetic policy')
    def frame(self,seed,n=90,labelled=True,normal=False,time='2026-10-06T09:00:00+09:00'):
        x,y=sample(n,seed,shift=9);x['record_id']=[f'P{seed}-{i}' for i in range(n)]
        if labelled:x['PassOrFail']=0 if normal else y
        x['production_time']=time;return x
    def ingest(self,o,f,bid):return o.ingest(f,batch_id=bid,label_source='synthetic inspection' if 'PassOrFail' in f else None,coordinates_confirmed=True,
                                            production_time_column='production_time',process_version='P1',metadata_source='synthetic source')
    def evidence(self):return dict(normal_process_confirmed=True,input_error_excluded=True,changed_representations=['Injection_Time'],label_source='synthetic inspection',evaluation_plan='separate holdout')

    def test_missing_configuration_holds_plan_and_blocks_approval(self):
        o,_,_,_=self.ops(configured=False);r=self.ingest(o,self.frame(1),'B')
        self.assertEqual(r['inspection_plan']['status'],'held_configuration')
        with self.assertRaisesRegex(ValueError,'담당자'):o.approve_action('ct',{},self.evidence(),'test-reviewer')

    def test_sampling_exact_budget_disjoint_and_reproducible(self):
        o,_,_,_=self.ops();f=self.frame(2);r=self.ingest(o,f,'B');p=r['inspection_plan']
        self.assertEqual(p['budget_k'],18);self.assertEqual(len(p['selected_ids']),18)
        self.assertFalse(set(p['priority_ids'])&set(p['random_ids']))
        self.assertEqual(len(p['random_ids']),8)
        initial=read(o.state/'batches/B/advice_latest.json')
        history=o.state/'batches/B/advice_history'/initial['revision']/'inspection_plan.json'
        self.assertEqual(read(history),p)
        self.assertEqual(o.ingest(f.iloc[::-1],batch_id='B',label_source='synthetic inspection',coordinates_confirmed=True,
                                  production_time_column='production_time',process_version='P1',metadata_source='synthetic source')['processing']['retry'],'already_completed')

    def test_real_time_requires_timezone_and_original_ids(self):
        o,_,_,_=self.ops();f=self.frame(3);f['production_time']='2026-10-06 09:00:00'
        with self.assertRaises(ValueError):self.ingest(o,f,'BAD')
        with self.assertRaises(ValueError):self.ingest(o,self.frame(4).drop(columns='record_id'),'BAD2')
        self.assertFalse((o.state/'batches/BAD').exists())

    def test_approval_binding_rejects_policy_and_input_changes(self):
        o,_,_,_=self.ops();r=self.ingest(o,self.frame(5),'B');payload=dict(kind='lr',batch_ids=['B'],drift_id=r['drift']['id'],cause='verified change',base_version=None)
        aid=o.approve_action('ct',payload,self.evidence(),'test-reviewer');self.configure(o)
        with self.assertRaisesRegex(ValueError,'재승인'):o.retrain('lr',['B'],r['drift']['id'],'verified change',approval_id=aid)

    def test_recent_selection_preserves_risk_history_without_inventing_time(self):
        o,_,_,old=self.ops();f1=self.frame(6,time='2026-10-01T09:00:00+09:00');self.ingest(o,f1,'OLD')
        f2=self.frame(7);risk_row=f1.index[f1.PassOrFail==1][0]
        f2.loc[0,feature_columns()]=f1.loc[risk_row,feature_columns()];f2.loc[0,'PassOrFail']=0
        r=self.ingest(o,f2,'NEW');self.assertEqual(r['drift']['status'],'review')
        selection=dict(mode='recent',start='2026-10-05T00:00:00+09:00',end='2026-10-07T00:00:00+09:00',process_version='P1',max_patterns=100)
        self.configure(o,selection)
        payload=dict(kind='lr',batch_ids=['OLD','NEW'],drift_id=r['drift']['id'],cause='verified change',base_version=None)
        aid=o.approve_action('ct',payload,self.evidence(),'test-reviewer');v=o.retrain('lr',['OLD','NEW'],r['drift']['id'],'verified change',approval_id=aid)
        b,m=o.load_model(v);self.assertEqual(m['parent'],old)
        self.assertLessEqual(len(b['corpus']),90)
        fp=fingerprints(f1.loc[[risk_row],feature_columns()])[0];corpus=b['corpus']
        actual=dict(zip(fingerprints(corpus[feature_columns()]),corpus.label));self.assertEqual(actual[fp],1)
        self.assertEqual(read(o.state/'governance/training_records'/f'{v}.json')['selection'],selection)

    def test_reference_update_is_approved_and_survives_model_rebuild(self):
        o,_,_,_=self.ops();initial=read(o.state/'baseline_pointer.json')['id'];self.ingest(o,self.frame(8,normal=True),'NORMAL')
        payload=dict(batch_ids=['NORMAL'],reason='confirmed normal process')
        aid=o.approve_action('reference',payload,self.evidence(),'test-reviewer')
        o.update_reference(['NORMAL'],'confirmed normal process',approval_id=aid)
        pointer=read(o.state/'baseline_pointer.json')['id'];self.assertNotEqual(pointer,initial)
        self.assertEqual(o.create_baseline(),pointer)
        ref=read(o.state/'baselines'/pointer/'manifest.json')['reference_sha256']
        rebuilt=o._build_baseline(o.active())
        self.assertEqual(read(o.state/'baselines'/rebuilt/'manifest.json')['reference_sha256'],ref)
        previous=read(o.state/'baselines'/pointer/'manifest.json')['reference_source']['previous']
        approval=o.approve_action('reference',dict(batch_ids=[],reason='reference rollback',rollback_baseline=previous),self.evidence(),'test-reviewer')
        o.rollback_reference('reference rollback',approval_id=approval)
        self.assertTrue(o.baseline_status()['valid'])

    def test_reference_cannot_absorb_risky_batch(self):
        o,_,_,_=self.ops();self.ingest(o,self.frame(9),'RISK')
        aid=o.approve_action('reference',dict(batch_ids=['RISK'],reason='change'),self.evidence(),'test-reviewer')
        with self.assertRaises(ValueError):o.update_reference(['RISK'],'change',approval_id=aid)

    def test_cumulative_ct_requires_approval_and_preserves_parent_parameters(self):
        o,_,_,old=self.ops();self.ingest(o,self.frame(20),'C1');r=self.ingest(o,self.frame(21),'C2')
        with self.assertRaises(ValueError):o.retrain('lr',['C1','C2'],r['drift']['id'],'confirmed process change')
        payload=dict(kind='lr',batch_ids=['C1','C2'],drift_id=r['drift']['id'],cause='confirmed process change',base_version=None)
        aid=o.approve_action('ct',payload,self.evidence(),'test-reviewer')
        new=o.retrain('lr',['C1','C2'],r['drift']['id'],'confirmed process change',approval_id=aid)
        b,_=o.load_model(new);parent,_=o.load_model(old)
        self.assertEqual(b['model'].steps[-1][1].get_params(),parent['model'].steps[-1][1].get_params())
        self.assertEqual(b['threshold'],parent['threshold'])

    def test_sampling_does_not_use_provided_labels_and_tiny_budget_holds(self):
        o,_,_,_=self.ops();self.ingest(o,self.frame(22),'B')
        original=read(o.state/'batches/B/inspection_plan.json')
        m=read(o.state/'batches/B/manifest.json');drift=read(o._processing_path('B'))['drift'];p=o.state/'batches/B/products.csv'
        products=pd.read_csv(p);products['label']=1-products.label;products.to_csv(p,index=False)
        o._rg3_advice(dict(m,drift=drift))
        self.assertEqual(read(o.state/'batches/B/inspection_plan.json')['selected_ids'],original['selected_ids'])
        settings=dict(approvers={a:['test-reviewer'] for a in ['ct','promote','reference','rollback']},training_selection={'mode':'cumulative'},
                      rg3_sampling={a:dict(fraction=.001,random_share=.4) for a in ['waiting','normal','watch','review']})
        o.configure_governance(settings,'test-owner','tiny synthetic budget')
        o._rg3_advice(dict(m,drift=drift))
        self.assertEqual(read(o.state/'batches/B/inspection_plan.json')['status'],'held_budget')

    def test_cn7_product_evidence_and_approval_gate_model_role(self):
        o=Operations('cn7',self.root,dict(min_eval_risk=3));self.configure(o,sampling=False)
        x,y=sample();old=fit_supervised('cn7',x,y,kind='rf');old['model'].steps[-1][1].set_params(n_estimators=20,max_depth=3,min_samples_leaf=1)
        old['model'].fit(x,1-y);ov=o.save_candidate(old,x,y,source={'synthetic_deliberately_weak_model':True});o.initialize(ov,'synthetic weak comparator');o.create_baseline()
        new=fit_supervised('cn7',x,y,kind='rf');new['model'].steps[-1][1].set_params(n_estimators=20,max_depth=3,min_samples_leaf=1);new['model'].fit(x,y)
        nv=o.save_candidate(new,x,y,source={'synthetic':True},parent=ov);ex,ey=sample(180,22)
        eid=o.register_evaluation(ex.assign(PassOrFail=ey),'synthetic independent');a=o.evaluate(nv,eid);self.assertTrue(a['passed'],a['reasons'])
        with self.assertRaises(ValueError):o.promote(a['id'])
        products=ex.loc[ex.index.repeat(2)].reset_index(drop=True);labels=np.repeat(ey.to_numpy(),2);labels[::2]=0
        products['PassOrFail']=labels;products['record_id']=[f'E{i}' for i in range(len(products))]
        evidence=o.record_product_validation(a['id'],products,'synthetic independent product inspections')
        report=read(o.state/'governance/product_validations'/evidence/'report.json');self.assertTrue(report['passed'])
        aid=o.approve_action('promote',dict(assessment_id=a['id'],product_validation_id=evidence),dict(reason='independent inspections improve'),'test-reviewer')
        self.assertEqual(o.promote(a['id'],approval_id=aid,product_validation_id=evidence),nv)
        self.assertEqual(o.registry()['inspection_role']['version'],nv)
        undo=o.approve_action('rollback',dict(kind='rf',reason='explicit rollback'),dict(reason='approved rollback'),'test-reviewer')
        self.assertEqual(o.rollback('rf','explicit rollback',approval_id=undo),ov)

    def test_cn7_approved_family_switch_changes_actual_inspection_model(self):
        o=Operations('cn7',self.root,dict(min_eval_risk=3));self.configure(o,sampling=False);x,y=sample()
        rf=fit_supervised('cn7',x,y,kind='rf');rf['model'].steps[-1][1].set_params(n_estimators=20,max_depth=3,min_samples_leaf=1)
        rf['model'].fit(x,1-y);rv=o.save_candidate(rf,x,y,source={'synthetic_weak_model':True});o.initialize(rv,'synthetic initial inspector')
        weak=fit_supervised('cn7',x,1-y,C=100);lv=o.save_candidate(weak,x,y,source={'synthetic_weak_model':True});o.initialize(lv,'synthetic LR comparator');o.create_baseline()
        candidate=fit_supervised('cn7',x,y,C=100);v=o.save_candidate(candidate,x,y,source={'synthetic':True},parent=lv)
        ex,ey=sample(180,24);eid=o.register_evaluation(ex.assign(PassOrFail=ey),'synthetic independent')
        assessment=o.evaluate(v,eid);self.assertTrue(assessment['passed'])
        products=ex.assign(PassOrFail=ey,record_id=[f'ID{i}' for i in range(len(ex))])
        pv=o.record_product_validation(assessment['id'],products,'synthetic product inspection')
        aid=o.approve_action('promote',dict(assessment_id=assessment['id'],product_validation_id=pv),dict(reason='verified product improvement'),'test-reviewer')
        o.promote(assessment['id'],approval_id=aid,product_validation_id=pv)
        self.assertEqual(o.policy['cn7_model_kind'],'lr')
        baseline=read(o.state/'baseline_pointer.json')['id'];self.assertEqual(read(o.state/'baselines'/baseline/'manifest.json')['policy']['cn7_model_kind'],'lr')
        r=o.ingest(sample(45,25)[0],coordinates_confirmed=True)
        self.assertEqual(r['inspection_priority']['model_kind'],'lr');self.assertEqual(r['inspection_priority']['model_version'],v)


if __name__=='__main__':unittest.main(verbosity=2)
