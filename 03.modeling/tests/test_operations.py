"""Real source integration tests; synthetic fixtures are NOT field validation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from pipeline_runtime import *

OUT=ROOT/'output/operations_tests/20261003'
OUT.mkdir(parents=True,exist_ok=True)


def synthetic(n,seed=42,shift=0):
    rng=np.random.default_rng(seed);a=rng.normal(size=(n,24))+shift
    x=pd.DataFrame(a,columns=feature_columns());y=(a[:,0]>shift).astype(int)
    return x,y


class OperationsTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='ops_test_')
        self.ops=Operations('rg3',Path(self.tmp.name),policy=dict(min_window_rows=40,min_window_patterns=30,bootstrap_repeats=40))
    def tearDown(self):self.tmp.cleanup()
    def seed(self,kind='lr'):
        x,y=synthetic(120)
        b=fit_supervised('rg3',x,y,kind=kind,C=100.) if kind in ('lr','rf') else fit_oneclass('rg3',x.loc[y==0],kind)
        v=self.ops.save_candidate(b,x,y,source={'fixture':'synthetic'})
        self.ops.initialize(v,'synthetic test baseline')
        return x,y,b,v
    def test_preprocess_preserves_counts_and_conflicts(self):
        x,y=synthetic(3);frame=pd.concat([x,x.iloc[[0]]],ignore_index=True)
        frame['PassOrFail']=[0,0,1,1]
        products,p=preprocess(frame,'fixture','synthetic')
        self.assertEqual(len(products),4);self.assertEqual(len(p),3)
        self.assertEqual(p.loc[p.product_count.eq(2),'label'].item(),1)
        self.assertEqual(p.product_count.sum(),4)
    def test_unaligned_unlabeled_held(self):
        self.seed();x,_=synthetic(50)
        r=self.ops.ingest(x,coordinates_confirmed=False)
        self.assertEqual(r['status'],'held')
        self.assertFalse((self.ops.state/'batch_ledger.json').exists())
    def test_insufficient_window_waits_and_does_not_consume(self):
        self.seed();x,y=synthetic(8,12)
        r=self.ops.ingest(x.assign(PassOrFail=y),label_source='synthetic',coordinates_confirmed=True)
        self.assertEqual(r['drift']['status'],'waiting')
        self.assertFalse((self.ops.state/'monitor.json').exists())
    def test_drift_persistence_and_ct_all_model_types(self):
        x,y,b,lr=self.seed()
        # Independent IF/RF/OCSVM templates, same isolated runtime.
        for kind in ['if','rf','ocsvm']:
            if kind=='ocsvm':
                xx,yy,_,dev,_,_=data('rg3');source=sorted((ROOT/'runtime/rg3/models').glob('ocsvm-*/model.joblib'))[0]
                other=joblib.load(source);other.update(kind='ocsvm',dataset='rg3')
                version=self.ops.save_candidate(other,xx.loc[dev],yy.loc[dev],source={'fixture':'legacy template'})
            else:
                other=fit_oneclass('rg3',x.loc[y==0],kind) if kind=='if' else fit_supervised('rg3',x,y,kind=kind)
                version=self.ops.save_candidate(other,x,y,source={'fixture':'synthetic'})
            self.ops.initialize(version,'synthetic baseline')
        batches=[]
        for seed in [71,72]:
            xx,yy=synthetic(45,seed,shift=9)
            r=self.ops.ingest(xx.assign(PassOrFail=yy),label_source='synthetic fixture',coordinates_confirmed=True)
            self.assertEqual(r['status'],'accepted');batches.append(r['id'])
        self.assertEqual(r['drift']['status'],'review')
        monitor=read(self.ops.state/'monitor.json')
        self.assertEqual(len(set(monitor['consumed'])),2)
        for kind in ['lr','rf','if','ocsvm']:
            v=self.ops.retrain(kind,batches,r['drift']['id'],'synthetic injected shift, not physical process evidence')
            _,m=self.ops.load_model(v);self.assertEqual(m['parent'],self.ops.active()[kind])
            self.assertNotEqual(v,self.ops.active()[kind])
        write(OUT/'drift_fixture_example.json',r['drift'])
    def test_unlabeled_ct_rejected(self):
        self.seed();ids=[]
        for seed in [81,82]:
            x,_=synthetic(45,seed,shift=9);r=self.ops.ingest(x,coordinates_confirmed=True);ids.append(r['id'])
        with self.assertRaisesRegex(ValueError,'비라벨'):self.ops.retrain('lr',ids,r['drift']['id'],'synthetic')
    def test_evaluation_cannot_overlap_training(self):
        x,y,_,_=self.seed()
        with self.assertRaisesRegex(ValueError,'학습 이력'):self.ops.register_evaluation(x.assign(PassOrFail=y),'synthetic')
    def test_candidate_cannot_train_on_registered_eval(self):
        x,y=synthetic(60,90);eid=self.ops.register_evaluation(x.assign(PassOrFail=y),'synthetic')
        b=fit_supervised('rg3',x,y)
        with self.assertRaisesRegex(ValueError,'평가 전용'):self.ops.save_candidate(b,x,y,source={'fixture':eid})
    def test_cd_promotion_rollback_and_guidance(self):
        x,y=synthetic(160,4);b=fit_supervised('rg3',x,y,C=100.)
        old=copy.deepcopy(b);old['threshold']=1.
        v0=self.ops.save_candidate(old,x,y,source={'fixture':'deliberately weak baseline'})
        self.ops.initialize(v0,'synthetic demonstration only')
        v1=self.ops.save_candidate(b,x,y,source={'fixture':'candidate'},parent=v0)
        ex,ey=synthetic(160,5);eid=self.ops.register_evaluation(ex.assign(PassOrFail=ey),'synthetic independent fixture')
        result=self.ops.evaluate(v1,eid)
        self.assertTrue(result['passed'],result)
        self.assertEqual(self.ops.promote(result['id']),v1)
        self.assertEqual(self.ops.rollback('lr','synthetic rollback exercise'),v0)
        write(OUT/'cd_fixture_example.json',result)
        guidance=read(self.ops.state/'assessments'/result['id']/'distribution_guidance.json')
        write(OUT/'rg3_guidance_fixture_example.json',guidance)
    def test_insufficient_labels_prevent_promotion(self):
        x,y,b,v=self.seed();candidate=self.ops.save_candidate(b,x,y,source={},parent=v)
        ex,ey=synthetic(20,91);eid=self.ops.register_evaluation(ex.assign(PassOrFail=ey),'synthetic')
        r=self.ops.evaluate(candidate,eid);self.assertFalse(r['passed'])
        with self.assertRaises(ValueError):self.ops.promote(r['id'])
    def test_model_tamper_rejected(self):
        _,_,_,v=self.seed();p=self.ops.state/'models'/v/'model.joblib'
        with p.open('ab') as f:f.write(b'tamper')
        with self.assertRaisesRegex(ValueError,'해시'):self.ops.load_model(v)
    def test_missing_risk_not_called_safe(self):
        x,_=synthetic(50);r=range_guidance(x,np.zeros(50),x,None)
        self.assertTrue(all(a['status']=='insufficient_support' for a in r['ranges']))
    def test_original_test_forced_historical(self):
        x,y,_,_,test,_=data('rg3')
        eid=self.ops.register_evaluation(x.loc[test].assign(PassOrFail=y.loc[test]),'provided labels')
        self.assertEqual(read(self.ops.state/'evaluations'/eid/'manifest.json')['purpose'],'historical_followup')


if __name__=='__main__':
    import io
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(OperationsTests)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (OUT/'test_log.txt').write_text(stream.getvalue(),encoding='utf-8')
    write(OUT/'summary.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                  scope='synthetic technical integration only; no real deployment'))
    print(stream.getvalue());sys.exit(0 if result.wasSuccessful() else 1)
