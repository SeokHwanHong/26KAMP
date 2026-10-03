import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from pipeline_runtime import *


def core_library():
    ns={'__name__':'registration_tests'}
    for c in read(ROOT/'03.modeling/common/00_pipeline_core.ipynb')['cells']:
        if c['cell_type']=='code' and 'pipeline_library' in c.get('metadata',{}).get('tags',[]):exec(''.join(c['source']),ns)
    return ns


class RegistrationAndLogisticTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='registration_test_')
        self.folder=Path(self.temp.name);self.ns=core_library()
    def tearDown(self):self.temp.cleanup()
    def fixture(self):
        run=self.folder/'run';run.mkdir()
        ctx=self.ns['Pipeline'](ROOT,'cn7',state_root=self.folder/'state')
        artifact=joblib.load(sorted((ROOT/'runtime/cn7/models').glob('ocsvm-*/model.joblib'))[0])
        write(run/'selection_manifest.json',{'choices':[artifact['choice']]})
        artifact['selection_sha256']=digest(run/'selection_manifest.json')
        path=run/'f1_selected_model.joblib';joblib.dump(artifact,path)
        source=ctx.data()[-1]
        write(run/'run_config.json',dict(dataset='cn7',source_sha256=source['source_sha256'],
             split_sha256=digest(ROOT/'data/processed/cn7/conservative/splits/split_assignments.csv'),
             notebook_code_sha256=ctx.code['notebook_code_digest']()))
        write(run/'postrun_audit.json',dict(passed=True,test_fixture=True))
        write(run/'verification.json',dict(artifact_sha256={p.name:digest(p) for p in run.iterdir() if p.name!='postrun_audit.json'}))
        return ctx,path
    def test_unchanged_audit_bound_artifact_registers(self):
        ctx,path=self.fixture();v=self.ns['register_ocsvm_baseline'](ctx,path)
        b,m=ctx.model(v,'ocsvm');self.assertEqual(b['threshold'],b['choice']['threshold'])
        self.assertIn('evidence',m)
    def test_changed_model_after_audit_rejected(self):
        ctx,path=self.fixture();b=joblib.load(path);b['threshold']+=1000;joblib.dump(b,path)
        with self.assertRaisesRegex(ValueError,'산출물이 변경'):self.ns['register_ocsvm_baseline'](ctx,path)
    def test_matching_hash_but_inconsistent_threshold_rejected(self):
        ctx,path=self.fixture();b=joblib.load(path);b['threshold']+=1000;joblib.dump(b,path)
        verification=read(path.parent/'verification.json');verification['artifact_sha256'][path.name]=digest(path)
        write(path.parent/'verification.json',verification)
        with self.assertRaisesRegex(ValueError,'설정·임계값'):self.ns['register_ocsvm_baseline'](ctx,path)
    def test_stale_split_rejected(self):
        ctx,path=self.fixture();run=read(path.parent/'run_config.json');run['split_sha256']='stale'
        write(path.parent/'run_config.json',run);v=read(path.parent/'verification.json')
        v['artifact_sha256']['run_config.json']=digest(path.parent/'run_config.json');write(path.parent/'verification.json',v)
        with self.assertRaisesRegex(ValueError,'데이터·분할'):self.ns['register_ocsvm_baseline'](ctx,path)
    def test_lr_all_training_rows_scaler(self):
        x,y,_,dev,test,_=data('rg3');b=fit_supervised('rg3',x.loc[dev],y.loc[dev])
        self.assertEqual(b['model'][1].n_samples_seen_,len(dev))
        self.assertEqual(set(b['model'].classes_),{0,1})
        before=b['model'][1].mean_.copy();p=score(b,x.loc[test]);score(b,x.loc[test]*1000)
        np.testing.assert_array_equal(before,b['model'][1].mean_)
        self.assertTrue(np.isfinite(p).all() and ((p>=0)&(p<=1)).all())
    def test_domain_normal_reference_alltrain_final_scaler(self):
        x,y,_,dev,_,_=data('cn7');b=fit_supervised('cn7',x.loc[dev],y.loc[dev],scenario='thermal')
        self.assertEqual(b['domain']['normal_scaler'].n_samples_seen_,int(y.loc[dev].eq(0).sum()))
        self.assertEqual(b['model'][1].n_samples_seen_,len(dev))
    def test_saved_threshold_matches_strict_rule(self):
        for ds in ['cn7','rg3']:
            folder=sorted((ROOT/'output/logistic'/ds).iterdir())[-1]
            b=joblib.load(folder/'model.joblib');x,y,_,_,test,_=data(ds)
            saved=pd.read_csv(folder/'test_predictions.csv',float_precision='round_trip')
            np.testing.assert_array_equal(score(b,x.loc[test]),saved.probability.to_numpy())
            np.testing.assert_array_equal((saved.probability>b['threshold']).astype(int),saved.prediction)
            for k,v in metrics(y.loc[test],saved.prediction).items():self.assertAlmostEqual(v,read(folder/'test_metrics.json')[k])


if __name__=='__main__':
    import io
    out=ROOT/'output/operations_tests/20261003';out.mkdir(parents=True,exist_ok=True)
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RegistrationAndLogisticTests))
    (out/'registration_logistic_tests.txt').write_text(stream.getvalue(),encoding='utf-8')
    write(out/'registration_logistic_summary.json',dict(passed=r.wasSuccessful(),tests=r.testsRun,errors=len(r.errors),failures=len(r.failures)))
    print(stream.getvalue());sys.exit(not r.wasSuccessful())
