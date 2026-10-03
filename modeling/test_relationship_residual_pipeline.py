"""Targeted leakage, threshold and saved-model contract checks."""
import unittest
import numpy as np
import pandas as pd
from relationship_residual_pipeline import (RelationshipBank, choose_threshold,
    counts, group_codes, load_pattern, load_temporal, past_context, reference_roles)

class RelationshipTests(unittest.TestCase):
    def test_past_features_do_not_read_future(self):
        x=pd.DataFrame({'a':np.arange(20,dtype=float),'b':np.arange(20,dtype=float)**2})
        first=past_context(x)
        x.loc[12:,:]=123456.
        second=past_context(x)
        pd.testing.assert_frame_equal(first.iloc[:12],second.iloc[:12])
        self.assertEqual(first.loc[5,'past3mean::a'],3.)

    def test_threshold_keeps_ties_together_and_matches_bruteforce(self):
        y=np.array([0,1,1,0,0]);s=np.array([.2,.2,.4,.8,.1])
        t=choose_threshold(y,s)
        candidates=np.r_[np.nextafter(s.min(),-np.inf),np.unique(s)]
        brute=max(candidates,key=lambda v:(counts(y,s,v)['F1'],-counts(y,s,v)['FP'],v))
        self.assertEqual(counts(y,s,t)['F1'],counts(y,s,brute)['F1'])
        self.assertEqual(counts(y,s,t)['FP'],counts(y,s,brute)['FP'])

    def test_reference_and_classifier_groups_separate(self):
        for ds in ('cn7','rg3'):
            ctx,x,y,g,dev,test,folds,_=load_pattern(ds)
            ref,cl=reference_roles(x.iloc[dev],y[dev],g[dev])
            self.assertFalse(set(g[dev][ref])&set(g[dev][cl]))
            self.assertTrue(np.all(y[dev][ref]==0))
            self.assertFalse(set(dev)&set(test))

    def test_temporal_partitions_and_gap(self):
        for ds in ('cn7','rg3'):
            ctx,x,y,g,tr,te,folds,_=load_temporal(ds)
            _,_,va=folds[0]
            self.assertGreater(va.min()-tr.max(),3)
            self.assertGreater(te.min()-va.max(),3)
            self.assertFalse(set(g[tr])&set(g[va]))
            self.assertFalse(set(g[tr])&set(g[te]))
            self.assertFalse(set(g[va])&set(g[te]))

    def test_relation_never_predicts_target_from_itself(self):
        x=pd.DataFrame({'a':np.arange(60,dtype=float),'b':2*np.arange(60,dtype=float)+3,
                        'constant':np.ones(60)})
        bank=RelationshipBank('ridge').fit(x,x,np.arange(60))
        self.assertNotIn('constant',bank.targets)
        for target,item in bank.models.items():
            self.assertNotIn(target,item['predictors'])
        self.assertEqual(len(bank.targets),2)
        same=pd.concat([x.iloc[[20]],x.iloc[[20]]],ignore_index=True)
        r=bank.residuals(same,same)
        np.testing.assert_array_equal(r.iloc[0],r.iloc[1])

if __name__=='__main__':
    unittest.main(verbosity=2)
