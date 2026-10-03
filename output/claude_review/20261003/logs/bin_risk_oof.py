# Feasibility: does a bin-based observed-risk score rank risk patterns on development OOF?
import sys,json;from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/home/claude/work/KAMP');sys.path.insert(0,str(ROOT/'03.modeling/common'))
from pipeline_runtime import data,bin_spec,bins,feature_columns,topk
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(0);res={}
for ds in ['cn7','rg3']:
    x,y,a,dev,test,_=data(ds);f=a.loc[dev,'cv_fold'].to_numpy();oof={k:np.zeros(len(dev)) for k in ['mean_smoothed','max_lower95']}
    for k in range(4):
        tr,va=dev[f!=k],dev[f==k];yt=y.loc[tr].to_numpy();base=yt.mean()
        sm=np.zeros(len(va));mx=np.zeros(len(va))
        for c in feature_columns():
            spec=bin_spec(x.loc[tr,c]);bt=bins(x.loc[tr,c],spec);bv=bins(x.loc[va,c],spec)
            tab=pd.DataFrame(dict(b=bt,y=yt)).groupby('b').y.agg(['sum','count'])
            rate=((tab['sum']+base*10)/(tab['count']+10))  # shrink to base rate
            z=1.96;n=tab['count'];p=tab['sum']/n;lo=(p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
            sm+=pd.Series(bv).map(rate).fillna(base).to_numpy();mx=np.maximum(mx,pd.Series(bv).map(lo).fillna(0).to_numpy())
        idx=np.where(f==k)[0];oof['mean_smoothed'][idx]=sm/24;oof['max_lower95'][idx]=mx
    yd=y.loc[dev].to_numpy();r={}
    for k,s in oof.items():
        auc=roc_auc_score(yd,s);perm=[roc_auc_score(rng.permutation(yd),s) for _ in range(1000)]
        r[k]=dict(OOF_AUC=round(auc,3),perm_p=round(float(np.mean(np.array(perm)>=auc)),3),top10=topk(yd,s,.1))
    res[ds]=dict(dev=len(dev),risks=int(yd.sum()),**r)
print(json.dumps(res,indent=1,ensure_ascii=False))
