import sys,json;from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'03.modeling/common'))
from pipeline_runtime import data,fingerprints,feature_columns,range_guidance,bins
from decision_runtime import inspection_advice
def classify(ref_x,ref_y,prod):
    g,rows=inspection_advice(ref_x,ref_y,prod)
    base=float(np.mean(ref_y));lookup={(r['variable'],r['interval']):r for r in g['ranges']}
    x=prod[feature_columns()];cats={c:bins(x[c],s) for c,s in g['definitions'].items()}
    for a,b in [('Injection_Time','Filling_Time'),('Max_Injection_Pressure','Mold_Temperature_3')]:
        cats[a+' × '+b]=np.char.add(np.char.add(cats[a],' / '),cats[b])
    out=[]
    for i in range(len(x)):
        m=[lookup[(c,str(v[i]))] for c,v in cats.items()]
        oor=any(r['reference_patterns']==0 or r['interval'] in('unseen','below_reference_range','above_reference_range') or 'unseen' in r['interval'] or 'reference_range' in r['interval'] for r in m)
        elev=any(r['status']=='descriptive_only' and r['reference_wilson95'][0] is not None and r['reference_wilson95'][0]>base for r in m)
        allsparse=all(r['status']=='insufficient_support' for r in m)
        out.append(dict(codex=rows[i]['evidence_level'],oor=oor,elev=elev,allsparse=allsparse))
    return pd.DataFrame(out)
res={}
for ds in ['rg3','cn7']:
    x,y,a,dev,test,_=data(ds)
    p=x.loc[test].reset_index(drop=True);p['record_id']=range(len(p));p['fingerprint']=fingerprints(p[feature_columns()])
    t=classify(x.loc[dev],y.loc[dev].to_numpy(),p);yt=y.loc[test].to_numpy()
    # dev OOF
    f=a.loc[dev,'cv_fold'].to_numpy();parts=[]
    for k in range(4):
        tr,va=dev[f!=k],dev[f==k];pp=x.loc[va].reset_index(drop=True);pp['record_id']=range(len(pp));pp['fingerprint']=fingerprints(pp[feature_columns()])
        c=classify(x.loc[tr],y.loc[tr].to_numpy(),pp);c['y']=y.loc[va].to_numpy();parts.append(c)
    o=pd.concat(parts)
    def summ(d,yy):
        r={}
        for col in ['oor','elev','allsparse']:
            s=d[col].to_numpy();r[col]=dict(flagged=int(s.sum()),risk_in_flagged=int(yy[s].sum()),risk_total=int(yy.sum()),n=len(yy))
        r['codex_levels']=d.codex.value_counts().to_dict()
        r['codex_recheck_risk']=int(yy[d.codex.ne('monitor').to_numpy()].sum())
        return r
    res[ds]=dict(test=summ(t,yt),dev_oof=summ(o,o.y.to_numpy()))
print(json.dumps(res,indent=1,ensure_ascii=False))
