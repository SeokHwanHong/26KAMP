# Adversarial probes in an isolated temp state root (no project file changes).
import sys,json,tempfile,copy
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/home/claude/work/KAMP');sys.path.insert(0,str(ROOT/'03.modeling/common'))
from pipeline_runtime import *
R={}
def synthetic(n,seed=42,shift=0):
    rng=np.random.default_rng(seed);a=rng.normal(size=(n,24))+shift
    x=pd.DataFrame(a,columns=feature_columns());return x,(a[:,0]>shift).astype(int)
tmp=tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='probe_')
ops=Operations('rg3',Path(tmp.name),policy=dict(min_window_rows=40,min_window_patterns=30,bootstrap_repeats=40))
x,y=synthetic(120);b=fit_supervised('rg3',x,y,C=100.);v=ops.save_candidate(b,x,y,source={'f':'syn'});ops.initialize(v,'probe')
# 1 input validation
bx,by=synthetic(50,7)
cases={}
f=bx.copy();f.iloc[0,0]=np.nan;cases['nan']=ops.ingest(f,coordinates_confirmed=True)['status']
f=bx.copy();f.iloc[0,0]=np.inf;cases['inf']=ops.ingest(f,coordinates_confirmed=True)['status']
cases['missing_col']=ops.ingest(bx.drop(columns=[feature_columns()[0]]),coordinates_confirmed=True)['status']
cases['reordered_cols']=ops.ingest(bx[feature_columns()[::-1]],coordinates_confirmed=True)['status']
f=bx.assign(product_id=['a']*50);cases['dup_id_in_batch']=ops.ingest(f,coordinates_confirmed=True)['status']
cases['label_2']=ops.ingest(bx.assign(PassOrFail=[2]+[0]*49),label_source='s',coordinates_confirmed=True)['status']
cases['label_no_source']=ops.ingest(bx.assign(PassOrFail=by),coordinates_confirmed=True)['status']
cases['coords_unconfirmed']=ops.ingest(bx,coordinates_confirmed=False)['status']
f=bx.copy().astype(object);f.iloc[0,0]='1.0';cases['string_value']=ops.ingest(f,coordinates_confirmed=True)['status']
R['1_input_hold']=cases
# 5 duplicate batch / cross-batch product id reuse
fr=bx.assign(product_id=[f'P{i}' for i in range(50)])
r1=ops.ingest(fr,coordinates_confirmed=True);r2=ops.ingest(fr,coordinates_confirmed=True)
R['5_same_file_twice']=dict(first=r1['status'],second=r2['status'],second_drift_rows=r2.get('drift',{}).get('rows'),
   note='identical products/IDs accepted twice -> drift window counts them twice')
try:ops.ingest(fr,batch_id=r1['id'],coordinates_confirmed=True);R['5_batch_id_reuse']='accepted'
except Exception as e:R['5_batch_id_reuse']=f'rejected: {type(e).__name__}'
# 3 window non-overlap
mon=read(ops.state/'monitor.json');R['3_consumed_unique']=len(mon['consumed'])==len(set(mon['consumed']))
# 6 CT keeps risk history
ops2=Operations('rg3',Path(tmp.name)/'b',policy=dict(min_window_rows=40,min_window_patterns=30,bootstrap_repeats=40))
x,y=synthetic(120);b=fit_supervised('rg3',x,y,C=100.);v=ops2.save_candidate(b,x,y,source={'f':'syn'});ops2.initialize(v,'probe')
risk_row=x.loc[y==1].iloc[[0]]
ids=[]
for s in [71,72]:
    xx,yy=synthetic(45,s,shift=9)
    if s==71: xx=pd.concat([xx,risk_row],ignore_index=True);yy=np.r_[yy,0]   # same input re-observed as normal
    r=ops2.ingest(xx.assign(PassOrFail=yy),label_source='syn',coordinates_confirmed=True);ids.append(r['id'])
R['6_drift_status']=r['drift']['status']
if r['drift']['status']=='review':
    cv=ops2.retrain('lr',ids,r['drift']['id'],'probe');bb,_=ops2.load_model(cv);c=bb['corpus']
    fp=fingerprints(risk_row)[0];R['6_risk_kept_after_normal_reobs']=int(c.loc[c.fingerprint==fp,'label'].iloc[0])
    # 4 version change then stale drift id
    R['6_candidate']=cv
# 7 evaluation gates: high FPR candidate must be kept
x,y=synthetic(160,4);b=fit_supervised('rg3',x,y,C=100.)
ops3=Operations('rg3',Path(tmp.name)/'c');old=copy.deepcopy(b);old['threshold']=1.
v0=ops3.save_candidate(old,x,y,source={});ops3.initialize(v0,'probe')
alarm=copy.deepcopy(b);alarm['threshold']=0.0   # alerts nearly everything
v1=ops3.save_candidate(alarm,x,y,source={},parent=v0)
ex,ey=synthetic(160,5);eid=ops3.register_evaluation(ex.assign(PassOrFail=ey),'syn')
a=ops3.evaluate(v1,eid);R['7_high_FPR_candidate']=dict(passed=a['passed'],reasons=a['reasons'],FPR=a['candidate_metrics']['FPR'])
same=ops3.save_candidate(copy.deepcopy(old),x,y,source={},parent=v0);a2=ops3.evaluate(same,eid)
R['7_no_improvement']=dict(passed=a2['passed'],reasons=a2['reasons'])
# 8 tamper evaluation after assessment
good=ops3.save_candidate(b,x,y,source={},parent=v0);a3=ops3.evaluate(good,eid)
p=ops3.state/'evaluations'/eid/'patterns.csv';t=p.read_text();p.write_text(t+'\n')
try:ops3.promote(a3['id']);R['8_eval_tamper']='promoted!'
except ValueError as e:R['8_eval_tamper']='rejected: '+str(e)
p.write_text(t)
# policy changed between evaluate and promote (not part of assessment integrity?)
ops3b=Operations('rg3',Path(tmp.name)/'c',policy=dict(max_FPR=0.0))
R['8_policy_changed_before_promote']='promoted (policy at evaluation time is used)' if ops3b.promote(a3['id'])==good else '?'
R['9_rollback']=ops3.rollback('lr','probe')==v0
# stale lock
(ops3.state/'.writer.lock').write_text('')
try:ops3.ingest(ex,coordinates_confirmed=True);R['H_stale_lock']='no error'
except Exception as e:R['H_stale_lock']=f'{type(e).__name__} (manual removal required)'
print(json.dumps(R,ensure_ascii=False,indent=1,default=str))
tmp.cleanup()
