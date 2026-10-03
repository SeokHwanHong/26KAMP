import sys,json,tempfile
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/home/claude/work/KAMP');sys.path.insert(0,str(ROOT/'03.modeling/common'))
from pipeline_runtime import *
def synthetic(n,seed=42,shift=0):
    rng=np.random.default_rng(seed);a=rng.normal(size=(n,24))+shift
    return pd.DataFrame(a,columns=feature_columns()),(a[:,0]>shift).astype(int)
tmp=tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='probe_dup_')
ops=Operations('rg3',Path(tmp.name),policy=dict(min_window_rows=40,min_window_patterns=30,bootstrap_repeats=40))
x,y=synthetic(120);v=ops.save_candidate(fit_supervised('rg3',x,y,C=100.),x,y,source={});ops.initialize(v,'probe')
xx,yy=synthetic(45,71,shift=9);frame=xx.assign(PassOrFail=yy,product_id=[f'LOT1-{i}' for i in range(45)])
out=[]
for k in range(2):
    r=ops.ingest(frame,label_source='same physical lot re-sent',coordinates_confirmed=True)
    out.append(dict(batch=r['id'],status=r['status'],drift=r['drift'].get('status'),streak=r['drift'].get('streak')))
print(json.dumps(out,indent=1))
if out[-1]['drift']=='review':
    v2=ops.retrain('lr',[o['batch'] for o in out],r['drift']['id'],'duplicate lot')
    print('CT candidate created from one physical lot sent twice:',v2)
tmp.cleanup()
