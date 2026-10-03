"""Local seven-stage KAMP runtime. No implicit deployment or pseudo-labeling."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[2]


def now(): return datetime.now(timezone.utc).isoformat()
def new_id(prefix): return prefix+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path, default=None): return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default
def clean_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',value):
        raise ValueError('Invalid identifier')
    return value
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    def convert(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert),encoding='utf-8')
    os.replace(tmp,path)


@lru_cache(None)
def library(dataset):
    if dataset not in ('cn7','rg3'):raise ValueError('dataset must be cn7 or rg3')
    p=ROOT/'03.modeling/models/ocsvm.ipynb';ns={'__name__':'kamp_runtime_library'}
    for c in read(p)['cells']:
        if c['cell_type']=='code' and 'pipeline_library' in c.get('metadata',{}).get('tags',[]):
            exec(compile(''.join(c['source']),str(p),'exec'),ns)
    ns.update(ROOT=ROOT,DATASET=dataset,NOTEBOOK_PATH=p)
    return ns


def data(dataset):return library(dataset)['load_data'](ROOT)
def feature_columns():return read(ROOT/'data/schema/input_features.json')['feature_columns']


def validate_x(x):
    if list(x)!=feature_columns() or x.empty:raise ValueError('24개 입력 열·순서 또는 빈 입력 확인 필요')
    if not all(pd.api.types.is_numeric_dtype(t) for t in x.dtypes):raise ValueError('비수치형 입력')
    if not np.isfinite(x.to_numpy(dtype=float)).all():raise ValueError('결측·무한값 입력')
    return x.astype(float)


def fingerprints(x):
    a=validate_x(x).to_numpy(dtype='<f8',copy=True);a[a==0]=0
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in a]


def metrics(y,pred):
    y=np.asarray(y,dtype=int);p=np.asarray(pred,dtype=int)
    if y.shape!=p.shape or not set(np.unique(y))<={0,1}:raise ValueError('metric label/shape mismatch')
    tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p==0)).sum());tn=int(((y==0)&(p==0)).sum())
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                recall=tp/(tp+fn) if tp+fn else None,FPR=fp/(fp+tn) if fp+tn else None,
                precision=tp/(tp+fp) if tp+fp else 0.)


def topk(y,scores,fraction=.1):
    y=np.asarray(y);s=np.asarray(scores);k=max(1,int(np.ceil(len(y)*fraction)))
    c=np.sort(s)[-k];hi=s>c;eq=s==c;seats=k-int(hi.sum());b=int(y[eq].sum());n=int(eq.sum());a=int(y[hi].sum())
    return dict(k=k,positives=int(y.sum()),expected_TP=float(a+seats*b/n),
                min_TP=a+max(0,seats-(n-b)),max_TP=a+min(seats,b))


def fit_supervised(dataset,x,y,kind='lr',scenario='raw_alltrain_scaled',C=1.,class_weight=None):
    validate_x(x);y=np.asarray(y,dtype=int)
    if set(y)!={0,1}:raise ValueError('지도학습에는 정상·위험 두 클래스 필요')
    domain=None;values=x
    if scenario!='raw_alltrain_scaled':
        domain,_=library(dataset)['fit_features'](x.loc[y==0],scenario)
        values=library(dataset)['transform_features'](domain,x)
    if kind=='lr':
        estimator=LogisticRegression(C=float(C),class_weight=class_weight,solver='lbfgs',
                                      max_iter=5000,tol=1e-6,random_state=42)
        model=make_pipeline(VarianceThreshold(),StandardScaler(),estimator)
    elif kind=='rf':
        estimator=RandomForestClassifier(n_estimators=300,max_depth=5,min_samples_leaf=5,
                  class_weight='balanced',random_state=42,n_jobs=1)
        model=make_pipeline(VarianceThreshold(),estimator)
    else:raise ValueError(kind)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(values,y)
    return dict(kind=kind,dataset=dataset,model=model,domain=domain,features=feature_columns(),
                scenario=scenario,C=float(C),class_weight=class_weight,threshold=.5,
                label_definition='max observed label per exact input pattern; not product defect probability')


def score(bundle,x):
    validate_x(x);kind=bundle['kind']
    if kind in ('lr','rf'):
        values=library(bundle['dataset'])['transform_features'](bundle['domain'],x) if bundle.get('domain') else x
        classes=bundle['model'].classes_
        return bundle['model'].predict_proba(values)[:,int(np.flatnonzero(classes==1)[0])]
    if kind=='ocsvm':return library(bundle['dataset'])['predict_artifact'](bundle,x)[0]
    if kind=='if':
        values=bundle['scaler'].transform(bundle['variance'].transform(x))
        return -bundle['model'].score_samples(values)
    raise ValueError(kind)


def fit_oneclass(dataset,x,kind,template=None):
    if kind=='if':
        variance=VarianceThreshold().fit(x);scaler=StandardScaler().fit(variance.transform(x))
        model=IsolationForest(n_estimators=300,random_state=42,n_jobs=1).fit(scaler.transform(variance.transform(x)))
        return dict(kind='if',dataset=dataset,features=feature_columns(),variance=variance,scaler=scaler,model=model,threshold=None)
    if not template:raise ValueError('OCSVM 재학습 설정의 기준 모델 필요')
    choice=template['choice'];ns=library(dataset)
    transform,values=ns['fit_features'](x,choice['scenario'])
    return dict(kind='ocsvm',dataset=dataset,features=transform,
                model=ns['fit_model'](values,choice['nu'],choice['gamma_multiplier']),
                choice=choice,threshold=template['threshold'])


def preprocess(frame,batch_id,label_source=None):
    cols=feature_columns()
    if frame.empty or not set(cols)<=set(frame):raise ValueError('빈 배치 또는 입력 열 누락')
    x=validate_x(frame[cols]).reset_index(drop=True)
    labels=None
    if 'PassOrFail' in frame:
        labels=pd.to_numeric(frame.PassOrFail,errors='raise').to_numpy()
        if not np.isfinite(labels).all() or not set(labels)<={0,1}:raise ValueError('라벨은 완전한 0/1이어야 함')
        if not label_source:raise ValueError('라벨 출처를 기록해야 함')
    ids=next((frame[c].astype(str).tolist() for c in ['record_id','product_id','Unnamed: 0'] if c in frame),None)
    ids=ids or [batch_id+':'+str(i) for i in range(len(x))]
    if len(set(ids))!=len(ids):raise ValueError('배치 내 제품 ID 중복')
    fp=fingerprints(x);products=x.copy();products['record_id']=ids;products['fingerprint']=fp
    if labels is not None:products['label']=labels.astype(int)
    groups=products.groupby('fingerprint',sort=False)
    patterns=groups[cols].first();patterns['product_count']=groups.size()
    if labels is not None:
        patterns['label']=groups.label.max();patterns['normal_count']=groups.label.apply(lambda v:int((v==0).sum()))
        patterns['defect_count']=groups.label.sum()
    return products,patterns.reset_index()


class Runtime:
    def __init__(self,dataset,state_root=None):
        if dataset not in ('cn7','rg3'):raise ValueError(dataset)
        self.dataset=dataset;self.state=Path(state_root or ROOT/'runtime')/dataset
        self.state.mkdir(parents=True,exist_ok=True)
        self.schema_hash=digest(ROOT/'data/schema/input_features.json')
    @contextmanager
    def lock(self):
        path=self.state/'.writer.lock';fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        try:yield
        finally:os.close(fd);path.unlink()
    def registry(self):
        r=read(self.state/'registry.json',{'active_ocsvm':None,'fixed_if':None,'history':[]})
        r.setdefault('active_models',{})
        for k,key in [('if','fixed_if'),('ocsvm','active_ocsvm')]:
            if r.get(key) and k not in r['active_models']:r['active_models'][k]=r[key]
        return r
    def load_model(self,version):
        folder=self.state/'models'/clean_id(version);m=read(folder/'manifest.json')
        if not m or m['dataset']!=self.dataset or m['schema_hash']!=self.schema_hash:raise ValueError('모델 데이터/스키마 불일치')
        if digest(folder/'model.joblib')!=m['artifact_sha256'] or m['sklearn']!=sklearn.__version__:raise ValueError('모델 해시/환경 불일치')
        b=joblib.load(folder/'model.joblib');b['kind']=m['kind'];b['dataset']=self.dataset
        return b,m
    def save_candidate(self,bundle,train_x,train_y,source,parent=None,extra=None):
        kind=bundle['kind'];validate_x(train_x)
        if bundle.get('dataset')!=self.dataset:raise ValueError('모델 데이터셋 불일치')
        fp=fingerprints(train_x)
        if len(set(fp))!=len(fp):raise ValueError('학습 패턴 중복')
        if set(fp)&self.protected():raise ValueError('평가 전용 패턴이 학습에 포함됨')
        version=new_id(kind);folder=self.state/'models'/version
        with self.lock():
            folder.mkdir(parents=True,exist_ok=False)
            bundle=dict(bundle,corpus=train_x.assign(label=np.asarray(train_y),fingerprint=fp),parent=parent)
            joblib.dump(bundle,folder/'model.joblib')
            manifest=dict(version=version,kind=kind,dataset=self.dataset,status='candidate',parent=parent,
                created_at=now(),schema_hash=self.schema_hash,sklearn=sklearn.__version__,
                artifact_sha256=digest(folder/'model.joblib'),source=source,
                runtime_code_sha256=digest(__file__),train_fingerprints=fp,
                training_patterns=len(train_x),training_risks=int(np.sum(train_y)),threshold=bundle.get('threshold'),extra=extra or {})
            write(folder/'manifest.json',manifest)
        self.load_model(version)
        return version
    def protected(self):
        x,_,_,_,test,_=data(self.dataset)
        p=set(fingerprints(x.loc[test]))
        for f in (self.state/'evaluations').glob('*/manifest.json'):
            m=read(f);p.update(m['fingerprints'])
        return p
    def initialize(self,version,reason):
        if not reason.strip():raise ValueError('초기 운영 지정 사유 필요')
        b,_=self.load_model(version);kind=b['kind']
        with self.lock():
            r=self.registry()
            if r['active_models'].get(kind):raise ValueError('운영 모델 존재: 검증·승격 경로 사용')
            r['active_models'][kind]=version
            if kind=='ocsvm':r['active_ocsvm']=version
            if kind=='if':r['fixed_if']=version
            r['history'].append(dict(action='initialize_reference',kind=kind,version=version,reason=reason,at=now()))
            write(self.state/'registry.json',r)
        return version
    def register_evaluation(self,frame,label_source,purpose='independent',evaluation_id=None):
        if purpose not in ('independent','historical_followup'):raise ValueError('evaluation purpose')
        eid=clean_id(evaluation_id or new_id('eval'));products,p=preprocess(frame,eid,label_source)
        if 'label' not in p:raise ValueError('평가에는 실제 라벨 필요')
        trained=set()
        for f in (self.state/'models').glob('*/manifest.json'):
            m=read(f);trained.update(m.get('train_fingerprints',[]))
        # Legacy models fit only development data; reserve those patterns as well.
        x,_,_,dev,test,_=data(self.dataset);trained.update(fingerprints(x.loc[dev]))
        if set(p.fingerprint)&trained:raise ValueError('학습 이력이 있는 패턴은 독립 평가 자료로 등록 불가')
        if set(p.fingerprint)&set(fingerprints(x.loc[test])):purpose='historical_followup'
        folder=self.state/'evaluations'/eid
        with self.lock():
            folder.mkdir(parents=True,exist_ok=False);p.to_csv(folder/'patterns.csv',index=False)
            write(folder/'manifest.json',dict(id=eid,dataset=self.dataset,purpose=purpose,
                label_source=label_source,fingerprints=p.fingerprint.tolist(),created_at=now(),
                sha256=digest(folder/'patterns.csv')))
        return eid


def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;d=1+z*z/n
    a=(p+z*z/(2*n))/d;b=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(max(0,a-b)),float(min(1,a+b))]


def bin_spec(values):
    values=np.asarray(values,float);unique=np.unique(values)
    if len(unique)<=12:return dict(type='discrete',values=unique.tolist(),minimum=float(unique.min()),maximum=float(unique.max()))
    return dict(type='continuous',cuts=np.unique(np.quantile(values,np.linspace(0,1,11)))[1:-1].tolist(),
                minimum=float(unique.min()),maximum=float(unique.max()))


def bins(values,spec):
    a=np.asarray(values,float)
    if spec['type']=='discrete':
        lookup={v:str(v) for v in spec['values']}
        return np.array([lookup.get(v,'unseen') for v in a])
    edges=[spec['minimum'],*spec['cuts'],spec['maximum']]
    out=[]
    for v in a:
        if v<edges[0]:out.append('below_reference_range')
        elif v>edges[-1]:out.append('above_reference_range')
        else:
            j=int(np.searchsorted(spec['cuts'],v,side='right'))
            out.append(f'[{edges[j]:.8g}, {edges[j+1]:.8g}'+(']' if j==len(edges)-2 else ')'))
    return np.array(out)


def tv(a,b):
    pa=pd.Series(a).value_counts(normalize=True);pb=pd.Series(b).value_counts(normalize=True)
    return float(pa.subtract(pb,fill_value=0).abs().sum()*.5)


def range_guidance(reference_x,reference_y,evaluation_x,evaluation_y,min_patterns=20,min_risks=5):
    """Training-defined ranges, never labels for unobserved cells. Pattern-level rates."""
    rows=[];columns=feature_columns()
    definitions={c:bin_spec(reference_x[c]) for c in columns}
    arrays={c:(bins(reference_x[c],definitions[c]),bins(evaluation_x[c],definitions[c])) for c in columns}
    for a,b in [('Injection_Time','Filling_Time'),('Max_Injection_Pressure','Mold_Temperature_3')]:
        arrays[a+' × '+b]=(np.char.add(np.char.add(arrays[a][0],' / '),arrays[b][0]),
                           np.char.add(np.char.add(arrays[a][1],' / '),arrays[b][1]))
    ry=np.asarray(reference_y);ey=None if evaluation_y is None else np.asarray(evaluation_y)
    for name,(rb,eb) in arrays.items():
        for category in sorted(set(rb)|set(eb)):
            rr=rb==category;ee=eb==category;n=int(rr.sum());k=int(ry[rr].sum())
            en=int(ee.sum());ek=None if ey is None else int(ey[ee].sum())
            enough=n>=min_patterns and k>=min_risks
            rows.append(dict(variable=name,interval=category,reference_patterns=n,reference_risks=k,
                observed_risk_history_rate=k/n if n else None,reference_wilson95=wilson(k,n),
                evaluation_patterns=en,evaluation_risks=ek,
                evaluation_wilson95=wilson(ek,en) if ek is not None else [None,None],
                evaluation_support='sufficient_descriptive_support' if ek is not None and en>=min_patterns and ek>=min_risks else 'insufficient_support',
                reference_average=float(ry.mean()),
                above_reference_average=bool(n and k/n>ry.mean()),
                status='descriptive_only' if enough else 'insufficient_support',
                guidance='관측 위험 이력 구간; 불량 발생 보장/원인 아님' if enough else '표본 부족 또는 미관측; 안전/불량 범위 확정 불가'))
    return dict(label_definition='risk-history pattern rate, not per-product defect probability',
        confidence_note='Wilson interval is descriptive; independence and multiple comparisons are not guaranteed',
        units='provided coordinates, not recovered physical units',ranges=rows,definitions=definitions)


def write_guidance(folder,guidance):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    write(folder/'distribution_guidance.json',guidance)
    pd.DataFrame(guidance['ranges']).to_csv(folder/'distribution_guidance.csv',index=False,encoding='utf-8-sig')
    ranked=sorted(guidance['ranges'],key=lambda r:(r['status']!='descriptive_only',-(r['observed_risk_history_rate'] or 0)))
    lines=['분포 기반 관측 위험 구간 안내',
           '아래 값은 제공 좌표계 기준이며 물리 단위로 복원한 온도·압력이 아닙니다.',
           '단위는 고유 입력 패턴입니다. 제품 불량 확률·원인·불량 발생 범위를 확정하지 않습니다.',
           '구간은 학습 자료에서 고정했습니다. 평가 구간 수치로 모델/임계값을 변경하지 않습니다.',
           '위험 0건은 안전 증거가 아닙니다. 표본 부족 구간은 추가 관측 대상으로 안내합니다.',
           'Wilson 95% 구간은 기술통계입니다. 관측 독립성과 다중 비교를 보장하지 않습니다.','']
    for r in ranked:
        lo,hi=r['reference_wilson95'];ci='미관측' if lo is None else f'{lo:.1%}~{hi:.1%}'
        lines.append(f"{r['variable']} / {r['interval']} | 참조 위험 {r['reference_risks']}/{r['reference_patterns']} 패턴 | 참고 구간 {ci} | 평가 위험 {r['evaluation_risks']}/{r['evaluation_patterns']} | {r['guidance']} | 평가표본: {r['evaluation_support']}")
    (folder/'distribution_guidance.txt').write_text('\n'.join(lines),encoding='utf-8')


class Operations(Runtime):
    """Explicit batch/drift/CT/CD API; all mutation commands are journaled locally."""
    def __init__(self,dataset,state_root=None,policy=None):
        super().__init__(dataset,state_root)
        self.policy=dict(min_window_rows=200,min_window_patterns=40,bootstrap_repeats=200,
            drift_quantile=.99,persistence=2,min_train_normal=30,min_train_risk=5,
            min_eval_normal=30,min_eval_risk=5,max_FPR=.2,inspection_fraction=.1)
        self.policy.update(policy or {})
    def active(self):return self.registry()['active_models']
    def create_baseline(self):
        x,y,_,dev,_,_=data(self.dataset)
        meta=pd.read_csv(ROOT/f'data/processed/{self.dataset}/conservative/labeled_metadata.csv')
        counts=meta.loc[dev,'total_count'].to_numpy(dtype=int)
        reference=x.loc[dev].iloc[np.repeat(np.arange(len(dev)),counts)].reset_index(drop=True)
        versions=self.active()
        if not versions:raise ValueError('명시적으로 초기 모델을 지정한 후 기준선 생성')
        bid=new_id('baseline');folder=self.state/'baselines'/bid
        with self.lock():
            folder.mkdir(parents=True,exist_ok=False)
            reference.to_csv(folder/'reference_products.csv',index=False)
            x.loc[dev].assign(label=y.loc[dev].to_numpy()).to_csv(folder/'reference_patterns.csv',index=False)
            specs={c:bin_spec(reference[c]) for c in reference}
            for kind,version in versions.items():
                b,_=self.load_model(version);specs['score::'+kind]=bin_spec(score(b,reference))
            write(folder/'manifest.json',dict(id=bid,dataset=self.dataset,created_at=now(),versions=versions,
                  specs=specs,policy=self.policy,reference_rows=len(reference),reference_patterns=len(dev),
                  reference_sha256=digest(folder/'reference_products.csv'),
                  patterns_sha256=digest(folder/'reference_patterns.csv'),
                  note='development reference; bootstrap thresholds exploratory, not field-calibrated'))
            write(self.state/'baseline_pointer.json',{'id':bid})
        return bid
    def baseline(self):
        p=read(self.state/'baseline_pointer.json')
        if not p or read(self.state/'baselines'/p['id']/'manifest.json')['versions']!=self.active():
            self.create_baseline();p=read(self.state/'baseline_pointer.json')
        folder=self.state/'baselines'/p['id'];m=read(folder/'manifest.json')
        if digest(folder/'reference_products.csv')!=m['reference_sha256']:raise ValueError('기준선 변조')
        return folder,m,pd.read_csv(folder/'reference_products.csv',float_precision='round_trip')
    def ingest(self,frame,batch_id=None,label_source=None,coordinates_confirmed=False):
        batch_id=clean_id(batch_id or new_id('batch'));folder=self.state/'batches'/batch_id
        with self.lock():folder.mkdir(parents=True,exist_ok=False)
        try:
            if not coordinates_confirmed:raise ValueError('제공 좌표/스케일 정합 확인 전 추론·드리프트 보류')
            products,patterns=preprocess(frame,batch_id,label_source)
            versions=self.active()
            if not versions:raise ValueError('운영 기준 모델 미지정')
            predictions=products[['record_id','fingerprint']].copy()
            observed_metrics={}
            if 'label' in products:predictions['provided_label']=products.label
            for kind,version in versions.items():
                b,_=self.load_model(version);s=score(b,products[feature_columns()])
                predictions[kind+'_score']=s
                predictions[kind+'_prediction']=(s>b['threshold']).astype(int) if b.get('threshold') is not None else pd.NA
                predictions[kind+'_version']=version
                if 'label' in patterns:
                    ps=score(b,patterns[feature_columns()]);yy=patterns.label.to_numpy()
                    observed_metrics[kind]=topk(yy,ps,self.policy['inspection_fraction']) if kind=='if' else metrics(yy,ps>b['threshold'])
            products.to_csv(folder/'products.csv',index=False);patterns.to_csv(folder/'patterns.csv',index=False)
            predictions.to_csv(folder/'predictions.csv',index=False)
            manifest=dict(id=batch_id,dataset=self.dataset,status='accepted',created_at=now(),
                label_source=label_source,labeled='label' in patterns,coordinates_confirmed=True,
                rows=len(products),patterns=len(patterns),versions=versions,
                confirmed_pattern_metrics=observed_metrics,
                performance_note='provided-label posthoc pattern metrics; no metrics for unlabeled batches',
                files={p.name:digest(p) for p in folder.glob('*.csv')})
            with self.lock():
                write(folder/'manifest.json',manifest)
                ledger=read(self.state/'batch_ledger.json',{'batches':[]})
                ledger['batches'].append(batch_id);write(self.state/'batch_ledger.json',ledger)
        except (ValueError,TypeError,KeyError) as exc:
            manifest=dict(id=batch_id,status='held',reason=str(exc),created_at=now())
            write(folder/'manifest.json',manifest);return manifest
        return dict(**manifest,drift=self.detect())
    def batch(self,bid):
        folder=self.state/'batches'/clean_id(bid);m=read(folder/'manifest.json')
        if not m or m['status']!='accepted':raise ValueError('유효한 배치 아님')
        if not all(digest(folder/n)==h for n,h in m['files'].items()):raise ValueError('배치 산출물 해시 불일치')
        return m,pd.read_csv(folder/'products.csv',float_precision='round_trip'),pd.read_csv(folder/'patterns.csv',float_precision='round_trip')
    def _represent(self,x,manifest):
        out={c:bins(x[c],manifest['specs'][c]) for c in x}
        for kind,v in manifest['versions'].items():
            b,_=self.load_model(v);out['score::'+kind]=bins(score(b,x),manifest['specs']['score::'+kind])
            if b.get('threshold') is not None:out['prediction::'+kind]=(score(b,x)>b['threshold']).astype(str)
        for a,b in [('Injection_Time','Filling_Time'),('Max_Injection_Pressure','Mold_Temperature_3')]:
            out[a+' × '+b]=np.char.add(np.char.add(out[a],' / '),out[b])
        return out
    def detect(self):
        folder,m,reference=self.baseline()
        with self.lock():
            monitor=read(self.state/'monitor.json',{'consumed':[],'history':[]})
            ids=[i for i in read(self.state/'batch_ledger.json',{'batches':[]})['batches'] if i not in monitor['consumed']]
            if not ids:return {'status':'waiting','reason':'no new batches'}
            products=pd.concat([self.batch(i)[1] for i in ids],ignore_index=True)
            x=products[feature_columns()];unique=len(set(fingerprints(x)))
            if len(x)<self.policy['min_window_rows'] or unique<self.policy['min_window_patterns']:
                result=dict(status='waiting',rows=len(x),patterns=unique,batch_ids=ids,reason='감지 표본 부족; 누적 대기')
                write(self.state/'drift_waiting.json',result);return result
            a=self._represent(reference,m);b=self._represent(x,m)
            changes={c:tv(a[c],b[c]) for c in a}
            # Finite-window empirical calibration. Use unique-pattern count as effective
            # sample size so duplicate products cannot inflate precision.
            n_eff=min(len(reference),unique)
            cache=folder/f'calibration_{n_eff}_{self.policy["bootstrap_repeats"]}.json'
            calibration=read(cache)
            if calibration is None:
                rng=np.random.default_rng(42);maxima=[]
                for _ in range(self.policy['bootstrap_repeats']):
                    ix=rng.integers(0,len(reference),size=n_eff)
                    maxima.append(max(tv(v,v[ix]) for v in a.values()))
                calibration=dict(threshold=float(np.quantile(maxima,self.policy['drift_quantile'])),
                    n_eff=n_eff,repeats=self.policy['bootstrap_repeats'],quantile=self.policy['drift_quantile'],
                    note='iid empirical bootstrap proxy, no temporal dependence claim')
                write(cache,calibration)
            threshold=calibration['threshold'];exceeded=max(changes.values())>threshold
            recent=[r for r in monitor['history'] if r['baseline']==m['id']]
            streak=(recent[-1]['streak'] if recent else 0)+1 if exceeded else 0
            status='review' if streak>=self.policy['persistence'] else 'watch' if exceeded else 'normal'
            reference_fingerprints=set(fingerprints(reference))
            did=new_id('drift');result=dict(id=did,baseline=m['id'],versions=m['versions'],created_at=now(),
                status=status,rows=len(x),patterns=unique,batch_ids=ids,changes=changes,
                threshold=threshold,streak=streak,calibration=calibration,
                new_pattern_fraction=float(np.mean([f not in reference_fingerprints for f in fingerprints(x)])),
                outside_reference={c:int(((x[c]<m['specs'][c]['minimum'])|(x[c]>m['specs'][c]['maximum'])).sum()) for c in x},
                interpretation='변화 검토 신호; 불량/원인 확정 또는 자동 재학습 명령 아님')
            write(self.state/'drift'/f'{did}.json',result)
            monitor['consumed'].extend(ids);monitor['history'].append(dict(id=did,baseline=m['id'],status=status,streak=streak))
            write(self.state/'monitor.json',monitor)
            return result
    def retrain(self,kind,batch_ids,drift_id,cause,base_version=None):
        if not cause.strip():raise ValueError('원인 확인 기록 필요')
        drift=read(self.state/'drift'/f'{clean_id(drift_id)}.json')
        if not drift or drift['status']!='review':raise ValueError('지속 변화 검토 단계 이후에만 CT 가능')
        if not set(batch_ids)<=set(read(self.state/'batch_ledger.json',{'batches':[]})['batches']):raise ValueError('미등록 배치')
        if not set(batch_ids)&set(drift['batch_ids']):raise ValueError('드리프트 검토 대상 배치를 포함해야 함')
        version=base_version or self.active().get(kind)
        if not version:raise ValueError('기준 모델 없음')
        old,oldm=self.load_model(version)
        if old['kind']!=kind:raise ValueError('모델 종류 불일치')
        if version!=self.active().get(kind):raise ValueError('현재 운영 기준과 다른 CT 기준 모델')
        if drift['versions']!=self.active():raise ValueError('모델 교체 이후에는 새 기준선 드리프트 확인 필요')
        x,y,_,dev,_,_=data(self.dataset)
        historical=old.get('corpus',x.loc[dev].assign(label=y.loc[dev].to_numpy()))
        parts=[historical[feature_columns()+['label']].copy()];new_patterns=[]
        for bid in batch_ids:
            m,_,p=self.batch(bid)
            if not m['labeled'] or not m['label_source']:raise ValueError('비라벨 배치는 재학습 제외')
            if kind in ('if','ocsvm'):
                # max(label) retains conflict history; only confirmed normal patterns fit.
                pass
            parts.append(p[feature_columns()+['label']]);new_patterns.extend(p.fingerprint)
        combined=pd.concat(parts,ignore_index=True)
        combined=combined.groupby(feature_columns(),sort=False,dropna=False).label.max().reset_index()
        if set(fingerprints(combined[feature_columns()]))&self.protected():raise ValueError('학습/평가 패턴 중복: 배치 분리 필요')
        n0=int(combined.label.eq(0).sum());n1=int(combined.label.eq(1).sum())
        if n0<self.policy['min_train_normal'] or (kind in ('lr','rf') and n1<self.policy['min_train_risk']):
            raise ValueError('학습용 고유 정상/위험 패턴 부족')
        xx=combined[feature_columns()];yy=combined.label
        if kind in ('lr','rf'):
            bundle=fit_supervised(self.dataset,xx,yy,kind=kind,scenario=old.get('scenario','raw_alltrain_scaled'),
                     C=old.get('C',1.),class_weight=old.get('class_weight'))
            bundle['threshold']=old['threshold']
        else:bundle=fit_oneclass(self.dataset,xx.loc[yy.eq(0)],kind,old)
        return self.save_candidate(bundle,xx,yy,source=dict(drift_id=drift_id,batch_ids=batch_ids,cause=cause),
             parent=version,extra=dict(old_patterns=len(historical),new_unique_patterns=len(set(new_patterns)),
                 normal_patterns=n0,risk_patterns=n1,threshold_policy='preserve parent; no evaluation tuning',
                 actual_fit_patterns=n0 if kind in ('if','ocsvm') else len(xx)))
    def evaluate(self,candidate,evaluation_id):
        b,bm=self.load_model(candidate);kind=b['kind'];current=self.active().get(kind)
        efolder=self.state/'evaluations'/clean_id(evaluation_id);em=read(efolder/'manifest.json')
        if not em or digest(efolder/'patterns.csv')!=em['sha256']:raise ValueError('평가 파일 해시 불일치')
        p=pd.read_csv(efolder/'patterns.csv',float_precision='round_trip');x=p[feature_columns()];y=p.label.to_numpy()
        if set(em['fingerprints'])&set(bm.get('train_fingerprints',[])):raise ValueError('후보 학습/평가 누수')
        old,om=self.load_model(current) if current else (None,None)
        if om and set(em['fingerprints'])&set(om.get('train_fingerprints',[])):raise ValueError('기준 모델 학습/평가 누수')
        report_id=new_id('assessment');cs=score(b,x);oscore=score(old,x) if old else None
        cm=topk(y,cs,self.policy['inspection_fraction']) if kind=='if' else metrics(y,cs>b['threshold'])
        ometrics=(topk(y,oscore,self.policy['inspection_fraction']) if kind=='if' else metrics(y,oscore>old['threshold'])) if old else None
        enough=int(y.sum())>=self.policy['min_eval_risk'] and int((y==0).sum())>=self.policy['min_eval_normal']
        reasons=[]
        if not enough:reasons.append('평가 정상/위험 고유 패턴 부족')
        if em['purpose']!='independent':reasons.append('이미 관찰한 후속 평가 자료: 운영 승격 근거로 사용 불가')
        if old is None:reasons.append('비교할 운영 기준 모델 미지정')
        if bm.get('parent') not in (None,current):reasons.append('후보 기준 버전이 현재 운영 모델과 다름')
        if old is not None and enough:
            if kind=='if':
                if cm['expected_TP']<=ometrics['expected_TP']:reasons.append('동일 검사량 위험 발견의 개선 없음')
            elif not(cm['F1']>ometrics['F1'] and cm['recall']>=ometrics['recall'] and cm['FPR']<=self.policy['max_FPR']):
                reasons.append('F1 개선·재현율 유지·오탐률 상한 조건 미충족')
        corpus=b.get('corpus')
        if corpus is None:
            xx,yy,_,dev,_,_=data(self.dataset);corpus=xx.loc[dev].assign(label=yy.loc[dev].to_numpy())
        guidance=range_guidance(corpus[feature_columns()],corpus.label,x,y)
        folder=self.state/'assessments'/report_id;folder.mkdir(parents=True,exist_ok=False)
        write_guidance(folder,guidance)
        result=dict(id=report_id,candidate=candidate,kind=kind,current=current,evaluation_id=evaluation_id,
             created_at=now(),candidate_metrics=cm,current_metrics=ometrics,passed=not reasons,reasons=reasons,
             policy=self.policy,candidate_sha256=bm['artifact_sha256'],current_sha256=om['artifact_sha256'] if om else None,
             evaluation_sha256=em['sha256'],guidance_sha256=digest(folder/'distribution_guidance.json'),
             guidance_use='구간별 관측 위험 안내; 임계값 조정/승격 점수에 사용하지 않음')
        write(folder/'assessment.json',result)
        write(folder/'integrity.json',{'assessment_sha256':digest(folder/'assessment.json')})
        return result
    def promote(self,assessment_id):
        folder=self.state/'assessments'/clean_id(assessment_id)
        integrity=read(folder/'integrity.json')
        if not integrity or digest(folder/'assessment.json')!=integrity['assessment_sha256']:raise ValueError('평가 기록 변조')
        a=read(folder/'assessment.json')
        if not a['passed']:raise ValueError('배포 조건 미통과: '+str(a['reasons']))
        _,m=self.load_model(a['candidate'])
        if m['artifact_sha256']!=a['candidate_sha256']:raise ValueError('평가 후 후보 변경')
        if digest(folder/'distribution_guidance.json')!=a['guidance_sha256']:raise ValueError('구간 안내 변경')
        epath=self.state/'evaluations'/a['evaluation_id']/'patterns.csv'
        if digest(epath)!=a['evaluation_sha256']:raise ValueError('평가 자료 변경')
        with self.lock():
            r=self.registry()
            if r['active_models'].get(a['kind'])!=a['current']:raise ValueError('운영 모델 변경: 재평가 필요')
            _,oldm=self.load_model(a['current'])
            if oldm['artifact_sha256']!=a['current_sha256']:raise ValueError('평가 후 기준 모델 변경')
            r['active_models'][a['kind']]=a['candidate']
            if a['kind']=='ocsvm':r['active_ocsvm']=a['candidate']
            if a['kind']=='if':r['fixed_if']=a['candidate']
            r['history'].append(dict(action='promote',kind=a['kind'],previous=a['current'],version=a['candidate'],
                 assessment=assessment_id,at=now()))
            write(self.state/'registry.json',r)
        self.create_baseline()
        return a['candidate']
    def rollback(self,kind,reason):
        if not reason.strip():raise ValueError('롤백 사유 필요')
        with self.lock():
            r=self.registry();current=r['active_models'].get(kind)
            history=[h for h in r['history'] if h['action']=='promote' and h['kind']==kind and h['version']==current]
            if not history:raise ValueError('되돌릴 승격 이력 없음')
            previous=history[-1]['previous'];self.load_model(previous)
            r['active_models'][kind]=previous
            if kind=='ocsvm':r['active_ocsvm']=previous
            if kind=='if':r['fixed_if']=previous
            r['history'].append(dict(action='rollback',kind=kind,previous=current,version=previous,reason=reason,at=now()))
            write(self.state/'registry.json',r)
        self.create_baseline();return previous
