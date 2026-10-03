"""KAMP normal-relationship residual experiment.

Run: python modeling/relationship_residual_pipeline.py --run-id <new-name>
No changes to original data, previous notebooks, splits or outputs.
Definitions: normal-relationship reference, classifier-training, validation and
test roles are separated. No test-driven feature/threshold/model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p/'kamp_data').is_dir())
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/matplotlib_cache'))
os.environ.setdefault('OMP_NUM_THREADS', '1')

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import beta
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, r2_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONFIG = dict(seed=42, relation_reference_normal_fraction=0.5,
              relation_inner_folds=3, relation_min_oof_r2=0.2,
              ridge_alpha=10., extra_trees=32, extra_depth=5, extra_min_leaf=8,
              logistic_C=1., rf_trees=128, rf_depth=5, rf_min_leaf=5,
              past_window=3, temporal_gap=3,
              threshold_rule='pooled validation F1; ties fewer FP, higher threshold',
              pattern_target='any observed defect in exact current-input pattern',
              temporal_target='original row label; file order assumed temporal',
              status='follow-up exploration on previously inspected data')

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                  default=lambda v: v.item() if isinstance(v,np.generic) else str(v)), encoding='utf-8')

def counts(y, score, threshold):
    y=np.asarray(y,dtype=int); score=np.asarray(score,dtype=float)
    assert np.isfinite(score).all()
    pred=(score > threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    p=int(y.sum()); n=len(y)-p
    return dict(n=len(y), positives=p, TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),
                precision=float(tp/(tp+fp)) if tp+fp else 0.,
                recall=float(tp/p) if p else None,
                FPR=float(fp/n) if n else None,
                F1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
                AP=float(average_precision_score(y,score)) if p else None,
                AUC=float(roc_auc_score(y,score)) if p and n else None)

def choose_threshold(y, score):
    y=np.asarray(y,dtype=int); score=np.asarray(score,dtype=float)
    assert 0 < y.sum() < len(y)
    order=np.argsort(-score,kind='stable'); ss=score[order]; yy=y[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]
    tp=np.cumsum(yy)[ends]; fp=ends+1-tp; fn=y.sum()-tp
    f1=2*tp/(2*tp+fp+fn)
    # A cutoff just below an observed score includes that entire tie group.
    thresholds=np.nextafter(ss[ends],-np.inf)
    i=sorted(range(len(ends)),key=lambda j:(-f1[j],fp[j],-thresholds[j]))[0]
    return float(thresholds[i])

def group_codes(x):
    return pd.factorize(pd.Series(list(map(tuple,x.to_numpy()))))[0]

def reference_roles(x, y, groups):
    """Keep mixed-label groups out of the normal reference entirely."""
    group_label=pd.Series(np.asarray(y)).groupby(groups).max()
    normal=group_label.index[group_label.eq(0)].to_numpy().copy()
    rng=np.random.default_rng(CONFIG['seed']); rng.shuffle(normal)
    selected=normal[:max(3,int(len(normal)*CONFIG['relation_reference_normal_fraction']))]
    reference=np.flatnonzero(np.isin(groups,selected))
    classifier=np.flatnonzero(~np.isin(groups,selected))
    assert len(set(groups[reference]) & set(groups[classifier]))==0
    assert np.all(np.asarray(y)[reference]==0)
    assert len(np.unique(np.asarray(y)[classifier]))==2
    return reference,classifier

def regression(backend):
    if backend=='ridge':
        return make_pipeline(StandardScaler(),Ridge(alpha=CONFIG['ridge_alpha']))
    return ExtraTreesRegressor(n_estimators=CONFIG['extra_trees'],max_depth=CONFIG['extra_depth'],
                               min_samples_leaf=CONFIG['extra_min_leaf'],random_state=42,n_jobs=1)

class RelationshipBank:
    """One regression per current variable; never use the target as predictor.

    OOF prediction on normal reference measures relationship learnability.
    Retain targets with fixed R2 >= .2. Residual scale is OOF normal Q90(abs error)
    with 5% of normal target SD floor. Classifier rows are never reference rows.
    """
    def __init__(self,backend):
        self.backend=backend

    def fit(self, context, current, groups):
        self.models={}; self.records=[]; self.targets=[]; self.context_columns=list(context)
        folds=list(GroupKFold(n_splits=3).split(context,groups=groups))
        for target in current:
            yy=current[target].to_numpy()
            if np.std(yy)<1e-12:
                self.records.append(dict(target=target,retained=False,reason='constant_reference_target',oof_r2=None))
                continue
            predictors=[c for c in context if c!=target and context[c].nunique()>1]
            assert target not in predictors
            if not predictors:
                continue
            oof=np.full(len(yy),np.nan)
            for tr,va in folds:
                assert not (set(groups[tr])&set(groups[va]))
                reg=regression(self.backend).fit(context.iloc[tr][predictors],yy[tr])
                oof[va]=reg.predict(context.iloc[va][predictors])
            assert np.isfinite(oof).all()
            quality=float(r2_score(yy,oof))
            scale=max(float(np.quantile(np.abs(yy-oof),.9)),float(np.std(yy))*.05,1e-8)
            record=dict(target=target,retained=quality>=CONFIG['relation_min_oof_r2'],
                        reason='retained' if quality>=CONFIG['relation_min_oof_r2'] else 'weak_normal_relationship',
                        oof_r2=quality,oof_mae=float(np.mean(np.abs(yy-oof))),residual_scale=scale,
                        predictor_count=len(predictors),normal_reference_rows=len(yy))
            self.records.append(record)
            if record['retained']:
                model=regression(self.backend).fit(context[predictors],yy)
                self.models[target]=dict(model=model,predictors=predictors,scale=scale)
                self.targets.append(target)
        return self

    def residuals(self, context, current):
        result=pd.DataFrame(index=context.index)
        for target in self.targets:
            item=self.models[target]
            result[target]=(current[target]-item['model'].predict(context[item['predictors']]))/item['scale']
        return result

    def features(self, context, current):
        residual=self.residuals(context,current)
        if not len(residual.columns):
            return pd.DataFrame({'no_retained_relationship':np.zeros(len(context))},index=context.index)
        a=residual.abs().to_numpy(); k=min(3,a.shape[1])
        return pd.concat([residual.add_prefix('residual::'),residual.abs().add_prefix('abs_residual::'),
            pd.DataFrame({'residual_mean':a.mean(axis=1),'residual_max':a.max(axis=1),
                          'residual_top3_mean':np.sort(a,axis=1)[:,-k:].mean(axis=1)},index=context.index)],axis=1)

def classifier(kind):
    if kind=='lr':
        return make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',
                                                                 max_iter=3000,random_state=42))
    return RandomForestClassifier(n_estimators=128,max_depth=5,min_samples_leaf=5,
                                   class_weight='balanced',max_features='sqrt',random_state=42,n_jobs=1)

class RelationshipExperiment:
    def fit(self,context,current,y,groups):
        ref,cl=reference_roles(current,y,groups)
        self.reference=ref; self.classifier_rows=cl; self.banks={}; self.models={}; self.specs={}
        self.fit_inputs=list(context); y=np.asarray(y)
        for kind in ('lr','rf'):
            for scope,indices in [('full',np.arange(len(y))),('matched',cl)]:
                name=f'raw_{scope}_{kind}'
                self.models[name]=classifier(kind).fit(context.iloc[indices],y[indices])
                self.specs[name]=dict(representation='raw',backend=None,kind=kind,columns=list(context))
        for backend in ('ridge','extra'):
            bank=RelationshipBank(backend).fit(context.iloc[ref].reset_index(drop=True),
                current.iloc[ref].reset_index(drop=True),groups[ref])
            self.banks[backend]=bank
            residual=bank.features(context,current)
            for representation,frame,kinds in [('residual',residual,('lr',)),
                                                ('augmented',pd.concat([context,residual],axis=1),('lr','rf'))]:
                for kind in kinds:
                    name=f'{backend}_{representation}_{kind}'
                    self.models[name]=classifier(kind).fit(frame.iloc[cl],y[cl])
                    self.specs[name]=dict(representation=representation,backend=backend,kind=kind,columns=list(frame))
        return self

    def predict(self,context,current):
        output={}; frames={b:bank.features(context,current) for b,bank in self.banks.items()}
        for name,model in self.models.items():
            spec=self.specs[name]; frame=context
            if spec['representation']=='residual': frame=frames[spec['backend']]
            elif spec['representation']=='augmented': frame=pd.concat([context,frames[spec['backend']]],axis=1)
            assert list(frame)==spec['columns']
            output[name]=model.predict_proba(frame)[:,list(model.classes_).index(1)]
        for backend,frame in frames.items():
            output[f'{backend}_unsupervised_top3']=frame.get('residual_top3_mean',pd.Series(np.zeros(len(frame)),index=frame.index)).to_numpy()
        return output

def load_pattern(ds):
    base=ROOT/f'data/processed/{ds}/conservative'
    manifest=json.loads((base/'splits/split_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['source_sha256'].items():
        assert digest(base/name)==expected, f'Changed source: {name}'
    assert digest(base/'splits/split_assignments.csv')==manifest['artifact_sha256']['split_assignments.csv']
    x=pd.read_csv(base/'X_labeled.csv',float_precision='round_trip')
    y=pd.read_csv(base/'y_labeled.csv').PassOrFail.to_numpy()
    assign=pd.read_csv(base/'splits/split_assignments.csv')
    assert np.array_equal(assign.pattern_row,np.arange(len(x))) and np.array_equal(assign.label,y)
    raw=pd.read_csv(ROOT/f'kamp_data/moldset_labeled_{ds}.csv',float_precision='round_trip')
    agg=raw.groupby(list(x),sort=False,dropna=False).PassOrFail.max()
    expected=pd.DataFrame(list(agg.index),columns=x.columns)
    assert np.array_equal(expected.to_numpy(),x.to_numpy()) and np.array_equal(agg.to_numpy(),y)
    dev=np.flatnonzero(assign.partition.eq('development')); test=np.flatnonzero(assign.partition.eq('test'))
    folds=[]
    for f in sorted(assign.loc[dev,'cv_fold'].unique()):
        va=dev[assign.loc[dev,'cv_fold'].to_numpy()==f]
        tr=dev[assign.loc[dev,'cv_fold'].to_numpy()!=f]
        folds.append((f'train_fold_{f}',tr,va))
    return x,x.copy(),y,group_codes(x),dev,test,folds,dict(protocol='fixed_pattern',dataset=ds,
        target=CONFIG['pattern_target'],source_sha256=digest(ROOT/f'kamp_data/moldset_labeled_{ds}.csv'))

def past_context(x,window=3):
    # Current target remains separate. Its own lag is a permitted past predictor.
    return pd.concat([x,x.shift(1).add_prefix('lag1::'),
                      x.shift(1).rolling(window,min_periods=window).mean().add_prefix('past3mean::'),
                      x.shift(1).rolling(window,min_periods=window).std(ddof=0).add_prefix('past3std::')],axis=1)

def load_temporal(ds):
    raw=pd.read_csv(ROOT/f'kamp_data/moldset_labeled_{ds}.csv',float_precision='round_trip')
    x=raw.drop(columns=['Unnamed: 0','PassOrFail']); y=raw.PassOrFail.to_numpy(); n=len(x)
    keys=list(map(tuple,x.to_numpy())); last={k:i for i,k in enumerate(keys)}
    def boundary(b):
        while max(last[k] for k in keys[:b])+1>b:
            b=max(last[k] for k in keys[:b])+1
        return b
    a=boundary(int(.6*n)); b=boundary(int(.8*n)); window=CONFIG['past_window']; gap=CONFIG['temporal_gap']
    tr=np.arange(window,a); va=np.arange(a+gap,b); te=np.arange(b+gap,n)
    groups=group_codes(x)
    assert not(set(groups[tr])&set(groups[va])) and not(set(groups[tr])&set(groups[te]))
    assert tr.max()<va.min() and va.max()<te.min()
    context=past_context(x,window)
    assert np.isfinite(context.iloc[np.r_[tr,va,te]].to_numpy()).all()
    info=dict(protocol='temporal_assumed',dataset=ds,target=CONFIG['temporal_target'],
              first_validation_row_1based=int(va[0]+1),first_test_row_1based=int(te[0]+1),
              gap=gap,split_counts=[len(tr),len(va),len(te)],split_defects=[int(y[i].sum()) for i in (tr,va,te)])
    return context,x,y,groups,tr,te,[('forward_validation',tr,va)],info

def normal_generalization(bank,context,current,y):
    out=[]; normal=np.asarray(y)==0
    for target in bank.targets:
        item=bank.models[target]; actual=current.loc[normal,target].to_numpy()
        predicted=item['model'].predict(context.loc[normal,item['predictors']])
        out.append(dict(target=target,normal_rows=len(actual),
                        normal_r2=float(r2_score(actual,predicted)) if len(actual)>1 and np.std(actual)>1e-12 else None,
                        normal_mae=float(np.mean(np.abs(actual-predicted)))))
    return out

def run_protocol(ds,protocol,out):
    out.mkdir(parents=True,exist_ok=False)
    context,current,y,groups,dev,test,folds,info=(load_pattern(ds) if protocol=='fixed_pattern' else load_temporal(ds))
    if protocol=='temporal_assumed' and any(len(np.unique(y[ix]))<2 for _,tr,va in folds for ix in (tr,va)):
        info['status']='not_evaluable: future validation contains no defects; no future defect detection claim'
        write_json(out/'protocol.json',info); print(ds,protocol,'SKIP: no future validation defects',flush=True)
        return info
    write_json(out/'protocol.json',info)
    validation_parts=[]; relation_records=[]; roles=[]; frozen_experiment=None
    for tag,tr,va in folds:
        print(ds,protocol,tag,'fit',len(tr),'validate',len(va),flush=True)
        assert not(set(groups[tr])&set(groups[va]))
        exp=RelationshipExperiment().fit(context.iloc[tr].reset_index(drop=True),current.iloc[tr].reset_index(drop=True),y[tr],groups[tr])
        roles.append(dict(fold=tag,reference_rows=tr[exp.reference].tolist(),classifier_rows=tr[exp.classifier_rows].tolist(),
                          validation_rows=va.tolist(),train_rows=tr.tolist()))
        scores=exp.predict(context.iloc[va].reset_index(drop=True),current.iloc[va].reset_index(drop=True))
        for name,s in scores.items():
            validation_parts.append(pd.DataFrame({'row':va,'fold':tag,'model':name,'y':y[va],'score':s}))
        for backend,bank in exp.banks.items():
            relation_records.extend([dict(fold=tag,backend=backend,**r) for r in bank.records])
        if protocol=='temporal_assumed': frozen_experiment=exp
    predictions=pd.concat(validation_parts,ignore_index=True)
    predictions.to_csv(out/'validation_predictions.csv',index=False)
    write_json(out/'role_plan.json',roles)
    pd.DataFrame(relation_records).to_csv(out/'relationship_oof_quality.csv',index=False)
    thresholds={}; selected_rows=[]
    for name,part in predictions.groupby('model'):
        assert part.row.is_unique
        t=choose_threshold(part.y,part.score); thresholds[name]=t
        selected_rows.append(dict(model=name,threshold=t,**counts(part.y,part.score,t)))
    comparison=pd.DataFrame(selected_rows).sort_values(['F1','FP','model'],ascending=[False,True,True])
    comparison.to_csv(out/'validation_comparison.csv',index=False)
    choice=comparison.iloc[0].to_dict()
    write_json(out/'selection_manifest.json',dict(selected=choice,thresholds=thresholds,test_used_for_selection=False,
        final_fit='development refit' if protocol=='fixed_pattern' else 'keep train-only model and validation threshold',config=CONFIG))
    selection_digest=digest(out/'selection_manifest.json')
    if frozen_experiment is None:
        print(ds,protocol,'final fit',len(dev),flush=True)
        exp=RelationshipExperiment().fit(context.iloc[dev].reset_index(drop=True),current.iloc[dev].reset_index(drop=True),y[dev],groups[dev])
    else:
        exp=frozen_experiment
    write_json(out/'final_fit_roles.json',dict(reference_rows=dev[exp.reference].tolist(),classifier_rows=dev[exp.classifier_rows].tolist(),test_rows=test.tolist()))
    joblib.dump(exp,out/'experiment.joblib')
    ctx=context.iloc[test].reset_index(drop=True); cur=current.iloc[test].reset_index(drop=True)
    scores=exp.predict(ctx,cur); reloaded=joblib.load(out/'experiment.joblib').predict(ctx,cur)
    assert all(np.allclose(scores[k],reloaded[k],atol=1e-12,rtol=0) for k in scores)
    test_rows=[]; test_predictions=[]; normal_records=[]; topk=[]; explanations=[]
    for name,s in scores.items():
        t=thresholds[name]; test_rows.append(dict(model=name,selected_on_validation=name==choice['model'],threshold=t,**counts(y[test],s,t)))
        test_predictions.append(pd.DataFrame({'row':test,'model':name,'y':y[test],'score':s,'threshold':t,'prediction':(s>t).astype(int)}))
        for fraction in (.1,.2,.3):
            k=int(np.ceil(len(test)*fraction)); cutoff=np.sort(s)[-k]
            greater=s>cutoff; equal=s==cutoff; seats=k-int(greater.sum())
            guaranteed=int(y[test][greater].sum()); tied_bad=int(y[test][equal].sum()); tied_n=int(equal.sum())
            topk.append(dict(model=name,fraction=fraction,k=k,
                expected_TP_random_ties=guaranteed+seats*tied_bad/tied_n,
                min_TP_with_ties=guaranteed+max(0,seats-(tied_n-tied_bad)),
                max_TP_with_ties=guaranteed+min(seats,tied_bad),total_defects=int(y[test].sum())))
    for backend,bank in exp.banks.items():
        pd.DataFrame(bank.records).to_csv(out/f'{backend}_final_relationships.csv',index=False)
        normal_records.extend([dict(backend=backend,**r) for r in normal_generalization(bank,ctx,cur,y[test])])
        residual=bank.residuals(ctx,cur)
        for i in range(len(test)):
            for target in residual.iloc[i].abs().sort_values(ascending=False).head(3).index:
                item=bank.models[target]; observed=float(cur.loc[i,target]); r=float(residual.loc[i,target])
                explanations.append(dict(row=int(test[i]),label=int(y[test[i]]),backend=backend,target=target,
                    observed=observed,expected=observed-r*item['scale'],scaled_residual=r,
                    meaning='relationship deviation, NOT causal attribution or defect probability'))
    pd.DataFrame(test_rows).to_csv(out/'test_comparison.csv',index=False)
    pd.concat(test_predictions).to_csv(out/'test_predictions.csv',index=False)
    pd.DataFrame(topk).to_csv(out/'test_topk_tie_aware.csv',index=False)
    pd.DataFrame(normal_records).to_csv(out/'test_normal_relationship_quality.csv',index=False)
    pd.DataFrame(explanations).to_csv(out/'test_residual_explanations.csv',index=False)
    # Export executable standardized linear relationship equations, not physical laws.
    equations=[]
    for target,item in exp.banks['ridge'].models.items():
        scaler,reg=item['model'].steps[0][1],item['model'].steps[1][1]
        equations.append(dict(target=target,intercept=float(reg.intercept_),
            terms=[dict(predictor=c,coefficient=float(co),center=float(mu),scale=float(sd))
                   for c,co,mu,sd in zip(item['predictors'],reg.coef_,scaler.mean_,scaler.scale_)],
            equation='predicted target = intercept + sum(coefficient * (predictor - center) / scale)'))
    write_json(out/'linear_relationship_equations.json',equations)
    assert digest(out/'selection_manifest.json')==selection_digest
    chosen=next(r for r in test_rows if r['selected_on_validation'])
    tp,p=chosen['TP'],chosen['positives']
    ci=[float(beta.ppf(.025,tp,p-tp+1)) if tp else 0.,float(beta.ppf(.975,tp+1,p-tp)) if tp<p else 1.] if p else None
    report=dict(**info,status='completed',validation_selected=choice['model'],
        validation_F1=choice['F1'],test=chosen,recall_binomial_95_interval=ci,
        interval_note='descriptive exact-binomial interval; independence may fail, selection/data shift not covered',
        selection_frozen_before_test=True,model_reload_identical=True,
        retained_relations={b:len(bank.targets) for b,bank in exp.banks.items()},
        test_role='previously inspected follow-up; no deployment claim')
    write_json(out/'verification.json',report)
    print(ds,protocol,'SELECTED',choice['model'],'validation F1',round(choice['F1'],4),
          'test',chosen['TP'],chosen['FP'],chosen['FN'],flush=True)
    return report

def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for folder in sorted(p for p in out.iterdir() if p.is_dir() and (p/'test_comparison.csv').exists()):
        test=pd.read_csv(folder/'test_comparison.csv').set_index('model')
        val=pd.read_csv(folder/'validation_comparison.csv').set_index('model')
        names=list(test.index); yy=np.arange(len(names))
        fig,ax=plt.subplots(figsize=(11,6))
        ax.barh(yy-.18,val.loc[names,'F1'],height=.36,label='Validation (selection)')
        ax.barh(yy+.18,test.loc[names,'F1'],height=.36,label='Test (follow-up)')
        ax.set_yticks(yy,names); ax.set_xlabel('F1');ax.set_title(folder.name+' | fixed model settings');ax.legend()
        fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=150);plt.close(fig)
        quality=pd.read_csv(folder/'relationship_oof_quality.csv')
        pivot=quality.pivot_table(index='target',columns='backend',values='oof_r2',aggfunc='median').clip(-1,1)
        fig,ax=plt.subplots(figsize=(8,8));im=ax.imshow(pivot.to_numpy(),vmin=-1,vmax=1,cmap='RdYlGn',aspect='auto')
        ax.set_yticks(range(len(pivot)),pivot.index,fontsize=8);ax.set_xticks(range(len(pivot.columns)),pivot.columns)
        ax.set_title(folder.name+' | normal-reference OOF R2');fig.colorbar(im,ax=ax)
        fig.tight_layout();fig.savefig(folder/'relationship_quality.png',dpi=140);plt.close(fig)

def summary_text(out,reports):
    lines=['KAMP 정상 관계식·잔차 학습 실험 결과','',
           '가설: 불량은 단일 값의 극단성보다 정상 공정 변수 관계의 어긋남으로 구별될 수 있는가?',
           '설계: 정상 참조 패턴 50%로 관계식 학습, 나머지 정상 및 위험으로 분류기 학습.',
           '관계식 대상: 현재 변수 하나를 나머지 변수로 예측. 시간 실험은 이전 값·과거 3행 요약을 추가.',
           '선형 Ridge와 비선형 ExtraTrees 비교. 참조 정상 내부 3-fold OOF R² >= 0.2인 관계만 사용.',
           '잔차: (실제값 - 예측값) / 정상 OOF 절대오차 90% 분위수(최소 척도 적용).',
           '분류기: 고정 설정 LR/RF. 모든 설정은 config.json에 기록. 검증 F1로 임계값·후보 선택.',
           'raw_full은 모든 학습행, raw_matched는 잔차 분류기와 같은 학습행 사용.',
           '패턴 실험은 기존 고정 분할과 위험 이력 목표 유지. 시간 실험은 원본 행 라벨로 별도 평가.',
           'CN7 시간순 후반에는 불량이 없으므로 탐지 성능 평가를 수행하지 않음.',
           '소수 위험 사례·기존 test 반복 관찰·시간순 가정·제공값의 물리 단위 미확인 한계가 있음.',
           '관계식은 예측 관계이며 물리 법칙이나 불량 원인으로 해석하지 않음.','']
    for report in reports:
        lines += [f"[{report['dataset']} / {report['protocol']}]",f"상태: {report['status']}"]
        if report['status']=='completed':
            folder=out/f"{report['dataset']}_{report['protocol']}"
            test=pd.read_csv(folder/'test_comparison.csv')
            lines += [f"검증에서 선택: {report['validation_selected']} / validation F1={report['validation_F1']:.4f}",
                      f"선택 모델 후속 test: TP={report['test']['TP']}, FP={report['test']['FP']}, FN={report['test']['FN']}, F1={report['test']['F1']:.4f}",
                      f"학습된 관계 수: {report['retained_relations']}",
                      test[['model','TP','FP','FN','F1','AP']].to_string(index=False), '']
    (out/'analysis_summary.txt').write_text('\n'.join(lines),encoding='utf-8')

def run_all(run_id):
    out=ROOT/'output/relationship_residual'/run_id
    if out.exists(): raise FileExistsError(f'Choose a new run id; preserving existing results: {out}')
    out.mkdir(parents=True)
    sources={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'kamp_data').glob('*.csv')}
    write_json(out/'config.json',dict(config=CONFIG,python=platform.python_version(),numpy=np.__version__,
        pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,
        source_hashes=sources,code_sha256=digest(ROOT/'modeling/relationship_residual_pipeline.py')))
    start=time.time();reports=[]
    for ds in ('cn7','rg3'):
        for protocol in ('fixed_pattern','temporal_assumed'):
            reports.append(run_protocol(ds,protocol,out/f'{ds}_{protocol}'))
    figures(out);summary_text(out,reports)
    assert all(digest(ROOT/name)==value for name,value in sources.items())
    write_json(out/'run_verification.json',dict(status='completed',elapsed_seconds=time.time()-start,
        original_csv_unchanged=True,reports=reports))
    print('DONE',out,round(time.time()-start,1),'seconds',flush=True)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    # Keep persisted class names importable from notebooks and other processes.
    from relationship_residual_pipeline import run_all as canonical_run
    canonical_run(args.run_id)
