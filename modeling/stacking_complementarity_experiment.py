"""Nested OOF stacking, fixed pattern splits; no time-order assumption.
Run from project root: python modeling/stacking_complementarity_experiment.py
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from relationship_residual_pipeline import (
    ROOT, RelationshipBank, classifier, load_pattern, choose_threshold, counts,
    write_json, digest,
)

NAMES = ['random_forest', 'relationship', 'isolation_forest']
MODELS = NAMES + ['mean_percentile', 'stacking_lr']
OUT = ROOT / 'output/stacking_complementarity/20261001_nested_v1'
CONFIG = dict(seed=42, inner_folds=3, outer_folds='existing 4 folds',
              meta_C=0.1, primary_budget=0.1, secondary_budgets=[0.2, 0.3],
              target='exact-input pattern has any observed defect history',
              relation='Ridge residual top3, trained on normal training patterns only',
              selection='OOF expected TP at top 10%; ties AP then model name',
              caveat='Previously inspected test; exploratory, not independent deployment validation')


class BaseModels:
    def fit(self, x, y):
        normal = np.flatnonzero(y == 0)
        assert len(normal) >= 6 and len(np.unique(y)) == 2
        self.rf = classifier('rf').fit(x, y)
        nx = x.iloc[normal].reset_index(drop=True)
        self.bank = RelationshipBank('ridge').fit(nx, nx, np.arange(len(nx)))
        self.iforest = IsolationForest(n_estimators=128, max_samples='auto',
                                      random_state=42, n_jobs=1).fit(nx)
        return self

    def predict(self, x):
        f = self.bank.features(x, x)
        residual = f.get('residual_top3_mean', pd.Series(np.zeros(len(x)))).to_numpy()
        result = np.column_stack([self.rf.predict_proba(x)[:, 1],
                                  np.log1p(residual), -self.iforest.score_samples(x)])
        assert np.isfinite(result).all()
        return result


def topk(y, scores, fraction=0.1):
    y, scores = np.asarray(y), np.asarray(scores)
    k = int(np.ceil(len(y) * fraction))
    cutoff = np.sort(scores)[-k]
    high, tied = scores > cutoff, scores == cutoff
    seats, nt = k - int(high.sum()), int(tied.sum())
    base, bad = int(y[high].sum()), int(y[tied].sum())
    return dict(k=k, positives=int(y.sum()), expected_TP=base+seats*bad/nt,
                min_TP=base+max(0, seats-(nt-bad)), max_TP=base+min(seats,bad))


def percentile_mean(reference, scores):
    cols = []
    for j in range(3):
        s = np.sort(reference[:, j])
        cols.append((np.searchsorted(s,scores[:,j],side='left') +
                     np.searchsorted(s,scores[:,j],side='right')) / (2*len(s)))
    return np.mean(cols, axis=0)


def fit_nested(x, y, row_ids, excluded, audit):
    assert not (set(row_ids) & set(excluded))
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.full((len(x), 3), np.nan)
    for fold, (tr, va) in enumerate(inner.split(x, y)):
        assert not (set(row_ids[tr]) & set(row_ids[va]))
        base = BaseModels().fit(x.iloc[tr].reset_index(drop=True), y[tr])
        oof[va] = base.predict(x.iloc[va].reset_index(drop=True))
        audit.append(dict(inner_fold=fold, train=row_ids[tr].tolist(),
                          validation=row_ids[va].tolist(), excluded=list(map(int, excluded))))
    assert np.isfinite(oof).all()
    meta = make_pipeline(StandardScaler(), LogisticRegression(
        C=0.1, class_weight='balanced', max_iter=3000, random_state=42)).fit(oof,y)
    final_base = BaseModels().fit(x,y)
    return final_base, meta, oof


def predict_all(fitted, x):
    base, meta, oof = fitted
    scores = base.predict(x)
    return np.column_stack([scores, percentile_mean(oof,scores),
                            meta.predict_proba(scores)[:,1]])


def metrics(y, scores, thresholds):
    rows=[]
    for j, name in enumerate(MODELS):
        row=dict(model=name, **counts(y,scores[:,j],thresholds[name]))
        for frac in [.1,.2,.3]:
            row.update({f'top{int(frac*100)}_{k}':v for k,v in topk(y,scores[:,j],frac).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def complementarity(ids,y,scores):
    frame=pd.DataFrame({'row':ids,'label':y})
    for j,name in enumerate(NAMES):
        s=scores[:,j]; k=int(np.ceil(len(s)*.1)); cutoff=np.sort(s)[-k]
        high=s>cutoff; tied=s==cutoff
        frame[name+'_inspection_probability']=np.where(high,1.,np.where(tied,(k-high.sum())/tied.sum(),0.))
    return frame[frame.label.eq(1)]


def run(ds):
    out=OUT/ds; out.mkdir()
    x,_,y,groups,dev,test,folds,info=load_pattern(ds)
    assert len(np.unique(groups))==len(x)
    scores=np.full((len(x),5),np.nan); audit=[]; coefficients=[]; fold_results=[]
    for tag,tr,va in folds:
        print(ds,tag,'nested fit',flush=True)
        fitted=fit_nested(x.iloc[tr].reset_index(drop=True),y[tr],tr,np.r_[va,test],audit)
        scores[va]=predict_all(fitted,x.iloc[va].reset_index(drop=True))
        coefficients.append(dict(fold=tag,**dict(zip(NAMES,fitted[1][-1].coef_[0]))))
        for j,name in enumerate(MODELS):
            fold_results.append(dict(fold=tag,model=name,**topk(y[va],scores[va,j])))
    assert np.isfinite(scores[dev]).all() and np.isnan(scores[test]).all()
    thresholds={name:choose_threshold(y[dev],scores[dev,j]) for j,name in enumerate(MODELS)}
    validation=metrics(y[dev],scores[dev],thresholds)
    selected=validation.sort_values(['top10_expected_TP','AP','model'],ascending=[False,False,True]).iloc[0].model
    manifest=dict(config=CONFIG,selected=selected,thresholds=thresholds,info=info,
                  note='F1 thresholds exploratory; primary endpoint is top10 inspection yield')
    write_json(out/'selection_manifest.json',manifest)
    frozen=digest(out/'selection_manifest.json')
    validation.to_csv(out/'validation_metrics.csv',index=False)
    pd.DataFrame(fold_results).to_csv(out/'per_fold_top10.csv',index=False)
    complementarity(dev,y[dev],scores[dev]).to_csv(out/'oof_defect_complementarity.csv',index=False)
    pd.DataFrame(scores[dev],columns=MODELS).assign(row=dev,label=y[dev]).to_csv(out/'oof_predictions.csv',index=False)
    print(ds,'final fit; frozen choice',selected,flush=True)
    fitted=fit_nested(x.iloc[dev].reset_index(drop=True),y[dev],dev,test,audit)
    coefficients.append(dict(fold='final',**dict(zip(NAMES,fitted[1][-1].coef_[0]))))
    predictions=predict_all(fitted,x.iloc[test].reset_index(drop=True))
    results=metrics(y[test],predictions,thresholds)
    results.to_csv(out/'test_metrics.csv',index=False)
    pd.DataFrame(predictions,columns=MODELS).assign(row=test,label=y[test]).to_csv(out/'test_predictions.csv',index=False)
    pd.DataFrame(coefficients).to_csv(out/'meta_coefficients.csv',index=False)
    write_json(out/'nested_roles.json',audit)
    assert digest(out/'selection_manifest.json')==frozen
    for a in audit:
        assert not(set(a['train'])&set(a['validation']))
        assert not((set(a['train'])|set(a['validation']))&set(a['excluded']))
    write_json(out/'verification.json',dict(nested_roles_disjoint=True,selection_frozen=True,
               oof_complete=True,unique_pattern_groups=True,source_sha256=info['source_sha256']))
    return ds,selected,validation,results


def main():
    started=time.time(); OUT.mkdir(parents=True,exist_ok=False)
    write_json(OUT/'config.json',dict(**CONFIG,code_sha256=digest(__file__)))
    results=[run(ds) for ds in ['cn7','rg3']]
    text=['KAMP 2단계 결합 실험 결과',
          '기존 고정 패턴 분할 / 외부 4-fold, 내부 3-fold OOF / 실제 라벨로 결합 학습',
          '시간순 가정 없음. 목표는 불량 이력이 있는 입력 패턴이며 개별 제품 불량 확률이 아님.',
          '검사 예산 10%를 주 지표로 사전 고정. 동점은 무작위 선택 시 기대 탐지 수와 최소·최대 기록.',
          '검증 F1 최적 임계값 수치는 보조 지표이며 검증 수치 자체는 선택 편향이 있음.',
          '기존에 관찰한 test를 사용한 탐색 실험. 독립적인 최종 검증을 대체하지 않음.']
    cols=['model','top10_k','top10_expected_TP','top10_min_TP','top10_max_TP','positives','AP','F1','TP','FP']
    for ds,selected,val,test in results:
        text.extend(['',ds+' / 검증 선택: '+selected,'[개발 nested OOF]',val[cols].to_string(index=False),
                     '[후속 test]',test[cols].to_string(index=False)])
    (OUT/'summary.txt').write_text('\n'.join(text),encoding='utf-8')
    write_json(OUT/'completion.json',dict(status='passed',seconds=time.time()-started))
    print('\n'.join(text),flush=True)


if __name__=='__main__':
    main()
