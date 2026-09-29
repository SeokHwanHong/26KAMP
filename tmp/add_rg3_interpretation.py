from pathlib import Path
import json,uuid
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'modeling/ocsvm_rg3_feature_scenarios.ipynb'
n=json.loads(p.read_text(encoding='utf-8'))
for c in n['cells']:
    s=''.join(c['source']).replace('이번 모델은 모든 test를 정상으로 판정하므로 예측상 위험 집단은 비어 있습니다. ', '')
    s=s.replace("print('위험 예측 0건의 정밀도와 단일 예측 클래스의 MCC는 구현 규칙상 0입니다.')", "print('정밀도는 위험 경보 중 실제 위험의 비율입니다.')")
    c['source']=s.splitlines(keepends=True)
    for o in c.get('outputs',[]):
        if o.get('output_type')=='stream':o['text']=[line for line in o['text'] if '위험 예측 0건의 정밀도' not in line]
def add(kind,s):
    c={'cell_type':kind,'id':uuid.uuid4().hex[:8],'metadata':{},'source':s.strip().splitlines(keepends=True)}
    if kind=='code':c.update(execution_count=None,outputs=[])
    n['cells'].append(c)
add('markdown','''## 15. 오탐 부담과 재학습 점수 변화

모두 위험으로 예측하는 단순 기준과 비교하고, 개발 OOF와 최종 test의 클래스별 점수 분포를 확인합니다. Fold 모델과 최종 모델은 학습 정상 수가 다르므로 동일 임계값이 같은 경보 비율을 보장하지 않습니다. 이 비교는 사후 진단이며 임계값을 변경하지 않습니다.''')
add('code','''final_out=out/'final_test'
failure_out=final_out/'failure_analysis'
saved=joblib.load(final_out/'model_bundle.joblib')
record=json.loads((final_out/'final_manifest.json').read_text(encoding='utf-8'))
protected={f:digest(final_out/f) for f in ['model_bundle.joblib','test_predictions.csv','test_metrics.csv','final_manifest.json']}
test_pred=pd.read_csv(final_out/'test_predictions.csv',float_precision='round_trip')
oof=pd.read_csv(out/'oof_predictions.csv',float_precision='round_trip')
oof=oof.loc[oof.scenario.eq(record['scenario'])].rename(columns={'label':'y_true'})
rows=[]
for split,frame in [('development_oof',oof),('test',test_pred)]:
    for rule,pred in [('선택 모델',frame.y_pred.to_numpy()),('모두 위험',np.ones(len(frame),dtype=int))]:
        truth=frame.y_true.to_numpy();tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel()
        rows.append({'partition':split,'rule':rule,'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn),
            'precision':precision_score(truth,pred,zero_division=0),'recall':recall_score(truth,pred,zero_division=0),
            'F2':fbeta_score(truth,pred,beta=2,zero_division=0),'alert_rate':float(pred.mean()),'false_positive_rate':float(fp/(fp+tn))})
burden=pd.DataFrame(rows);burden.to_csv(failure_out/'alert_burden.csv',index=False)
print(burden.to_string(index=False))
shift=pd.read_csv(failure_out/'fold_to_final_scores.csv')
shift_summary=shift.groupby(['pattern_id','y_true']).agg(fold_score_mean=('fold_risk_score','mean'),fold_score_min=('fold_risk_score','min'),fold_score_max=('fold_risk_score','max'),final_score=('final_risk_score','first'),fold_alert_count=('fold_above_deployed_threshold','sum')).reset_index()
shift_summary['final_minus_fold_mean']=shift_summary.final_score-shift_summary.fold_score_mean
shift_summary.to_csv(failure_out/'refit_score_shift.csv',index=False)
print(shift_summary.loc[shift_summary.y_true.eq(1)].to_string(index=False))
quantiles=[]
for split,frame in [('development_oof',oof),('test',test_pred)]:
    for label,part in frame.groupby('y_true'):
        quantiles.append({'partition':split,'label':int(label),'count':len(part),**{str(q):float(part.risk_score.quantile(q)) for q in [0,.1,.5,.9,1]}})
pd.DataFrame(quantiles).to_csv(failure_out/'score_quantiles.csv',index=False)
''')
add('markdown','''## 16. 선택된 변수 구성의 개발 검증 중요도

S1의 선택 파라미터를 고정한 뒤, 각 fold의 정상 학습으로 모델을 적합합니다. 검증 원본 컬럼 하나를 20회 섞고 모든 변환을 다시 적용하여 AP 감소를 계산합니다. Test는 사용하지 않습니다. 이 중요도는 선택된 모델에 대한 개발 진단이며 인과 영향이나 독립 검증된 변수 선택 기준이 아닙니다. 제외된 컬럼은 중요도가 0이 됩니다.''')
add('code','''importance_rows=[]
for fold in range(4):
    train=assignment.loc[assignment.partition.eq('development')&assignment.cv_fold.ne(fold),'pattern_row'].to_numpy()
    valid=assignment.loc[assignment.partition.eq('development')&assignment.cv_fold.eq(fold),'pattern_row'].to_numpy()
    normal=train[y.iloc[train].to_numpy()==0]
    assert not (set(train)|set(valid))&set(data['test'])
    bundle,z=fit_feature_bundle(X.iloc[normal],record['scenario'])
    model=OneClassSVM(kernel='rbf',nu=record['nu'],gamma=record['gamma_multiplier']/z.shape[1],tol=1e-4,max_iter=100000).fit(z)
    assert model.fit_status_==0
    xv=X.iloc[valid].copy();truth=y.iloc[valid]
    baseline=average_precision_score(truth,-model.decision_function(transform_feature_bundle(bundle,xv)))
    rng=np.random.default_rng(42+fold)
    for column in X.columns:
        for repeat in range(20):
            shuffled=xv.copy();shuffled[column]=rng.permutation(shuffled[column].to_numpy())
            score=-model.decision_function(transform_feature_bundle(bundle,shuffled))
            importance_rows.append({'fold':fold,'feature':column,'repeat':repeat,'baseline_AP':baseline,'AP_drop':baseline-average_precision_score(truth,score)})
imp=pd.DataFrame(importance_rows)
imp.to_csv(out/'selected_permutation_repeats.csv',index=False)
fold_imp=imp.groupby(['feature','fold']).AP_drop.mean().reset_index()
summary=fold_imp.groupby('feature').AP_drop.agg(['mean','std']).reset_index()
summary['positive_folds']=summary.feature.map(fold_imp.assign(positive=fold_imp.AP_drop.gt(0)).groupby('feature').positive.sum())
summary=summary.sort_values('mean',ascending=False)
summary.to_csv(out/'selected_permutation_importance.csv',index=False)
print(summary.to_string(index=False))
fig,ax=plt.subplots(figsize=(10,7));plot=summary.sort_values('mean')
ax.barh(plot.feature,plot['mean'],xerr=plot['std'].fillna(0),capsize=2)
ax.axvline(0,color='black',lw=.8);ax.set(xlabel='검증 AP 감소 평균 ± fold 표준편차',title='RG3 선택 모델의 개발 컬럼 중요도')
fig.tight_layout();fig.savefig(out/'selected_importance.png',dpi=140);plt.close(fig)
display(Image(filename=str(out/'selected_importance.png')))
assert len(imp)==24*4*20
assert all(digest(final_out/f)==sha for f,sha in protected.items())
(failure_out/'supplement_verification.json').write_text(json.dumps({'test_used_for_importance':False,'permutation_evaluations':len(imp),'final_artifacts_unchanged':True,'protected_hashes':protected,'sklearn_version':sklearn.__version__},indent=2),encoding='utf-8')
''')
p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
