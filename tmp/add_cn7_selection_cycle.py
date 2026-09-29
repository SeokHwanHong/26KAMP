from pathlib import Path
import json,uuid
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'modeling/ocsvm_cn7_feature_scenarios.ipynb'
n=json.loads(p.read_text(encoding='utf-8'))
def add(kind,s):
 c={'cell_type':kind,'id':uuid.uuid4().hex[:8],'metadata':{'selection_cycle':True},'source':s.strip().splitlines(keepends=True)}
 if kind=='code':c.update(execution_count=None,outputs=[])
 n['cells'].append(c)
add('markdown','''## 16. 중요도 기반 변수 선택과 재검증 계획

앞의 1~15절은 기존 시나리오 실험 기록입니다. **16~20절은 전체 원변수+13개 파생변수를 출발점으로 중요도 → 후보 구성 → 4-fold 재검증 → 최종 재학습 → test 평가를 수행하는 후속 실험**입니다. 이 단계의 결과는 `selection_cycle`에 따로 저장합니다.

검증 중요도는 실제 모델 입력 특징을 하나씩 20회 섞었을 때의 AP 감소입니다. 원변수와 파생변수의 상관관계가 깨지는 한계가 있으므로 중요도만으로 삭제를 확정하지 않습니다. 평균 양수 및 4-fold 중 3개 이상 양수라는 규칙은 후보 생성용 휴리스틱이며 최적성이 보장되지 않습니다. 원변수 전체, 원변수+파생 전체, 기존 S1도 비교에 포함합니다.

동일 개발 fold에서 중요도·파라미터·변수를 반복 선택하므로 재검증 성능 역시 선택된 개발 성능입니다. 독립 성능 추정이나 nested CV가 아닙니다. 이미 본 test는 후속 평가에만 사용하고, 그 결과로 후보·임계값을 바꾸지 않습니다. 표준화와 상수 제거는 각 fold의 정상 학습에만 적합합니다.''')
add('code','''cycle_out=out/'selection_cycle'
cycle_out.mkdir(exist_ok=True)
previous_protected={str(p.relative_to(out)):digest(p) for p in (out/'final_test').glob('*') if p.is_file()}

def cycle_fit(normal):
    scaler=StandardScaler().fit(normal)
    z=pd.DataFrame(scaler.transform(normal),columns=normal.columns,index=normal.index)
    frame=pd.concat([z,derived(z,True,True)],axis=1)
    vt=VarianceThreshold(0).fit(frame)
    names=list(frame.columns[vt.get_support()])
    fs=StandardScaler().fit(vt.transform(frame))
    return {'normal_scaler':scaler,'columns':list(normal),'vt':vt,'scaler':fs,'names':names},fs.transform(vt.transform(frame))

def cycle_transform(bundle,inputs):
    z=pd.DataFrame(bundle['normal_scaler'].transform(inputs[bundle['columns']]),columns=bundle['columns'],index=inputs.index)
    frame=pd.concat([z,derived(z,True,True)],axis=1)
    return bundle['scaler'].transform(bundle['vt'].transform(frame))

cycle_cache=[]
for fold in range(4):
    tr=assignment.loc[assignment.partition.eq('development')&assignment.cv_fold.ne(fold),'pattern_row'].to_numpy()
    va=assignment.loc[assignment.partition.eq('development')&assignment.cv_fold.eq(fold),'pattern_row'].to_numpy()
    normal=tr[y.iloc[tr].to_numpy()==0]
    assert not (set(tr)|set(va))&set(data['test']) and not set(tr)&set(va)
    b,an=cycle_fit(X.iloc[normal]);av=cycle_transform(b,X.iloc[va])
    cycle_cache.append((fold,b,an,av,va))
cycle_names=cycle_cache[0][1]['names']
assert all(b['names']==cycle_names for _,b,_,_,_ in cycle_cache),'fold별 활성 변수가 다르므로 공통 변수 정의가 필요합니다.'

def cycle_search(label,names):
    indices=[cycle_names.index(c) for c in names]
    rows=[];fold_rows=[];best=None
    for nu in NU_VALUES:
        for multiplier in GAMMA_MULTIPLIERS:
            scores=pd.Series(np.nan,index=dev);aps=[]
            for fold,b,an,av,va in cycle_cache:
                model=OneClassSVM(kernel='rbf',nu=nu,gamma=multiplier/len(indices),tol=1e-4,max_iter=100000).fit(an[:,indices])
                assert model.fit_status_==0
                s=-model.decision_function(av[:,indices]);scores.loc[va]=s
                ap=average_precision_score(y.iloc[va],s);aps.append(ap)
                fold_rows.append({'candidate':label,'nu':nu,'gamma_multiplier':multiplier,'fold':fold,'AP':ap})
            row={'candidate':label,'nu':nu,'gamma_multiplier':multiplier,'mean_AP':float(np.mean(aps)),'std_AP':float(np.std(aps,ddof=1)),'feature_count':len(names)}
            rows.append(row)
            if best is None or row['mean_AP']>best['mean_AP']:
                best={**row,'scores':scores.loc[dev].to_numpy(),'names':names}
    pd.DataFrame(rows).to_csv(cycle_out/f'{label}_search.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(cycle_out/f'{label}_fold_search.csv',index=False)
    assert np.isfinite(best['scores']).all()
    print(label,'평균 AP:',best['mean_AP'],'변수 수:',len(names),flush=True)
    return best

cycle_full=cycle_search('full',cycle_names)
''')
add('markdown','''## 17. 검증 중요도로 파생변수 후보 구성

전체 입력 모델의 개발 평균 AP 최적 설정을 사용합니다. 파생변수 중요도는 변환된 입력의 해당 열만 섞어 계산합니다. Test 라벨·입력은 중요도 계산에 사용하지 않습니다. 원변수는 유지하고 파생변수만 선택한 후보와, 모든 입력에서 안정적인 변수만 선택한 후보를 함께 재검증합니다.''')
add('code','''cycle_imp=[]
for fold,b,an,av,va in cycle_cache:
    model=OneClassSVM(kernel='rbf',nu=cycle_full['nu'],gamma=cycle_full['gamma_multiplier']/len(cycle_names),tol=1e-4,max_iter=100000).fit(an)
    assert model.fit_status_==0
    base=average_precision_score(y.iloc[va],-model.decision_function(av))
    rng=np.random.default_rng(42+fold)
    for j,feature in enumerate(cycle_names):
        for repeat in range(20):
            perm=av.copy();perm[:,j]=rng.permutation(perm[:,j])
            decrease=base-average_precision_score(y.iloc[va],-model.decision_function(perm))
            cycle_imp.append({'fold':fold,'feature':feature,'repeat':repeat,'AP_drop':decrease})
cycle_imp=pd.DataFrame(cycle_imp)
cycle_imp.to_csv(cycle_out/'importance_repeats.csv',index=False)
cycle_fold_imp=cycle_imp.groupby(['feature','fold']).AP_drop.mean().reset_index()
cycle_summary=cycle_fold_imp.groupby('feature').AP_drop.agg(['mean','std']).reset_index()
cycle_summary['positive_folds']=cycle_summary.feature.map(cycle_fold_imp.assign(positive=cycle_fold_imp.AP_drop.gt(0)).groupby('feature').positive.sum())
cycle_summary['derived']=~cycle_summary.feature.isin(X.columns)
cycle_summary=cycle_summary.sort_values('mean',ascending=False)
cycle_summary.to_csv(cycle_out/'importance_summary.csv',index=False)
originals=[c for c in cycle_names if c in X.columns]
positive=set(cycle_summary.loc[cycle_summary['mean'].gt(0),'feature'])
stable=set(cycle_summary.loc[cycle_summary['mean'].gt(0)&cycle_summary.positive_folds.ge(3),'feature'])
cycle_candidates={'full':cycle_names,'originals':originals,
    'stable_derived':[c for c in cycle_names if c in originals or c in stable],
    'positive_derived':[c for c in cycle_names if c in originals or c in positive],
    'stable_all':[c for c in cycle_names if c in stable],
    'previous_S1':[c for c in originals if c not in GROUPS['형체와 전체 주기']]}
# 동일한 입력 조합은 한 번만 탐색합니다.
cycle_unique={};cycle_alias={}
for label,names in cycle_candidates.items():
    if not names:continue
    prior=next((key for key,value in cycle_unique.items() if value==names),None)
    if prior:cycle_alias[label]=prior
    else:cycle_unique[label]=names
(cycle_out/'candidate_definitions.json').write_text(json.dumps({'candidates':cycle_unique,'aliases':cycle_alias,'rule':'mean AP drop > 0; stable also positive in >=3 of 4 folds','test_used':False},ensure_ascii=False,indent=2),encoding='utf-8')
print(cycle_summary.to_string(index=False))
print('후보별 입력 수:',{k:len(v) for k,v in cycle_unique.items()})
fig,ax=plt.subplots(figsize=(12,10));plot=cycle_summary.sort_values('mean')
ax.barh(plot.feature,plot['mean'],xerr=plot['std'].fillna(0),color=['tomato' if x else 'steelblue' for x in plot.derived])
ax.axvline(0,color='black');ax.set(xlabel='검증 AP 감소 평균 ± fold 표준편차',title='CN7 입력 중요도: 빨강은 파생변수')
fig.tight_layout();fig.savefig(cycle_out/'importance.png',dpi=140);plt.close(fig)
display(Image(filename=str(cycle_out/'importance.png')))
''')
add('markdown','''## 18. 후보별 재학습과 4-fold 재검증

각 후보를 동일한 300개 파라미터 조합으로 재탐색합니다. 최고 평균 fold AP로 후보를 선택하고 정확한 동률은 후보 정의 순서로 결정합니다. F1·F2·Recall·FP도 함께 보고합니다. 임계값은 해당 후보의 개발 OOF F2로 결정하며, 각 fold의 지표는 그 공통 임계값을 적용한 개발 진단입니다.''')
add('code','''cycle_results={'full':cycle_full}
for label,names in cycle_unique.items():
    if label!='full':cycle_results[label]=cycle_search(label,names)
cycle_rows=[];cycle_oof=[];cycle_fold_metrics=[]
for label,result in cycle_results.items():
    threshold,_=choose_threshold(y.iloc[dev].to_numpy(),result['scores'])
    result['threshold']=float(threshold)
    cycle_rows.append({k:v for k,v in result.items() if k not in ['scores','names']}|metrics(y.iloc[dev],result['scores'],threshold))
    frame=assignment.iloc[dev][['pattern_row','pattern_id','cv_fold','label']].copy()
    frame['candidate']=label;frame['risk_score']=result['scores'];frame['threshold']=threshold
    frame['y_pred']=(frame.risk_score>threshold).astype(int);cycle_oof.append(frame)
    for fold,part in frame.groupby('cv_fold'):
        cycle_fold_metrics.append({'candidate':label,'fold':int(fold),**metrics(part.label,part.risk_score.to_numpy(),threshold)})
cycle_comparison=pd.DataFrame(cycle_rows).sort_values('mean_AP',ascending=False,kind='stable')
cycle_comparison.to_csv(cycle_out/'candidate_comparison.csv',index=False)
pd.concat(cycle_oof).to_csv(cycle_out/'oof_predictions.csv',index=False)
pd.DataFrame(cycle_fold_metrics).to_csv(cycle_out/'selected_fold_metrics.csv',index=False)
cycle_choice=str(cycle_comparison.iloc[0].candidate)
cycle_selected=cycle_results[cycle_choice]
cycle_selection={k:v for k,v in cycle_selected.items() if k!='scores'}
cycle_selection.update(selection_metric='development mean fold AP',threshold_metric='development OOF F2',test_used_for_selection=False,independent_validation=False,source_hashes=data['source_hashes'],split_hash=data['split_hash'])
(cycle_out/'selection_manifest.json').write_text(json.dumps(cycle_selection,ensure_ascii=False,indent=2),encoding='utf-8')
print(cycle_comparison.to_string(index=False))
print('확정 후보:',cycle_choice,'선택 변수:',cycle_selected['names'])
fig,ax=plt.subplots(figsize=(10,5));plot=cycle_comparison.sort_values('mean_AP')
ax.barh(plot.candidate,plot.mean_AP,xerr=plot.std_AP,capsize=3)
ax.set(xlabel='개발 평균 AP ± fold 표준편차',title='중요도 기반 후보의 재검증')
fig.tight_layout();fig.savefig(cycle_out/'candidate_comparison.png',dpi=140);plt.close(fig)
display(Image(filename=str(cycle_out/'candidate_comparison.png')))
''')
add('markdown','''## 19. 개발 정상 전체 재학습과 고정 Test 후속 평가

18절에서 변수·파라미터·임계값을 저장한 후 개발 정상 전체에 변환과 모델을 다시 적합합니다. Test에는 transform과 predict만 적용합니다. 이전 test를 이미 관찰한 후속 실험이므로 독립적인 일반화 성능 증거로 해석하지 않습니다. 결과가 나빠도 test로 후보를 재선택하지 않습니다.''')
add('code','''selection_sha=digest(cycle_out/'selection_manifest.json')
normal=dev[y.iloc[dev].to_numpy()==0]
cycle_bundle,an=cycle_fit(X.iloc[normal])
assert all(c in cycle_bundle['names'] for c in cycle_selected['names'])
indices=[cycle_bundle['names'].index(c) for c in cycle_selected['names']]
model=OneClassSVM(kernel='rbf',nu=cycle_selected['nu'],gamma=cycle_selected['gamma_multiplier']/len(indices),tol=1e-4,max_iter=100000).fit(an[:,indices])
assert model.fit_status_==0
test_rows=data['test'];truth=y.iloc[test_rows].to_numpy()
values=cycle_transform(cycle_bundle,X.iloc[test_rows])[:,indices]
scores=-model.decision_function(values);threshold=cycle_selected['threshold']
test_metrics=metrics(truth,scores,threshold)
pd.DataFrame([test_metrics]).to_csv(cycle_out/'test_metrics.csv',index=False)
pred=assignment.iloc[test_rows][['pattern_row','pattern_id','label']].copy()
pred['risk_score']=scores;pred['y_pred']=(scores>threshold).astype(int);pred['threshold']=threshold
pred.to_csv(cycle_out/'test_predictions.csv',index=False)
joblib.dump({'features':cycle_bundle,'selected_names':cycle_selected['names'],'indices':indices,'model':model,'threshold':threshold},cycle_out/'model_bundle.joblib')
loaded=joblib.load(cycle_out/'model_bundle.joblib')
reloaded=-loaded['model'].decision_function(cycle_transform(loaded['features'],X.iloc[test_rows])[:,loaded['indices']])
assert np.allclose(scores,reloaded,rtol=0,atol=1e-12)
assert digest(cycle_out/'selection_manifest.json')==selection_sha
assert all(digest(out/f)==sha for f,sha in previous_protected.items())
verification={'all_fits_converged':True,'reload_scores_equal':True,'previous_final_artifacts_unchanged':True,'selection_sha256':selection_sha,'test_used_for_selection':False,'test_previously_seen':True,'sklearn_version':sklearn.__version__,'importance_evaluations':len(cycle_imp),'candidate_count':len(cycle_results)}
(cycle_out/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print('후속 test 결과:',test_metrics)
fig,axes=plt.subplots(1,3,figsize=(15,4))
pcurve,rcurve,_=precision_recall_curve(truth,scores);axes[0].plot(rcurve,pcurve)
axes[0].set(xlabel='Recall',ylabel='Precision',title='후속 Test PR 곡선')
fpr,tpr,_=roc_curve(truth,scores);axes[1].plot(fpr,tpr);axes[1].plot([0,1],[0,1],'--',color='gray')
axes[1].set(xlabel='오탐률',ylabel='Recall',title='후속 Test ROC 곡선')
cm=confusion_matrix(truth,pred.y_pred,labels=[0,1]);axes[2].imshow(cm,cmap='Blues')
for i in range(2):
    for j in range(2):axes[2].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
axes[2].set(xticks=[0,1],yticks=[0,1],xticklabels=['정상','위험'],yticklabels=['정상','위험'],xlabel='예측',ylabel='실제',title='후속 Test 혼동행렬')
fig.tight_layout();fig.savefig(cycle_out/'test_evaluation.png',dpi=140);plt.close(fig)
display(Image(filename=str(cycle_out/'test_evaluation.png')))
''')
add('markdown','''## 20. 재검증 결과 요약

아래 요약은 실행 결과에서 생성합니다. 중요도와 재검증은 모두 개발 데이터 기반이며 최종 test는 선택에 사용하지 않았습니다. 상관된 파생변수의 개별 permutation 결과만으로 공정 원인을 단정하지 않습니다.''')
add('code','''summary_text=(f"선택 후보: {cycle_choice}\\n입력 수: {len(cycle_selected['names'])}\\n"
    f"개발 평균 AP: {cycle_selected['mean_AP']:.6f} (전체 입력: {cycle_full['mean_AP']:.6f})\\n"
    f"Test TP={test_metrics['TP']}, FP={test_metrics['FP']}, FN={test_metrics['FN']}, TN={test_metrics['TN']}\\n"
    f"Test F1={test_metrics['F1']:.6f}, F2={test_metrics['F2']:.6f}, AP={test_metrics['AP']:.6f}\\n"
    "검증 데이터를 반복 사용한 개발 선택이며, 이미 관찰한 test의 후속 평가입니다.\\n"
    "기존 test 결과에 맞춰 변수를 선택한 것은 아니지만, 독립 검증은 새 데이터가 필요합니다.")
print(summary_text)
(cycle_out/'analysis_summary.md').write_text('# CN7 중요도 기반 변수 선택 후속 실험\\n\\n'+summary_text,encoding='utf-8')
''')
n['cells'][0]['source'].append('\n\n추가: 16~20절에 중요도 기반 변수 후보 구성, 4-fold 재검증, 최종 재학습 및 test 후속 평가를 포함합니다.\n')
p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
