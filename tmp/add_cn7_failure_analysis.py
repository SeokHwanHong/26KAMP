from pathlib import Path
import json,uuid
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'modeling/ocsvm_cn7_feature_scenarios.ipynb'
n=json.loads(p.read_text(encoding='utf-8'))
def add(kind,s):
 c={'cell_type':kind,'id':uuid.uuid4().hex[:8],'metadata':{},'source':s.strip().splitlines(keepends=True)}
 if kind=='code':c.update(execution_count=None,outputs=[])
 n['cells'].append(c)
add('markdown','''## 11. 탐지 실패 원인 분석

현재 test 위험 3개가 모두 누락된 원인을 진단합니다. **라벨 유형 차이, 점수 순위와 임계값, 정상 영역과의 유사성, 재학습에 따른 점수 변화**를 분리합니다. Test 라벨을 이용한 사후 설명이며, 아래 수치를 새로운 임계값이나 특징 선택의 독립 검증 근거로 사용하지 않습니다. 모델과 최종 예측은 변경하지 않습니다.''')
add('code','''from sklearn.metrics import pairwise_distances
failure_out = out / 'final_test/failure_analysis'
failure_out.mkdir(exist_ok=True)
final_out = out / 'final_test'
saved = joblib.load(final_out/'model_bundle.joblib')
final_record = json.loads((final_out/'final_manifest.json').read_text(encoding='utf-8'))
protected_hashes = {f:digest(final_out/f) for f in ['model_bundle.joblib','test_predictions.csv','test_metrics.csv','final_manifest.json']}
meta = pd.read_csv(data['folder']/'labeled_metadata.csv')
test_pred = pd.read_csv(final_out/'test_predictions.csv',float_precision='round_trip')
oof_all = pd.read_csv(out/'oof_predictions.csv',float_precision='round_trip')
oof = oof_all.loc[oof_all.scenario.eq(final_record['scenario'])].copy()
oof = oof.rename(columns={'label':'y_true'})
oof['partition']='development_oof';test_pred['partition']='test'
joined = pd.concat([oof,test_pred],ignore_index=True).merge(
    meta[['pattern_id','pattern_type','normal_count','defect_count']],on='pattern_id',validate='many_to_one')
types=[]
for (partition,kind),part in joined.groupby(['partition','pattern_type']):
    risk = part.loc[part.y_true.eq(1)]
    types.append({'partition':partition,'pattern_type':kind,'patterns':len(part),
        'risk_patterns':len(risk),'detected_risk':int(risk.y_pred.sum()),
        'missed_risk':int((risk.y_pred==0).sum()),
        'false_positives':int(((part.y_true==0)&(part.y_pred==1)).sum())})
type_table=pd.DataFrame(types)
type_table.to_csv(failure_out/'pattern_type_performance.csv',index=False)
print('원본 관측 유형별 탐지 결과')
print(type_table.to_string(index=False))
''')
add('markdown','## 12. 임계값 문제와 순위 문제 구분\n\n각 위험의 점수·순위와 그 위험까지 탐지하려면 함께 경보해야 하는 정상 수를 계산합니다. 위험 점수의 동점까지 포함하는 사후 비용 설명이며 배포 임계값 제안이 아닙니다. OCSVM 기본 경계 0에서도 탐지되는지 별도로 확인합니다.')
add('code','''threshold=saved['threshold']
rank_rows=[]
normal_scores=test_pred.loc[test_pred.y_true.eq(0),'risk_score'].to_numpy()
for _,r in test_pred.loc[test_pred.y_true.eq(1)].iterrows():
    s=float(r.risk_score)
    alerts=test_pred.risk_score>=s
    rank_rows.append({'pattern_id':r.pattern_id,'risk_score':s,'deployed_threshold':threshold,
        'score_minus_threshold':s-threshold,'rank_from_most_risky':1+int((test_pred.risk_score>s).sum()),
        'normal_scores_below_percent':100*float((normal_scores<s).mean()),
        'TP_if_including_this_score':int((alerts & test_pred.y_true.eq(1)).sum()),
        'FP_if_including_this_score':int((alerts & test_pred.y_true.eq(0)).sum())})
rank_table=pd.DataFrame(rank_rows).sort_values('risk_score',ascending=False)
rank_table.to_csv(failure_out/'missed_risk_ranks.csv',index=False)
fixed=[]
for label,t in [('開発'.replace('開発','개발')+' OOF 선택',threshold),('OCSVM 기본 경계',0.0)]:
    fixed.append({'rule':label,'threshold':t,**metrics(test_pred.y_true,test_pred.risk_score.to_numpy(),t)})
pd.DataFrame(fixed).to_csv(failure_out/'fixed_threshold_diagnostics.csv',index=False)
print(rank_table.to_string(index=False))
print(pd.DataFrame(fixed)[['rule','threshold','TP','FP','FN','F2']].to_string(index=False))
''')
add('markdown','## 13. 정상 학습 영역과의 유사성\n\n최종 모델의 21차원 변환 공간에서 최근접 정상 패턴을 찾습니다. 정상 학습끼리의 자기 자신을 제외한 최근접 거리 분포와 비교합니다. 이는 거리 진단이며 확률·인과 설명이 아닙니다. 각 변수의 정상 범위 안에 있다고 해서 결합 분포까지 정상이라는 뜻은 아닙니다.')
add('code','''normal_rows=data['dev'][y.iloc[data['dev']].to_numpy()==0]
normal_z=transform_feature_bundle(saved['features'],X.iloc[normal_rows])
test_rows=test_pred.pattern_row.to_numpy()
test_z=transform_feature_bundle(saved['features'],X.iloc[test_rows])
normal_dist=pairwise_distances(normal_z)
np.fill_diagonal(normal_dist,np.inf)
normal_nn=normal_dist.min(axis=1)
dist=pairwise_distances(test_z,normal_z)
nearest_index=dist.argmin(axis=1)
nearest_distance=dist.min(axis=1)
retained=saved['features']['retained_features']
distance_rows=[];difference_rows=[]
for i,r in test_pred.reset_index(drop=True).iterrows():
    nearest_row=int(normal_rows[nearest_index[i]])
    within=(test_z[i]>=normal_z.min(axis=0))&(test_z[i]<=normal_z.max(axis=0))
    distance_rows.append({'pattern_id':r.pattern_id,'y_true':int(r.y_true),
        'nearest_normal_pattern_id':assignment.iloc[nearest_row].pattern_id,
        'nearest_distance':float(nearest_distance[i]),
        'normal_NN_distance_percentile':100*float((normal_nn<=nearest_distance[i]).mean()),
        'within_normal_feature_ranges':int(within.sum()),'feature_count':len(retained)})
    if r.y_true==1:
        for j,feature in enumerate(retained):
            difference_rows.append({'pattern_id':r.pattern_id,'feature':feature,
                'risk_value':test_z[i,j],'nearest_normal_value':normal_z[nearest_index[i],j],
                'difference':test_z[i,j]-normal_z[nearest_index[i],j]})
distance_table=pd.DataFrame(distance_rows)
distance_table.to_csv(failure_out/'normal_neighborhood.csv',index=False)
pd.DataFrame(difference_rows).to_csv(failure_out/'missed_risk_feature_differences.csv',index=False)
print(distance_table.loc[distance_table.y_true.eq(1)].to_string(index=False))
''')
add('markdown','## 14. Fold 모델과 최종 모델의 점수 비교\n\n개발에서 선택한 동일 설정으로 fold 모델 4개를 다시 적합하고 test 위험의 점수를 진단합니다. 어떤 fold 모델에서도 점수가 음수라면 최종 재학습만으로 생긴 누락이라는 설명은 약해집니다. fold 모델은 학습 수가 다르고, 다른 fold에서는 개발 OOF 행이 학습에 포함될 수 있으므로 이 비교를 별도 성능 평가로 사용하지 않습니다.')
add('code','''shift_rows=[]
for fold in range(4):
    train=assignment.loc[assignment.partition.eq('development')&assignment.cv_fold.ne(fold),'pattern_row'].to_numpy()
    normal=train[y.iloc[train].to_numpy()==0]
    assert not set(train)&set(data['test'])
    b,z=fit_feature_bundle(X.iloc[normal],final_record['scenario'])
    m=OneClassSVM(kernel='rbf',nu=final_record['nu'],gamma=final_record['gamma_multiplier']/z.shape[1],tol=1e-4,max_iter=100000).fit(z)
    assert m.fit_status_==0
    s=-m.decision_function(transform_feature_bundle(b,X.iloc[test_rows]))
    for i,r in test_pred.reset_index(drop=True).iterrows():
        shift_rows.append({'pattern_id':r.pattern_id,'y_true':int(r.y_true),'fold_model':fold,
            'fold_risk_score':float(s[i]),'final_risk_score':float(r.risk_score),
            'fold_above_zero':bool(s[i]>0),'fold_above_deployed_threshold':bool(s[i]>threshold)})
shift=pd.DataFrame(shift_rows)
shift.to_csv(failure_out/'fold_to_final_scores.csv',index=False)
print(shift.loc[shift.y_true.eq(1)].to_string(index=False))
fig,axes=plt.subplots(1,3,figsize=(16,4.8))
risks=joined.loc[joined.y_true.eq(1)]
for i,(kind,label,color) in enumerate([('conflicting','정상·불량 공존','steelblue'),('defect_only','불량 전용','tomato')]):
    part=risks.loc[risks.partition.eq('development_oof')&risks.pattern_type.eq(kind)]
    axes[0].scatter(np.full(len(part),i),part.risk_score,c=color,label=label)
axes[0].scatter(np.full(3,2),test_pred.loc[test_pred.y_true.eq(1),'risk_score'],c='crimson',marker='X',label='Test 위험')
axes[0].axhline(threshold,ls='--',color='black',label='고정 임계값')
axes[0].set(xticks=[0,1,2],xticklabels=['개발 공존','개발 불량 전용','Test 공존'],ylabel='위험 점수',title='위험 유형별 점수');axes[0].legend(fontsize=8)
order=np.argsort(test_pred.risk_score.to_numpy())
for label,color,marker in [(0,'steelblue','o'),(1,'crimson','X')]:
    mask=test_pred.y_true.to_numpy()[order]==label
    axes[1].scatter(np.arange(len(order))[mask],test_pred.risk_score.to_numpy()[order][mask],c=color,marker=marker,s=50 if label else 15)
axes[1].axhline(threshold,color='black',ls='--');axes[1].axhline(0,color='gray',ls=':')
axes[1].set(xlabel='점수 정렬 순서',ylabel='위험 점수',title='선택 임계값과 기본 경계 0')
axes[2].hist(normal_nn,bins=25,color='lightgray',label='정상 학습의 최근접 거리')
for _,r in distance_table.loc[distance_table.y_true.eq(1)].iterrows():axes[2].axvline(r.nearest_distance,label=r.pattern_id)
axes[2].set(xlabel='변환 공간의 최근접 정상 거리',ylabel='패턴 수',title='누락 위험의 정상 근접도');axes[2].legend(fontsize=8)
fig.tight_layout();fig.savefig(failure_out/'failure_diagnostics.png',dpi=150);plt.close(fig)
display(Image(filename=str(failure_out/'failure_diagnostics.png')))
assert all(digest(final_out/f)==sha for f,sha in protected_hashes.items())
(failure_out/'verification.json').write_text(json.dumps({'final_model_and_predictions_unchanged':True,
    'test_analysis_only':True,'threshold_retuned':False,'protected_hashes':protected_hashes},indent=2),encoding='utf-8')
print('실패 원인 진단 저장:',failure_out)
''')
p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
