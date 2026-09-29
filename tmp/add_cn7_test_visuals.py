from pathlib import Path
import json,uuid
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'modeling/ocsvm_cn7_feature_scenarios.ipynb'
n=json.loads(p.read_text(encoding='utf-8'))
def add(kind,s):
 c={'cell_type':kind,'id':uuid.uuid4().hex[:8],'metadata':{},'source':s.strip().splitlines(keepends=True)}
 if kind=='code':c.update(execution_count=None,outputs=[])
 n['cells'].append(c)
add('markdown','''## 10. 고정 Test 평가지표와 2차원 분포

저장된 최종 모델과 임계값을 그대로 사용합니다. OCSVM은 군집 번호를 만드는 모델이 아니므로, 아래 그림은 군집 결과 대신 **입력 공간의 분포와 정상·위험 판정**을 시각화합니다. PCA는 최종 모델 입력으로 변환한 개발 정상 데이터에서만 적합하고 test에 적용합니다. 2차원 그림의 거리만으로 원래 고차원 모델의 경계를 해석하지 않습니다.

실제 라벨, 예측 라벨, 연속 위험 점수, 행별 임계값 비교를 나란히 봅니다. 이번 모델은 모든 test를 정상으로 판정하므로 예측상 위험 집단은 비어 있습니다. 위험 예측이 0건이면 정밀도는 정의되지 않으며, 보고 규칙에 따라 0으로 표기합니다.''')
add('code','''from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef

saved = joblib.load(out / 'final_test/model_bundle.joblib')
final_out = out / 'final_test'
stored = pd.read_csv(final_out / 'test_predictions.csv', float_precision='round_trip')
rows = stored.pattern_row.to_numpy()
assert set(rows) == set(data['test'])
actual = y.iloc[rows].to_numpy()
values = transform_feature_bundle(saved['features'], X.iloc[rows])
scores = -saved['model'].decision_function(values)
threshold = saved['threshold']
predicted = (scores > threshold).astype(int)
assert np.array_equal(actual, stored.y_true) and np.array_equal(predicted, stored.y_pred)
assert np.allclose(scores, stored.risk_score, rtol=0, atol=1e-12)
tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0,1]).ravel()
base_metrics = metrics(actual, scores, threshold)
metric_table = pd.DataFrame([
    ('Test 패턴 수', len(actual)), ('실제 위험 수', int(actual.sum())),
    ('위험 예측 수', int(predicted.sum())), ('TP 탐지 위험', int(tp)), ('FP 정상 오탐', int(fp)),
    ('FN 위험 누락', int(fn)), ('TN 정상 판정', int(tn)),
    ('Precision 정밀도', base_metrics['precision']), ('Recall 재현율', base_metrics['recall']),
    ('F1',base_metrics['F1']), ('F2',base_metrics['F2']), ('AP',base_metrics['AP']),
    ('ROC-AUC',base_metrics['ROC_AUC']), ('Accuracy 정확도',accuracy_score(actual,predicted)),
    ('Balanced accuracy 균형 정확도',balanced_accuracy_score(actual,predicted)),
    ('Specificity 특이도',float(tn/(tn+fp))), ('MCC',matthews_corrcoef(actual,predicted)),
    ('고정 임계값',threshold),
],columns=['평가 항목','값'])
metric_table.to_csv(final_out/'test_metrics_detailed.csv',index=False,encoding='utf-8-sig')
print(metric_table.to_string(index=False))
print('위험 예측 0건의 정밀도와 단일 예측 클래스의 MCC는 구현 규칙상 0입니다.')

normal_dev = data['dev'][y.iloc[data['dev']].to_numpy()==0]
normal_values = transform_feature_bundle(saved['features'],X.iloc[normal_dev])
projection = PCA(n_components=2,svd_solver='full').fit(normal_values)
normal_xy = projection.transform(normal_values)
test_xy = projection.transform(values)
ratios = projection.explained_variance_ratio_
coordinates = stored.copy()
coordinates['PC1'],coordinates['PC2'] = test_xy[:,0],test_xy[:,1]
coordinates.to_csv(final_out/'test_pca_coordinates.csv',index=False)
joblib.dump(projection,final_out/'visualization_pca.joblib')
fig,axes=plt.subplots(2,2,figsize=(13,10))
for ax in axes.flat[:3]:
    ax.scatter(normal_xy[:,0],normal_xy[:,1],s=12,c='lightgray',alpha=.35,label='개발 정상 배경')
    ax.set(xlabel=f'PC1 ({ratios[0]:.1%})',ylabel=f'PC2 ({ratios[1]:.1%})')
for label,color,marker,text in [(0,'steelblue','o','정상'),(1,'crimson','X','위험')]:
    mask=actual==label
    axes[0,0].scatter(test_xy[mask,0],test_xy[mask,1],s=85 if label else 30,c=color,marker=marker,label=f'실제 {text} {mask.sum()}개',edgecolors='black',linewidths=.4)
    mask=predicted==label
    axes[0,1].scatter(test_xy[mask,0],test_xy[mask,1],s=85 if label else 30,c=color,marker=marker,label=f'예측 {text} {mask.sum()}개',edgecolors='black',linewidths=.4)
for ax in axes[0]:ax.legend(fontsize=9)
axes[0,0].set_title('Test 실제 라벨의 분포')
axes[0,1].set_title('Test 예측 판정의 분포')
color_plot=axes[1,0].scatter(test_xy[:,0],test_xy[:,1],c=scores,cmap='viridis',s=35)
fig.colorbar(color_plot,ax=axes[1,0],label='위험 점수 확률 아님')
axes[1,0].set_title('연속 위험 점수')
for index in np.flatnonzero(actual==1):
    axes[0,0].annotate(str(stored.iloc[index].pattern_id),test_xy[index],xytext=(5,7),textcoords='offset points',fontsize=7)
order=np.argsort(scores)
sorted_scores=scores[order];sorted_actual=actual[order]
for label,color,marker,text in [(0,'steelblue','o','실제 정상'),(1,'crimson','X','실제 위험')]:
    mask=sorted_actual==label
    axes[1,1].scatter(np.arange(len(order))[mask],sorted_scores[mask],c=color,marker=marker,s=60 if label else 20,label=text)
axes[1,1].axhline(threshold,c='black',ls='--',label='개발 OOF 임계값')
axes[1,1].set(xlabel='점수 정렬 순서 시간순 아님',ylabel='위험 점수',title='Test 점수와 고정 임계값')
axes[1,1].legend(fontsize=9)
fig.suptitle(f'CN7 최종 Test 분포와 판정 | PCA 설명분산 합계 {ratios.sum():.1%}')
fig.tight_layout();fig.savefig(final_out/'test_distribution.png',dpi=150);plt.close(fig)
display(Image(filename=str(final_out/'test_distribution.png')))
(final_out/'visualization_info.json').write_text(json.dumps({'pca_fit':'development normal only after final feature transform',
    'explained_variance_ratio':ratios.tolist(),'threshold_unchanged':True,'test_predictions_unchanged':True,
    'is_clustering':False},ensure_ascii=False,indent=2),encoding='utf-8')
''')
p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
