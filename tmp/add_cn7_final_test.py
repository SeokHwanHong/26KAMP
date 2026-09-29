from pathlib import Path
import json,uuid
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'modeling/ocsvm_cn7_feature_scenarios.ipynb'
n=json.loads(p.read_text(encoding='utf-8'))
def add(kind,s):
 c={'cell_type':kind,'id':uuid.uuid4().hex[:8],'metadata':{},'source':s.strip().splitlines(keepends=True)}
 if kind=='code':c.update(execution_count=None,outputs=[])
 n['cells'].append(c)
add('markdown','''## 7. 개발 성능으로 최종 변수 구성 확정

시나리오별 최고 평균 검증 AP를 비교합니다. 정확한 동률이면 시나리오 ID 순서로 선택합니다. 파라미터와 임계값도 해당 시나리오의 개발 결과만 사용합니다. 학습 점수나 test 점수로 선택하지 않습니다.

기존 기준 모델의 test를 이미 확인한 이력이 있으므로, 이번 결과는 같은 holdout의 후속 평가입니다. 새로운 독립 test 검증이라고 표현하지 않습니다. 이번 평가 후 test에 맞춰 변수나 임계값을 바꾸지 않습니다.''')
add('code','''# 저장된 개발 결과를 사용하므로 전체 탐색을 다시 실행하지 않아도 됩니다.
comparison_for_final = pd.read_csv(out / 'scenario_comparison.csv', float_precision='round_trip')
experiment = json.loads((out / 'experiment_manifest.json').read_text(encoding='utf-8'))
assert experiment['source_hashes'] == data['source_hashes']
assert experiment['split_hash'] == data['split_hash']
assert digest(out / 'scenario_comparison.csv') == experiment['artifact_sha256']['scenario_comparison.csv']
chosen = comparison_for_final.sort_values(['mean_AP', 'scenario'], ascending=[False, True], kind='stable').iloc[0]
selected_scenario = str(chosen.scenario)
selected_nu, selected_multiplier = float(chosen.nu), float(chosen.gamma_multiplier)
selected_threshold = float(chosen.threshold)
print('선택 시나리오:', selected_scenario, SCENARIOS[selected_scenario])
print('평균 검증 AP:', chosen.mean_AP, '| nu:', selected_nu, '| gamma 배수:', selected_multiplier)
print('개발 OOF에서 확정한 임계값:', selected_threshold)
''')
add('markdown','## 8. 개발 정상 전체 적합과 고정 Test 평가\n\n파생변수 변환까지 저장 가능한 형태로 구성합니다. 정상 개발 데이터에서 모든 변환을 적합하고 test에는 transform만 적용합니다. 기존 기준 모델 산출물은 덮어쓰지 않습니다.')
add('code','''def fit_feature_bundle(normal, scenario):
    scaler = StandardScaler().fit(normal)
    z = pd.DataFrame(scaler.transform(normal), columns=normal.columns, index=normal.index)
    bundle = {'scenario': scenario, 'input_columns': list(normal), 'normal_scaler': scaler, 'pcas': {}}
    if scenario == 'S5':
        frame = pd.DataFrame(index=normal.index)
        for group, cols in GROUPS.items():
            active = [c for c in cols if normal[c].nunique() > 1]
            if not active: continue
            pca = PCA(n_components=min(2, len(active)), svd_solver='full').fit(z[active])
            scores = pca.transform(z[active])
            bundle['pcas'][group] = {'columns': active, 'pca': pca}
            for j in range(scores.shape[1]): frame[f'{group}_PC{j+1}'] = scores[:, j]
            if len(active) > scores.shape[1]:
                frame[f'{group}_재구성RMSE'] = np.sqrt(np.mean((z[active].to_numpy()-pca.inverse_transform(scores))**2, axis=1))
    else:
        cols = [c for c in normal if scenario not in ['S1','S4'] or c not in GROUPS['형체와 전체 주기']]
        bundle['base_columns'] = cols
        frame = z[cols].copy()
        if scenario in ['S2','S3','S4']:
            frame = pd.concat([frame, derived(z, scenario in ['S2','S4'], scenario in ['S3','S4'])], axis=1)
    bundle['feature_columns'] = list(frame)
    vt = VarianceThreshold(0).fit(frame)
    transformed = vt.transform(frame)
    bundle['constant_filter'] = vt
    bundle['feature_scaler'] = None
    if scenario not in ['S0','S1']:
        bundle['feature_scaler'] = StandardScaler().fit(transformed)
        transformed = bundle['feature_scaler'].transform(transformed)
    bundle['retained_features'] = list(frame.columns[vt.get_support()])
    return bundle, transformed

def transform_feature_bundle(bundle, inputs):
    inputs = inputs[bundle['input_columns']]
    z = pd.DataFrame(bundle['normal_scaler'].transform(inputs), columns=inputs.columns, index=inputs.index)
    scenario = bundle['scenario']
    if scenario == 'S5':
        frame = pd.DataFrame(index=inputs.index)
        for group, record in bundle['pcas'].items():
            cols, pca = record['columns'], record['pca']
            scores = pca.transform(z[cols])
            for j in range(scores.shape[1]): frame[f'{group}_PC{j+1}'] = scores[:, j]
            if len(cols) > scores.shape[1]:
                frame[f'{group}_재구성RMSE'] = np.sqrt(np.mean((z[cols].to_numpy()-pca.inverse_transform(scores))**2, axis=1))
    else:
        frame = z[bundle['base_columns']].copy()
        if scenario in ['S2','S3','S4']:
            frame = pd.concat([frame, derived(z, scenario in ['S2','S4'], scenario in ['S3','S4'])], axis=1)
    values = bundle['constant_filter'].transform(frame[bundle['feature_columns']])
    if bundle['feature_scaler'] is not None: values = bundle['feature_scaler'].transform(values)
    assert np.isfinite(values).all()
    return values

development_rows, test_rows = data['dev'], data['test']
assert not set(development_rows) & set(test_rows)
normal_rows = development_rows[y.iloc[development_rows].to_numpy() == 0]
bundle, normal_values = fit_feature_bundle(X.iloc[normal_rows], selected_scenario)
test_values = transform_feature_bundle(bundle, X.iloc[test_rows])
# 探索時と同じ変換かを検証します。
reference_normal, reference_test, reference_columns, _ = representation(X.iloc[normal_rows], X.iloc[test_rows], selected_scenario)
assert np.allclose(normal_values, reference_normal) and np.allclose(test_values, reference_test)
assert bundle['retained_features'] == reference_columns
gamma = selected_multiplier / normal_values.shape[1]
final_model = OneClassSVM(kernel='rbf', nu=selected_nu, gamma=gamma, tol=1e-4, max_iter=100000).fit(normal_values)
assert final_model.fit_status_ == 0
test_scores = -final_model.decision_function(test_values)
assert np.isfinite(test_scores).all()
test_result = metrics(y.iloc[test_rows], test_scores, selected_threshold)
print('최종 학습 정상 패턴 수:', len(normal_rows), '| 실제 입력 변수 수:', normal_values.shape[1])
print('고정 test 결과:', test_result)
print('최종 입력 변수:', bundle['retained_features'])
'''.replace('# 探索時と同じ変換かを検証します。','# 개발 탐색과 동일한 특징 변환인지 검증합니다.'))
add('markdown','## 9. 최종 평가와 변환 묶음 저장\n\n최종 변환·모델·임계값을 final_test 폴더에 저장합니다. 비라벨 추론은 표준화 좌표계 정합을 확인한 이후에 수행합니다. 이 노트북의 transform_feature_bundle과 derived 함수로 저장된 변환 묶음을 적용할 수 있습니다.')
add('code','''final_out = out / 'final_test'
final_out.mkdir(exist_ok=True)
final_predictions = assignment.iloc[test_rows][['pattern_row','pattern_id']].copy()
final_predictions['y_true'] = y.iloc[test_rows].to_numpy()
final_predictions['risk_score'] = test_scores
final_predictions['threshold'] = selected_threshold
final_predictions['y_pred'] = (test_scores > selected_threshold).astype(int)
final_predictions['scenario'] = selected_scenario
final_predictions.to_csv(final_out/'test_predictions.csv', index=False)
pd.DataFrame([{'scenario':selected_scenario,'rows':len(test_rows),'risk_patterns':int(y.iloc[test_rows].sum()),**test_result}]).to_csv(final_out/'test_metrics.csv',index=False)
artifact = {'features':bundle,'model':final_model,'threshold':selected_threshold,'score_definition':'-decision_function'}
joblib.dump(artifact, final_out/'model_bundle.joblib')
loaded = joblib.load(final_out/'model_bundle.joblib')
reload_scores = -loaded['model'].decision_function(transform_feature_bundle(loaded['features'],X.iloc[test_rows]))
assert np.array_equal(test_scores,reload_scores)
fig,axes=plt.subplots(1,3,figsize=(15,4))
yt=y.iloc[test_rows].to_numpy()
precision,recall,_=precision_recall_curve(yt,test_scores)
axes[0].plot(recall,precision);axes[0].axhline(yt.mean(),ls='--',color='gray')
axes[0].set(xlabel='재현율',ylabel='정밀도',title='고정 Test PR 곡선')
fpr,tpr,_=roc_curve(yt,test_scores);axes[1].plot(fpr,tpr);axes[1].plot([0,1],[0,1],'--',color='gray')
axes[1].set(xlabel='오탐률',ylabel='재현율',title='고정 Test ROC 곡선')
cm=confusion_matrix(yt,final_predictions.y_pred,labels=[0,1]);axes[2].imshow(cm,cmap='Blues')
for i in range(2):
    for j in range(2): axes[2].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
axes[2].set(xticks=[0,1],yticks=[0,1],xticklabels=['정상','위험'],yticklabels=['정상','위험'],xlabel='예측',ylabel='실제',title='혼동행렬')
fig.suptitle('CN7 '+selected_scenario+' 개발 선택 후 고정 Test 평가');fig.tight_layout()
fig.savefig(final_out/'test_evaluation.png',dpi=140);plt.close(fig)
display(Image(filename=str(final_out/'test_evaluation.png')))
config={'scenario':selected_scenario,'nu':selected_nu,'gamma_multiplier':selected_multiplier,'gamma':gamma,
    'threshold':selected_threshold,'selection_metric':'development mean fold AP',
    'threshold_metric':'development OOF F2','source_hashes':data['source_hashes'],'split_hash':data['split_hash'],
    'retained_features':bundle['retained_features'],'normal_train_rows':len(normal_rows),
    'test_rows':len(test_rows),'test_used_for_selection':False,'previous_baseline_test_seen':True,
    'reload_predictions_equal':True,'sklearn_version':sklearn.__version__,
    'artifact_sha256':{p.name:digest(p) for p in final_out.iterdir() if p.is_file() and p.name!='final_manifest.json'}}
(final_out/'final_manifest.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
print('최종 평가 저장:',final_out)
''')
# Update the scope statement now that a separate final-test section exists.
s=''.join(n['cells'][0]['source']).replace('고정 test와 비라벨 데이터는 평가하지 않습니다.', '시나리오 탐색에는 고정 test를 사용하지 않으며, 7~9절에서 개발 선택을 확정한 뒤 test를 평가합니다. 비라벨 추론은 수행하지 않습니다.')
n['cells'][0]['source']=s.splitlines(keepends=True)
p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
