from pathlib import Path
import json,uuid
ROOT=Path(__file__).resolve().parents[1]
base=json.loads((ROOT/'modeling/ocsvm_cn7_exploration.ipynb').read_text(encoding='utf-8'))
cells=[]
def add(kind,s):
 c={'cell_type':kind,'id':uuid.uuid4().hex[:8],'metadata':{},'source':s.strip().splitlines(keepends=True)}
 if kind=='code':c.update(execution_count=None,outputs=[])
 cells.append(c)
add('markdown','''# CN7 공정 파생변수 시나리오

개발 4-fold에서 파생변수가 OCSVM 위험 조건 탐지에 도움이 되는지 비교합니다. 고정 test와 비라벨 데이터는 평가하지 않습니다. 기존 중요도 결과를 보고 구성한 탐색이므로 결과는 개발 성능이며 독립 검증된 개선이 아닙니다.

제공값은 컬럼별로 이미 표준화되어 있습니다. 아래 z는 각 fold의 정상 학습 기준으로 다시 표준화한 값입니다. z의 차이는 물리 단위의 차이가 아니라 정상 대비 상대적 편차의 차이이고, 곱은 통계적 상호작용입니다. 압력×속도를 실제 동력으로, 온도 차이를 실제 섭씨 차이로 해석하지 않습니다.

모든 시나리오에서 nu 20개, gamma 배수 15개를 각각 다시 탐색합니다. 300개×4-fold×6개 시나리오입니다. 전처리·파생변수 정규화·PCA는 정상 학습에만 적합합니다.''')
setup=''.join(base['cells'][2]['source']).replace("OUTPUT = ROOT / 'output/ocsvm_development_exploration'", "OUTPUT = ROOT / 'output/ocsvm_cn7_feature_scenarios'")
add('markdown','## 1. 설정과 고정 분할 확인')
add('code',setup)
add('code',''.join(base['cells'][3]['source']))
add('code',''.join(base['cells'][4]['source']))
add('markdown','''## 2. 시나리오 정의

| ID | 입력 구성 | 검토할 가설 |
|---|---|---|
| S0 | 전체 원변수 | 기준선 |
| S1 | 형체·전체 주기 제외 | 기존 개발 분석의 제외 효과를 재최적화로 확인 |
| S2 | 전체 원변수 + 편차·온도 분포 특징 | 공정 변수 사이 상대적 불일치가 도움이 되는가 |
| S3 | 전체 원변수 + 선택 상호작용 | 사출 속도와 시간·압력·쿠션의 결합이 도움이 되는가 |
| S4 | 형체·전체 주기 제외 + S2·S3 특징 | 제외 효과와 파생변수 효과를 결합하면 어떤가 |
| S5 | 공정별 PCA 점수와 재구성 오차 | 상관된 입력을 공정별 상태로 압축하면 어떤가 |

S2: 사출시간−충전시간, 최대−평균 스크루 RPM, 최대−평균 배압, 사출−전환 압력, 금형온도3−4, 배럴 z 평균·표준편차·범위, 금형 z 평균.

S3: 최대사출속도×충전시간, 최대사출속도×최대사출압력, 최대사출속도×쿠션위치, 계량시간×평균배압.

S5는 각 공정 변수군에서 최대 2개 PC와 재구성 RMSE를 사용합니다. 잔차가 항상 0인 변수는 학습에서 제거됩니다. PCA는 물리 의미가 같은 단위로 복원하는 변환이 아닙니다.''')
groups=''.join(base['cells'][6]['source']).split('name = DATASETS[0]')[0]
add('code',groups+'''
from sklearn.decomposition import PCA
name = 'cn7'
data = datasets[name]
X, y, assignment, dev = data['X'], data['y'], data['assignment'], data['dev']
out = OUTPUT
assert sorted(sum(GROUPS.values(), [])) == sorted(X.columns)
SCENARIOS = {
    'S0': '전체 원변수', 'S1': '형체·전체 주기 제외',
    'S2': '전체와 편차 특징', 'S3': '전체와 상호작용',
    'S4': '형체·주기 제외와 파생변수', 'S5': '공정별 PCA와 재구성 오차'}

def derived(z, differences, interactions):
    new = pd.DataFrame(index=z.index)
    if differences:
        for feature, a, b in [
            ('사출_충전_상대편차', 'Injection_Time', 'Filling_Time'),
            ('스크루RPM_상대편차', 'Max_Screw_RPM', 'Average_Screw_RPM'),
            ('배압_상대편차', 'Max_Back_Pressure', 'Average_Back_Pressure'),
            ('사출_전환압력_상대편차', 'Max_Injection_Pressure', 'Max_Switch_Over_Pressure'),
            ('금형온도_상대편차', 'Mold_Temperature_3', 'Mold_Temperature_4')]:
            new[feature] = z[a] - z[b]
        barrel = z[['Barrel_Temperature_' + str(i) for i in range(1, 7)]]
        new['배럴_상대수준평균'] = barrel.mean(axis=1)
        new['배럴_상대편차표준편차'] = barrel.std(axis=1, ddof=0)
        new['배럴_상대편차범위'] = barrel.max(axis=1) - barrel.min(axis=1)
        new['금형_상대수준평균'] = z[['Mold_Temperature_3', 'Mold_Temperature_4']].mean(axis=1)
    if interactions:
        for feature, a, b in [
            ('속도_충전시간_상호작용', 'Max_Injection_Speed', 'Filling_Time'),
            ('속도_사출압력_상호작용', 'Max_Injection_Speed', 'Max_Injection_Pressure'),
            ('속도_쿠션_상호작용', 'Max_Injection_Speed', 'Cushion_Position'),
            ('계량시간_평균배압_상호작용', 'Plasticizing_Time', 'Average_Back_Pressure')]:
            new[feature] = z[a] * z[b]
    return new

def representation(normal, valid, scenario):
    # 분모가 0인 상수 입력도 z=0으로 유지하여 파생식을 안정적으로 계산합니다.
    scaler = StandardScaler().fit(normal)
    zn = pd.DataFrame(scaler.transform(normal), columns=normal.columns, index=normal.index)
    zv = pd.DataFrame(scaler.transform(valid), columns=valid.columns, index=valid.index)
    pca_audit = []
    if scenario == 'S5':
        fn, fv = pd.DataFrame(index=normal.index), pd.DataFrame(index=valid.index)
        for group, cols in GROUPS.items():
            active = [c for c in cols if normal[c].nunique() > 1]
            if not active: continue
            pca = PCA(n_components=min(2, len(active)), svd_solver='full').fit(zn[active])
            sn, sv = pca.transform(zn[active]), pca.transform(zv[active])
            for j in range(sn.shape[1]):
                fn[f'{group}_PC{j+1}'], fv[f'{group}_PC{j+1}'] = sn[:, j], sv[:, j]
            if len(active) > sn.shape[1]:
                fn[f'{group}_재구성RMSE'] = np.sqrt(np.mean((zn[active].to_numpy() - pca.inverse_transform(sn))**2, axis=1))
                fv[f'{group}_재구성RMSE'] = np.sqrt(np.mean((zv[active].to_numpy() - pca.inverse_transform(sv))**2, axis=1))
            pca_audit.append({'group': group, 'features': active,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(), 'components': pca.components_.tolist()})
    else:
        cols = [c for c in normal if scenario not in ['S1', 'S4'] or c not in GROUPS['형체와 전체 주기']]
        fn, fv = zn[cols].copy(), zv[cols].copy()
        if scenario in ['S2', 'S3', 'S4']:
            fn = pd.concat([fn, derived(zn, scenario in ['S2','S4'], scenario in ['S3','S4'])], axis=1)
            fv = pd.concat([fv, derived(zv, scenario in ['S2','S4'], scenario in ['S3','S4'])], axis=1)
    # 기준선에는 기존과 같은 1회 표준화, 추가 특징에는 전체 표현의 스케일을 맞춥니다.
    vt = VarianceThreshold(0).fit(fn)
    an, av = vt.transform(fn), vt.transform(fv)
    if scenario not in ['S0', 'S1']:
        feature_scaler = StandardScaler().fit(an)
        an, av = feature_scaler.transform(an), feature_scaler.transform(av)
    assert np.isfinite(an).all() and np.isfinite(av).all()
    return an, av, list(fn.columns[vt.get_support()]), pca_audit
''')
add('markdown','## 3. fold별 특징 생성\n\n각 fold의 정상 학습에서 특징을 적합한 뒤 검증에 적용합니다. 학습 결과를 파라미터 후보 간 재사용하여 반복 계산을 줄입니다. 파생변수를 추가하면 RBF 거리의 변수별 비중도 달라지므로, 새로운 정보뿐 아니라 거리 구조 변화의 효과가 포함됩니다.')
add('code','''cache, feature_audit = {}, []
for scenario in SCENARIOS:
    cache[scenario] = []
    for fold in range(4):
        train = assignment.loc[assignment.partition.eq('development') & assignment.cv_fold.ne(fold), 'pattern_row'].to_numpy()
        valid = assignment.loc[assignment.partition.eq('development') & assignment.cv_fold.eq(fold), 'pattern_row'].to_numpy()
        assert not (set(train) | set(valid)) & set(data['test'])
        assert not set(train) & set(valid)
        normal = train[y.iloc[train].to_numpy() == 0]
        an, av, columns, pca_info = representation(X.iloc[normal], X.iloc[valid], scenario)
        cache[scenario].append((fold, an, av, valid))
        feature_audit.append({'scenario': scenario, 'fold': fold, 'normal_rows': len(normal),
            'validation_rows': len(valid), 'feature_count': len(columns), 'feature_names': columns, 'pca': pca_info})
(out / 'feature_audit.json').write_text(json.dumps(feature_audit, ensure_ascii=False, indent=2), encoding='utf-8')
print(pd.DataFrame(feature_audit)[['scenario', 'fold', 'normal_rows', 'validation_rows', 'feature_count']].to_string(index=False))
''')
add('markdown','## 4. 시나리오별 파라미터 재탐색\n\n각 시나리오마다 평균 검증 AP가 가장 큰 설정을 선택합니다. 정확한 동률은 후보 순서로 결정합니다. 다른 시나리오의 최적 파라미터를 그대로 사용하지 않습니다.')
add('code','''all_search, all_folds, best_results = [], [], {}
for scenario in SCENARIOS:
    best_ap = -np.inf
    candidate_id = 0
    rows, folds = [], []
    for nu in NU_VALUES:
        for multiplier in GAMMA_MULTIPLIERS:
            oof = pd.Series(np.nan, index=dev, dtype=float)
            ap_values = []
            for fold, an, av, valid in cache[scenario]:
                gamma = multiplier / an.shape[1]
                model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma, tol=1e-4, max_iter=100000).fit(an)
                assert model.fit_status_ == 0, '수렴 실패'
                scores = -model.decision_function(av)
                assert np.isfinite(scores).all()
                oof.loc[valid] = scores
                ap = average_precision_score(y.iloc[valid], scores)
                ap_values.append(ap)
                folds.append({'scenario': scenario, 'candidate_id': candidate_id, 'fold': fold,
                    'AP': ap, 'ROC_AUC': roc_auc_score(y.iloc[valid], scores), 'gamma': gamma})
            assert oof.notna().all()
            mean = float(np.mean(ap_values))
            row = {'scenario': scenario, 'candidate_id': candidate_id, 'nu': nu,
                'gamma_multiplier': multiplier, 'mean_AP': mean, 'std_AP': float(np.std(ap_values, ddof=1))}
            rows.append(row)
            if mean > best_ap:
                best_ap = mean
                best_results[scenario] = {'selection': row.copy(), 'oof': oof.copy()}
            candidate_id += 1
    all_search.extend(rows); all_folds.extend(folds)
    pd.DataFrame(rows).to_csv(out / f'{scenario}_parameter_search.csv', index=False)
    print(scenario, SCENARIOS[scenario], '300設定'.replace('設定','개 설정'), '완료:', best_results[scenario]['selection'], flush=True)
search = pd.DataFrame(all_search)
fold_search = pd.DataFrame(all_folds)
search.to_csv(out / 'all_parameter_search.csv', index=False)
fold_search.to_csv(out / 'all_fold_metrics.csv', index=False)
assert len(search) == 1800 and len(fold_search) == 7200
''')
add('markdown','## 5. 기준선 대비 성능과 위험 패턴별 변화\n\nOOF F2 최대 임계값은 시나리오별로 선택합니다. 동일 OOF의 F1·F2는 선택된 개발 성능입니다. 평균 fold AP와 통합 OOF AP는 다를 수 있습니다. 위험 패턴별 점수·누락을 함께 저장합니다.')
add('code','''comparisons, predictions, paired = [], [], []
baseline_best = best_results['S0']['selection']
base_folds = fold_search.loc[fold_search.scenario.eq('S0') & fold_search.candidate_id.eq(baseline_best['candidate_id'])].set_index('fold')
for scenario, result in best_results.items():
    scores = result['oof'].loc[dev].to_numpy()
    threshold, _ = choose_threshold(y.iloc[dev].to_numpy(), scores)
    selection = result['selection']
    comparisons.append({**selection, 'description': SCENARIOS[scenario], 'threshold': threshold,
        'delta_mean_AP': selection['mean_AP'] - baseline_best['mean_AP'],
        'nu_at_boundary': selection['nu'] in [NU_VALUES[0], NU_VALUES[-1]],
        'gamma_at_boundary': selection['gamma_multiplier'] in [GAMMA_MULTIPLIERS[0], GAMMA_MULTIPLIERS[-1]],
        **metrics(y.iloc[dev], scores, threshold)})
    frame = assignment.iloc[dev][['pattern_row', 'pattern_id', 'cv_fold', 'label']].copy()
    frame['scenario'], frame['risk_score'], frame['threshold'] = scenario, scores, threshold
    frame['y_pred'] = (scores > threshold).astype(int)
    predictions.append(frame)
    for _, row in fold_search.loc[fold_search.scenario.eq(scenario) & fold_search.candidate_id.eq(selection['candidate_id'])].iterrows():
        paired.append({'scenario': scenario, 'fold': int(row.fold), 'AP': row.AP,
            'baseline_AP': base_folds.loc[row.fold, 'AP'], 'delta_AP': row.AP-base_folds.loc[row.fold, 'AP']})
comparison = pd.DataFrame(comparisons).sort_values('mean_AP', ascending=False)
prediction = pd.concat(predictions, ignore_index=True)
comparison.to_csv(out / 'scenario_comparison.csv', index=False)
prediction.to_csv(out / 'oof_predictions.csv', index=False)
prediction.loc[prediction.label.eq(1)].to_csv(out / 'risk_pattern_diagnostics.csv', index=False)
pd.DataFrame(paired).to_csv(out / 'paired_fold_comparison.csv', index=False)
print(comparison[['scenario','description','mean_AP','std_AP','delta_mean_AP','F1','F2','TP','FP','FN']].to_string(index=False))
''')
add('markdown','## 6. 시각화와 재현 기록\n\n오차막대는 네 fold AP의 표준편차입니다. 개발 데이터에서 반복 선택했으므로 신뢰구간이나 독립 test 성능으로 해석하지 않습니다. 이번 결과만으로 최종 모델을 교체하지 않습니다.')
add('code','''fig, ax = plt.subplots(figsize=(11, 5))
ordered = comparison.sort_values('mean_AP')
ax.barh(ordered.scenario + ' ' + ordered.description, ordered.mean_AP, xerr=ordered.std_AP, capsize=3)
ax.set(xlabel='개발 평균 AP ± fold 표준편차', title='CN7 파생변수 시나리오별 재최적화 결과')
fig.tight_layout(); fig.savefig(out/'scenario_AP.png',dpi=140);plt.close(fig)
display(Image(filename=str(out/'scenario_AP.png')))
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, scenario in zip(axes.flat, SCENARIOS):
    part=search.loc[search.scenario.eq(scenario)].pivot(index='nu',columns='gamma_multiplier',values='mean_AP').sort_index()
    im=ax.imshow(part.to_numpy(),aspect='auto',origin='lower',vmin=search.mean_AP.min(),vmax=search.mean_AP.max())
    ax.set_title(scenario + ' ' + SCENARIOS[scenario])
    ticks=[0,7,14]
    ax.set_xticks(ticks,[f'{part.columns[i]:.3g}' for i in ticks]);ax.set_yticks([0,9,19],[f'{part.index[i]:.3f}' for i in [0,9,19]])
    ax.set(xlabel='gamma 배수',ylabel='nu');fig.colorbar(im,ax=ax,label='평균 AP')
fig.tight_layout();fig.savefig(out/'scenario_heatmaps.png',dpi=140);plt.close(fig)
display(Image(filename=str(out/'scenario_heatmaps.png')))
record={'dataset':'cn7','development_only':True,'test_scored':False,'unlabeled_scored':False,
    'scenarios':SCENARIOS,'nu_values':NU_VALUES,'gamma_multipliers':GAMMA_MULTIPLIERS,
    'source_hashes':data['source_hashes'],'split_hash':data['split_hash'],
    'sklearn_version':sklearn.__version__,'all_fits_converged':True,
    'fit_count':7200,'normal_only_preprocessing':True,
    'artifact_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='experiment_manifest.json'}}
(out/'experiment_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
assert all(digest(data['folder']/f)==sha for f,sha in data['source_hashes'].items())
print('파생변수 개발 분석 저장:',out)
''')
(ROOT/'modeling/ocsvm_cn7_feature_scenarios.ipynb').write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5},ensure_ascii=False,indent=1),encoding='utf-8')
