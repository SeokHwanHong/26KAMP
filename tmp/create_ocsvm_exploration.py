from pathlib import Path
import json,uuid
ROOT=Path(__file__).resolve().parents[1]
for name in ['cn7','rg3']:
    base=json.loads((ROOT/f'modeling/ocsvm_{name}_conservative.ipynb').read_text(encoding='utf-8'))
    cells=[]
    def add(kind,s):
        c={'cell_type':kind,'id':uuid.uuid4().hex[:8],'metadata':{},'source':s.strip().splitlines(keepends=True)}
        if kind=='code':c.update(execution_count=None,outputs=[])
        cells.append(c)
    add('markdown',f'''# {name.upper()} OCSVM 개발 시나리오와 변수 중요도

기존 기준 실험과 별도로 실행하는 개발 분석입니다. 고정 test는 점수를 산출하지 않고, 비라벨 추론도 수행하지 않습니다. 기존 평가·모델 파일은 변경하지 않습니다.

1. 정상 기준 표준화와 전체 변수를 사용해 nu 0.005~0.10의 20개 값, gamma 배수 0.03~1의 로그 간격 15개 값을 비교합니다.
2. 전체 변수에서 선택한 동일 nu·gamma 배수로 공정 변수군 제외 및 재표준화 생략 시나리오를 비교합니다. 시나리오별 재최적화 결과가 아니라 입력 처리 변화에 대한 민감도 분석입니다.
3. 전체 변수 선택 모델에서 컬럼별 및 변수군별 permutation importance를 검증 AP 감소량으로 계산합니다. 각 검증 fold에서 20회 반복합니다.

모든 결과는 선택·해석에 사용한 개발 성능입니다. 변수 중요도는 인과관계가 아니며, 중요도에 따른 자동 변수 제거는 하지 않습니다.''')
    setup=''.join(base['cells'][2]['source']).replace("OUTPUT = ROOT / 'output/ocsvm_conservative'", "OUTPUT = ROOT / 'output/ocsvm_development_exploration'")
    setup=setup.replace('NU_VALUES = [0.01, 0.05, 0.10]','NU_VALUES = np.round(np.arange(1, 21) * 0.005, 3).tolist()').replace('GAMMA_MULTIPLIERS = [0.1, 1.0, 10.0]','GAMMA_MULTIPLIERS = np.geomspace(0.03, 1.0, 15).tolist()')
    add('markdown','## 1. 設定'.replace('設定','설정과 데이터 확인'))
    add('code',setup)
    add('code',''.join(base['cells'][4]['source']))
    add('code',''.join(base['cells'][6]['source']))
    add('markdown','## 2. 공정 변수군과 학습 함수\n\n공정 변수군은 해석용 가설입니다. Cycle_Time은 전체 공정 시간이며 단계 전용 변수가 아닙니다. 최대 사출 압력과 전환 압력은 충전·전환 그룹으로 묶습니다. 모든 24개 컬럼을 중복 없이 포함합니다.')
    add('code','''GROUPS = {
    '충전과 전환': ['Injection_Time', 'Filling_Time', 'Cushion_Position', 'Max_Injection_Speed', 'Max_Injection_Pressure', 'Max_Switch_Over_Pressure'],
    '계량과 가소화': ['Plasticizing_Time', 'Plasticizing_Position', 'Max_Screw_RPM', 'Average_Screw_RPM', 'Max_Back_Pressure', 'Average_Back_Pressure'],
    '실린더와 호퍼 온도': ['Barrel_Temperature_' + str(i) for i in range(1, 7)] + ['Hopper_Temperature'],
    '금형 온도': ['Mold_Temperature_3', 'Mold_Temperature_4'],
    '형체와 전체 주기': ['Clamp_Close_Time', 'Clamp_Open_Position', 'Cycle_Time'],
}
name = DATASETS[0]
data = datasets[name]
X, y, assignment, dev = data['X'], data['y'], data['assignment'], data['dev']
assert sorted(sum(GROUPS.values(), [])) == sorted(X.columns)
out = OUTPUT / name
out.mkdir(exist_ok=True)

def cv_run(columns, nu, multiplier, scale=True, keep_models=False):
    oof = pd.Series(np.nan, index=dev, dtype=float)
    records, models = [], []
    for fold in range(4):
        train = assignment.loc[assignment.partition.eq('development') & assignment.cv_fold.ne(fold), 'pattern_row'].to_numpy()
        valid = assignment.loc[assignment.partition.eq('development') & assignment.cv_fold.eq(fold), 'pattern_row'].to_numpy()
        assert not (set(train) | set(valid)) & set(data['test'])
        assert not set(train) & set(valid)
        normal = train[y.iloc[train].to_numpy() == 0]
        variance = VarianceThreshold(0)
        z = variance.fit_transform(X.iloc[normal][columns])
        scaler = StandardScaler() if scale else None
        if scale: z = scaler.fit_transform(z)
        gamma = multiplier / z.shape[1]
        estimator = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma, tol=1e-4, max_iter=100000)
        estimator.fit(z)
        if estimator.fit_status_ != 0:
            raise RuntimeError('후보 수렴 실패: 해당 후보를 점검하세요.')
        steps = [('constant_filter', variance)]
        if scale: steps.append(('scaler', scaler))
        model = Pipeline(steps + [('ocsvm', estimator)])
        scores = risk_score(model, X.iloc[valid][columns])
        oof.loc[valid] = scores
        records.append({'fold': fold, 'AP': average_precision_score(y.iloc[valid], scores),
            'ROC_AUC': roc_auc_score(y.iloc[valid], scores), 'retained_count': z.shape[1],
            'normal_train_rows': len(normal), 'gamma': gamma, 'support_vectors': len(estimator.support_)})
        if keep_models: models.append((fold, model, valid, columns))
    assert oof.notna().all()
    return pd.DataFrame(records), oof, models
''')
    add('markdown','## 3. 300개 설정의 개발 교차검증\n\n평균 AP가 높은 순으로 정렬하며 정확한 동률은 후보 순서를 따릅니다. 범위 끝에 최적값이 있는지 기록하되 자동 확장하지 않습니다.')
    add('code','''search_rows, search_folds = [], []
best_mean, best_oof = -np.inf, None
candidate = 0
for nu in NU_VALUES:
    for multiplier in GAMMA_MULTIPLIERS:
        fold_scores, oof, _ = cv_run(list(X), nu, multiplier)
        mean = float(fold_scores.AP.mean())
        search_rows.append({'candidate_id': candidate, 'nu': nu, 'gamma_multiplier': multiplier,
            'mean_AP': mean, 'std_AP': float(fold_scores.AP.std()), 'mean_ROC_AUC': float(fold_scores.ROC_AUC.mean())})
        search_folds.append(fold_scores.assign(candidate_id=candidate, nu=nu, gamma_multiplier=multiplier))
        if mean > best_mean:
            best_mean, best_oof = mean, oof.copy()
        candidate += 1
    print(f'완료: {candidate}/300개 설정', flush=True)
search = pd.DataFrame(search_rows).sort_values(['mean_AP', 'candidate_id'], ascending=[False, True], kind='stable')
search.to_csv(out / 'parameter_search.csv', index=False)
pd.concat(search_folds, ignore_index=True).to_csv(out / 'parameter_fold_metrics.csv', index=False)
best = search.iloc[0]
threshold, threshold_table = choose_threshold(y.iloc[dev].to_numpy(), best_oof.loc[dev].to_numpy())
threshold_table.to_csv(out / 'baseline_threshold_search.csv', index=False)
print('상위 10개 설정')
print(search.head(10).to_string(index=False))
print('범위 경계 여부:', 'nu', best.nu in [NU_VALUES[0], NU_VALUES[-1]],
      'gamma 배수', best.gamma_multiplier in [GAMMA_MULTIPLIERS[0], GAMMA_MULTIPLIERS[-1]])
''')
    add('markdown','## 4. 입력 처리 시나리오 비교\n\n전체 변수에서 고른 nu와 gamma 배수를 고정합니다. 변수군 제외 시 gamma는 남은 변수 수 d에 따라 재계산되므로 순수한 단일 변수 효과와 구분하세요. 각 시나리오의 OOF F2 임계값도 개발 데이터에서 선택합니다. 모델 선택에 test는 사용하지 않습니다.')
    add('code','''scenarios = [('전체 변수와 정상 재표준화', list(X), True), ('전체 변수와 제공 스케일 유지', list(X), False)]
scenarios += [(group + ' 제외', [c for c in X if c not in columns], True) for group, columns in GROUPS.items()]
scenario_rows, scenario_folds, scenario_predictions = [], [], []
baseline_models = None
for scenario, columns, scale in scenarios:
    folds, oof, models = cv_run(columns, float(best.nu), float(best.gamma_multiplier), scale, keep_models=True)
    selected_threshold, _ = choose_threshold(y.iloc[dev].to_numpy(), oof.loc[dev].to_numpy())
    scenario_rows.append({'scenario': scenario, 'input_count': len(columns), 'standardize': scale,
        'mean_fold_AP': folds.AP.mean(), 'std_fold_AP': folds.AP.std(), 'threshold': selected_threshold,
        **metrics(y.iloc[dev], oof.loc[dev].to_numpy(), selected_threshold)})
    scenario_folds.append(folds.assign(scenario=scenario))
    pred = assignment.iloc[dev][['pattern_row', 'pattern_id', 'cv_fold', 'label']].copy()
    pred['scenario'], pred['risk_score'], pred['threshold'] = scenario, oof.loc[dev].to_numpy(), selected_threshold
    pred['y_pred'] = (pred.risk_score > selected_threshold).astype(int)
    scenario_predictions.append(pred)
    if scenario == scenarios[0][0]: baseline_models = models
scenario_table = pd.DataFrame(scenario_rows)
scenario_table.to_csv(out / 'scenario_comparison.csv', index=False)
pd.concat(scenario_folds, ignore_index=True).to_csv(out / 'scenario_fold_metrics.csv', index=False)
pd.concat(scenario_predictions, ignore_index=True).to_csv(out / 'scenario_oof_predictions.csv', index=False)
print(scenario_table[['scenario', 'mean_fold_AP', 'std_fold_AP', 'F1', 'F2', 'TP', 'FP', 'FN']].to_string(index=False))
''')
    add('markdown','## 5. 컬럼과 변수군 중요도\n\n선택된 전체 변수 모델을 고정하고 validation의 컬럼을 섞어 AP 감소량을 계산합니다. 값이 클수록 해당 모델이 의존한 정보입니다. 음수는 섞은 후 점수가 개선된 경우이며 오류가 아닙니다. 변수군은 같은 행 순열로 함께 섞어 그룹 내부 관계를 보존합니다. 상관 변수의 대체 효과와 정상 학습에서 제거된 상수 컬럼을 구분합니다. 반복 표준편차는 소표본 불확실성의 신뢰구간이 아닙니다.')
    add('code','''REPEATS = 20
importance_rows = []
items = [('column', c, [c]) for c in X] + [('group', g, columns) for g, columns in GROUPS.items()]
for fold, model, valid, columns in baseline_models:
    validation = X.iloc[valid][columns].copy()
    labels = y.iloc[valid].to_numpy()
    base_ap = average_precision_score(labels, risk_score(model, validation))
    retained = set(np.array(columns)[model.named_steps['constant_filter'].get_support()])
    for kind, feature, selected in items:
        for repeat in range(REPEATS):
            rng = np.random.default_rng(42000 + fold * 100 + repeat)
            permutation = rng.permutation(len(validation))
            shuffled = validation.copy()
            shuffled.loc[:, selected] = validation.iloc[permutation][selected].to_numpy()
            permuted_ap = average_precision_score(labels, risk_score(model, shuffled))
            importance_rows.append({'kind': kind, 'feature': feature, 'fold': fold, 'repeat': repeat,
                'baseline_AP': base_ap, 'permuted_AP': permuted_ap, 'AP_decrease': base_ap-permuted_ap,
                'active_in_model': any(c in retained for c in selected)})
importance = pd.DataFrame(importance_rows)
fold_importance = importance.groupby(['kind', 'feature', 'fold'], as_index=False).agg(
    mean_AP_decrease=('AP_decrease', 'mean'), repeat_std=('AP_decrease', 'std'), active_in_model=('active_in_model', 'max'))
importance_summary = fold_importance.groupby(['kind', 'feature'], as_index=False).agg(
    mean_AP_decrease=('mean_AP_decrease', 'mean'), fold_std=('mean_AP_decrease', 'std'),
    positive_folds=('mean_AP_decrease', lambda s: int((s > 0).sum())), active_folds=('active_in_model', 'sum'))
importance_summary = importance_summary.sort_values(['kind', 'mean_AP_decrease'], ascending=[True, False])
importance.to_csv(out / 'permutation_importance_repeats.csv', index=False)
fold_importance.to_csv(out / 'permutation_importance_folds.csv', index=False)
importance_summary.to_csv(out / 'permutation_importance_summary.csv', index=False)
assert len(importance) == (len(X.columns) + len(GROUPS)) * 4 * REPEATS
assert np.allclose(importance.loc[~importance.active_in_model, 'AP_decrease'], 0)
print('上位'.replace('上位','상위'), '컬럼 중요도')
print(importance_summary.loc[importance_summary.kind.eq('column')].head(10).to_string(index=False))
print('변수군 중요도')
print(importance_summary.loc[importance_summary.kind.eq('group')].to_string(index=False))
''')
    add('markdown','## 6. 탐색 영역과 중요도 시각화\n\n히트맵의 최고점과 주변 영역을 함께 확인합니다. 중요도 오차막대는 네 fold 평균 중요도의 표준편차입니다. 하나의 fold만으로 변수 제거를 결정하지 않습니다.')
    add('code','''matrix = search.pivot(index='nu', columns='gamma_multiplier', values='mean_AP').sort_index()
fig, ax = plt.subplots(figsize=(12, 7))
img = ax.imshow(matrix.to_numpy(), aspect='auto', origin='lower', cmap='viridis')
ax.set_xticks(range(len(matrix.columns)), [f'{v:.3g}' for v in matrix.columns], rotation=45)
ax.set_yticks(range(len(matrix.index)), [f'{v:.3f}' for v in matrix.index])
ax.set(xlabel='gamma 배수 / d', ylabel='nu', title=name.upper() + ' 개발 4-fold 평균 AP')
fig.colorbar(img, ax=ax, label='평균 AP'); fig.tight_layout()
fig.savefig(out / 'parameter_heatmap.png', dpi=140); plt.close(fig)
display(Image(filename=str(out / 'parameter_heatmap.png')))
for kind, title in [('column', '컬럼별'), ('group', '공정 변수군별')]:
    values = importance_summary.loc[importance_summary.kind.eq(kind)].sort_values('mean_AP_decrease')
    fig, ax = plt.subplots(figsize=(11, 8 if kind == 'column' else 4))
    ax.barh(values.feature, values.mean_AP_decrease, xerr=values.fold_std.fillna(0), color='steelblue', capsize=3)
    ax.axvline(0, color='black', lw=.8)
    ax.set(xlabel='검증 AP 감소량 평균 ± fold 표준편차', title=name.upper() + ' ' + title + ' 중요도')
    fig.tight_layout(); fig.savefig(out / f'{kind}_importance.png', dpi=140); plt.close(fig)
    display(Image(filename=str(out / f'{kind}_importance.png')))
record = {'dataset': name, 'development_only': True, 'test_scored': False,
    'nu_values': NU_VALUES, 'gamma_multipliers': GAMMA_MULTIPLIERS,
    'selected_nu': float(best.nu), 'selected_gamma_multiplier': float(best.gamma_multiplier),
    'selected_mean_AP': float(best.mean_AP), 'scenario_parameters': 'fixed full-feature selected nu and gamma multiplier',
    'importance_repeats': REPEATS, 'groups': GROUPS,
    'source_hashes': data['source_hashes'], 'split_hash': data['split_hash'],
    'sklearn_version': sklearn.__version__, 'all_fits_converged': True,
    'artifact_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'experiment_manifest.json'}}
(out / 'experiment_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
assert all(digest(data['folder'] / f) == sha for f, sha in data['source_hashes'].items())
print('개발 분석 저장 완료:', out)
''')
    path=ROOT/f'modeling/ocsvm_{name}_exploration.ipynb'
    path.write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5},ensure_ascii=False,indent=1),encoding='utf-8')
