"""CN7 domain hypotheses and development recall selection under FPR constraints.

No test observations are accepted by search_development or select_candidates.
AP is diagnostic only. Thresholds belong to a specific fitted model.
"""
from pathlib import Path
import hashlib
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

VERSION = 'domain_fpr_v1'
SEED = 42
CALIBRATION_FRACTION = 0.35
BUDGETS = (0.01, 0.03, 0.05)
NU_VALUES = sorted(set([0.001, 0.0025] + np.round(np.arange(1, 21)*0.005, 3).tolist()
                       + [0.125, 0.15, 0.20, 0.25, 0.30, 0.40]))
GAMMA_MULTIPLIERS = sorted(set(np.geomspace(0.03, 1., 15).tolist()
                              + [0.003, 0.005, 0.01, 0.02, 2., 3., 10.]))
GROUPS = {
    '충전과 전환': ['Injection_Time', 'Filling_Time', 'Cushion_Position',
                'Max_Injection_Speed', 'Max_Injection_Pressure', 'Max_Switch_Over_Pressure'],
    '계량과 가소화': ['Plasticizing_Time', 'Plasticizing_Position', 'Max_Screw_RPM',
                 'Average_Screw_RPM', 'Max_Back_Pressure', 'Average_Back_Pressure'],
    '실린더와 호퍼 온도': [f'Barrel_Temperature_{i}' for i in range(1, 7)] + ['Hopper_Temperature'],
    '금형 온도': ['Mold_Temperature_3', 'Mold_Temperature_4'],
    '형체와 전체 주기': ['Clamp_Close_Time', 'Clamp_Open_Position', 'Cycle_Time'],
}
SCENARIOS = {
    'raw': (),
    'legacy_without_cycle': (),
    'pressure': ('pressure',),
    'plasticizing': ('plasticizing',),
    'thermal': ('thermal',),
    'domain_all': ('pressure', 'plasticizing', 'thermal'),
    'group_pca': (),
}
DOMAIN_SPEC = [
    dict(feature='사출_전환압력_상대편차', group='pressure', operation='difference',
         columns=['Max_Injection_Pressure', 'Max_Switch_Over_Pressure'],
         hypothesis='최대 사출압력과 전환압력이 정상 대비 함께 변하는지 확인',
         limitation='서로 다른 시점의 요약값이며 실제 압력 강하가 아님', source='ARBURG 64/2017 p27'),
    dict(feature='스크루RPM_상대편차', group='plasticizing', operation='difference',
         columns=['Max_Screw_RPM', 'Average_Screw_RPM'],
         hypothesis='가소화 회전수의 최대·평균 상대 변화 불일치',
         limitation='실제 RPM 폭이나 시간적 변동성으로 해석할 수 없음', source='Sensors 2022 22(13):4792; 제공 변수명'),
    dict(feature='배압_상대편차', group='plasticizing', operation='difference',
         columns=['Max_Back_Pressure', 'Average_Back_Pressure'],
         hypothesis='가소화 배압의 최대·평균 상대 변화 불일치',
         limitation='실제 배압 차이 또는 불량 인과 효과가 아님', source='Sensors 2022 22(13):4792; 제공 변수명'),
    dict(feature='배럴_상대수준평균', group='thermal', operation='mean',
         columns=[f'Barrel_Temperature_{i}' for i in range(1, 7)],
         hypothesis='활성 배럴 센서의 정상 대비 공통 온도 이동',
         limitation='평균 섭씨 온도 아님; 센서 위치·목표 온도는 미확인', source='제공 변수명; 공정 가설'),
    dict(feature='배럴_상대편차표준편차', group='thermal', operation='std',
         columns=[f'Barrel_Temperature_{i}' for i in range(1, 7)],
         hypothesis='활성 배럴 센서의 정상 대비 변화가 서로 다른 정도',
         limitation='물리적 온도 균일도·공간 구배 아님', source='제공 변수명; 공정 가설'),
    dict(feature='금형_상대수준평균', group='thermal', operation='mean',
         columns=['Mold_Temperature_3', 'Mold_Temperature_4'],
         hypothesis='금형 센서의 정상 대비 공통 온도 이동',
         limitation='센서가 상·하형인지 미확인; 평균 섭씨 온도 아님', source='제공 변수명; 공정 가설'),
    dict(feature='금형온도_상대편차', group='thermal', operation='difference',
         columns=['Mold_Temperature_3', 'Mold_Temperature_4'],
         hypothesis='금형 두 센서의 정상 대비 상대 변화 불일치',
         limitation='실제 온도차·냉각 효율 아님', source='제공 변수명; 공정 가설'),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, record):
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_data(root):
    folder = Path(root)/'data/processed/cn7/conservative'
    split_folder = folder/'splits'
    manifest = json.loads((split_folder/'split_manifest.json').read_text(encoding='utf-8'))
    for filename, expected in manifest['source_sha256'].items():
        assert digest(folder/filename) == expected
    for filename, expected in manifest['artifact_sha256'].items():
        assert digest(split_folder/filename) == expected
    x = pd.read_csv(folder/'X_labeled.csv', float_precision='round_trip')
    y = pd.read_csv(folder/'y_labeled.csv').PassOrFail
    assignment = pd.read_csv(split_folder/'split_assignments.csv')
    assert sorted(x.columns) == sorted(sum(GROUPS.values(), []))
    assert np.isfinite(x.to_numpy()).all() and not x.duplicated().any()
    assert np.array_equal(assignment.pattern_row, np.arange(len(x)))
    assert np.array_equal(assignment.label, y) and assignment.pattern_id.is_unique
    assert set(y) == {0, 1}
    dev = assignment.loc[assignment.partition.eq('development'), 'pattern_row'].to_numpy()
    test = assignment.loc[assignment.partition.eq('test'), 'pattern_row'].to_numpy()
    assert not set(dev) & set(test) and set(dev) | set(test) == set(range(len(x)))
    assert set(assignment.loc[dev, 'cv_fold']) == {0, 1, 2, 3}
    assert assignment.loc[test, 'cv_fold'].eq(-1).all()
    return x, y, assignment, dev, test, manifest


def split_normal(rows, seed):
    rows = np.asarray(rows)
    shuffled = np.random.default_rng(seed).permutation(rows)
    count = int(np.ceil(len(rows)*CALIBRATION_FRACTION))
    if count < 1 or count >= len(rows):
        raise ValueError('정상 학습/보정 자료가 부족합니다.')
    return np.sort(shuffled[count:]), np.sort(shuffled[:count])


def derived_frame(z, specs):
    result = pd.DataFrame(index=z.index)
    for spec in specs:
        values = z[spec['active_columns']]
        operation = spec['operation']
        if operation == 'difference':
            result[spec['feature']] = values.iloc[:, 0] - values.iloc[:, 1]
        elif operation == 'mean':
            result[spec['feature']] = values.mean(axis=1)
        elif operation == 'std':
            result[spec['feature']] = values.std(axis=1, ddof=0)
        else:
            raise ValueError(operation)
    return result


def unscaled_frame(bundle, inputs):
    inputs = inputs[bundle['input_columns']]
    z = pd.DataFrame(bundle['normal_scaler'].transform(inputs), index=inputs.index, columns=inputs.columns)
    if bundle['scenario'] == 'group_pca':
        frame = pd.DataFrame(index=inputs.index)
        for group, item in bundle['pcas'].items():
            values = z[item['columns']].to_numpy()
            scores = item['pca'].transform(values)
            for j in range(scores.shape[1]):
                frame[f'{group}_PC{j+1}'] = scores[:, j]
            if len(item['columns']) > scores.shape[1]:
                frame[f'{group}_재구성RMSE'] = np.sqrt(np.mean((values-item['pca'].inverse_transform(scores))**2, axis=1))
        return frame
    return pd.concat([z[bundle['base_columns']], derived_frame(z, bundle['specs'])], axis=1)


def fit_features(normal, scenario):
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    active = [c for c in normal if normal[c].nunique() > 1]
    scaler = StandardScaler().fit(normal)
    bundle = dict(scenario=scenario, input_columns=list(normal), normal_scaler=scaler,
                  specs=[], skipped_specs=[], pcas={})
    for spec in DOMAIN_SPEC:
        if spec['group'] not in SCENARIOS[scenario]:
            continue
        cols = [c for c in spec['columns'] if c in active]
        # Pairs need both sensors; aggregates need >=2 to avoid raw-feature duplication.
        if len(cols) < 2:
            bundle['skipped_specs'].append({'feature': spec['feature'], 'reason': '활성 센서 2개 미만'})
        else:
            bundle['specs'].append({**spec, 'active_columns': cols})
    bundle['base_columns'] = [c for c in active if scenario != 'legacy_without_cycle'
                              or c not in GROUPS['형체와 전체 주기']]
    if scenario == 'group_pca':
        z = pd.DataFrame(scaler.transform(normal), columns=normal.columns, index=normal.index)
        for group, cols in GROUPS.items():
            cols = [c for c in cols if c in active]
            if cols:
                bundle['pcas'][group] = {'columns': cols, 'pca': PCA(n_components=min(2, len(cols)),
                                                  svd_solver='full').fit(z[cols].to_numpy())}
    frame = unscaled_frame(bundle, normal)
    bundle['variance'] = VarianceThreshold(0).fit(frame)
    values = bundle['variance'].transform(frame)
    bundle['feature_scaler'] = StandardScaler().fit(values)
    bundle['feature_names'] = list(frame.columns[bundle['variance'].get_support()])
    return bundle, transform_features(bundle, normal)


def transform_features(bundle, inputs):
    frame = unscaled_frame(bundle, inputs)
    values = bundle['feature_scaler'].transform(bundle['variance'].transform(frame))
    assert np.isfinite(values).all()
    return values


def calibration_threshold(normal_scores, budget):
    """Conservative order statistic; strict > handles ties without splitting them.

    This is a finite-sample calibration rule, not a guarantee after model selection
    or under distribution shift. No labels from validation/test are used.
    """
    scores = np.asarray(normal_scores, dtype=float)
    if not len(scores) or not np.isfinite(scores).all() or not 0 < budget < 1:
        raise ValueError('정상 점수와 0~1 사이 오탐 예산이 필요합니다.')
    rank = math.ceil((len(scores)+1)*(1-budget))
    return float(np.sort(scores)[rank-1]) if rank <= len(scores) else float('inf')


def classification_metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    tp = int(((truth == 1) & (prediction == 1)).sum())
    fn = int(((truth == 1) & (prediction == 0)).sum())
    fp = int(((truth == 0) & (prediction == 1)).sum())
    tn = int(((truth == 0) & (prediction == 0)).sum())
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, recall=tp/(tp+fn) if tp+fn else 0.,
                FPR=fp/(fp+tn) if fp+tn else 0., precision=tp/(tp+fp) if tp+fp else 0.,
                F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                F2=5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.)


def summarize_folds(rows):
    counts = {key: int(sum(row[key] for row in rows)) for key in ('TP', 'FP', 'FN', 'TN')}
    tp, fp, fn, tn = (counts[key] for key in ('TP', 'FP', 'FN', 'TN'))
    return {**counts, 'recall': tp/(tp+fn), 'FPR': fp/(fp+tn),
            'precision': tp/(tp+fp) if tp+fp else 0.,
            'recall_std': float(np.std([r['recall'] for r in rows], ddof=1)),
            'worst_fold_recall': min(r['recall'] for r in rows),
            'max_fold_FPR': max(r['FPR'] for r in rows),
            'mean_AP_reference': float(np.mean([r['AP_reference'] for r in rows])),
            'mean_ROC_AUC_reference': float(np.mean([r['ROC_AUC_reference'] for r in rows]))}


def make_fold_plan(xdev, ydev, fold_ids):
    assert xdev.index.equals(ydev.index) and xdev.index.equals(fold_ids.index)
    plans = []
    for fold in sorted(fold_ids.unique()):
        valid = xdev.index[fold_ids.eq(fold)].to_numpy()
        normal = xdev.index[fold_ids.ne(fold) & ydev.eq(0)].to_numpy()
        fit, cal = split_normal(normal, SEED+int(fold))
        assert not (set(fit) & set(cal) or set(fit) & set(valid) or set(cal) & set(valid))
        assert set(fit) | set(cal) == set(normal)
        assert ydev.loc[fit].eq(0).all() and ydev.loc[cal].eq(0).all()
        assert set(ydev.loc[valid]) == {0, 1}
        plans.append(dict(fold=int(fold), fit=fit, cal=cal, valid=valid))
    assert sorted(np.concatenate([p['valid'] for p in plans])) == sorted(xdev.index)
    return plans


def fit_model(values, nu, multiplier):
    model = OneClassSVM(kernel='rbf', nu=float(nu), gamma=float(multiplier)/values.shape[1],
                       tol=1e-4, max_iter=100000).fit(values)
    assert model.fit_status_ == 0, 'OCSVM 수렴 실패'
    return model


def select_candidates(table, budgets=BUDGETS):
    selected = []
    for budget in budgets:
        allowed = table.loc[table.budget.eq(budget) & table.FPR.le(budget+1e-12)]
        # Pooled recall, then fewer false alarms, lower fold variability, smaller model.
        ranked = allowed.sort_values(['recall', 'FP', 'recall_std', 'feature_count', 'candidate_id'],
                                     ascending=[False, True, True, True, True], kind='stable')
        if ranked.empty or ranked.iloc[0].TP == 0:
            selected.append(dict(budget=budget, status='no_supported_model',
                                 reason='오탐 제약을 만족하며 개발 불량을 탐지한 후보 없음'))
        else:
            selected.append({**ranked.iloc[0].to_dict(), 'status': 'selected_for_followup'})
    return selected


def search_development(xdev, ydev, fold_ids, out, nu_values=NU_VALUES,
                       multipliers=GAMMA_MULTIPLIERS, scenarios=None, budgets=BUDGETS):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plans = make_fold_plan(xdev, ydev, fold_ids)
    write_json(out/'fold_plan.json', [{k: v.tolist() if isinstance(v, np.ndarray) else v
                                     for k, v in plan.items()} for plan in plans])
    rows, audit, fold_rows = [], [], []
    candidate_id = 0
    for scenario in (list(SCENARIOS) if scenarios is None else scenarios):
        cache = []
        for plan in plans:
            bundle, train = fit_features(xdev.loc[plan['fit']], scenario)
            cal = transform_features(bundle, xdev.loc[plan['cal']])
            valid = transform_features(bundle, xdev.loc[plan['valid']])
            cache.append((plan, train, cal, valid))
            audit.append(dict(scenario=scenario, fold=plan['fold'], fit_count=len(train),
                              calibration_count=len(cal), validation_count=len(valid),
                              feature_names=bundle['feature_names'], domain_specs=bundle['specs'],
                              skipped_specs=bundle['skipped_specs']))
        for nu in nu_values:
            for multiplier in multipliers:
                per_budget = {budget: [] for budget in budgets}
                for plan, train, cal, valid in cache:
                    model = fit_model(train, nu, multiplier)
                    cal_scores, scores = -model.decision_function(cal), -model.decision_function(valid)
                    assert np.isfinite(scores).all() and np.isfinite(cal_scores).all()
                    truth = ydev.loc[plan['valid']].to_numpy()
                    refs = dict(AP_reference=float(average_precision_score(truth, scores)),
                                ROC_AUC_reference=float(roc_auc_score(truth, scores)))
                    for budget in budgets:
                        threshold = calibration_threshold(cal_scores, budget)
                        row = dict(candidate_id=candidate_id, scenario=scenario, nu=float(nu),
                                   gamma_multiplier=float(multiplier), budget=budget, fold=plan['fold'],
                                   threshold=threshold, **classification_metrics(truth, scores > threshold), **refs)
                        per_budget[budget].append(row)
                        fold_rows.append(row)
                for budget, values in per_budget.items():
                    rows.append(dict(candidate_id=candidate_id, scenario=scenario, nu=float(nu),
                                     gamma_multiplier=float(multiplier), budget=budget,
                                     feature_count=max(c[1].shape[1] for c in cache), **summarize_folds(values)))
                candidate_id += 1
        print(f'{scenario}: {len(nu_values)*len(multipliers)}개 설정 × {len(plans)} folds 완료', flush=True)
        pd.DataFrame(rows).to_csv(out/'candidate_search.csv', index=False)
    table = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out/'candidate_fold_metrics.csv', index=False)
    write_json(out/'feature_audit.json', audit)
    write_json(out/'domain_feature_definitions.json', DOMAIN_SPEC)
    selected = select_candidates(table, budgets)
    write_json(out/'selection_manifest.json', dict(version=VERSION, budgets=list(budgets),
        selection='development pooled recall max subject to pooled FPR <= budget; ties FP, recall_std, feature_count, candidate_id',
        threshold='normal calibration ceil((n+1)*(1-budget)) order statistic; risk > threshold',
        calibration_fraction=CALIBRATION_FRACTION, seed=SEED, choices=selected,
        test_used_for_selection=False, independent_validation=False,
        budget_is_operating_requirement=False, AP_used_for_selection=False,
        nu_values=list(nu_values), gamma_multipliers=list(multipliers)))
    return table, folds, selected, plans


def development_diagnostics(xdev, ydev, plans, selected, out, repeats=20):
    """Group permutation diagnostic. Recompute derived features from raw inputs.

    Fixed model + calibration threshold; report recall loss AND FPR change.
    This diagnostic never generates new candidates or changes selection.
    """
    predictions, importance = [], []
    selection_sha = digest(Path(out)/'selection_manifest.json')
    for choice in selected:
        if choice['status'] != 'selected_for_followup':
            continue
        for plan in plans:
            bundle, train = fit_features(xdev.loc[plan['fit']], choice['scenario'])
            model = fit_model(train, choice['nu'], choice['gamma_multiplier'])
            cal_scores = -model.decision_function(transform_features(bundle, xdev.loc[plan['cal']]))
            threshold = calibration_threshold(cal_scores, choice['budget'])
            inputs = xdev.loc[plan['valid']]
            truth = ydev.loc[plan['valid']].to_numpy()
            scores = -model.decision_function(transform_features(bundle, inputs))
            base = classification_metrics(truth, scores > threshold)
            for index, label, score in zip(inputs.index, truth, scores):
                predictions.append(dict(budget=choice['budget'], scenario=choice['scenario'],
                    fold=plan['fold'], pattern_row=int(index), y_true=int(label), risk_score=float(score),
                    threshold=threshold, y_pred=int(score > threshold)))
            for group, cols in GROUPS.items():
                rng = np.random.default_rng(SEED+plan['fold'])
                for repeat in range(repeats):
                    perm = inputs.copy()
                    perm.loc[:, cols] = inputs[cols].to_numpy()[rng.permutation(len(inputs))]
                    changed_scores = -model.decision_function(transform_features(bundle, perm))
                    changed = classification_metrics(truth, changed_scores > threshold)
                    importance.append(dict(budget=choice['budget'], scenario=choice['scenario'],
                        fold=plan['fold'], group=group, repeat=repeat,
                        recall_drop=base['recall']-changed['recall'], FPR_increase=changed['FPR']-base['FPR']))
    pred = pd.DataFrame(predictions, columns=['budget','scenario','fold','pattern_row','y_true','risk_score','threshold','y_pred'])
    imp = pd.DataFrame(importance, columns=['budget','scenario','fold','group','repeat','recall_drop','FPR_increase'])
    pred.to_csv(Path(out)/'selected_oof_predictions.csv', index=False)
    imp.to_csv(Path(out)/'group_importance_repeats.csv', index=False)
    assert digest(Path(out)/'selection_manifest.json') == selection_sha
    return pred, imp


def predict_artifact(artifact, inputs):
    scores = -artifact['model'].decision_function(transform_features(artifact['features'], inputs))
    return scores, (scores > artifact['threshold']).astype(int)


def evaluate_fixed_test(xdev, ydev, xtest, ytest, selected, out):
    out = Path(out)
    selection_sha = digest(out/'selection_manifest.json')
    normal = xdev.index[ydev.eq(0)].to_numpy()
    fit, cal = split_normal(normal, SEED+1000)
    assert not (set(xdev.index) & set(xtest.index))
    write_json(out/'final_fit_plan.json', dict(fit=fit.tolist(), cal=cal.tolist(), test=xtest.index.tolist(),
        reason='보정 후 재학습하지 않음: 모델과 임계값의 점수 척도를 유지'))
    rows, predictions = [], []
    for choice in selected:
        budget = choice['budget']
        tag = f'fpr_{int(round(budget*100)):02d}'
        if choice['status'] != 'selected_for_followup':
            # Explicit no-alert reference; not a validated/deployed detector.
            prediction = np.zeros(len(xtest), dtype=int)
            scores = np.full(len(xtest), np.nan)
            threshold = None
            scenario = 'no_alert_reference'
            reference = dict(AP_reference=None, ROC_AUC_reference=None)
        else:
            scenario = choice['scenario']
            bundle, values = fit_features(xdev.loc[fit], scenario)
            model = fit_model(values, choice['nu'], choice['gamma_multiplier'])
            threshold = calibration_threshold(-model.decision_function(transform_features(bundle, xdev.loc[cal])), budget)
            artifact = dict(version=VERSION, features=bundle, model=model, threshold=threshold,
                            budget=budget, selection_sha256=selection_sha, choice=choice)
            scores, prediction = predict_artifact(artifact, xtest)
            joblib.dump(artifact, out/f'{tag}_model.joblib')
            reloaded_scores, reloaded_pred = predict_artifact(joblib.load(out/f'{tag}_model.joblib'), xtest)
            assert np.array_equal(scores, reloaded_scores) and np.array_equal(prediction, reloaded_pred)
            reference = dict(AP_reference=float(average_precision_score(ytest, scores)),
                             ROC_AUC_reference=float(roc_auc_score(ytest, scores)))
        result = classification_metrics(ytest, prediction)
        rows.append(dict(budget=budget, status=choice['status'], scenario=scenario,
                         threshold=threshold, **result, **reference,
                         observed_FPR_exceeds_budget=result['FPR'] > budget+1e-12))
        for index, label, score, pred in zip(xtest.index, ytest, scores, prediction):
            predictions.append(dict(budget=budget, scenario=scenario, pattern_row=int(index),
                y_true=int(label), risk_score=score, threshold=threshold, y_pred=int(pred)))
    table = pd.DataFrame(rows)
    table.to_csv(out/'test_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(out/'test_predictions.csv', index=False)
    assert digest(out/'selection_manifest.json') == selection_sha
    return table
