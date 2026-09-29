from pathlib import Path
import json, uuid
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def add(kind, text):
    c={'cell_type':kind,'id':uuid.uuid4().hex[:8],'metadata':{},'source':text.strip().splitlines(keepends=True)}
    if kind=='code': c.update(execution_count=None,outputs=[])
    cells.append(c)
add('markdown','''# OCSVM 보수적 공정 위험 탐지

CN7·RG3를 별도로 학습합니다. 고정 test 20%는 모델 선택에서 제외하고, 개발 80%의 저장된 4-fold를 그대로 사용합니다. 각 학습 구간에서 **정상 패턴만으로 상수 제거 → 표준화 → RBF One-Class SVM**을 적합합니다.

9개 설정의 평균 검증 AP로 설정을 선택하고, 선택 설정의 개발 OOF 점수에서 F2 임계값을 정합니다. 이후 개발 정상 전체로 재적합하여 고정 test를 평가합니다. OOF 지표는 설정·임계값 선택에 사용된 개발 성능입니다. Test 결과를 보고 재선택하지 않습니다.

위험 점수는 `-decision_function`이며 확률이 아닙니다. 양성 1은 불량 이력이 있는 공정 조건입니다. 비라벨 좌표계 정합은 미확인 상태이므로 이번 노트북은 비라벨 추론을 수행하지 않습니다.''')
add('markdown','## 1. 설정과 라이브러리\n\n후보는 nu 0.01·0.05·0.10과 gamma=(0.1·1·10)/d의 9개 조합입니다. d는 정상 학습에서 상수 제거 후 남은 변수 수입니다. nu는 실제 불량률을 의미하지 않습니다.')
add('code','''from pathlib import Path
import os, json, hashlib, platform
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'data/origin').is_dir()), None)
if ROOT is None:
    raise RuntimeError('프로젝트 내부에서 실행하세요.')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'tmp/matplotlib_cache')
import numpy as np
import pandas as pd
import sklearn, joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Image
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
    roc_curve, confusion_matrix, precision_score, recall_score, f1_score, fbeta_score)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
DATASETS = ['cn7', 'rg3']
NU_VALUES = [0.01, 0.05, 0.10]
GAMMA_MULTIPLIERS = [0.1, 1.0, 10.0]
OUTPUT = ROOT / 'output/ocsvm_conservative'
OUTPUT.mkdir(parents=True, exist_ok=True)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
''')
add('markdown','## 2. 저장된 분할과 입력 검증\n\npattern_row는 전체 X_labeled의 행 위치입니다. 전처리 및 분할 해시와 라벨 정렬을 확인합니다. 입력은 24개 공정 컬럼이며 ID·fold·관측 횟수를 포함하지 않습니다.')
add('code','''datasets = {}
for name in DATASETS:
    folder = ROOT / 'data/processed' / name / 'conservative'
    splits = folder / 'splits'
    source = json.loads((folder / 'preprocessing_manifest.json').read_text(encoding='utf-8'))
    split_record = json.loads((splits / 'split_manifest.json').read_text(encoding='utf-8'))
    for f, sha in split_record['source_sha256'].items():
        assert digest(folder / f) == sha, '분할 시점과 전처리 데이터가 다릅니다.'
    for f, sha in split_record['artifact_sha256'].items():
        assert digest(splits / f) == sha, '저장된 분할 자료가 변경되었습니다.'
    X = pd.read_csv(folder / 'X_labeled.csv', float_precision='round_trip')
    y = pd.read_csv(folder / 'y_labeled.csv')['PassOrFail']
    assignment = pd.read_csv(splits / 'split_assignments.csv')
    assert list(X) == source['feature_columns'] and X.shape[1] == 24
    assert np.isfinite(X.to_numpy()).all() and not X.duplicated().any()
    assert np.array_equal(assignment.pattern_row, np.arange(len(X)))
    assert np.array_equal(assignment.label, y) and assignment.pattern_id.is_unique
    dev = assignment.loc[assignment.partition.eq('development'), 'pattern_row'].to_numpy()
    test = assignment.loc[assignment.partition.eq('test'), 'pattern_row'].to_numpy()
    assert not set(dev) & set(test)
    assert set(dev) | set(test) == set(range(len(X)))
    assert set(assignment.loc[dev, 'cv_fold']) == {0, 1, 2, 3}
    assert (assignment.loc[test, 'cv_fold'] == -1).all()
    datasets[name] = dict(X=X, y=y, assignment=assignment, dev=dev, test=test,
        folder=folder, source_hashes=split_record['source_sha256'],
        split_hash=digest(splits / 'split_assignments.csv'))
    print(name.upper(), '개발 패턴:', len(dev), '고정 test 패턴:', len(test))
''')
add('markdown','## 3. 정상 전용 학습과 평가 함수\n\n스케일러와 상수 제거도 정상 학습에만 적합합니다. 클래스 가중치는 사용하지 않습니다. 위험 점수가 임계값보다 큰 경우 1로 판정합니다. 수렴 실패는 묵인하지 않고 중단합니다.')
add('code','''def fit_normal(X_normal, nu, multiplier):
    variance = VarianceThreshold(0)
    kept = variance.fit_transform(X_normal)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(kept)
    gamma = multiplier / scaled.shape[1]
    estimator = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma, tol=1e-4, max_iter=100000)
    estimator.fit(scaled)
    assert estimator.fit_status_ == 0, 'OCSVM 수렴 실패'
    pipeline = Pipeline([('constant_filter', variance), ('scaler', scaler), ('ocsvm', estimator)])
    info = {'nu': nu, 'gamma_multiplier': multiplier, 'gamma': gamma,
        'normal_train_rows': len(X_normal), 'retained_features': list(X_normal.columns[variance.get_support()]),
        'support_vectors': len(estimator.support_),
        'normal_train_outside_fraction': float((estimator.decision_function(scaled) < 0).mean()),
        'fit_status': int(estimator.fit_status_)}
    return pipeline, info

def risk_score(pipeline, X):
    scores = -pipeline.decision_function(X)
    assert np.isfinite(scores).all()
    return scores

def metrics(y_true, score, threshold):
    prediction = (score > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {'AP': float(average_precision_score(y_true, score)),
        'ROC_AUC': float(roc_auc_score(y_true, score)),
        'precision': float(precision_score(y_true, prediction, zero_division=0)),
        'recall': float(recall_score(y_true, prediction, zero_division=0)),
        'F1': float(f1_score(y_true, prediction, zero_division=0)),
        'F2': float(fbeta_score(y_true, prediction, beta=2, zero_division=0)),
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)}

def choose_threshold(y_true, score):
    candidates = np.r_[np.nextafter(score.min(), -np.inf), np.unique(score)]
    rows = []
    for threshold in candidates:
        prediction = score > threshold
        rows.append({'threshold': float(threshold),
            'F2': float(fbeta_score(y_true, prediction, beta=2, zero_division=0)),
            'precision': float(precision_score(y_true, prediction, zero_division=0))})
    table = pd.DataFrame(rows).sort_values(['F2', 'precision', 'threshold'], ascending=False, kind='stable')
    return float(table.iloc[0].threshold), table
''')
add('markdown','## 4. 개발 4-fold 후보 비교\n\n각 후보의 평균 검증 AP로 선택합니다. 정확한 동률은 미리 정한 후보 순서를 따릅니다. Test 점수는 이 셀에서 산출하지 않습니다. 각 fold의 정상 학습 점수 범위를 기록해 OCSVM 점수 척도의 변동을 확인합니다.')
add('code','''results = {}
for name, data in datasets.items():
    X, y, assignment, dev = data['X'], data['y'], data['assignment'], data['dev']
    records, diagnostics, candidate_scores = [], [], {}
    candidate_id = 0
    for nu in NU_VALUES:
        for multiplier in GAMMA_MULTIPLIERS:
            oof = pd.Series(np.nan, index=dev, dtype=float)
            visits = pd.Series(0, index=dev)
            for fold in range(4):
                train = assignment.loc[assignment.partition.eq('development') & assignment.cv_fold.ne(fold), 'pattern_row'].to_numpy()
                valid = assignment.loc[assignment.partition.eq('development') & assignment.cv_fold.eq(fold), 'pattern_row'].to_numpy()
                assert not set(train) & set(valid) and not (set(train) | set(valid)) & set(data['test'])
                normal = train[y.iloc[train].to_numpy() == 0]
                model, info = fit_normal(X.iloc[normal], nu, multiplier)
                scores = risk_score(model, X.iloc[valid])
                oof.loc[valid] = scores
                visits.loc[valid] += 1
                normal_scores = risk_score(model, X.iloc[normal])
                records.append({'candidate_id': candidate_id, 'nu': nu, 'gamma_multiplier': multiplier,
                    'fold': fold, 'AP': average_precision_score(y.iloc[valid], scores),
                    'ROC_AUC': roc_auc_score(y.iloc[valid], scores)})
                diagnostics.append({'candidate_id': candidate_id, 'fold': fold, **info,
                    'normal_score_q50': float(np.quantile(normal_scores, .5)),
                    'normal_score_q95': float(np.quantile(normal_scores, .95)),
                    'normal_score_q99': float(np.quantile(normal_scores, .99))})
            assert oof.notna().all() and (visits == 1).all()
            candidate_scores[candidate_id] = oof
            candidate_id += 1
    fold_table = pd.DataFrame(records)
    comparison = fold_table.groupby(['candidate_id', 'nu', 'gamma_multiplier'], as_index=False).agg(
        mean_AP=('AP', 'mean'), std_AP=('AP', 'std'), mean_ROC_AUC=('ROC_AUC', 'mean'))
    comparison = comparison.sort_values(['mean_AP', 'candidate_id'], ascending=[False, True], kind='stable')
    best = comparison.iloc[0]
    oof = candidate_scores[int(best.candidate_id)]
    threshold, threshold_table = choose_threshold(y.iloc[dev].to_numpy(), oof.loc[dev].to_numpy())
    results[name] = dict(best=best, threshold=threshold, threshold_table=threshold_table,
        oof=oof, comparison=comparison, fold_table=fold_table, diagnostics=diagnostics)
    print(name.upper(), '개발 검증 결과')
    print(comparison.to_string(index=False))
    print('선택된 nu:', best.nu, '| gamma 배수:', best.gamma_multiplier, '| F2 임계값:', threshold)
''')
add('markdown','## 5. 최종 적합과 고정 Test 평가\n\n선택을 마친 뒤 개발 정상 전체로 다시 적합합니다. OOF에서 정한 수치 임계값을 그대로 test에 적용합니다. 재적합 모델과 fold 모델의 점수 척도 차이는 진단 기록으로 남깁니다. 이는 확률 보정이 아니며 test를 이용해 임계값을 수정하지 않습니다.')
add('code','''evaluation_rows = []
for name, data in datasets.items():
    result = results[name]
    X, y, dev, test = data['X'], data['y'], data['dev'], data['test']
    best, threshold = result['best'], result['threshold']
    normal = dev[y.iloc[dev].to_numpy() == 0]
    model, info = fit_normal(X.iloc[normal], float(best.nu), float(best.gamma_multiplier))
    final_normal_scores = risk_score(model, X.iloc[normal])
    info.update({f'normal_score_q{q}': float(np.quantile(final_normal_scores, q / 100)) for q in [50, 95, 99]})
    predictions = []
    for partition, indices, scores in [
        ('development_oof', dev, result['oof'].loc[dev].to_numpy()),
        ('test', test, risk_score(model, X.iloc[test]))]:
        frame = data['assignment'].iloc[indices][['pattern_row', 'pattern_id', 'cv_fold']].copy()
        frame['dataset'], frame['model'], frame['feature_set'] = name, 'OCSVM', 'all_process_features'
        frame['partition'], frame['y_true'] = partition, y.iloc[indices].to_numpy()
        frame['risk_score'], frame['threshold'] = scores, threshold
        frame['y_pred'] = (scores > threshold).astype(int)
        predictions.append(frame)
        evaluation_rows.append({'dataset': name, 'partition': partition, 'rows': len(indices),
            'risk_patterns': int(y.iloc[indices].sum()), **metrics(y.iloc[indices], scores, threshold)})
    prediction = pd.concat(predictions, ignore_index=True)
    assert prediction.pattern_id.is_unique and len(prediction) == len(X)
    for fold in range(4):
        part = prediction.loc[prediction.partition.eq('development_oof') & prediction.cv_fold.eq(fold)]
        evaluation_rows.append({'dataset': name, 'partition': f'development_fold_{fold}',
            'rows': len(part), 'risk_patterns': int(part.y_true.sum()),
            **metrics(part.y_true, part.risk_score.to_numpy(), threshold)})
    result.update(model=model, final_fit=info, predictions=prediction)
evaluation = pd.DataFrame(evaluation_rows)
print(evaluation.loc[evaluation.partition.isin(['development_oof', 'test'])].to_string(index=False))
print('OOF는 개발 성능이며 최종 test와 구분해서 해석합니다.')
''')
add('markdown','## 6. 그림과 결과 저장\n\nOOF·test 각각 PR 곡선, ROC 곡선, 정상·위험 점수 분포, 혼동행렬을 저장합니다. 목적함수 값 대신 수렴 상태·서포트 벡터 수·정상 학습의 경계 밖 비율을 학습 진단으로 기록합니다. 모델 파이프라인과 임계값은 함께 재사용해야 합니다.')
add('code','''for name, data in datasets.items():
    result = results[name]
    out = OUTPUT / name
    out.mkdir(exist_ok=True)
    result['comparison'].to_csv(out / 'candidate_comparison.csv', index=False)
    result['fold_table'].to_csv(out / 'candidate_fold_metrics.csv', index=False)
    result['threshold_table'].to_csv(out / 'threshold_search.csv', index=False)
    result['predictions'].to_csv(out / 'predictions.csv', index=False)
    evaluation.loc[evaluation.dataset.eq(name)].to_csv(out / 'evaluation.csv', index=False)
    joblib.dump(result['model'], out / 'pipeline.joblib')
    restored = joblib.load(out / 'pipeline.joblib')
    assert np.array_equal(risk_score(restored, data['X'].iloc[data['dev']]),
                          risk_score(result['model'], data['X'].iloc[data['dev']]))
    config = {'dataset': name, 'model': 'OCSVM', 'input_features': list(data['X']),
        'nu': float(result['best'].nu), 'gamma_multiplier': float(result['best'].gamma_multiplier),
        'threshold': result['threshold'], 'score_definition': '-decision_function',
        'prediction_rule': 'risk_score > threshold', 'selection': 'mean development fold AP; ties candidate order',
        'threshold_selection': 'development OOF F2; ties precision then higher threshold',
        'source_hashes': data['source_hashes'], 'split_hash': data['split_hash'],
        'final_fit': result['final_fit'], 'fold_diagnostics': result['diagnostics'],
        'python': platform.python_version(), 'sklearn': sklearn.__version__,
        'unlabeled_inference': False, 'test_used_for_selection': False}
    (out / 'model_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    for partition in ['development_oof', 'test']:
        frame = result['predictions'].loc[result['predictions'].partition.eq(partition)]
        yt, scores = frame.y_true.to_numpy(), frame.risk_score.to_numpy()
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        precision, recall, _ = precision_recall_curve(yt, scores)
        axes[0, 0].plot(recall, precision)
        axes[0, 0].axhline(yt.mean(), ls='--', color='gray', label='위험 패턴 비율')
        axes[0, 0].set(xlabel='재현율', ylabel='정밀도', title='PR 곡선', xlim=(0, 1), ylim=(0, 1.05))
        axes[0, 0].legend()
        fpr, tpr, _ = roc_curve(yt, scores)
        axes[0, 1].plot(fpr, tpr); axes[0, 1].plot([0, 1], [0, 1], '--', color='gray')
        axes[0, 1].set(xlabel='오탐률', ylabel='재현율', title='ROC 곡선')
        bins = np.histogram_bin_edges(scores, bins=20)
        for label, text, color in [(0, '정상', 'steelblue'), (1, '위험', 'tomato')]:
            axes[1, 0].hist(scores[yt == label], bins=bins, alpha=.6, label=text, color=color)
        axes[1, 0].axvline(result['threshold'], color='black', ls='--', label='판정 임계값')
        axes[1, 0].set(xlabel='위험 점수', ylabel='패턴 수', title='점수 분포'); axes[1, 0].legend()
        cm = confusion_matrix(yt, frame.y_pred, labels=[0, 1])
        axes[1, 1].imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2): axes[1, 1].text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
        axes[1, 1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['정상', '위험'], yticklabels=['정상', '위험'], xlabel='예측', ylabel='실제', title='혼동행렬')
        fig.suptitle(name.upper() + (' 개발 OOF 선택 결과' if partition == 'development_oof' else ' 고정 Test 평가'))
        fig.tight_layout()
        figure = out / f'{partition}_evaluation.png'
        fig.savefig(figure, dpi=140); plt.close(fig)
        display(Image(filename=str(figure)))
    artifact_hashes = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'verification.json'}
    (out / 'verification.json').write_text(json.dumps({'passed': True, 'artifact_sha256': artifact_hashes,
        'pipeline_reload_predictions_equal': True, 'all_fits_converged': True,
        'OOF_coverage_verified': True, 'test_excluded_from_selection': True}, indent=2), encoding='utf-8')
    assert all(digest(data['folder'] / f) == sha for f, sha in data['source_hashes'].items())
    print(name.upper(), '모델과 결과 저장:', out)
''')
add('markdown','## 해석 범위\n\n불량 이력 패턴 수가 적으므로 TP·FP·FN을 점수와 함께 확인합니다. 서로 다른 OCSVM의 원시 점수는 같은 척도를 보장하지 않으며 OOF 임계값을 재적합 모델에 적용하는 결과는 고정 test로 평가합니다. 이번 test를 확인한 뒤 같은 test에 맞춰 반복 튜닝하면 독립 평가의 의미가 약해집니다. 파생변수 실험은 별도로 기록하세요.')
(ROOT/'modeling').mkdir(exist_ok=True)
(ROOT/'modeling/ocsvm_conservative.ipynb').write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5},ensure_ascii=False,indent=1),encoding='utf-8')
