from pathlib import Path
import json
import uuid

root=Path(__file__).resolve().parents[1]
cells=[]
def add(kind, text):
    cell={'cell_type':kind,'id':uuid.uuid4().hex[:8],'metadata':{},'source':text.strip().splitlines(keepends=True)}
    if kind=='code': cell.update(execution_count=None,outputs=[])
    cells.append(cell)

add('markdown', '''# 보수적 공정 관리 데이터 분할

CN7·RG3의 통합 패턴 데이터를 각각 분할합니다. **Test 20%를 먼저 고정**하고 **개발 80% 안에서 Stratified 4-fold**를 구성합니다. 매 반복에서 전체의 약 60%가 학습, 20%가 검증입니다. LR·RF·OCSVM은 동일 매핑을 재사용합니다.

위에서부터 실행하세요. 이미 저장된 매핑이 있으면 재생성 결과와 일치하는지 확인하며, 다른 매핑으로 덮어쓰지 않습니다. 비라벨 데이터는 분할하지 않습니다. 특징 선택·스케일링·모델 학습은 수행하지 않습니다.''')
add('markdown','## 1. 경로와 분할 설정\n\nseed는 42로 고정합니다. 패턴 통합 결과만 사용하고, 상위 폴더의 기존 3-fold 자료는 사용하지 않습니다.')
add('code','''from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'data/origin').is_dir()), None)
if ROOT is None:
    raise RuntimeError('26KAMP 프로젝트 안에서 실행하세요.')
SEED = 42
TEST_SIZE = 0.2
N_SPLITS = 4
DATASETS = ['cn7', 'rg3']

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
''')
add('markdown','## 2. 전처리 결과와 정렬 검증\n\n입력·라벨·메타데이터의 행 순서, 해시, 고유 패턴, 대표 라벨을 확인합니다. 메타데이터는 분할 추적에만 쓰며 모델 입력에 포함하지 않습니다.')
add('code','''datasets = {}
for name in DATASETS:
    folder = ROOT / 'data/processed' / name / 'conservative'
    manifest = json.loads((folder / 'preprocessing_manifest.json').read_text(encoding='utf-8'))
    files = ['X_labeled.csv', 'y_labeled.csv', 'labeled_metadata.csv']
    hashes = {f: sha256(folder / f) for f in files}
    assert all(hashes[f] == manifest['artifact_sha256'][f] for f in files), '전처리 결과가 기록과 다릅니다.'
    X = pd.read_csv(folder / 'X_labeled.csv', float_precision='round_trip')
    y = pd.read_csv(folder / 'y_labeled.csv')['PassOrFail']
    meta = pd.read_csv(folder / 'labeled_metadata.csv')
    assert len(X) == len(y) == len(meta)
    assert list(X) == manifest['feature_columns'] and not X.duplicated().any()
    assert meta.pattern_id.is_unique
    assert np.array_equal(meta.pattern_row.to_numpy(), np.arange(len(X)))
    assert np.array_equal(y.to_numpy(), meta.representative_label.to_numpy())
    assert set(y) == {0, 1}
    datasets[name] = {'folder': folder, 'X': X, 'y': y, 'meta': meta, 'hashes': hashes}
    print(f'{name.upper()}: 전체 {len(y)}개 / 정상 {int((y == 0).sum())}개 / 위험 {int((y == 1).sum())}개')
''')
add('markdown','## 3. 고정 Test와 개발 4-fold 생성\n\n`cv_fold`는 해당 패턴이 검증에 사용되는 fold 번호입니다. 개발 데이터는 0~3, test는 -1입니다. 반복 k의 학습은 개발 데이터 중 cv_fold가 k가 아닌 행, 검증은 k인 행입니다. pattern_row는 전처리 CSV의 0부터 시작하는 행 위치입니다.')
add('code','''def make_assignment(y, meta):
    indices = np.arange(len(y))
    development, test = train_test_split(indices, test_size=TEST_SIZE,
        stratify=y.to_numpy(), random_state=SEED, shuffle=True)
    # 원본 패턴 순서로 정렬한 뒤 내부 fold를 생성하여 순서 규칙도 고정합니다.
    development, test = np.sort(development), np.sort(test)
    result = meta[['pattern_row', 'pattern_id']].copy()
    result['label'] = y.to_numpy()
    result['partition'] = 'test'
    result['cv_fold'] = -1
    result.loc[development, 'partition'] = 'development'
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (_, valid_pos) in enumerate(splitter.split(development, y.iloc[development])):
        result.loc[development[valid_pos], 'cv_fold'] = fold
    return result

def indices_for_fold(assignment, fold):
    assert fold in range(N_SPLITS)
    dev = assignment.partition.eq('development')
    train = assignment.loc[dev & assignment.cv_fold.ne(fold), 'pattern_row'].to_numpy()
    valid = assignment.loc[dev & assignment.cv_fold.eq(fold), 'pattern_row'].to_numpy()
    test = assignment.loc[~dev, 'pattern_row'].to_numpy()
    return train, valid, test

for name, data in datasets.items():
    data['assignment'] = make_assignment(data['y'], data['meta'])
''')
add('markdown','## 4. 누수와 재현성 검증\n\n학습·검증·test 간 패턴 교집합이 없어야 합니다. 모든 개발 패턴은 검증 1회·학습 3회에 참여하고 test는 어떤 학습·검증에도 참여하지 않아야 합니다. 각 구간에 정상과 위험 클래스가 모두 있는지 확인합니다.')
add('code','''summaries = []
for name, data in datasets.items():
    assignment, y = data['assignment'], data['y']
    pd.testing.assert_frame_equal(assignment, make_assignment(y, data['meta']))
    train_count = np.zeros(len(y), dtype=int)
    valid_count = np.zeros(len(y), dtype=int)
    test_reference = set(assignment.loc[assignment.partition.eq('test'), 'pattern_row'])
    assert set(assignment.loc[assignment.partition.eq('development'), 'cv_fold']) == set(range(N_SPLITS))
    for fold in range(N_SPLITS):
        train, valid, test = indices_for_fold(assignment, fold)
        assert not (set(train) & set(valid) or set(train) & set(test) or set(valid) & set(test))
        assert set(train) | set(valid) | set(test) == set(range(len(y)))
        assert set(test) == test_reference
        train_count[train] += 1
        valid_count[valid] += 1
        row = {'dataset': name, 'fold': fold}
        for role, idx in [('train', train), ('validation', valid), ('test', test)]:
            assert set(y.iloc[idx]) == {0, 1}
            row[role + '_rows'] = len(idx)
            row[role + '_normal'] = int((y.iloc[idx] == 0).sum())
            row[role + '_risk'] = int((y.iloc[idx] == 1).sum())
            row[role + '_risk_fraction'] = float(y.iloc[idx].mean())
        summaries.append(row)
    dev = assignment.partition.eq('development').to_numpy()
    assert (train_count[dev] == 3).all() and (valid_count[dev] == 1).all()
    assert (train_count[~dev] == 0).all() and (valid_count[~dev] == 0).all()
summary = pd.DataFrame(summaries)
shown = summary[['dataset', 'fold', 'train_rows', 'train_risk', 'validation_rows', 'validation_risk', 'test_rows', 'test_risk']].rename(columns={
    'dataset': '데이터', 'fold': '검증 fold', 'train_rows': '학습 수', 'train_risk': '학습 위험',
    'validation_rows': '검증 수', 'validation_risk': '검증 위험', 'test_rows': '고정 test 수', 'test_risk': 'test 위험'})
print(shown.to_string(index=False))
print('검증 완료: 구간 간 중복 없음, test 고정, 개발 패턴 검증 1회·학습 3회, 동일 seed 재현')
''')
add('markdown','## 5. 공통 분할 저장\n\n각 conservative/splits 폴더에 매핑, 구간별 입력·라벨, fold별 인덱스와 요약을 저장합니다. 기존 고정 분할이 다르면 오류로 중단합니다. 이후 모델은 이 매핑을 불러오며 독립적으로 다시 분할하지 않습니다.')
add('code','''# 먼저 두 데이터셋의 기존 매핑을 모두 확인합니다.
for name, data in datasets.items():
    out = data['folder'] / 'splits'
    if (out / 'split_assignments.csv').exists():
        pd.testing.assert_frame_equal(pd.read_csv(out / 'split_assignments.csv'), data['assignment'])
    if (out / 'split_manifest.json').exists():
        old = json.loads((out / 'split_manifest.json').read_text(encoding='utf-8'))
        assert old['source_sha256'] == data['hashes'], '원본 변경: 기존 test 분할을 자동 갱신하지 않습니다.'

for name, data in datasets.items():
    out = data['folder'] / 'splits'
    out.mkdir(exist_ok=True)
    assignment = data['assignment']
    tables = {'split_assignments.csv': assignment,
        'fold_summary.csv': summary.loc[summary.dataset.eq(name)].reset_index(drop=True)}
    for part in ['development', 'test']:
        rows = assignment.loc[assignment.partition.eq(part), 'pattern_row'].to_numpy()
        tables['X_' + part + '.csv'] = data['X'].iloc[rows].reset_index(drop=True)
        tables['y_' + part + '.csv'] = data['y'].iloc[rows].reset_index(drop=True).to_frame()
        tables[part + '_metadata.csv'] = assignment.iloc[rows].reset_index(drop=True)
    for filename, frame in tables.items():
        frame.to_csv(out / filename, index=False, encoding='utf-8')
        pd.testing.assert_frame_equal(pd.read_csv(out / filename, float_precision='round_trip'), frame, check_dtype=False)
    folds = {}
    for fold in range(N_SPLITS):
        train, valid, test = indices_for_fold(assignment, fold)
        folds[str(fold)] = {'train_pattern_rows': train.tolist(), 'validation_pattern_rows': valid.tolist()}
    (out / 'fold_indices.json').write_text(json.dumps({'index_base': 0, 'index_reference': '../X_labeled.csv',
        'test_pattern_rows': test.tolist(), 'folds': folds}, indent=2), encoding='utf-8')
    record = {'dataset': name, 'seed': SEED, 'test_size': TEST_SIZE, 'n_splits': N_SPLITS,
        'method': 'Stratified holdout test followed by StratifiedKFold on development',
        'development_order': 'ascending original pattern_row before StratifiedKFold',
        'source_sha256': data['hashes'], 'python_version': platform.python_version(),
        'sklearn_version': sklearn.__version__, 'pandas_version': pd.__version__,
        'preprocessing_fitted': False, 'models_trained': False,
        'checks': {'disjoint_partitions': True, 'test_fixed': True, 'both_classes_present': True,
                   'development_validation_once': True, 'development_train_three_times': True,
                   'seed_reproducible': True, 'saved_tables_verified': True},
        'artifact_sha256': {f: sha256(out / f) for f in [*tables, 'fold_indices.json']}}
    (out / 'split_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'README.md').write_text('# 고정 test와 개발 4-fold\\n\\n'
        'test 20%는 최종 평가 전용입니다. development 80%에서 cv_fold를 번갈아 검증합니다.\\n'
        'split_assignments.csv의 pattern_row는 상위 X_labeled.csv 기준 0부터 시작하는 행 위치입니다.\\n'
        'fold_indices.json도 같은 원본 패턴 행 위치를 사용하며 X_development의 행 번호가 아닙니다.\\n'
        'development_metadata.csv와 test_metadata.csv는 각 구간 CSV의 행 순서에 대응합니다.\\n'
        '각 fold 학습에서 전처리를 적합하세요. OCSVM은 분할 후 학습 라벨 0만 추출합니다.\\n'
        '비라벨 데이터는 포함하지 않습니다. 모델과 임계값 선택에 test를 사용하지 마세요.\\n', encoding='utf-8')
    assert all(sha256(data['folder'] / f) == sha for f, sha in data['hashes'].items())
    print(name.upper(), '분할 저장 완료:', out)
''')
add('markdown','## 6. 후속 학습에서 사용하는 방법\n\n반복 k는 `indices_for_fold(assignment, k)`로 학습·검증 행을 가져옵니다. LR·RF는 학습 전체, OCSVM은 학습 중 라벨 0만 적합합니다. 특징 선택과 스케일러는 학습 구간에서만 적합하세요. 개발 검증에서 설정과 임계값을 확정한 뒤 개발 80%로 재학습하고 고정 test에서 평가합니다.')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
path=root/'preprocessing/split_conservative_data.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
