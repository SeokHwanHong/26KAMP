from pathlib import Path
import contextlib, hashlib, io, json

ROOT = Path(__file__).resolve().parents[1]
for dataset, expected in [('cn7', 28771), ('rg3', 29707)]:
    path = ROOT / 'preprocessing' / f'preprocess_{dataset}_conservative.ipynb'
    nb = json.loads(path.read_text(encoding='utf-8'))
    folder = ROOT / 'data/processed' / dataset / 'conservative'
    protected = ['X_labeled.csv', 'y_labeled.csv', 'labeled_metadata.csv', 'source_row_mapping.csv']
    before = {f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in protected}
    for cell in nb['cells']:
        s = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            s = s.replace('원본 파일과 비라벨 행은 보존합니다.', '원본 파일은 보존하며, 비라벨도 동일 입력을 한 행으로 통합하고 원본 연결표를 저장합니다.')
            s = s.replace('비라벨 데이터는 전체 행을 유지합니다.', '비라벨도 동일 입력을 통합하고 모든 원본 행과의 연결을 보존합니다. 비라벨에는 라벨을 생성하지 않습니다.')
        else:
            if 'U = unlabeled[features].copy()' in s:
                s = s[:s.index('U = unlabeled[features].copy()')] + '''# 비라벨도 같은 24개 입력 기준으로 통합하며 첫 원본 행을 대표로 선택합니다.
u_codes = unlabeled.groupby(features, sort=False, dropna=False).ngroup()
u_work = unlabeled.assign(source_row=np.arange(len(unlabeled)),
    pattern_id=u_codes.map(lambda i: f'PREFIX_U_{i:06d}'))
u_groups = u_work.groupby('pattern_id', sort=False)
u_meta = u_groups.agg(total_count=('source_row', 'size'),
    representative_source_row=('source_row', 'first'),
    representative_source_id=('Unnamed: 0', 'first'))
u_meta['source_ids'] = u_groups['Unnamed: 0'].agg(lambda x: json.dumps(x.tolist()))
u_meta['source_rows'] = u_groups['source_row'].agg(lambda x: json.dumps(x.tolist()))
u_representatives = u_work.drop_duplicates('pattern_id').set_index('pattern_id').loc[u_meta.index]
U = u_representatives[features].reset_index(drop=True)
u_meta = u_meta.reset_index()
u_meta.insert(0, 'pattern_row', np.arange(len(u_meta)))
u_mapping = u_work[['source_row', 'Unnamed: 0', 'pattern_id']].rename(columns={'Unnamed: 0': 'source_id'})
u_mapping = u_mapping.merge(u_meta[['pattern_id', 'pattern_row']], on='pattern_id',
    how='left', validate='many_to_one', sort=False).sort_values('source_row').reset_index(drop=True)
'''.replace('PREFIX', dataset.upper())
            if 'assert list(X) == list(U) == features' in s:
                s += f'''
# 통합된 비라벨 입력으로 모든 원본 행을 정확히 복원할 수 있어야 합니다.
assert len(U) == {expected}
assert not U.duplicated().any()
assert len(u_mapping) == len(unlabeled) and u_mapping.source_row.is_unique
assert int(u_meta.total_count.sum()) == len(unlabeled)
assert np.array_equal(U.iloc[u_mapping.pattern_row.to_numpy()].to_numpy(), unlabeled[features].to_numpy())
assert 'PassOrFail' not in U.columns and 'PassOrFail' not in u_meta.columns
# 패턴별 예측도 이 인덱스로 원본 행 순서에 복원할 수 있습니다.
pattern_prediction_example = np.arange(len(U))
restored_prediction_example = pattern_prediction_example[u_mapping.pattern_row.to_numpy()]
assert len(restored_prediction_example) == len(unlabeled)
'''
            s = s.replace("'X_unlabeled.csv': U, 'unlabeled_metadata.csv': u_meta,", "'X_unlabeled.csv': U, 'unlabeled_metadata.csv': u_meta,\n    'unlabeled_source_row_mapping.csv': u_mapping,")
            s = s.replace("'unlabeled_rows': len(U), 'unlabeled_unique_patterns': int(u_codes.nunique()),", "'unlabeled_rows': len(U), 'unlabeled_rows_before': len(unlabeled),\n    'unlabeled_unique_patterns': len(U), 'unlabeled_removed_duplicate_rows': len(unlabeled) - len(U),")
            s = s.replace('Preserve all source rows and values for later row-level inference; no labels imputed.', 'Aggregate exact 24-feature duplicates; retain first row and reversible source mapping; no labels imputed.')
            s = s.replace("'all CSV round trips equal',", "'unlabeled patterns unique and all source inputs reconstructed exactly',\n    'unlabeled prediction mapping verified', 'all CSV round trips equal',")
            s = s.replace('- X_unlabeled.csv: 원본 35,239행과 값을 유지합니다. labeled와 같은 24개 컬럼 순서입니다.', '- X_unlabeled.csv: 비라벨 35,239행을 28,771개 고유 패턴으로 통합합니다. 입력값과 24개 컬럼 순서는 유지합니다.')
            s = s.replace('- X_unlabeled.csv: 원본 35,941행과 값을 유지합니다. labeled와 같은 24개 컬럼 순서입니다.', '- X_unlabeled.csv: 비라벨 35,941행을 29,707개 고유 패턴으로 통합합니다. 입력값과 24개 컬럼 순서는 유지합니다.')
            s = s.replace('- unlabeled_metadata.csv: 비라벨 원본 행과 비라벨 파일 내부 패턴 ID를 연결합니다.', '- unlabeled_metadata.csv: 비라벨 패턴별 대표 행, 관측 횟수와 원본 ID 목록입니다.\n- unlabeled_source_row_mapping.csv: 비라벨 원본의 모든 행을 통합 패턴에 연결합니다. pattern_row로 예측을 원본 순서에 복원할 수 있습니다.')
            s = s.replace("print('비라벨 행 수:', len(U))", "print('비라벨 원본 행 수:', len(unlabeled))\nprint('비라벨 통합 패턴 수:', len(U))\nprint('비라벨 제거 중복 행 수:', len(unlabeled) - len(U))")
        cell['source'] = s.splitlines(keepends=True)
    ns = {}; count = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code': continue
        count += 1
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(''.join(cell['source']), f'{dataset}_cell_{count}', 'exec'), ns)
        cell['execution_count'] = count
        cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': output.getvalue().splitlines(keepends=True)}] if output.getvalue() else []
    assert all(hashlib.sha256((folder / f).read_bytes()).hexdigest() == sha for f, sha in before.items())
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(dataset, 'unique unlabeled patterns:', len(ns['U']), 'removed:', len(ns['unlabeled'])-len(ns['U']), 'all checks passed')
