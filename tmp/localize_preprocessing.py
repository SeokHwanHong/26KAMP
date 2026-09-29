from pathlib import Path
import contextlib
import hashlib
import io
import json

ROOT = Path(__file__).resolve().parents[1]
sections = [
    ('작업 경로와 라이브러리 준비', '프로젝트 경로를 찾고 전처리에 필요한 라이브러리를 불러옵니다.'),
    ('원본 데이터 불러오기', '라벨·비라벨 원본과 24개 입력 컬럼을 확인하고 원본 파일의 해시를 기록합니다.'),
    ('입력 데이터 품질 점검', '자료형, 결측값, 무한대, 상수 컬럼, 중복 입력과 원본 ID의 고유성을 확인합니다. 상수 제거는 분할 후 학습 구간에서 수행합니다.'),
    ('불량 우선 패턴 통합과 원본 연결', '24개 입력값이 완전히 같은 행을 하나로 통합합니다. 불량이 하나라도 있으면 대표 라벨을 1로 지정하고 원본 행과 관측 횟수를 보존합니다. 비라벨 데이터는 전체 행을 유지합니다.'),
    ('대표 라벨과 원본 복원 검증', '패턴 수와 라벨 수, 불량 이력 보존 여부를 확인합니다. 연결표로 복원한 입력이 원본과 정확히 일치하는지도 검증합니다.'),
    ('전처리 결과 저장과 재확인', '입력, 라벨, 메타데이터와 원본 연결표를 CSV로 저장한 뒤 다시 읽어 값이 유지되는지 확인합니다.'),
    ('전처리 정책과 검증 기록 저장', '적용한 정책, 데이터 품질, 파일 해시와 검증 결과를 기록합니다. 분할과 스케일링은 아직 수행하지 않습니다.'),
    ('전처리 결과 요약', '정상·위험 패턴의 구성과 원본 및 비라벨 행 수를 확인합니다.'),
]
replacements = {
    'Open from the 26KAMP workspace.': '26KAMP 프로젝트 폴더에서 노트북을 실행하세요.',
    'Missing/nonfinite values require an explicit policy': '결측값 또는 무한대가 발견되었습니다. 처리 기준을 먼저 정해야 합니다.',
    '# IDs are local to this source file, assigned in first-occurrence order.': '# 패턴 ID는 파일별로 구분하며 원본에서 처음 등장한 순서대로 부여합니다.',
    '# Choose a defect source row for risk patterns, otherwise the first normal row.': '# 위험 패턴은 불량 원본 행을, 정상 패턴은 첫 정상 행을 대표 행으로 선택합니다.',
    '# Semantic checks: exact coverage, label policy and reversible input mapping.': '# 모든 원본 행의 연결, 대표 라벨 정책과 입력값의 정확한 복원을 검증합니다.',
}
for path in sorted((ROOT / 'preprocessing').glob('preprocess_*_conservative.ipynb')):
    dataset = path.stem.split('_')[1]
    notebook = json.loads(path.read_text(encoding='utf-8'))
    index = 0
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            if source.startswith('# '):
                source = f'# {dataset.upper()} 보수적 공정 관리 전처리\n\n위에서부터 모든 셀을 순서대로 실행합니다. 동일한 24개 입력을 하나의 패턴으로 통합하고, 불량 이력이 하나라도 있으면 위험 라벨 1을 부여합니다. 원본 파일과 비라벨 행은 보존합니다.\n\n정상은 0, 위험은 1입니다. 데이터 분할, 상수 제거, 특징 선택과 스케일링은 후속 단계에서 수행합니다. 컬럼명과 정책 식별자는 기존 데이터와의 호환성을 위해 유지합니다.\n'
            else:
                title, explanation = sections[index]
                index += 1
                source = f'## {index}. {title}\n\n{explanation}\n'
        else:
            for before, after in replacements.items():
                source = source.replace(before, after)
            if "print(json.dumps({'output':" in source:
                source = source[:source.index("print(json.dumps({'output':")] + "print('전처리 저장 및 검증 완료')\nprint('저장 위치:', OUT)\n"
            if "print(metadata.groupby(" in source:
                source = """요약 = metadata.groupby(['representative_label', 'pattern_type']).size().reset_index(name='패턴 수')
요약['representative_label'] = 요약['representative_label'].map({0: '정상', 1: '위험'})
요약['pattern_type'] = 요약['pattern_type'].map({'normal_only': '정상만 관측', 'conflicting': '정상·불량 함께 관측', 'defect_only': '불량만 관측'})
요약 = 요약.rename(columns={'representative_label': '대표 라벨', 'pattern_type': '원본 관측 유형'})
print(요약.to_string(index=False))
print('원본 라벨 행 수:', len(mapping))
print('통합된 입력 패턴 수:', len(X))
print('비라벨 행 수:', len(U))
print('검증 결과: 모든 검사 통과')
"""
        cell['source'] = source.splitlines(keepends=True)
    folder = ROOT / 'data/processed' / dataset / 'conservative'
    def hashes():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
    before = hashes()
    namespace = {}
    count = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(''.join(cell['source']), f'{dataset}_cell_{count}', 'exec'), namespace)
        cell['execution_count'] = count
        cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': capture.getvalue().splitlines(keepends=True)}] if capture.getvalue() else []
    assert hashes() == before, '전처리 산출물이 변경되었습니다.'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{dataset}: {count} cells executed; artifact hashes unchanged')
