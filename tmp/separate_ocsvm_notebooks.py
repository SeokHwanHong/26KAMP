from pathlib import Path
import base64
import contextlib
import copy
import hashlib
import io
import json
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'modeling/ocsvm_conservative.ipynb'
original = json.loads(source.read_text(encoding='utf-8'))
for name in ['cn7', 'rg3']:
    notebook = copy.deepcopy(original)
    for cell in notebook['cells']:
        cell['id'] = uuid.uuid4().hex[:8]
        text = ''.join(cell['source'])
        text = text.replace('# OCSVM 보수적 공정 위험 탐지', f'# {name.upper()} OCSVM 보수적 공정 위험 탐지')
        text = text.replace('CN7·RG3를 별도로 학습합니다.', f'{name.upper()} 데이터만 학습하고 평가하는 독립 노트북입니다.')
        text = text.replace("DATASETS = ['cn7', 'rg3']", f"DATASETS = ['{name}']  # 이 노트북은 {name.upper()}만 실행합니다.")
        cell['source'] = text.splitlines(keepends=True)
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
    out = ROOT / 'output/ocsvm_conservative' / name
    csv_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')}
    other = 'rg3' if name == 'cn7' else 'cn7'
    other_dir = ROOT / 'output/ocsvm_conservative' / other
    other_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in other_dir.iterdir() if p.is_file()}
    ns = {}
    count = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        capture = io.StringIO()
        images = []
        def capture_image(obj):
            images.append({'output_type': 'display_data', 'data': {
                'image/png': base64.b64encode(obj.data).decode('ascii'),
                'text/plain': ['평가 그래프']}, 'metadata': {}})
        if count > 1:
            ns['display'] = capture_image
        with contextlib.redirect_stdout(capture):
            exec(compile(''.join(cell['source']), f'{name}_cell_{count}', 'exec'), ns)
        cell['execution_count'] = count
        text = capture.getvalue()
        cell['outputs'] = ([{'output_type': 'stream', 'name': 'stdout', 'text': text.splitlines(keepends=True)}] if text else []) + images
    assert set(ns['datasets']) == {name} and set(ns['results']) == {name}
    assert csv_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')}
    assert other_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in other_dir.iterdir() if p.is_file()}
    notebook['metadata']['language_info']['version'] = sys.version.split()[0]
    target = ROOT / 'modeling' / f'ocsvm_{name}_conservative.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    print(name, 'independent execution passed; prior CSV results unchanged; other dataset untouched', flush=True)
