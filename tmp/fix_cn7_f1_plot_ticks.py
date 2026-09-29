"""Refresh only the changed plot code/output, preserving all computed results."""
from pathlib import Path
import base64
import json
import os
import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT/'tmp/matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import Image
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
OUT = ROOT/'output/ocsvm_cn7_integrated/validation_f1_20260929'
path = ROOT/'modeling/ocsvm_cn7_integrated.ipynb'
notebook = nbformat.read(path, as_version=4)
cell = notebook.cells[7]
old = "    ax.set_xscale('symlog', linthresh=1e-6)"
new = old + "\n    ticks = [-1e3,-1.,-1e-3,-1e-6,0.,1e-6,1e-3,1.,1e3]\n    ax.set_xticks(ticks, [f'{v:g}' for v in ticks])\n    ax.minorticks_off()"
assert old in cell.source and 'ticks = [-1e3' not in cell.source
cell.source = cell.source.replace(old, new)
choice = json.loads((OUT/'selection_manifest.json').read_text(encoding='utf-8'))['choices'][0]
def display(item):
    pass
# The modified section reads saved validation results and regenerates one figure only.
source = cell.source[cell.source.index('    sensitivity ='):]
import textwrap
exec(compile(textwrap.dedent(source), 'threshold_plot_refresh', 'exec'))
images = [o for o in cell.outputs if o.output_type=='display_data' and 'image/png' in o.data]
assert len(images) == 2
images[-1].data['image/png'] = base64.b64encode((OUT/'threshold_sensitivity.png').read_bytes()).decode()
nbformat.validate(notebook); nbformat.write(notebook, path)
import hashlib
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
record_path = OUT/'notebook_execution.json'
record = json.loads(record_path.read_text(encoding='utf-8'))
record['notebook_sha256'] = digest(path)
record['plot_only_refresh_executed'] = True
record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
verification_path = OUT/'verification.json'
verification = json.loads(verification_path.read_text(encoding='utf-8'))
for name in ['threshold_sensitivity.png','selected_threshold_sensitivity.csv']:
    verification['artifact_sha256'][name] = digest(OUT/name)
verification_path.write_text(json.dumps(verification, indent=2), encoding='utf-8')
print('Plot ticks refreshed; search, model, selection and test results unchanged.')
