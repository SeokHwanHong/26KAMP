"""Execute every notebook code cell in one fresh process, preserving outputs."""
from pathlib import Path
import base64
import contextlib
import hashlib
import io
import json
import time
import traceback
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'modeling/ocsvm_cn7_integrated.ipynb'
notebook = nbf.read(path, as_version=4)
namespace = {'__name__':'__main__'}
started = time.time()
count = 0
for index, cell in enumerate(notebook.cells):
    if cell.cell_type != 'code': continue
    count += 1
    cell.outputs = []
    cell.execution_count = count
    def capture(*items, **kwargs):
        for item in items:
            if hasattr(item, '_repr_png_'):
                raw = item._repr_png_()
                if isinstance(raw, tuple): raw = raw[0]
                data = {'image/png': raw if isinstance(raw,str) else base64.b64encode(raw).decode()}
            else:
                data = {'text/plain': str(item)}
                if hasattr(item, '_repr_html_'):
                    html = item._repr_html_()
                    if html: data['text/html'] = html
            cell.outputs.append(nbf.v4.new_output('display_data', data=data))
    namespace['display'] = capture
    buffer = io.StringIO()
    print('Executing cell', index, flush=True)
    try:
        source = cell.source.replace('from IPython.display import display, Image', 'from IPython.display import Image')
        with contextlib.redirect_stdout(buffer):
            exec(compile(source, f'cn7_cell_{index}', 'exec'), namespace)
    except Exception as exc:
        cell.outputs.append(nbf.v4.new_output('error', ename=type(exc).__name__, evalue=str(exc),
                                             traceback=traceback.format_exc().splitlines()))
        raise
    finally:
        if buffer.getvalue():
            cell.outputs.insert(0, nbf.v4.new_output('stream', name='stdout', text=buffer.getvalue()))
        nbf.write(notebook, path)
    print(buffer.getvalue()[-3000:], flush=True)
nbf.validate(notebook)
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs)
           for c in notebook.cells if c.cell_type=='code')
record = dict(all_cells_executed=True, code_cells=count, elapsed_seconds=time.time()-started,
              notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(namespace['OUT']/'notebook_execution.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(record, flush=True)
