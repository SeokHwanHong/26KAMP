from pathlib import Path
import json,io,contextlib,base64,traceback,sys,ast
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'modeling/ocsvm_rg3_feature_scenarios.ipynb'
n=json.loads(p.read_text(encoding='utf-8'))
ns={'__name__':'__main__'}
count=0
for cell in n['cells']:
    if cell['cell_type']!='code':continue
    count+=1
    if '--supplement' in sys.argv and 5<=count<=16:
        if count==10:
            tree=ast.parse(''.join(cell['source']))
            defs=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef)],type_ignores=[])
            exec(compile(defs,'feature_functions','exec'),ns)
        continue
    cell['execution_count']=count;cell['outputs']=[]
    def capture(*items,**kwargs):
        for item in items:
            if hasattr(item,'_repr_png_'):
                raw=item._repr_png_()
                if isinstance(raw,tuple):raw=raw[0]
                if isinstance(raw,str):raw=raw.encode()
                cell['outputs'].append({'output_type':'display_data','data':{'image/png':base64.b64encode(raw).decode()},'metadata':{}})
            else:
                cell['outputs'].append({'output_type':'display_data','data':{'text/plain':[str(item)]},'metadata':{}})
    source=''.join(cell['source']).replace('from IPython.display import display, Image','from IPython.display import Image')
    ns['display']=capture
    buffer=io.StringIO()
    print(f'Running cell {count}',flush=True)
    try:
        with contextlib.redirect_stdout(buffer):exec(compile(source,f'cell_{count}','exec'),ns)
    except Exception:
        cell['outputs'].append({'output_type':'stream','name':'stderr','text':traceback.format_exc().splitlines(keepends=True)})
        raise
    finally:
        if buffer.getvalue():cell['outputs'].insert(0,{'output_type':'stream','name':'stdout','text':buffer.getvalue().splitlines(keepends=True)})
        p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
    print(buffer.getvalue()[-1400:],flush=True)
print('All cells completed.')
