from pathlib import Path
import json,io,contextlib,base64,traceback,sys,hashlib
ROOT=Path(__file__).resolve().parents[1]
dataset=sys.argv[1]
assert dataset in ['cn7','rg3']
p=ROOT/f'modeling/ocsvm_{dataset}_integrated.ipynb'
n=json.loads(p.read_text(encoding='utf-8'));ns={'__name__':'__main__'};count=0
for cell in n['cells']:
    if cell['cell_type']!='code':continue
    count+=1;cell['execution_count']=count;cell['outputs']=[]
    def capture(*items,**kwargs):
        for item in items:
            if hasattr(item,'_repr_png_'):
                raw=item._repr_png_()
                if isinstance(raw,tuple):raw=raw[0]
                encoded=raw if isinstance(raw,str) else base64.b64encode(raw).decode()
                cell['outputs'].append({'output_type':'display_data','data':{'image/png':encoded},'metadata':{}})
            else:cell['outputs'].append({'output_type':'display_data','data':{'text/plain':[str(item)]},'metadata':{}})
    ns['display']=capture
    source=''.join(cell['source']).replace('from IPython.display import display, Image','from IPython.display import Image')
    buffer=io.StringIO();print(dataset,count,cell['metadata']['integration_source'],flush=True)
    try:
        with contextlib.redirect_stdout(buffer):exec(compile(source,f'cell_{count}','exec'),ns)
    except Exception:
        cell['outputs'].append({'output_type':'stream','name':'stderr','text':traceback.format_exc().splitlines(keepends=True)})
        raise
    finally:
        if buffer.getvalue():cell['outputs'].insert(0,{'output_type':'stream','name':'stdout','text':buffer.getvalue().splitlines(keepends=True)})
        p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
    print(buffer.getvalue()[-450:],flush=True)
for path,sha in n['metadata']['integration']['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
out=ROOT/f'output/ocsvm_{dataset}_integrated/expanded_search_20260929'
(out/'integration_verification.json').write_text(json.dumps({'all_cells_executed':True,'code_cells':count,'sources_unchanged':True,'source_sha256':n['metadata']['integration']['source_sha256'],'notebook_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
print(dataset,'completed',count,flush=True)
