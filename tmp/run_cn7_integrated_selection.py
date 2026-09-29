from pathlib import Path
import json,io,contextlib,base64,traceback
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'modeling/ocsvm_cn7_integrated.ipynb'
n=json.loads(p.read_text(encoding='utf-8'));ns={'__name__':'__main__'}
count=0
for index,cell in enumerate(n['cells']):
 if cell['cell_type']!='code':continue
 count+=1
 origin=cell['metadata'].get('integration_source',{})
 if not (origin.get('notebook')=='ocsvm_cn7_feature_scenarios.ipynb' and origin.get('cell_index') in [2,3,4,6]) and not cell['metadata'].get('selection_cycle'):continue
 cell['execution_count']=count;cell['outputs']=[]
 def capture(*items,**kwargs):
  for item in items:
   if hasattr(item,'_repr_png_'):
    raw=item._repr_png_()
    if isinstance(raw,tuple):raw=raw[0]
    cell['outputs'].append({'output_type':'display_data','data':{'image/png':raw if isinstance(raw,str) else base64.b64encode(raw).decode()},'metadata':{}})
   else:cell['outputs'].append({'output_type':'display_data','data':{'text/plain':[str(item)]},'metadata':{}})
 ns['display']=capture;buf=io.StringIO()
 print('Executing',index,flush=True)
 try:
  with contextlib.redirect_stdout(buf):exec(compile(''.join(cell['source']).replace('from IPython.display import display, Image','from IPython.display import Image'),f'cell_{index}','exec'),ns)
 except Exception:
  cell['outputs'].append({'output_type':'stream','name':'stderr','text':traceback.format_exc().splitlines(keepends=True)})
  raise
 finally:
  if buf.getvalue():cell['outputs'].insert(0,{'output_type':'stream','name':'stdout','text':buf.getvalue().splitlines(keepends=True)})
  p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
 print(buf.getvalue()[-2000:],flush=True)
