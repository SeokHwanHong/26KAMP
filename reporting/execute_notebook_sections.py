"""Append and execute only the report cells; preserve every original cell."""
from pathlib import Path
import hashlib,json,subprocess,sys,os
_cache=Path(__file__).resolve().parents[1]/'tmp/v4_qa/jupyter'
for _key,_sub in [('IPYTHONDIR','ipython'),('JUPYTER_RUNTIME_DIR','runtime')]:
    _folder=_cache/_sub;_folder.mkdir(parents=True,exist_ok=True);os.environ[_key]=str(_folder)
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from visualizations import ROOT,OUT,NOTEBOOKS

TAG='report_v4_visualization'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    protected={p.as_posix():sha(p) for folder in ['runtime','data/processed'] for p in (ROOT/folder).rglob('*') if p.is_file()}
    originals={};nb=nbformat.v4.new_notebook()
    for key in NOTEBOOKS:
        path=ROOT/key;doc=json.loads(path.read_text(encoding='utf-8'))
        old=[c for c in doc['cells'] if TAG not in c.get('metadata',{}).get('tags',[])]
        originals[key]=(doc,old)
        code=f'''# 보고서용 시각화: 기존 학습 셀을 실행하지 않고 저장 결과를 표시한다.
from pathlib import Path
import sys
_report_root = next(p for p in [Path.cwd(), *Path.cwd().parents]
                    if (p / "reporting/visualizations.py").is_file())
if str(_report_root) not in sys.path:
    sys.path.insert(0, str(_report_root))
from reporting.visualizations import show_notebook
_report_figure_ids = show_notebook({key!r})
print("보고서 수록 그림:", ", ".join(_report_figure_ids))
'''
        nb.cells.append(nbformat.v4.new_code_cell(code,metadata={'tags':[TAG]}))
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv[0]=sys.executable
    client=NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    records=[]
    for (key,(doc,old)),executed in zip(originals.items(),nb.cells):
        markdown=nbformat.v4.new_markdown_cell('''## 결과보고서 v4 시각화

아래 셀만 실행하면 이 노트북에 해당하는 보고서 그림을 표시한다. 기존 학습·전처리·운영 셀을 다시 실행할 필요가 없다.
공통 생성 코드는 `reporting/visualizations.py`에 있다. 그림이 없으면 해당 코드를 먼저 실행한다.
고정 Test는 이미 관찰한 후속 평가이며, RF 연구 v1과 표준화 수정 후 v2를 구분한다.
시각화용 중앙값·IQR 변환은 표시 목적에 한정되며 모델 입력이나 기존 표준화 로직을 변경하지 않는다.
''',metadata={'tags':[TAG]})
        doc['cells']=old+[dict(markdown),dict(executed)]
        path=ROOT/key;path.write_text(json.dumps(doc,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        saved=json.loads(path.read_text(encoding='utf-8'))
        assert saved['cells'][:len(old)]==old, key+' original cells changed'
        errors=[o for o in executed.outputs if o.output_type=='error'];assert not errors
        count=sum(o.output_type in ('display_data','execute_result') and 'image/png' in o.get('data',{}) for o in executed.outputs)
        assert count>0
        records.append(dict(notebook=key,original_cells_preserved=len(old),figure_outputs=count,
                            execution_count=executed.execution_count,original_cells_sha256=hashlib.sha256(json.dumps(old,ensure_ascii=False,sort_keys=True).encode()).hexdigest()))
    for path,expected in protected.items():assert sha(Path(path))==expected,'Protected artifact changed: '+path
    result=dict(notebooks=records,figure_outputs=sum(r['figure_outputs'] for r in records),
                protected_artifacts_unchanged=len(protected),training_cells_executed=False,
                execution_python=sys.executable,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    (OUT/'notebook_execution_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
