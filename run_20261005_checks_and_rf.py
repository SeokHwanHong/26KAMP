"""2026-10-05 Claude: 전체 검사 → (통과 시) CN7 RF 운영 지정 → 기준선 생성.

- run_checks.py가 실패하면 RF 등록은 하지 않는다.
- RF는 pipeline_cli.py의 'initial --kind rf'와 같은 방법(개발 자료 전체, 고정 설정)으로 만든다.
- 이미 RF 운영 모델이 있으면 새로 만들지 않는다.
- 결과: output/claude_review/20261005_실행로그.json
"""
from datetime import datetime
from pathlib import Path
import json
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'03.modeling/common'))
LOG=ROOT/'output/claude_review/20261005_실행로그.json'
LOG.parent.mkdir(parents=True,exist_ok=True)
out=dict(started=datetime.now().isoformat(timespec='seconds'),python=sys.version,executable=sys.executable)


def save():
    LOG.write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


print('[1/2] 전체 검사(run_checks.py) 실행 중... 몇 분 걸릴 수 있습니다.',flush=True)
r=subprocess.run([sys.executable,str(ROOT/'03.modeling/tests/run_checks.py')],cwd=ROOT,
                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace')
out['run_checks']=dict(exit_code=r.returncode,passed=r.returncode==0,output=r.stdout[-8000:])
print(r.stdout[-3000:],flush=True);save()

if r.returncode!=0:
    out['rf']='건너뜀: 전체 검사 실패. 로그를 Claude에게 전달하세요.'
else:
    print('[2/2] CN7 RF 운영 지정 + 기준선 생성 중...',flush=True)
    try:
        import sklearn
        from pipeline_runtime import data,fit_supervised
        from decision_runtime import Operations
        ops=Operations('cn7')
        current=ops.active().get('rf')
        if current:
            out['rf']=dict(status='already_active',version=current)
        else:
            x,y,_,dev,_,_=data('cn7')
            bundle=fit_supervised('cn7',x.loc[dev],y.loc[dev],kind='rf')
            version=ops.save_candidate(bundle,x.loc[dev],y.loc[dev],source={'kind':'fixed baseline; not grid-selected'})
            ops.initialize(version,'CN7 RF + 검사 예산 운영 기준 (2026-10-05 팀 결정)')
            baseline=ops.create_baseline()
            out['rf']=dict(status='initialized',version=version,baseline=baseline,sklearn=sklearn.__version__,
                           registry=ops.registry())
    except Exception:
        out['rf']=dict(status='error',traceback=traceback.format_exc())
out['finished']=datetime.now().isoformat(timespec='seconds');save()
print(json.dumps(dict(run_checks_passed=out['run_checks']['passed'],rf=out['rf']),ensure_ascii=False,indent=2,default=str))
print('\n결과 저장:',LOG)
