"""2026-10-05 Claude: 전체 검사 → (통과 시) CN7 RF 운영 지정 → 기준선 생성.

주의: 이 실행기는 실제 운영 runtime을 바꿀 수 있다(RF가 없을 때). 검사만 하려면 run_checks_only.bat을 쓴다.
- run_checks.py가 실패하면 RF 등록은 하지 않는다.
- RF는 pipeline_cli.py의 'initial --kind rf'와 같은 방법(개발 자료 전체, 고정 설정)으로 만든다.
- 이미 RF 운영 모델이 있으면 새로 만들지 않고, 기준선 유효성만 읽기 전용으로 확인한다(자동 재생성 안 함).
- 종료 코드: 0 성공, 1 검사 실패, 3 RF/기준선 준비 실패 또는 기준선 무효, 2 실행기 오류
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


def main():
    print('[1/2] 전체 검사(run_checks.py) 실행 중... 몇 분 걸릴 수 있습니다.',flush=True)
    r=subprocess.run([sys.executable,str(ROOT/'03.modeling/tests/run_checks.py')],cwd=ROOT,
                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace')
    out['run_checks']=dict(exit_code=r.returncode,passed=r.returncode==0,output=r.stdout[-8000:])
    print(r.stdout[-3000:],flush=True);save()
    if r.returncode!=0:
        out['rf']='건너뜀: 전체 검사 실패. 로그를 Claude에게 전달하세요.';return 1
    print('[2/2] CN7 RF 운영 기준 확인/지정...',flush=True)
    try:
        import sklearn
        from pipeline_runtime import data,fit_supervised
        from workflow_runtime import Operations
        ops=Operations('cn7');current=ops.active().get('rf')
        if current:
            out['rf']=dict(status='already_active',version=current)
        else:
            x,y,_,dev,_,_=data('cn7')
            bundle=fit_supervised('cn7',x.loc[dev],y.loc[dev],kind='rf')
            version=ops.save_candidate(bundle,x.loc[dev],y.loc[dev],source={'kind':'fixed baseline; not grid-selected'})
            ops.initialize(version,'CN7 RF + 검사 예산 운영 기준 (2026-10-05 팀 결정)')
            out['rf']=dict(status='initialized',version=version,baseline=ops.create_baseline(),sklearn=sklearn.__version__)
        out['baseline_status']=ops.baseline_status()
        if not out['baseline_status']['valid']:
            out['rf_note']='기준선이 현재 운영 모델과 맞지 않거나 파일이 바뀜: 자동 재생성하지 않음. 원인 확인 후 pipeline_cli.py baseline'
            return 3
        return 0
    except Exception:
        out['rf']=dict(status='error',traceback=traceback.format_exc());return 3


if __name__=='__main__':
    try:code=main()
    except Exception:
        out['runner_error']=traceback.format_exc();code=2
    out['exit_code']=code;out['finished']=datetime.now().isoformat(timespec='seconds');save()
    print(json.dumps(dict(exit_code=code,run_checks_passed=(out.get('run_checks') or {}).get('passed'),rf=out.get('rf'),
                          baseline=out.get('baseline_status')),ensure_ascii=False,indent=2,default=str))
    print('\n결과 저장:',LOG)
    sys.exit(code)
