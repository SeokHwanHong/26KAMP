"""Check-only runner (2026-10-05). Runs the full regression/audit suite and never changes the operating runtime.

- Runs 03.modeling/tests/run_checks.py (it verifies runtime/ is unchanged before/after).
- Records Python/package versions and source hashes with the result.
- Read-only status of CN7/RG3 operating models and baselines (never creates or registers anything).
- Exit code: 0 = regression and runtime readiness passed, 1 = either failed, 2 = runner error.
Result: output/claude_review/checks_only_<timestamp>.json (+ checks_only_latest.json)
"""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/claude_review'
SOURCES=['03.modeling/common/pipeline_runtime.py','03.modeling/common/decision_runtime.py','03.modeling/pipeline_cli.py',
         '03.modeling/common/workflow_runtime.py','03.modeling/tests/test_workflow_runtime.py',
         'run_checks_only.py','03.modeling/tests/test_recheck_fixes.py',
         '03.modeling/common/source_provenance.py','03.modeling/tests/test_review_20261006.py',
         '03.modeling/tests/test_decision_runtime.py','03.modeling/tests/audit_logistic_runs.py']


def sha(path):
    p=ROOT/path
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def versions():
    out={'python':platform.python_version(),'executable':sys.executable}
    for name in ['numpy','pandas','scipy','sklearn','joblib','pyarrow']:
        try:out[name]=__import__(name).__version__
        except Exception as exc:out[name]=f'unavailable: {exc}'
    return out


def validate_runtime_status(status):
    """Readiness is separate from regression. Only explicit, error-free waiting is benign."""
    errors=[];waiting=[]
    for ds,item in status.items():
        if 'error' in item:
            errors.append(dict(dataset=ds,reason=item['error']));continue
        if not item.get('active'):errors.append(dict(dataset=ds,reason='운영 모델 미지정'))
        if item.get('baseline',{}).get('valid') is not True:
            errors.append(dict(dataset=ds,reason='기준선 유효성 확인 실패',details=item.get('baseline')))
        for pending in item.get('pending',[]):
            if pending.get('stage') in ('waiting','label_pending') and not pending.get('error'):
                waiting.append(dict(dataset=ds,**pending))
            else:
                errors.append(dict(dataset=ds,reason='복구/확인이 필요한 미완료 배치',details=pending))
    return dict(passed=not errors,errors=errors,normal_waiting=waiting,
                pending_policy='오류 없는 waiting/label_pending만 정상 대기; 복구 단계·근거 손상·미지 상태는 실패')


def main():
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S')
    result=dict(started=datetime.now().isoformat(timespec='seconds'),kind='checks_only',versions=versions(),
                source_sha256={p:sha(p) for p in SOURCES})
    code=2
    try:
        print('전체 검사(run_checks.py) 실행 중... 운영 runtime은 변경하지 않습니다.',flush=True)
        r=subprocess.run([sys.executable,str(ROOT/'03.modeling/tests/run_checks.py')],cwd=ROOT,
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace')
        result['run_checks']=dict(exit_code=r.returncode,passed=r.returncode==0,output=r.stdout[-8000:])
        print(r.stdout[-3000:],flush=True)
        sys.path.insert(0,str(ROOT/'03.modeling/common'))
        from workflow_runtime import Operations
        status={}
        for ds in ['cn7','rg3']:
            try:
                ops=Operations(ds);status[ds]=dict(active=ops.active(),baseline=ops.baseline_status(),pending=ops.pending_batches())
            except Exception as exc:status[ds]=dict(error=f'{type(exc).__name__}: {exc}')
        result['runtime_status_read_only']=status
        result['runtime_validation']=validate_runtime_status(status)
        code=0 if r.returncode==0 and result['runtime_validation']['passed'] else 1
    except Exception:
        result['error']=traceback.format_exc()
    result['exit_code']=code;result['passed']=code==0;result['finished']=datetime.now().isoformat(timespec='seconds')
    OUT.mkdir(parents=True,exist_ok=True)
    text=json.dumps(result,ensure_ascii=False,indent=2,default=str)
    (OUT/f'checks_only_{stamp}.json').write_text(text,encoding='utf-8');(OUT/'checks_only_latest.json').write_text(text,encoding='utf-8')
    print(json.dumps(dict(passed=code==0,exit_code=code,result=str(OUT/f'checks_only_{stamp}.json')),ensure_ascii=False,indent=2))
    return code


if __name__=='__main__':sys.exit(main())
