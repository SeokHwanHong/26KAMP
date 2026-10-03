"""One-command handoff verification. Reuses LR artifacts, no full retrain/deploy."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding='utf-8')
run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
out=ROOT/'output/test_runs'/run_id;out.mkdir(parents=True,exist_ok=False)
def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'runtime').rglob('*') if p.is_file()}
before=hashes();results=[]
env=dict(os.environ,PYTHONIOENCODING='utf-8')
for name in ['test_operations.py','test_registration_and_logistic.py','test_existing_ocsvm.py','test_decision_runtime.py','audit_logistic_runs.py']:
    result=subprocess.run([sys.executable,str(Path(__file__).parent/name)],cwd=ROOT,env=env,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace')
    (out/(name+'.log')).write_text(result.stdout,encoding='utf-8')
    results.append(dict(script=name,exit_code=result.returncode,passed=result.returncode==0))
    print(name,'PASS' if result.returncode==0 else 'FAIL',flush=True)
unchanged=before==hashes()
summary=dict(passed=all(r['passed'] for r in results) and unchanged,results=results,
             runtime_unchanged=unchanged,scope='58 regression/integration invocations plus saved LR audit; no field validation')
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(dict(**summary,report=str(out)),ensure_ascii=False,indent=2))
sys.exit(0 if summary['passed'] else 1)
