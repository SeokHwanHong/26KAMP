"""Run RF checks, preserve logs and confirm shared artifacts are unchanged."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

import rf_pipeline as rf


def main():
    output = rf.ROOT / "output/rf_test_runs" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    snapshot = rf.protected_snapshot()
    steps = []
    for name in ("test_rf_threshold_parts.py", "test_rf_pipeline.py"):
        print(f"Running {name}", flush=True)
        result = subprocess.run([sys.executable, str(rf.ROOT / "modeling" / name)], cwd=rf.ROOT,
                                capture_output=True, encoding="utf-8", errors="replace")
        (output / f"{Path(name).stem}.log").write_text(result.stdout+result.stderr, encoding="utf-8")
        steps.append(dict(script=name, passed=result.returncode == 0, exit_code=result.returncode))
        print(f"{name}: {'PASS' if result.returncode == 0 else 'FAIL'}", flush=True)
    preservation = rf.check_protected(snapshot)
    summary = dict(passed=all(s["passed"] for s in steps), steps=steps, environment=rf.environment(), preservation=preservation)
    rf.write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"TEST_FOLDER={output}", flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
