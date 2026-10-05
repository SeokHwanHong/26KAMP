"""Capture the full cold RF experiment log without changing existing results."""
from pathlib import Path
import os
import subprocess
import sys
import uuid


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = Path(__file__).resolve().parents[1]
    log = root / ".work" / f"rf-training-{uuid.uuid4().hex[:12]}.log"
    log.parent.mkdir(exist_ok=True)
    command = [sys.executable, "-u", str(root / "modeling/rf_pipeline.py"), "train", "--dataset", "both", "--jobs", "4"]
    folder = None
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                                   encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
            if line.startswith("RUN_FOLDER="):
                folder = Path(line.strip().split("=", 1)[1])
        code = process.wait()
    if folder is not None:
        (folder / "training.log").write_text(log.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"LOG={log}", flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
