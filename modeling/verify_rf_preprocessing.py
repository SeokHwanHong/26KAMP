"""Re-run original preprocessing and the fixed split in an isolated temp root.

The live processed CSVs are never overwritten. Generated artifact hashes must
match the fixed CN7/RG3 inputs used by the Random Forest notebooks.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def run_notebook(path: Path, namespace: dict):
    notebook = nbformat.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            print(f"{path.name}: cell {index}", flush=True)
            exec(compile(cell.source, f"{path.name}:cell{index}", "exec"), namespace)


def copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


results = {}
with tempfile.TemporaryDirectory(prefix="rf_preprocess_repro_") as temp:
    temp_root = Path(temp)
    for dataset in ("cn7", "rg3"):
        for label_status in ("labeled", "unlabeled"):
            name = f"moldset_{label_status}_{dataset}.csv"
            copy(ROOT / "data" / "origin" / name, temp_root / "data" / "origin" / name)
        source_schema = ROOT / "data" / "processed" / dataset / "preprocessing_manifest.json"
        copy(source_schema, temp_root / "data" / "processed" / dataset / "preprocessing_manifest.json")
    document = next((ROOT / "output" / "conservative_process_strategy").glob("*.docx"))
    copy(document, temp_root / "output" / "conservative_process_strategy" / document.name)
    previous = Path.cwd()
    try:
        os.chdir(temp_root)
        for dataset in ("cn7", "rg3"):
            run_notebook(ROOT / "preprocessing" / f"preprocess_{dataset}_conservative.ipynb",
                         {"__name__": "__main__", "display": lambda *args, **kwargs: None})
        run_notebook(ROOT / "preprocessing" / "split_conservative_data.ipynb",
                     {"__name__": "__main__", "display": lambda *args, **kwargs: None})
    finally:
        os.chdir(previous)
    for dataset in ("cn7", "rg3"):
        original = ROOT / "data" / "processed" / dataset / "conservative"
        reproduced = temp_root / "data" / "processed" / dataset / "conservative"
        preprocessing_manifest = json.loads((original / "preprocessing_manifest.json").read_text(encoding="utf-8"))
        split_manifest = json.loads((original / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
        names = list(preprocessing_manifest["artifact_sha256"]) + [
            f"splits/{name}" for name in split_manifest["artifact_sha256"]
        ]
        mismatches = []
        import hashlib
        for name in names:
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            if digest(original / name) != digest(reproduced / name):
                mismatches.append(name)
        results[dataset] = dict(compared_artifacts=len(names), mismatches=mismatches,
                                identical=not mismatches)
        print(dataset, results[dataset], flush=True)
        if mismatches:
            raise AssertionError(f"{dataset} reproduction mismatch: {mismatches}")

for dataset in ("cn7", "rg3"):
    output = ROOT / "output" / f"random_forest_{dataset}" / "manual_seven_scenarios_v1"
    (output / "preprocessing_reproduction.json").write_text(
        json.dumps(results[dataset], ensure_ascii=False, indent=2), encoding="utf-8")
