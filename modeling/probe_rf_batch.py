"""Exercise the RF CLI using explicitly synthetic product IDs and labels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

import rf_pipeline as rf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    checks = []
    for dataset in ("cn7", "rg3"):
        folder = run / dataset
        artifact = rf.load_artifact(folder)
        X, _, _, _, test, _ = rf.definitions(dataset)["load_data"]()
        # Values remain in the supplied labeled coordinates. These are test
        # fixtures, not new production observations or independent truth.
        frame = X.iloc[[test[0], test[0], test[1]]].reset_index(drop=True)
        frame.insert(0, "Unnamed: 0", [f"synthetic-{dataset}-{i}" for i in range(3)])
        frame["PassOrFail"] = [0, 1, 0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"synthetic.csv"
            frame.to_csv(path, index=False)
            base = [sys.executable, str(rf.ROOT/"modeling/rf_pipeline.py"), "infer", str(folder), str(path)]
            rejected = subprocess.run(base, cwd=rf.ROOT, capture_output=True, encoding="utf-8", errors="replace")
            assert rejected.returncode != 0
            allowed = subprocess.run(base+["--coordinates-confirmed", "--label-source", "synthetic CLI test; not production truth"],
                                     cwd=rf.ROOT, capture_output=True, encoding="utf-8", errors="replace")
            if allowed.returncode:
                raise RuntimeError(allowed.stdout+allowed.stderr)
            output = Path(allowed.stdout.strip())
            products = pd.read_csv(output/"product_predictions.csv")
            patterns = pd.read_csv(output/"pattern_predictions.csv")
            assert len(products) == 3 and len(patterns) == 2
            assert patterns.total_count.tolist() == [2, 1]
            assert patterns.representative_label.tolist() == [1, 0]
            np.testing.assert_array_equal(products.prediction, rf.predict(artifact, frame[artifact["input_columns"]])[1])
            check = dict(dataset=dataset, passed=True, unconfirmed_input_rejected=True,
                         product_rows=3, unique_patterns=2, conflict_max_label_preserved=True,
                         synthetic=True, output=str(output))
            checks.append(check)
            (run/f"batch_cli_{dataset}.log").write_text(rejected.stdout+rejected.stderr+"\nCONFIRMED SYNTHETIC FIXTURE\n"+allowed.stdout+allowed.stderr, encoding="utf-8")
    rf.write_json(run/"batch_cli_checks.json", dict(passed=True, checks=checks))
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
