"""Storage-only regression checks; no forest fitting or experiment changes."""

import ast
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "modeling/random_forest_manual.py").read_text(encoding="utf-8")
FUNCTIONS = {"sha256", "write_threshold_search", "read_threshold_search", "metrics"}
helper_ast = ast.Module(body=[node for node in ast.parse(SOURCE).body
                             if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS],
                        type_ignores=[])
HELPERS = dict(Path=Path, hashlib=hashlib, json=json, pd=pd, np=np)
exec(compile(helper_ast, "rf_storage_helpers", "exec"), HELPERS)


class ThresholdPartTests(unittest.TestCase):
    def test_real_exports_and_actual_verification_cell_without_original_csv(self):
        for dataset in ("cn7", "rg3"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temporary:
                folder = ROOT / "output" / f"random_forest_{dataset}" / "manual_seven_scenarios_v1"
                original = pd.read_csv(folder / "threshold_search.csv")
                split = HELPERS["read_threshold_search"](folder)
                pd.testing.assert_frame_equal(original, split, check_exact=True)
                shared = Path(temporary)
                for name in ("threshold_search_part1.csv", "threshold_search_part2.csv",
                             "threshold_search_parts.json", "selection_manifest.json",
                             "selected_oof_predictions.csv", "test_predictions.csv"):
                    shutil.copy2(folder / name, shared / name)
                assert not (shared / "threshold_search.csv").exists()
                pd.testing.assert_frame_equal(original, HELPERS["read_threshold_search"](shared),
                                              check_exact=True)
                selection = json.loads((shared / "selection_manifest.json").read_text(encoding="utf-8"))
                summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                assignment = pd.read_csv(ROOT / "data" / "processed" / dataset /
                                         "conservative/splits/split_assignments.csv")
                artifact = joblib.load(folder / "model.joblib")
                namespace = dict(HELPERS, OUT=shared, DATASET=dataset,
                                 winner_id=selection["candidate_id"], threshold=selection["threshold"],
                                 dev=assignment.loc[assignment.partition.eq("development"), "pattern_row"].to_numpy(),
                                 test=assignment.loc[assignment.partition.eq("test"), "pattern_row"].to_numpy(),
                                 original_metrics=selection["oof_metrics"], test_metrics=summary["test"],
                                 restored=artifact, final_forest=artifact["forest"],
                                 SCENARIOS=range(7), PARAMS=range(108), summary=summary)
                verification = SOURCE.split("# %% G. ", 1)[1].split("\n", 1)[1]
                exec(compile(verification, "actual_rf_verification_cell", "exec"), namespace)
                audit = json.loads((shared / "postrun_audit.json").read_text(encoding="utf-8"))
                self.assertEqual(audit["status"], "passed")
                self.assertEqual(audit["threshold_rows"], 756756)
                print(f"{dataset}: all values/order equal; verification passed without legacy CSV")

    def test_writer_reader_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            expected = pd.DataFrame(dict(candidate_id=[0, 0, 1], threshold=[0.0, 0.5, 1.0]))
            expected.to_csv(folder / "threshold_search.csv", index=False)
            pd.testing.assert_frame_equal(expected, HELPERS["read_threshold_search"](folder))
            HELPERS["write_threshold_search"](expected, folder)
            pd.testing.assert_frame_equal(expected, HELPERS["read_threshold_search"](folder))
            self.assertEqual(json.loads((folder / "threshold_search_parts.json").read_text())["total_rows"], 3)

    def test_missing_part_and_tampered_part_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            table = pd.DataFrame(dict(candidate_id=[0, 1], threshold=[0.0, 1.0]))
            HELPERS["write_threshold_search"](table, folder)
            first = folder / "threshold_search_part1.csv"
            original = first.read_bytes()
            first.unlink()
            with self.assertRaises(FileNotFoundError):
                HELPERS["read_threshold_search"](folder)
            first.write_bytes(original + b" ")
            with self.assertRaises(AssertionError):
                HELPERS["read_threshold_search"](folder)

    def test_notebooks_match_source_and_compile(self):
        for dataset in ("cn7", "rg3"):
            notebook = nbformat.read(ROOT / "modeling" / f"random_forest_{dataset}_integrated.ipynb",
                                     as_version=4)
            code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
            expected = [section.split("\n", 1)[1].replace(
                'DATASET = os.environ.get("RF_DATASET", "cn7")', f'DATASET = "{dataset}"').rstrip()
                for section in SOURCE.split("# %% ")[1:]]
            self.assertEqual(code, expected)
            for index, cell in enumerate(code):
                compile(cell, f"{dataset}:cell{index}", "exec")


if __name__ == "__main__":
    unittest.main(verbosity=2)
