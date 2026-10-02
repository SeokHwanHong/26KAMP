"""Execute one integrated notebook's code cells in a fresh Python interpreter.

This verifies the notebook is self-contained without requiring a Jupyter UI.
Completed search scenarios are loaded from their immutable, audited caches.
"""

import argparse
from pathlib import Path

import nbformat


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=("cn7", "rg3"))
args = parser.parse_args()
path = Path(__file__).resolve().parent / f"random_forest_{args.dataset}_integrated.ipynb"
notebook = nbformat.read(path, as_version=4)
namespace = {"__name__": "__main__"}
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        print(f"Executing code cell {index}", flush=True)
        exec(compile(cell.source, f"{path.name}:cell{index}", "exec"), namespace)
print(f"Verified {path.name}", flush=True)
