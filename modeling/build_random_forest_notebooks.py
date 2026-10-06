"""Mechanically copy the self-contained A-G source into two notebooks."""

from pathlib import Path
import nbformat


HERE = Path(__file__).resolve().parent
source = (HERE / "random_forest_manual.py").read_text(encoding="utf-8")
sections = source.split("# %% ")
assert len(sections) == 8, len(sections)
for dataset in ("cn7", "rg3"):
    cells = [nbformat.v4.new_markdown_cell(
        f"# {dataset.upper()} Random Forest — manual seven-scenario experiment\n\n"
        "Fixed pattern split, seven predeclared input scenarios, pooled 4-fold OOF F1 selection, "
        "and post-selection Test assessment. The Test partition was observed in prior work.\n\n"
        "Use the supplied standardized coordinates directly for derived features and PCA. "
        "No additional StandardScaler is fitted; PCA retains mean centering. "
        "New results use manual_seven_scenarios_provided_scale_v2."
    )]
    for section in sections[1:]:
        heading, rest = section.split("\n", 1)
        rest = rest.replace('DATASET = os.environ.get("RF_DATASET", "cn7")',
                            f'DATASET = "{dataset}"')
        cells.append(nbformat.v4.new_markdown_cell(f"## {heading}"))
        cells.append(nbformat.v4.new_code_cell(rest.rstrip()))
    notebook = nbformat.v4.new_notebook(cells=cells,
                                       metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    path = HERE / f"random_forest_{dataset}_integrated.ipynb"
    if path.exists():
        previous = nbformat.read(path, as_version=4)
        for cell, old_cell in zip(notebook.cells, previous.cells):
            if cell.cell_type == old_cell.cell_type and "id" in old_cell:
                cell["id"] = old_cell["id"]
    nbformat.write(notebook, path)
    print(path)
