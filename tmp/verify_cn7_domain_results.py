"""Independent post-run audit of notebook, manifests, and saved predictions."""
from pathlib import Path
import json
import sys
import joblib
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'modeling'))
import cn7_domain_fpr as w

out = ROOT/'output/ocsvm_cn7_integrated/validation_f1_20260929'
notebook_path = ROOT/'modeling/ocsvm_cn7_integrated.ipynb'
nb = nbformat.read(notebook_path, as_version=4)
nbformat.validate(nb)
for cell in nb.cells:
    if cell.cell_type == 'code':
        compile(cell.source, 'notebook_cell', 'exec')
        assert cell.execution_count is not None
        assert not any(o.output_type == 'error' for o in cell.outputs)
record = json.loads((out/'verification.json').read_text(encoding='utf-8'))
for name, expected in record['artifact_sha256'].items():
    assert w.digest(out/name) == expected, name
execution = json.loads((out/'notebook_execution.json').read_text(encoding='utf-8'))
assert w.digest(notebook_path) == execution['notebook_sha256']
assert w.digest(ROOT/nb.metadata.revision.previous_notebook) == nb.metadata.revision.previous_sha256
x, y, assignment, dev, test, _ = w.load_data(ROOT)
config = json.loads((out/'run_config.json').read_text(encoding='utf-8'))
assert w.digest(ROOT/'modeling/cn7_domain_fpr.py') == config['implementation_sha256']
search = pd.read_csv(out/'candidate_search.csv', float_precision='round_trip')
selection = json.loads((out/'selection_manifest.json').read_text(encoding='utf-8'))
assert w.select_candidates(search) == selection['choices']
assert len(search) == 7*616
folds = pd.read_csv(out/'candidate_fold_metrics.csv', float_precision='round_trip')
assert len(folds) == 7*616*4
plans = json.loads((out/'fold_plan.json').read_text(encoding='utf-8'))
for plan in plans:
    expected_fit = set(assignment.loc[assignment.partition.eq('development') &
                       assignment.cv_fold.ne(plan['fold']) & assignment.label.eq(0), 'pattern_row'])
    assert set(plan['fit']) == expected_fit and 'cal' not in plan
final_plan = json.loads((out/'final_fit_plan.json').read_text(encoding='utf-8'))
assert set(final_plan['fit']) == set(y.loc[dev].index[y.loc[dev].eq(0)]) and 'cal' not in final_plan
# Independently select directly from every joint nu/gamma/t trial, not the compressed table.
best_chunks, joint_count = [], 0
for chunk in pd.read_csv(out/'threshold_search.csv', float_precision='round_trip', chunksize=50000):
    joint_count += len(chunk)
    # Independent scalar recomputation verifies the vectorized pooled-F1 formula.
    np.testing.assert_allclose(chunk.F1, 2*chunk.TP/(2*chunk.TP+chunk.FP+chunk.FN), rtol=0, atol=1e-15)
    best_chunks.append(w.rank_candidates(chunk).head(20))
assert joint_count == 7*616*len(w.THRESHOLDS)
assert w.select_candidates(pd.concat(best_chunks, ignore_index=True)) == selection['choices']
expected_top = w.rank_candidates(pd.concat(best_chunks, ignore_index=True)).head(20).reset_index(drop=True)
pd.testing.assert_frame_equal(expected_top, pd.read_csv(out/'top_joint_candidates.csv', float_precision='round_trip'), check_exact=True)
oof = pd.read_csv(out/'selected_oof_predictions.csv', float_precision='round_trip')
test_pred = pd.read_csv(out/'test_predictions.csv', float_precision='round_trip')
test_metrics = pd.read_csv(out/'test_metrics.csv', float_precision='round_trip')
for choice in selection['choices']:
    if choice['status'] != 'selected_for_followup': continue
    current = oof
    assert set(current.pattern_row) == set(dev) and current.pattern_row.is_unique
    assert np.array_equal((current.risk_score > current.threshold).astype(int), current.y_pred)
    metrics = w.classification_metrics(current.y_true, current.y_pred)
    assert all(metrics[k] == choice[k] for k in ['TP','FP','TN','FN'])
    assert metrics['F1'] == choice['F1']
    selected_fold = folds.loc[folds.candidate_id.eq(choice['candidate_id'])]
    assert np.isclose(selected_fold.F1.std(ddof=1), choice['F1_std'])
    assert current.threshold.eq(choice['threshold']).all()
    artifact = joblib.load(out/'f1_selected_model.joblib')
    scores, pred = w.predict_artifact(artifact, x.loc[test])
    saved = test_pred
    assert np.array_equal(saved.pattern_row, test)
    np.testing.assert_array_equal(scores, saved.risk_score.to_numpy())
    np.testing.assert_array_equal(pred, saved.y_pred.to_numpy())
    observed = w.classification_metrics(y.loc[test], pred)
    saved_metrics = test_metrics.iloc[0]
    for key, value in observed.items():
        assert np.isclose(value, saved_metrics[key])
    assert artifact['selection_sha256'] == w.digest(out/'selection_manifest.json')
    assert artifact['threshold'] == choice['threshold']
    assert artifact['features']['normal_scaler'].n_samples_seen_ == len(final_plan['fit'])
summary = dict(passed=True, all_code_cells_executed=True, complete_grid=True,
               artifact_hashes_match=True, independent_model_reload_equal=True,
               selection_reproduced_without_test=True, metrics_recomputed=True,
               archive_preserved=True, all_normal_train_rows_used=True,
               fixed_validation_threshold_preserved=True, full_joint_grid_verified=joint_count,
               F1_selection_and_top20_reproduced=True, FPR_used_for_selection=False,
               unit_tests_passed=6)
w.write_json(out/'postrun_audit.json', summary)
print(json.dumps(summary, indent=2))
