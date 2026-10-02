"""Reproducible raw-feature Random Forest experiment for the fixed KAMP folds.

Run from the repository root with:
    .venv/Scripts/python.exe modeling/random_forest_baseline.py cn7
    .venv/Scripts/python.exe modeling/random_forest_baseline.py rg3

This is the manual's first, raw_no_scaler input scenario. It does not search
domain-derived feature sets. Test is scored only after the development search
and its frozen selection manifest have been saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = np.linspace(0.0, 1.0, 1001)
EXPERIMENT = "raw_no_scaler_grid_v1"
CANDIDATES = [
    dict(max_depth=depth, min_samples_leaf=leaf,
         max_features=features, class_weight=weight)
    for depth, leaf, features, weight in product(
        [3, 6, None], [1, 3, 5, 10],
        ["sqrt", 0.5, 1.0], [None, "balanced", "balanced_subsample"]
    )
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(dataset: str):
    folder = ROOT / "data" / "processed" / dataset / "conservative"
    split_dir = folder / "splits"
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    pre = json.loads((folder / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["source_sha256"].items():
        assert sha256(folder / filename) == expected, filename
    for filename, expected in manifest["artifact_sha256"].items():
        assert sha256(split_dir / filename) == expected, filename
    X = pd.read_csv(folder / "X_labeled.csv", float_precision="round_trip")
    y = pd.read_csv(folder / "y_labeled.csv")["PassOrFail"].to_numpy()
    assignment = pd.read_csv(split_dir / "split_assignments.csv")
    assert list(X.columns) == pre["feature_columns"] and X.shape[1] == 24
    assert len(X) == len(y) == len(assignment)
    assert np.isfinite(X.to_numpy()).all() and not X.duplicated().any()
    assert np.array_equal(assignment.pattern_row, np.arange(len(X)))
    assert np.array_equal(assignment.label, y) and set(y) == {0, 1}
    assert assignment.pattern_id.is_unique
    dev = assignment.loc[assignment.partition.eq("development"), "pattern_row"].to_numpy()
    test = assignment.loc[assignment.partition.eq("test"), "pattern_row"].to_numpy()
    assert not set(dev) & set(test)
    assert set(dev) | set(test) == set(range(len(X)))
    assert set(assignment.loc[dev, "cv_fold"]) == {0, 1, 2, 3}
    assert assignment.loc[test, "cv_fold"].eq(-1).all()
    return X, y, assignment, dev, test, manifest


def make_model(params: dict) -> Pipeline:
    return Pipeline([
        ("constant", VarianceThreshold(threshold=0.0)),
        ("rf", RandomForestClassifier(
            n_estimators=300, criterion="gini", bootstrap=True,
            max_samples=None, min_samples_split=2, ccp_alpha=0.0,
            oob_score=False, warm_start=False, random_state=42,
            n_jobs=1, **params,
        )),
    ])


def probability_of_risk(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classes = model.named_steps["rf"].classes_
    col = np.flatnonzero(classes == 1)
    assert len(col) == 1
    p = model.predict_proba(X)[:, col[0]]
    assert p.shape == (len(X),) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    return p


def counts(y: np.ndarray, prediction: np.ndarray) -> dict:
    tp = int(np.sum((y == 1) & prediction))
    fp = int(np.sum((y == 0) & prediction))
    fn = int(np.sum((y == 1) & ~prediction))
    tn = int(np.sum((y == 0) & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                precision=precision, recall=recall,
                fpr=fp / (fp + tn) if fp + tn else 0.0)


def threshold_table(y: np.ndarray, folds: np.ndarray, p: np.ndarray, candidate_id: int,
                    max_features_used: int) -> pd.DataFrame:
    predictions = p[:, None] > THRESHOLDS[None, :]
    positive = y == 1
    tp = np.sum(predictions & positive[:, None], axis=0)
    fp = np.sum(predictions & ~positive[:, None], axis=0)
    fn = int(positive.sum()) - tp
    tn = int((~positive).sum()) - fp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
    fold_f1 = []
    for fold in range(4):
        mask = folds == fold
        yp = positive[mask]
        pp = predictions[mask]
        ftp = np.sum(pp & yp[:, None], axis=0)
        ffp = np.sum(pp & ~yp[:, None], axis=0)
        ffn = int(yp.sum()) - ftp
        den = 2 * ftp + ffp + ffn
        fold_f1.append(np.divide(2 * ftp, den, out=np.zeros_like(ftp, dtype=float), where=den != 0))
    return pd.DataFrame(dict(
        candidate_id=candidate_id, threshold=THRESHOLDS, tp=tp, fp=fp, fn=fn, tn=tn,
        f1=f1, fold_f1_std=np.std(fold_f1, axis=0, ddof=1),
        max_features_used=max_features_used,
        **{f"fold_{k}_f1": fold_f1[k] for k in range(4)},
    ))


def run(dataset: str):
    X, y, assignment, dev, test, split_manifest = load_data(dataset)
    output = ROOT / "output" / f"random_forest_{dataset}" / EXPERIMENT
    output.mkdir(parents=True, exist_ok=True)
    X_dev, y_dev = X.iloc[dev], y[dev]
    folds = assignment.iloc[dev].cv_fold.to_numpy()
    assert len(np.unique(dev)) == len(dev) and set(folds) == {0, 1, 2, 3}
    assert set(y_dev) == {0, 1}
    print(f"{dataset}: development={len(dev)}, risk={sum(y_dev)}, test={len(test)}", flush=True)

    tables, audits, selected_oof_cache = [], [], {}
    for candidate_id, params in enumerate(CANDIDATES):
        p = np.full(len(dev), np.nan)
        seen = np.zeros(len(dev), dtype=int)
        feature_counts = []
        for fold in range(4):
            train_pos = np.flatnonzero(folds != fold)
            valid_pos = np.flatnonzero(folds == fold)
            assert set(y_dev[train_pos]) == set(y_dev[valid_pos]) == {0, 1}
            model = make_model(params)
            started = time.perf_counter()
            model.fit(X_dev.iloc[train_pos], y_dev[train_pos])
            elapsed = time.perf_counter() - started
            p[valid_pos] = probability_of_risk(model, X_dev.iloc[valid_pos])
            seen[valid_pos] += 1
            forest = model.named_steps["rf"]
            n_features = int(model.named_steps["constant"].get_support().sum())
            feature_counts.append(n_features)
            audits.append(dict(candidate_id=candidate_id, fold=fold,
                               train_rows=len(train_pos), train_risk=int(y_dev[train_pos].sum()),
                               validation_rows=len(valid_pos), validation_risk=int(y_dev[valid_pos].sum()),
                               n_features=n_features, n_trees=len(forest.estimators_),
                               max_fitted_depth=max(tree.get_depth() for tree in forest.estimators_),
                               fit_seconds=elapsed))
        assert (seen == 1).all() and np.isfinite(p).all()
        selected_oof_cache[candidate_id] = p
        tables.append(threshold_table(y_dev, folds, p, candidate_id, max(feature_counts)))
        if (candidate_id + 1) % 12 == 0:
            print(f"{dataset}: {candidate_id + 1}/{len(CANDIDATES)} candidates", flush=True)

    search = pd.concat(tables, ignore_index=True)
    ranked = search.sort_values(
        ["f1", "fold_f1_std", "max_features_used", "candidate_id", "threshold"],
        # All thresholds are nonnegative, so ascending t implements the
        # manual's abs(t) ascending tie-breaker; the later descending t
        # tie-breaker cannot differ within this grid.
        ascending=[False, True, True, True, True], kind="stable",
    )
    winner = ranked.iloc[0]
    cid, threshold = int(winner.candidate_id), float(winner.threshold)
    params = CANDIDATES[cid]
    selection = dict(
        dataset=dataset, experiment=EXPERIMENT, input_scenario="raw_no_scaler",
        training_mode="supervised", label_1="pattern_has_defect_history",
        candidate_id=cid, candidate_count=len(CANDIDATES),
        threshold_grid_count=len(THRESHOLDS), threshold=threshold,
        params=params, oof_metrics=counts(y_dev, selected_oof_cache[cid] > threshold),
        oof_fold_f1_std=float(winner.fold_f1_std),
        split_source_sha256=split_manifest["source_sha256"],
        split_artifact_sha256=split_manifest["artifact_sha256"],
        python=platform.python_version(), sklearn=sklearn.__version__,
        note="Test not used for candidate or threshold selection; OOF is tuned, not independent.",
    )
    search.to_csv(output / "threshold_search.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "forest_audit.csv", index=False)
    pd.DataFrame([dict(candidate_id=i, **p) for i, p in enumerate(CANDIDATES)]).to_csv(
        output / "candidates.csv", index=False)
    ranked.head(20).to_csv(output / "top20.csv", index=False)
    (output / "selection_manifest.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    oof = pd.DataFrame(dict(pattern_row=dev, fold=folds, label=y_dev,
                            probability=selected_oof_cache[cid], threshold=threshold,
                            prediction=(selected_oof_cache[cid] > threshold).astype(int)))
    oof.to_csv(output / "selected_oof_predictions.csv", index=False)
    print(f"{dataset}: frozen OOF winner {cid}, t={threshold:.3f}, F1={selection['oof_metrics']['f1']:.4f}", flush=True)

    # Only this section can access the held-out labels for performance assessment.
    model = make_model(params)
    model.fit(X_dev, y_dev)
    joblib.dump(dict(model=model, threshold=threshold, columns=list(X.columns),
                     selection_manifest_sha256=sha256(output / "selection_manifest.json")),
                output / "model.joblib")
    restored = joblib.load(output / "model.joblib")
    X_test, y_test = X.iloc[test], y[test]
    p_test = probability_of_risk(restored["model"], X_test)
    assert np.array_equal(p_test, probability_of_risk(model, X_test))
    test_predictions = pd.DataFrame(dict(pattern_row=test, label=y_test,
                                         probability=p_test, threshold=threshold,
                                         prediction=(p_test > threshold).astype(int)))
    test_predictions.to_csv(output / "test_predictions.csv", index=False)
    test_metrics = counts(y_test, p_test > threshold)
    test_metrics["average_precision"] = float(average_precision_score(y_test, p_test))
    test_metrics["roc_auc"] = float(roc_auc_score(y_test, p_test))
    feature_names = np.asarray(X.columns)[model.named_steps["constant"].get_support()]
    pd.DataFrame(dict(feature=feature_names,
                      impurity_importance=model.named_steps["rf"].feature_importances_))\
        .sort_values("impurity_importance", ascending=False)\
        .to_csv(output / "impurity_importance.csv", index=False)
    summary = dict(dataset=dataset, experiment=EXPERIMENT,
                   oof=selection["oof_metrics"], test=test_metrics,
                   all_normal_test=counts(y_test, np.zeros(len(test), dtype=bool)),
                   all_risk_test=counts(y_test, np.ones(len(test), dtype=bool)),
                   selection_manifest_sha256=sha256(output / "selection_manifest.json"),
                   test_status="follow-up; OCSVM work had already observed this test partition")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["cn7", "rg3"])
    args = parser.parse_args()
    run(args.dataset)
