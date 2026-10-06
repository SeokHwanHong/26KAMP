"""RF-only initial modeling, saved inference and reference distributions.

Uses the existing RF definitions without executing their old output-writing
workflow. Every training run is new; shared code/data/runtime are read-only.
CT, drift alarm policy, promotion and rollback remain outside this module.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import uuid

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "modeling/random_forest_manual.py"
CONSTANTS = {"SEED", "THRESHOLDS", "SCENARIOS", "PARAMS", "GROUPS", "DOMAIN_SPEC", "SCENARIO_GROUPS"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


@lru_cache(maxsize=2)
def definitions(dataset):
    if dataset not in {"cn7", "rg3"}:
        raise ValueError("Unknown dataset")
    tree = ast.parse(LEGACY.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            body.append(node)
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id in CONSTANTS for t in node.targets):
            body.append(node)
    namespace = {"__name__": "rf_legacy_definitions", "__file__": str(LEGACY), "ROOT": ROOT, "DATASET": dataset}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(LEGACY), "exec"), namespace)
    return namespace


def environment():
    return {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}


def protected_snapshot():
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    return {p: sha256(ROOT / p) for p in paths if p and (ROOT / p).is_file()}


def check_protected(snapshot):
    changed = [p for p, expected in snapshot.items() if not (ROOT / p).is_file() or sha256(ROOT / p) != expected]
    if changed:
        raise RuntimeError(f"Protected files changed: {changed}")
    return {"passed": True, "checked_files": len(snapshot), "changed_files": changed, "runtime_unchanged": True}


def rank_search(table):
    return table.sort_values(["f1", "fold_f1_std", "max_features_used", "candidate_id", "threshold"],
                             ascending=[False, True, True, True, True], kind="stable")


def fit_candidate(dataset, candidate_id, params, prepared, ydev, folds, scenario, feature_count):
    lib = definitions(dataset)
    probability = np.full(len(ydev), np.nan)
    seen = np.zeros(len(ydev), dtype=int)
    records = []
    for fold, train_pos, valid_pos, train_values, valid_values in prepared:
        forest = lib["make_forest"](params)
        started = time.perf_counter()
        forest.fit(train_values, ydev[train_pos])
        probability[valid_pos] = lib["risk_probability"](forest, valid_values)
        seen[valid_pos] += 1
        records.append(dict(candidate_id=candidate_id, scenario=scenario, fold=fold,
                            train_rows=len(train_pos), train_risk=int(ydev[train_pos].sum()),
                            validation_rows=len(valid_pos), validation_risk=int(ydev[valid_pos].sum()),
                            n_features=train_values.shape[1], n_trees=len(forest.estimators_),
                            fit_seconds=time.perf_counter()-started))
    if not (seen == 1).all() or not np.isfinite(probability).all():
        raise RuntimeError("Incomplete OOF predictions")
    table = lib["threshold_table"](ydev, folds, probability, scenario, candidate_id, feature_count)
    return candidate_id, probability, table, records


def describe_features(bundle, fold=None, train_rows=None, normal_rows=None):
    return dict(fold=fold, train_rows=train_rows, normal_reference_rows=normal_rows,
                feature_coordinates=bundle.get("feature_coordinates", "normal_standardized_v1"),
                scenario=bundle["scenario"], feature_names=bundle["feature_names"],
                derived_specs=bundle["specs"], skipped_specs=bundle["skipped_specs"],
                pca_groups={k: {"columns": v["columns"], "components": v["pca"].n_components_}
                            for k, v in bundle["pcas"].items()})


def predict(artifact, frame):
    columns = artifact["input_columns"]
    if list(frame.columns) != columns or frame.empty:
        raise ValueError("RF input columns/order or row count invalid")
    if not all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes) or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("RF input must contain finite numeric values")
    lib = definitions(artifact["dataset"])
    probability = lib["risk_probability"](artifact["forest"], lib["transform_features"](artifact["bundle"], frame))
    return probability, (probability > artifact["threshold"]).astype(int)


def load_artifact(folder):
    folder = Path(folder)
    audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise ValueError("Run did not pass audit")
    for name in ("model.joblib", "selection_manifest.json", "run_plan.json", "reference_baseline.json"):
        if sha256(folder / name) != audit["artifact_sha256"][name]:
            raise ValueError(f"Artifact changed after audit: {name}")
    if audit["source_sha256"] != sha256(LEGACY) or audit["pipeline_sha256"] != sha256(__file__):
        raise ValueError("Source changed since run")
    model = joblib.load(folder / "model.joblib")
    selection = json.loads((folder / "selection_manifest.json").read_text(encoding="utf-8"))
    if model["environment"] != environment():
        raise ValueError("Saved model environment differs; validate in its recorded environment")
    if model["threshold"] != selection["threshold"] or model["selection_sha256"] != sha256(folder / "selection_manifest.json"):
        raise ValueError("Selection/threshold mismatch")
    schema = json.loads((ROOT / "data/schema/input_features.json").read_text(encoding="utf-8"))
    if model["schema_sha256"] != sha256(ROOT / "data/schema/input_features.json") or model["input_columns"] != schema["feature_columns"]:
        raise ValueError("Input schema differs")
    return model


def predict_batch(artifact, frame, *, coordinates_confirmed=False, label_source=None):
    if not coordinates_confirmed:
        raise ValueError("Batch coordinates/scale have not been confirmed")
    columns = artifact["input_columns"]
    required = set(columns) | {"Unnamed: 0"}
    if not required.issubset(frame.columns) or set(frame.columns)-required-{"PassOrFail"}:
        raise ValueError("Batch schema invalid")
    if frame["Unnamed: 0"].isna().any() or not frame["Unnamed: 0"].is_unique:
        raise ValueError("Product IDs must be present and unique within batch")
    inputs = frame.loc[:, columns]
    # Validate values before grouping; invalid rows never disappear silently.
    predict(artifact, inputs)
    labeled = "PassOrFail" in frame
    if labeled and (not label_source or not frame.PassOrFail.isin([0, 1]).all()):
        raise ValueError("Labeled batch requires valid labels and a source")
    codes = inputs.groupby(columns, sort=False, dropna=False).ngroup().to_numpy()
    positions = np.flatnonzero(~pd.Series(codes).duplicated().to_numpy())
    patterns = inputs.iloc[positions].reset_index(drop=True)
    probability, prediction = predict(artifact, patterns)
    products = pd.DataFrame(dict(source_row=np.arange(len(frame)), source_id=frame["Unnamed: 0"].to_numpy(),
                                 pattern_row=codes, probability=probability[codes], prediction=prediction[codes],
                                 model_version=artifact["model_version"]))
    pattern_meta = pd.DataFrame(dict(pattern_row=np.arange(len(patterns)),
                                     total_count=np.bincount(codes, minlength=len(patterns)),
                                     probability=probability, prediction=prediction))
    if labeled:
        products["original_label"] = frame.PassOrFail.to_numpy()
        labels = products.groupby("pattern_row", sort=True).original_label.max().to_numpy()
        pattern_meta["representative_label"] = labels
        products["representative_label"] = labels[codes]
    return patterns, pattern_meta, products


def distribution(values, spec=None):
    values = np.asarray(values, dtype=float)
    if spec is None:
        unique = np.unique(values)
        spec = {"kind": "values", "values": unique.tolist()} if len(unique) <= 12 else {
            "kind": "bins", "cuts": np.unique(np.quantile(values, np.linspace(0, 1, 11)[1:-1])).tolist()}
        spec.update(minimum=float(values.min()), maximum=float(values.max()))
    if spec["kind"] == "values":
        reference = np.asarray(spec["values"])
        counts = [int((values == v).sum()) for v in reference]
        counts.append(int((~np.isin(values, reference)).sum()))
    else:
        codes = np.searchsorted(spec["cuts"], values, side="right")
        counts = np.bincount(codes, minlength=len(spec["cuts"])+1).tolist()
    result = {**spec, "proportions": (np.asarray(counts)/len(values)).tolist(),
              "out_of_range_fraction": float(((values < spec["minimum"]) | (values > spec["maximum"])).mean())}
    return result


def category_codes(values, spec):
    if spec["kind"] == "bins":
        return np.searchsorted(spec["cuts"], values, side="right")
    lookup = {v: i for i, v in enumerate(spec["values"])}
    return np.array([lookup.get(v, len(lookup)) for v in values], dtype=int)


def joint_distribution(frame, columns, specs):
    pairs = np.column_stack([category_codes(frame[c].to_numpy(), specs[c]) for c in columns])
    keys, counts = np.unique(pairs, axis=0, return_counts=True)
    return {":".join(str(int(x)) for x in key): int(count)/len(frame) for key, count in zip(keys, counts)}


def build_baseline(artifact, inputs, weights):
    weights = np.asarray(weights, dtype=int)
    if len(weights) != len(inputs) or (weights < 1).any():
        raise ValueError("Positive product counts required")
    expanded = inputs.iloc[np.repeat(np.arange(len(inputs)), weights)].reset_index(drop=True)
    probability, prediction = predict(artifact, expanded)
    features = {c: distribution(expanded[c].to_numpy()) for c in expanded}
    pairs = [["Injection_Time", "Filling_Time"], ["Max_Injection_Pressure", "Max_Switch_Over_Pressure"],
             ["Barrel_Temperature_1", "Mold_Temperature_3"]]
    return dict(dataset=artifact["dataset"], model_version=artifact["model_version"],
                product_rows=len(expanded), unique_patterns=len(inputs), reference="development only; original product frequencies",
                features=features, pairs=[dict(columns=cols, proportions=joint_distribution(expanded, cols, features)) for cols in pairs],
                score=distribution(probability), risk_fraction=float(prediction.mean()),
                coordinate_id=f"{artifact['dataset']}:provided_labeled_coordinates:v1",
                drift_policy="metrics only; no calibrated alert/persistence policy", ct_status="deferred")


def compare_batch(artifact, baseline, frame):
    if baseline["model_version"] != artifact["model_version"] or baseline["dataset"] != artifact["dataset"]:
        raise ValueError("Baseline must use the same dataset and model version")
    probability, prediction = predict(artifact, frame)
    rows = []
    for column, spec in baseline["features"].items():
        current = distribution(frame[column].to_numpy(), spec)
        rows.append(dict(feature=column, tv=0.5*float(np.abs(np.asarray(current["proportions"])-spec["proportions"]).sum()),
                         out_of_range_fraction=current["out_of_range_fraction"]))
    pair_rows = []
    for pair in baseline["pairs"]:
        current = joint_distribution(frame, pair["columns"], baseline["features"])
        keys = set(current) | set(pair["proportions"])
        pair_rows.append(dict(columns=pair["columns"], tv=0.5*sum(abs(current.get(k, 0)-pair["proportions"].get(k, 0)) for k in keys)))
    score = distribution(probability, baseline["score"])
    return dict(model_version=artifact["model_version"], product_rows=len(frame), unique_patterns=len(frame.drop_duplicates()),
                feature_changes=rows, pair_changes=pair_rows,
                score_tv=0.5*float(np.abs(np.asarray(score["proportions"])-baseline["score"]["proportions"]).sum()),
                risk_fraction=float(prediction.mean()), risk_fraction_change=float(prediction.mean()-baseline["risk_fraction"]),
                status="metrics_only; drift thresholds/persistence and CT not implemented")


def audit_run(folder):
    folder = Path(folder)
    plan = json.loads((folder / "run_plan.json").read_text(encoding="utf-8"))
    lib = definitions(plan["dataset"])
    X, y, assignment, dev, test, split = lib["load_data"]()
    if split["source_sha256"] != plan["split_source_sha256"] or split["artifact_sha256"] != plan["split_artifact_sha256"]:
        raise AssertionError("Input/split changed")
    table = lib["read_threshold_search"](folder)
    if len(table) != len(plan["scenarios"])*len(plan["params"])*len(lib["THRESHOLDS"]):
        raise AssertionError("Incomplete joint grid")
    np.testing.assert_allclose(table.f1, 2*table.tp/(2*table.tp+table.fp+table.fn), rtol=0, atol=1e-15)
    winner = rank_search(table).iloc[0]
    selection = json.loads((folder / "selection_manifest.json").read_text(encoding="utf-8"))
    assert int(winner.candidate_id) == selection["candidate_id"]
    assert np.isclose(winner.threshold, selection["threshold"], rtol=0, atol=1e-15)
    model = joblib.load(folder / "model.joblib")
    assert model["selection_sha256"] == sha256(folder / "selection_manifest.json")
    assert model["threshold"] == selection["threshold"]
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    for partition, indices in (("oof", dev), ("test", test)):
        saved = pd.read_csv(folder / f"{'selected_oof' if partition == 'oof' else 'test'}_predictions.csv", float_precision="round_trip")
        np.testing.assert_array_equal(saved.pattern_row, indices)
        np.testing.assert_array_equal(saved.label, y[indices])
        np.testing.assert_array_equal(saved.prediction, saved.probability > selection["threshold"])
        recomputed = lib["metrics"](saved.label.to_numpy(), saved.prediction.to_numpy())
        for key, value in recomputed.items():
            assert np.isclose(value, summary[partition][key], rtol=0, atol=1e-15)
        if partition == "test":
            probability, prediction = predict(model, X.iloc[test])
            np.testing.assert_allclose(probability, saved.probability, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(prediction, saved.prediction)
    fold_plan = json.loads((folder / "fold_plan.json").read_text(encoding="utf-8"))
    for fold in fold_plan:
        expected = dev[assignment.iloc[dev].cv_fold.to_numpy() != fold["fold"]]
        np.testing.assert_array_equal(fold["train_pattern_rows"], expected)
        assert not set(fold["train_pattern_rows"]) & (set(fold["validation_pattern_rows"]) | set(test))
    baseline = json.loads((folder / "reference_baseline.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(ROOT / f"data/processed/{plan['dataset']}/conservative/labeled_metadata.csv")
    expected_baseline = build_baseline(model, X.iloc[dev], metadata.iloc[dev].total_count.to_numpy())
    assert baseline == expected_baseline
    report = dict(passed=True, dataset=plan["dataset"], joint_rows=len(table), source_sha256=sha256(LEGACY),
                  pipeline_sha256=sha256(__file__), selection_reproduced=True, test_prediction_reload_equal=True,
                  no_test_in_training=True, reference_frequency_preserved=True,
                  artifact_sha256={p.name: sha256(p) for p in folder.iterdir() if p.is_file() and p.name != "audit.json"})
    write_json(folder / "audit.json", report)
    return report


def run_dataset(dataset, folder, *, jobs=4, scenarios=None, params=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    lib = definitions(dataset)
    X, y, assignment, dev, test, split = lib["load_data"]()
    Xdev, ydev = X.iloc[dev], y[dev]
    folds = assignment.iloc[dev].cv_fold.to_numpy(dtype=int)
    scenarios = list(lib["SCENARIOS"] if scenarios is None else scenarios)
    params = list(lib["PARAMS"] if params is None else params)
    plan = dict(dataset=dataset, scenarios=scenarios, params=params, seed=42, n_estimators=300,
                feature_coordinates="provided_v1", additional_standardization=False,
                threshold_values=lib["THRESHOLDS"].tolist(), environment=environment(), training="normal+risk patterns",
                selection="pooled OOF F1 desc; fold std asc; feature count asc; candidate ID asc; threshold asc",
                source_sha256=sha256(LEGACY), pipeline_sha256=sha256(__file__), test_used_for_selection=False,
                split_source_sha256=split["source_sha256"], split_artifact_sha256=split["artifact_sha256"],
                design="user PNG seven-stage pipeline", cache_reused=False, ct_status="deferred",
                test_status="historical follow-up; previously observed", jobs=jobs)
    write_json(folder / "run_plan.json", plan)
    fold_plan = []
    for fold in range(4):
        fold_plan.append(dict(fold=fold, train_pattern_rows=dev[folds != fold].tolist(),
                              validation_pattern_rows=dev[folds == fold].tolist(), test_pattern_rows=test.tolist()))
    write_json(folder / "fold_plan.json", fold_plan)
    all_tables, all_audits, features, score_cache = [], [], {}, {}
    for sid, scenario in enumerate(scenarios):
        prepared, feature_rows = [], []
        for fold in range(4):
            train_pos, valid_pos = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            bundle, values = lib["fit_features"](Xdev.iloc[train_pos], ydev[train_pos], scenario)
            prepared.append((fold, train_pos, valid_pos, values, lib["transform_features"](bundle, Xdev.iloc[valid_pos])))
            feature_rows.append(describe_features(bundle, fold, len(train_pos), int((ydev[train_pos] == 0).sum())))
        print(f"{dataset}: {scenario} {len(params)} settings x 4 folds, cold fit", flush=True)
        started = time.perf_counter()
        results = Parallel(n_jobs=jobs, prefer="processes", verbose=5)(
            delayed(fit_candidate)(dataset, sid*len(params)+pid, param, prepared, ydev, folds,
                                   scenario, max(len(row["feature_names"]) for row in feature_rows))
            for pid, param in enumerate(params))
        tables, audits = [], []
        scores = np.empty((len(params), len(dev)))
        for cid, probability, table, records in results:
            scores[cid % len(params)] = probability
            tables.append(table)
            audits.extend(records)
        table = pd.concat(tables, ignore_index=True)
        table.to_csv(folder / f"search_{scenario}.csv", index=False)
        np.save(folder / f"oof_probabilities_{scenario}.npy", scores)
        all_tables.append(table)
        all_audits.extend(audits)
        features[scenario] = feature_rows
        score_cache[scenario] = scores
        print(f"{dataset}: {scenario} complete ({time.perf_counter()-started:.1f}s)", flush=True)
    search = pd.concat(all_tables, ignore_index=True)
    lib["write_threshold_search"](search, folder)
    pd.DataFrame(all_audits).to_csv(folder / "forest_audit.csv", index=False)
    write_json(folder / "feature_audit.json", features)
    ranked = rank_search(search)
    winner = ranked.iloc[0]
    cid, threshold = int(winner.candidate_id), float(winner.threshold)
    sid, pid = divmod(cid, len(params))
    scenario = scenarios[sid]
    oof_probability = score_cache[scenario][pid]
    selection = dict(dataset=dataset, scenario=scenario, candidate_id=cid, params=params[pid], threshold=threshold,
                     oof_metrics=lib["metrics"](ydev, oof_probability > threshold),
                     fold_f1_std=float(winner.fold_f1_std), feature_count=int(winner.max_features_used),
                     selection_metric="pooled OOF F1", test_used_for_selection=False, plan_sha256=sha256(folder / "run_plan.json"))
    write_json(folder / "selection_manifest.json", selection)
    ranked.head(20).to_csv(folder / "top20.csv", index=False)
    ranked.groupby("scenario", sort=False).head(1).to_csv(folder / "scenario_best.csv", index=False)
    pd.DataFrame(dict(pattern_row=dev, fold=folds, label=ydev, probability=oof_probability,
                      threshold=threshold, prediction=(oof_probability > threshold).astype(int))).to_csv(
                          folder / "selected_oof_predictions.csv", index=False)
    bundle, values = lib["fit_features"](Xdev, ydev, scenario)
    forest = lib["make_forest"](params[pid]).fit(values, ydev)
    artifact = dict(dataset=dataset, bundle=bundle, forest=forest, threshold=threshold,
                    input_columns=list(X), environment=environment(), schema_sha256=sha256(ROOT / "data/schema/input_features.json"),
                    selection_sha256=sha256(folder / "selection_manifest.json"), model_version=f"{folder.parent.name}-{dataset}")
    joblib.dump(artifact, folder / "model.joblib")
    restored = joblib.load(folder / "model.joblib")
    probability, prediction = predict(restored, X.iloc[test])
    original_probability, original_prediction = predict(artifact, X.iloc[test])
    np.testing.assert_array_equal(probability, original_probability)
    np.testing.assert_array_equal(prediction, original_prediction)
    pd.DataFrame(dict(pattern_row=test, label=y[test], probability=probability,
                      threshold=threshold, prediction=prediction)).to_csv(folder / "test_predictions.csv", index=False)
    test_metrics = lib["metrics"](y[test], prediction)
    test_metrics.update(average_precision=float(average_precision_score(y[test], probability)),
                        roc_auc=float(roc_auc_score(y[test], probability)))
    summary = dict(dataset=dataset, scenario=scenario, params=params[pid], threshold=threshold,
                   oof=selection["oof_metrics"], test=test_metrics, test_status=plan["test_status"])
    write_json(folder / "summary.json", summary)
    write_json(folder / "final_features.json", describe_features(bundle, train_rows=len(dev), normal_rows=int((ydev == 0).sum())))
    pd.DataFrame(dict(feature=bundle["feature_names"], impurity_importance=forest.feature_importances_)).sort_values(
        "impurity_importance", ascending=False).to_csv(folder / "impurity_importance.csv", index=False)
    # Permute original sensor groups together before applying saved transforms.
    # The threshold stays frozen and these diagnostics never select a candidate.
    selected_models = []
    for fold in range(4):
        train_pos, valid_pos = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fold_bundle, train_values = lib["fit_features"](Xdev.iloc[train_pos], ydev[train_pos], scenario)
        fold_forest = lib["make_forest"](params[pid]).fit(train_values, ydev[train_pos])
        fold_probability = lib["risk_probability"](fold_forest, lib["transform_features"](fold_bundle, Xdev.iloc[valid_pos]))
        np.testing.assert_array_equal(fold_probability, oof_probability[valid_pos])
        selected_models.append((fold, valid_pos, fold_bundle, fold_forest))
    repeats = []
    for gid, (group, columns) in enumerate(lib["GROUPS"].items()):
        for repeat in range(20):
            perturbed_probability = np.full(len(dev), np.nan)
            for fold, valid_pos, fold_bundle, fold_forest in selected_models:
                changed = Xdev.iloc[valid_pos].copy()
                permutation = np.random.default_rng(42+gid*1000+repeat*10+fold).permutation(len(valid_pos))
                changed.loc[:, columns] = changed[columns].to_numpy()[permutation]
                perturbed_probability[valid_pos] = lib["risk_probability"](fold_forest, lib["transform_features"](fold_bundle, changed))
            perturbed = lib["metrics"](ydev, perturbed_probability > threshold)
            repeats.append(dict(group=group, repeat=repeat, f1_drop=selection["oof_metrics"]["f1"]-perturbed["f1"],
                                recall_drop=selection["oof_metrics"]["recall"]-perturbed["recall"],
                                fpr_increase=perturbed["fpr"]-selection["oof_metrics"]["fpr"]))
    repeat_table = pd.DataFrame(repeats)
    repeat_table.to_csv(folder / "group_permutation_repeats.csv", index=False)
    repeat_table.groupby("group", sort=False).agg(
        f1_drop_mean=("f1_drop", "mean"), f1_drop_std=("f1_drop", "std"),
        recall_drop_mean=("recall_drop", "mean"), fpr_increase_mean=("fpr_increase", "mean")).to_csv(
            folder / "group_permutation_importance.csv")
    metadata = pd.read_csv(ROOT / f"data/processed/{dataset}/conservative/labeled_metadata.csv")
    baseline = build_baseline(artifact, Xdev, metadata.iloc[dev].total_count.to_numpy())
    write_json(folder / "reference_baseline.json", baseline)
    for partition, rows, preds in (("oof", dev, (oof_probability > threshold).astype(int)), ("test", test, prediction)):
        risk_types = metadata.iloc[rows].assign(label=y[rows], prediction=preds)
        risk_types.loc[risk_types.label == 1].groupby("pattern_type").agg(
            risk_patterns=("label", "size"), detected=("prediction", "sum")).to_csv(folder / f"{partition}_risk_types.csv")
    audit = audit_run(folder)
    print(f"{dataset}: selected {scenario}, threshold={threshold}, OOF F1={summary['oof']['f1']:.6f}, Test F1={test_metrics['f1']:.6f}; audit={audit['passed']}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--dataset", choices=["cn7", "rg3", "both"], default="both")
    train.add_argument("--jobs", type=int, default=4)
    audit = sub.add_parser("audit")
    audit.add_argument("folder", type=Path)
    infer = sub.add_parser("infer")
    infer.add_argument("folder", type=Path)
    infer.add_argument("input", type=Path)
    infer.add_argument("--coordinates-confirmed", action="store_true")
    infer.add_argument("--label-source")
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit_run(args.folder), ensure_ascii=False, indent=2))
    elif args.command == "infer":
        artifact = load_artifact(args.folder)
        frame = pd.read_csv(args.input, float_precision="round_trip")
        patterns, metadata, products = predict_batch(artifact, frame, coordinates_confirmed=args.coordinates_confirmed, label_source=args.label_source)
        folder = args.folder / f"batch-{uuid.uuid4().hex[:12]}"
        folder.mkdir(exist_ok=False)
        patterns.to_csv(folder / "patterns.csv", index=False)
        metadata.to_csv(folder / "pattern_predictions.csv", index=False)
        products.to_csv(folder / "product_predictions.csv", index=False)
        baseline = json.loads((args.folder / "reference_baseline.json").read_text(encoding="utf-8"))
        write_json(folder / "distribution_comparison.json", compare_batch(artifact, baseline, frame[artifact["input_columns"]]))
        write_json(folder / "batch_manifest.json", dict(input_sha256=sha256(args.input), model_version=artifact["model_version"],
                   label_source=args.label_source, coordinates_confirmed=args.coordinates_confirmed, product_rows=len(products), unique_patterns=len(patterns)))
        print(folder)
    else:
        snapshot = protected_snapshot()
        run_id = f"rf-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        folder = ROOT / "output/random_forest_pipeline" / run_id
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / "protected_before.json", snapshot)
        print(f"RUN_FOLDER={folder}", flush=True)
        try:
            datasets = ["cn7", "rg3"] if args.dataset == "both" else [args.dataset]
            summaries = [run_dataset(d, folder / d, jobs=args.jobs) for d in datasets]
            write_json(folder / "summary.json", dict(run_id=run_id, completed=True, datasets=summaries, environment=environment()))
        finally:
            write_json(folder / "preservation.json", check_protected(snapshot))


if __name__ == "__main__":
    # Windows workers must resolve cached helpers from an importable module.
    # Executing functions as __main__ makes the cached definitions unpicklable.
    import rf_pipeline
    rf_pipeline.main()
