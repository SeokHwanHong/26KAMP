"""Self-contained source for the two Random Forest integrated notebooks.

The notebook builder copies the A-G sections into each notebook; no project
module is imported by the notebooks. Run this source from the repository root
with RF_DATASET=cn7 or RF_DATASET=rg3. Results are a follow-up assessment,
because the fixed Test partition was already observed in earlier work.
"""

# %% A. Environment, fixed plan, and input integrity
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from itertools import product
from pathlib import Path

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import average_precision_score, roc_auc_score


DATASET = os.environ.get("RF_DATASET", "cn7")
assert DATASET in {"cn7", "rg3"}
EXPERIMENT = "manual_seven_scenarios_provided_scale_v2"
SEED = 42
THRESHOLDS = np.linspace(0.0, 1.0, 1001)
SCENARIOS = (
    "raw_no_scaler", "without_cycle", "pressure", "plasticizing",
    "thermal", "domain_all", "group_pca",
)
PARAMS = [
    dict(max_depth=depth, min_samples_leaf=leaf,
         max_features=features, class_weight=weight)
    for depth, leaf, features, weight in product(
        [3, 6, None], [1, 3, 5, 10],
        ["sqrt", 0.5, 1.0], [None, "balanced", "balanced_subsample"]
    )
]
GROUPS = {
    "충전과 전환": ["Injection_Time", "Filling_Time", "Cushion_Position",
                 "Max_Injection_Speed", "Max_Injection_Pressure", "Max_Switch_Over_Pressure"],
    "계량과 가소화": ["Plasticizing_Time", "Plasticizing_Position", "Max_Screw_RPM",
                  "Average_Screw_RPM", "Max_Back_Pressure", "Average_Back_Pressure"],
    "실린더와 호퍼 온도": [f"Barrel_Temperature_{i}" for i in range(1, 7)] + ["Hopper_Temperature"],
    "금형 온도": ["Mold_Temperature_3", "Mold_Temperature_4"],
    "형체와 전체 주기": ["Clamp_Close_Time", "Clamp_Open_Position", "Cycle_Time"],
}
DOMAIN_SPEC = [
    dict(feature="사출_전환압력_제공값차", group="pressure", operation="difference",
         columns=["Max_Injection_Pressure", "Max_Switch_Over_Pressure"]),
    dict(feature="스크루RPM_제공값차", group="plasticizing", operation="difference",
         columns=["Max_Screw_RPM", "Average_Screw_RPM"]),
    dict(feature="배압_제공값차", group="plasticizing", operation="difference",
         columns=["Max_Back_Pressure", "Average_Back_Pressure"]),
    dict(feature="배럴_제공값평균", group="thermal", operation="mean",
         columns=[f"Barrel_Temperature_{i}" for i in range(1, 7)]),
    dict(feature="배럴_제공값센서간산포", group="thermal", operation="std",
         columns=[f"Barrel_Temperature_{i}" for i in range(1, 7)]),
    dict(feature="금형_제공값평균", group="thermal", operation="mean",
         columns=["Mold_Temperature_3", "Mold_Temperature_4"]),
    dict(feature="금형온도_제공값차", group="thermal", operation="difference",
         columns=["Mold_Temperature_3", "Mold_Temperature_4"]),
]


def find_root() -> Path:
    for base in (Path.cwd(), Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()):
        for candidate in (base, *base.parents):
            if (candidate / "data" / "origin").is_dir() and (candidate / "modeling").is_dir():
                return candidate
    raise FileNotFoundError("Project root with data/origin and modeling not found")


ROOT = find_root()
OUT = ROOT / "output" / f"random_forest_{DATASET}" / EXPERIMENT
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_threshold_search(table: pd.DataFrame, folder: Path) -> None:
    """Keep every search row, but publish two Git-friendly CSV files."""
    midpoint = (len(table) + 1) // 2
    parts = []
    for index, chunk in enumerate((table.iloc[:midpoint], table.iloc[midpoint:]), 1):
        path = folder / f"threshold_search_part{index}.csv"
        chunk.to_csv(path, index=False)
        if path.stat().st_size >= 100_000_000:
            raise ValueError(f"CSV part still too large: {path}")
        parts.append(dict(file=path.name, rows=len(chunk), bytes=path.stat().st_size,
                          sha256=sha256(path)))
    manifest = dict(format_version=1, total_rows=len(table),
                    columns=list(table.columns), parts=parts)
    (folder / "threshold_search_parts.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def read_threshold_search(folder: Path) -> pd.DataFrame:
    """Prefer validated split CSVs; accept the legacy single-file result."""
    paths = [folder / f"threshold_search_part{i}.csv" for i in (1, 2)]
    if any(path.exists() for path in paths):
        if not all(path.exists() for path in paths):
            raise FileNotFoundError("Both threshold_search_part1.csv and part2.csv are required")
        manifest = json.loads((folder / "threshold_search_parts.json").read_text(encoding="utf-8"))
        assert manifest["format_version"] == 1
        assert len(manifest["parts"]) == 2
        tables = []
        for path, record in zip(paths, manifest["parts"]):
            assert record["file"] == path.name
            assert record["bytes"] == path.stat().st_size
            assert record["sha256"] == sha256(path), f"Changed CSV part: {path}"
            chunk = pd.read_csv(path)
            assert list(chunk.columns) == manifest["columns"]
            assert len(chunk) == record["rows"]
            tables.append(chunk)
        table = pd.concat(tables, ignore_index=True)
        assert len(table) == manifest["total_rows"]
        return table
    return pd.read_csv(folder / "threshold_search.csv")


def load_data():
    folder = ROOT / "data" / "processed" / DATASET / "conservative"
    split_dir = folder / "splits"
    split_manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    preprocessing_manifest = json.loads((folder / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    for name, expected in split_manifest["source_sha256"].items():
        assert sha256(folder / name) == expected, name
    for name, expected in split_manifest["artifact_sha256"].items():
        assert sha256(split_dir / name) == expected, name
    X = pd.read_csv(folder / "X_labeled.csv", float_precision="round_trip")
    y = pd.read_csv(folder / "y_labeled.csv")["PassOrFail"].to_numpy(dtype=int)
    assignment = pd.read_csv(split_dir / "split_assignments.csv")
    assert list(X.columns) == preprocessing_manifest["feature_columns"] and X.shape[1] == 24
    assert len(X) == len(y) == len(assignment)
    assert np.isfinite(X.to_numpy()).all() and not X.duplicated().any()
    assert np.array_equal(assignment.pattern_row, np.arange(len(X)))
    assert np.array_equal(assignment.label, y) and set(y) == {0, 1}
    assert assignment.pattern_id.is_unique
    dev = assignment.loc[assignment.partition.eq("development"), "pattern_row"].to_numpy()
    test = assignment.loc[assignment.partition.eq("test"), "pattern_row"].to_numpy()
    assert not set(dev) & set(test) and set(dev) | set(test) == set(range(len(X)))
    assert set(assignment.loc[dev, "cv_fold"]) == {0, 1, 2, 3}
    assert assignment.loc[test, "cv_fold"].eq(-1).all()
    expected = {"cn7": (606, 484, 122, 11, 3), "rg3": (591, 472, 119, 20, 5)}[DATASET]
    assert (len(X), len(dev), len(test), int(y[dev].sum()), int(y[test].sum())) == expected
    return X, y, assignment, dev, test, split_manifest


X, y, assignment, dev, test, split_manifest = load_data()
Xdev, ydev = X.iloc[dev], y[dev]
folds = assignment.iloc[dev].cv_fold.to_numpy(dtype=int)
PLAN = dict(
    dataset=DATASET, experiment=EXPERIMENT, seed=SEED, scenarios=list(SCENARIOS),
    feature_coordinates="provided_v1", additional_standardization=False,
    model="RandomForestClassifier", n_estimators=300, criterion="gini", bootstrap=True,
    candidate_settings=len(PARAMS), threshold_values=THRESHOLDS.tolist(),
    selection="pooled OOF F1 desc; fold F1 sample std asc; max feature count asc; candidate ID asc; abs(t) asc; t desc",
    labels="1 = pattern with any original defect observation; not product defect probability",
    test_status="Already observed in earlier OCSVM and raw RF work; post-selection follow-up only",
    split_source_sha256=split_manifest["source_sha256"],
    split_artifact_sha256=split_manifest["artifact_sha256"],
)
plan_path = OUT / "run_plan.json"
if plan_path.exists():
    assert json.loads(plan_path.read_text(encoding="utf-8")) == PLAN
else:
    plan_path.write_text(json.dumps(PLAN, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{DATASET}: {len(Xdev)} development patterns, {int(ydev.sum())} risk; {len(test)} held-out patterns", flush=True)


# %% B. Domain hypotheses and fold-fitted feature transforms
SCENARIO_GROUPS = {
    "raw_no_scaler": (), "without_cycle": (), "pressure": ("pressure",),
    "plasticizing": ("plasticizing",), "thermal": ("thermal",),
    "domain_all": ("pressure", "plasticizing", "thermal"), "group_pca": (),
}


def derived_frame(coordinates: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    result = pd.DataFrame(index=coordinates.index)
    for spec in specs:
        values = coordinates[spec["active_columns"]]
        if spec["operation"] == "difference":
            result[spec["feature"]] = values.iloc[:, 0] - values.iloc[:, 1]
        elif spec["operation"] == "mean":
            result[spec["feature"]] = values.mean(axis=1)
        elif spec["operation"] == "std":
            result[spec["feature"]] = values.std(axis=1, ddof=0)
        else:
            raise ValueError(spec["operation"])
    return result


def feature_frame(bundle: dict, inputs: pd.DataFrame) -> pd.DataFrame:
    inputs = inputs.loc[:, bundle["input_columns"]]
    scenario = bundle["scenario"]
    if scenario == "raw_no_scaler" or scenario == "without_cycle":
        return inputs.loc[:, bundle["base_columns"]].copy()
    if bundle.get("feature_coordinates") == "provided_v1":
        coordinates = inputs
    elif "feature_coordinates" not in bundle and "normal_scaler" in bundle:
        # Read-only compatibility for models fitted before the scale change.
        # Existing forests must receive the coordinates they were trained on.
        coordinates = pd.DataFrame(bundle["normal_scaler"].transform(inputs),
                                   index=inputs.index, columns=inputs.columns)
    else:
        raise ValueError("Unknown RF feature coordinates; retrain the model")
    if scenario == "group_pca":
        result = pd.DataFrame(index=inputs.index)
        for group, item in bundle["pcas"].items():
            values = coordinates[item["columns"]].to_numpy()
            scores = item["pca"].transform(values)
            for j in range(scores.shape[1]):
                result[f"{group}_PC{j+1}"] = scores[:, j]
            if len(item["columns"]) > scores.shape[1]:
                reconstruction = item["pca"].inverse_transform(scores)
                result[f"{group}_재구성RMSE"] = np.sqrt(np.mean((values - reconstruction) ** 2, axis=1))
        return result
    # Unlike OCSVM, RF retains the supplied 24-feature scale for base columns.
    return pd.concat([inputs.loc[:, bundle["base_columns"]],
                      derived_frame(coordinates, bundle["specs"])], axis=1)


def fit_features(Xtrain: pd.DataFrame, ytrain: np.ndarray, scenario: str):
    assert scenario in SCENARIOS and set(ytrain) == {0, 1}
    normal = Xtrain.iloc[np.flatnonzero(ytrain == 0)]
    active = [column for column in Xtrain if normal[column].nunique() > 1]
    base = [column for column in Xtrain if scenario != "without_cycle"
            or column not in GROUPS["형체와 전체 주기"]]
    bundle = dict(scenario=scenario, input_columns=list(Xtrain.columns),
                  base_columns=base, specs=[], skipped_specs=[], pcas={},
                  feature_coordinates="provided_v1")
    # Supplied inputs are already standardized. Keep that scale for derived
    # features and PCA; do not fit another per-sensor StandardScaler.
    for spec in DOMAIN_SPEC:
        if spec["group"] not in SCENARIO_GROUPS[scenario]:
            continue
        cols = [column for column in spec["columns"] if column in active]
        if len(cols) < 2:
            bundle["skipped_specs"].append(dict(feature=spec["feature"], reason="fewer than 2 active sensors"))
        else:
            bundle["specs"].append({**spec, "active_columns": cols})
    if scenario == "group_pca":
        for group, columns in GROUPS.items():
            cols = [column for column in columns if column in active]
            if cols:
                bundle["pcas"][group] = dict(
                    columns=cols,
                    pca=PCA(n_components=min(2, len(cols)), svd_solver="full").fit(normal[cols].to_numpy()),
                )
    frame = feature_frame(bundle, Xtrain)
    bundle["variance"] = VarianceThreshold(0.0).fit(frame)
    bundle["feature_names"] = list(frame.columns[bundle["variance"].get_support()])
    values = bundle["variance"].transform(frame)
    assert values.shape[1] > 0 and np.isfinite(values).all()
    return bundle, values


def transform_features(bundle: dict, inputs: pd.DataFrame) -> np.ndarray:
    values = bundle["variance"].transform(feature_frame(bundle, inputs))
    assert values.shape[1] == len(bundle["feature_names"]) and np.isfinite(values).all()
    return values


# %% C. Forest scoring and fixed OOF ranking
def make_forest(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, criterion="gini", bootstrap=True, max_samples=None,
        min_samples_split=2, ccp_alpha=0.0, oob_score=False, warm_start=False,
        random_state=SEED, n_jobs=1, **params,
    )


def risk_probability(forest: RandomForestClassifier, values: np.ndarray) -> np.ndarray:
    risk_col = np.flatnonzero(forest.classes_ == 1)
    assert len(risk_col) == 1
    p = forest.predict_proba(values)[:, risk_col[0]]
    assert p.shape == (len(values),) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    return p


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth, prediction = np.asarray(truth), np.asarray(prediction, dtype=bool)
    tp = int(((truth == 1) & prediction).sum())
    fp = int(((truth == 0) & prediction).sum())
    fn = int(((truth == 1) & ~prediction).sum())
    tn = int(((truth == 0) & ~prediction).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                fpr=fp / (fp + tn) if fp + tn else 0.0)


def threshold_table(truth: np.ndarray, fold_ids: np.ndarray, p: np.ndarray,
                    scenario: str, candidate_id: int, feature_count: int) -> pd.DataFrame:
    predictions = p[:, None] > THRESHOLDS[None, :]
    positive = truth == 1
    tp = np.sum(predictions & positive[:, None], axis=0)
    fp = np.sum(predictions & ~positive[:, None], axis=0)
    fn = int(positive.sum()) - tp
    tn = int((~positive).sum()) - fp
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0)
    fold_f1, fold_counts = [], {}
    for fold in range(4):
        mask = fold_ids == fold
        pred, target = predictions[mask], positive[mask]
        ftp = np.sum(pred & target[:, None], axis=0)
        ffp = np.sum(pred & ~target[:, None], axis=0)
        ffn = int(target.sum()) - ftp
        ftn = int((~target).sum()) - ffp
        fden = 2 * ftp + ffp + ffn
        ff1 = np.divide(2 * ftp, fden, out=np.zeros_like(ftp, dtype=float), where=fden != 0)
        fold_f1.append(ff1)
        fold_counts[f"fold_{fold}_tp"] = ftp
        fold_counts[f"fold_{fold}_fp"] = ffp
        fold_counts[f"fold_{fold}_fn"] = ffn
        fold_counts[f"fold_{fold}_tn"] = ftn
        fold_counts[f"fold_{fold}_f1"] = ff1
    return pd.DataFrame(dict(
        scenario=scenario, candidate_id=candidate_id, threshold=THRESHOLDS,
        tp=tp, fp=fp, fn=fn, tn=tn, f1=f1,
        fold_f1_std=np.std(fold_f1, axis=0, ddof=1),
        mean_fold_f1=np.mean(fold_f1, axis=0), max_features_used=feature_count,
        **fold_counts,
    ))


# %% D. Seven-scenario 4-fold search and frozen selection
PARALLEL_CANDIDATES = 4  # forest n_jobs remains 1; only independent candidates run concurrently


def fit_candidate(param_id, params, prepared, scenario, scenario_id, scenario_features):
    candidate_id = scenario_id * len(PARAMS) + param_id
    score_row = np.full(len(dev), np.nan)
    seen = np.zeros(len(dev), dtype=int)
    audits = []
    for fold, train_pos, valid_pos, train_values, valid_values in prepared:
        forest = make_forest(params)
        started = time.perf_counter()
        forest.fit(train_values, ydev[train_pos])
        elapsed = time.perf_counter() - started
        score_row[valid_pos] = risk_probability(forest, valid_values)
        seen[valid_pos] += 1
        audits.append(dict(candidate_id=candidate_id, scenario=scenario, fold=fold,
                           train_rows=len(train_pos), train_risk=int(ydev[train_pos].sum()),
                           validation_rows=len(valid_pos), validation_risk=int(ydev[valid_pos].sum()),
                           n_features=train_values.shape[1], n_trees=len(forest.estimators_),
                           max_fitted_depth=max(tree.get_depth() for tree in forest.estimators_),
                           fit_seconds=elapsed))
    assert (seen == 1).all() and np.isfinite(score_row).all()
    table = threshold_table(ydev, folds, score_row, scenario, candidate_id,
                            max(len(item["feature_names"]) for item in scenario_features))
    return param_id, score_row, table, audits


candidate_rows = [dict(candidate_id=scenario_id * len(PARAMS) + param_id,
                       scenario=scenario, param_id=param_id, **params)
                  for scenario_id, scenario in enumerate(SCENARIOS)
                  for param_id, params in enumerate(PARAMS)]
pd.DataFrame(candidate_rows).to_csv(OUT / "candidates.csv", index=False)
all_tables, all_audits, feature_audit = [], [], {}
for scenario_id, scenario in enumerate(SCENARIOS):
    scenario_file = OUT / f"search_{scenario}.csv"
    score_file = OUT / f"oof_probabilities_{scenario}.npy"
    audit_file = OUT / f"forest_audit_{scenario}.csv"
    feature_file = OUT / f"feature_audit_{scenario}.json"
    if all(path.exists() for path in (scenario_file, score_file, audit_file, feature_file)):
        table = pd.read_csv(scenario_file)
        scores = np.load(score_file)
        audit = pd.read_csv(audit_file)
        scenario_features = json.loads(feature_file.read_text(encoding="utf-8"))
        assert len(table) == len(PARAMS) * len(THRESHOLDS)
        assert scores.shape == (len(PARAMS), len(dev)) and np.isfinite(scores).all()
        assert len(audit) == len(PARAMS) * 4
        print(f"{DATASET}: reused completed {scenario}", flush=True)
    else:
        # Feature fitting is independent of forest parameters, but always inside
        # the current fold's train partition. It is reused only within that fold.
        prepared, scenario_features = [], []
        for fold in range(4):
            train_pos = np.flatnonzero(folds != fold)
            valid_pos = np.flatnonzero(folds == fold)
            assert set(ydev[train_pos]) == set(ydev[valid_pos]) == {0, 1}
            bundle, train_values = fit_features(Xdev.iloc[train_pos], ydev[train_pos], scenario)
            valid_values = transform_features(bundle, Xdev.iloc[valid_pos])
            prepared.append((fold, train_pos, valid_pos, train_values, valid_values))
            scenario_features.append(dict(fold=fold, feature_names=bundle["feature_names"],
                                          skipped_specs=bundle["skipped_specs"],
                                          normal_fit_rows=int((ydev[train_pos] == 0).sum()),
                                          transform_fit_rows=len(train_pos)))
        results = Parallel(n_jobs=PARALLEL_CANDIDATES, prefer="processes", verbose=5)(
            delayed(fit_candidate)(param_id, params, prepared, scenario, scenario_id, scenario_features)
            for param_id, params in enumerate(PARAMS)
        )
        scores = np.full((len(PARAMS), len(dev)), np.nan)
        tables, audits = [], []
        for param_id, score_row, candidate_table, candidate_audit in results:
            scores[param_id] = score_row
            tables.append(candidate_table)
            audits.extend(candidate_audit)
        table = pd.concat(tables, ignore_index=True)
        audit = pd.DataFrame(audits)
        table.to_csv(scenario_file, index=False)
        np.save(score_file, scores)
        audit.to_csv(audit_file, index=False)
        feature_file.write_text(json.dumps(scenario_features, ensure_ascii=False, indent=2), encoding="utf-8")
    all_tables.append(table)
    all_audits.append(audit)
    feature_audit[scenario] = scenario_features

search = pd.concat(all_tables, ignore_index=True)
audit = pd.concat(all_audits, ignore_index=True)
assert len(search) == len(SCENARIOS) * len(PARAMS) * len(THRESHOLDS)
assert search.candidate_id.nunique() == len(SCENARIOS) * len(PARAMS)
ranked = search.sort_values(
    ["f1", "fold_f1_std", "max_features_used", "candidate_id", "threshold"],
    ascending=[False, True, True, True, True], kind="stable",
)
winner = ranked.iloc[0]
winner_id = int(winner.candidate_id)
winner_scenario_id, winner_param_id = divmod(winner_id, len(PARAMS))
winner_scenario = SCENARIOS[winner_scenario_id]
winner_params = PARAMS[winner_param_id]
threshold = float(winner.threshold)
winner_p = np.load(OUT / f"oof_probabilities_{winner_scenario}.npy")[winner_param_id]
assert len(winner_p) == len(dev)
write_threshold_search(search, OUT)
audit.to_csv(OUT / "forest_audit.csv", index=False)
ranked.head(20).to_csv(OUT / "top20.csv", index=False)
(OUT / "feature_audit.json").write_text(json.dumps(feature_audit, ensure_ascii=False, indent=2), encoding="utf-8")
selection = dict(
    dataset=DATASET, experiment=EXPERIMENT, scenario=winner_scenario,
    candidate_id=winner_id, param_id=winner_param_id, params=winner_params,
    threshold=threshold, oof_metrics=metrics(ydev, winner_p > threshold),
    fold_f1_std=float(winner.fold_f1_std), max_features_used=int(winner.max_features_used),
    scenario_count=len(SCENARIOS), setting_count=len(SCENARIOS) * len(PARAMS),
    threshold_grid_count=len(THRESHOLDS), plan_sha256=sha256(plan_path),
    candidate_parallel_jobs=PARALLEL_CANDIDATES, forest_n_jobs=1,
    python=platform.python_version(), sklearn=sklearn.__version__,
    test_used_for_selection=False,
    test_status="Already observed in earlier work; subsequent assessment is not an independent holdout",
)
(OUT / "selection_manifest.json").write_text(
    json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
oof = pd.DataFrame(dict(pattern_row=dev, fold=folds, label=ydev,
                        probability=winner_p, threshold=threshold,
                        prediction=(winner_p > threshold).astype(int)))
oof.to_csv(OUT / "selected_oof_predictions.csv", index=False)
fold_metrics = pd.DataFrame([dict(fold=k, **metrics(ydev[folds == k], winner_p[folds == k] > threshold))
                             for k in range(4)])
fold_metrics.to_csv(OUT / "selected_fold_metrics.csv", index=False)
print(f"{DATASET}: frozen {winner_scenario}, candidate {winner_id}, t={threshold:.3f}, OOF F1={selection['oof_metrics']['f1']:.4f}", flush=True)


# %% E. Refit on all development patterns, then follow-up Test assessment
final_bundle, final_values = fit_features(Xdev, ydev, winner_scenario)
final_forest = make_forest(winner_params)
final_forest.fit(final_values, ydev)
artifact = dict(bundle=final_bundle, forest=final_forest, threshold=threshold,
                input_columns=list(X.columns), label_1="pattern_has_defect_history",
                selection_manifest_sha256=sha256(OUT / "selection_manifest.json"))
joblib.dump(artifact, OUT / "model.joblib")
restored = joblib.load(OUT / "model.joblib")
Xtest, ytest = X.iloc[test], y[test]
test_p = risk_probability(restored["forest"], transform_features(restored["bundle"], Xtest))
assert np.array_equal(test_p, risk_probability(final_forest, transform_features(final_bundle, Xtest)))
test_prediction = test_p > threshold
pd.DataFrame(dict(pattern_row=test, label=ytest, probability=test_p,
                  threshold=threshold, prediction=test_prediction.astype(int)))\
    .to_csv(OUT / "test_predictions.csv", index=False)
test_metrics = metrics(ytest, test_prediction)
test_metrics["average_precision"] = float(average_precision_score(ytest, test_p))
test_metrics["roc_auc"] = float(roc_auc_score(ytest, test_p))
summary = dict(dataset=DATASET, scenario=winner_scenario, params=winner_params,
               threshold=threshold, oof=selection["oof_metrics"], test=test_metrics,
               all_normal_test=metrics(ytest, np.zeros(len(test), dtype=bool)),
               all_risk_test=metrics(ytest, np.ones(len(test), dtype=bool)),
               test_status=selection["test_status"])
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


# %% F. Fixed-threshold importance and failure diagnostics
pd.DataFrame(dict(feature=final_bundle["feature_names"],
                  impurity_importance=final_forest.feature_importances_))\
    .sort_values("impurity_importance", ascending=False)\
    .to_csv(OUT / "impurity_importance.csv", index=False)
original_metrics = metrics(ydev, winner_p > threshold)
group_rows, repeat_rows = [], []
selected_fold_models = []
for fold in range(4):
    train_pos = np.flatnonzero(folds != fold)
    valid_pos = np.flatnonzero(folds == fold)
    bundle, train_values = fit_features(Xdev.iloc[train_pos], ydev[train_pos], winner_scenario)
    forest = make_forest(winner_params).fit(train_values, ydev[train_pos])
    assert np.array_equal(risk_probability(forest, transform_features(bundle, Xdev.iloc[valid_pos])),
                          winner_p[valid_pos])
    selected_fold_models.append((fold, valid_pos, bundle, forest))
for group_id, (group, columns) in enumerate(GROUPS.items()):
    active_columns = [column for column in columns if column in Xdev]
    if not active_columns:
        continue
    for repeat in range(20):
        permuted_p = np.full(len(dev), np.nan)
        for fold, valid_pos, bundle, forest in selected_fold_models:
            perturbed = Xdev.iloc[valid_pos].copy()
            rng = np.random.default_rng(SEED + group_id * 1000 + repeat * 10 + fold)
            shuffled = rng.permutation(len(valid_pos))
            perturbed.loc[:, active_columns] = perturbed.loc[:, active_columns].to_numpy()[shuffled]
            permuted_p[valid_pos] = risk_probability(forest, transform_features(bundle, perturbed))
        assert np.isfinite(permuted_p).all()
        current = metrics(ydev, permuted_p > threshold)
        repeat_rows.append(dict(group=group, repeat=repeat,
                                f1_drop=original_metrics["f1"] - current["f1"],
                                recall_drop=original_metrics["recall"] - current["recall"],
                                fpr_increase=current["fpr"] - original_metrics["fpr"],
                                **current))
repeat_table = pd.DataFrame(repeat_rows)
repeat_table.to_csv(OUT / "group_permutation_repeats.csv", index=False)
for group, subset in repeat_table.groupby("group", sort=False):
    group_rows.append(dict(group=group, repeats=len(subset),
                           f1_drop_mean=subset.f1_drop.mean(),
                           f1_drop_std=subset.f1_drop.std(ddof=1),
                           recall_drop_mean=subset.recall_drop.mean(),
                           fpr_increase_mean=subset.fpr_increase.mean()))
pd.DataFrame(group_rows).to_csv(OUT / "group_permutation_importance.csv", index=False)

test_failed = pd.read_csv(OUT / "test_predictions.csv")
test_failed = test_failed.loc[test_failed.label.ne(test_failed.prediction)]
test_failed.to_csv(OUT / "test_errors.csv", index=False)
metadata = pd.read_csv(ROOT / "data" / "processed" / DATASET / "conservative" / "labeled_metadata.csv")
risk_types = []
for partition, predictions in (("개발 OOF", oof),
                               ("Test 후속 평가", pd.read_csv(OUT / "test_predictions.csv"))):
    joined = predictions.merge(metadata[["pattern_row", "pattern_type"]], on="pattern_row")
    for pattern_type, subset in joined.loc[joined.label.eq(1)].groupby("pattern_type"):
        risk_types.append(dict(partition=partition, pattern_type=pattern_type,
                               risk_patterns=len(subset), detected=int(subset.prediction.sum()),
                               missed=int((subset.prediction == 0).sum())))
risk_type_table = pd.DataFrame(risk_types)
risk_type_table.to_csv(OUT / "risk_pattern_type_breakdown.csv", index=False)
type_lines = ["| 구간 | 위험 패턴 유형 | 개수 | 탐지 | 미탐 |", "|---|---|---:|---:|---:|"]
for row in risk_type_table.itertuples():
    type_lines.append(f"| {row.partition} | {row.pattern_type} | {row.risk_patterns} | "
                      f"{row.detected} | {row.missed} |")
scenario_best = ranked.groupby("scenario", sort=False).head(1).copy()
scenario_best.to_csv(OUT / "scenario_best.csv", index=False)
scenario_lines = ["| 입력 구성 | OOF F1 | Fold F1 표준편차 | 임계값 | TP/FP/FN |",
                  "|---|---:|---:|---:|---:|"]
for row in scenario_best.itertuples():
    scenario_lines.append(f"| {row.scenario} | {row.f1:.4f} | {row.fold_f1_std:.4f} | "
                          f"{row.threshold:.3f} | {row.tp}/{row.fp}/{row.fn} |")
group_importance = pd.DataFrame(group_rows).sort_values("f1_drop_mean", ascending=False)
importance_lines = ["| 공정군 | 평균 OOF F1 감소 | 재현율 감소 | FPR 증가 |",
                    "|---|---:|---:|---:|"]
for row in group_importance.itertuples():
    importance_lines.append(f"| {row.group} | {row.f1_drop_mean:.4f} | "
                            f"{row.recall_drop_mean:.4f} | {row.fpr_increase_mean:.4f} |")
report = f"""# {DATASET.upper()} 랜덤 포레스트: 매뉴얼 기준 7개 입력 구성 비교

고정된 개발 4-fold에서 사전 선언한 입력 구성 7개, 구성당 모델 설정 108개,
설정당 엄격한 `p > t` 임계값 1,001개를 비교했다. 합산 OOF F1과 매뉴얼의
동률 기준으로 **{winner_scenario}** 구성(후보 {winner_id}, 임계값 {threshold:.3f})을 선택했다.
개발 OOF F1은 **{original_metrics['f1']:.6f}**
(TP {original_metrics['tp']}, FP {original_metrics['fp']}, FN {original_metrics['fn']})이었다.
개발 전체로 다시 학습한 뒤 고정된 임계값을 적용한 Test 후속 평가 F1은
**{test_metrics['f1']:.6f}**
(TP {test_metrics['tp']}, FP {test_metrics['fp']}, FN {test_metrics['fn']}, TN {test_metrics['tn']})이었다.

## 입력 구성별 개발 OOF 최상위 후보

{chr(10).join(scenario_lines)}

## 위험 패턴 유형별 탐지와 실패

{chr(10).join(type_lines)}

## 고정 임계값 공정군 순열 진단

{chr(10).join(importance_lines)}

OOF는 후보 선택에 사용된 튜닝 성적이며 독립적인 일반화 성능 추정이 아니다. Test는
앞선 OCSVM 및 원변수 RF 작업에서 이미 관찰한 구간이므로 완전히 미사용인 독립
holdout으로 표현하지 않는다. 이번 입력 구성·모델 설정·임계값 선택에는 Test 지표를
사용하지 않았다. 라벨 1은 동일한 24개 입력 패턴에서 불량이 한 번이라도 관측된
이력을 뜻하며, 개별 제품의 불량 확률이 아니다. 특히 정상·불량이 같은 입력으로
함께 관측된 패턴의 개별 결과는 이 입력만으로 구별할 수 없다. 공정군 순열 중요도는
개발 검증 민감도일 뿐 물리적 공정 단계의 불량 원인 증명이 아니다.
"""
(OUT / "analysis_summary.md").write_text(report, encoding="utf-8")


# %% G. Independent checks of saved search, predictions, and model artifact
saved_search = read_threshold_search(OUT)
assert len(saved_search) == 7 * 108 * 1001
independent_rank = saved_search.sort_values(
    ["f1", "fold_f1_std", "max_features_used", "candidate_id", "threshold"],
    ascending=[False, True, True, True, True], kind="stable",
).iloc[0]
assert int(independent_rank.candidate_id) == winner_id
assert np.isclose(float(independent_rank.threshold), threshold, rtol=0, atol=1e-15)
saved_oof = pd.read_csv(OUT / "selected_oof_predictions.csv")
saved_test = pd.read_csv(OUT / "test_predictions.csv")
assert len(saved_oof) == len(dev) and saved_oof.pattern_row.is_unique
assert set(saved_oof.fold) == {0, 1, 2, 3}
assert np.array_equal(saved_oof.pattern_row, dev)
assert np.array_equal(saved_oof.prediction, saved_oof.probability > threshold)
assert np.array_equal(saved_test.pattern_row, test)
assert np.array_equal(saved_test.prediction, saved_test.probability > threshold)
assert metrics(saved_oof.label.to_numpy(), saved_oof.prediction.to_numpy())["f1"] == original_metrics["f1"]
assert metrics(saved_test.label.to_numpy(), saved_test.prediction.to_numpy())["f1"] == test_metrics["f1"]
assert restored["selection_manifest_sha256"] == sha256(OUT / "selection_manifest.json")
assert len(final_forest.estimators_) == 300
audit_result = dict(status="passed", dataset=DATASET, scenarios=len(SCENARIOS),
                    candidates=len(SCENARIOS) * len(PARAMS),
                    threshold_rows=len(saved_search), oof_rows=len(saved_oof),
                    test_rows=len(saved_test), selection_manifest_sha256=sha256(OUT / "selection_manifest.json"))
(OUT / "postrun_audit.json").write_text(json.dumps(audit_result, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
