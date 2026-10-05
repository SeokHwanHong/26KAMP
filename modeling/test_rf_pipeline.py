"""RF feature, split, batch, baseline, audit and artifact rejection tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

import rf_pipeline as rf


class RFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = rf.definitions("cn7")
        rng = np.random.default_rng(123)
        cols = sum(cls.lib["GROUPS"].values(), [])
        # Use schema order, including sensors that are constant in normals
        # but vary in risk rows, to test reference/fit scope distinctions.
        cols = json.loads((rf.ROOT / "data/schema/input_features.json").read_text(encoding="utf-8"))["feature_columns"]
        cls.X = pd.DataFrame(rng.normal(size=(80, 24)), columns=cols)
        cls.y = np.r_[np.zeros(60, dtype=int), np.ones(20, dtype=int)]
        cls.X.loc[:59, "Mold_Temperature_4"] = 0.
        cls.X.loc[:59, "Barrel_Temperature_6"] = 0.
        cls.bundle, values = cls.lib["fit_features"](cls.X, cls.y, "domain_all")
        cls.forest = cls.lib["make_forest"](cls.lib["PARAMS"][0]).fit(values, cls.y)
        cls.artifact = dict(dataset="cn7", input_columns=cols, bundle=cls.bundle, forest=cls.forest,
                            threshold=.2, model_version="rf-test")

    def test_normal_reference_and_supervised_all_train(self):
        normal = self.X.iloc[:60]
        np.testing.assert_allclose(self.bundle["normal_scaler"].mean_, normal.mean().to_numpy(), rtol=0, atol=1e-14)
        self.assertEqual(self.bundle["normal_scaler"].n_samples_seen_, 60)
        frame = self.lib["feature_frame"](self.bundle, self.X)
        np.testing.assert_allclose(self.bundle["variance"].variances_, frame.var(ddof=0).to_numpy(), rtol=0, atol=1e-14)
        self.assertIn("Mold_Temperature_4", self.bundle["feature_names"])
        np.testing.assert_array_equal(self.forest.classes_, [0, 1])

    def test_all_seven_transforms_and_skipped_sensor_pairs(self):
        for scenario in self.lib["SCENARIOS"]:
            with self.subTest(scenario=scenario):
                bundle, values = self.lib["fit_features"](self.X, self.y, scenario)
                np.testing.assert_array_equal(values, self.lib["transform_features"](bundle, self.X))
                self.assertTrue(np.isfinite(values).all())
                if scenario == "domain_all":
                    self.assertNotIn("금형온도_상대편차", bundle["feature_names"])
                    barrel = next(s for s in bundle["specs"] if s["feature"] == "배럴_상대수준평균")
                    self.assertNotIn("Barrel_Temperature_6", barrel["active_columns"])

    def test_derived_values_against_hand_calculation(self):
        z = pd.DataFrame([[2., -1., 4.], [0., 2., 2.]], columns=["a", "b", "c"])
        specs = [dict(feature="diff", operation="difference", active_columns=["a", "b"]),
                 dict(feature="mean", operation="mean", active_columns=["a", "b", "c"]),
                 dict(feature="std", operation="std", active_columns=["a", "b", "c"])]
        got = self.lib["derived_frame"](z, specs)
        np.testing.assert_array_equal(got["diff"], [3., -2.])
        np.testing.assert_allclose(got["mean"], [5/3, 4/3])
        np.testing.assert_allclose(got["std"], np.std(z.to_numpy(), axis=1, ddof=0))

    def test_threshold_counts_match_brute_force_and_strict_boundary(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        folds = np.repeat(np.arange(4), 2)
        p = np.array([.0, .1, .1, .4, .4, .5, .5, 1.])
        table = self.lib["threshold_table"](y, folds, p, "raw_no_scaler", 0, 24)
        for t in (0., .1, .4, .5, 1.):
            row = table.loc[np.isclose(table.threshold, t)].iloc[0]
            for key, expected in self.lib["metrics"](y, p > t).items():
                if key in row:
                    self.assertAlmostEqual(row[key], expected)
        self.assertEqual(table.iloc[-1].tp, 0)

    def test_pooled_f1_selection_and_ties_ignore_test(self):
        table = pd.DataFrame([dict(f1=.5, fold_f1_std=.2, max_features_used=5, candidate_id=0, threshold=.3),
                              dict(f1=.5, fold_f1_std=.1, max_features_used=20, candidate_id=1, threshold=.5),
                              dict(f1=.5, fold_f1_std=.1, max_features_used=10, candidate_id=2, threshold=.2),
                              dict(f1=.5, fold_f1_std=.1, max_features_used=10, candidate_id=2, threshold=.1)])
        best = rf.rank_search(table).iloc[0]
        self.assertEqual(best.candidate_id, 2)
        self.assertEqual(best.threshold, .1)

    def test_saved_prediction_and_transform_immutable(self):
        before = self.bundle["normal_scaler"].mean_.copy()
        reference = rf.predict(self.artifact, self.X)
        rf.predict(self.artifact, self.X*1000)
        np.testing.assert_array_equal(before, self.bundle["normal_scaler"].mean_)
        artifact = copy.copy(self.artifact)
        artifact["threshold"] = float(reference[0][0])
        self.assertEqual(rf.predict(artifact, self.X.iloc[:1])[1][0], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.joblib"
            joblib.dump(self.artifact, path)
            restored = rf.predict(joblib.load(path), self.X)
            np.testing.assert_array_equal(reference[0], restored[0])
            np.testing.assert_array_equal(reference[1], restored[1])

    def batch(self):
        frame = self.X.iloc[[0, 0, 1]].reset_index(drop=True)
        frame.insert(0, "Unnamed: 0", [10, 11, 12])
        return frame

    def test_batch_conflicts_products_and_prediction_order(self):
        frame = self.batch()
        frame["PassOrFail"] = [0, 1, 0]
        patterns, meta, products = rf.predict_batch(self.artifact, frame, coordinates_confirmed=True, label_source="synthetic test")
        self.assertEqual(len(patterns), 2)
        self.assertEqual(meta.total_count.tolist(), [2, 1])
        self.assertEqual(meta.representative_label.tolist(), [1, 0])
        np.testing.assert_array_equal(patterns.iloc[products.pattern_row].to_numpy(), frame[self.artifact["input_columns"]].to_numpy())
        np.testing.assert_array_equal(products.prediction, rf.predict(self.artifact, frame[self.artifact["input_columns"]])[1])
        self.assertEqual(products.source_id.tolist(), [10, 11, 12])

    def test_unlabeled_keeps_no_labels_and_alignment_required(self):
        with self.assertRaises(ValueError):
            rf.predict_batch(self.artifact, self.batch())
        _, meta, products = rf.predict_batch(self.artifact, self.batch(), coordinates_confirmed=True)
        self.assertNotIn("representative_label", meta)
        self.assertNotIn("original_label", products)

    def test_bad_input_and_ids_rejected(self):
        for kind in ("nan", "inf", "missing", "duplicate_id", "label", "label_source", "string"):
            with self.subTest(kind=kind):
                frame = self.batch()
                if kind in ("nan", "inf"):
                    frame.loc[0, "Injection_Time"] = np.nan if kind == "nan" else np.inf
                elif kind == "missing":
                    frame = frame.drop(columns="Injection_Time")
                elif kind == "duplicate_id":
                    frame["Unnamed: 0"] = [1, 1, 2]
                elif kind == "string":
                    frame["Injection_Time"] = "bad"
                else:
                    frame["PassOrFail"] = [0, 2, 0] if kind == "label" else [0, 1, 0]
                with self.assertRaises(ValueError):
                    rf.predict_batch(self.artifact, frame, coordinates_confirmed=True)

    def test_baseline_frequency_same_version_and_fixed_bins(self):
        reference = self.X.iloc[:2]
        baseline = rf.build_baseline(self.artifact, reference, [3, 1])
        self.assertEqual(baseline["product_rows"], 4)
        spec = baseline["features"]["Injection_Time"]
        self.assertEqual(sorted(spec["proportions"]), [0., .25, .75])
        expanded = reference.iloc[[0, 0, 0, 1]]
        original = copy.deepcopy(baseline)
        same = rf.compare_batch(self.artifact, baseline, expanded)
        self.assertEqual(same["score_tv"], 0.)
        self.assertEqual(max(v["tv"] for v in same["feature_changes"]), 0.)
        rf.compare_batch(self.artifact, baseline, expanded+1000)
        self.assertEqual(baseline, original)
        other = copy.copy(self.artifact)
        other["model_version"] = "other"
        with self.assertRaises(ValueError):
            rf.compare_batch(other, baseline, expanded)

    def test_actual_fixed_folds_no_overlap(self):
        for dataset in ("cn7", "rg3"):
            _, y, assignment, dev, test, _ = rf.definitions(dataset)["load_data"]()
            folds = assignment.iloc[dev].cv_fold.to_numpy()
            seen = []
            for fold in range(4):
                train, valid = dev[folds != fold], dev[folds == fold]
                self.assertFalse(set(train) & (set(valid) | set(test)))
                self.assertEqual(set(y[train]), {0, 1})
                seen.extend(valid)
            self.assertEqual(sorted(seen), sorted(dev))

    def test_cold_runs_audit_and_tamper_rejected(self):
        for dataset in ("cn7", "rg3"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)/dataset
                lib = rf.definitions(dataset)
                rf.run_dataset(dataset, folder, jobs=2, scenarios=["raw_no_scaler", "domain_all"], params=lib["PARAMS"][:1])
                artifact = rf.load_artifact(folder)
                self.assertEqual(artifact["dataset"], dataset)
                audit = json.loads((folder/"audit.json").read_text(encoding="utf-8"))
                self.assertEqual(audit["joint_rows"], 2002)
                with (folder/"model.joblib").open("ab") as handle:
                    handle.write(b"tampered")
                with self.assertRaises(ValueError):
                    rf.load_artifact(folder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
