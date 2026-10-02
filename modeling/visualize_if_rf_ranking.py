"""Exploratory IF features versus existing RF OOF; no hybrid ranker is fitted."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "if_rf_ranking_visualization_v1"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib_cache"))

import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.ensemble import IsolationForest
import sklearn

font_manager.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                     "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 12,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
COLORS = {"normal_only": "#8BA5BE", "conflicting": "#D66E24", "defect_only": "#9C3C62"}
LABELS = {"normal_only": "정상만 관측", "conflicting": "양품·불량 공존", "defect_only": "불량만 관측"}
PARAMS = dict(n_estimators=200, max_samples="auto", contamination="auto",
              max_features=1.0, bootstrap=False, random_state=42, n_jobs=1)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tie_key(row):
    # Deterministic, label-independent tie breaking; IDs never enter a model.
    return hashlib.sha256(f"seed42:{int(row)}".encode()).hexdigest()


def budget_metrics(frame, score, fraction):
    selected, tp, tie_cut_folds = 0, 0, 0
    for _, group in frame.groupby("fold", sort=True):
        k = int(np.ceil(len(group) * fraction)) if fraction > 0 else 0
        ordered = group.sort_values([score, "tie_key"], ascending=[False, True])
        picked = ordered.iloc[:k]
        selected += k
        tp += int(picked.label.sum())
        if 0 < k < len(ordered):
            tie_cut_folds += int(ordered.iloc[k - 1][score] == ordered.iloc[k][score])
    positives = int(frame.label.sum())
    return dict(requested_pct=100 * fraction, selected=selected,
                selected_pct=100 * selected / len(frame), captured_risk=tp,
                total_risk=positives, missed_risk=positives - tp,
                selected_normal=selected - tp, recall=tp / positives,
                precision=tp / selected if selected else 0., tie_cut_folds=tie_cut_folds)


frames, curves, audit, summaries = {}, {}, {}, {}
for dataset in ("cn7", "rg3"):
    base = ROOT / "data" / "processed" / dataset / "conservative"
    rf_path = ROOT / "output" / f"random_forest_{dataset}" / "manual_seven_scenarios_v1" / "selected_oof_predictions.csv"
    paths = [base / "X_labeled.csv", base / "y_labeled.csv",
             base / "labeled_metadata.csv", base / "splits" / "split_assignments.csv", rf_path]
    X = pd.read_csv(paths[0])
    y = pd.read_csv(paths[1]).iloc[:, 0].astype(int)
    metadata = pd.read_csv(paths[2])
    assignment = pd.read_csv(paths[3])
    rf = pd.read_csv(rf_path)
    assert np.isfinite(X.to_numpy()).all() and X.shape[1] == 24
    assert assignment.pattern_row.tolist() == list(range(len(X)))
    assert metadata.pattern_row.tolist() == list(range(len(X)))
    assert np.array_equal(assignment.label, y)
    assert rf.pattern_row.is_unique
    dev = assignment.loc[assignment.partition.eq("development")].copy()
    assert set(dev.pattern_row) == set(rf.pattern_row)
    frame = rf.merge(dev[["pattern_row", "cv_fold", "label"]], on="pattern_row",
                     validate="one_to_one", suffixes=("", "_assignment"))
    assert np.array_equal(frame.fold, frame.cv_fold)
    assert np.array_equal(frame.label, frame.label_assignment)
    frame = frame.drop(columns=["cv_fold", "label_assignment"])
    frame = frame.merge(metadata[["pattern_row", "pattern_id", "pattern_type", "normal_count", "defect_count"]],
                        on="pattern_row", validate="one_to_one")
    frame["if_score"] = np.nan
    fold_audit = []
    test_rows = set(assignment.loc[assignment.partition.eq("test"), "pattern_row"])
    for fold in range(4):
        train = dev.loc[dev.cv_fold.ne(fold), "pattern_row"].to_numpy()
        valid = dev.loc[dev.cv_fold.eq(fold), "pattern_row"].to_numpy()
        normal = train[y.iloc[train].to_numpy() == 0]
        assert not (set(train) | set(valid)) & test_rows
        assert set(train).isdisjoint(valid)
        keep = X.iloc[normal].nunique().gt(1).to_numpy()
        model = IsolationForest(**PARAMS).fit(X.iloc[normal, keep])
        score = -model.score_samples(X.iloc[valid, keep])
        score_map = dict(zip(valid, score))
        mask = frame.fold.eq(fold)
        frame.loc[mask, "if_score"] = frame.loc[mask, "pattern_row"].map(score_map)
        fold_audit.append(dict(fold=fold, fit_rows=normal.tolist(),
                               validation_rows=valid.tolist(), removed_columns=X.columns[~keep].tolist()))
    assert np.isfinite(frame.if_score).all()
    frame["tie_key"] = frame.pattern_row.map(tie_key)
    frame["rf_rank_within_fold"] = 0
    frame["if_rank_within_fold"] = 0
    for _, group in frame.groupby("fold"):
        for score_col, rank_col in (("probability", "rf_rank_within_fold"), ("if_score", "if_rank_within_fold")):
            ordered_index = group.sort_values([score_col, "tie_key"], ascending=[False, True]).index
            frame.loc[ordered_index, rank_col] = np.arange(1, len(group) + 1)
    rows = []
    for name, score in (("RF", "probability"), ("IF", "if_score")):
        for q in np.linspace(0, 1, 101):
            rows.append(dict(method=name, **budget_metrics(frame, score, float(q))))
    curve = pd.DataFrame(rows)
    for _, group in curve.groupby("method"):
        assert group.captured_risk.is_monotonic_increasing
        assert group.iloc[-1].captured_risk == int(frame.label.sum())
    frame.drop(columns="tie_key").sort_values("pattern_row").to_csv(OUT / f"{dataset}_oof_diagnostic.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(OUT / f"{dataset}_selection_curve.csv", index=False, encoding="utf-8-sig")
    frames[dataset], curves[dataset] = frame, curve
    audit[dataset] = dict(input_hashes={str(p.relative_to(ROOT)): digest(p) for p in paths},
                          development_count=len(frame), risk_count=int(frame.label.sum()), folds=fold_audit)
    summaries[dataset] = dict(count=len(frame), risk=int(frame.label.sum()),
                             pattern_types=frame.pattern_type.value_counts().to_dict(),
                             budgets={str(pct): {name: budget_metrics(frame, score, pct / 100)
                                                for name, score in (("RF", "probability"), ("IF", "if_score"))}
                                      for pct in (5, 10, 20)})


fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.7), sharex=True, sharey=True)
all_if = pd.concat([f.if_score for f in frames.values()])
all_rf = pd.concat([f.probability for f in frames.values()])
xpad = (all_if.max() - all_if.min()) * .06
ymax = max(all_rf.max(), max(f.threshold.iloc[0] for f in frames.values())) * 1.10
for ax, dataset in zip(axes, ("cn7", "rg3")):
    frame = frames[dataset]
    for kind, marker, size in (("normal_only", "o", 24), ("conflicting", "^", 90), ("defect_only", "s", 75)):
        points = frame.loc[frame.pattern_type.eq(kind)]
        if len(points):
            ax.scatter(points.if_score, points.probability, marker=marker, s=size,
                       color=COLORS[kind], alpha=.45 if kind == "normal_only" else .95,
                       linewidths=.6, edgecolors="white", label=f"{LABELS[kind]} ({len(points)}개)",
                       zorder=2 if kind == "normal_only" else 3)
    t = float(frame.threshold.iloc[0])
    ax.axhline(t, color="#374151", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(.98, (t + .025) / (ymax + .025) + .02, f"기존 RF 판정 기준 t={t:.3f}", transform=ax.transAxes, fontsize=10, ha="right")
    ax.set_title(f"{dataset.upper()}  개발 {len(frame)}패턴 · 위험 이력 {int(frame.label.sum())}패턴", pad=14)
    ax.set_xlabel("IF 이상 점수 (-score_samples, 클수록 비전형적)", labelpad=12)
    ax.set_xlim(all_if.min() - xpad, all_if.max() + xpad)
    ax.set_ylim(-.025, ymax)
    ax.grid(alpha=.13)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("기존 RF의 위험 이력 예측 점수", labelpad=10)
fig.suptitle("IF 이상 점수와 기존 RF 점수는 같은 판단인가?", fontsize=20, y=.985)
fig.text(.5, .912, "각 점은 개발 데이터의 미학습 검증 패턴 · IF는 fold별 정상 Train에서만 적합", ha="center", fontsize=11, color="#475569")
fig.text(.5, .028, "탐색 진단: 기존 RF OOF와 새 IF 점수를 비교한 그림이며, IF+RF 또는 순위 학습 결합 모델의 결과가 아닙니다.",
         ha="center", fontsize=11, color="#475569")
fig.subplots_adjust(top=.81, bottom=.18, left=.085, right=.98, wspace=.15)
fig.savefig(OUT / "01_if_vs_rf_oof.png", dpi=180)
plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2), sharex=True, sharey=True)
for ax, dataset in zip(axes, ("cn7", "rg3")):
    frame, curve = frames[dataset], curves[dataset]
    for name, color, line in (("RF", "#245DB6", "-"), ("IF", "#D66E24", "--")):
        series = curve.loc[curve.method.eq(name)]
        ax.step(series.selected_pct, series.recall * 100, where="post", color=color,
                linewidth=2.4, linestyle=line, label=f"{name} 점수순 선택")
    ax.plot([0, 100], [0, 100], color="#7C8592", linestyle=":", linewidth=1.4, label="무작위 선택 기대값")
    ax.axvspan(0, 20, color="#245DB6", alpha=.045)
    ax.set_title(f"{dataset.upper()}  위험 이력 {int(frame.label.sum())}패턴", pad=14)
    ax.set_xlabel("실제 선택한 개발 패턴 비율 (%)", labelpad=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 10, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(alpha=.13)
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    rf10, if10 = summaries[dataset]["budgets"]["10"]["RF"], summaries[dataset]["budgets"]["10"]["IF"]
    ax.text(.03, .98, f"각 fold 상위 10% 선택: 총 {rf10['selected']}패턴\nRF {rf10['captured_risk']}/{rf10['total_risk']} 포착 · IF {if10['captured_risk']}/{if10['total_risk']} 포착",
            transform=ax.transAxes, va="top", fontsize=11)
axes[0].set_ylabel("위험 이력 패턴 포착률 (%)", labelpad=10)
fig.suptitle("검사 대상을 적게 선택해도 위험 패턴이 포함되는가?", fontsize=20, y=.985)
fig.text(.5, .912, "각 validation fold 안에서 순위를 매긴 뒤 선택 수·포착 수를 합산 · 패턴 단위의 오프라인 선택 실험", ha="center", fontsize=11, color="#475569")
fig.text(.5, .057, "동점은 정답과 무관한 seed 42 기반 ID 해시로 결정 · 선택 수는 fold마다 올림 · 결합 모델은 아직 학습하지 않음", ha="center", fontsize=10, color="#475569")
fig.text(.5, .025, "RF는 개발 OOF로 이미 선택된 모델이므로 선택 낙관성이 있음 · Test 미사용 · 제품 단위 검사 효과나 독립 일반화 성능이 아님", ha="center", fontsize=10, color="#475569")
fig.subplots_adjust(top=.81, bottom=.20, left=.085, right=.98, wspace=.15)
fig.savefig(OUT / "02_risk_capture_by_selection.png", dpi=180)
plt.close(fig)

manifest = dict(purpose="Exploratory visualization, not hybrid training or deployment validation",
                sklearn_version=sklearn.__version__, if_parameters=PARAMS, test_used=False,
                rf_refitted=False, ranker_fitted=False, feature_space="provided raw 24 columns, normal-train constant removal, no additional scaler",
                tie_break="descending score then SHA256(seed42:pattern_row), no labels",
                evaluation_unit="unique input patterns with max aggregated label, not individual product quality",
                caveats=["IF score is derived from existing inputs and cannot resolve identical-X label conflict",
                         "IF reference model differs by fold; cross-fold score calibration is not validated",
                         "RF OOF reused after model selection; curves are descriptive diagnostics, not unbiased estimates",
                         "No real batch or temporal validation is claimed"],
                datasets=audit, summary=summaries)
(OUT / "visualization_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
for dataset, info in summaries.items():
    print(dataset.upper(), json.dumps(info["budgets"], ensure_ascii=False))
print("Figures:", str(OUT / "01_if_vs_rf_oof.png"), str(OUT / "02_risk_capture_by_selection.png"))
