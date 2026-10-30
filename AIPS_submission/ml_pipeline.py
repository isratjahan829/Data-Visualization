"""Leakage-controlled ML pipeline for the nano-K yield dataset.

Re-runs the model benchmark with
  * outcome-derived columns removed from the predictors (market value, % yield gain, ...),
  * leave-one-extract-out (grouped) outer CV with a grouped inner grid search,
  * fold-internal imputation and scaling (no information from the test fold),
  * SHAP + permutation importance on held-out folds.

Usage
-----
    pip install pandas numpy scikit-learn xgboost shap matplotlib
    python ml_pipeline.py data.csv --target Total_Yield_kg_per_plot --group Plant_Extract

Outputs (in ./ml_outputs): table7_model_performance.csv, oof_predictions.csv,
fig_obs_vs_pred.png, fig_model_comparison.png, fig_shap_beeswarm.png,
fig_permutation_importance.png, run_log.txt (paste into Methods/Results).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import TransformedTargetRegressor  # noqa: F401  (kept for easy extension)
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SEED = 42

# Columns whose value is computed from (or recorded together with) the target.
# Any column whose name matches one of these patterns is dropped from the predictors.
LEAKY_PATTERNS = [
    r"market", r"value", r"yield", r"increase", r"gain", r"fruit_?count", r"fruit_?w",
    r"avg_?wt", r"revenue", r"profit",
]


def parse_mean(x):
    """'115.8 ± 4.2' -> 115.8 ; leaves numbers untouched."""
    if isinstance(x, str):
        m = re.search(r"-?\d+(?:\.\d+)?", x.replace("−", "-"))
        return float(m.group()) if m else np.nan
    return x


def build_models():
    lin = lambda est: Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler()), ("m", est)])
    tree = lambda est: Pipeline([("imp", SimpleImputer()), ("m", est)])
    alphas = np.logspace(-3, 3, 13)
    models = {
        "Mean baseline": (tree(DummyRegressor()), {}),
        "Linear regression": (lin(LinearRegression()), {}),
        "Ridge": (lin(Ridge(random_state=SEED)), {"m__alpha": alphas}),
        "Lasso": (lin(Lasso(max_iter=50000, random_state=SEED)), {"m__alpha": alphas}),
        "Random forest": (tree(RandomForestRegressor(random_state=SEED, n_jobs=-1)),
                          {"m__n_estimators": [100, 300, 500], "m__max_depth": [3, 5, 10, None]}),
        "SVR": (lin(SVR(kernel="rbf")), {"m__C": [1, 10, 100], "m__gamma": ["scale", 0.01, 0.1]}),
        "KNN": (lin(KNeighborsRegressor()), {"m__n_neighbors": [3, 5, 7]}),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = (tree(XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0)),
                             {"m__learning_rate": [0.03, 0.1], "m__n_estimators": [100, 300, 500],
                              "m__max_depth": [2, 4, 6]})
    except ImportError:
        print("xgboost not installed - skipping XGBoost")
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--target", default="Total_Yield_kg_per_plot")
    ap.add_argument("--group", default="Plant_Extract", help="column identifying the plant source")
    ap.add_argument("--features", nargs="*", help="explicit predictor list (overrides auto-selection)")
    ap.add_argument("--drop", nargs="*", default=[], help="extra columns to exclude")
    ap.add_argument("--out", default="ml_outputs")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(exist_ok=True)
    log = []
    say = lambda s: (print(s), log.append(s))

    df = pd.read_csv(a.csv)
    groups = df[a.group].astype(str).values
    num = df.drop(columns=[a.group]).apply(lambda c: c.map(parse_mean)).apply(pd.to_numeric, errors="coerce")
    y = num[a.target].values
    if a.features:
        feats = a.features
    else:
        feats = [c for c in num.columns
                 if c != a.target and c not in a.drop
                 and not any(re.search(p, c, re.I) for p in LEAKY_PATTERNS)
                 and num[c].notna().mean() > 0.5]
    dropped = [c for c in num.columns if c not in feats and c != a.target]
    X = num[feats]
    keep = ~np.isnan(y)
    X, y, groups = X[keep], y[keep], groups[keep]

    say(f"n = {len(y)} observations, {len(feats)} predictors, {len(set(groups))} groups ({a.group})")
    say("Predictors: " + ", ".join(feats))
    say("Excluded (leakage / non-numeric / user): " + ", ".join(dropped))
    within = X.groupby(groups).nunique().max()
    const = [c for c in feats if within[c] <= 1]
    if const:
        say(f"NOTE: {len(const)} predictors are constant within each group "
            f"(effective n for them = {len(set(groups))}): " + ", ".join(const))

    outer = LeaveOneGroupOut()
    n_inner = min(5, len(set(groups)) - 1)
    rows, oof = [], {}
    for name, (pipe, grid) in build_models().items():
        pred = np.full(len(y), np.nan); fold_scores = []
        for tr, te in outer.split(X, y, groups):
            est = pipe
            if grid:
                est = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=n_inner),
                                   scoring="neg_root_mean_squared_error", n_jobs=-1)
                est.fit(X.iloc[tr], y[tr], groups=groups[tr])
            else:
                est.fit(X.iloc[tr], y[tr])
            pred[te] = est.predict(X.iloc[te])
            fold_scores.append(np.sqrt(mean_squared_error(y[te], pred[te])))
        oof[name] = pred
        rows.append({"Model": name,
                     "R2_oof": r2_score(y, pred),
                     "MAE": mean_absolute_error(y, pred),
                     "RMSE": np.sqrt(mean_squared_error(y, pred)),
                     "RMSE_fold_mean": np.mean(fold_scores),
                     "RMSE_fold_sd": np.std(fold_scores, ddof=1)})
        say(f"{name:18s} R2={rows[-1]['R2_oof']:.3f}  MAE={rows[-1]['MAE']:.3f}  "
            f"RMSE={rows[-1]['RMSE']:.3f} (fold {rows[-1]['RMSE_fold_mean']:.3f} ± {rows[-1]['RMSE_fold_sd']:.3f})")

    res = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    res.insert(0, "Rank", range(1, len(res) + 1))
    res.round(3).to_csv(out / "table7_model_performance.csv", index=False)
    pd.DataFrame({"group": groups, "observed": y, **oof}).to_csv(out / "oof_predictions.csv", index=False)

    best = res.loc[res.Model != "Mean baseline", "Model"].iloc[0]
    say(f"Best model: {best}")

    # Fig: model comparison
    r = res[res.Model != "Mean baseline"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(r.Model, r.RMSE_fold_mean, xerr=r.RMSE_fold_sd, color="#4C72B0", capsize=3)
    ax.set_xlabel("RMSE (kg/plot), leave-one-extract-out; mean ± SD across folds")
    fig.tight_layout(); fig.savefig(out / "fig_model_comparison.png", dpi=300); plt.close(fig)

    # Fig: observed vs predicted (out-of-fold)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for g in sorted(set(groups)):
        m = groups == g
        ax.scatter(y[m], oof[best][m], label=g, s=22)
    lo, hi = np.nanmin([y, oof[best]]), np.nanmax([y, oof[best]])
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Observed yield (kg/plot)"); ax.set_ylabel(f"Predicted yield ({best}, out-of-fold)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig_obs_vs_pred.png", dpi=300); plt.close(fig)

    # Interpretation: refit tuned best model and a random forest on held-out folds
    models = build_models()
    shap_vals, X_te_all, perm = [], [], []
    for tr, te in outer.split(X, y, groups):
        pipe, grid = models["Random forest"]
        est = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=n_inner),
                           scoring="neg_root_mean_squared_error", n_jobs=-1).fit(
            X.iloc[tr], y[tr], groups=groups[tr]).best_estimator_
        pi = permutation_importance(est, X.iloc[te], y[te], n_repeats=10, random_state=SEED,
                                    scoring="neg_root_mean_squared_error")
        perm.append(pi.importances_mean)
        try:
            import shap
            Xt = pd.DataFrame(est.named_steps["imp"].transform(X.iloc[te]), columns=feats)
            shap_vals.append(shap.TreeExplainer(est.named_steps["m"]).shap_values(Xt))
            X_te_all.append(Xt)
        except ImportError:
            pass
    perm = np.array(perm)
    order = np.argsort(perm.mean(0))
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(feats) + 1.2))
    ax.barh(np.array(feats)[order], perm.mean(0)[order], xerr=perm.std(0)[order], color="#55A868", capsize=2)
    ax.set_xlabel("Permutation importance (increase in RMSE, held-out folds)")
    fig.tight_layout(); fig.savefig(out / "fig_permutation_importance.png", dpi=300); plt.close(fig)
    if shap_vals:
        import shap
        shap.summary_plot(np.vstack(shap_vals), pd.concat(X_te_all), show=False)
        plt.tight_layout(); plt.savefig(out / "fig_shap_beeswarm.png", dpi=300); plt.close()
    say("Permutation ranking: " + " > ".join(np.array(feats)[order[::-1]][:6]))

    (out / "run_log.txt").write_text("\n".join(log))
    print(f"\nDone. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
