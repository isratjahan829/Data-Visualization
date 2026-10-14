"""End-to-end analysis for the manuscript
"AI as a Substitute for Classroom Teachers in Higher Education:
A Soft Computing-Based Feasibility Study".

    python run_pipeline.py            # full run (about 20 min on 4 CPU cores)
    python run_pipeline.py --fast     # 5-fold x 1 CV and 100 Monte Carlo draws

Every number, table and figure in the manuscript is written by this script to
results/tables, results/figures and results/summary.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import figures
from config import (ACTIVITIES, CATEGORIES, DEMOGRAPHICS, FACTOR_ORDER, FACTORS,
                    FIGURES, REVERSED, SEED, T_HIGH, T_LOW, TABLES, TARGET,
                    TARGET_ITEMS)
from data_prep import build_sample, load_raw, save_processed
from fuzzy_system import (ActivityFIS, HierarchicalFIS, final_rule_base,
                          rule_table, trust_rule_base, mean_rule_base)
from models import corrected_ttest, model_zoo, repeated_cv
from psychometrics import CFA, cronbach_alpha, htmt

SUMMARY: dict = {}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def categorize(score):
    return np.where(score >= T_HIGH, CATEGORIES[2],
                    np.where(score >= T_LOW, CATEGORIES[1], CATEGORIES[0]))


# --------------------------------------------------------------------------- #
# 1. Sample and descriptives
# --------------------------------------------------------------------------- #
def describe_sample(raw, d, flow):
    SUMMARY["flow"] = flow
    rows = []
    for q, mapping in DEMOGRAPHICS.items():
        vc = d[q].map(mapping).value_counts()
        for k, v in vc.items():
            rows.append({"variable": q, "category": k, "n": int(v), "pct": 100 * v / len(d)})
    demo = pd.DataFrame(rows)
    demo.to_csv(TABLES / "table_demographics.csv", index=False)
    SUMMARY["age"] = {"mean": float(d.Q3.mean()), "sd": float(d.Q3.std()),
                      "median": float(d.Q3.median())}
    SUMMARY["n_countries_study"] = int(d.Q4.nunique())
    SUMMARY["n_countries_citizenship"] = int(d.Q1.nunique())
    top = d.Q4.value_counts().head(10)
    SUMMARY["top_countries"] = {k: int(v) for k, v in top.items()}
    SUMMARY["languages"] = {k: int(v) for k, v in d.source.value_counts().items()}
    return demo


# --------------------------------------------------------------------------- #
# 2. Measurement model
# --------------------------------------------------------------------------- #
MEASUREMENT_ALTERNATIVES = {
    "M0 (initial a-priori mapping)": {
        "TE": ["Q19f", "Q19g", "Q26b", "Q26c"], "SE": ["Q26f", "Q26h", "Q27e", "Q27i"],
        "P": ["Q26d", "Q19a", "Q26e"], "AC": ["Q27a", "Q27b", "Q26j", "Q26i"],
        "C": ["Q19b", "Q19c", "Q28f", "Q28g"]},
    "M1": {"TE": ["Q19d", "Q19f", "Q19g"], "SE": ["Q26f", "Q26h", "Q25b", "Q25c"],
           "P": ["Q26d", "Q26a", "Q26e"], "AC": ["Q27a", "Q27b", "Q26j"],
           "C": ["Q19a", "Q19b", "Q19c"]},
    "M2": {"TE": ["Q19d", "Q19f", "Q19g"], "SE": ["Q25b", "Q25c", "Q25d"],
           "P": ["Q26d", "Q26a", "Q26e"], "AC": ["Q27a", "Q27b", "Q26j"],
           "C": ["Q19a", "Q19b", "Q19c"]},
    "M3 (final)": {f: FACTORS[f]["items"] for f in ["TE", "SE", "P", "AC", "C"]},
}


def measurement(raw, scored):
    blocks = {f: FACTORS[f]["items"] for f in FACTOR_ORDER}
    items = [i for b in blocks.values() for i in b]
    cfa = CFA(blocks).fit(scored)
    fit = cfa.fit_indices()
    ct = cfa.construct_table()
    ct["alpha"] = [cronbach_alpha(scored[blocks[f]]) for f in FACTOR_ORDER]
    ct["mean"] = [scored[blocks[f]].mean(axis=1).mean() for f in FACTOR_ORDER]
    ct["sd"] = [scored[blocks[f]].mean(axis=1).std() for f in FACTOR_ORDER]
    ct["n_items"] = [len(blocks[f]) for f in FACTOR_ORDER]
    ct["name"] = [FACTORS[f]["name"] for f in FACTOR_ORDER]
    ct.to_csv(TABLES / "table_reliability_validity.csv")
    cfa.loadings().to_csv(TABLES / "table_cfa_loadings.csv", index=False)
    phi = pd.DataFrame(cfa.Phi, index=FACTOR_ORDER, columns=FACTOR_ORDER)
    phi.to_csv(TABLES / "table_latent_correlations.csv")
    H = htmt(scored[items], blocks)
    H.to_csv(TABLES / "table_htmt.csv")
    # Fornell-Larcker: sqrt(AVE) on the diagonal vs latent correlations
    fl = phi.copy()
    for f in FACTOR_ORDER:
        fl.loc[f, f] = ct.loc[f, "sqrtAVE"]
    fl.to_csv(TABLES / "table_fornell_larcker.csv")
    fl_ok = all(ct.loc[f, "sqrtAVE"] > phi.loc[f].drop(f).abs().max() for f in FACTOR_ORDER)
    # Harman's single-factor test (share of variance of first unrotated component)
    ev = np.linalg.eigvalsh(np.corrcoef(scored[items].to_numpy(float), rowvar=False))[::-1]
    harman = float(ev[0] / ev.sum() * 100)

    # transparency: alternative item mappings that were compared
    alt_rows = []
    base = {f: FACTORS[f]["items"] for f in ["A", "R", "E"]}
    pool = raw[(raw.Q13 == 1) & raw.Q3.between(18, 100)].copy()
    for it in REVERSED:
        pool[it] = 6 - pool[it]
    for name, spec in MEASUREMENT_ALTERNATIVES.items():
        V = {**spec, **base}
        V = {f: V[f] for f in FACTOR_ORDER}
        its = [i for b in V.values() for i in b]
        s = pool.dropna(subset=its + TARGET_ITEMS)
        s = s[s[its].std(axis=1) > 0]
        c = CFA(V).fit(s)
        fi = c.fit_indices()
        t = c.construct_table()
        alt_rows.append({"mapping": name, "n": fi["n"], "CFI": fi["CFI"], "TLI": fi["TLI"],
                         "RMSEA": fi["RMSEA"], "SRMR": fi["SRMR"],
                         "min_alpha": min(cronbach_alpha(s[b]) for b in V.values()),
                         "min_AVE": t.AVE.min(),
                         "max_HTMT": float(np.nanmax(htmt(s[its], V).values))})
    pd.DataFrame(alt_rows).to_csv(TABLES / "measurement_alternatives.csv", index=False)

    SUMMARY["cfa_fit"] = {k: (float(v) if not isinstance(v, bool) else v) for k, v in fit.items()}
    SUMMARY["fornell_larcker_satisfied"] = bool(fl_ok)
    SUMMARY["max_htmt"] = float(np.nanmax(H.values))
    SUMMARY["harman_first_factor_pct"] = harman
    SUMMARY["measurement_alternatives"] = alt_rows
    return ct, phi, H


# --------------------------------------------------------------------------- #
# 3. Fuzzy inference
# --------------------------------------------------------------------------- #
def fuzzy_scores(d):
    fis = HierarchicalFIS()
    out = fis.predict(d)
    for k in ("PC", "IQ", "TW", "TRFS"):
        d[k] = out[k]
    d["TRFS_class"] = categorize(d.TRFS)
    # rule tables for the appendix
    pd.DataFrame(rule_table(trust_rule_base(), ["R", "E"], "TW")).to_csv(
        TABLES / "rules_TW.csv", index=False)
    pd.DataFrame(rule_table(mean_rule_base(2), ["SE", "C"], "IQ")).to_csv(
        TABLES / "rules_IQ.csv", index=False)
    pd.DataFrame(rule_table(mean_rule_base(4), ["TE", "P", "AC", "A"], "PC")).to_csv(
        TABLES / "rules_PC.csv", index=False)
    pd.DataFrame(rule_table(final_rule_base(), ["PC", "IQ", "TW"], "TRFS")).to_csv(
        TABLES / "rules_TRFS.csv", index=False)
    pd.DataFrame(rule_table(ActivityFIS().rules, ["Capability", "R", "E"], "Feasibility")).to_csv(
        TABLES / "rules_activity.csv", index=False)

    # convergent / criterion validity of the fuzzy score
    crit = {
        "TSI (ChatGPT easier/clearer than professors)": d[TARGET],
        "Q24e satisfied with assistance": d["Q24e"],
        "Q16 overall experience": d["Q16"],
        "Q15 intensity of use": d["Q15"],
        "Q22h ChatGPT might replace formal education": d["Q22h"],
    }
    vrows = []
    for k, v in crit.items():
        m = v.notna()
        r, p = stats.spearmanr(d.loc[m, "TRFS"], v[m])
        vrows.append({"criterion": k, "n": int(m.sum()), "spearman_rho": r, "p": p})
    val = pd.DataFrame(vrows)
    val.to_csv(TABLES / "table_trfs_validity.csv", index=False)
    # known-groups: TRFS by intensity of use
    g = d.groupby("Q15")["TRFS"].agg(["count", "mean", "std"])
    g.to_csv(TABLES / "table_trfs_by_usage.csv")
    kw = stats.kruskal(*[x.values for _, x in d.groupby("Q15")["TRFS"]])
    mu = d.Q15.notna()
    jt_rho = stats.spearmanr(d.loc[mu, "Q15"], d.loc[mu, "TRFS"])

    cls = d.TRFS_class.value_counts(normalize=True) * 100
    SUMMARY["trfs"] = {"mean": float(d.TRFS.mean()), "sd": float(d.TRFS.std()),
                       "median": float(d.TRFS.median()),
                       "class_pct": {k: float(v) for k, v in cls.items()},
                       "kw_usage_H": float(kw.statistic), "kw_usage_p": float(kw.pvalue),
                       "rho_usage": float(jt_rho.statistic)}
    SUMMARY["intermediate"] = {k: {"mean": float(d[k].mean()), "sd": float(d[k].std())}
                               for k in ("PC", "IQ", "TW")}
    SUMMARY["trfs_validity"] = vrows
    return d


def activity_scores(d, scored):
    fis = ActivityFIS()
    rng = np.random.default_rng(SEED)
    B = 2000
    idx = rng.integers(0, len(d), size=(B, len(d)))
    rows, mat = [], {}
    for a, spec in ACTIVITIES.items():
        s, _ = fis.predict(d[a].to_numpy(), d["R"].to_numpy(), d["E"].to_numpy())
        mat[a] = s
        boot = s[idx].mean(axis=1)
        cls = pd.Series(categorize(s)).value_counts(normalize=True) * 100
        rows.append({"code": a, "activity": spec["name"], "items": ", ".join(spec["items"]),
                     "capability_mean": d[a].mean(), "capability_sd": d[a].std(),
                     "alpha": cronbach_alpha(scored[spec["items"]]) if len(spec["items"]) > 1 else np.nan,
                     "TRFS_mean": s.mean(), "TRFS_sd": s.std(), "TRFS_median": np.median(s),
                     "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                     "pct_full": cls.get(CATEGORIES[2], 0.0),
                     "pct_partial": cls.get(CATEGORIES[1], 0.0),
                     "pct_human": cls.get(CATEGORIES[0], 0.0)})
    act = pd.DataFrame(rows)
    act["category"] = categorize(act.TRFS_mean)
    # Within-category tier: is the activity reliably above or below the
    # neutral centre (50) of the Medium output set? (bootstrap 95% CI)
    act["tier"] = np.where(act.ci_low > 50, "AI-leaning",
                           np.where(act.ci_high < 50, "Human-leaning", "Balanced"))
    act["net_endorsement"] = act.pct_full - act.pct_human
    act = act.sort_values("TRFS_mean", ascending=False).reset_index(drop=True)
    act["rank"] = np.arange(1, len(act) + 1)
    act.to_csv(TABLES / "table_activity_feasibility.csv", index=False)
    M = pd.DataFrame(mat)
    for a in M:
        d[f"TRFS_{a}"] = M[a].values

    # Friedman test + Kendall's W, then adjacent-rank Wilcoxon with Holm
    fr = stats.friedmanchisquare(*[M[a] for a in M])
    n, k = M.shape
    W = fr.statistic / (n * (k - 1))
    order = list(act.code)
    prow = []
    for a, b in zip(order[:-1], order[1:]):
        diff = (M[a] - M[b]).to_numpy()
        nz = diff[diff != 0]
        rk = stats.rankdata(np.abs(nz))
        wp, wm = rk[nz > 0].sum(), rk[nz < 0].sum()
        res = stats.wilcoxon(M[a], M[b])
        prow.append({"pair": f"{a} vs {b}", "mean_diff": diff.mean(), "p_raw": res.pvalue,
                     "rank_biserial": (wp - wm) / (wp + wm)})
    pw = pd.DataFrame(prow)
    # Holm adjustment
    o = np.argsort(pw.p_raw.values)
    m = len(o)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * pw.p_raw.values[j]))
        adj[j] = run
    pw["p_holm"] = adj
    pw.to_csv(TABLES / "table_activity_pairwise.csv", index=False)
    SUMMARY["activity_friedman"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue),
                                    "kendall_W": float(W), "n": int(n), "k": int(k)}
    SUMMARY["activities"] = act.to_dict(orient="records")
    return act, M


# --------------------------------------------------------------------------- #
# 4. Predictive modelling (ANN vs baselines)
# --------------------------------------------------------------------------- #
def predictive(d, fast=False):
    F = d[FACTOR_ORDER].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    Xsets = {"F": F, "T": d[["TRFS"]].to_numpy(float),
             "FH": np.column_stack([F, d[["PC", "IQ", "TW", "TRFS"]].to_numpy(float)])}
    splits, reps = (5, 1) if fast else (10, 3)
    cv, chosen = repeated_cv(Xsets, y, model_zoo(), splits, reps, log=log)
    cv.to_csv(TABLES / "cv_fold_scores.csv", index=False)
    chosen.to_csv(TABLES / "ann_selected_hyperparameters.csv", index=False)
    summ = cv.groupby("model")[["MSE", "RMSE", "MAE", "R2", "seconds"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.sort_values("RMSE_mean")
    ref = "ANN (MLP)"
    n_tr, n_te = cv.n_train.iloc[0], cv.n_test.iloc[0]
    tt = []
    for mname in summ.index:
        if mname == ref:
            tt.append((np.nan, np.nan))
            continue
        a = cv[cv.model == ref].sort_values("fold").RMSE.values
        b = cv[cv.model == mname].sort_values("fold").RMSE.values
        tt.append(corrected_ttest(a, b, n_tr, n_te))
    summ["t_vs_ANN"] = [t for t, _ in tt]
    summ["p_vs_ANN"] = [p for _, p in tt]
    summ.to_csv(TABLES / "table_model_comparison.csv")
    SUMMARY["cv_design"] = {"splits": splits, "repeats": reps}
    SUMMARY["model_comparison"] = summ.reset_index().to_dict(orient="records")
    SUMMARY["ann_hyperparameters_mode"] = (
        chosen[chosen.model == ref].drop(columns=["model", "fold"]).astype(str)
        .value_counts().head(3).reset_index().to_dict(orient="records"))

    # ANN as a surrogate (emulator) of the fuzzy system: can a network learn
    # the HFIS mapping from the eight factors?
    sur = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=400,
                                                       early_stopping=True,
                                                       random_state=SEED))
    pred = cross_val_predict(sur, F, d.TRFS.to_numpy(), cv=KFold(5, shuffle=True,
                                                                 random_state=SEED))
    r2 = 1 - ((d.TRFS - pred) ** 2).sum() / ((d.TRFS - d.TRFS.mean()) ** 2).sum()
    SUMMARY["ann_surrogate_of_fis"] = {"R2": float(r2),
                                       "RMSE": float(np.sqrt(((d.TRFS - pred) ** 2).mean()))}
    return cv, summ


def importance(d):
    """RQ2: which factors drive feasibility? Three complementary views."""
    F = d[FACTOR_ORDER].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    kf = KFold(5, shuffle=True, random_state=SEED)
    imps = []
    for tr, te in kf.split(F):
        m = make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=400, early_stopping=True,
                                       random_state=SEED)).fit(F[tr], y[tr])
        pi = permutation_importance(m, F[te], y[te], n_repeats=10, random_state=SEED,
                                    scoring="r2")
        imps.append(pi.importances_mean)
    imps = np.array(imps)
    # FIS local sensitivity: change in TRFS when one factor moves +1 SD
    fis = HierarchicalFIS()
    base = d.TRFS.to_numpy()
    sens = []
    for f in FACTOR_ORDER:
        dd = d[FACTOR_ORDER].copy()
        dd[f] = np.clip(dd[f] + dd[f].std(), 1, 5)
        sens.append((fis.predict(dd)["TRFS"] - base).mean())
    rho = [stats.spearmanr(d[f], y).statistic for f in FACTOR_ORDER]
    imp = pd.DataFrame({"factor": FACTOR_ORDER,
                        "name": [FACTORS[f]["name"] for f in FACTOR_ORDER],
                        "ann_perm_importance_R2_drop": imps.mean(axis=0),
                        "ann_perm_importance_sd": imps.std(axis=0),
                        "fis_sensitivity_plus1SD": sens,
                        "spearman_with_TSI": rho})
    imp["rank_ann"] = imp.ann_perm_importance_R2_drop.rank(ascending=False).astype(int)
    imp["rank_fis"] = imp.fis_sensitivity_plus1SD.rank(ascending=False).astype(int)
    imp.to_csv(TABLES / "table_factor_importance.csv", index=False)
    SUMMARY["importance"] = imp.to_dict(orient="records")
    SUMMARY["importance_rank_agreement_tau"] = float(
        stats.kendalltau(imp.rank_ann, imp.rank_fis).statistic)
    return imp


# --------------------------------------------------------------------------- #
# 5. Robustness of the fuzzy design
# --------------------------------------------------------------------------- #
def robustness(d, act, draws=500):
    rng = np.random.default_rng(SEED)
    base_rank = act.set_index("code")["rank"]
    base_cat = act.set_index("code")["category"]
    rows, scores = [], []
    codes = list(ACTIVITIES)
    for i in range(draws):
        cfg = dict(in_shift=rng.uniform(-0.25, 0.25), in_spread=rng.uniform(-0.25, 0.25),
                   out_shift=rng.uniform(-5, 5), out_spread=rng.uniform(-5, 5),
                   rounding=rng.choice(["half_down", "half_up"]),
                   t_norm=rng.choice(["min", "prod"]))
        fis = ActivityFIS(**cfg)
        means = {a: fis.predict(d[a].to_numpy(), d.R.to_numpy(), d.E.to_numpy())[0].mean()
                 for a in codes}
        s = pd.Series(means)
        rk = s.rank(ascending=False)
        tau = stats.kendalltau(base_rank[codes], rk[codes]).statistic
        cat = pd.Series(categorize(s.values), index=s.index)
        rows.append({**cfg, "kendall_tau": tau,
                     "category_agreement": float((cat[codes] == base_cat[codes]).mean())})
        scores.append(s)
    rob = pd.DataFrame(rows)
    rob.to_csv(TABLES / "robustness_draws.csv", index=False)
    S = pd.DataFrame(scores)
    band = pd.DataFrame({"p2.5": S.quantile(0.025), "p50": S.median(), "p97.5": S.quantile(0.975)})
    band.to_csv(TABLES / "robustness_activity_bands.csv")
    # alternative factor weighting for the overall TRFS: equal weights in FIS_4
    SUMMARY["robustness"] = {"draws": draws,
                             "tau_mean": float(rob.kendall_tau.mean()),
                             "tau_p5": float(rob.kendall_tau.quantile(0.05)),
                             "category_agreement_mean": float(rob.category_agreement.mean()),
                             "category_agreement_min": float(rob.category_agreement.min())}
    return rob, band


# --------------------------------------------------------------------------- #
# 6. Subgroups
# --------------------------------------------------------------------------- #
def subgroups(d):
    rows = []
    for q, mapping in DEMOGRAPHICS.items():
        g = d[q].map(mapping)
        m = g.notna()
        groups = [d.loc[m & (g == lvl), "TRFS"].values for lvl in g[m].unique()]
        groups = [x for x in groups if len(x) >= 30]
        H, p = stats.kruskal(*groups)
        n = sum(len(x) for x in groups)
        eps2 = (H - len(groups) + 1) / (n - len(groups))
        for lvl in sorted(g[m].unique()):
            x = d.loc[m & (g == lvl), "TRFS"]
            rows.append({"variable": q, "level": lvl, "n": len(x), "TRFS_mean": x.mean(),
                         "TRFS_sd": x.std(), "KW_H": H, "KW_p": p, "epsilon2": eps2})
    sg = pd.DataFrame(rows)
    sg.to_csv(TABLES / "table_subgroups.csv", index=False)
    # activity x field heatmap data
    fld = d.Q10.map(DEMOGRAPHICS["Q10"])
    hm = d.groupby(fld)[[f"TRFS_{a}" for a in ACTIVITIES]].mean()
    hm.columns = list(ACTIVITIES)
    hm.to_csv(TABLES / "table_activity_by_field.csv")
    SUMMARY["subgroups"] = (sg.groupby("variable")[["KW_H", "KW_p", "epsilon2"]].first()
                            .reset_index().to_dict(orient="records"))
    return sg, hm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    log("loading data")
    raw = load_raw()
    d, scored, flow = build_sample(raw)
    log(f"analytic sample n={len(d)}")
    describe_sample(raw, d, flow)
    item_desc = scored[[i for f in FACTOR_ORDER for i in FACTORS[f]["items"]]].describe().T
    item_desc.to_csv(TABLES / "table_item_descriptives.csv")

    log("measurement model")
    ct, phi, H = measurement(raw, scored)
    log("fuzzy inference")
    d = fuzzy_scores(d)
    act, M = activity_scores(d, scored)
    log("predictive models")
    cv, summ = predictive(d, fast=args.fast)
    log("importance")
    imp = importance(d)
    log("robustness")
    rob, band = robustness(d, act, draws=100 if args.fast else 500)
    log("subgroups")
    sg, hm = subgroups(d)
    save_processed(d)

    log("figures")
    figures.make_all(d, ct, phi, act, cv, summ, imp, rob, band, hm)
    with open(TABLES.parent / "summary.json", "w") as fh:
        json.dump(SUMMARY, fh, indent=2, default=float)
    log("done")


if __name__ == "__main__":
    main()
