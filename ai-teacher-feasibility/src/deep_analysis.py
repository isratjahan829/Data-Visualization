"""Extended ("deep") analyses for the revised manuscript.

    python deep_analysis.py   (after run_pipeline.py; ~3 min)

1. Mathematical properties of the fuzzy systems (thresholds, centroids, rule
   complexity, exhaustive monotonicity / range checks, Mamdani vs. provably
   monotone weighted-average inference).
2. Shapley (dominance-analysis) decomposition of R^2 with bootstrap CIs.
3. Multilevel models: students nested in countries (ICC, random intercepts,
   fixed effects of the eight factors).
4. Quantile regression of the TSI on the TRFS.
5. Effect sizes and equivalence (TOST) tests for the activity scores.
6. Measurement invariance across groups (per-group CFA, Tucker congruence).
7. Student typologies (k-means, silhouette and bootstrap stability).
Writes results/tables/deep_*.csv, results/deep_summary.json, figures 11-17.
"""
from __future__ import annotations

import itertools
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from config import (ACTIVITIES, DEMOGRAPHICS, FACTOR_ORDER, FACTORS, PROCESSED_CSV,
                    REVERSED, SEED, T_HIGH, T_LOW, TABLES, TARGET)
from fuzzy_system import (ActivityFIS, HierarchicalFIS, MonotoneActivityFIS,
                          MonotoneHFIS, Partition, mamdani, rule_table_is_monotone,
                          trap_centroid, weighted_average)
from psychometrics import CFA

warnings.filterwarnings("ignore")
OUT = {}


# --------------------------------------------------------------------------- #
# 1. Mathematical properties
# --------------------------------------------------------------------------- #
def grid_check(fn, lo, hi, n_inputs, pts):
    g = np.linspace(lo, hi, pts)
    mesh = np.meshgrid(*([g] * n_inputs), indexing="ij")
    flat = [m.ravel() for m in mesh]
    out = np.empty(flat[0].size)
    step = 20000
    for s in range(0, out.size, step):
        out[s:s + step] = fn([f[s:s + step] for f in flat])
    Z = out.reshape(mesh[0].shape)
    worst = min(np.diff(Z, axis=a).min() for a in range(n_inputs))
    share = np.mean([(np.diff(Z, axis=a) < -1e-9).mean() for a in range(n_inputs)])
    return {"min": Z.min(), "max": Z.max(), "worst_local_decrease": worst,
            "share_decreasing_steps": share, "grid_points": out.size}


def math_properties(d):
    sp = Partition.score()
    cL = trap_centroid(*sp.params["Low"])
    cH = trap_centroid(*sp.params["High"])
    # numerical centroid for validation of the closed form
    y = np.linspace(0, 100, 200001)
    muL = sp.memberships(y)[:, 0]
    cL_num = (y * muL).sum() / muL.sum()
    # crossover points: Low(y) = Medium(y) on [20,50] -> (50-y)/30 = (y-20)/30
    x_LM = (50 + 20) / 2
    x_MH = (50 + 80) / 2

    hf, mh = HierarchicalFIS(), MonotoneHFIS()
    afis, mafis = ActivityFIS(), MonotoneActivityFIS()
    rows = []
    systems = {
        "FIS1 PC(TE,P,AC,A)": (4, 1, 5, 13, hf.rules["PC"], hf.in_part),
        "FIS2 IQ(SE,C)": (2, 1, 5, 41, hf.rules["IQ"], hf.in_part),
        "FIS3 TW(R,E)": (2, 1, 5, 41, hf.rules["TW"], hf.in_part),
        "FIS4 TRFS(PC,IQ,TW)": (3, 0, 100, 41, hf.rules["TRFS"], hf.mid_part),
        "Activity FIS(Cap,R,E)": (3, 1, 5, 33, afis.rules, afis.in_part),
    }
    for name, (k, lo, hi, pts, rules, part) in systems.items():
        mam = grid_check(lambda X: mamdani(X, [part] * k, rules, sp)[0], lo, hi, k, pts)
        wa = grid_check(lambda X: weighted_average(X, [part] * k, rules, mh.c), lo, hi, k, pts)
        rows.append({"system": name, "rules": len(rules),
                     "rule_table_monotone": rule_table_is_monotone(rules, k),
                     "grid_points": mam["grid_points"],
                     "mamdani_min": mam["min"], "mamdani_max": mam["max"],
                     "mamdani_worst_decrease": mam["worst_local_decrease"],
                     "mamdani_share_decreasing": mam["share_decreasing_steps"],
                     "wa_min": wa["min"], "wa_max": wa["max"],
                     "wa_worst_decrease": wa["worst_local_decrease"]})
    prop = pd.DataFrame(rows)
    prop.to_csv(TABLES / "deep_math_properties.csv", index=False)

    # Ruspini (partition-of-unity) check
    xl = np.linspace(1, 5, 4001)
    xs = np.linspace(0, 100, 4001)
    rusp = max(abs(hf.in_part.memberships(xl).sum(1) - 1).max(),
               abs(sp.memberships(xs).sum(1) - 1).max())

    # Mamdani vs monotone scores on the real data
    m = mh.predict(d)
    d["TRFS_WA"] = m["TRFS"]
    act_rows = []
    for a, spec in ACTIVITIES.items():
        s_m, _ = afis.predict(d[a].values, d.R.values, d.E.values)
        s_w = mafis.predict(d[a].values, d.R.values, d.E.values)
        d[f"WA_{a}"] = s_w
        act_rows.append({"code": a, "activity": spec["name"], "mamdani": s_m.mean(),
                         "wa": s_w.mean(), "wa_pct_full": 100 * (s_w >= T_HIGH).mean(),
                         "wa_pct_human": 100 * (s_w < T_LOW).mean()})
    comp = pd.DataFrame(act_rows)
    comp["rank_mamdani"] = comp.mamdani.rank(ascending=False).astype(int)
    comp["rank_wa"] = comp.wa.rank(ascending=False).astype(int)
    comp.to_csv(TABLES / "deep_mamdani_vs_monotone.csv", index=False)
    flat_rules = 3 ** 8
    hier_rules = 3 ** 4 + 3 ** 2 + 3 ** 2 + 3 ** 3
    OUT["math"] = {
        "cL_closed_form": cL, "cL_numeric": cL_num, "cH": cH,
        "crossover_LM": x_LM, "crossover_MH": x_MH, "ruspini_max_error": float(rusp),
        "flat_rules": flat_rules, "hier_rules": hier_rules,
        "reduction_pct": 100 * (1 - hier_rules / flat_rules),
        "rho_mamdani_wa": float(stats.spearmanr(d.TRFS, d.TRFS_WA).statistic),
        "r_mamdani_wa": float(np.corrcoef(d.TRFS, d.TRFS_WA)[0, 1]),
        "rho_wa_tsi": float(stats.spearmanr(d.TRFS_WA, d[TARGET]).statistic),
        "rho_mamdani_tsi": float(stats.spearmanr(d.TRFS, d[TARGET]).statistic),
        "wa_mean": float(d.TRFS_WA.mean()), "wa_sd": float(d.TRFS_WA.std()),
        "activity_rank_tau": float(stats.kendalltau(comp.rank_mamdani, comp.rank_wa).statistic),
        "wa_max_activity": float(comp.wa.max()), "wa_min_activity": float(comp.wa.min()),
        "worst_mamdani_decrease": float(prop.mamdani_worst_decrease.min()),
        "max_share_decreasing": float(prop.mamdani_share_decreasing.max()),
        "worst_wa_decrease": float(prop.wa_worst_decrease.min()),
    }
    return prop, comp


# --------------------------------------------------------------------------- #
# 2. Shapley decomposition of R^2 (general dominance weights)
# --------------------------------------------------------------------------- #
def r2_from_corr(R, r, idx):
    if not idx:
        return 0.0
    idx = list(idx)
    return float(r[idx] @ np.linalg.solve(R[np.ix_(idx, idx)], r[idx]))


def shapley_r2(X, y):
    Z = np.column_stack([X, y])
    C = np.corrcoef(Z, rowvar=False)
    R, r = C[:-1, :-1], C[:-1, -1]
    p = X.shape[1]
    cache = {}
    for k in range(p + 1):
        for S in itertools.combinations(range(p), k):
            cache[S] = r2_from_corr(R, r, S)
    phi = np.zeros(p)
    from math import factorial
    for j in range(p):
        others = [i for i in range(p) if i != j]
        for k in range(p):
            wk = factorial(k) * factorial(p - k - 1) / factorial(p)
            for S in itertools.combinations(others, k):
                Sj = tuple(sorted(S + (j,)))
                phi[j] += wk * (cache[Sj] - cache[S])
    return phi, cache[tuple(range(p))]


def dominance(d, B=500):
    X = d[FACTOR_ORDER].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    phi, r2 = shapley_r2(X, y)
    rng = np.random.default_rng(SEED)
    boots = np.array([shapley_r2(X[i], y[i])[0]
                      for i in (rng.integers(0, len(y), len(y)) for _ in range(B))])
    tab = pd.DataFrame({"factor": FACTOR_ORDER, "name": [FACTORS[f]["name"] for f in FACTOR_ORDER],
                        "shapley_R2": phi, "ci_low": np.percentile(boots, 2.5, axis=0),
                        "ci_high": np.percentile(boots, 97.5, axis=0),
                        "share_pct": 100 * phi / r2})
    tab = tab.sort_values("shapley_R2", ascending=False)
    tab.to_csv(TABLES / "deep_shapley_r2.csv", index=False)
    OUT["shapley"] = {"R2_total": r2, "sum_check": float(phi.sum()), "B": B,
                      "top": tab.iloc[0].factor, "top_share": float(tab.iloc[0].share_pct),
                      "rows": tab.to_dict(orient="records")}
    return tab


# --------------------------------------------------------------------------- #
# 3. Multilevel models
# --------------------------------------------------------------------------- #
def multilevel(d, min_n=30):
    cnt = d.Q4.value_counts()
    keep = cnt[cnt >= min_n].index
    m = d[d.Q4.isin(keep)].copy()
    m["country"] = m.Q4.astype(str)
    z = m[FACTOR_ORDER].apply(lambda s: (s - s.mean()) / s.std())
    for f in FACTOR_ORDER:
        m[f"z{f}"] = z[f]
    res = {}
    for outcome in ["TSI", "TRFS"]:
        null = smf.mixedlm(f"{outcome} ~ 1", m, groups=m["country"]).fit(reml=True)
        tau2 = float(null.cov_re.iloc[0, 0])
        icc = tau2 / (tau2 + null.scale)
        res[outcome] = {"icc": icc, "tau2": tau2, "sigma2": float(null.scale)}
        if outcome == "TRFS":
            re = null.random_effects
            ce = pd.DataFrame({"country": list(re.keys()),
                               "u": [float(v.iloc[0]) for v in re.values()]})
            ce["n"] = ce.country.map(m.country.value_counts())
            # conditional SD of the BLUP: sqrt(tau2*sigma2/(n*tau2+sigma2))
            ce["se"] = np.sqrt(tau2 * null.scale / (ce.n * tau2 + null.scale))
            ce["mean_TRFS"] = null.fe_params.iloc[0] + ce.u
            ce.sort_values("u").to_csv(TABLES / "deep_country_effects.csv", index=False)
    full = smf.mixedlm("TSI ~ " + " + ".join(f"z{f}" for f in FACTOR_ORDER), m,
                       groups=m["country"]).fit(reml=True)
    ols = smf.ols("TSI ~ " + " + ".join(f"z{f}" for f in FACTOR_ORDER), m).fit()
    tau2_full = float(full.cov_re.iloc[0, 0])
    fe = pd.DataFrame({"term": full.fe_params.index, "beta": full.fe_params.values,
                       "se": full.bse_fe.values, "p": full.pvalues[full.fe_params.index].values,
                       "ci_low": full.conf_int().loc[full.fe_params.index, 0].values,
                       "ci_high": full.conf_int().loc[full.fe_params.index, 1].values,
                       "ols_beta": ols.params[full.fe_params.index].values})
    fe.to_csv(TABLES / "deep_mixed_model.csv", index=False)
    # marginal / conditional R^2 (Nakagawa & Schielzeth 2013)
    var_f = float(np.var(full.model.exog @ full.fe_params.values))
    r2m = var_f / (var_f + tau2_full + full.scale)
    r2c = (var_f + tau2_full) / (var_f + tau2_full + full.scale)
    # likelihood-ratio test for the random intercept (ML fits, boundary-corrected)
    fml = smf.mixedlm("TSI ~ " + " + ".join(f"z{f}" for f in FACTOR_ORDER), m,
                      groups=m["country"]).fit(reml=False)
    ols_ml = sm.OLS(m.TSI, sm.add_constant(m[[f"z{f}" for f in FACTOR_ORDER]])).fit()
    lr = 2 * (fml.llf - ols_ml.llf)
    p_lr = 0.5 * stats.chi2.sf(lr, 1)
    OUT["multilevel"] = {"n": len(m), "countries": int(m.country.nunique()), "min_n": min_n,
                         "icc_TSI": res["TSI"]["icc"], "icc_TRFS": res["TRFS"]["icc"],
                         "icc_TSI_conditional": tau2_full / (tau2_full + full.scale),
                         "R2_marginal": r2m, "R2_conditional": r2c,
                         "LR": lr, "p_LR": p_lr,
                         "fixed": fe.to_dict(orient="records")}
    return fe


# --------------------------------------------------------------------------- #
# 4. Quantile regression
# --------------------------------------------------------------------------- #
def quantile_reg(d, reps=20, step=12.5):
    """Quantile regression of the TSI on the TRFS. The TSI takes only nine
    values, so quantiles are estimated on jittered outcomes y + U(-h, h),
    h = step/2, and averaged over replications (Machado & Santos Silva, 2005)."""
    taus = np.round(np.arange(0.1, 0.91, 0.05), 2)
    x = d.TRFS / 10.0  # effect per 10 TRFS points
    X = sm.add_constant(x.rename("TRFS10"))
    rng = np.random.default_rng(SEED)
    rows = []
    for t in taus:
        b, lo, hi = [], [], []
        for _ in range(reps):
            yj = d[TARGET] + rng.uniform(-step / 2, step / 2, len(d))
            r = sm.QuantReg(yj, X).fit(q=t)
            ci = r.conf_int().loc["TRFS10"]
            b.append(r.params["TRFS10"]); lo.append(ci[0]); hi.append(ci[1])
        rows.append({"tau": t, "beta": np.mean(b), "ci_low": np.mean(lo), "ci_high": np.mean(hi)})
    ols = sm.OLS(d[TARGET], X).fit()
    q = pd.DataFrame(rows)
    q.to_csv(TABLES / "deep_quantile_regression.csv", index=False)
    OUT["quantile"] = {"reps": reps, "ols_beta": float(ols.params["TRFS10"]),
                       "ols_ci": [float(v) for v in ols.conf_int().loc["TRFS10"]],
                       "beta_q10": float(q.beta.iloc[0]),
                       "beta_q50": float(q[q.tau == 0.5].beta.iloc[0]),
                       "beta_q90": float(q.beta.iloc[-1]),
                       "beta_min": float(q.beta.min()), "beta_max": float(q.beta.max())}
    return q, ols


# --------------------------------------------------------------------------- #
# 5. Effect sizes and equivalence tests for activities
# --------------------------------------------------------------------------- #
def tost_one_sample(x, mu0, delta):
    n = len(x)
    m, s = x.mean(), x.std(ddof=1)
    se = s / np.sqrt(n)
    t1 = (m - (mu0 - delta)) / se
    t2 = (m - (mu0 + delta)) / se
    p1 = stats.t.sf(t1, n - 1)
    p2 = stats.t.cdf(t2, n - 1)
    return max(p1, p2)


def activity_effects(d, delta=2.0):
    rows = []
    for a, spec in ACTIVITIES.items():
        x = d[f"TRFS_{a}"].to_numpy()
        n = len(x)
        dz = (x.mean() - 50) / x.std(ddof=1)
        se_d = np.sqrt(1 / n + dz ** 2 / (2 * n))
        rows.append({"code": a, "activity": spec["name"], "mean": x.mean(),
                     "cohen_d_vs_50": dz, "d_ci_low": dz - 1.96 * se_d,
                     "d_ci_high": dz + 1.96 * se_d,
                     "p_tost_equiv_50": tost_one_sample(x, 50.0, delta),
                     "equivalent_to_50": tost_one_sample(x, 50.0, delta) < 0.05,
                     "p_below_65": stats.ttest_1samp(x, T_HIGH, alternative="less").pvalue})
    eff = pd.DataFrame(rows).sort_values("mean", ascending=False)
    eff.to_csv(TABLES / "deep_activity_effects.csv", index=False)
    OUT["activity_effects"] = {"delta": delta,
                               "n_equivalent": int(eff.equivalent_to_50.sum()),
                               "equivalent": list(eff[eff.equivalent_to_50].activity),
                               "d_max": float(eff.cohen_d_vs_50.max()),
                               "d_min": float(eff.cohen_d_vs_50.min()),
                               "max_p_below_65": float(eff.p_below_65.max())}
    return eff


# --------------------------------------------------------------------------- #
# 6. Measurement invariance (per-group CFA, Tucker congruence)
# --------------------------------------------------------------------------- #
def tucker(a, b):
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def invariance(d):
    blocks = {f: FACTORS[f]["items"] for f in FACTOR_ORDER}
    sc = d.copy()
    for it in REVERSED:
        if it in sc:
            sc[it] = 6 - sc[it]
    groups = {
        "Gender": ("Q2", {1: "Male", 2: "Female"}),
        "Level": ("Q8", {1: "Undergraduate", 2: "Postgraduate", 3: "Postgraduate"}),
        "Field": ("Q10", DEMOGRAPHICS["Q10"]),
        "Language": ("source", {"EN": "English", "ES": "Spanish", "AR": "Arabic",
                                "IT": "Italian"}),
    }
    ref = CFA(blocks).fit(sc)
    L_ref = ref.loadings().loading.values
    rows = []
    for gname, (col, mp) in groups.items():
        lab = sc[col].map(mp)
        for lvl in sorted(lab.dropna().unique()):
            sub = sc[lab == lvl]
            if len(sub) < 300:
                continue
            c = CFA(blocks).fit(sub)
            fi = c.fit_indices()
            L = c.loadings()
            cong = [tucker(L[L.factor == f].loading.values,
                           ref.loadings()[ref.loadings().factor == f].loading.values)
                    for f in FACTOR_ORDER]
            rows.append({"grouping": gname, "group": lvl, "n": len(sub), "CFI": fi["CFI"],
                         "RMSEA": fi["RMSEA"], "SRMR": fi["SRMR"],
                         "min_congruence": min(cong), "mean_congruence": np.mean(cong),
                         "max_abs_loading_diff": float(np.abs(L.loading.values - L_ref).max())})
    inv = pd.DataFrame(rows)
    inv.to_csv(TABLES / "deep_invariance.csv", index=False)
    OUT["invariance"] = {"groups": len(inv), "min_congruence": float(inv.min_congruence.min()),
                         "min_CFI": float(inv.CFI.min()), "max_RMSEA": float(inv.RMSEA.max()),
                         "max_loading_diff": float(inv.max_abs_loading_diff.max())}
    return inv


# --------------------------------------------------------------------------- #
# 7. Student typologies (k-means on activity capability profiles)
# --------------------------------------------------------------------------- #
def typology(d, kmax=6, boots=20):
    """k chosen by the pre-declared rule: maximal mean silhouette among the
    solutions whose bootstrap stability (mean adjusted Rand index) is >= 0.90."""
    codes = list(ACTIVITIES)
    X = StandardScaler().fit_transform(d[codes].to_numpy(float))
    rng = np.random.default_rng(SEED)
    rows, fits = [], {}
    for k in range(2, kmax + 1):
        km = KMeans(k, n_init=10, random_state=SEED).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=6000, random_state=SEED)
        aris = []
        for b in range(boots):
            i = rng.integers(0, len(X), len(X))
            kb = KMeans(k, n_init=5, random_state=b).fit(X[i])
            aris.append(adjusted_rand_score(km.labels_, kb.predict(X)))
        rows.append({"k": k, "silhouette": sil, "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
                     "bootstrap_ARI": float(np.mean(aris))})
        fits[k] = km
    sel = pd.DataFrame(rows)
    ok = sel[sel.bootstrap_ARI >= 0.90]
    k_sel = int(ok.loc[ok.silhouette.idxmax(), "k"])
    ev = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))[::-1]
    out = {}
    for k in sorted({k_sel, 3}):
        lab = fits[k].labels_
        tmp = d.assign(cluster=lab)
        prof = tmp.groupby("cluster")[codes + ["TRFS", TARGET] + FACTOR_ORDER].mean()
        prof["n"] = tmp.cluster.value_counts().sort_index()
        prof["pct"] = 100 * prof.n / len(d)
        order = list(prof.TRFS.sort_values().index)
        names = (["AI-sceptical", "AI-receptive"] if k == 2 else
                 ["AI-sceptical", "Ambivalent", "AI-enthusiastic"])
        mapping = {c: names[i] for i, c in enumerate(order)}
        prof["label"] = prof.index.map(mapping)
        prof = prof.loc[order]
        prof.to_csv(TABLES / f"deep_typology_k{k}.csv")
        lab_named = tmp.cluster.map(mapping)
        H = stats.kruskal(*[d.loc[lab_named == t, TARGET] for t in prof.label])
        eps2 = (H.statistic - k + 1) / (len(d) - k)
        ct = pd.crosstab(lab_named, d.Q10.map(DEMOGRAPHICS["Q10"]))
        chi = stats.chi2_contingency(ct)
        cramer = np.sqrt(chi.statistic / (ct.values.sum() * (min(ct.shape) - 1)))
        out[k] = {"profiles": prof.reset_index().to_dict(orient="records"),
                  "kw_TSI_H": float(H.statistic), "kw_TSI_eps2": float(eps2),
                  "field_cramers_v": float(cramer)}
        if k == 3:
            d["typology3"] = lab_named.values
    sel.to_csv(TABLES / "deep_typology_selection.csv", index=False)
    OUT["typology"] = {"k_selected": k_sel, "selection": rows,
                       "first_pc_share": float(ev[0] / ev.sum()), "solutions": out}
    return sel


def main():
    d = pd.read_csv(PROCESSED_CSV)
    print("math properties"); prop, comp = math_properties(d)
    print("shapley"); sh = dominance(d)
    print("multilevel"); fe = multilevel(d)
    print("quantile"); q, ols = quantile_reg(d)
    print("activity effects"); eff = activity_effects(d)
    print("invariance"); inv = invariance(d)
    print("typology"); sel = typology(d)
    with open(TABLES.parent / "deep_summary.json", "w") as fh:
        json.dump(OUT, fh, indent=2, default=float)
    import figures_deep
    figures_deep.make_all(d, prop, comp, sh, q, eff, OUT)
    print(json.dumps({k: v for k, v in OUT.items() if k not in ("typology", "multilevel", "shapley")},
                     indent=1, default=float))
    print("done")


if __name__ == "__main__":
    main()
