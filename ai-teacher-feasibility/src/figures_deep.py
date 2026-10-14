"""Figures 11-17 for the extended analyses (same visual system as figures.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

import figures as F
from config import ACTIVITIES, FACTORS, T_HIGH, T_LOW, TABLES
from fuzzy_system import ActivityFIS, MonotoneActivityFIS

BLUE, ORANGE, AQUA, INK, INK2, MUTED, GRID = F.BLUE, F.ORANGE, F.AQUA, F.INK, F.INK2, F.MUTED, F.GRID
TYPE_COLORS = {"AI-enthusiastic": BLUE, "Ambivalent": ORANGE, "AI-sceptical": AQUA}


def fig_raincloud(d):
    codes = list(pd.Series({a: d[f"TRFS_{a}"].mean() for a in ACTIVITIES}).sort_values().index)
    fig, ax = plt.subplots(figsize=(F.DOUBLE, 4.6))
    rng = np.random.default_rng(1)
    ax.axvspan(0, T_LOW, color=AQUA, alpha=0.07, lw=0)
    ax.axvspan(T_LOW, T_HIGH, color=ORANGE, alpha=0.07, lw=0)
    ax.axvspan(T_HIGH, 100, color=BLUE, alpha=0.07, lw=0)
    for i, a in enumerate(codes):
        x = d[f"TRFS_{a}"].to_numpy()
        kde = stats.gaussian_kde(x, bw_method=0.12)
        xs = np.linspace(15, 85, 400)
        dens = kde(xs)
        dens = dens / dens.max() * 0.42
        ax.fill_between(xs, i + 0.08, i + 0.08 + dens, color=BLUE, alpha=0.35, lw=0)
        ax.plot(xs, i + 0.08 + dens, color=BLUE, lw=0.8)
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        ax.plot([q1, q3], [i, i], color=INK, lw=2.4, solid_capstyle="round")
        ax.scatter([x.mean()], [i], s=26, color="white", edgecolor=INK, zorder=4, linewidth=1)
        sub = rng.choice(x, 400, replace=False)
        ax.scatter(sub + rng.uniform(-0.6, 0.6, sub.size), i - 0.22 + rng.uniform(-0.12, 0.12, sub.size),
                   s=1.2, color=INK2, alpha=0.35, linewidth=0)
        ax.text(86.5, i, f"{x.mean():.1f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(codes)), [ACTIVITIES[a]["name"] for a in codes])
    ax.set_xlim(15, 90)
    ax.set_ylim(-0.6, len(codes) - 0.3)
    for t in (T_LOW, T_HIGH):
        ax.axvline(t, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(25, len(codes) - 0.45, "Human involvement", ha="center", fontsize=7, color=INK2)
    ax.text(50, len(codes) - 0.45, "Partial AI assistance", ha="center", fontsize=7, color=INK2)
    ax.text(75, len(codes) - 0.45, "Full AI replacement", ha="center", fontsize=7, color=INK2)
    ax.set_xlabel("Activity feasibility score (0-100)")
    ax.set_title("Raincloud view: density, quartiles (bar), mean (○) and raw responses", loc="left")
    F._save(fig, "fig11_activity_raincloud")


def fig_shapley(sh, r2):
    sh = sh.sort_values("shapley_R2")
    fig, ax = plt.subplots(figsize=(F.SINGLE * 1.45, 2.8))
    y = np.arange(len(sh))
    ax.hlines(y, 0, sh.shapley_R2, color=GRID, lw=6)
    ax.errorbar(sh.shapley_R2, y, xerr=[sh.shapley_R2 - sh.ci_low, sh.ci_high - sh.shapley_R2],
                fmt="o", color=BLUE, ms=6, mec="white", mew=1.2, elinewidth=1, capsize=0)
    for yi, (hi, s) in enumerate(zip(sh.ci_high, sh.share_pct)):
        ax.text(hi + 0.002, yi, f"{s:.1f}%", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [f"{n} ({f})" for n, f in zip(sh.name, sh.factor)])
    ax.set_xlabel("Shapley share of R² (95% bootstrap CI)")
    ax.set_xlim(0, sh.ci_high.max() * 1.3)
    ax.set_title(f"Exact decomposition: Σφ = R² = {r2:.3f}", loc="left")
    ax.grid(axis="x")
    F._save(fig, "fig12_shapley_r2")


def fig_multilevel():
    ce = pd.read_csv(TABLES / "deep_country_effects.csv")
    ce = ce[ce.n >= 100].sort_values("mean_TRFS")
    fe = pd.read_csv(TABLES / "deep_mixed_model.csv")
    fe = fe[fe.term != "Intercept"].copy()
    fe["f"] = fe.term.str[1:]
    fe = fe.sort_values("beta")
    fig, axes = plt.subplots(1, 2, figsize=(F.DOUBLE, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    y = np.arange(len(ce))
    grand = ce.mean_TRFS.mean() - ce.u.mean()
    ax.axvline(grand, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.errorbar(ce.mean_TRFS, y, xerr=1.96 * ce.se, fmt="o", ms=3.5, color=BLUE,
                elinewidth=0.9, capsize=0, mec="white", mew=0.6)
    ax.set_yticks(y, [f"{c} ({n})" for c, n in zip(ce.country, ce.n)], fontsize=6.3)
    ax.set_xlabel("Country mean TRFS (BLUP ± 95% CI)")
    ax.set_title("Country random intercepts (n ≥ 100)", loc="left")
    ax.grid(axis="x")
    ax = axes[1]
    y = np.arange(len(fe))
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.errorbar(fe.beta, y + 0.12, xerr=[fe.beta - fe.ci_low, fe.ci_high - fe.beta], fmt="o", ms=5,
                color=BLUE, mec="white", mew=1, elinewidth=1, capsize=0, label="Multilevel (country RI)")
    ax.scatter(fe.ols_beta, y - 0.12, marker="D", s=14, color=ORANGE, label="Single-level OLS", zorder=3)
    ax.set_yticks(y, [f"{FACTORS[f]['name']} ({f})" for f in fe.f])
    ax.set_xlabel("Effect on TSI per +1 SD (points)")
    ax.set_title("Fixed effects on the substitution index", loc="left")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.grid(axis="x")
    F._panel(axes[0], "a")
    F._panel(axes[1], "b")
    fig.tight_layout()
    F._save(fig, "fig13_multilevel")


def fig_quantile(q, ols_beta, ols_ci):
    fig, ax = plt.subplots(figsize=(F.SINGLE * 1.45, 2.5))
    ax.fill_between(q.tau, q.ci_low, q.ci_high, color=BLUE, alpha=0.18, lw=0, label="95% CI")
    ax.plot(q.tau, q.beta, color=BLUE, lw=2, marker="o", ms=3.5, label="Quantile regression")
    ax.axhline(ols_beta, color=ORANGE, lw=1.4, label=f"OLS ({ols_beta:.2f})")
    ax.axhspan(ols_ci[0], ols_ci[1], color=ORANGE, alpha=0.12, lw=0)
    ax.set_xlabel("Quantile τ of the teacher substitution index")
    ax.set_ylabel("ΔTSI per +10 TRFS points")
    ax.set_title("Effect of the fuzzy score across the TSI distribution", loc="left")
    ax.legend(loc="lower center", fontsize=6.5, ncol=3)
    ax.set_ylim(min(q.ci_low.min(), ols_ci[0]) - 1.0, max(q.ci_high.max(), ols_ci[1]) + 0.4)
    ax.grid(axis="y")
    F._save(fig, "fig14_quantile_regression")


def fig_typology(d):
    prof = pd.read_csv(TABLES / "deep_typology_k3.csv")
    codes = list(ACTIVITIES)
    labels = [ACTIVITIES[a]["name"].replace(" & ", " &\n").replace(" ", "\n", 1) for a in codes]
    ang = np.linspace(0, 2 * np.pi, len(codes), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=(F.DOUBLE, 3.6))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    for _, r in prof.iterrows():
        v = r[codes].to_numpy(float)
        v = np.concatenate([v, v[:1]])
        ax.plot(ang_c, v, color=TYPE_COLORS[r.label], lw=1.8, label=f"{r.label} ({r.pct:.0f}%)")
        ax.fill(ang_c, v, color=TYPE_COLORS[r.label], alpha=0.10)
    ax.set_xticks(ang, labels, fontsize=6.2)
    ax.set_ylim(1.5, 4.6)
    ax.set_yticks([2, 3, 4], ["2", "3", "4"], fontsize=6, color=INK2)
    ax.spines["polar"].set_color(GRID)
    ax.grid(color=GRID, lw=0.6)
    ax.set_title("Perceived AI capability profiles (1-5)", loc="left", pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=6.5)
    ax2 = fig.add_subplot(1, 2, 2)
    order = list(prof.label)
    act_order = list(pd.Series({a: d[f"TRFS_{a}"].mean() for a in codes}).sort_values().index)
    y = np.arange(len(act_order))
    for lab in order:
        m = d[d.typology3 == lab][[f"TRFS_{a}" for a in act_order]].mean().values
        ax2.plot(m, y, "o", color=TYPE_COLORS[lab], ms=5, mec="white", mew=0.8)
    lo = d.groupby("typology3")[[f"TRFS_{a}" for a in act_order]].mean().min().values
    hi = d.groupby("typology3")[[f"TRFS_{a}" for a in act_order]].mean().max().values
    ax2.hlines(y, lo, hi, color=GRID, lw=2, zorder=0)
    ax2.axvspan(T_HIGH, 80, color=BLUE, alpha=0.07, lw=0)
    ax2.axvspan(20, T_LOW, color=AQUA, alpha=0.07, lw=0)
    for t in (T_LOW, T_HIGH):
        ax2.axvline(t, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax2.set_xlim(20, 80)
    ax2.set_yticks(y, [ACTIVITIES[a]["name"] for a in act_order], fontsize=6.8)
    ax2.set_xlabel("Mean activity feasibility (0-100)")
    ax2.set_title("Activity feasibility by student typology", loc="left")
    ax2.grid(axis="x")
    fig.text(0.01, 0.97, "a", fontsize=9, fontweight="bold")
    fig.text(0.50, 0.97, "b", fontsize=9, fontweight="bold")
    fig.tight_layout()
    F._save(fig, "fig15_typology")


def fig_surfaces3d():
    g = np.linspace(1, 5, 61)
    C, R = np.meshgrid(g, g)
    e = np.full(C.size, 3.0)
    zm = ActivityFIS().predict(C.ravel(), R.ravel(), e)[0].reshape(C.shape)
    zw = MonotoneActivityFIS().predict(C.ravel(), R.ravel(), e).reshape(C.shape)
    fig = plt.figure(figsize=(F.DOUBLE, 3.3))
    for i, (z, ttl) in enumerate([(zm, "Mamdani (min-max, centroid)"),
                                  (zw, "Monotone weighted-average (Prop. 3)")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot_surface(C, R, z, cmap=F.BLUES, linewidth=0, antialiased=True, rcount=61, ccount=61,
                        vmin=15, vmax=85)
        ax.contour(C, R, z, levels=[T_LOW, T_HIGH], colors=ORANGE, linewidths=1.2, offset=10)
        ax.set_zlim(10, 85)
        ax.set_xlabel("Capability", fontsize=7, labelpad=-2)
        ax.set_ylabel("Reliability", fontsize=7, labelpad=-2)
        ax.set_zlabel("Feasibility", fontsize=7, labelpad=-4)
        ax.tick_params(labelsize=6, pad=-2)
        ax.view_init(elev=24, azim=-128)
        ax.set_title(ttl, fontsize=8, loc="left")
        for p in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            p.set_facecolor("white")
            p.set_edgecolor(GRID)
    fig.text(0.02, 0.93, "a", fontsize=9, fontweight="bold")
    fig.text(0.52, 0.93, "b", fontsize=9, fontweight="bold")
    fig.subplots_adjust(wspace=0.05, left=0.0, right=0.98)
    F._save(fig, "fig16_surfaces_3d")


def fig_effects(eff, comp, delta, sd):
    eff = eff.sort_values("mean")
    fig, axes = plt.subplots(1, 2, figsize=(F.DOUBLE, 2.9), sharey=True)
    ax = axes[0]
    y = np.arange(len(eff))
    band = delta / sd
    ax.axvspan(-band, band, color=MUTED, alpha=0.15, lw=0, label=f"Equivalence bounds (±{delta:.0f} pts)")
    ax.axvline(0, color=MUTED, lw=0.7)
    col = [BLUE if e else ORANGE for e in eff.equivalent_to_50]
    ax.hlines(y, eff.d_ci_low, eff.d_ci_high, color=col, lw=2)
    ax.scatter(eff.cohen_d_vs_50, y, color=col, s=28, zorder=3, edgecolor="white", linewidth=1)
    ax.set_yticks(y, list(eff.activity))
    ax.set_xlabel("Cohen's d vs. neutral score 50 (95% CI)")
    ax.set_title("Effect sizes and TOST equivalence", loc="left")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", lw=2, label="Equivalent to 50 (TOST p<.05)"),
                       Line2D([], [], color=ORANGE, marker="o", lw=2, label="Not equivalent")],
              loc="lower right", fontsize=6.3)
    ax.grid(axis="x")
    ax = axes[1]
    c = comp.set_index("activity").loc[eff.activity]
    ax.hlines(y, c.wa, c.mamdani, color=GRID, lw=3)
    ax.scatter(c.mamdani, y, color=BLUE, s=26, zorder=3, label="Mamdani", edgecolor="white", linewidth=1)
    ax.scatter(c.wa, y, color=ORANGE, s=26, zorder=3, marker="D", label="Monotone WA",
               edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Mean activity feasibility (0-100)")
    ax.set_title("Inference variant comparison", loc="left")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.grid(axis="x")
    F._panel(axes[0], "a")
    F._panel(axes[1], "b")
    fig.tight_layout()
    F._save(fig, "fig17_effects_and_variants")


def make_all(d, prop, comp, sh, q, eff, OUT):
    fig_raincloud(d)
    fig_shapley(sh, OUT["shapley"]["R2_total"])
    fig_multilevel()
    fig_quantile(q, OUT["quantile"]["ols_beta"], OUT["quantile"]["ols_ci"])
    fig_typology(d)
    fig_surfaces3d()
    sd = float(np.mean([d[f"TRFS_{a}"].std() for a in ACTIVITIES]))
    fig_effects(eff, comp, OUT["activity_effects"]["delta"], sd)
