"""Publication figures (vector PDF + 300 dpi PNG).

Palette: validated categorical slots 1-3 (blue, orange, aqua) for the three
decision categories, a single blue ramp for magnitude, grey for context marks.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from config import (ACTIVITIES, CATEGORIES, FACTOR_ORDER, FACTORS, FIGURES,  # noqa: E402
                    T_HIGH, T_LOW)
from fuzzy_system import (LEVELS, ActivityFIS, HierarchicalFIS,  # noqa: E402
                          Partition)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
CAT_COLORS = {CATEGORIES[2]: BLUE, CATEGORIES[1]: ORANGE, CATEGORIES[0]: AQUA}
LEVEL_COLORS = [AQUA, ORANGE, BLUE]  # Low, Medium, High
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

SINGLE, DOUBLE = 3.54, 7.48  # inches (90 mm / 190 mm)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
})


def _save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)


def _panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="right", va="bottom")


def fig_membership():
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.0))
    for ax, part, xl, title in [
        (axes[0], Partition.likert(), "Factor score (Likert 1-5)", "Input variables (TE, SE, P, AC, C, A, R, E)"),
        (axes[1], Partition.score(), "Score (0-100)", "Intermediate (PC, IQ, TW) and output (TRFS)"),
    ]:
        x = np.linspace(part.lo, part.hi, 501)
        mu = part.memberships(x)
        for j, lvl in enumerate(LEVELS):
            ax.plot(x, mu[:, j], color=LEVEL_COLORS[j], lw=1.6, label=lvl)
        ax.set_xlabel(xl)
        ax.set_ylabel("Membership degree")
        ax.set_ylim(0, 1.08)
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y")
    for t in (T_LOW, T_HIGH):
        axes[1].axvline(t, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        axes[1].text(t, 1.06, f"{t:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
    axes[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.5))
    _panel(axes[0], "a")
    _panel(axes[1], "b")
    fig.tight_layout()
    _save(fig, "fig2_membership_functions")


def fig_surfaces():
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.7))
    g = np.linspace(0, 100, 81)
    PC, TW = np.meshgrid(g, g)
    fis = HierarchicalFIS()
    from fuzzy_system import mamdani
    z, _ = mamdani([PC.ravel(), np.full(PC.size, 50.0), TW.ravel()], [fis.mid_part] * 3,
                   fis.rules["TRFS"], fis.out_part)
    cs = axes[0].contourf(PC, TW, z.reshape(PC.shape), levels=np.linspace(15, 85, 15), cmap=BLUES)
    axes[0].contour(PC, TW, z.reshape(PC.shape), levels=[T_LOW, T_HIGH], colors="white",
                    linewidths=1.0)
    axes[0].set_xlabel("Pedagogical capability (PC)")
    axes[0].set_ylabel("Trustworthiness (TW)")
    axes[0].set_title("FIS$_4$: TRFS surface (IQ fixed at 50)", loc="left")
    gl = np.linspace(1, 5, 81)
    CAP, REL = np.meshgrid(gl, gl)
    afis = ActivityFIS()
    z2, _ = afis.predict(CAP.ravel(), REL.ravel(), np.full(CAP.size, 3.0))
    axes[1].contourf(CAP, REL, z2.reshape(CAP.shape), levels=np.linspace(15, 85, 15), cmap=BLUES)
    axes[1].contour(CAP, REL, z2.reshape(CAP.shape), levels=[T_LOW, T_HIGH], colors="white",
                    linewidths=1.0)
    axes[1].set_xlabel("Activity capability (1-5)")
    axes[1].set_ylabel("Reliability (1-5)")
    axes[1].set_title("Activity FIS surface (E fixed at 3)", loc="left")
    cb = fig.colorbar(cs, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label("Feasibility score (0-100)")
    cb.outline.set_visible(False)
    _panel(axes[0], "a")
    _panel(axes[1], "b")
    _save(fig, "fig3_control_surfaces")


def fig_factor_profile(ct, phi):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.9), gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axes[0]
    order = ct.sort_values("mean").index
    y = np.arange(len(order))
    ax.errorbar(ct.loc[order, "mean"], y, xerr=ct.loc[order, "sd"], fmt="none",
                ecolor=MUTED, elinewidth=1.0, capsize=0)
    ax.scatter(ct.loc[order, "mean"], y, s=36, color=BLUE, zorder=3,
               edgecolor="white", linewidth=1.2)
    for yi, f in zip(y, order):
        ax.text(ct.loc[f, "mean"], yi + 0.28, f"{ct.loc[f, 'mean']:.2f}", ha="center",
                fontsize=7, color=INK2)
    ax.set_yticks(y, [f"{FACTORS[f]['name']} ({f})" for f in order])
    ax.axvline(3, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xlim(1.5, 5)
    ax.set_xlabel("Factor score, mean ± SD (1-5)")
    ax.grid(axis="x")
    ax.set_title("Perceived AI capability by factor", loc="left")
    ax = axes[1]
    P = phi.loc[FACTOR_ORDER, FACTOR_ORDER].values.copy()
    mask = np.triu(np.ones_like(P, bool), 0)
    Pm = np.ma.array(P, mask=mask)
    im = ax.imshow(Pm, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(P)):
        for j in range(i):
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if P[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(P)), FACTOR_ORDER)
    ax.set_yticks(range(len(P)), FACTOR_ORDER)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Latent factor correlations (CFA)", loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.outline.set_visible(False)
    _panel(axes[0], "a")
    _panel(axes[1], "b")
    fig.tight_layout()
    _save(fig, "fig4_factor_profile")


def fig_trfs(d):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    ax = axes[0]
    bins = np.arange(15, 87, 2.5)
    for lo, hi, cat in [(0, T_LOW, CATEGORIES[0]), (T_LOW, T_HIGH, CATEGORIES[1]),
                        (T_HIGH, 100, CATEGORIES[2])]:
        x = d.TRFS[(d.TRFS >= lo) & (d.TRFS < hi)]
        ax.hist(x, bins=bins, color=CAT_COLORS[cat], edgecolor="white", linewidth=0.5,
                label=f"{cat} ({100 * len(x) / len(d):.1f}%)")
    ax.set_xlabel("Overall TRFS (0-100)")
    ax.set_ylabel("Respondents")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), fontsize=6.8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.set_title("Distribution of the overall feasibility score", loc="left")
    ax = axes[1]
    hb = ax.hexbin(d.TRFS, d.TSI, gridsize=28, cmap=BLUES, mincnt=1, linewidths=0.2,
                   edgecolors="white")
    q = pd.qcut(d.TRFS, 10, duplicates="drop")
    mm = d.groupby(q, observed=True)[["TRFS", "TSI"]].mean()
    ax.plot(mm.TRFS, mm.TSI, color=ORANGE, lw=2, marker="o", ms=4, label="Decile means")
    ax.set_xlabel("Overall TRFS (fuzzy)")
    ax.set_ylabel("TSI (0-100)")
    from scipy.stats import spearmanr
    rho = spearmanr(d.TRFS, d.TSI).statistic
    ax.set_title(f"TRFS vs. teacher substitution index (ρ = {rho:.2f})", loc="left")
    ax.legend(loc="upper left")
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Respondents")
    cb.outline.set_visible(False)
    _panel(axes[0], "a")
    _panel(axes[1], "b")
    fig.tight_layout()
    _save(fig, "fig5_trfs_distribution_validity")


def fig_models(cv, summ):
    order = list(summ.index[::-1])  # best at top
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, metric, xl in [(axes[0], "RMSE", "RMSE (TSI points, lower is better)"),
                           (axes[1], "R2", "R² (higher is better)")]:
        for i, m in enumerate(order):
            v = cv[cv.model == m][metric].values
            c = BLUE if "ANN" in m else MUTED
            ax.scatter(v, i + rng.uniform(-0.15, 0.15, len(v)), s=6, color=c, alpha=0.45,
                       linewidth=0)
            ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=INK if "ANN" not in m else BLUE,
                    lw=2)
        ax.set_yticks(range(len(order)), order)
        ax.set_xlabel(xl)
        ax.grid(axis="x")
    axes[0].set_title("Out-of-fold error, repeated k-fold CV", loc="left")
    axes[1].set_title("Explained variance", loc="left")
    _panel(axes[0], "a")
    _panel(axes[1], "b")
    fig.tight_layout()
    _save(fig, "fig6_model_comparison")


def fig_importance(imp):
    imp = imp.sort_values("ann_perm_importance_R2_drop")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), sharey=True)
    y = np.arange(len(imp))
    labels = [f"{n} ({f})" for n, f in zip(imp.name, imp.factor)]
    axes[0].barh(y, imp.ann_perm_importance_R2_drop, xerr=imp.ann_perm_importance_sd,
                 color=BLUE, height=0.6, error_kw={"ecolor": INK2, "elinewidth": 0.8})
    axes[0].set_xlabel("Drop in R² when permuted")
    axes[0].set_title("ANN permutation importance (TSI)", loc="left")
    axes[1].barh(y, imp.fis_sensitivity_plus1SD, color=ORANGE, height=0.6)
    axes[1].set_xlabel("Mean ΔTRFS for +1 SD in factor")
    axes[1].set_title("Fuzzy system sensitivity (TRFS)", loc="left")
    axes[0].set_yticks(y, labels)
    for ax in axes:
        ax.grid(axis="x")
        ax.axvline(0, color=MUTED, lw=0.6)
    _panel(axes[0], "a")
    _panel(axes[1], "b")
    fig.tight_layout()
    _save(fig, "fig7_factor_importance")


def fig_activities(act, band):
    act = act.sort_values("TRFS_mean")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.3), sharey=True,
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(act))
    ax.axvspan(0, T_LOW, color=AQUA, alpha=0.08, lw=0)
    ax.axvspan(T_LOW, T_HIGH, color=ORANGE, alpha=0.08, lw=0)
    ax.axvspan(T_HIGH, 100, color=BLUE, alpha=0.08, lw=0)
    lo = band.loc[act.code, "p2.5"].values
    hi = band.loc[act.code, "p97.5"].values
    ax.hlines(y, lo, hi, color=MUTED, lw=4, alpha=0.5, label="Range under 500 MF perturbations")
    colors = [CAT_COLORS[c] for c in act.category]
    ax.scatter(act.TRFS_mean, y, s=40, color=colors, edgecolor="white", linewidth=1.2, zorder=3)
    for yi, (m, l, h) in enumerate(zip(act.TRFS_mean, act.ci_low, act.ci_high)):
        ax.text(m + 1.2, yi + 0.22, f"{m:.1f}", fontsize=7, color=INK2)
    ax.set_yticks(y, [f"{n} ({c})" for n, c in zip(act.activity, act.code)])
    ax.set_xlim(25, 75)
    ax.set_xlabel("Activity feasibility score, mean (0-100)")
    for t, txt in [(T_LOW, f"{T_LOW:.0f}"), (T_HIGH, f"{T_HIGH:.0f}")]:
        ax.axvline(t, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_title("Feasibility of AI substitution by activity", loc="left")
    handles = [Patch(color=CAT_COLORS[c], label=c) for c in CATEGORIES[::-1]]
    handles.append(plt.Line2D([], [], color=MUTED, lw=4, alpha=0.5, label="Sensitivity range"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06),
               fontsize=7)
    ax = axes[1]
    left = np.zeros(len(act))
    for col, cat in [("pct_human", CATEGORIES[0]), ("pct_partial", CATEGORIES[1]),
                     ("pct_full", CATEGORIES[2])]:
        ax.barh(y, act[col], left=left, color=CAT_COLORS[cat], height=0.62,
                edgecolor="white", linewidth=1.0)
        for yi, (l, v) in enumerate(zip(left, act[col])):
            if v >= 9:
                ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white")
        left += act[col].values
    ax.set_xlim(0, 100)
    ax.set_xlabel("Respondents per category (%)")
    ax.set_title("Respondent-level classification", loc="left")
    _panel(axes[0], "a")
    _panel(axes[1], "b")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "fig8_activity_feasibility")


def fig_robustness(rob):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.2))
    axes[0].hist(rob.kendall_tau, bins=np.linspace(0.5, 1.0, 21), color=BLUE,
                 edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Kendall's τ with baseline activity ranking")
    axes[0].set_ylabel("MF configurations")
    axes[0].set_title("Rank stability", loc="left")
    vc = rob.category_agreement.round(2).value_counts().sort_index()
    axes[1].bar([f"{100 * k:.0f}%" for k in vc.index], vc.values, color=BLUE, width=0.6)
    axes[1].set_xlabel("Activities keeping their baseline category")
    axes[1].set_ylabel("MF configurations")
    axes[1].set_title("Category stability", loc="left")
    for ax in axes:
        ax.grid(axis="y")
    _panel(axes[0], "a")
    _panel(axes[1], "b")
    fig.tight_layout()
    _save(fig, "fig9_robustness")


def fig_field_heatmap(hm, act):
    order = list(act.sort_values("TRFS_mean", ascending=False).code)
    H = hm[order]
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
    im = ax.imshow(H.values, cmap=BLUES, aspect="auto", vmin=H.values.min() - 5,
                   vmax=H.values.max())
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            v = H.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.8,
                    color="white" if v > np.percentile(H.values, 55) else INK)
    ax.set_xticks(range(len(order)), [ACTIVITIES[a]["name"] for a in order], fontsize=6.5,
                  rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(H.shape[0]), H.index)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Mean feasibility")
    cb.outline.set_visible(False)
    ax.set_title("Activity feasibility by field of study", loc="left")
    _save(fig, "fig10_activity_by_field")


def make_all(d, ct, phi, act, cv, summ, imp, rob, band, hm):
    fig_membership()
    fig_surfaces()
    fig_factor_profile(ct, phi)
    fig_trfs(d)
    fig_models(cv, summ)
    fig_importance(imp)
    fig_activities(act, band)
    fig_robustness(rob)
    fig_field_heatmap(hm, act)
