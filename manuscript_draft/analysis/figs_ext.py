"""Figures for the extended analysis. Usage: python figs_ext.py <extended_dir> <per_fold_csv> <fig_dir>"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
ext, src, fig = sys.argv[1:4]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
B, O, G = '#2a78d6', '#eb6834', '#9a9993'
A = pd.read_csv(f'{ext}/all_models_extended.csv'); FR = pd.read_csv(f'{ext}/friedman.csv')

# --- Average ranks with Nemenyi critical difference ---
f, axs = plt.subplots(2, 2, figsize=(7.2, 6.4), sharex=True)
for ax, (ch, pr, t) in zip(axs.flat, [('62ch', 'zero-shot', 'a  62 electrodes, zero-shot'), ('62ch', 'calibrated', 'b  62 electrodes, calibrated'), ('6ch', 'zero-shot', 'c  6 electrodes, zero-shot'), ('6ch', 'calibrated', 'd  6 electrodes, calibrated')]):
    s = A[A.ch == ch].sort_values(f'rank_{pr}', ascending=False); r = s[f'rank_{pr}'].values
    fr = FR[(FR.ch == ch) & (FR.protocol == pr)].iloc[0]; best = r.min(); cd = fr.CD
    y = np.arange(len(s))
    ax.axvspan(best, best + cd, color='#e6eef9', zorder=0)
    ax.scatter(r, y, s=26, color=[O if m == 'CLISA-EWU' else B for m in s.model], zorder=3, ec='white', lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(s.model, fontsize=7.5)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == 'CLISA-EWU': lbl.set_fontweight('bold')
    ax.set_title(t, loc='left', fontweight='bold', fontsize=8.5)
    ax.text(0.98, 0.97, f"Friedman χ²={fr.chi2:.1f}, p<0.001\nKendall W={fr.kendall_W:.2f}, CD={cd:.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=7, color='#52514e')
    ax.grid(axis='x', color='#ecebe7', lw=0.6); ax.set_xlim(0.5, 15.5)
f.supxlabel('Average rank across 15 held-out participants (1 = best); shaded band = Nemenyi critical difference from the best model', fontsize=8)
f.tight_layout(); f.savefig(f'{fig}/new_average_ranks_cd.png', dpi=300); plt.close(f)

# --- Mean confusion matrices (CLISA-EWU) ---
M = json.load(open(f'{ext}/clisa_mean_confusion.json')); cls = ['Neutral', 'Sad', 'Fear', 'Happy']
cmap = LinearSegmentedColormap.from_list('b', ['#ffffff', '#cfe0f6', '#2a78d6', '#123d73'])
f, axs = plt.subplots(2, 2, figsize=(6.4, 5.8))
for ax, (k, t) in zip(axs.flat, [('62ch_cm_a', 'a  62 electrodes, zero-shot'), ('62ch_cm_b', 'b  62 electrodes, calibrated'), ('6ch_cm_a', 'c  6 electrodes, zero-shot'), ('6ch_cm_b', 'd  6 electrodes, calibrated')]):
    m = np.array(M[k]) * 100; im = ax.imshow(m, cmap=cmap, vmin=0, vmax=80)
    for i in range(4):
        for j in range(4): ax.text(j, i, f'{m[i, j]:.1f}', ha='center', va='center', fontsize=8, color='white' if m[i, j] > 45 else '#0b0b0b')
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(cls, fontsize=7.5); ax.set_yticklabels(cls, fontsize=7.5)
    ax.set_title(t, loc='left', fontweight='bold', fontsize=8.5); ax.spines[:].set_visible(False)
for ax in axs[1]: ax.set_xlabel('Predicted class')
for ax in axs[:, 0]: ax.set_ylabel('True class')
for ax in axs[:, 1]: ax.set_yticklabels([])
cb = f.colorbar(im, ax=axs, shrink=0.6, pad=0.02); cb.set_label('Row-normalised percentage (mean of 45 runs)')
f.savefig(f'{fig}/new_mean_confusion.png', dpi=300, bbox_inches='tight'); plt.close(f)

# --- Per-class metrics (CLISA-EWU, calibrated) ---
pc = pd.read_csv(f'{ext}/clisa_per_class.csv')
f, axs = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
for ax, met, t in zip(axs, ['P', 'R', 'F1', 'Spec'], ['Precision', 'Recall', 'F1 score', 'Specificity']):
    x = np.arange(4); w = 0.38
    for off, ch, col in [(-w / 2, '62ch', B), (w / 2, '6ch', O)]:
        row = pc[(pc.ch == ch) & (pc.protocol == 'calibrated')].iloc[0]
        ax.bar(x + off, [row[f'{met}_{c}'] for c in cls], w * 0.92, color=col, label='62 electrodes' if ch == '62ch' else '6 electrodes')
    ax.set_xticks(x); ax.set_xticklabels(cls, rotation=35, fontsize=7.5); ax.set_title(t, fontsize=8.5, fontweight='bold', loc='left'); ax.set_ylim(0, 1); ax.grid(axis='y', color='#ecebe7', lw=0.6)
axs[0].set_ylabel('Mean over 15 participants'); f.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.02))
f.tight_layout(rect=(0, 0, 1, 0.92)); f.savefig(f'{fig}/new_per_class_metrics.png', dpi=300); plt.close(f)

# --- Calibration gain vs zero-shot, and 62 vs 6 agreement ---
d = pd.read_csv(src); d = d[d.model_id == 'M20']
s = {ch: d[d.ch == ch].groupby('test_sub')[['acc_a', 'acc_b']].mean() for ch in ['62ch', '6ch']}
f, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
for ch, col in [('62ch', B), ('6ch', O)]:
    x = s[ch].acc_a; g = s[ch].acc_b - s[ch].acc_a; rho, p = stats.spearmanr(x, g)
    axs[0].scatter(x, g * 100, s=26, color=col, ec='white', lw=0.8, label=f"{'62' if ch == '62ch' else '6'} electrodes (ρ={rho:.2f}, p={p:.3f})".replace('p=0.000', 'p<0.001'))
    b = np.polyfit(x, g * 100, 1); xx = np.linspace(x.min(), x.max(), 10); axs[0].plot(xx, np.polyval(b, xx), color=col, lw=1.2)
axs[0].set_xlabel('Zero-shot accuracy'); axs[0].set_ylabel('Calibration gain (percentage points)'); axs[0].set_title('a  Gain versus zero-shot accuracy', loc='left', fontweight='bold', fontsize=8.5); axs[0].legend(frameon=False, fontsize=7)
for pr, col, lab in [('acc_a', B, 'Zero-shot'), ('acc_b', O, 'Calibrated')]:
    rho, p = stats.spearmanr(s['62ch'][pr], s['6ch'][pr])
    axs[1].scatter(s['62ch'][pr], s['6ch'][pr], s=26, color=col, ec='white', lw=0.8, label=f'{lab} (ρ={rho:.2f}, p={p:.3f})'.replace('p=0.000', 'p<0.001'))
axs[1].plot([0.2, 0.85], [0.2, 0.85], ls=':', color=G, lw=1); axs[1].set_xlim(0.2, 0.85); axs[1].set_ylim(0.2, 0.85)
axs[1].set_xlabel('Accuracy, 62 electrodes'); axs[1].set_ylabel('Accuracy, 6 electrodes'); axs[1].set_title('b  Agreement between montages', loc='left', fontweight='bold', fontsize=8.5); axs[1].legend(frameon=False, fontsize=7, loc='upper left')
for ax in axs: ax.grid(color='#ecebe7', lw=0.6)
f.tight_layout(); f.savefig(f'{fig}/new_gain_and_agreement.png', dpi=300); plt.close(f)
print('ok')
