"""Power, equivalence, mixed-model, jackknife, seed-wise, balanced-accuracy and rank-probability analyses.
Usage: python sensitivity.py <deep_models_all_folds.csv> <out.json>"""
import sys, json, warnings, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestPower
from statsmodels.stats.weightstats import ttost_paired
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
d = pd.read_csv(sys.argv[1]); d = d[d.model_id.notna() & d.acc_b.notna()].copy()
N = {'M11': 'Two-layer perceptron', 'M12': 'Deep perceptron', 'M17': 'EEG Conformer', 'M19': 'DANN', 'M20': 'CLISA-EWU', 'M21': 'SimCLR', 'M22': 'BYOL', 'M23': 'PseudoLabel'}
R = {}
subj = lambda m, ch, col: d[(d.model_id == m) & (d.ch == ch)].groupby('test_sub')[col].mean()
# A. power
tp = TTestPower()
R['power'] = {'mde_dz_alpha05': tp.solve_power(nobs=15, alpha=0.05, power=0.8), 'mde_dz_alpha_bonf14': tp.solve_power(nobs=15, alpha=0.05 / 14, power=0.8),
              'mde_dz_alpha05_pow90': tp.solve_power(nobs=15, alpha=0.05, power=0.9)}
for ch in ['62ch', '6ch']:  # translate to accuracy points using SD of paired differences vs DANN
    diff = subj('M20', ch, 'acc_b') - subj('M19', ch, 'acc_b')
    R['power'][f'mde_points_vs_DANN_{ch}'] = 100 * R['power']['mde_dz_alpha05'] * diff.std(ddof=1)
# B. TOST equivalence, zero-shot CLISA vs perceptrons
R['tost'] = {}
for ch in ['62ch', '6ch']:
    for m in ['M11', 'M12']:
        a, b = subj('M20', ch, 'acc_a'), subj(m, ch, 'acc_a'); diff = a - b
        ci90 = stats.t.interval(0.90, 14, loc=diff.mean(), scale=diff.std(ddof=1) / np.sqrt(15))
        for marg in [0.02, 0.03]:
            p, _, _ = ttost_paired(a.values, b.values, -marg, marg)
            R['tost'][f'{ch}|{N[m]}|{int(marg*100)}'] = {'diff': diff.mean(), 'ci90': ci90, 'p_tost': p}
# C. mixed model on run-level data: acc ~ model + seed + (1|participant)
R['mixed'] = {}
for ch in ['62ch', '6ch']:
    for col, pr in [('acc_b', 'calibrated'), ('acc_a', 'zero-shot')]:
        for m in ['M19', 'M23', 'M22', 'M21', 'M11']:
            x = d[(d.ch == ch) & d.model_id.isin(['M20', m])][['model_id', 'seed', 'test_sub', col]].rename(columns={col: 'y'})
            x['clisa'] = (x.model_id == 'M20').astype(int)
            fit = smf.mixedlm('y ~ clisa + C(seed)', x, groups=x['test_sub'], re_formula='~clisa').fit(reml=True, method='lbfgs')
            ci = fit.conf_int().loc['clisa']
            R['mixed'][f'{pr}|{ch}|{N[m]}'] = {'beta': fit.params['clisa'], 'lo': ci[0], 'hi': ci[1], 'p': fit.pvalues['clisa']}
# D. jackknife (leave one participant out)
R['jack'] = {}
for ch in ['62ch', '6ch']:
    for m in ['M19', 'M23', 'M22']:
        a, b = subj('M20', ch, 'acc_b'), subj(m, ch, 'acc_b'); res = []
        for p in a.index:
            aa, bb = a.drop(p), b.drop(p); res.append(((aa - bb).mean(), stats.wilcoxon(aa, bb).pvalue))
        res = np.array(res)
        R['jack'][f'{ch}|{N[m]}'] = {'diff_min': res[:, 0].min(), 'diff_max': res[:, 0].max(), 'p_min': res[:, 1].min(), 'p_max': res[:, 1].max(), 'n_sig': int((res[:, 1] < 0.05).sum())}
    a62, a6 = subj('M20', '62ch', 'acc_b'), subj('M20', '6ch', 'acc_b')
    ret = [a6.drop(p).mean() / a62.drop(p).mean() for p in a62.index]
R['jack']['retention'] = {'min': min(ret), 'max': max(ret)}
# E. seed-wise
R['seed'] = {}
ms = sorted(d.model_id.unique())
for ch in ['62ch', '6ch']:
    for s in sorted(d.seed.unique()):
        g = d[(d.ch == ch) & (d.seed == s)]
        means = g.groupby('model_id').acc_b.mean().sort_values(ascending=False)
        rank = list(means.index).index('M20') + 1
        c = g[g.model_id == 'M20'].set_index('test_sub').acc_b; o = g[g.model_id == 'M19'].set_index('test_sub').acc_b
        R['seed'][f'{ch}|{int(s)}'] = {'clisa_mean': means['M20'], 'rank': rank, 'best': means.index[0], 'diff_vs_dann': (c - o).mean(), 'p_vs_dann': stats.wilcoxon(c, o).pvalue}
# F. balanced accuracy comparison CLISA vs DANN
def ba(cm):
    c = np.array(json.loads(cm), float); return (np.diag(c) / c.sum(1)).mean()
R['ba'] = {}
for ch in ['62ch', '6ch']:
    for col, pr in [('cm_b', 'calibrated'), ('cm_a', 'zero-shot')]:
        g = d[(d.ch == ch) & d.model_id.isin(['M19', 'M20']) & d[col].notna()].copy(); g['ba'] = g[col].map(ba)
        a = g[g.model_id == 'M20'].groupby('test_sub').ba.mean(); b = g[g.model_id == 'M19'].groupby('test_sub').ba.mean()
        R['ba'][f'{pr}|{ch}'] = {'diff': (a - b).mean(), 'p': stats.wilcoxon(a, b).pvalue, 'wins': int(((a - b) > 0).sum())}
# G. bootstrap probability of being the best model
rng = np.random.default_rng(1); R['pbest'] = {}
for ch in ['62ch', '6ch']:
    for col, pr in [('acc_b', 'calibrated'), ('acc_a', 'zero-shot')]:
        M = pd.DataFrame({m: subj(m, ch, col) for m in d[d.ch == ch].model_id.unique()})
        wins = pd.Series(0, index=M.columns)
        for _ in range(10000):
            idx = rng.integers(0, 15, 15); wins[M.iloc[idx].mean().idxmax()] += 1
        p = (wins / 10000).sort_values(ascending=False)
        R['pbest'][f'{pr}|{ch}'] = {k: round(float(v), 4) for k, v in p.head(4).items()}
# H. calibration gain heterogeneity
R['gain'] = {}
for ch in ['62ch', '6ch']:
    g = subj('M20', ch, 'acc_b') - subj('M20', ch, 'acc_a')
    R['gain'][ch] = {'n_gt10': int((g > 0.10).sum()), 'n_gt20': int((g > 0.20).sum()), 'min': g.min(), 'max': g.max(), 'median': g.median()}
json.dump(R, open(sys.argv[2], 'w'), indent=1, default=float)
print(json.dumps(R, indent=1, default=lambda x: round(float(x), 4)))
