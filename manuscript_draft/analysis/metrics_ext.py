"""Extended metric and statistical analysis from per-fold LOSO results.
Usage: python metrics_ext.py <deep_models_all_folds.csv> <out_dir>
Confusion-matrix metrics are available for CLISA-EWU (M20) and DANN (M19) only;
accuracy and macro-F1 are available for every deep model."""
import sys, json, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src); d = d[d.model_id.notna() & d.acc_b.notna()].copy()
NAMES = {'M11':'Two-layer perceptron','M12':'Deep perceptron','M13':'LSTM','M14':'GRU','M15':'Conv1D','M16':'Transformer','M17':'EEG Conformer','M18':'ChanDrop transformer','M19':'DANN','M20':'CLISA-EWU','M21':'SimCLR','M22':'BYOL','M23':'PseudoLabel','M24':'MixMatch','M25':'DANCE teacher','M26':'DANCE student'}
rng = np.random.default_rng(0)
def boot_ci(x, B=10000):
    x = np.asarray(x); m = rng.choice(x, (B, len(x))).mean(1); return np.percentile(m, [2.5, 97.5])

# ---------- 1. Confusion-matrix metrics (CLISA-EWU, DANN) ----------
CLS = ['Neutral', 'Sad', 'Fear', 'Happy']
def cm_metrics(c):
    c = np.asarray(c, float); n = c.sum(); K = len(c)
    tp = np.diag(c); fn = c.sum(1) - tp; fp = c.sum(0) - tp; tn = n - tp - fn - fp
    rec = tp / (tp + fn); prec = np.divide(tp, tp + fp, out=np.zeros(K), where=(tp + fp) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    spec = tn / (tn + fp); acc = tp.sum() / n; po = acc
    pe = (c.sum(0) * c.sum(1)).sum() / n**2; kappa = (po - pe) / (1 - pe)
    t, p = c.sum(1), c.sum(0)
    mcc = (tp.sum() * n - (t * p).sum()) / np.sqrt((n**2 - (p**2).sum()) * (n**2 - (t**2).sum()))
    P = acc; itr = np.log2(K) + (P * np.log2(P) if P > 0 else 0) + ((1 - P) * np.log2((1 - P) / (K - 1)) if P < 1 else 0)
    return dict(acc=acc, bacc=rec.mean(), macro_p=prec.mean(), macro_r=rec.mean(), macro_f1=f1.mean(),
                w_f1=(f1 * t / n).sum(), kappa=kappa, mcc=mcc, gmean=np.prod(rec) ** (1 / K), spec=spec.mean(),
                itr_bits=itr, itr_min=itr * 60 / 4.0, **{f'{a}_{CLS[i]}': v[i] for a, v in [('P', prec), ('R', rec), ('F1', f1), ('Spec', spec)] for i in range(K)})
rows = []
for _, x in d[d.cm_b.notna() & d.model_id.isin(['M19', 'M20'])].iterrows():
    for proto, col in [('zero-shot', 'cm_a'), ('calibrated', 'cm_b')]:
        rows.append(dict(model=NAMES[x.model_id], ch=x.ch, seed=x.seed, sub=x.test_sub, protocol=proto, **cm_metrics(json.loads(x[col]))))
CM = pd.DataFrame(rows); CM.to_csv(f'{out}/cm_metrics_per_run.csv', index=False)
keys = ['acc', 'bacc', 'macro_p', 'macro_r', 'macro_f1', 'w_f1', 'kappa', 'mcc', 'gmean', 'spec', 'itr_bits', 'itr_min']
ext = []
for (m, ch, pr), g in CM.groupby(['model', 'ch', 'protocol']):
    s = g.groupby('sub')[keys].mean(); r = dict(model=m, ch=ch, protocol=pr)
    for k in keys: r[k] = s[k].mean(); lo, hi = boot_ci(s[k]); r[k + '_lo'], r[k + '_hi'] = lo, hi
    ext.append(r)
EXT = pd.DataFrame(ext); EXT.to_csv(f'{out}/cm_metrics_summary.csv', index=False)
pc = CM[CM.model == 'CLISA-EWU'].groupby(['ch', 'protocol', 'sub']).mean(numeric_only=True).groupby(['ch', 'protocol']).mean()
pc[[c for c in pc.columns if '_' in c and c.split('_')[0] in ('P', 'R', 'F1', 'Spec')]].to_csv(f'{out}/clisa_per_class.csv')
# mean row-normalised confusion matrices for CLISA-EWU
mcm = {}
for ch in ['62ch', '6ch']:
    for col in ['cm_a', 'cm_b']:
        mats = [np.array(json.loads(v), float) for v in d[(d.model_id == 'M20') & (d.ch == ch)][col]]
        mcm[f'{ch}_{col}'] = np.mean([m / m.sum(1, keepdims=True) for m in mats], 0).tolist()
json.dump(mcm, open(f'{out}/clisa_mean_confusion.json', 'w'))

# ---------- 2. All-model participant-level table (accuracy, macro-F1) ----------
allrows, subj = [], {}
for (m, ch), g in d.groupby(['model_id', 'ch']):
    s = g.groupby('test_sub')[['acc_a', 'f1_a', 'acc_b', 'f1_b']].mean(); subj[(m, ch)] = s
    gain = s.acc_b - s.acc_a
    allrows.append(dict(model=NAMES[m], ch=ch, acc_zs=s.acc_a.mean(), f1_zs=s.f1_a.mean(), acc_cal=s.acc_b.mean(), acc_cal_sd=s.acc_b.std(ddof=1),
                        f1_cal=s.f1_b.mean(), gain=gain.mean(), gain_lo=boot_ci(gain)[0], gain_hi=boot_ci(gain)[1],
                        cv=s.acc_b.std(ddof=1) / s.acc_b.mean(), worst=s.acc_b.min(), best=s.acc_b.max(),
                        seed_sd=g.groupby('seed').acc_b.mean().std(ddof=1), rho_zs_cal=stats.spearmanr(s.acc_a, s.acc_b).statistic))
ALL = pd.DataFrame(allrows)

# ---------- 3. Friedman test, average ranks, Nemenyi critical difference ----------
fr = []
for ch in ['62ch', '6ch']:
    for col, pr in [('acc_a', 'zero-shot'), ('acc_b', 'calibrated')]:
        ms = sorted([m for (m, c) in subj if c == ch])
        M = np.column_stack([subj[(m, ch)][col].values for m in ms])
        chi, p = stats.friedmanchisquare(*M.T); n, k = M.shape
        ranks = np.mean([stats.rankdata(-row) for row in M], 0)
        q = stats.studentized_range.ppf(0.95, k, 1e6) / np.sqrt(2); cd = q * np.sqrt(k * (k + 1) / (6 * n))
        W = chi / (n * (k - 1))
        fr.append(dict(ch=ch, protocol=pr, k=k, n=n, chi2=chi, p=p, kendall_W=W, CD=cd))
        for m, r in zip(ms, ranks): ALL.loc[(ALL.model == NAMES[m]) & (ALL.ch == ch), f'rank_{pr}'] = r
FR = pd.DataFrame(fr); FR.to_csv(f'{out}/friedman.csv', index=False)
ALL.to_csv(f'{out}/all_models_extended.csv', index=False)

# ---------- 4. Paired effect sizes for CLISA-EWU ----------
eff = []
for ch in ['62ch', '6ch']:
    c = subj[('M20', ch)]
    for m in ['M19', 'M23', 'M22', 'M21', 'M11', 'M12', 'M17']:
        o = subj[(m, ch)]
        for col, pr in [('acc_b', 'calibrated'), ('acc_a', 'zero-shot')]:
            diff = (c[col] - o[col]).values; w = stats.wilcoxon(c[col], o[col])
            nz = diff[diff != 0]; rk = stats.rankdata(np.abs(nz)); rrb = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
            t = stats.ttest_rel(c[col], o[col]); lo, hi = boot_ci(diff)
            eff.append(dict(ch=ch, protocol=pr, comparator=NAMES[m], diff=diff.mean(), lo=lo, hi=hi, dz=diff.mean() / diff.std(ddof=1),
                            r_rb=rrb, wins=int((diff > 0).sum()), W=w.statistic, p_w=w.pvalue, t=t.statistic, p_t=t.pvalue))
EF = pd.DataFrame(eff)
for pr in ['calibrated', 'zero-shot']:
    i = EF.protocol == pr; EF.loc[i, 'p_holm'] = multipletests(EF.loc[i, 'p_w'], method='holm')[1]
EF.to_csv(f'{out}/effect_sizes.csv', index=False)

# ---------- 5. Variance decomposition (participant x seed, CLISA-EWU) ----------
vd = []
for ch in ['62ch', '6ch']:
    for col, pr in [('acc_a', 'zero-shot'), ('acc_b', 'calibrated')]:
        g = d[(d.model_id == 'M20') & (d.ch == ch)].pivot_table(index='test_sub', columns='seed', values=col)
        Y = g.values; n, k = Y.shape; gm = Y.mean()
        ss_s = k * ((Y.mean(1) - gm) ** 2).sum(); ss_r = n * ((Y.mean(0) - gm) ** 2).sum(); ss_t = ((Y - gm) ** 2).sum(); ss_e = ss_t - ss_s - ss_r
        ms_s, ms_r, ms_e = ss_s / (n - 1), ss_r / (k - 1), ss_e / ((n - 1) * (k - 1))
        icc = (ms_s - ms_e) / (ms_s + (k - 1) * ms_e)       # ICC(3,1), consistency across seeds
        F_s = ms_s / ms_e; p_s = 1 - stats.f.cdf(F_s, n - 1, (n - 1) * (k - 1)); F_r = ms_r / ms_e; p_r = 1 - stats.f.cdf(F_r, k - 1, (n - 1) * (k - 1))
        vd.append(dict(ch=ch, protocol=pr, pct_participant=100 * ss_s / ss_t, pct_seed=100 * ss_r / ss_t, pct_resid=100 * ss_e / ss_t, F_participant=F_s, p_participant=p_s, F_seed=F_r, p_seed=p_r, ICC=icc))
VD = pd.DataFrame(vd); VD.to_csv(f'{out}/variance_decomposition.csv', index=False)

# ---------- 6. Correlations ----------
c62, c6 = subj[('M20', '62ch')], subj[('M20', '6ch')]
cor = dict(
    zs_vs_cal_62=stats.spearmanr(c62.acc_a, c62.acc_b), zs_vs_cal_6=stats.spearmanr(c6.acc_a, c6.acc_b),
    gain_vs_zs_62=stats.spearmanr(c62.acc_a, c62.acc_b - c62.acc_a), gain_vs_zs_6=stats.spearmanr(c6.acc_a, c6.acc_b - c6.acc_a),
    m62_vs_m6_cal=stats.spearmanr(c62.acc_b, c6.acc_b), m62_vs_m6_zs=stats.spearmanr(c62.acc_a, c6.acc_a))
json.dump({k: [float(v.statistic), float(v.pvalue)] for k, v in cor.items()}, open(f'{out}/correlations.json', 'w'), indent=1)
# acc vs balanced accuracy agreement
for (m, ch, pr), g in CM.groupby(['model', 'ch', 'protocol']):
    print(f'{m:10s} {ch:5s} {pr:10s} acc={g.acc.mean():.4f} bacc={g.bacc.mean():.4f} max|diff|={np.abs(g.acc - g.bacc).max():.4f}')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(EXT[['model', 'ch', 'protocol'] + keys].round(3).to_string(index=False)); print(pc.round(3).T.to_string())
print(ALL.round(3).sort_values(['ch', 'acc_cal'], ascending=[True, False]).to_string(index=False))
print(FR.round(4).to_string(index=False)); print(EF.round(3).to_string(index=False)); print(VD.round(3).to_string(index=False))
print({k: (round(v.statistic, 3), round(v.pvalue, 4)) for k, v in cor.items()})
