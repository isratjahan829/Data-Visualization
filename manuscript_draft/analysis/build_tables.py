"""Build manuscript table rows (JSON) from the extended analysis outputs. Usage: python build_tables.py <extended_dir> <out.json>"""
import sys, json, numpy as np, pandas as pd
ext, out = sys.argv[1:3]
f3 = lambda x: f'{x:.3f}'.replace('-', '−')
fp = lambda p: '<0.001' if p < 0.001 else f'{p:.3f}'
A = pd.read_csv(f'{ext}/all_models_extended.csv')
rng = np.random.default_rng(0)
T = {}
# extended all-model table
rows = []
for ch, lab in [('62ch', '62 electrodes'), ('6ch', 'Six electrodes')]:
    rows.append([lab] + [''] * 10)
    for _, r in A[A.ch == ch].sort_values('rank_calibrated').iterrows():
        rows.append([r.model, f3(r.acc_zs), f3(r.f1_zs), f3(r.acc_cal), f3(r.acc_cal_sd), f3(r.f1_cal), f'{100*r.gain:.1f} ({100*r.gain_lo:.1f}, {100*r.gain_hi:.1f})'.replace('-', '−'), f'{100*r.cv:.1f}', f3(r.worst), f3(r.best), f'{r.rank_calibrated:.2f}'])
T['allmodels'] = rows
# confusion-matrix metrics
E = pd.read_csv(f'{ext}/cm_metrics_summary.csv'); rows = []
for _, r in E.sort_values(['model', 'ch', 'protocol'], ascending=[False, True, False]).iterrows():
    rows.append([r.model, '62' if r.ch == '62ch' else '6', r.protocol.capitalize()] + [f3(r[k]) for k in ['acc', 'bacc', 'macro_p', 'macro_f1', 'w_f1', 'kappa', 'mcc', 'gmean', 'spec']] + [f'{r.itr_min:.2f}'])
T['cmext'] = rows
# kappa CI for text
T['kappa_ci'] = {f'{r.model}|{r.ch}|{r.protocol}': [round(r.kappa_lo, 3), round(r.kappa_hi, 3)] for _, r in E.iterrows()}
# per-class
P = pd.read_csv(f'{ext}/clisa_per_class.csv'); rows = []
for ch, lab in [('62ch', '62 electrodes'), ('6ch', 'Six electrodes')]:
    rows.append([lab] + [''] * 8)
    for c in ['Neutral', 'Sad', 'Fear', 'Happy']:
        z = P[(P.ch == ch) & (P.protocol == 'zero-shot')].iloc[0]; k = P[(P.ch == ch) & (P.protocol == 'calibrated')].iloc[0]
        rows.append([c] + [f3(z[f'{m}_{c}']) for m in ['P', 'R', 'F1', 'Spec']] + [f3(k[f'{m}_{c}']) for m in ['P', 'R', 'F1', 'Spec']])
T['perclass'] = rows
# effect sizes
F = pd.read_csv(f'{ext}/effect_sizes.csv'); rows = []
for pr in ['calibrated', 'zero-shot']:
    for ch, lab in [('62ch', '62 electrodes'), ('6ch', 'six electrodes')]:
        rows.append([f'{pr.capitalize()}, {lab}'] + [''] * 7)
        for _, r in F[(F.protocol == pr) & (F.ch == ch)].iterrows():
            rows.append([f'vs {r.comparator}', f'{100*r["diff"]:.1f} ({100*r.lo:.1f}, {100*r.hi:.1f})'.replace('-', '−'), f'{r.dz:.2f}'.replace('-', '−'), f'{r.r_rb:.2f}'.replace('-', '−'), f'{r.wins}/15', fp(r.p_t), fp(r.p_w), fp(r.p_holm)])
T['paired'] = rows
# friedman
R = pd.read_csv(f'{ext}/friedman.csv'); rows = []
for _, r in R.iterrows():
    s = A[A.ch == r.ch].sort_values(f'rank_{r.protocol}'); b = s.iloc[0]; c = s[s.model == 'CLISA-EWU'].iloc[0]
    rows.append(['62' if r.ch == '62ch' else '6', r.protocol.capitalize(), str(r.k), str(r.n), f'{r.chi2:.1f}', fp(r.p), f'{r.kendall_W:.2f}', f'{r.CD:.2f}', f'{b.model} ({b[f"rank_{r.protocol}"]:.2f})', f'{c[f"rank_{r.protocol}"]:.2f}'])
T['friedman'] = rows
# variance decomposition
V = pd.read_csv(f'{ext}/variance_decomposition.csv'); rows = []
for _, r in V.iterrows():
    rows.append(['62' if r.ch == '62ch' else '6', r.protocol.capitalize(), f'{r.pct_participant:.1f}', f'{r.pct_seed:.1f}', f'{r.pct_resid:.1f}', f'{r.F_participant:.2f} ({fp(r.p_participant)})', f'{r.F_seed:.2f} ({fp(r.p_seed)})', f'{r.ICC:.2f}'])
T['vardec'] = rows
json.dump(T, open(out, 'w'), indent=1, ensure_ascii=False); print({k: len(v) for k, v in T.items()}); print(T['kappa_ci'])
