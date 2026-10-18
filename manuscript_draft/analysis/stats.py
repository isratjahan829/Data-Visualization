import pandas as pd, numpy as np, json, sys
from scipy import stats
d=pd.read_csv(sys.argv[1])
d=d[d.model_id.notna()&d.acc_b.notna()].copy()
names={'M11':'Shallow MLP','M12':'Deep MLP','M13':'LSTM','M14':'GRU','M15':'Conv1D','M16':'Transformer','M17':'EEG Conformer','M18':'ChanDrop Transformer','M19':'DANN','M20':'CLISA-EWU','M21':'SimCLR','M22':'BYOL','M23':'PseudoLabel','M24':'MixMatch','M25':'DANCE teacher','M26':'DANCE student'}
rng=np.random.default_rng(0)
def boot(x,B=10000):
    x=np.asarray(x); m=[rng.choice(x,len(x)).mean() for _ in range(B)]; return np.percentile(m,[2.5,97.5])
rows=[]
subj={}
for (m,ch),g in d.groupby(['model_id','ch']):
    s=g.groupby('test_sub')[['acc_a','f1_a','acc_b','f1_b']].mean()
    subj[(m,ch)]=s
    for met in ['acc_a','acc_b','f1_b']:
        pass
    lo,hi=boot(s.acc_b); loa,hia=boot(s.acc_a)
    rows.append(dict(model=names[m],ch=ch,n=len(g),accA=g.acc_a.mean(),accA_ci=f"{loa:.3f}-{hia:.3f}",accB=g.acc_b.mean(),accB_sd_runs=g.acc_b.std(),accB_ci=f"{lo:.3f}-{hi:.3f}",f1B=g.f1_b.mean(),gain=g.acc_b.mean()-g.acc_a.mean(),
       min_sub=s.acc_b.min(),max_sub=s.acc_b.max(),best_run=g.acc_b.max()))
T=pd.DataFrame(rows).sort_values(['ch','accB'],ascending=[True,False])
pd.set_option('display.width',250); print(T.round(4).to_string(index=False))
print()
# paired subject-level Wilcoxon CLISA vs comparators
out=[]
for ch in ['62ch','6ch']:
    c=subj[('M20',ch)]
    for m in ['M19','M23','M22','M21','M11','M12','M17']:
        o=subj[(m,ch)]
        for met in ['acc_b','acc_a']:
            diff=(c[met]-o[met]).values
            w=stats.wilcoxon(c[met],o[met])
            t=stats.ttest_rel(c[met],o[met])
            lo,hi=boot(diff)
            out.append(dict(ch=ch,comp=names[m],metric=met,diff=diff.mean(),ci=f"{lo:.3f} to {hi:.3f}",wins=int((diff>0).sum()),W=w.statistic,p_wil=w.pvalue,p_t=t.pvalue))
O=pd.DataFrame(out)
from statsmodels.stats.multitest import multipletests
for met in ['acc_b','acc_a']:
    idx=O.metric==met
    O.loc[idx,'p_holm']=multipletests(O.loc[idx,'p_wil'],method='holm')[1]
print(O.round(4).to_string(index=False))
# 62 vs 6 for CLISA
c62=subj[('M20','62ch')].acc_b; c6=subj[('M20','6ch')].acc_b
print('\nCLISA 62 vs 6 subject-level: diff',(c62-c6).mean(), stats.wilcoxon(c62,c6), 'retention', c6.mean()/c62.mean())
# retention bootstrap
r=[ ]
idx=np.arange(15)
for _ in range(10000):
    b=rng.choice(idx,15); r.append(c6.values[b].mean()/c62.values[b].mean())
print('retention CI',np.percentile(r,[2.5,97.5]))
# chance: one-sample vs 0.25
print('6ch CLISA vs chance', stats.wilcoxon(c6-0.25))
print('per-subject CLISA 62 accB', c62.round(3).to_dict()); print('per-subject CLISA 6 accB', c6.round(3).to_dict())
print('per-subject CLISA 62 accA', subj[('M20','62ch')].acc_a.round(3).to_dict())
# seed variability for CLISA
g=d[(d.model_id=='M20')]
print(g.groupby(['ch','seed']).acc_b.mean())
print('best fold rows'); print(g.sort_values('acc_b').tail(3)[['ch','seed','test_sub','acc_a','acc_b','best_val_f1']])
print('best_val_f1 summary by model'); print(d.groupby('model_id').best_val_f1.describe()[['mean','min','max']])
T.to_csv('summary_table.csv',index=False); O.to_csv('paired_tests.csv',index=False)
