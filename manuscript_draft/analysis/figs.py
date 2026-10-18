import pandas as pd, numpy as np, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','axes.spines.top':False,'axes.spines.right':False})
B,O,G='#2a78d6','#eb6834','#9a9993'
d=pd.read_csv(sys.argv[1]); d=d[d.model_id.notna()&d.acc_b.notna()]
names={'M11':'Shallow MLP','M12':'Deep MLP','M13':'LSTM','M14':'GRU','M15':'Conv1D','M16':'Transformer','M17':'EEG Conformer','M18':'ChanDrop Transf.','M19':'DANN','M20':'CLISA-EWU','M21':'SimCLR','M22':'BYOL','M23':'PseudoLabel','M24':'MixMatch','M25':'DANCE teacher','M26':'DANCE student'}
rng=np.random.default_rng(0)
def ci(x): m=[rng.choice(x,len(x)).mean() for _ in range(4000)]; return np.percentile(m,[2.5,97.5])
fig,axs=plt.subplots(1,2,figsize=(7.2,4.4),sharex=True)
for ax,ch,t in zip(axs,['62ch','6ch'],['a  62 electrodes','b  6 electrodes']):
    rows=[]
    for m,g in d[d.ch==ch].groupby('model_id'):
        s=g.groupby('test_sub')[['acc_a','acc_b']].mean()
        rows.append((names[m],s.acc_a.mean(),ci(s.acc_a.values),s.acc_b.mean(),ci(s.acc_b.values)))
    rows.sort(key=lambda r:r[3])
    for i,(n,a,ac,b,bc) in enumerate(rows):
        ax.plot([a,b],[i,i],color='#d6d5d0',lw=2,zorder=1)
        ax.errorbar(a,i,xerr=[[a-ac[0]],[ac[1]-a]],fmt='o',color=B,ms=5,lw=1,capsize=0,zorder=3,mec='white',mew=1)
        ax.errorbar(b,i,xerr=[[b-bc[0]],[bc[1]-b]],fmt='o',color=O,ms=5,lw=1,capsize=0,zorder=3,mec='white',mew=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    for lbl in ax.get_yticklabels():
        if lbl.get_text()=='CLISA-EWU': lbl.set_fontweight('bold')
    ax.axvline(0.25,color=G,ls=':',lw=1); ax.text(0.252,-0.9,'chance',color='#52514e',fontsize=7)
    ax.set_title(t,loc='left',fontsize=9,fontweight='bold'); ax.set_xlim(0.2,0.75); ax.grid(axis='x',color='#ecebe7',lw=0.6)

axs[0].plot([],[],'o',color=B,label='Zero-shot (no target data)'); axs[0].plot([],[],'o',color=O,label='After 20-window calibration')
fig.legend(loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(0.55,1.02))
fig.supxlabel('Balanced accuracy (subject-level mean, 95% bootstrap CI)',fontsize=9); fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig('fig/new_zeroshot_vs_calibrated.png',dpi=300)
# fig 3 per-subject
c=d[d.model_id=='M20']
s62=c[c.ch=='62ch'].groupby('test_sub')[['acc_a','acc_b']].mean(); s6=c[c.ch=='6ch'].groupby('test_sub')[['acc_a','acc_b']].mean()
fig,axs=plt.subplots(1,2,figsize=(7.2,3.4),sharey=True)
for ax,met,t in zip(axs,['acc_a','acc_b'],['a  Zero-shot','b  Calibrated (20 windows)']):
    for sub in s62.index:
        ax.plot([0,1],[s62.loc[sub,met],s6.loc[sub,met]],color='#c9c8c2',lw=1,zorder=1)
    ax.scatter(np.zeros(15),s62[met],color=B,s=28,zorder=2,ec='white',lw=1); ax.scatter(np.ones(15),s6[met],color=O,s=28,zorder=2,ec='white',lw=1)
    ax.plot([-0.12,0.12],[s62[met].mean()]*2,color='#0b0b0b',lw=2); ax.plot([0.88,1.12],[s6[met].mean()]*2,color='#0b0b0b',lw=2)
    ax.axhline(0.25,color=G,ls=':',lw=1)
    ax.set_xticks([0,1]); ax.set_xticklabels(['62 electrodes','6 electrodes']); ax.set_xlim(-0.4,1.4)
    ax.set_title(t,loc='left',fontsize=9,fontweight='bold'); ax.grid(axis='y',color='#ecebe7',lw=0.6)
axs[0].set_ylabel('CLISA-EWU balanced accuracy\n(per held-out participant, mean of 3 seeds)')
fig.tight_layout(); fig.savefig('fig/new_per_participant_montage.png',dpi=300)
print('ok')
