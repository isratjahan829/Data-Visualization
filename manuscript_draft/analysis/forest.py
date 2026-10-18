import pandas as pd, numpy as np, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e'})
O=pd.read_csv(sys.argv[1]); O=O[O.metric=='acc_b']
fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),sharey=True)
comps=['DANN','PseudoLabel','BYOL','SimCLR','Shallow MLP','Deep MLP','EEG Conformer']
lab={'Shallow MLP':'Two-layer perceptron','Deep MLP':'Deep perceptron'}
for ax,ch,t in zip(axs,['62ch','6ch'],['a  62 electrodes','b  6 electrodes']):
    for i,c in enumerate(comps[::-1]):
        r=O[(O.ch==ch)&(O.comp==c)].iloc[0]; lo,hi=[float(v) for v in r.ci.replace(' to ',',').split(',')]
        sig=r.p_holm<0.05; col='#2a78d6'
        ax.plot([lo*100,hi*100],[i,i],color=col,lw=1.5)
        ax.scatter(r['diff']*100,i,s=36,color=col if sig else 'white',ec=col,lw=1.5,zorder=3)
        ax.text(31,i,f"p={r.p_holm:.3f}" if r.p_holm>=0.001 else "p<0.001",va="center",fontsize=7.5,color="#52514e",clip_on=False)
    ax.axvline(0,color='#9a9993',lw=1,ls=':')
    ax.set_yticks(range(len(comps))); ax.set_yticklabels([lab.get(c,c) for c in comps[::-1]])
    ax.set_xlim(-5,30); ax.set_title(t,loc='left',fontweight='bold',fontsize=9); ax.grid(axis='x',color='#ecebe7',lw=0.6)
fig.supxlabel('CLISA-EWU minus comparator, calibrated accuracy (percentage points, 95% CI)',fontsize=8.5)
axs[0].scatter([],[],s=36,color='#2a78d6',label='Holm-adjusted p < 0.05'); axs[0].scatter([],[],s=36,color='white',ec='#2a78d6',lw=1.5,label='not significant')
fig.legend(loc='upper center',ncol=2,frameon=False,fontsize=8,bbox_to_anchor=(0.55,1.0))
fig.tight_layout(rect=(0,0,1,0.93)); fig.subplots_adjust(wspace=0.38); fig.savefig('fig/new_pairwise_wilcoxon.png',dpi=300); print('ok')
