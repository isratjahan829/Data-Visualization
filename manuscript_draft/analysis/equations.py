# Render the manuscript equations as PNG images (matplotlib mathtext, STIX fonts).
import sys, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'mathtext.fontset': 'stix', 'font.family': 'STIXGeneral'})
EQ = {
 'recall': r'$\mathrm{Recall}_i=\dfrac{TP_i}{TP_i+FN_i},\qquad \mathrm{Precision}_i=\dfrac{TP_i}{TP_i+FP_i}$',
 'bacc': r'$\mathrm{Balanced\ accuracy}=\dfrac{1}{C}\sum_{i=1}^{C}\mathrm{Recall}_i$',
 'f1': r'$F1_i=2\cdot\dfrac{\mathrm{Precision}_i\cdot\mathrm{Recall}_i}{\mathrm{Precision}_i+\mathrm{Recall}_i}$',
 'macrof1': r'$F1_{\mathrm{macro}}=\dfrac{1}{C}\sum_{i=1}^{C}F1_i$',
 'retention': r'$\mathrm{Retention}=\dfrac{BA_{6}}{BA_{62}}$',
 'ntxent': r'$\ell_{i,j}=-\log\dfrac{\exp(\mathrm{sim}(z_i,z_j)/\tau)}{\sum_{k=1,\,k\neq i}^{2N}\exp(\mathrm{sim}(z_i,z_k)/\tau)}$',
}
out = sys.argv[1]; sizes = {}
for k, e in EQ.items():
    fig = plt.figure(figsize=(0.01, 0.01)); fig.text(0, 0, e, fontsize=13)
    p = f'{out}/eq_{k}.png'; fig.savefig(p, dpi=300, bbox_inches='tight', pad_inches=0.03, transparent=False, facecolor='white'); plt.close(fig)
    from PIL import Image; sizes[k] = Image.open(p).size
json.dump(sizes, open(f'{out}/sizes.json', 'w')); print(sizes)
