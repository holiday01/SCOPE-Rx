"""
Fig 5 — Method ablation (Tier A embedding, Tier B deconv, Tier C scoring, MPNN held-out).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _style import (use_style, panel_label, save_both,
                    GENEFORMER_COLOR, SCVI_COLOR, PCA_COLOR, RANDOM_COLOR,
                    ATTENTION_COLOR, NNLS_COLOR, SCADEN_COLOR,
                    LIHC_COLOR, LUAD_COLOR)
use_style()

ROOT = Path('/home/holiday01/drug_sc')
A = pd.read_parquet(ROOT/'results/ablation_table/embedding.parquet')
B = pd.read_parquet(ROOT/'results/ablation_table/deconv.parquet')
C = pd.read_parquet(ROOT/'results/ablation_table/score_fusion.parquet')
M = pd.read_parquet(ROOT/'results/ablation_mpnn_holdout/cv_results.parquet')

fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.6), constrained_layout=True)

# ----- Panel a: Tier A embedding -----
ax = axes[0]
labels_a = [v.replace('_',' ').replace('A0 ','').replace('A1 ','').replace('A2 ','').replace('A3 ','')
            for v in A['variant']]
colors_a = [GENEFORMER_COLOR, SCVI_COLOR, PCA_COLOR, RANDOM_COLOR][:len(A)]
x = np.arange(len(A)); w = 0.27
ax.bar(x-w, A['cox_p05'],  w, label='p<0.05',  color='#9ec5e8', edgecolor='k', lw=0.4)
ax.bar(x,   A['cox_p01'],  w, label='p<0.01',  color='#4a8db8', edgecolor='k', lw=0.4)
ax.bar(x+w, A['cox_p001'], w, label='p<0.001', color='#1f4f70', edgecolor='k', lw=0.4)
ax.set_xticks(x); ax.set_xticklabels(labels_a, rotation=20, ha='right')
ax.set_ylabel('TCGA-LUAD Cox-significant prototypes')
ax.set_title('Tier A — embedding')
ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1), frameon=False, fontsize=6.5)
panel_label(ax, 'a')

# ----- Panel b: Tier B deconv -----
ax = axes[1]
labels_b = ['Attention\n(SCOPE-Rx)', 'NNLS', 'Scaden-MLP']
colors_b = [ATTENTION_COLOR, NNLS_COLOR, SCADEN_COLOR]
x = np.arange(len(B)); w = 0.36
ax.bar(x-w/2, B['entropy_pct']*100, w, label='Composition entropy (% of max)',
       color=[ATTENTION_COLOR, NNLS_COLOR, SCADEN_COLOR], edgecolor='k', lw=0.4)
ax2 = ax.twinx()
ax2.bar(x+w/2, B['top1_collapse_pct']*100, w, label='Patients > 50% in one prototype (%)',
        color=[ATTENTION_COLOR, NNLS_COLOR, SCADEN_COLOR], edgecolor='k', lw=0.4, alpha=0.45, hatch='///')
ax.set_xticks(x); ax.set_xticklabels(labels_b)
ax.set_ylabel('Composition entropy (% of max)', color='#222')
ax2.set_ylabel('Top-1 collapse (%)', color='#222')
ax.set_ylim(0, 100); ax2.set_ylim(0, 100)
ax.set_title('Tier B — deconv method')
# Combine legends
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5, -0.20),
          frameon=False, fontsize=6, ncol=2)
panel_label(ax, 'b')

# ----- Panel c: Tier C score fusion -----
ax = axes[2]
soc_drugs = ['gefitinib','trametinib','osimertinib','selumetinib','alectinib',
             'erlotinib','afatinib','brigatinib','pemetrexed']
short_lab = {'C0_kill+0.5*onc+0.7*prior':'C0\n(default)',
             'C1_kill_only':'C1\nkill only',
             'C2_kill+0.5*onc':'C2\nkill+onc',
             'C3_prior_only':'C3\nprior only',
             'C4_equal_third':'C4\nequal'}
xlabels = [short_lab.get(v, v) for v in C['variant']]
x = np.arange(len(C))
colors_c = ['#2ca02c'] + ['#999999']*4
colors_c[0] = ATTENTION_COLOR  # default highlighted
# default and equal both work — keep default green and equal a darker cyan
ax.bar(x, C['mean_pct'], 0.7, color=['#2ca02c','#bb6464','#bb9b64','#bb6488','#3a8a8c'],
       edgecolor='k', lw=0.4)
for i, (mp, mr) in enumerate(zip(C['mean_pct'], C['max_rank'])):
    ax.text(i, mp+0.5, f'max={int(mr)}', ha='center', va='bottom', fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7)
ax.set_ylabel('Mean rank (%) of 9 LUAD SOC drugs')
ax.set_title('Tier C — score-fusion weights')
ax.set_ylim(0, max(C['mean_pct'])*1.25)
panel_label(ax, 'c')

# ----- Panel d: MPNN held-out -----
ax = axes[3]
order = ['MPNN','CellLine_mean','Random_emb','ECFP4+Ridge']
M['method'] = pd.Categorical(M['method'], categories=order, ordered=True)
M_sorted = M.sort_values('method')
parts = ax.violinplot([M[M['method']==m]['pearson'].values for m in order],
                      positions=range(len(order)), showmeans=True, showmedians=False, widths=0.7)
colors_d = [ATTENTION_COLOR, '#88aabb', '#aaaaaa', '#cc6644']
for body, c in zip(parts['bodies'], colors_d):
    body.set_facecolor(c); body.set_edgecolor('k'); body.set_linewidth(0.4); body.set_alpha(0.7)
parts['cmeans'].set_color('k'); parts['cmeans'].set_linewidth(0.8)
parts['cbars'].set_color('k'); parts['cbars'].set_linewidth(0.6)
parts['cmins'].set_color('k'); parts['cmaxes'].set_color('k')
# Add medians text (below x-axis, not above)
for i, m in enumerate(order):
    med = M[M['method']==m]['pearson'].median()
    ax.text(i, -0.21, f'med {med:+.2f}', ha='center', va='top', fontsize=6.5,
            transform=ax.get_xaxis_transform())
ax.set_xticks(range(len(order)))
ax.set_xticklabels(['MPNN','Cell-line\nmean','Random\nemb','ECFP4 +\nRidge'], fontsize=7)
ax.set_ylabel('Per-drug Pearson r (held-out, 5-fold CV)')
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_title(f'MPNN unseen-drug ({len(M)//4} drugs)')
panel_label(ax, 'd')

save_both(fig, ROOT/'results/figures_main/fig5_ablation')
print(f'Saved Fig 5')
