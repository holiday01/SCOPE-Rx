"""
Fig 1 — Pipeline schematic + cohort schema.
(a) Multi-stage pipeline flow chart
(b) Cross-cancer schema (LIHC + LUAD share the same architecture)
(c) Cohort summary stats table panel
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
from _style import use_style, panel_label, save_both, LIHC_COLOR, LUAD_COLOR
use_style()

ROOT = Path('/home/holiday01/drug_sc')

fig = plt.figure(figsize=(12.0, 9.5), constrained_layout=True)
gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 0.7, 1])

# =============================================================
# Panel a: Pipeline flow chart
# =============================================================
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 10); ax.set_ylim(0, 5)
ax.axis('off')

def box(ax, x, y, w, h, text, fc='#e8eef5', ec='#333', fs=7, lw=0.8):
    p = FancyBboxPatch((x,y), w, h, boxstyle='round,pad=0.05', fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs)

def arrow(ax, x0, y0, x1, y1, color='#444'):
    a = FancyArrowPatch((x0,y0), (x1,y1), arrowstyle='->,head_width=2,head_length=3',
                        color=color, lw=0.9, mutation_scale=8)
    ax.add_patch(a)

# Inputs
box(ax, 0.1, 3.6, 1.6, 0.7, 'scRNA atlas\n(tissue-matched)', fc='#fce8d4', fs=7.5)
box(ax, 0.1, 2.5, 1.6, 0.7, 'TCGA bulk\nRNA-seq + clinical', fc='#dbe7f5', fs=7.5)
box(ax, 0.1, 1.4, 1.6, 0.7, 'DepMap pan-cancer\n+ GDSC + PRISM', fc='#dbf5e7', fs=7.5)
box(ax, 0.1, 0.3, 1.6, 0.7, 'External validation\ncohorts (GEO)', fc='#f5dbe7', fs=7.5)

# Tower 1
box(ax, 2.2, 3.6, 1.7, 0.7, 'Geneformer V2-104M\nzero-shot embed', fc='#fff', fs=7.5)
box(ax, 4.1, 3.6, 1.7, 0.7, 'Leiden cluster\n→ N cell-state\nprototypes', fc='#fff', fs=7.5)
arrow(ax, 1.7, 3.95, 2.2, 3.95)
arrow(ax, 3.9, 3.95, 4.1, 3.95)

# Tower 2 — attention deconv
box(ax, 6.0, 3.05, 2.0, 1.3,
    'Trust-tiered\nattention deconv\n(bulk → prototypes,\nDepMap-calibrated trust)',
    fc='#dff5ec', fs=7)
arrow(ax, 5.8, 3.95, 6.0, 3.85)
arrow(ax, 1.7, 2.85, 6.0, 3.45)

# Tower 3 — drug encoder
box(ax, 2.2, 1.4, 1.7, 0.7, 'MPNN drug encoder\n(SMILES → 256-d)', fc='#fff', fs=7.5)
arrow(ax, 1.7, 1.75, 2.2, 1.75)

# T2b scDEAL
box(ax, 4.1, 1.4, 1.7, 0.7, 'scDEAL DANN\n(bulk → drug AUC)', fc='#fff', fs=7.5)
arrow(ax, 3.9, 1.75, 4.1, 1.75)

# Scoring layers (right column)
box(ax, 8.2, 4.1, 1.6, 0.55, 'S_kill = scDEAL × Cox HR × trust', fc='#dff5ec', fs=6.5)
box(ax, 8.2, 3.4, 1.6, 0.55, 'S_onc = PRISM oncology / phase / MOA', fc='#fff5d8', fs=6.5)
box(ax, 8.2, 2.7, 1.6, 0.55, 'S_prior = drug-target × pathway-Cox', fc='#f5d8eb', fs=6.5)
box(ax, 8.2, 1.7, 1.6, 0.7,
    '$S_\\mathrm{final} = z\\,S_\\mathrm{kill}$\n$+\\,0.5\\,zS_\\mathrm{onc} + 0.7\\,zS_\\mathrm{prior}$',
    fc='#222', fs=8)
ax.text(8.2+0.8, 1.7+0.35, '$S_\\mathrm{final} = zS_\\mathrm{kill} + 0.5\\,zS_\\mathrm{onc} + 0.7\\,zS_\\mathrm{prior}$',
        ha='center', va='center', color='white', fontsize=7.5)

arrow(ax, 8.0, 3.7, 8.2, 4.4)  # deconv → kill
arrow(ax, 8.0, 3.7, 8.2, 3.7)
arrow(ax, 8.0, 3.7, 8.2, 3.0)

# Final outputs
box(ax, 8.2, 0.4, 1.6, 0.9, 'Cohort top-K + per-patient top-5\n+ wet-lab brief\n(FACS markers, PDO suggestion)', fc='#eee', fs=6.8)
arrow(ax, 9.0, 1.7, 9.0, 1.3)

# External validation feed-back arrow
box(ax, 6.0, 0.4, 2.0, 0.7, 'Multi-cohort Cox + meta-analysis\n+ composite TCGA-trained risk', fc='#fce8d4', fs=7)
arrow(ax, 1.7, 0.65, 6.0, 0.65)
arrow(ax, 8.0, 0.65, 8.2, 0.85)

ax.set_title('SCOPE-Rx pipeline overview', fontsize=10, pad=4)
panel_label(ax, 'a', x=-0.02, y=1.05)

# =============================================================
# Panel b: cross-cancer schema (text panel)
# =============================================================
ax = fig.add_subplot(gs[1])
ax.set_xlim(0, 10); ax.set_ylim(0, 1)
ax.axis('off')

# LIHC arrow
ax.text(0.5, 0.55, 'LIHC', fontsize=14, fontweight='bold', color=LIHC_COLOR, ha='center')
ax.text(0.5, 0.20, 'HCC scRNA atlas\n(GSE125449/149614/...)\n+ TCGA-LIHC + 2 ext',
        fontsize=6.5, ha='center')

ax.text(9.5, 0.55, 'LUAD', fontsize=14, fontweight='bold', color=LUAD_COLOR, ha='center')
ax.text(9.5, 0.20, 'LUAD scRNA atlas\n(GSE131907)\n+ TCGA-LUAD + 3 ext',
        fontsize=6.5, ha='center')

box_y = 0.40; box_h = 0.30
p = FancyBboxPatch((1.4, box_y), 7.2, box_h, boxstyle='round,pad=0.05',
                   fc='#f5f1e8', ec='#666', lw=1.2)
ax.add_patch(p)
ax.text(5.0, box_y+box_h/2, 'Identical pipeline · 0 hyperparameter retuning · same architecture',
        ha='center', va='center', fontsize=10, fontweight='bold')

ax.annotate('', xy=(1.4, 0.55), xytext=(0.85, 0.55),
            arrowprops=dict(arrowstyle='->', lw=1.6, color=LIHC_COLOR))
ax.annotate('', xy=(9.15, 0.55), xytext=(8.6, 0.55),
            arrowprops=dict(arrowstyle='<-', lw=1.6, color=LUAD_COLOR))

panel_label(ax, 'b', x=-0.02, y=1.05)

# =============================================================
# Panel c: cohort summary table
# =============================================================
ax = fig.add_subplot(gs[2])
ax.axis('off')

import pandas as pd
import numpy as np
import json

# Build cohort summary
cohort_rows = [
    {'Cohort':'TCGA-LIHC','Cancer':'LIHC','Platform':'RNA-seq','Role':'Train',
     'n':423,'Endpoints':'OS','Final composite c-index':'—'},
    {'Cohort':'GSE14520','Cancer':'LIHC','Platform':'Affy HG-U133A 2.0','Role':'External',
     'n':225,'Endpoints':'OS, RFS','Final composite c-index':'0.727 (OS) / 0.685 (RFS)'},
    {'Cohort':'GSE76427','Cancer':'LIHC','Platform':'Illumina HT-12 v4','Role':'External',
     'n':115,'Endpoints':'OS, RFS','Final composite c-index':'0.686 (OS) / 0.639 (RFS)'},
    {'Cohort':'TCGA-LUAD','Cancer':'LUAD','Platform':'RNA-seq','Role':'Train',
     'n':576,'Endpoints':'OS','Final composite c-index':'—'},
    {'Cohort':'GSE68465','Cancer':'LUAD','Platform':'Affy HG-U133A','Role':'External',
     'n':462,'Endpoints':'OS, RFS','Final composite c-index':'0.639 (OS) / 0.591 (RFS)'},
    {'Cohort':'GSE72094','Cancer':'LUAD','Platform':'Affy HuRSTA','Role':'External',
     'n':442,'Endpoints':'OS','Final composite c-index':'0.670 (OS)'},
    {'Cohort':'GSE31210','Cancer':'LUAD','Platform':'Affy HG-U133 Plus 2','Role':'External',
     'n':204,'Endpoints':'OS, RFS','Final composite c-index':'0.779 (OS) / 0.618 (RFS)'},
]
ct = pd.DataFrame(cohort_rows)

# Render as table inside ax
columns = list(ct.columns)
cell_text = ct.values.tolist()
table = ax.table(cellText=cell_text, colLabels=columns,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False); table.set_fontsize(8)
table.scale(1, 1.45)

# Color cells by cancer
for i, row in ct.iterrows():
    color = '#fce4dc' if row['Cancer']=='LIHC' else '#dde9f2'
    for j in range(len(columns)):
        table.get_celld()[(i+1, j)].set_facecolor(color)
        table.get_celld()[(i+1, j)].set_edgecolor('white')
        table.get_celld()[(i+1, j)].set_linewidth(1.2)
# Header row
for j in range(len(columns)):
    table.get_celld()[(0, j)].set_facecolor('#444')
    table.get_celld()[(0, j)].set_text_props(color='white', fontweight='bold')

ax.set_title('Cohorts (2,447 patients across 2 cancers, 5 external validation cohorts)', fontsize=10, pad=8)
panel_label(ax, 'c', x=-0.02, y=1.05)

save_both(fig, ROOT/'results/figures_main/fig1_pipeline')
print('Saved Fig 1')
