"""
Graphical abstract for SCOPE-Rx: single horizontal panel
showing pipeline + key cross-cancer result.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
sys.path.insert(0, str(Path(__file__).parent))
from _style import use_style, save_both, LIHC_COLOR, LUAD_COLOR, ATTENTION_COLOR
use_style()

ROOT = Path('/home/holiday01/drug_sc')
fig = plt.figure(figsize=(11, 4.8), constrained_layout=False)

# Layout: 3 vertical zones — Inputs (left), Pipeline (center), Results (right)
ax = fig.add_axes([0, 0, 1, 1])  # full canvas
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

# Title
ax.text(50, 95, 'SCOPE-Rx', ha='center', va='center', fontsize=18, fontweight='bold', color='#222')
ax.text(50, 89, 'Trust-tiered attention deconvolution of single-cell foundation embeddings',
        ha='center', va='center', fontsize=9.5, color='#444')
ax.text(50, 84, 'for cell-state-resolved survival-anchored drug repositioning',
        ha='center', va='center', fontsize=9.5, color='#444')

# === Left zone: Inputs ===
def labbox(x, y, w, h, lines, fc='#fde7d4', fs=8.5):
    p = FancyBboxPatch((x,y), w, h, boxstyle='round,pad=0.4', fc=fc, ec='#666', lw=0.8)
    ax.add_patch(p)
    for i, ln in enumerate(lines):
        ax.text(x+w/2, y+h-3-i*4, ln, ha='center', va='top', fontsize=fs)

labbox(2, 60, 19, 16, ['scRNA atlas', '(tissue-matched)', 'Geneformer V2'])
labbox(2, 40, 19, 16, ['TCGA bulk RNA', '+ overall survival', 'LIHC + LUAD'])
labbox(2, 20, 19, 16, ['DepMap +', 'GDSC + PRISM', '1,684 lines × 1,806 drugs'])
labbox(2, 0, 19, 16, ['1,790 pathways', 'MSigDB / KEGG /', 'Reactome'], fc='#eee0fa')

# === Center zone: Pipeline core ===
core_x, core_y, core_w, core_h = 26, 30, 28, 50
p = FancyBboxPatch((core_x, core_y), core_w, core_h,
                   boxstyle='round,pad=0.6', fc='#dff5ec', ec=ATTENTION_COLOR, lw=2.2)
ax.add_patch(p)

ax.text(core_x+core_w/2, core_y+core_h-5, 'Trust-tiered attention',
        ha='center', va='top', fontsize=10.5, fontweight='bold', color=ATTENTION_COLOR)
ax.text(core_x+core_w/2, core_y+core_h-10, 'deconvolution',
        ha='center', va='top', fontsize=10.5, fontweight='bold', color=ATTENTION_COLOR)
ax.text(core_x+core_w/2, core_y+core_h-15, '+ DepMap-calibrated trust',
        ha='center', va='top', fontsize=8, color='#444', style='italic')

# 3-layer scoring
ax.text(core_x+core_w/2, core_y+25, '─── 3-layer scoring ───', ha='center', va='center', fontsize=8, color='#666')
ax.text(core_x+core_w/2, core_y+19, r'$z\,S_{\mathrm{kill}} \;+\; 0.5\,zS_{\mathrm{onc}} \;+\; 0.7\,zS_{\mathrm{prior}}$',
        ha='center', va='center', fontsize=11)
ax.text(core_x+core_w/2, core_y+13, '$S_\\mathrm{kill}$: scDEAL × Cox HR × trust',
        ha='center', va='center', fontsize=7.5, color='#444')
ax.text(core_x+core_w/2, core_y+9.5, '$S_\\mathrm{onc}$: PRISM oncology / phase / MOA',
        ha='center', va='center', fontsize=7.5, color='#444')
ax.text(core_x+core_w/2, core_y+6, '$S_\\mathrm{prior}$: drug-target × pathway-Cox',
        ha='center', va='center', fontsize=7.5, color='#444')

# Arrows from inputs
for y_in in [68, 48, 28, 8]:
    ax.add_patch(FancyArrowPatch((21, y_in), (core_x, 55), arrowstyle='->,head_width=2,head_length=3',
                                  color='#999', lw=0.7, mutation_scale=8))

# === Right zone: Results ===
right_x = 60
def res_box(y, h, lines, color, fc='#fff'):
    p = FancyBboxPatch((right_x, y), 38, h, boxstyle='round,pad=0.3', fc=fc, ec=color, lw=1.2)
    ax.add_patch(p)
    for i, ln in enumerate(lines):
        ax.text(right_x+1.5, y+h-3-i*3.4, ln, ha='left', va='top', fontsize=7.5)

# LIHC
res_box(53, 22, [
    r'$\bullet$ LIHC: Lapatinib (EGFR/HER) at rank 1',
    r'$\bullet$ 9 mv-significant prognostic prototypes',
    r'$\bullet$ 8/9 replicate in 2 ext cohorts (n = 340)',
    r'$\bullet$ Composite c-index 0.727 in GSE14520',
    r'$\bullet$ Mechanism axes: EGFR/HER, proteasome,',
    r'$\;\;$  HSP90, AKT/PI3K, MEK',
], LIHC_COLOR, '#fde9e3')
ax.text(right_x+1.5, 75.5, 'LIHC (n = 763)', fontsize=9.5, fontweight='bold', color=LIHC_COLOR)

# LUAD
res_box(28, 22, [
    r'$\bullet$ LUAD: Gefitinib (EGFR) at rank 1',
    r'$\bullet$ 18/58 prototypes p < 0.001 in 4-cohort meta',
    r'$\bullet$ 9/9 LUAD SOC drugs in top 61 of 1,806',
    r'$\bullet$ Composite c-index 0.639–0.779 in 3 ext',
    r'$\bullet$ Mechanism axes: EGFR, MEK, ALK,',
    r'$\;\;$  proteasome, PI3K/mTOR, BCL inhibitors',
], LUAD_COLOR, '#dff0f5')
ax.text(right_x+1.5, 50.5, 'LUAD (n = 1,684)', fontsize=9.5, fontweight='bold', color=LUAD_COLOR)

# Method-validation strip
res_box(2, 22, [
    r'$\bullet$ Identical pipeline · 0 retuning across cancers',
    r'$\bullet$ Attention deconv: 0% composition collapse',
    r'$\;\;$  vs NNLS 95%, Scaden-MLP 99% collapse',
    r'$\bullet$ 3 scoring layers each individually necessary',
    r'$\bullet$ End-to-end runtime 12 min on 16 GB GPU',
    r'$\bullet$ 67/67 numerical claims independently audited',
], '#444', '#f5f5f5')
ax.text(right_x+1.5, 25.5, 'Method validation', fontsize=9.5, fontweight='bold', color='#444')

# Arrow from core to results
ax.add_patch(FancyArrowPatch((core_x+core_w, 55), (right_x, 55),
                              arrowstyle='->,head_width=3,head_length=4',
                              color=ATTENTION_COLOR, lw=1.5, mutation_scale=10))

save_both(fig, ROOT/'results/figures_main/graphical_abstract')
print('Saved graphical_abstract')
