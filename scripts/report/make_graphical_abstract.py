"""Generate the graphical abstract for SCOPE-Rx — single landscape figure."""
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse
from matplotlib.lines import Line2D

ROOT = Path('/home/holiday01/drug_sc')
OUT  = ROOT/'results/figures'

plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.titleweight':'bold',
                     'figure.dpi':140,'savefig.dpi':300,'savefig.bbox':'tight',
                     'font.family':'sans-serif'})

# ---- color palette ----
ACCENT  = '#264653'
INPUTC  = '#cddafd'
PIPEC   = '#ffe5d9'
SCOREC  = '#caffbf'
OUTPUTC = '#ffadad'
GOOD    = '#2a9d8f'
BAD     = '#e76f51'

def rounded(ax, x, y, w, h, label, color, fontsize=9, lw=1, alpha=1.0, sub=None):
    box = FancyBboxPatch((x,y), w, h, boxstyle='round,pad=0.06,rounding_size=0.18',
                         facecolor=color, edgecolor='#333', lw=lw, alpha=alpha, zorder=3)
    ax.add_patch(box)
    cx = x + w/2; cy = y + h/2
    ax.text(cx, cy + (0.10*h if sub else 0), label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', zorder=4)
    if sub:
        ax.text(cx, cy - 0.32*h, sub, ha='center', va='center',
                fontsize=fontsize-1.5, color='#444', zorder=4, style='italic')

def arrow(ax, x1, y1, x2, y2, color='#444'):
    a = FancyArrowPatch((x1,y1),(x2,y2),
        arrowstyle='-|>,head_width=4,head_length=6',
        color=color, lw=1.4, mutation_scale=15, zorder=2)
    ax.add_patch(a)

# ============= main figure: 14 x 8 inches, 4 columns =============
fig, ax = plt.subplots(figsize=(14, 7.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 7.5); ax.axis('off')

# title
ax.text(7, 7.2, 'SCOPE-Rx', ha='center', va='center',
        fontsize=24, fontweight='bold', color=ACCENT)
ax.text(7, 6.85, 'Multi-modal single-cell × drug-structure × TCGA survival pipeline for HCC drug repositioning',
        ha='center', va='center', fontsize=11.5, color='#444', style='italic')

# ===== Column 1: INPUTS (x = 0.2 - 2.7) =====
ax.text(1.45, 6.3, 'INPUTS', ha='center', fontsize=12, fontweight='bold', color=ACCENT)
inputs = [
    ('TCGA-LIHC',        '423 patients\nbulk RNA-seq + OS', 5.4),
    ('HCC scRNA atlas',  '172,538 cells\n6 GEO series',     4.4),
    ('DepMap + GDSC\n+ PRISM + CTRPv2', '1,806 drugs\n1.13M AUC pairs',  3.4),
    ('MSigDB + KEGG\n+ Reactome',       '1,790 pathways', 2.4),
    ('External cohorts', 'GSE14520 (n=225)\nGSE76427 (n=115)', 1.4),
]
for txt, sub, y in inputs:
    rounded(ax, 0.2, y, 2.5, 0.7, txt, INPUTC, fontsize=9, sub=sub)

# arrows from inputs to pipeline
for _,_,y in inputs:
    arrow(ax, 2.75, y+0.35, 3.2, y+0.35)

# ===== Column 2: PIPELINE STAGES (x = 3.3 - 6.8) =====
ax.text(5.05, 6.3, 'PIPELINE', ha='center', fontsize=12, fontweight='bold', color=ACCENT)
pipe = [
    ('Geneformer V2-104M\nzero-shot embedding',
     '768-d · 10-NN purity 0.80', 5.4, '#fff1e6'),
    ('Leiden → 57 cell-state\nprototypes',
     '57 prototypes · 89% entropy', 4.4, '#fff1e6'),
    ('Attention deconvolver\n(bulk → composition)',
     '0% collapse vs Scaden 91%',   3.4, PIPEC),
    ('Multivariate Cox\n+ trust calibration',
     '9 prognostic prototypes',     2.4, '#fde2e4'),
    ('scDEAL kill + oncology\n+ pathway-Cox prior',
     '4-method consensus',          1.4, SCOREC),
]
for txt, sub, y, c in pipe:
    rounded(ax, 3.3, y, 3.5, 0.7, txt, c, fontsize=9.5, sub=sub)

# vertical arrows down the pipeline
for y_top, y_bot in [(5.4, 5.1), (4.4, 4.1), (3.4, 3.1), (2.4, 2.1)]:
    arrow(ax, 5.05, y_top, 5.05, y_bot+0.0)

# arrow from pipeline to output
arrow(ax, 6.85, 1.75, 7.4, 1.75)
arrow(ax, 6.85, 3.75, 7.4, 3.75)
arrow(ax, 6.85, 5.75, 7.4, 5.75)

# ===== Column 3: KEY RESULTS (x = 7.5 - 10.5) =====
ax.text(9, 6.3, 'KEY RESULTS', ha='center', fontsize=12, fontweight='bold', color=ACCENT)

# Result 1: External validation
rounded(ax, 7.5, 5.05, 3.0, 1.05, 'External validation\n(2 cohorts, 3 platforms)',
        '#a8dadc', fontsize=10)
ax.text(9.0, 5.40, '8 / 9', ha='center', fontsize=22, fontweight='bold', color=GOOD, zorder=5)
ax.text(9.0, 5.13, 'prognostic prototypes replicate', ha='center', fontsize=8.5, color='#222', zorder=5)

# Result 2: c-index
rounded(ax, 7.5, 3.55, 3.0, 1.05, 'TCGA-trained risk score\n(GSE14520, n=219)',
        '#a8dadc', fontsize=10)
ax.text(9.0, 3.90, 'c-index = 0.727', ha='center', fontsize=15, fontweight='bold', color=GOOD, zorder=5)
ax.text(9.0, 3.62, 'OS p = 0.006   ·   RFS p = 0.021', ha='center', fontsize=8.5, color='#222', zorder=5)

# Result 3: Method consistency
rounded(ax, 7.5, 2.05, 3.0, 1.05, 'Methodological consensus',
        '#a8dadc', fontsize=10)
ax.text(9.0, 2.40, '5 + 4 methods', ha='center', fontsize=14, fontweight='bold', color=ACCENT, zorder=5)
ax.text(9.0, 2.12, 'agree on 9 prototypes & 11 drugs', ha='center', fontsize=8.5, color='#222', zorder=5)

# Result 4: Meta-analysis
rounded(ax, 7.5, 0.65, 3.0, 1.05, 'Fixed-effect meta-analysis\n(3 cohorts, n = 763)',
        '#a8dadc', fontsize=10)
ax.text(9.0, 1.0, '16 / 57', ha='center', fontsize=22, fontweight='bold', color=GOOD, zorder=5)
ax.text(9.0, 0.73, 'pooled-significant prototypes', ha='center', fontsize=8.5, color='#222', zorder=5)

# arrow to outputs
arrow(ax, 10.55, 3.75, 11.1, 3.75)

# ===== Column 4: TOP DRUGS / WET-LAB BRIEF (x = 11.2 - 13.9) =====
ax.text(12.55, 6.3, 'TOP DRUGS', ha='center', fontsize=12, fontweight='bold', color=ACCENT)
ax.text(12.55, 6.05, '5 mechanism classes converge', ha='center', fontsize=9, style='italic', color='#444')

drugs = [
    ('Lapatinib',     'EGFR/HER2 (Launched)',    5.40, OUTPUTC),
    ('Afatinib',      'pan-HER (Launched)',      4.85, OUTPUTC),
    ('Trametinib',    'MEK (Launched)',          4.30, OUTPUTC),
    ('Copanlisib',    'PI3K (Launched)',         3.75, OUTPUTC),
    ('MK-2206',       'AKT (Phase 2)',           3.20, '#ffe0b3'),
    ('Tanespimycin',  'HSP90 (Phase 3)',         2.65, '#ffe0b3'),
    ('Cediranib',     'VEGFR (Phase 3, rescue)', 2.10, '#ffe0b3'),
]
for d, sub, y, c in drugs:
    rounded(ax, 11.2, y, 2.7, 0.45, d, c, fontsize=10, sub=sub)

# wet-lab brief footer
rounded(ax, 11.2, 0.85, 2.7, 0.95, 'Wet-lab brief',
        '#ffd6a5', fontsize=10)
ax.text(12.55, 1.50, 'per drug:', ha='center', fontsize=8.5, fontweight='bold', color='#222', zorder=5)
ax.text(12.55, 1.25, 'subpop · markers · cell line', ha='center', fontsize=7.8, color='#222', zorder=5)
ax.text(12.55, 1.05, 'MOA · phase · SMILES', ha='center', fontsize=7.8, color='#222', zorder=5)

# bottom strip — patient deliverable
rounded(ax, 0.2, 0.15, 13.7, 0.50,
        '423 TCGA-LIHC patients × per-patient Top-5 drug ranking · 176 high-confidence drugs · 1,077 low-trust flagged',
        '#e9c46a', fontsize=10)

plt.savefig(OUT/'graphical_abstract.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(OUT/'graphical_abstract.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT/"graphical_abstract.png"} ({(OUT/"graphical_abstract.png").stat().st_size/1024:.0f} KB)')
print(f'Saved: {OUT/"graphical_abstract.pdf"} ({(OUT/"graphical_abstract.pdf").stat().st_size/1024:.0f} KB)')
