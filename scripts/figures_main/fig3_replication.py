"""
Fig 3 — Prognostic prototype reproducibility across cohorts.
(a) LIHC forest plot: 9 multivariate-sig prototypes — TCGA + GSE14520 + GSE76427 HRs
(b) LUAD forest plot: top-15 4-cohort meta-pooled prototypes
(c) Composite TCGA-trained risk c-index across 5 external cohorts
(d) LUAD 4-cohort meta volcano plot
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _style import use_style, panel_label, save_both, LIHC_COLOR, LUAD_COLOR
use_style()

ROOT = Path('/home/holiday01/drug_sc')

# ----- Load LIHC data -----
lihc_mv = pd.read_parquet(ROOT/'results/t3h/multivariate_cox.parquet')
lihc_mv_sig = lihc_mv[lihc_mv['p_x_mv']<0.05].sort_values('p_x_mv').reset_index(drop=True)
lihc_g14520_os  = pd.read_parquet(ROOT/'results/t3j/all_external_cox.parquet')
g14_os  = lihc_g14520_os[lihc_g14520_os['cohort']=='GSE14520_OS']
g76_os  = lihc_g14520_os[lihc_g14520_os['cohort']=='GSE76427_OS']

def lookup_hr(df, p):
    sub = df[df['proto']==p]
    if len(sub): return float(sub['HR'].iloc[0]), float(sub['p'].iloc[0])
    return None, None

# Build LIHC forest plot data
lihc_rows = []
for _, r in lihc_mv_sig.iterrows():
    p = int(r['proto'])
    hr_t = float(r['HR_x_uni']); p_t = float(r['p_x_mv'])
    hr_g14, p_g14 = lookup_hr(g14_os, p)
    hr_g76, p_g76 = lookup_hr(g76_os, p)
    lihc_rows.append({'proto':p, 'label':r.get('label', r.get('dominant','')),
                      'hr_tcga':hr_t, 'p_tcga':p_t,
                      'hr_g14':hr_g14, 'p_g14':p_g14,
                      'hr_g76':hr_g76, 'p_g76':p_g76})
lihc_fp = pd.DataFrame(lihc_rows)

# ----- Load LUAD data -----
luad_meta = pd.read_parquet(ROOT/'results/t3j_luad/meta_analysis_pooled.parquet')
luad_meta_top = luad_meta.sort_values('pooled_p').head(15)
luad_proto_meta = pd.read_parquet(ROOT/'results/t3c_luad/prototype_meta.parquet').set_index('proto')
luad_tcga = pd.read_parquet(ROOT/'results/t3c_luad/cox_per_prototype.parquet')
luad_g68 = pd.read_parquet(ROOT/'results/t3i_luad/GSE68465_cox_os.parquet')
luad_g72 = pd.read_parquet(ROOT/'results/t3i_luad/GSE72094_cox_os.parquet')
luad_g31 = pd.read_parquet(ROOT/'results/t3i_luad/GSE31210_cox_os.parquet')

luad_rows = []
for _, r in luad_meta_top.iterrows():
    p = int(r['proto'])
    hr_t,_ = lookup_hr(luad_tcga, p)
    hr_68,_ = lookup_hr(luad_g68, p)
    hr_72,_ = lookup_hr(luad_g72, p)
    hr_31,_ = lookup_hr(luad_g31, p)
    luad_rows.append({'proto':p, 'label':r.get('dominant_cell_type','') + '|' + str(r.get('dominant_sample_type','')),
                      'hr_tcga':hr_t, 'hr_68':hr_68, 'hr_72':hr_72, 'hr_31':hr_31,
                      'pooled_HR':float(r['pooled_HR']), 'pooled_p':float(r['pooled_p'])})
luad_fp = pd.DataFrame(luad_rows)

# ----- Composite c-index data -----
lihc_c = pd.read_parquet(ROOT/'results/t3j/composite_risk_results.parquet')
# Read LUAD composite from t3i_luad eval_metrics
import json
luad_eval = json.loads((ROOT/'results/t3i_luad/eval_metrics.json').read_text())
c_rows = []
for _, r in lihc_c.iterrows():
    if 'error' in str(r.get('error','')): continue
    c_rows.append({'cohort':r['cohort'], 'cancer':'LIHC', 'n':int(r['n']),
                   'c_index':float(r['c_index']), 'p':float(r['p'])})
for cid, sub in luad_eval['cohorts'].items():
    for endpt, e in sub['composite_risk'].items():
        c_rows.append({'cohort':f'{cid}_{endpt}', 'cancer':'LUAD', 'n':int(e['n']),
                       'c_index':float(e['c_index']), 'p':float(e['p'])})
ccompare = pd.DataFrame(c_rows)

# =============================================================
# Build figure
# =============================================================
fig = plt.figure(figsize=(12.0, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])

# ---------- Panel a: LIHC forest plot ----------
ax = fig.add_subplot(gs[0, 0])
y = np.arange(len(lihc_fp))[::-1]
def safe_log(x):
    return np.log(x) if x and x > 0 else np.nan
def plot_dot(ax, ys, hrs, color, marker, label, offset=0):
    xs = [safe_log(h) for h in hrs]
    ax.scatter(xs, np.array(ys)+offset, marker=marker, color=color, s=22, edgecolor='k',
               linewidth=0.4, label=label, zorder=3)
plot_dot(ax, y, lihc_fp['hr_tcga'].values, '#222', 'o', 'TCGA-LIHC', offset=0.18)
plot_dot(ax, y, lihc_fp['hr_g14'].values,  LIHC_COLOR, 's', 'GSE14520', offset=0)
plot_dot(ax, y, lihc_fp['hr_g76'].values,  '#cb997e', '^', 'GSE76427', offset=-0.18)
ax.axvline(0, color='k', lw=0.5, ls='--')
ax.set_yticks(y)
ax.set_yticklabels([f"#{r['proto']} {r['label'][:18]}" for _, r in lihc_fp.iterrows()], fontsize=6.5)
ax.set_xlabel('log Hazard Ratio (per SD)')
ax.set_title('LIHC — 9 prog-sig prototypes  (multivariate-adj)', color=LIHC_COLOR)
ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1), frameon=False, fontsize=6.5)
panel_label(ax, 'a')

# ---------- Panel b: LUAD forest plot ----------
ax = fig.add_subplot(gs[0, 1])
y = np.arange(len(luad_fp))[::-1]
plot_dot(ax, y, luad_fp['hr_tcga'].values, '#222', 'o', 'TCGA-LUAD', offset=0.27)
plot_dot(ax, y, luad_fp['hr_68'].values, LUAD_COLOR, 's', 'GSE68465', offset=0.09)
plot_dot(ax, y, luad_fp['hr_72'].values, '#7baccc', '^', 'GSE72094', offset=-0.09)
plot_dot(ax, y, luad_fp['hr_31'].values, '#1d3a52', 'D', 'GSE31210', offset=-0.27)
# pooled in star
xs = [safe_log(h) for h in luad_fp['pooled_HR'].values]
ax.scatter(xs, y, marker='*', color='#d62728', s=80, edgecolor='k', linewidth=0.4,
           label='Pooled (4-cohort)', zorder=4)
ax.axvline(0, color='k', lw=0.5, ls='--')
ax.set_yticks(y)
ax.set_yticklabels([f"#{r['proto']} {r['label'][:22]}" for _, r in luad_fp.iterrows()], fontsize=6.3)
ax.set_xlabel('log Hazard Ratio (per SD)')
ax.set_title('LUAD — Top-15 meta-pooled prog-sig prototypes', color=LUAD_COLOR)
ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1), frameon=False, fontsize=6.5)
panel_label(ax, 'b')

# ---------- Panel c: composite c-index ----------
ax = fig.add_subplot(gs[1, 0])
ccompare = ccompare.sort_values(['cancer','cohort']).reset_index(drop=True)
colors = [LIHC_COLOR if c=='LIHC' else LUAD_COLOR for c in ccompare['cancer']]
y = np.arange(len(ccompare))
ax.barh(y, ccompare['c_index'], color=colors, edgecolor='k', lw=0.4, height=0.65)
ax.set_yticks(y)
labels = [c.replace('_',' / ') for c in ccompare['cohort']]
ax.set_yticklabels(labels, fontsize=7)
ax.axvline(0.5, color='k', lw=0.5, ls=':', alpha=0.5)
ax.set_xlim(0.4, 0.85)
ax.set_xlabel('Composite TCGA-trained risk score — concordance index')
for i,(c,n,p) in enumerate(zip(ccompare['c_index'], ccompare['n'], ccompare['p'])):
    ax.text(c+0.005, i, f'{c:.3f}  (n={int(n)}, p={p:.1g})', va='center', fontsize=6.5)
# Legend for cancer types
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color=LIHC_COLOR, label='LIHC'),
                   mpatches.Patch(color=LUAD_COLOR, label='LUAD')],
          loc='lower left', bbox_to_anchor=(1.005, 0), frameon=False, fontsize=7)
ax.set_title('External validation — composite TCGA-trained risk c-index')
panel_label(ax, 'c')

# ---------- Panel d: LUAD 4-cohort meta volcano ----------
ax = fig.add_subplot(gs[1, 1])
m = luad_meta.dropna(subset=['pooled_p','pooled_HR']).copy()
m['log_HR'] = np.log(m['pooled_HR'])
m['neg_log10_p'] = -np.log10(m['pooled_p'].clip(lower=1e-300))
# Color by significance
colors = ['#d62728' if p<0.001 else ('#ff7f0e' if p<0.01 else ('#9ec5e8' if p<0.05 else '#cccccc'))
          for p in m['pooled_p']]
ax.scatter(m['log_HR'], m['neg_log10_p'], c=colors, s=22, edgecolor='k', linewidth=0.3, alpha=0.85)
# Top-5 labels
top5 = m.sort_values('pooled_p').head(5)
for _, r in top5.iterrows():
    label = f"#{int(r['proto'])} {str(r.get('dominant_cell_type','')[:8])}"
    ax.annotate(label, (r['log_HR'], r['neg_log10_p']),
                xytext=(4, 0), textcoords='offset points', fontsize=6, va='center')
ax.axhline(-np.log10(0.05), color='gray', lw=0.5, ls=':', label='p=0.05')
ax.axhline(-np.log10(0.001), color='gray', lw=0.5, ls='--', label='p=0.001')
ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel('Pooled log HR (per SD)')
ax.set_ylabel('-log10 pooled p')
ax.set_title('LUAD 4-cohort fixed-effect meta-analysis volcano')
ax.legend(loc='lower right', frameon=False, fontsize=6.5)
panel_label(ax, 'd')

save_both(fig, ROOT/'results/figures_main/fig3_replication')
print('Saved Fig 3')
