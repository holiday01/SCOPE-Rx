"""
Fig 4 — Cancer-specific SOC drug rediscovery.
(a) LIHC top-25 final ranking with mechanism color bands
(b) LUAD top-25 final ranking with mechanism color bands
(c) Slope chart: LIHC clinical drugs T3d → T3e → T3f
(d) Slope chart: LUAD SOC drugs T3d → T3e → T3f
"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _style import use_style, panel_label, save_both, LIHC_COLOR, LUAD_COLOR
use_style()

ROOT = Path('/home/holiday01/drug_sc')

# Mechanism axes for color bands
def mech_axis(target, moa, drug):
    """Map drug → coarse mechanism axis label."""
    s = ' '.join([str(target or ''), str(moa or ''), str(drug or '')]).upper()
    if any(k in s for k in ['EGFR','ERBB']) and 'ALK' not in s: return 'EGFR/HER'
    if 'ALK' in s and 'EGFR' not in s: return 'ALK'
    if any(k in s for k in ['MAP2K','MEK ','MEK1','MEK2']): return 'MEK'
    if any(k in s for k in ['HSP90','TANESPI','ALVESPI','GELDANAMYCIN']): return 'HSP90'
    if 'PSM' in s or 'PROTEASOME' in s: return 'Proteasome'
    if any(k in s for k in ['BCL2','BCL_','VENETOCLAX','NAVITOCLAX']): return 'BCL'
    if any(k in s for k in ['PIK3','MTOR','AKT','PI3K']): return 'PI3K/AKT/mTOR'
    if any(k in s for k in ['CDK4','CDK6','PALBOCICLIB']): return 'CDK4/6'
    if any(k in s for k in ['MDM2','MDM ']): return 'MDM2'
    if any(k in s for k in ['PARP','RUCAPARIB','VELIPARIB']): return 'PARP'
    if any(k in s for k in ['AURK','GSK1070916']): return 'Aurora'
    if any(k in s for k in ['CHEK1','CHK1','AZD7762']): return 'CHK1/2'
    if any(k in s for k in ['HDAC','VORINOSTAT']): return 'HDAC'
    if any(k in s for k in ['TOP1','TOPOISOMERASE','SN-38']): return 'TOP1'
    if any(k in s for k in ['MET ','KDR','VEGFR','CABOZANTINIB','FORETINIB','SORAFENIB','LENVATINIB','REGORAFENIB','SUNITINIB']): return 'VEGFR/MET'
    if any(k in s for k in ['DHFR','TYMS','PEMETREXED','METHOTREXATE']): return 'Antifolate'
    if any(k in s for k in ['DNA POLYMERASE','POLE','POLA','MICROTUBULE','TUBULIN','PACLITAXEL','DOCETAXEL','VINORELBINE']): return 'Cytoskel/DNA-rep'
    if any(k in s for k in ['SMO ','HEDGEHOG']): return 'Hedgehog'
    return 'Other'

mech_colors = {
    'EGFR/HER':'#d62728','ALK':'#8c564b','MEK':'#ff7f0e','HSP90':'#1f77b4',
    'Proteasome':'#2ca02c','BCL':'#9467bd','PI3K/AKT/mTOR':'#17becf',
    'CDK4/6':'#e377c2','MDM2':'#bcbd22','PARP':'#7f7f7f','Aurora':'#9c755f',
    'CHK1/2':'#76b7b2','HDAC':'#edc949','TOP1':'#59a14f','VEGFR/MET':'#af7aa1',
    'Antifolate':'#4e79a7','Cytoskel/DNA-rep':'#f28e2b','Hedgehog':'#e15759',
    'Other':'#bbbbbb'
}

def panel_top25(ax, df, title, color_axis):
    df = df.head(25).copy()
    df['axis'] = df.apply(lambda r: mech_axis(r.get('target'), r.get('moa'), r.get('drug')), axis=1)
    df = df[::-1]  # top at top
    y = np.arange(len(df))
    cols = [mech_colors[a] for a in df['axis']]
    ax.barh(y, df['score_final'], color=cols, edgecolor='k', lw=0.3, height=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{i+1}. {d}' for i,d in zip(range(len(df))[::-1], df['drug'].str[:22])],
                       fontsize=6.6)
    ax.set_xlabel('Final score (z_kill + 0.5·z_onc + 0.7·z_prior)')
    ax.set_title(title, color=color_axis)
    # Mechanism legend
    seen_axes = list(dict.fromkeys(df['axis']))
    handles = [plt.Rectangle((0,0),1,1, color=mech_colors[a]) for a in seen_axes]
    ax.legend(handles, seen_axes, loc='upper left', bbox_to_anchor=(1.005, 1.0),
              frameon=False, fontsize=6, handlelength=0.7, ncol=1)

def panel_slope(ax, ranks_df, n_total, title, color):
    """ranks_df: cols = ['drug', 'r_T3d', 'r_T3e', 'r_T3f']"""
    stages = ['T3d\nkill','T3e\nonc filter','T3f\nfinal']
    x = np.arange(3)
    for _, r in ranks_df.iterrows():
        ys = [r['r_T3d'], r['r_T3e'], r['r_T3f']]
        ax.plot(x, ys, '-o', color=color, alpha=0.7, lw=0.7, ms=3)
        # label at right side
        ax.annotate(r['drug'], (2, r['r_T3f']), xytext=(3,0), textcoords='offset points',
                    fontsize=6, va='center', color=color)
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.invert_yaxis()
    ax.set_yscale('log')
    ax.set_ylabel(f'Rank (of {n_total} drugs)')
    ax.set_title(title, color=color)
    ax.set_xlim(-0.3, 3.5)  # extra room for drug labels at right
    ax.grid(axis='y', alpha=0.3, lw=0.4)

# ----- Load LIHC data -----
lihc_final = pd.read_parquet(ROOT/'results/t3f/drug_final_score.parquet')
lihc_t3d = pd.read_parquet(ROOT/'results/t3d/drug_score_cohort.parquet').sort_values('score', ascending=False).reset_index(drop=True)
lihc_t3e = pd.read_parquet(ROOT/'results/t3e/drug_score_soft_combined.parquet').sort_values('score_combined', ascending=False).reset_index(drop=True)
lihc_t3f = lihc_final.sort_values('score_final', ascending=False).reset_index(drop=True)

# ----- Load LUAD data -----
luad_final = pd.read_parquet(ROOT/'results/t3f_luad/drug_final_score.parquet')
luad_t3d = pd.read_parquet(ROOT/'results/t3d_luad/drug_score_cohort.parquet').sort_values('score', ascending=False).reset_index(drop=True)
luad_t3e = pd.read_parquet(ROOT/'results/t3e_luad/drug_score_soft_combined.parquet').sort_values('score_combined', ascending=False).reset_index(drop=True)
luad_t3f = luad_final.sort_values('score_final', ascending=False).reset_index(drop=True)

# ----- Build slope-rank tables -----
def get_rank_table(t3d, t3e, t3f, drug_list):
    """For each drug name (lowercase match), return (rank_T3d, rank_T3e, rank_T3f)."""
    rows = []
    for d in drug_list:
        h_d = t3d[t3d['drug'].str.lower()==d]; r_d = (int(h_d.index[0])+1) if len(h_d) else None
        h_e = t3e[t3e['drug'].str.lower()==d]; r_e = (int(h_e.index[0])+1) if len(h_e) else None
        h_f = t3f[t3f['drug'].str.lower()==d]; r_f = (int(h_f.index[0])+1) if len(h_f) else None
        if r_d and r_e and r_f:
            rows.append({'drug':d, 'r_T3d':r_d, 'r_T3e':r_e, 'r_T3f':r_f})
    return pd.DataFrame(rows)

lihc_clinical = ['lapatinib','sorafenib','lenvatinib','regorafenib','cabozantinib',
                 'erlotinib','gefitinib','afatinib','paclitaxel','doxorubicin',
                 'cisplatin','5-fluorouracil']
luad_soc = ['gefitinib','trametinib','osimertinib','selumetinib','alectinib',
            'erlotinib','afatinib','brigatinib','pemetrexed']

lihc_slope = get_rank_table(lihc_t3d, lihc_t3e, lihc_t3f, lihc_clinical)
luad_slope = get_rank_table(luad_t3d, luad_t3e, luad_t3f, luad_soc)

# ----- Build figure -----
fig, axes = plt.subplots(2, 2, figsize=(12.0, 10), constrained_layout=True,
                         gridspec_kw={'width_ratios':[1.4, 1], 'height_ratios':[1, 1]})

panel_top25(axes[0,0], lihc_t3f, f'LIHC — Top-25 final ranking (n={len(lihc_t3f):,} drugs)', LIHC_COLOR)
panel_label(axes[0,0], 'a')
panel_top25(axes[1,0], luad_t3f, f'LUAD — Top-25 final ranking (n={len(luad_t3f):,} drugs)', LUAD_COLOR)
panel_label(axes[1,0], 'c')

panel_slope(axes[0,1], lihc_slope, len(lihc_t3f), 'LIHC clinical drug rank trajectory', LIHC_COLOR)
panel_label(axes[0,1], 'b')
panel_slope(axes[1,1], luad_slope, len(luad_t3f), 'LUAD SOC drug rank trajectory', LUAD_COLOR)
panel_label(axes[1,1], 'd')

save_both(fig, ROOT/'results/figures_main/fig4_drug_rediscovery')
print('Saved Fig 4')
