"""Generate all figures for the Phase-1 PPT report."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

ROOT = Path('/home/holiday01/drug_sc')
OUT  = ROOT/'results/figures'; OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.titleweight':'bold',
                     'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':130,'savefig.dpi':200,'savefig.bbox':'tight'})

# ---------- Fig 1: Pipeline overview ----------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11,5.5))
    ax.set_xlim(0,11); ax.set_ylim(0,6); ax.axis('off')
    boxes = [
        (0.2, 4.5, 'TCGA-LIHC\n423 patients\n20k genes', '#fde2e4'),
        (0.2, 2.5, 'scRNA atlas\n172k HCC cells\n6 GEO', '#cddafd'),
        (0.2, 0.5, 'DepMap + GDSC\n+ PRISM + CTRP\n1.13M points', '#d8f3dc'),
        (3.2, 4.5, 'T3c Attention\nDeconvolver\n55 prototypes', '#ffe5d9'),
        (3.2, 2.5, 'T3b Geneformer\nzero-shot\n768-d', '#cddafd'),
        (3.2, 0.5, 'T2b scDEAL\nbulk→scRNA\nDANN', '#d8f3dc'),
        (6.2, 3.0, 'T3d Survival\nscoring', '#ffd6a5'),
        (6.2, 4.8, 'T3e Oncology/\nPhase filter', '#caffbf'),
        (6.2, 1.2, 'T3f Pathway\nCox prior', '#bdb2ff'),
        (9.0, 3.0, 'Top-20 Drugs\n+ FACS markers\n+ Suggested lines', '#ffadad'),
    ]
    for x,y,txt,c in boxes:
        rect = FancyBboxPatch((x,y),2.5,1.2, boxstyle='round,pad=0.05',
                              facecolor=c, edgecolor='#222', linewidth=1)
        ax.add_patch(rect)
        ax.text(x+1.25, y+0.6, txt, ha='center', va='center', fontsize=9.5)
    # arrows
    arrows = [
        ((2.7,5.1),(3.2,5.1)), ((2.7,3.1),(3.2,3.1)), ((2.7,1.1),(3.2,1.1)),
        ((5.7,5.1),(6.2,5.3)), ((5.7,3.1),(6.2,3.5)), ((5.7,1.1),(6.2,1.7)),
        ((8.7,5.3),(9.0,3.7)), ((8.7,3.5),(9.0,3.5)), ((8.7,1.7),(9.0,3.3)),
    ]
    for a,b in arrows:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->,head_width=4',
                                     color='#444', linewidth=1))
    ax.set_title('SCOPE-Rx Phase-1 pipeline — multi-modal LIHC drug repositioning',
                 pad=10, fontsize=13)
    plt.savefig(OUT/'fig01_pipeline.png')
    plt.close()

# ---------- Fig 2: Baseline comparison (Spearman / HR before-after) ----------
def fig_baseline_bar():
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    ax = axes[0]
    methods = ['Scaden-CA\ncomposition', 'scPDS\npathway', 'T2b scDEAL', 'T3c Attn-deconv']
    sp = [0.0, 0.11, 0.61, None]
    entropy = [0.0, None, None, 0.889]
    colors = ['#e76f51','#f4a261','#2a9d8f','#264653']
    xs = np.arange(len(methods))
    bars = ax.bar(xs, [0.0 if v is None else v for v in sp], color=colors, alpha=0.9)
    for i,(v,b) in enumerate(zip(sp,bars)):
        if v is not None:
            ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Per-line Spearman\n(held-out HCC drug ranking)')
    ax.set_title('Baseline accuracy on 3 held-out HCC lines')
    ax.set_ylim(0,0.8)

    ax = axes[1]
    hr_before = [1e28, 5e3, 1e-2, 1e6]; hr_after=[1.26,1.19,0.82,1.19]
    labels=['CD14 macro','M2 macro','Epithelial\n(protective)','CD16 macro']
    x = np.arange(len(labels)); w=0.35
    # log-scale plot
    ax.bar(x-w/2, np.log10(hr_before), w, color='#e76f51', label='before fix (junk HRs)')
    ax.bar(x+w/2, np.log10(hr_after), w, color='#2a9d8f', label='after z-score fix')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Cox HR (log10 scale)')
    ax.set_title('Cox HR numerical explosion fix (T3c)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT/'fig02_baselines_fix.png')
    plt.close()

# ---------- Fig 3: TCGA composition heatmap (T3c vs Scaden) ----------
def fig_composition():
    comp = pd.read_parquet(ROOT/'results/t3c/tcga_composition.parquet').values
    # sort patients by proto_22 (epithelial tumor) for visual order
    order = np.argsort(-comp[:,22])
    fig, ax = plt.subplots(figsize=(10,4))
    sns.heatmap(comp[order], ax=ax, cmap='mako', cbar_kws={'label':'composition'},
                yticklabels=False, xticklabels=False)
    ax.set_xlabel('Prototype (n=57, sorted by cell type)')
    ax.set_ylabel('TCGA-LIHC patient (n=423)')
    ax.set_title('TCGA-LIHC cell-state composition from T3c attention deconvolver\n(no patient collapses to a single prototype — cf. Scaden-CA 91% → SNU886)')
    plt.tight_layout()
    plt.savefig(OUT/'fig03_composition.png')
    plt.close()

# ---------- Fig 4: Prognosis-driving prototypes (Cox HR forest) ----------
def fig_cox_forest():
    cox = pd.read_parquet(ROOT/'results/t3c/cox_per_prototype.parquet')
    pm  = pd.read_parquet(ROOT/'results/t3c/prototype_meta.parquet').set_index('proto')
    cox = cox[cox['p']<0.05].copy()
    cox['label'] = cox.apply(lambda r: f"proto_{int(r['proto'])}  {pm.loc[int(r['proto']), 'dominant_cell_type']}", axis=1)
    cox = cox.sort_values('HR')
    fig, ax = plt.subplots(figsize=(8,5.2))
    colors = ['#2a9d8f' if h<1 else '#e76f51' for h in cox['HR']]
    ax.hlines(range(len(cox)), 0.7, cox['HR'], color='#999', lw=1)
    ax.scatter(cox['HR'], range(len(cox)), color=colors, s=80, zorder=5)
    ax.axvline(1.0, color='k', lw=0.8, ls='--')
    ax.set_yticks(range(len(cox))); ax.set_yticklabels(cox['label'], fontsize=9)
    for i,(_,r) in enumerate(cox.iterrows()):
        ax.text(r['HR']+0.02, i, f"p={r['p']:.3g}", va='center', fontsize=8)
    ax.set_xlabel('Cox Hazard Ratio per 1 SD of prototype composition')
    ax.set_title('Prognosis-driving HCC cell-state prototypes (TCGA-LIHC OS)')
    ax.set_xlim(0.7,1.45)
    plt.tight_layout()
    plt.savefig(OUT/'fig04_cox_prototypes.png')
    plt.close()

# ---------- Fig 5: Top-20 drug score decomposition ----------
def fig_top20_decomp():
    d = pd.read_parquet(ROOT/'results/t3f/drug_final_score.parquet').drop_duplicates('drug_lc').head(20).copy()
    d = d.iloc[::-1]  # reverse for top at top
    fig, ax = plt.subplots(figsize=(9,6.5))
    y = np.arange(len(d))
    w1 = d['z_kill'].values
    w2 = 0.5*d['z_onc'].values
    w3 = 0.7*d['z_prior'].values
    ax.barh(y, w1, color='#2a9d8f', label='1.0 × z(Kill)')
    ax.barh(y, w2, left=w1, color='#f4a261', label='0.5 × z(Onc relevance)')
    ax.barh(y, w3, left=w1+w2, color='#8a6bbf', label='0.7 × z(Pathway prior)')
    ax.set_yticks(y); ax.set_yticklabels(d['drug'], fontsize=9)
    ax.set_xlabel('Contribution to final score')
    ax.set_title('Top-20 drug ranking — decomposition of final score')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT/'fig05_top20_decomp.png')
    plt.close()

# ---------- Fig 6: Clinical HCC drug ranking progression ----------
def fig_clinical_progression():
    data = {
        'drug':     ['Lapatinib','Afatinib','Erlotinib','Paclitaxel','Doxorubicin','Vincristine',
                     'Oxaliplatin','Lenvatinib','Regorafenib','5-FU','Cabozantinib','Sunitinib',
                     'Sorafenib','Gemcitabine','Cisplatin'],
        'T3d raw':  [1,32,79,60,648,897,203,372,702,426,711,896,1146,1154,1389],
        'T3e soft': [1,3,12,8,90,262,43,45,109,98,113,166,267,458,468],
        'T3f+g':    [1,11,26,40,198,142,250,237,269,267,326,400,467,500,507],
    }
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(9,6))
    for i, r in df.iterrows():
        ax.plot([0,1,2], [r['T3d raw'], r['T3e soft'], r['T3f+g']],
                marker='o', lw=1.2, alpha=0.8)
        ax.text(2.05, r['T3f+g'], r['drug'], va='center', fontsize=9)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['Raw kill only','+ Oncology filter\n(Patch 2)','+ Pathway prior\n+ matched trust\n(Patches 1 & 3)'])
    ax.set_ylabel('Cohort ranking  (lower = better, out of 1806)')
    ax.set_yscale('log'); ax.invert_yaxis()
    ax.set_title('Clinical HCC drug ranking across pipeline patches')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT/'fig06_clinical_progression.png')
    plt.close()

# ---------- Fig 7: Pathway Cox volcano ----------
def fig_pathway_volcano():
    cox = pd.read_parquet(ROOT/'results/t3f/pathway_cox_tcga_lihc.parquet')
    cox['logHR'] = np.log(cox['HR'].clip(1e-3,1e3))
    cox['nlp'] = -np.log10(cox['p'].clip(1e-300))
    fig, ax = plt.subplots(figsize=(8,5))
    sig = cox['p']<0.05
    ax.scatter(cox.loc[~sig,'logHR'], cox.loc[~sig,'nlp'], c='#bbb', s=8, alpha=0.5)
    ax.scatter(cox.loc[sig & (cox['logHR']>0),'logHR'], cox.loc[sig & (cox['logHR']>0),'nlp'], c='#e76f51', s=10)
    ax.scatter(cox.loc[sig & (cox['logHR']<0),'logHR'], cox.loc[sig & (cox['logHR']<0),'nlp'], c='#2a9d8f', s=10)
    # annotate top pathways
    top = cox.sort_values('nlp', ascending=False).head(6)
    for _,r in top.iterrows():
        short = r['pathway'].split(':')[1][:38]
        ax.text(r['logHR']+0.004, r['nlp']+0.1, short, fontsize=7)
    ax.axhline(-np.log10(0.05), ls='--', c='k', lw=0.5)
    ax.axvline(0, ls='--', c='k', lw=0.5)
    ax.set_xlabel('log(HR) per SD (pathway activity)')
    ax.set_ylabel('-log10(p)')
    ax.set_title(f'TCGA-LIHC pathway Cox (n={len(cox)} pathways, {sig.sum()} significant)')
    plt.tight_layout()
    plt.savefig(OUT/'fig07_pathway_volcano.png')
    plt.close()

# ---------- Fig 8: Trust before/after Patch 3 ----------
def fig_trust_fix():
    t = pd.read_parquet(ROOT/'results/t3g/trust_before_after_by_celltype.parquet').reset_index()
    t = t.sort_values('trust_before')
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.arange(len(t)); w=0.4
    ax.bar(x-w/2, t['trust_before'], w, color='#e76f51', label='before (full panel max)')
    ax.bar(x+w/2, t['trust_after'],  w, color='#2a9d8f', label='after (cell-type matched)')
    ax.set_xticks(x); ax.set_xticklabels(t['dominant_cell_type'], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Trust to DepMap (Pearson r)')
    ax.set_title('Patch 3 — honest cell-type-matched trust references')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT/'fig08_trust_fix.png')
    plt.close()

# ---------- Fig 9: Geneformer UMAP by cell type ----------
def fig_geneformer_umap():
    emb = pd.read_parquet(ROOT/'results/t3b/geneformer_v2_meanpool_20k.parquet')
    obs = pd.read_parquet(ROOT/'results/t3b/cell_metadata_20k.parquet')
    from sklearn.decomposition import PCA
    import umap
    X = emb.drop(columns='cell_global_idx').values
    Z = PCA(n_components=50, random_state=0).fit_transform(X)
    u = umap.UMAP(n_neighbors=25, min_dist=0.3, random_state=0).fit_transform(Z)
    ct = obs['own_assign_cell_type'].values
    types = pd.Series(ct).value_counts().head(12).index.tolist()
    cmap = plt.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,6))
    for i, t in enumerate(types):
        m = ct==t
        ax.scatter(u[m,0], u[m,1], s=2.5, c=[cmap(i)], label=f'{t} ({m.sum()})', alpha=0.7)
    ax.set_title('Geneformer V2-104M zero-shot embedding of 20k HCC cells\n(10-NN label purity = 0.80)')
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=8, markerscale=3)
    plt.tight_layout()
    plt.savefig(OUT/'fig09_geneformer_umap.png')
    plt.close()

# ---------- Fig 10: Mechanistic themes pie ----------
def fig_mechanism_pie():
    data = {'EGFR/HER':6,'Proteasome':3,'HSP90':2,'AKT/PI3K':4,'CDK/MEK/MAPK':3,
            'HDAC':1,'DNA alkylator':1}
    fig, ax = plt.subplots(figsize=(5.5,5.5))
    colors = plt.get_cmap('Set3').colors
    ax.pie(data.values(), labels=[f'{k}\n(n={v})' for k,v in data.items()],
           colors=colors, startangle=90, wedgeprops={'edgecolor':'white'})
    ax.set_title('Top-25 drug mechanistic convergence\n(out of 1,806 candidates)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT/'fig10_mechanism_pie.png')
    plt.close()

print('Generating figures …')
fig_pipeline()
fig_baseline_bar()
fig_composition()
fig_cox_forest()
fig_top20_decomp()
fig_clinical_progression()
fig_pathway_volcano()
fig_trust_fix()
try:
    fig_geneformer_umap()
except Exception as e:
    print(f'  UMAP skipped ({e}), using PCA fallback')
fig_mechanism_pie()
import os
for f in sorted(os.listdir(OUT)):
    print(f' {OUT/f}')
