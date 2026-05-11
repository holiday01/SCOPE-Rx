"""
Fig 2 — Trust-tiered attention deconvolution prevents composition collapse.
(a) UMAP of Geneformer embeddings, LIHC | LUAD side by side
(b) Composition entropy distribution: Attention vs NNLS vs Scaden
(c) Top-1 prototype usage histogram (Attention vs NNLS vs Scaden)
(d) Trust-to-DepMap distribution per prototype (LIHC + LUAD)
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from _style import (use_style, panel_label, save_both,
                    LIHC_COLOR, LUAD_COLOR,
                    ATTENTION_COLOR, NNLS_COLOR, SCADEN_COLOR)
use_style()

ROOT = Path('/home/holiday01/drug_sc')

# ============================================================
# Panel a — UMAPs of Geneformer embeddings
# ============================================================
import scanpy as sc
import anndata as ad

def umap_panel(ax, parquet_path, meta_path, title, color):
    emb_df = pd.read_parquet(parquet_path)
    cells = emb_df['cell_global_idx'].values
    Xg = emb_df.drop(columns='cell_global_idx').values.astype(np.float32)
    obs = pd.read_parquet(meta_path)
    a = ad.AnnData(Xg)
    sc.pp.pca(a, n_comps=50, zero_center=True)
    sc.pp.neighbors(a, n_neighbors=15, use_rep='X_pca')
    sc.tl.umap(a, random_state=0)
    ct_col = 'celltype' if 'celltype' in obs.columns else (
        'own_assign_cell_type' if 'own_assign_cell_type' in obs.columns else 'cell_type')
    a.obs[ct_col] = obs[ct_col].values if ct_col in obs.columns else 'NA'
    cats = a.obs[ct_col].value_counts().index.tolist()[:10]
    palette = plt.colormaps.get_cmap('tab10').colors
    coords = a.obsm['X_umap']
    for i, ct in enumerate(cats):
        m = a.obs[ct_col].values == ct
        ax.scatter(coords[m,0], coords[m,1], s=1.2, c=[palette[i]], label=str(ct)[:18],
                   alpha=0.6, linewidth=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_title(title, color=color)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), frameon=False,
              fontsize=6, markerscale=3.5, handletextpad=0.2, ncol=3)

# ============================================================
# Panels b,c — composition diversity (LUAD)
# ============================================================
B0 = pd.read_parquet(ROOT/'results/t3c_luad/tcga_composition.parquet').values
# Recompute B1 (NNLS) and B2 (Scaden) from ablation_table run if available; else recompute via NNLS.
# We saved B-tier comp as parquet -> not directly. Use the deconv.parquet summary; for visualisation,
# we must regenerate. To save time: replicate using existing prototype_expression + Xt.

# Read ablation deconv results table for the bar metrics:
deconv_summary = pd.read_parquet(ROOT/'results/ablation_table/deconv.parquet')

# Recompute B1/B2 compositions (cheap, <30s)
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import nnls
from scipy.stats import entropy as _ent
from pathlib import Path as _P

def reload_pb_and_tcga():
    pb = pd.read_parquet(ROOT/'results/t3c_luad/prototype_expression.parquet').values.astype(np.float32)
    genes = list(pd.read_parquet(ROOT/'results/t3c_luad/prototype_expression.parquet').columns)
    TCGA = _P('/mnt/10t/scrna_atac/data/raw/TCGA_LUAD/TCGA_LUAD_expression.gz')
    t_expr = pd.read_csv(TCGA, sep='\t', index_col=0, low_memory=False)
    if t_expr.values.max() < 30:
        t_lin = np.clip(np.expm1(t_expr.values * np.log(2)), 0, None).astype(np.float32)
    else:
        t_lin = t_expr.values.astype(np.float32)
    t_g = {g:i for i,g in enumerate(t_expr.index)}
    sel = np.array([t_g.get(g,-1) for g in genes])
    valid = sel>=0
    Xt_raw = np.zeros((t_expr.shape[1], len(genes)), dtype=np.float32)
    Xt_raw[:, valid] = t_lin[sel[valid],:].T
    rs = Xt_raw.sum(1, keepdims=True) + 1e-6
    Xt_norm = np.log1p(Xt_raw / rs * 1e4).astype(np.float32)
    return pb, Xt_norm

pb, Xt = reload_pb_and_tcga()
n_proto = pb.shape[0]

# B1 NNLS
print('Recomputing NNLS comp …', flush=True)
B1 = np.zeros((Xt.shape[0], n_proto), dtype=np.float32)
for i in range(Xt.shape[0]):
    w, _ = nnls(pb.T, Xt[i])
    s = w.sum(); B1[i] = w/s if s>1e-9 else np.full_like(w, 1.0/n_proto)

# B2 Scaden — quick re-train (smaller epochs to save time)
print('Re-training Scaden quickly …', flush=True)
class ScadenMLP(nn.Module):
    def __init__(self, d_in, n_proto):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, n_proto))
    def forward(self,x): return F.softmax(self.net(x), dim=-1)
rng = np.random.default_rng(0)
W = rng.dirichlet(np.ones(n_proto)*0.5, size=20000).astype(np.float32)
W[:6000] = rng.dirichlet(np.ones(n_proto)*2.0, size=6000).astype(np.float32)
Xs = W @ pb + rng.normal(0, 0.10, (20000, pb.shape[1])).astype(np.float32)
g_mu = Xs.mean(0); g_sd = Xs.std(0)+1e-6
Xs_z = (Xs - g_mu)/g_sd
Xt_z = (Xt - g_mu)/g_sd
mdl = ScadenMLP(pb.shape[1], n_proto).cuda()
opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-5)
Xs_tt = torch.from_numpy(Xs_z).cuda(); Wt = torch.from_numpy(W).cuda()
for _ep in range(15):
    mdl.train()
    perm = torch.randperm(len(Xs_tt), device='cuda')
    for s in range(0, len(perm), 512):
        idx = perm[s:s+512]
        loss = F.l1_loss(mdl(Xs_tt[idx]), Wt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
mdl.eval()
with torch.no_grad():
    B2 = mdl(torch.from_numpy(Xt_z).cuda()).cpu().numpy()

def entropy_per_row(C):
    return -(C * np.log(C + 1e-9)).sum(1) / np.log(C.shape[1])

ent_B0 = entropy_per_row(B0)
ent_B1 = entropy_per_row(B1)
ent_B2 = entropy_per_row(B2)

# ============================================================
# Build figure
# ============================================================
fig = plt.figure(figsize=(13.0, 8.5), constrained_layout=True)
gs = fig.add_gridspec(2, 4)

# Panel a — UMAP (2 sub-panels)
ax_a1 = fig.add_subplot(gs[0, 0])
ax_a2 = fig.add_subplot(gs[0, 1])
print('Building LIHC UMAP …', flush=True)
try:
    umap_panel(ax_a1, ROOT/'results/t3b/geneformer_v2_meanpool_20k.parquet',
               ROOT/'results/t3b/cell_metadata_20k.parquet',
               'LIHC scRNA (n=20,000)', LIHC_COLOR)
except Exception as e:
    ax_a1.text(0.5, 0.5, f'LIHC UMAP unavailable\n{e}', ha='center', va='center',
               transform=ax_a1.transAxes, fontsize=7)
print('Building LUAD UMAP …', flush=True)
umap_panel(ax_a2, ROOT/'results/t3b_luad/geneformer_v2_meanpool_luad.parquet',
           ROOT/'results/t3b_luad/cell_metadata_luad.parquet',
           'LUAD scRNA (n=20,000)', LUAD_COLOR)
panel_label(ax_a1, 'a')

# Panel b — entropy distribution (LUAD)
ax_b = fig.add_subplot(gs[0, 2])
parts = ax_b.violinplot([ent_B0, ent_B1, ent_B2], positions=[0,1,2],
                        showmeans=True, widths=0.7)
for body, c in zip(parts['bodies'], [ATTENTION_COLOR, NNLS_COLOR, SCADEN_COLOR]):
    body.set_facecolor(c); body.set_edgecolor('k'); body.set_linewidth(0.5); body.set_alpha(0.75)
parts['cmeans'].set_color('k'); parts['cmeans'].set_linewidth(0.8)
parts['cbars'].set_color('k'); parts['cbars'].set_linewidth(0.5)
parts['cmins'].set_color('k'); parts['cmaxes'].set_color('k')
ax_b.set_xticks([0,1,2])
ax_b.set_xticklabels(['Attention\n(SCOPE-Rx)', 'NNLS', 'Scaden\nMLP'])
ax_b.set_ylabel('Composition entropy (% of max)')
ax_b.set_ylim(-0.05, 1.05)
ax_b.set_title('TCGA-LUAD per-patient composition diversity')
panel_label(ax_b, 'b')

# Panel c — Top-1 prototype usage histogram
ax_c = fig.add_subplot(gs[0, 3])
top1_B0 = B0.max(1); top1_B1 = B1.max(1); top1_B2 = B2.max(1)
bins = np.linspace(0, 1, 25)
ax_c.hist([top1_B0, top1_B1, top1_B2], bins=bins,
          label=['Attention','NNLS','Scaden'], color=[ATTENTION_COLOR, NNLS_COLOR, SCADEN_COLOR],
          stacked=False, edgecolor='k', linewidth=0.3, alpha=0.75)
ax_c.axvline(0.5, color='k', lw=0.5, ls='--')
ax_c.set_xlabel('Max prototype assignment per patient')
ax_c.set_ylabel('Number of TCGA-LUAD patients')
ax_c.set_title('Top-1 prototype usage')
ax_c.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=6.5, ncol=3)
# Annotate collapse fraction
# (Stats summary moved to caption — was previously redundant with legend below)
panel_label(ax_c, 'c')

# Panel d — Trust-to-DepMap distribution (LIHC + LUAD)
ax_d = fig.add_subplot(gs[1, :])
luad_meta = pd.read_parquet(ROOT/'results/t3c_luad/prototype_meta.parquet')
lihc_meta = pd.read_parquet(ROOT/'results/t3c/prototype_meta.parquet')
def categorize(label, ct):
    s = (str(label) + ' ' + str(ct)).lower()
    if 'epithelial' in s or 'malignant' in s or 'hepato' in s or 'tumor' in s.split('|')[0]:
        return 'Tumor / epithelial'
    if 'lymphoc' in s or 't cell' in s or 'b lymph' in s or 'cd4' in s or 'cd8' in s or 'nk' in s:
        return 'Lymphoid'
    if 'myeloid' in s or 'mac' in s or 'dc' in s or 'mono' in s:
        return 'Myeloid'
    if 'fibro' in s: return 'Fibroblast'
    if 'endothel' in s: return 'Endothelial'
    return 'Other'
luad_meta['cat'] = luad_meta.apply(lambda r: categorize(r.get('label',''), r.get('dominant_cell_type','')), axis=1)
lihc_meta['cat'] = lihc_meta.apply(lambda r: categorize(r.get('label',''), r.get('dominant_cell_type','')), axis=1)
cat_order = ['Tumor / epithelial','Myeloid','Lymphoid','Endothelial','Fibroblast','Other']
cat_color = {'Tumor / epithelial':'#d62728','Myeloid':'#ff7f0e','Lymphoid':'#2ca02c',
             'Endothelial':'#9467bd','Fibroblast':'#8c564b','Other':'#7f7f7f'}
positions = []; data_lists=[]; colors=[]; labels=[]
x = 0
for cat in cat_order:
    luad_v = luad_meta[luad_meta['cat']==cat]['trust_to_depmap'].values
    lihc_v = lihc_meta[lihc_meta['cat']==cat]['trust_to_depmap'].values
    if len(lihc_v)>0:
        positions.append(x); data_lists.append(lihc_v); colors.append(cat_color[cat]); labels.append(f'LIHC\n{cat}'); x+=1
    if len(luad_v)>0:
        positions.append(x); data_lists.append(luad_v); colors.append(cat_color[cat]); labels.append(f'LUAD\n{cat}'); x+=1
    x += 0.4

bplots = ax_d.boxplot(data_lists, positions=positions, widths=0.6, patch_artist=True,
                     showfliers=True, flierprops=dict(marker='o', ms=2.5, alpha=0.5),
                     medianprops=dict(color='k', lw=0.8), boxprops=dict(linewidth=0.5))
for box, c in zip(bplots['boxes'], colors):
    box.set_facecolor(c); box.set_alpha(0.6)
ax_d.axhline(0.5, color='gray', lw=0.5, ls='--', label='High-trust threshold (0.5)')
ax_d.set_xticks(positions); ax_d.set_xticklabels(labels, fontsize=6.5)
ax_d.set_ylabel('Trust to DepMap (max Pearson r vs cell-line panel)')
ax_d.set_title('Per-prototype trust calibration to DepMap (LIHC + LUAD)')
ax_d.legend(loc='upper left', bbox_to_anchor=(1.005, 1), frameon=False, fontsize=7)
panel_label(ax_d, 'd', x=-0.06)

save_both(fig, ROOT/'results/figures_main/fig2_attention')
print('Saved Fig 2')
