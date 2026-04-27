"""
T3c — Cell-state attention deconvolver replacing Scaden-CA.

Basis = HCC scRNA "cell-state prototypes" obtained by Leiden-clustering the
Geneformer embeddings (768-d). Each prototype carries:
  * a 768-d cell-state centroid (from Geneformer)
  * a 17 460-d pseudobulk expression profile (from raw counts)

We train a soft-attention model that predicts a distribution over prototypes
from a bulk expression vector. Trained on Dirichlet-mixed pseudobulk of
prototypes (+ Gaussian noise). Then applied to TCGA-LIHC.

Outputs:
  checkpoints/t3c_attn_deconv.pt
  results/t3c/tcga_composition.parquet  (424 × P)
  results/t3c/prototype_expression.parquet
  results/t3c/prototype_geneformer_centroid.parquet
  results/t3c/prototype_meta.parquet     (id, dominant cell_type, n_cells)
  results/t3c/eval_metrics.json
"""
from __future__ import annotations
import json, time, math, random, pickle
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import anndata as ad

ROOT = Path('/home/holiday01/drug_sc')
T3B  = ROOT/'results/t3b'
PROC = ROOT/'data/processed/hcc_drug'
TCGA_EXPR = ROOT/'data/TCGA_LIHC/TCGA_LIHC_expression.gz'
OUT  = ROOT/'results/t3c'; OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT/'checkpoints'

DEV = 'cuda'
SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)

N_PROTOS     = 60       # target number of cell-state prototypes (Leiden resolution tuned)
SIM_N        = 60000    # number of simulated bulk mixtures for training
SIM_NOISE    = 0.10
EPOCHS       = 25
BS           = 512
LR           = 1e-3

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Load Geneformer embeddings + cell metadata ----------
log('Loading Geneformer meanpool + cell metadata …')
emb = pd.read_parquet(T3B/'geneformer_v2_meanpool_20k.parquet')
obs = pd.read_parquet(T3B/'cell_metadata_20k.parquet')
assert (emb['cell_global_idx'].values == obs['cell_global_idx'].values).all()
global_idx = emb['cell_global_idx'].values
Xg = emb.drop(columns='cell_global_idx').values.astype(np.float32)
log(f'Geneformer: {Xg.shape}  cells with metadata: {len(obs)}')

# ---------- 2. Cluster into prototypes (Leiden on Geneformer neighbourhood) ----------
log('Building neighbour graph + Leiden clustering on Geneformer embeddings …')
import scanpy as sc
_ad = ad.AnnData(Xg); _ad.obs = obs.reset_index(drop=True)
sc.pp.pca(_ad, n_comps=50, zero_center=True)
sc.pp.neighbors(_ad, n_neighbors=15, use_rep='X_pca')
# adjust resolution to hit ~N_PROTOS clusters
res = 1.5
for _ in range(6):
    sc.tl.leiden(_ad, resolution=res, random_state=0)
    n_c = _ad.obs['leiden'].nunique()
    log(f'  resolution={res:.2f} → {n_c} clusters')
    if n_c >= N_PROTOS*0.9 and n_c <= N_PROTOS*1.3: break
    res *= (N_PROTOS/max(n_c,1))**0.7
proto_id = _ad.obs['leiden'].astype(int).values
n_proto = proto_id.max()+1
log(f'Final clusters: {n_proto}')

# ---------- 3. Prototype centroid (Geneformer) + pseudobulk expression ----------
log('Computing prototype Geneformer centroid + pseudobulk expression …')
centroid = np.zeros((n_proto, Xg.shape[1]), dtype=np.float32)
for p in range(n_proto):
    m = proto_id == p
    centroid[p] = Xg[m].mean(0)

# pseudobulk on 17,460 shared genes: reload the h5ad backed, slice by global_idx
log('Reading raw counts for sampled cells (17,460 shared genes) …')
genes_shared = pd.read_parquet(PROC/'gene_universe.parquet')['gene'].tolist()
a = ad.read_h5ad(ROOT/'data/scRNA_HCC_integrated/GSE223204GSE202642GSE162616_bbknn_tumornormal.h5ad', backed='r')
sc_genes = list(a.var_names); sc_i = {g:i for i,g in enumerate(sc_genes)}
sel_sc_idx = np.array([sc_i[g] for g in genes_shared])
assert 'counts' in a.layers, "expected layers['counts'] in h5ad"
rows=[]
CHUNK = 4000
order = np.argsort(global_idx)
for s in range(0, len(global_idx), CHUNK):
    ix = np.sort(global_idx[order[s:s+CHUNK]])  # sorted access for backed h5ad
    sub = a.layers['counts'][ix, :][:, sel_sc_idx]
    if hasattr(sub,'toarray'): sub = sub.toarray()
    rows.append(np.asarray(sub, dtype=np.float32))
a.file.close()
Xc = np.vstack(rows)[np.argsort(order)]   # re-sort to original 20k order
log(f'Raw counts matrix: {Xc.shape}  max={Xc.max():.0f}  (should be >>1 if real counts)')

# CP10K-log normalise per cell
rs = Xc.sum(1, keepdims=True) + 1e-6
Xn = np.log1p(Xc / rs * 1e4).astype(np.float32)

# pseudobulk per prototype
pb = np.zeros((n_proto, Xn.shape[1]), dtype=np.float32)
sizes = np.zeros(n_proto, dtype=np.int64)
for p in range(n_proto):
    m = proto_id == p
    pb[p] = Xn[m].mean(0)
    sizes[p] = m.sum()
log(f'Prototype pseudobulk: {pb.shape}  min size={sizes.min()}  max size={sizes.max()}')

# dominant cell-type per prototype + sample_type split
ct_col = 'own_assign_cell_type' if 'own_assign_cell_type' in obs.columns else 'cell_type'
dom=[]; dom_st=[]; tumor_frac=[]; label=[]
for p in range(n_proto):
    m = proto_id == p
    ct = obs.loc[m, ct_col].value_counts().index[0]
    st_counts = obs.loc[m, 'sample_type'].value_counts() if 'sample_type' in obs.columns else None
    if st_counts is not None:
        st_top = st_counts.index[0]
        tfrac = float(st_counts.get('tumor',0) / max(st_counts.sum(),1))
    else:
        st_top = 'NA'; tfrac = np.nan
    dom.append(ct)
    dom_st.append(st_top)
    tumor_frac.append(tfrac)
    # disambiguate Epithelial: tag as _malignant or _normal by sample_type
    if ct == 'Epithelial Cells':
        label.append(f'Epithelial_{"tumor" if tfrac>0.7 else ("normal" if tfrac<0.3 else "mixed")}')
    else:
        label.append(ct)

# Per-prototype trust weight: similarity to nearest DepMap cell line
log('Computing trust weights (prototype vs DepMap cell lines) …')
panC = pd.read_parquet(ROOT/'data/processed/hcc_drug/cellline_expression_panCancer.parquet')
# cell lines are TPM; convert to log1p to match pb scale
Xcl = np.log1p(panC.values.astype(np.float32))
# standardize each gene on its own (so correlation on ranked axes)
cl_mu = Xcl.mean(0); cl_sd = Xcl.std(0)+1e-6
Xcl_z = (Xcl - cl_mu) / cl_sd
pb_mu = pb.mean(0); pb_sd = pb.std(0)+1e-6
pb_z  = (pb - pb_mu) / pb_sd
# Pearson per (proto, line) via normalized dot product
pb_n = pb_z / (np.linalg.norm(pb_z, axis=1, keepdims=True)+1e-6)
cl_n = Xcl_z / (np.linalg.norm(Xcl_z, axis=1, keepdims=True)+1e-6)
cor = pb_n @ cl_n.T   # (n_proto, n_cellline)
trust = cor.max(1)
best_line_idx = cor.argmax(1)
best_line_id = np.array(panC.index)[best_line_idx]
log(f'Trust weights: mean={trust.mean():.3f}  min={trust.min():.3f}  max={trust.max():.3f}')
log(f'Best line r distribution: p10={np.percentile(trust,10):.2f}  median={np.median(trust):.2f}  p90={np.percentile(trust,90):.2f}')

pd.DataFrame({'proto':np.arange(n_proto),'n_cells':sizes,
              'dominant_cell_type':dom,'dominant_sample_type':dom_st,
              'tumor_fraction':tumor_frac,'label':label,
              'trust_to_depmap':trust,'best_cellline':best_line_id}).to_parquet(OUT/'prototype_meta.parquet', index=False)

pd.DataFrame(pb, columns=genes_shared).to_parquet(OUT/'prototype_expression.parquet')
pd.DataFrame(centroid, columns=[f'e{i}' for i in range(centroid.shape[1])]).to_parquet(OUT/'prototype_geneformer_centroid.parquet')

# ---------- 4. Simulate mixtures ----------
log('Simulating Dirichlet-weighted bulk mixtures …')
alphas = np.random.uniform(0.3, 1.5, N_PROTOS if False else n_proto).astype(np.float32)  # varied sparsity
W_tr = np.random.dirichlet(alpha=np.ones(n_proto)*0.5, size=SIM_N).astype(np.float32)
# Use a mix of sparsities so model sees both focal and diffuse tumours
W_tr[:SIM_N//3] = np.random.dirichlet(alpha=np.ones(n_proto)*2.0, size=SIM_N//3).astype(np.float32)
X_tr = W_tr @ pb
X_tr += np.random.normal(0, SIM_NOISE, X_tr.shape).astype(np.float32)

W_va = np.random.dirichlet(alpha=np.ones(n_proto)*0.5, size=5000).astype(np.float32)
X_va = W_va @ pb + np.random.normal(0, SIM_NOISE, (5000, pb.shape[1])).astype(np.float32)

g_mu = X_tr.mean(0); g_sd = X_tr.std(0) + 1e-6
X_tr_s = (X_tr - g_mu) / g_sd
X_va_s = (X_va - g_mu) / g_sd
log(f'Simulated train {X_tr.shape}  val {X_va.shape}')

# ---------- 5. Attention-based deconvolver ----------
class AttnDeconv(nn.Module):
    def __init__(self, d_in, n_proto, d_hid=256, p=0.1):
        super().__init__()
        self.query_enc = nn.Sequential(
            nn.Linear(d_in, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(p),
            nn.Linear(512, d_hid))
        self.proto_key = nn.Parameter(torch.randn(n_proto, d_hid) * 0.02)
        self.temp = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        q = self.query_enc(x)                              # (B, d)
        logits = (q @ self.proto_key.T) / self.temp.clamp(min=0.1)
        return F.softmax(logits, dim=-1)

model = AttnDeconv(X_tr.shape[1], n_proto).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

Xt = torch.from_numpy(X_tr_s).to(DEV); Wt = torch.from_numpy(W_tr).to(DEV)
Xv = torch.from_numpy(X_va_s).to(DEV); Wv = torch.from_numpy(W_va).to(DEV)

for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(Xt), device=DEV)
    tot=0.0
    for s in range(0,len(perm),BS):
        idx = perm[s:s+BS]
        pr = model(Xt[idx])
        loss = F.l1_loss(pr, Wt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*len(idx)
    model.eval()
    with torch.no_grad():
        p_va = model(Xv)
        va_mae = F.l1_loss(p_va, Wv).item()
        per_r=[]
        pv = p_va.cpu().numpy()
        for j in range(n_proto):
            if pv[:,j].std()>1e-6 and W_va[:,j].std()>1e-6:
                per_r.append(np.corrcoef(pv[:,j], W_va[:,j])[0,1])
        mean_r = np.mean(per_r) if per_r else 0.0
    log(f'ep {ep+1:02d}/{EPOCHS}  tr_mae={tot/len(Xt):.4f}  va_mae={va_mae:.4f}  mean_proto_r={mean_r:.3f}')

# ---------- 6. Apply to TCGA-LIHC ----------
log('Loading TCGA-LIHC expression …')
t_expr = pd.read_csv(TCGA_EXPR, sep='\t', index_col=0, low_memory=False)
log(f'TCGA shape: {t_expr.shape}')
if t_expr.values.max() < 30:
    log('  TCGA appears log2-scaled; de-log before log1p(CP10K)')
    t_expr_lin = np.clip(np.expm1(t_expr.values * np.log(2)), 0, None).astype(np.float32)
else:
    t_expr_lin = t_expr.values.astype(np.float32)
# genes align
t_g = {g:i for i,g in enumerate(t_expr.index)}
sel = np.array([t_g[g] if g in t_g else -1 for g in genes_shared])
valid = sel>=0
log(f'TCGA genes aligned: {valid.sum()}/{len(genes_shared)}')
Xt_raw = np.zeros((t_expr.shape[1], len(genes_shared)), dtype=np.float32)
Xt_raw[:, valid] = t_expr_lin[sel[valid], :].T
# CPM-normalise if we de-logged (rough), else treat as FPKM+1 already
rs = Xt_raw.sum(1, keepdims=True) + 1e-6
Xt_norm = np.log1p(Xt_raw / rs * 1e4).astype(np.float32)
Xt_std = (Xt_norm - g_mu) / g_sd
log(f'TCGA processed: {Xt_std.shape}')

model.eval()
with torch.no_grad():
    comp = model(torch.from_numpy(Xt_std).to(DEV)).cpu().numpy()
comp_df = pd.DataFrame(comp, index=t_expr.columns, columns=[f'proto_{i}' for i in range(n_proto)])
comp_df.to_parquet(OUT/'tcga_composition.parquet')
log(f'TCGA composition: {comp_df.shape}')

# Diversity diagnostics
max_frac = comp.max(1)
top1_prop = (max_frac > 0.5).mean()
entropy = -(comp * np.log(comp + 1e-9)).sum(1)
log(f'Composition diagnostics:')
log(f'  Top-1 prototype usage distribution (mean): {comp.mean(0).round(3).tolist()[:10]} …')
log(f'  Fraction of patients with >50% in one prototype: {top1_prop:.3f}')
log(f'  Mean entropy: {entropy.mean():.2f} / max possible {math.log(n_proto):.2f}  ({entropy.mean()/math.log(n_proto):.2%} of max)')

# Top prototypes across the cohort
mean_comp = comp.mean(0)
top_order = np.argsort(-mean_comp)
log('Top-10 prototypes in TCGA-LIHC cohort:')
proto_meta = pd.DataFrame({'proto':np.arange(n_proto),'n_cells':sizes,'dominant_cell_type':dom})
for p in top_order[:10]:
    log(f'  proto_{p:<3d}  mean_comp={mean_comp[p]:.3f}  dominant={dom[p]:<30s}  n_cells={sizes[p]}')

# ---------- 7. TCGA survival sanity: Cox per prototype ----------
log('Running Cox(OS) per prototype on TCGA-LIHC …')
clin = pd.read_csv(ROOT/'data/TCGA_LIHC/TCGA_LIHC_clinical.tsv', sep='\t', low_memory=False)
cols = {c.lower():c for c in clin.columns}
sid = cols.get('sampleid') or clin.columns[0]
# death info: days_to_death + vital_status
if 'vital_status' in cols and 'days_to_death' in cols and 'days_to_last_followup' in cols:
    clin['event'] = clin[cols['vital_status']].str.upper().map({'DEAD':1, 'DECEASED':1, 'ALIVE':0, 'LIVING':0})
    dtd = pd.to_numeric(clin[cols['days_to_death']], errors='coerce')
    dtl = pd.to_numeric(clin[cols['days_to_last_followup']], errors='coerce')
    clin['time'] = dtd.where(clin['event']==1, dtl)
    surv = clin[[sid,'event','time']].rename(columns={sid:'sample'}).dropna()
    merged = comp_df.reset_index().rename(columns={'index':'sample'}).merge(surv, on='sample', how='inner')
    log(f'  survival rows: {len(merged)}')
    from lifelines import CoxPHFitter
    rows=[]
    for p in range(n_proto):
        col = f'proto_{p}'
        d = merged[['event','time',col]].dropna()
        d = d[d['time']>0].copy()
        if len(d)<50 or d[col].std()<1e-4: continue
        # **FIX**: z-score composition column before Cox so that HR is per-SD, not per-unit
        mu, sd = d[col].mean(), d[col].std()
        d['x'] = (d[col] - mu) / sd
        try:
            cph = CoxPHFitter(penalizer=0.05).fit(d[['event','time','x']], 'time','event')
            rows.append({'proto':p,'dominant':dom[p],'n':len(d),
                         'HR':float(np.exp(cph.params_['x'])),
                         'p':float(cph.summary.loc['x','p']),
                         'mean_comp':float(mean_comp[p]),
                         'comp_std':float(sd)})
        except Exception as e:
            rows.append({'proto':p,'dominant':dom[p],'n':len(d),'error':str(e)})
    cox_df = pd.DataFrame(rows).sort_values('p')
    cox_df.to_parquet(OUT/'cox_per_prototype.parquet', index=False)
    log(cox_df.head(10).to_string())
else:
    log('  Could not find expected clinical columns; skipping Cox')
    cox_df = pd.DataFrame()

# ---------- 8. Save ----------
torch.save({'model':model.state_dict(),'n_proto':n_proto,'g_mu':g_mu,'g_sd':g_sd,
            'genes':genes_shared,'centroid':centroid,'pb':pb,'proto_id':proto_id,
            'global_idx':global_idx,'dominant':dom,'sizes':sizes.tolist()},
           CKPT/'t3c_attn_deconv.pt')

(OUT/'eval_metrics.json').write_text(json.dumps({
    'n_prototypes': int(n_proto),
    'sim_val_mae': float(va_mae),
    'sim_mean_proto_r': float(mean_r),
    'tcga_samples': int(comp_df.shape[0]),
    'tcga_entropy_pct_of_max': float(entropy.mean()/math.log(n_proto)),
    'tcga_top1_gt_50pct': float(top1_prop),
    'top_prototypes': [{'proto':int(p), 'mean_comp':float(mean_comp[p]),
                        'dominant':dom[p], 'n_cells':int(sizes[p])}
                       for p in top_order[:10]],
    'cox_significant_prototypes': int((cox_df['p']<0.05).sum()) if len(cox_df) else 0,
}, indent=2, default=str))
log('== T3c done ==')
