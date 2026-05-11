"""
T3c-LUAD — Cell-state attention deconvolver on TCGA-LUAD.

Mirrors t3c_celltype_attention_deconv.py for LIHC. Differences:
  - Reads results/t3b_luad/{geneformer_v2_meanpool_luad, cell_metadata_luad}.parquet
  - LUAD scRNA h5ad lacks raw counts → pseudobulk computed as
        log1p(mean over cells of expm1(X) * 1e4 / mean(expm1(X).sum(1)))
    (X is log1p(normalize_total(target_sum=1e4)); rank/ratio identical to
    pseudobulk-from-raw-counts for the deconvolution objective).
  - Source target = TCGA-LUAD (576 patients).
  - Tumor/normal split based on Sample_Origin ∈ {tLung,tL/B,mBrain,mLN,PE}
    (tumor) vs {nLung,nLN} (normal).
"""
from __future__ import annotations
import json, time, math, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import anndata as ad

ROOT = Path('/home/holiday01/drug_sc')
T3B  = ROOT/'results/t3b_luad'
PROC = ROOT/'data/processed/luad_drug'
TCGA_EXPR = Path('/mnt/10t/scrna_atac/data/raw/TCGA_LUAD/TCGA_LUAD_expression.gz')
TCGA_CLIN = Path('/mnt/10t/scrna_atac/data/raw/TCGA_LUAD/TCGA_LUAD_clinical.tsv')
SCR = Path('/mnt/10t/scrna_atac/data/processed/LUAD/luad_scrna_annotated.h5ad')
OUT  = ROOT/'results/t3c_luad'; OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT/'checkpoints'

DEV = 'cuda'
SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)

N_PROTOS  = 60
SIM_N     = 60000
SIM_NOISE = 0.10
EPOCHS    = 25
BS        = 512
LR        = 1e-3

NORMAL_ORIGINS = {'nLung', 'nLN'}

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Load Geneformer embeddings + cell metadata ----------
log('Loading Geneformer meanpool + cell metadata …')
emb = pd.read_parquet(T3B/'geneformer_v2_meanpool_luad.parquet')
obs = pd.read_parquet(T3B/'cell_metadata_luad.parquet')
assert (emb['cell_global_idx'].values == obs['cell_global_idx'].values).all()
global_idx = emb['cell_global_idx'].values
Xg = emb.drop(columns='cell_global_idx').values.astype(np.float32)
log(f'Geneformer: {Xg.shape}  cells with metadata: {len(obs)}')

# Tumor/normal flag
obs = obs.copy()
obs['tn'] = np.where(obs['Sample_Origin'].isin(NORMAL_ORIGINS), 'normal', 'tumor')

# ---------- 2. Cluster into prototypes (Leiden on Geneformer neighbourhood) ----------
log('Building neighbour graph + Leiden clustering on Geneformer embeddings …')
import scanpy as sc
_ad = ad.AnnData(Xg); _ad.obs = obs.reset_index(drop=True)
sc.pp.pca(_ad, n_comps=50, zero_center=True)
sc.pp.neighbors(_ad, n_neighbors=15, use_rep='X_pca')
res = 1.5
for _ in range(8):
    sc.tl.leiden(_ad, resolution=res, random_state=0)
    n_c = _ad.obs['leiden'].nunique()
    log(f'  resolution={res:.2f} → {n_c} clusters')
    if N_PROTOS*0.9 <= n_c <= N_PROTOS*1.3: break
    res *= (N_PROTOS/max(n_c,1))**0.7
proto_id = _ad.obs['leiden'].astype(int).values
n_proto = int(proto_id.max())+1
log(f'Final clusters: {n_proto}')

# ---------- 3. Prototype centroid + pseudobulk ----------
log('Computing prototype Geneformer centroid + pseudobulk expression …')
centroid = np.zeros((n_proto, Xg.shape[1]), dtype=np.float32)
for p in range(n_proto):
    centroid[p] = Xg[proto_id == p].mean(0)

# Pseudobulk from log-normalized X. Reconstruct counts-per-10k via expm1.
genes_shared = pd.read_parquet(PROC/'gene_universe.parquet')['gene'].tolist()
log(f'Reading scRNA expression for {len(genes_shared)} shared genes …')
a = ad.read_h5ad(SCR, backed='r')
sc_genes = list(a.var_names); sc_i = {g:i for i,g in enumerate(sc_genes)}
sel_sc_idx = np.array([sc_i[g] for g in genes_shared])
rows = []
CHUNK = 4000
for s in range(0, len(global_idx), CHUNK):
    ix = global_idx[s:s+CHUNK]
    sub = a.X[ix, :][:, sel_sc_idx]
    if hasattr(sub,'toarray'): sub = sub.toarray()
    rows.append(np.asarray(sub, dtype=np.float32))
a.file.close()
Xn_log = np.vstack(rows)            # (N, G_shared) — log1p(CP10K) values
Xn_lin = np.expm1(Xn_log)           # CP10K counts-per-10k (rows ≈ 1e4)
log(f'Lin pseudo-counts: {Xn_lin.shape}  median row sum={np.median(Xn_lin.sum(1)):.0f}')

# pseudobulk per prototype
pb = np.zeros((n_proto, Xn_lin.shape[1]), dtype=np.float32)
sizes = np.zeros(n_proto, dtype=np.int64)
for p in range(n_proto):
    m = proto_id == p
    if m.sum() == 0: continue
    pb[p] = np.log1p(Xn_lin[m].mean(0))
    sizes[p] = m.sum()
log(f'Prototype pseudobulk: {pb.shape}  min size={sizes.min()}  max size={sizes.max()}')

# Dominant cell type + sample type per prototype, with malignant/normal disambiguation
dom=[]; dom_st=[]; tumor_frac=[]; label=[]
for p in range(n_proto):
    m = proto_id == p
    if m.sum() == 0:
        dom.append('NA'); dom_st.append('NA'); tumor_frac.append(np.nan); label.append('NA')
        continue
    ct = obs.loc[m,'celltype'].value_counts().index[0]
    sub_ct = obs.loc[m,'Cell_subtype'].value_counts().index[0] if 'Cell_subtype' in obs.columns else 'NA'
    st_counts = obs.loc[m,'tn'].value_counts()
    tfrac = float(st_counts.get('tumor',0) / max(st_counts.sum(),1))
    dom.append(str(ct)); dom_st.append(st_counts.idxmax())
    tumor_frac.append(tfrac)
    if str(ct) == 'Epithelial cells':
        tag = 'tumor' if tfrac > 0.7 else ('normal' if tfrac < 0.3 else 'mixed')
        label.append(f'Epithelial_{tag} ({sub_ct})')
    else:
        label.append(f'{ct} ({sub_ct})')

# Trust weight: prototype vs DepMap pan-cancer (Pearson on z-scored log-TPM)
log('Computing trust weights (prototype vs DepMap cell lines) …')
panC = pd.read_parquet(PROC/'cellline_expression_panCancer.parquet')
Xcl = np.log1p(panC.values.astype(np.float32))
cl_mu = Xcl.mean(0); cl_sd = Xcl.std(0)+1e-6
Xcl_z = (Xcl - cl_mu) / cl_sd
pb_mu = pb.mean(0); pb_sd = pb.std(0)+1e-6
pb_z  = (pb - pb_mu) / pb_sd
pb_n = pb_z / (np.linalg.norm(pb_z, axis=1, keepdims=True)+1e-6)
cl_n = Xcl_z / (np.linalg.norm(Xcl_z, axis=1, keepdims=True)+1e-6)
cor = pb_n @ cl_n.T
trust = cor.max(1)
best_line_idx = cor.argmax(1)
best_line_id = np.array(panC.index)[best_line_idx]
log(f'Trust: mean={trust.mean():.3f}  min={trust.min():.3f}  max={trust.max():.3f}  '
    f'p10={np.percentile(trust,10):.3f}  median={np.median(trust):.3f}  p90={np.percentile(trust,90):.3f}')

pd.DataFrame({'proto':np.arange(n_proto),'n_cells':sizes,
              'dominant_cell_type':dom,'dominant_sample_type':dom_st,
              'tumor_fraction':tumor_frac,'label':label,
              'trust_to_depmap':trust,'best_cellline':best_line_id}).to_parquet(OUT/'prototype_meta.parquet', index=False)

pd.DataFrame(pb, columns=genes_shared).to_parquet(OUT/'prototype_expression.parquet')
pd.DataFrame(centroid, columns=[f'e{i}' for i in range(centroid.shape[1])]).to_parquet(OUT/'prototype_geneformer_centroid.parquet')

# ---------- 4. Simulate Dirichlet mixtures ----------
log('Simulating Dirichlet-weighted bulk mixtures …')
W_tr = np.random.dirichlet(alpha=np.ones(n_proto)*0.5, size=SIM_N).astype(np.float32)
W_tr[:SIM_N//3] = np.random.dirichlet(alpha=np.ones(n_proto)*2.0, size=SIM_N//3).astype(np.float32)
X_tr = W_tr @ pb
X_tr += np.random.normal(0, SIM_NOISE, X_tr.shape).astype(np.float32)
W_va = np.random.dirichlet(alpha=np.ones(n_proto)*0.5, size=5000).astype(np.float32)
X_va = W_va @ pb + np.random.normal(0, SIM_NOISE, (5000, pb.shape[1])).astype(np.float32)
g_mu = X_tr.mean(0); g_sd = X_tr.std(0) + 1e-6
X_tr_s = (X_tr - g_mu) / g_sd
X_va_s = (X_va - g_mu) / g_sd
log(f'Train {X_tr.shape}  Val {X_va.shape}')

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
        q = self.query_enc(x)
        logits = (q @ self.proto_key.T) / self.temp.clamp(min=0.1)
        return F.softmax(logits, dim=-1)

model = AttnDeconv(X_tr.shape[1], n_proto).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
Xt = torch.from_numpy(X_tr_s).to(DEV); Wt = torch.from_numpy(W_tr).to(DEV)
Xv = torch.from_numpy(X_va_s).to(DEV); Wv = torch.from_numpy(W_va).to(DEV)

for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(Xt), device=DEV); tot = 0.0
    for s in range(0, len(perm), BS):
        idx = perm[s:s+BS]
        pr = model(Xt[idx]); loss = F.l1_loss(pr, Wt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*len(idx)
    model.eval()
    with torch.no_grad():
        p_va = model(Xv); va_mae = F.l1_loss(p_va, Wv).item()
        pv = p_va.cpu().numpy(); per_r = []
        for j in range(n_proto):
            if pv[:,j].std() > 1e-6 and W_va[:,j].std() > 1e-6:
                per_r.append(np.corrcoef(pv[:,j], W_va[:,j])[0,1])
        mean_r = float(np.mean(per_r)) if per_r else 0.0
    log(f'ep {ep+1:02d}/{EPOCHS}  tr_mae={tot/len(Xt):.4f}  va_mae={va_mae:.4f}  mean_proto_r={mean_r:.3f}')

# ---------- 6. Apply to TCGA-LUAD ----------
log('Loading TCGA-LUAD expression …')
t_expr = pd.read_csv(TCGA_EXPR, sep='\t', index_col=0, low_memory=False)
log(f'TCGA shape: {t_expr.shape}')
if t_expr.values.max() < 30:
    log('  TCGA appears log2-scaled; de-log before log1p(CP10K)')
    t_expr_lin = np.clip(np.expm1(t_expr.values * np.log(2)), 0, None).astype(np.float32)
else:
    t_expr_lin = t_expr.values.astype(np.float32)

t_g = {g:i for i,g in enumerate(t_expr.index)}
sel = np.array([t_g[g] if g in t_g else -1 for g in genes_shared])
valid = sel >= 0
log(f'TCGA genes aligned: {valid.sum()}/{len(genes_shared)}')
Xt_raw = np.zeros((t_expr.shape[1], len(genes_shared)), dtype=np.float32)
Xt_raw[:, valid] = t_expr_lin[sel[valid], :].T
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
top1_prop = float((max_frac > 0.5).mean())
entropy = -(comp * np.log(comp + 1e-9)).sum(1)
log(f'Composition diagnostics:')
log(f'  fraction patients with >50% in one prototype: {top1_prop:.3f}')
log(f'  mean entropy: {entropy.mean():.2f} / max possible {math.log(n_proto):.2f}  '
    f'({entropy.mean()/math.log(n_proto):.2%} of max)')

mean_comp = comp.mean(0)
top_order = np.argsort(-mean_comp)
log('Top-10 prototypes in TCGA-LUAD cohort:')
for p in top_order[:10]:
    log(f'  proto_{p:<3d}  mean_comp={mean_comp[p]:.3f}  '
        f'dominant={dom[p]:<25s} tn={dom_st[p]:<6s} n_cells={sizes[p]} trust={trust[p]:.2f}')

# ---------- 7. TCGA-LUAD survival sanity: Cox per prototype ----------
log('Running Cox(OS) per prototype on TCGA-LUAD …')
clin = pd.read_csv(TCGA_CLIN, sep='\t', low_memory=False)
cols = {c.lower():c for c in clin.columns}
sid = cols.get('sampleid') or clin.columns[0]
clin['event'] = clin[cols['vital_status']].astype(str).str.upper().map({'DEAD':1,'DECEASED':1,'ALIVE':0,'LIVING':0})
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
    mu, sd = d[col].mean(), d[col].std()
    d['x'] = (d[col] - mu) / sd
    try:
        cph = CoxPHFitter(penalizer=0.05).fit(d[['event','time','x']], 'time','event')
        rows.append({'proto':p,'dominant':dom[p],'tn':dom_st[p],'n':len(d),
                     'HR':float(np.exp(cph.params_['x'])),
                     'p':float(cph.summary.loc['x','p']),
                     'mean_comp':float(mean_comp[p]),'trust':float(trust[p])})
    except Exception as e:
        rows.append({'proto':p,'dominant':dom[p],'tn':dom_st[p],'n':len(d),'error':str(e)})
cox_df = pd.DataFrame(rows).sort_values('p')
cox_df.to_parquet(OUT/'cox_per_prototype.parquet', index=False)
log(cox_df.head(10).to_string())

# ---------- 8. Save checkpoint ----------
torch.save({'model':model.state_dict(),'n_proto':n_proto,'g_mu':g_mu,'g_sd':g_sd,
            'genes':genes_shared,'centroid':centroid,'pb':pb,'proto_id':proto_id,
            'global_idx':global_idx,'dominant':dom,'sizes':sizes.tolist()},
           CKPT/'t3c_attn_deconv_luad.pt')

(OUT/'eval_metrics.json').write_text(json.dumps({
    'n_prototypes': int(n_proto),
    'sim_val_mae': float(va_mae),
    'sim_mean_proto_r': float(mean_r),
    'tcga_samples': int(comp_df.shape[0]),
    'tcga_entropy_pct_of_max': float(entropy.mean()/math.log(n_proto)),
    'tcga_top1_gt_50pct': float(top1_prop),
    'cox_significant_prototypes_p05': int((cox_df['p']<0.05).sum()) if 'p' in cox_df.columns else 0,
    'cox_significant_prototypes_p01': int((cox_df['p']<0.01).sum()) if 'p' in cox_df.columns else 0,
    'cox_significant_prototypes_p001': int((cox_df['p']<0.001).sum()) if 'p' in cox_df.columns else 0,
    'top_prototypes': [{'proto':int(p),'mean_comp':float(mean_comp[p]),
                        'dominant':dom[p],'tn':dom_st[p],
                        'n_cells':int(sizes[p]),'trust':float(trust[p])}
                       for p in top_order[:10]],
}, indent=2, default=str))
log('== T3c-LUAD done ==')
