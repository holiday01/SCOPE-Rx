"""
LUAD ablation table — Tier A (embedding) × Tier B (deconv) × Tier C (score fusion).

Tier A: which embedding feeds prototype clustering?
  A0 Geneformer V2-104M (current)
  A1 scVI 64-d trained on the LUAD scRNA
  A2 PCA(50) on log1p(X)
  A3 Random Gaussian 50-d
  → re-run Leiden + attention deconv pipeline; compare prototype Cox count + GSE72094 c-index.

Tier B: which deconv method maps bulk → prototype composition?
  B0 Attention deconv (current)         — uses existing results/t3c_luad output
  B1 NNLS                                — non-negative least-squares per patient
  B2 Scaden-style MLP                    — fresh-trained on Dirichlet pseudobulk mixtures

Tier C: score fusion weights (T3f final formula)
  C0 1·z_kill + 0.5·z_onc + 0.7·z_prior  (current)
  C1 z_kill only
  C2 z_kill + 0.5·z_onc                  (no prior)
  C3 z_prior only
  C4 (1/3)·each (equal weights)
  → mean rank percentile of 9 LUAD SOC drugs.

Outputs:
  results/ablation_table/{embedding,deconv,score_fusion}.parquet
  comparison.md
"""
from __future__ import annotations
import json, time, math, os
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import anndata as ad
from lifelines import CoxPHFitter
from scipy.optimize import nnls

ROOT = Path('/home/holiday01/drug_sc')
PROC = ROOT/'data/processed/luad_drug'
T3B  = ROOT/'results/t3b_luad'
T3C  = ROOT/'results/t3c_luad'
T3D  = ROOT/'results/t3d_luad'
T3E  = ROOT/'results/t3e_luad'
T3F  = ROOT/'results/t3f_luad'
T3I  = ROOT/'results/t3i_luad'
SCR  = Path('/mnt/10t/scrna_atac/data/processed/LUAD/luad_scrna_annotated.h5ad')
OUT  = ROOT/'results/ablation_table'; OUT.mkdir(parents=True, exist_ok=True)
DEV  = 'cuda'
SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# =============================================================
# Common loaders
# =============================================================
log('Loading shared resources …')
genes_shared = list(pd.read_parquet(T3C/'prototype_expression.parquet').columns)
proto_meta_existing = pd.read_parquet(T3C/'prototype_meta.parquet')
panC = pd.read_parquet(PROC/'cellline_expression_panCancer.parquet')

# Read scRNA expression for prototype pseudobulk computation
log('Loading LUAD scRNA expression for shared genes …')
a = ad.read_h5ad(SCR, backed='r')
sc_genes = list(a.var_names); sc_i = {g:i for i,g in enumerate(sc_genes)}
sel_sc_idx = np.array([sc_i[g] for g in genes_shared])
N = a.n_obs
rows = []
CHUNK = 4000
for s in range(0, N, CHUNK):
    sub = a.X[s:s+CHUNK, :][:, sel_sc_idx]
    if hasattr(sub,'toarray'): sub = sub.toarray()
    rows.append(np.asarray(sub, dtype=np.float32))
Xn_log = np.vstack(rows)              # log1p(CP10K) values per cell × gene
Xn_lin = np.expm1(Xn_log)             # CP10K
log(f'scRNA expression matrix: {Xn_log.shape}')
obs = a.obs.copy()
a.file.close()

# TCGA-LUAD expression and clinical (for Cox)
log('Loading TCGA-LUAD expression + clinical …')
TCGA_EXPR = Path('/mnt/10t/scrna_atac/data/raw/TCGA_LUAD/TCGA_LUAD_expression.gz')
TCGA_CLIN = Path('/mnt/10t/scrna_atac/data/raw/TCGA_LUAD/TCGA_LUAD_clinical.tsv')
t_expr = pd.read_csv(TCGA_EXPR, sep='\t', index_col=0, low_memory=False)
if t_expr.values.max() < 30:
    t_lin = np.clip(np.expm1(t_expr.values * np.log(2)), 0, None).astype(np.float32)
else:
    t_lin = t_expr.values.astype(np.float32)
t_g = {g:i for i,g in enumerate(t_expr.index)}
sel = np.array([t_g.get(g,-1) for g in genes_shared])
valid = sel >= 0
Xt_raw = np.zeros((t_expr.shape[1], len(genes_shared)), dtype=np.float32)
Xt_raw[:, valid] = t_lin[sel[valid],:].T
rs = Xt_raw.sum(1, keepdims=True) + 1e-6
Xt_norm = np.log1p(Xt_raw / rs * 1e4).astype(np.float32)
tcga_samples = list(t_expr.columns)
log(f'  TCGA-LUAD: {Xt_norm.shape}')

clin = pd.read_csv(TCGA_CLIN, sep='\t', low_memory=False)
cols = {c.lower():c for c in clin.columns}
sid = cols.get('sampleid') or clin.columns[0]
clin['event'] = clin[cols['vital_status']].astype(str).str.upper().map({'DEAD':1,'DECEASED':1,'ALIVE':0,'LIVING':0})
dtd = pd.to_numeric(clin[cols['days_to_death']], errors='coerce')
dtl = pd.to_numeric(clin[cols['days_to_last_followup']], errors='coerce')
clin['time'] = dtd.where(clin['event']==1, dtl)
surv = clin[[sid,'event','time']].rename(columns={sid:'sample'}).dropna()

# =============================================================
# Tier A — embedding ablation
# =============================================================
log('\n' + '='*70 + '\n  Tier A — embedding ablation\n' + '='*70)

import scanpy as sc

def cluster_to_proto(emb, target=58):
    _ad = ad.AnnData(emb.astype(np.float32))
    n_pca = min(50, emb.shape[1]-1, emb.shape[0]-1)
    if n_pca >= 5:
        sc.pp.pca(_ad, n_comps=n_pca, zero_center=True)
        sc.pp.neighbors(_ad, n_neighbors=15, use_rep='X_pca')
    else:
        sc.pp.neighbors(_ad, n_neighbors=15, use_rep='X')
    res = 1.5
    for _ in range(7):
        sc.tl.leiden(_ad, resolution=res, random_state=0)
        n_c = _ad.obs['leiden'].nunique()
        if target*0.85 <= n_c <= target*1.25: break
        res *= (target/max(n_c,1))**0.7
    proto_id = _ad.obs['leiden'].astype(int).values
    return proto_id

def pseudobulk_log_cp10k(proto_id, Xn_lin):
    n_proto = int(proto_id.max())+1
    pb = np.zeros((n_proto, Xn_lin.shape[1]), dtype=np.float32)
    sizes = np.zeros(n_proto, dtype=np.int64)
    for p in range(n_proto):
        m = proto_id == p
        if m.sum() == 0: continue
        pb[p] = np.log1p(Xn_lin[m].mean(0))
        sizes[p] = m.sum()
    return pb, sizes

class AttnDeconv(nn.Module):
    def __init__(self, d_in, n_proto, d_hid=256, p=0.1):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(d_in,512), nn.LayerNorm(512), nn.ReLU(),
                               nn.Dropout(p), nn.Linear(512,d_hid))
        self.k = nn.Parameter(torch.randn(n_proto, d_hid)*0.02)
        self.t = nn.Parameter(torch.tensor(1.0))
    def forward(self,x):
        return F.softmax((self.q(x)@self.k.T)/self.t.clamp(min=0.1), dim=-1)

def train_deconv_eval_tcga(pb, label, sim_n=40000, epochs=20):
    """Train attention deconv on Dirichlet sim → apply to TCGA-LUAD → return composition + per-prototype Cox."""
    n_proto = pb.shape[0]
    rng = np.random.default_rng(SEED)
    W = rng.dirichlet(alpha=np.ones(n_proto)*0.5, size=sim_n).astype(np.float32)
    W[:sim_n//3] = rng.dirichlet(alpha=np.ones(n_proto)*2.0, size=sim_n//3).astype(np.float32)
    Xs = W @ pb + rng.normal(0, 0.10, (sim_n, pb.shape[1])).astype(np.float32)
    g_mu = Xs.mean(0); g_sd = Xs.std(0)+1e-6
    Xs_z = (Xs - g_mu)/g_sd
    Xtcga = (Xt_norm - g_mu)/g_sd

    model = AttnDeconv(pb.shape[1], n_proto).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt_t = torch.from_numpy(Xs_z).to(DEV); Wt = torch.from_numpy(W).to(DEV)
    BS = 512
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt_t), device=DEV)
        for s in range(0, len(perm), BS):
            idx = perm[s:s+BS]
            pred = model(Xt_t[idx])
            loss = F.l1_loss(pred, Wt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        comp = model(torch.from_numpy(Xtcga).to(DEV)).cpu().numpy()
    return comp, n_proto

def cox_per_proto(comp, samples, surv_df, n_proto):
    df = pd.DataFrame(comp, index=samples, columns=[f'p{i}' for i in range(n_proto)])
    m = df.reset_index().rename(columns={'index':'sample'}).merge(surv_df, on='sample', how='inner')
    m = m[m['time']>0]
    rows = []
    for p in range(n_proto):
        c = f'p{p}'
        d = m[['event','time',c]].dropna()
        if len(d)<50 or d[c].std()<1e-5: continue
        d = d.assign(x=(d[c]-d[c].mean())/d[c].std())
        try:
            cph = CoxPHFitter(penalizer=0.05).fit(d[['event','time','x']], 'time','event')
            rows.append({'proto':p,'HR':float(np.exp(cph.params_['x'])),
                         'p':float(cph.summary.loc['x','p'])})
        except: pass
    return pd.DataFrame(rows)

# Inline GSE72094 phenotype loader for composite-risk c-index evaluation
import gzip
def parse_g72_phen():
    path = ROOT/'data/external_LUAD/GSE72094_series_matrix.txt.gz'
    sample_ids = None
    char_rows = []
    with gzip.open(path, 'rt', errors='replace') as f:
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                sample_ids = [x.strip().strip('"') for x in line.split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1'):
                cells = [x.strip().strip('"') for x in line.split('\t')[1:]]
                char_rows.append(cells)
            elif line.startswith('!series_matrix_table_begin'):
                break
    n = len(sample_ids)
    per = [{} for _ in range(n)]
    for cells in char_rows:
        for i, c in enumerate(cells[:n]):
            if not c or ':' not in c: continue
            k,v = c.split(':',1); per[i][k.strip().lower()] = v.strip()
    p_full = pd.DataFrame(per, index=sample_ids)
    p_full['os_event'] = p_full.get('vital_status', pd.Series(['']*n)).astype(str).str.lower().map(
        lambda s: 1 if 'dead' in s else (0 if 'alive' in s else np.nan))
    p_full['os_time'] = pd.to_numeric(p_full.get('survival_time_in_days', pd.Series([np.nan]*n)), errors='coerce')
    p_full['age'] = pd.to_numeric(p_full.get('age_at_diagnosis', pd.Series([np.nan]*n)), errors='coerce')
    sex = p_full.get('gender', pd.Series(['']*n)).astype(str).str.lower()
    p_full['male'] = sex.apply(lambda s: 1.0 if s in ('m','male') else (0.0 if s in ('f','female') else np.nan))
    return p_full

g72_phen = parse_g72_phen()
log(f'GSE72094 phen: {g72_phen.shape}, OS events parsed: {g72_phen["os_event"].notna().sum()}')

def composite_cindex_g72(comp_g72_arr, cox_proto, n_proto):
    risk_w = np.zeros(n_proto, dtype=np.float32)
    for _, r in cox_proto.iterrows():
        if float(r['p']) < 0.1:
            risk_w[int(r['proto'])] = float(np.log(r['HR']))
    risk = comp_g72_arr.dot(risk_w)
    df = g72_phen.copy()
    df['risk'] = (risk - risk.mean())/(risk.std()+1e-6)
    cov = ['age','male']
    d = df[['os_event','os_time','risk']+cov].dropna()
    d = d[d['os_time']>0]
    if len(d) < 50: return None
    try:
        cph = CoxPHFitter(penalizer=0.05).fit(d, 'os_time','os_event')
        return {'n':int(len(d)), 'HR_per_SD':float(np.exp(cph.params_['risk'])),
                'p':float(cph.summary.loc['risk','p']),
                'c_index':float(cph.concordance_index_)}
    except: return None

# ----- A0: Geneformer (current) -----
log('\n[A0] Geneformer V2-104M …')
A0_emb = pd.read_parquet(T3B/'geneformer_v2_meanpool_luad.parquet').drop(columns='cell_global_idx').values.astype(np.float32)
A0_proto_id = cluster_to_proto(A0_emb)
A0_n = int(A0_proto_id.max())+1
A0_pb, _ = pseudobulk_log_cp10k(A0_proto_id, Xn_lin)
A0_comp, _ = train_deconv_eval_tcga(A0_pb, 'A0_Geneformer')
A0_cox = cox_per_proto(A0_comp, tcga_samples, surv, A0_n)
log(f'  A0: n_proto={A0_n}  cox p<0.05={(A0_cox["p"]<0.05).sum()}  '
    f'entropy={(-(A0_comp*np.log(A0_comp+1e-9)).sum(1)).mean()/np.log(A0_n):.0%}')

# ----- A1: scVI -----
log('\n[A1] scVI 64-d …')
try:
    import scvi
    a2 = ad.AnnData(Xn_lin.astype(np.float32))
    a2.obs = obs.copy(); a2.var_names = genes_shared
    # convert to integer counts approximation: round CP10K to int
    a2.X = np.round(a2.X).astype(np.int32)
    scvi.model.SCVI.setup_anndata(a2)
    sv = scvi.model.SCVI(a2, n_layers=2, n_latent=64)
    sv.train(max_epochs=80, train_size=0.9, batch_size=256, accelerator='gpu', devices=1, plan_kwargs={'lr':1e-3})
    A1_emb = sv.get_latent_representation()
    log(f'  scVI emb: {A1_emb.shape}')
    A1_proto_id = cluster_to_proto(A1_emb)
    A1_n = int(A1_proto_id.max())+1
    A1_pb, _ = pseudobulk_log_cp10k(A1_proto_id, Xn_lin)
    A1_comp, _ = train_deconv_eval_tcga(A1_pb, 'A1_scVI')
    A1_cox = cox_per_proto(A1_comp, tcga_samples, surv, A1_n)
    A1_ok = True
    log(f'  A1: n_proto={A1_n}  cox p<0.05={(A1_cox["p"]<0.05).sum()}  '
        f'entropy={(-(A1_comp*np.log(A1_comp+1e-9)).sum(1)).mean()/np.log(A1_n):.0%}')
except Exception as e:
    log(f'  A1 scVI failed: {e}')
    A1_ok = False; A1_cox = pd.DataFrame(); A1_n = 0; A1_comp = None

# ----- A2: PCA(50) -----
log('\n[A2] PCA(50) …')
from sklearn.decomposition import PCA
A2_emb = PCA(n_components=50, random_state=0).fit_transform(Xn_log.astype(np.float32))
A2_proto_id = cluster_to_proto(A2_emb)
A2_n = int(A2_proto_id.max())+1
A2_pb, _ = pseudobulk_log_cp10k(A2_proto_id, Xn_lin)
A2_comp, _ = train_deconv_eval_tcga(A2_pb, 'A2_PCA')
A2_cox = cox_per_proto(A2_comp, tcga_samples, surv, A2_n)
log(f'  A2: n_proto={A2_n}  cox p<0.05={(A2_cox["p"]<0.05).sum()}  '
    f'entropy={(-(A2_comp*np.log(A2_comp+1e-9)).sum(1)).mean()/np.log(A2_n):.0%}')

# ----- A3: Random -----
log('\n[A3] Random Gaussian 50-d …')
A3_emb = np.random.default_rng(SEED).standard_normal((N, 50)).astype(np.float32)
A3_proto_id = cluster_to_proto(A3_emb)
A3_n = int(A3_proto_id.max())+1
A3_pb, _ = pseudobulk_log_cp10k(A3_proto_id, Xn_lin)
A3_comp, _ = train_deconv_eval_tcga(A3_pb, 'A3_Random')
A3_cox = cox_per_proto(A3_comp, tcga_samples, surv, A3_n)
log(f'  A3: n_proto={A3_n}  cox p<0.05={(A3_cox["p"]<0.05).sum()}  '
    f'entropy={(-(A3_comp*np.log(A3_comp+1e-9)).sum(1)).mean()/np.log(A3_n):.0%}')

A_results = []
for label, n_proto, comp, cox in [('A0_Geneformer', A0_n, A0_comp, A0_cox),
                                   ('A1_scVI', A1_n, A1_comp, A1_cox) if A1_ok else (None, 0, None, None),
                                   ('A2_PCA',  A2_n,  A2_comp,  A2_cox),
                                   ('A3_Random', A3_n, A3_comp, A3_cox)]:
    if label is None: continue
    ent = (-(comp*np.log(comp+1e-9)).sum(1)).mean()/np.log(n_proto)
    top1 = float((comp.max(1)>0.5).mean())
    A_results.append({'variant':label, 'n_proto':n_proto,
                      'entropy_pct':ent, 'top1_collapse_pct':top1,
                      'cox_p05':int((cox['p']<0.05).sum()),
                      'cox_p01':int((cox['p']<0.01).sum()),
                      'cox_p001':int((cox['p']<0.001).sum())})
A_df = pd.DataFrame(A_results)
A_df.to_parquet(OUT/'embedding.parquet', index=False)
log(f'\nTier A summary:\n{A_df.to_string(index=False)}')

# =============================================================
# Tier B — deconv method ablation (use Geneformer prototypes as anchor)
# =============================================================
log('\n' + '='*70 + '\n  Tier B — deconv method ablation\n' + '='*70)

# B0: existing attention deconv
B0_comp = pd.read_parquet(T3C/'tcga_composition.parquet').values
B0_n = B0_comp.shape[1]
log(f'[B0] Attention deconv: comp shape {B0_comp.shape}')

# B1: NNLS — for each TCGA bulk sample, solve W >= 0 minimising ||W·pb - x||
log('[B1] NNLS …')
pb_ref = pd.read_parquet(T3C/'prototype_expression.parquet').values.astype(np.float32)  # n_proto × n_genes
B1_comp = np.zeros((Xt_norm.shape[0], pb_ref.shape[0]), dtype=np.float32)
for i in range(Xt_norm.shape[0]):
    w, _ = nnls(pb_ref.T, Xt_norm[i])
    s = w.sum()
    B1_comp[i] = w / s if s > 1e-9 else np.full_like(w, 1.0/len(w))
log(f'  B1 done: comp shape {B1_comp.shape}')

# B2: Scaden-style MLP — train an MLP regressor on Dirichlet pseudobulk mixtures
log('[B2] Scaden-style MLP …')
class ScadenMLP(nn.Module):
    def __init__(self, d_in, n_proto):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, n_proto))
    def forward(self,x): return F.softmax(self.net(x), dim=-1)

n_proto = pb_ref.shape[0]
rng = np.random.default_rng(SEED)
W = rng.dirichlet(np.ones(n_proto)*0.5, size=40000).astype(np.float32)
W[:13000] = rng.dirichlet(np.ones(n_proto)*2.0, size=13000).astype(np.float32)
Xs = W @ pb_ref + rng.normal(0, 0.10, (40000, pb_ref.shape[1])).astype(np.float32)
g_mu = Xs.mean(0); g_sd = Xs.std(0)+1e-6
Xs_z = (Xs - g_mu)/g_sd
Xt_z = (Xt_norm - g_mu)/g_sd
mdl = ScadenMLP(pb_ref.shape[1], n_proto).to(DEV)
opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-5)
Xs_tt = torch.from_numpy(Xs_z).to(DEV); Wt = torch.from_numpy(W).to(DEV)
BS = 512
for ep in range(20):
    mdl.train()
    perm = torch.randperm(len(Xs_tt), device=DEV)
    for s in range(0, len(perm), BS):
        idx = perm[s:s+BS]
        loss = F.l1_loss(mdl(Xs_tt[idx]), Wt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
mdl.eval()
with torch.no_grad():
    B2_comp = mdl(torch.from_numpy(Xt_z).to(DEV)).cpu().numpy()
log(f'  B2 done: comp shape {B2_comp.shape}')

# Evaluate Tier B
B_results = []
for label, comp in [('B0_Attention', B0_comp), ('B1_NNLS', B1_comp), ('B2_Scaden', B2_comp)]:
    ent = (-(comp*np.log(comp+1e-9)).sum(1)).mean()/np.log(comp.shape[1])
    top1 = float((comp.max(1)>0.5).mean())
    cox = cox_per_proto(comp, tcga_samples, surv, comp.shape[1])
    B_results.append({'method':label, 'entropy_pct':float(ent), 'top1_collapse_pct':top1,
                      'cox_p05':int((cox['p']<0.05).sum()),
                      'cox_p01':int((cox['p']<0.01).sum()),
                      'cox_p001':int((cox['p']<0.001).sum())})
B_df = pd.DataFrame(B_results)
B_df.to_parquet(OUT/'deconv.parquet', index=False)
log(f'\nTier B summary:\n{B_df.to_string(index=False)}')

# =============================================================
# Tier C — score-fusion ablation
# =============================================================
log('\n' + '='*70 + '\n  Tier C — score-fusion ablation\n' + '='*70)

f = pd.read_parquet(T3F/'drug_final_score.parquet')
SOC = ['gefitinib','trametinib','osimertinib','selumetinib','alectinib',
       'erlotinib','afatinib','brigatinib','pemetrexed']

variants = {
    'C0_kill+0.5*onc+0.7*prior': (1.0, 0.5, 0.7),
    'C1_kill_only':              (1.0, 0.0, 0.0),
    'C2_kill+0.5*onc':           (1.0, 0.5, 0.0),
    'C3_prior_only':             (0.0, 0.0, 1.0),
    'C4_equal_third':            (1.0/3, 1.0/3, 1.0/3),
}

C_rows = []
for vlabel, (wk, wo, wp) in variants.items():
    f2 = f.copy()
    f2['score'] = wk*f2['z_kill'] + wo*f2['z_onc'] + wp*f2['z_prior']
    f2 = f2.sort_values('score', ascending=False).reset_index(drop=True)
    f2['drug_lc'] = f2['drug'].str.lower()
    rank_pcts = []
    rank_dict = {}
    for d in SOC:
        h = f2[f2['drug_lc']==d]
        if len(h):
            r = int(h.index[0])+1
            rank_pcts.append(r/len(f2)*100)
            rank_dict[d] = r
    C_rows.append({'variant':vlabel,'weights':f'{wk}/{wo}/{wp}',
                   'n_soc_found':len(rank_pcts),
                   'mean_pct':float(np.mean(rank_pcts)),
                   'median_pct':float(np.median(rank_pcts)),
                   'max_rank':int(max(rank_dict.values())) if rank_dict else None,
                   **{f'rank_{d}':rank_dict.get(d) for d in SOC}})
C_df = pd.DataFrame(C_rows)
C_df.to_parquet(OUT/'score_fusion.parquet', index=False)
log(f'\nTier C summary:\n{C_df[["variant","mean_pct","median_pct","max_rank"]].to_string(index=False)}')

# =============================================================
# Comparison report
# =============================================================
md = ['# LUAD ablation table — A (embedding) × B (deconv) × C (score fusion)\n',
      '## Tier A — embedding choice (Leiden→58 prototypes; same attention deconv)',
      '| Variant | n_proto | entropy% | top-1 collapse% | Cox p<0.05 | p<0.01 | p<0.001 |',
      '|---|---:|---:|---:|---:|---:|---:|']
for _, r in A_df.iterrows():
    md.append(f"| **{r['variant']}** | {r['n_proto']} | {r['entropy_pct']:.0%} | "
              f"{r['top1_collapse_pct']:.1%} | {r['cox_p05']} | {r['cox_p01']} | {r['cox_p001']} |")

md += ['\n## Tier B — deconvolution method (Geneformer prototypes fixed)',
       '| Method | entropy% | top-1 collapse% | Cox p<0.05 | p<0.01 | p<0.001 |',
       '|---|---:|---:|---:|---:|---:|']
for _, r in B_df.iterrows():
    md.append(f"| **{r['method']}** | {r['entropy_pct']:.0%} | {r['top1_collapse_pct']:.1%} | "
              f"{r['cox_p05']} | {r['cox_p01']} | {r['cox_p001']} |")

md += ['\n## Tier C — score-fusion weights (T3f formula); rank of 9 LUAD SOC drugs',
       '| Variant | weights k/o/p | mean rank% | median rank% | max rank |',
       '|---|---|---:|---:|---:|']
for _, r in C_df.iterrows():
    md.append(f"| **{r['variant']}** | {r['weights']} | {r['mean_pct']:.1f}% | "
              f"{r['median_pct']:.1f}% | {r['max_rank']} |")

md += ['\n### SOC drug rank by Tier-C variant\n',
       '| Drug | ' + ' | '.join(C_df['variant'].tolist()) + ' |',
       '|---|' + ''.join('---:|' for _ in C_df['variant']) ]
for d in SOC:
    row = f'| {d} |'
    for _, r in C_df.iterrows():
        v = r.get(f'rank_{d}', None)
        row += f' {v} |' if v is not None else ' — |'
    md.append(row)

md += ['\n## Interpretation',
       '- **Tier A**: if Random (A3) yields zero or near-zero Cox-significant prototypes while Geneformer (A0) gives 15+, the embedding tower is load-bearing. scVI vs Geneformer comparison shows whether foundation-model pre-training adds signal beyond standard scRNA latent.',
       '- **Tier B**: if Attention (B0) keeps composition entropy ~88% and Cox p<0.05 ~15 while NNLS / Scaden collapse, the attention design is what enables non-collapsing deconv.',
       '- **Tier C**: comparing C1 (kill only) and C3 (prior only) against C0 reveals the marginal contribution of each scoring layer. C0 should out-rank both ablations on SOC drug recovery.']
(OUT/'comparison.md').write_text('\n'.join(md))
log(f'\nReport: {OUT/"comparison.md"}')

(OUT/'eval_metrics.json').write_text(json.dumps({
    'tier_A_embedding': A_results,
    'tier_B_deconv': B_results,
    'tier_C_score_fusion': C_rows,
}, indent=2, default=str))
log('== ablation table done ==')
