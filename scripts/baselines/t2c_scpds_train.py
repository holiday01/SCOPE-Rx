"""
T2c — scPDS-style pathway transformer baseline.

Same supervision / eval as T2b, but:
  input:  pathway activation score per sample (mean of z-scored log-TPM over pathway genes)
  encoder: Transformer over pathway tokens (learnable embedding per pathway + score)
  heads:  same multi-drug regression head + DANN domain head

Pathway libraries: MSigDB Hallmark (50) + KEGG 2026 (352) = 402 tokens.
"""
from __future__ import annotations
import json, time, math, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Function
import gseapy as gp

ROOT = Path('/home/holiday01/drug_sc')
PROC = ROOT/'data/processed/hcc_drug'
OUT  = ROOT/'results/t2c'; OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT/'checkpoints'; CKPT.mkdir(parents=True, exist_ok=True)
DEV  = 'cuda'
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

HELD_OUT_HCC = ['ACH-000221','ACH-000480','ACH-000625']
MIN_BULK_OBS_PER_DRUG = 30
N_SC_CELLS = 20000
D_MODEL    = 128
N_HEAD     = 4
N_LAYERS   = 3
LR         = 5e-4
EPOCHS     = 40
BS_BULK    = 128
BS_SC      = 128
LAMBDA_DOM_MAX = 0.3

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Pathway gene sets ----------
log('Loading pathway libraries …')
H = gp.get_library('MSigDB_Hallmark_2020', organism='Human')
K = gp.get_library('KEGG_2026', organism='Human')
pathway_map = {f'H::{k}':set(v) for k,v in H.items()}
pathway_map.update({f'K::{k}':set(v) for k,v in K.items()})
log(f'Total pathways: {len(pathway_map)}')

# ---------- 2. Data ----------
genes = pd.read_parquet(PROC/'gene_universe.parquet')['gene'].tolist()
gene_to_i = {g:i for i,g in enumerate(genes)}
# Build pathway × gene sparse index (list of gene indices per pathway)
pw_names, pw_gene_idx = [], []
for pn, gs in pathway_map.items():
    idx = [gene_to_i[g] for g in gs if g in gene_to_i]
    if len(idx) >= 5:
        pw_names.append(pn); pw_gene_idx.append(np.array(idx, dtype=np.int64))
log(f'Pathways kept (≥5 genes in expression): {len(pw_names)}')

# ---------- 3. Expression + AUC ----------
expr_all = pd.read_parquet(PROC/'cellline_expression_panCancer.parquet')
long_tab = pd.read_parquet(PROC/'drug_response_long.parquet')
agg = (long_tab.dropna(subset=['auc']).groupby(['drug','ModelID'])['auc'].mean().unstack('ModelID'))
drug_obs = agg.notna().sum(axis=1)
agg = agg.loc[drug_obs[drug_obs>=MIN_BULK_OBS_PER_DRUG].index]
common_lines = [c for c in expr_all.index if c in agg.columns]
agg = agg[common_lines]
agg_mean = agg.mean(1); agg_std = agg.std(1).replace(0,np.nan)
agg_z = agg.sub(agg_mean,axis=0).div(agg_std,axis=0)
Y = agg_z.T.values.astype(np.float32); M = (~np.isnan(Y)).astype(np.float32); Y=np.nan_to_num(Y)

X_tpm = expr_all.loc[common_lines].values.astype(np.float32)
X = np.log1p(X_tpm)
g_mu = X.mean(0); g_sd = X.std(0) + 1e-6
Xz = (X - g_mu) / g_sd  # (lines × genes), z-scored per gene

def sample_pathway_scores(Xz):
    """Compute per-sample pathway activation = mean of z-scored genes in pathway."""
    out = np.zeros((Xz.shape[0], len(pw_gene_idx)), dtype=np.float32)
    for j, idx in enumerate(pw_gene_idx):
        out[:, j] = Xz[:, idx].mean(axis=1)
    return out

log('Computing bulk pathway scores …')
P_bulk = sample_pathway_scores(Xz)
log(f'P_bulk {P_bulk.shape}')

# ---------- 4. scRNA sample ----------
import anndata as ad
a = ad.read_h5ad(ROOT/'data/scRNA_HCC_integrated/GSE223204GSE202642GSE162616_bbknn_tumornormal.h5ad', backed='r')
sc_genes = list(a.var_names); sc_i = {g:i for i,g in enumerate(sc_genes)}
sel_sc_idx = np.array([sc_i[g] for g in genes])
sample_idx = np.random.choice(a.n_obs, size=min(N_SC_CELLS,a.n_obs), replace=False); sample_idx.sort()
assert 'counts' in a.layers, "expected layers['counts']"
Xs_parts = []
for i in range(0,len(sample_idx),4000):
    chunk = sample_idx[i:i+4000]
    sub = a.layers['counts'][chunk,:][:,sel_sc_idx]
    if hasattr(sub,'toarray'): sub = sub.toarray()
    Xs_parts.append(np.asarray(sub,dtype=np.float32))
a.file.close()
Xs = np.vstack(Xs_parts)
rs = Xs.sum(1,keepdims=True)+1e-6
Xs = np.log1p(Xs/rs*1e4)
Xs = (Xs - g_mu) / g_sd
P_sc = sample_pathway_scores(Xs)
log(f'P_sc {P_sc.shape}')

# ---------- 5. Train / held-out split ----------
held_mask = np.array([ln in HELD_OUT_HCC for ln in common_lines])
Pb_tr, Pb_te = P_bulk[~held_mask], P_bulk[held_mask]
Y_tr, Y_te = Y[~held_mask], Y[held_mask]
M_tr, M_te = M[~held_mask], M[held_mask]
log(f'Train lines={len(Pb_tr)}  Held-out={len(Pb_te)}  drugs={Y.shape[1]}  pathways={Pb_tr.shape[1]}')

# ---------- 6. Model ----------
class GRL(Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam=lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return g.neg()*ctx.lam, None
def grl(x,lam): return GRL.apply(x,lam)

N_PW = P_bulk.shape[1]

class PathwayTransformer(nn.Module):
    """Each pathway is a token: embedding(pathway_id) + linear(score).
       Transformer encoder -> [CLS]-like mean pool -> latent."""
    def __init__(self, n_pw, d=128, nhead=4, nlayers=3, drop=0.2):
        super().__init__()
        self.pw_emb = nn.Embedding(n_pw, d)
        self.score_proj = nn.Linear(1, d)
        self.cls = nn.Parameter(torch.randn(1,1,d)*0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=2*d,
                                               dropout=drop, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d)
    def forward(self, scores):  # (B, n_pw)
        B = scores.shape[0]
        pw_ids = torch.arange(N_PW, device=scores.device).unsqueeze(0).expand(B,-1)
        tok = self.pw_emb(pw_ids) + self.score_proj(scores.unsqueeze(-1))
        cls = self.cls.expand(B,-1,-1)
        x = torch.cat([cls, tok], dim=1)
        x = self.enc(x)
        return self.norm(x[:,0])  # (B, d)

class DrugHead(nn.Module):
    def __init__(self,d,n): super().__init__(); self.lin=nn.Linear(d,n)
    def forward(self,z): return self.lin(z)
class DomainHead(nn.Module):
    def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,z): return self.net(z)

enc = PathwayTransformer(N_PW, D_MODEL, N_HEAD, N_LAYERS).to(DEV)
dhead = DrugHead(D_MODEL, Y.shape[1]).to(DEV)
dom   = DomainHead(D_MODEL).to(DEV)
opt = torch.optim.Adam(list(enc.parameters())+list(dhead.parameters())+list(dom.parameters()),
                       lr=LR, weight_decay=1e-5)

log(f'Model params: {sum(p.numel() for p in enc.parameters())/1e6:.2f}M (encoder) + {sum(p.numel() for p in dhead.parameters())/1e6:.2f}M (drug head)')

# ---------- 7. Train ----------
Pbt = torch.from_numpy(Pb_tr).to(DEV); Yt=torch.from_numpy(Y_tr).to(DEV); Mt=torch.from_numpy(M_tr).to(DEV)
Pst = torch.from_numpy(P_sc).to(DEV)
Pbte = torch.from_numpy(Pb_te).to(DEV); Yte=torch.from_numpy(Y_te); Mte=torch.from_numpy(M_te)

metrics=[]
for ep in range(EPOCHS):
    enc.train(); dhead.train(); dom.train()
    p = ep/max(EPOCHS-1,1)
    lam = LAMBDA_DOM_MAX * (2/(1+math.exp(-10*p)) - 1)
    iters = max(len(Pbt)//BS_BULK, 1) * 4
    loss_r=loss_d=0.0
    for it in range(iters):
        bi = torch.randint(0,len(Pbt),(BS_BULK,),device=DEV)
        si = torch.randint(0,len(Pst),(BS_SC,), device=DEV)
        z_b = enc(Pbt[bi]); z_s = enc(Pst[si])
        pred = dhead(z_b)
        reg = (((pred - Yt[bi])**2) * Mt[bi]).sum() / (Mt[bi].sum()+1e-6)
        z_all = torch.cat([z_b,z_s],0)
        dom_logits = dom(grl(z_all, lam))
        dom_label = torch.cat([torch.zeros(len(z_b),dtype=torch.long,device=DEV),
                               torch.ones(len(z_s), dtype=torch.long,device=DEV)],0)
        dloss = F.cross_entropy(dom_logits, dom_label)
        loss = reg + lam*dloss
        opt.zero_grad(); loss.backward(); opt.step()
        loss_r+=reg.item(); loss_d+=dloss.item()

    enc.eval(); dhead.eval()
    with torch.no_grad():
        pred_te = dhead(enc(Pbte)).cpu().numpy()
    # per-line Spearman (primary metric)
    from scipy.stats import spearmanr
    sps=[]
    for i in range(len(Pbte)):
        m = M_te[i].astype(bool)
        if m.sum()<20: continue
        s,_ = spearmanr(pred_te[i,m], Y_te[i,m])
        if not np.isnan(s): sps.append(s)
    mean_sp = float(np.mean(sps)) if sps else float('nan')
    metrics.append({'epoch':ep+1,'reg_loss':loss_r/iters,'dom_loss':loss_d/iters,
                    'mean_line_spearman_heldout':mean_sp,'lambda':lam})
    log(f'ep {ep+1:02d}/{EPOCHS}  reg={loss_r/iters:.3f}  dom={loss_d/iters:.3f}  lam={lam:.3f}  sp̄={mean_sp:.3f}')

# ---------- 8. Final eval + save ----------
enc.eval(); dhead.eval()
with torch.no_grad():
    pred_te = dhead(enc(Pbte)).cpu().numpy()
pred_raw = pred_te * agg_std.values[None,:] + agg_mean.values[None,:]
from scipy.stats import spearmanr, pearsonr
lines_meta = pd.read_parquet(PROC/'cellline_meta.parquet').set_index('ModelID')
rows=[]
for i,ln in enumerate(HELD_OUT_HCC):
    m = M_te[i].astype(bool)
    sp,_ = spearmanr(pred_te[i,m], Y_te[i,m]) if m.sum()>=20 else (np.nan,0)
    pr,_ = pearsonr(pred_te[i,m], Y_te[i,m]) if m.sum()>=20 else (np.nan,0)
    rows.append({'line':ln,'cell':lines_meta.loc[ln,'cell_line'],'n':int(m.sum()),
                 'spearman':float(sp),'pearson':float(pr)})
perline = pd.DataFrame(rows)
print('\n=== Per-held-out-line ranking (scPDS) ===')
print(perline.to_string(index=False))
perline.to_parquet(OUT/'per_line_rank_eval.parquet', index=False)

torch.save({'enc':enc.state_dict(),'dhead':dhead.state_dict(),'dom':dom.state_dict(),
            'drugs':list(agg.index),'pw_names':pw_names,
            'g_mu':g_mu,'g_sd':g_sd,'drug_mean':agg_mean.values,'drug_std':agg_std.values,
            'genes':genes,'pw_gene_idx':[idx.tolist() for idx in pw_gene_idx]},
           CKPT/'t2c_scpds.pt')

# scRNA inference
with torch.no_grad():
    preds=[]
    for i in range(0,len(Pst),1024):
        preds.append(dhead(enc(Pst[i:i+1024])).cpu().numpy())
    preds = np.vstack(preds)
pd.DataFrame(preds, columns=agg.index).to_parquet(OUT/'scrna_drug_predictions_zscore_20k.parquet', index=False)

summary = {
    'baseline':'scPDS',
    'n_drugs':int(Y.shape[1]),'n_pathways':N_PW,
    'n_train_lines':int(len(Pb_tr)),'n_heldout':int(len(Pb_te)),
    'heldout_lines':HELD_OUT_HCC,
    'per_line':rows,
    'mean_spearman':float(np.nanmean([r['spearman'] for r in rows])),
    'mean_pearson':float(np.nanmean([r['pearson']  for r in rows])),
    'per_epoch':metrics,
}
(OUT/'eval_metrics.json').write_text(json.dumps(summary, indent=2))
log(f'mean per-line Spearman (scPDS) = {summary["mean_spearman"]:.3f}')
log('== T2c done ==')
