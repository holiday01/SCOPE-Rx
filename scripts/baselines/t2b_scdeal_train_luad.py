"""
T2b-LUAD — scDEAL drug encoder retrained for LUAD scRNA target domain.

Same architecture / training schedule as t2b_scdeal_train.py, with three changes:
  - PROC = data/processed/luad_drug (LUAD-specific gene universe = 17,180 genes)
  - Target domain = all 20k cells in luad_scrna_annotated.h5ad
  - Held-out LUAD lines (EGFR/KRAS-mutant references):
        ACH-000012 HCC827  (EGFR ex19)
        ACH-000587 NCIH1975 (EGFR T790M/L858R)
        ACH-000779 PC9     (EGFR ex19)
  - Source h5ad is log1p(normalize_total(1e4)); raw counts reconstructed via
    expm1(X) for the DANN target distribution.
"""
from __future__ import annotations
import json, time, math, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Function

ROOT = Path('/home/holiday01/drug_sc')
PROC = ROOT/'data/processed/luad_drug'
SCR  = Path('/mnt/10t/scrna_atac/data/processed/LUAD/luad_scrna_annotated.h5ad')
OUT  = ROOT/'results/t2b_luad'; OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT/'checkpoints'
DEV  = 'cuda'
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

HELD_OUT_LUAD = ['ACH-000012','ACH-000587','ACH-000779']  # HCC827, H1975, PC9
MIN_BULK_OBS_PER_DRUG = 30
N_SC_CELLS = 20000
EMB_DIM    = 256
HID_DIM    = 1024
LR         = 1e-3
EPOCHS     = 40
BS_BULK    = 128
BS_SC      = 128
LAMBDA_DOM_MAX = 0.3

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Tables ----------
log('Loading processed parquet tables …')
expr_all = pd.read_parquet(PROC/'cellline_expression_panCancer.parquet')
long_tab = pd.read_parquet(PROC/'drug_response_long.parquet')
luad_meta = pd.read_parquet(PROC/'cellline_meta.parquet')
genes = pd.read_parquet(PROC/'gene_universe.parquet')['gene'].tolist()
assert expr_all.shape[1] == len(genes)
log(f'Bulk: {expr_all.shape}  Drug long: {len(long_tab)}  LUAD lines: {len(luad_meta)}')

# ---------- 2. drug × line AUC matrix ----------
agg = (long_tab.dropna(subset=['auc'])
       .groupby(['drug','ModelID'])['auc'].mean().unstack('ModelID'))
log(f'Initial drug×line AUC matrix: {agg.shape}  non-null={agg.notna().sum().sum():,}')
drug_obs_count = agg.notna().sum(axis=1)
drugs_keep = drug_obs_count[drug_obs_count >= MIN_BULK_OBS_PER_DRUG].index
agg = agg.loc[drugs_keep]
log(f'After min-obs filter: {agg.shape}')

common_lines = [c for c in expr_all.index if c in agg.columns]
agg = agg[common_lines]
log(f'Aligned lines: {len(common_lines)}')

agg_mean = agg.mean(axis=1); agg_std = agg.std(axis=1).replace(0, np.nan)
agg_z = agg.sub(agg_mean, axis=0).div(agg_std, axis=0)
Y = agg_z.T.values.astype(np.float32)
M = (~np.isnan(Y)).astype(np.float32)
Y = np.nan_to_num(Y, nan=0.0)
log(f'Y (line×drug): {Y.shape}  observed={M.sum():,.0f}  per-drug mean obs={M.sum(0).mean():.1f}')

# ---------- 3. Expression tensors ----------
X = expr_all.loc[common_lines].values.astype(np.float32)
X = np.log1p(X)
gene_mu = X.mean(0); gene_sd = X.std(0) + 1e-6
X = (X - gene_mu) / gene_sd
log(f'X bulk: {X.shape}')

# ---------- 4. scRNA target domain ----------
import anndata as ad
a = ad.read_h5ad(SCR, backed='r')
sc_all_genes = list(a.var_names)
gene_to_sc_idx = {g:i for i,g in enumerate(sc_all_genes)}
sel_sc_idx = np.array([gene_to_sc_idx[g] for g in genes])
n_total = a.n_obs
sample_idx = np.random.choice(n_total, size=min(N_SC_CELLS, n_total), replace=False); sample_idx.sort()
log(f'Sampling {len(sample_idx)} of {n_total} cells …')

Xs_list = []
CHUNK = 4000
for i in range(0, len(sample_idx), CHUNK):
    chunk = sample_idx[i:i+CHUNK]
    sub = a.X[chunk, :][:, sel_sc_idx]
    if hasattr(sub, 'toarray'): sub = sub.toarray()
    Xs_list.append(np.asarray(sub, dtype=np.float32))
a.file.close()
Xs_log = np.vstack(Xs_list)              # log1p(CP10K)
# Convert to counts-per-10k → re-log1p+CP10K → standardise with bulk gene stats.
# Since X is already log1p(CP10K) and per-cell row-sums(expm1) ≈ 1e4, we can
# just feed the log values directly through bulk normalisation.
Xs = Xs_log.astype(np.float32)
Xs = (Xs - gene_mu) / gene_sd
log(f'X scRNA: {Xs.shape}')

# ---------- 5. Train / held-out split ----------
held_mask = np.array([ln in HELD_OUT_LUAD for ln in common_lines])
log(f'Held-out present in expression: {held_mask.sum()} / {len(HELD_OUT_LUAD)}')
X_tr, X_te = X[~held_mask], X[held_mask]
Y_tr, Y_te = Y[~held_mask], Y[held_mask]
M_tr, M_te = M[~held_mask], M[held_mask]
log(f'Train lines={len(X_tr)}  Held-out LUAD lines={len(X_te)}  drugs={Y.shape[1]}')

# ---------- 6. Model ----------
class GRL(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.lam, None
def grl(x, lam): return GRL.apply(x, lam)

class Encoder(nn.Module):
    def __init__(self, d_in, d_hid, d_emb, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.BatchNorm1d(d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_emb), nn.BatchNorm1d(d_emb), nn.ReLU(), nn.Dropout(p))
    def forward(self, x): return self.net(x)
class DrugHead(nn.Module):
    def __init__(self, d_emb, n): super().__init__(); self.lin = nn.Linear(d_emb, n)
    def forward(self, z): return self.lin(z)
class DomainHead(nn.Module):
    def __init__(self, d_emb):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_emb,128), nn.ReLU(), nn.Linear(128,2))
    def forward(self, z): return self.net(z)

enc = Encoder(X.shape[1], HID_DIM, EMB_DIM).to(DEV)
dhead = DrugHead(EMB_DIM, Y.shape[1]).to(DEV)
dom  = DomainHead(EMB_DIM).to(DEV)
opt = torch.optim.Adam(list(enc.parameters())+list(dhead.parameters())+list(dom.parameters()),
                       lr=LR, weight_decay=1e-5)

Xt = torch.from_numpy(X_tr).to(DEV); Yt=torch.from_numpy(Y_tr).to(DEV); Mt=torch.from_numpy(M_tr).to(DEV)
Xs_t = torch.from_numpy(Xs).to(DEV)
Xte = torch.from_numpy(X_te).to(DEV); Yte=torch.from_numpy(Y_te); Mte=torch.from_numpy(M_te)

def sample_indices(n, k): return torch.randint(0, n, (k,), device=DEV)
metrics = []
for ep in range(EPOCHS):
    enc.train(); dhead.train(); dom.train()
    p = ep / max(EPOCHS-1,1)
    lam = LAMBDA_DOM_MAX * (2/(1+math.exp(-10*p)) - 1)
    iters = max(len(Xt)//BS_BULK, 1) * 4
    loss_r = loss_d = 0.0
    for it in range(iters):
        bi = sample_indices(len(Xt), BS_BULK); si = sample_indices(len(Xs_t), BS_SC)
        xb, yb, mb = Xt[bi], Yt[bi], Mt[bi]; xs = Xs_t[si]
        z_b = enc(xb); z_s = enc(xs)
        pred = dhead(z_b)
        reg_loss = (((pred - yb)**2) * mb).sum() / (mb.sum() + 1e-6)
        z_all = torch.cat([z_b, z_s], 0)
        dom_logits = dom(grl(z_all, lam))
        dom_label = torch.cat([torch.zeros(len(z_b), dtype=torch.long, device=DEV),
                               torch.ones(len(z_s),  dtype=torch.long, device=DEV)], 0)
        dom_loss = F.cross_entropy(dom_logits, dom_label)
        loss = reg_loss + lam * dom_loss
        opt.zero_grad(); loss.backward(); opt.step()
        loss_r += reg_loss.item(); loss_d += dom_loss.item()

    enc.eval(); dhead.eval()
    with torch.no_grad():
        pred_te = dhead(enc(Xte)).cpu().numpy()
    Y_te_np = Yte.numpy(); M_te_np = Mte.numpy()
    pers = []
    for d in range(Y.shape[1]):
        m_d = M_te_np[:, d].astype(bool)
        if m_d.sum() < 2: continue
        p_ = pred_te[m_d, d]; t_ = Y_te_np[m_d, d]
        if np.std(p_)==0 or np.std(t_)==0: continue
        pers.append(np.corrcoef(p_, t_)[0,1])
    mean_r = float(np.mean(pers)) if pers else float('nan')
    metrics.append({'epoch':ep+1,'reg_loss':loss_r/iters,'dom_loss':loss_d/iters,
                    'mean_drug_pearson_heldout':mean_r,'lambda':lam})
    log(f'ep {ep+1:02d}/{EPOCHS}  reg={loss_r/iters:.3f}  dom={loss_d/iters:.3f}  lam={lam:.3f}  r̄(held-out)={mean_r:.3f}')

# ---------- 7. Save ----------
torch.save({'enc':enc.state_dict(),'dhead':dhead.state_dict(),'dom':dom.state_dict(),
            'drugs': list(agg.index),
            'lines_train': [ln for ln,h in zip(common_lines,held_mask) if not h],
            'lines_heldout': [ln for ln,h in zip(common_lines,held_mask) if h],
            'gene_mu':gene_mu,'gene_sd':gene_sd,
            'drug_mean':agg_mean.values,'drug_std':agg_std.values,
            'genes':genes},
           CKPT/'t2b_scdeal_luad.pt')

pearson_rows = []
enc.eval(); dhead.eval()
with torch.no_grad():
    pred_te = dhead(enc(Xte)).cpu().numpy()
for d, dn in enumerate(agg.index):
    m_d = M_te[:, d].astype(bool)
    if m_d.sum() < 2:
        pearson_rows.append({'drug':dn,'n_heldout':int(m_d.sum()),'pearson':np.nan}); continue
    p_ = pred_te[m_d, d]; t_ = Y_te[m_d, d]
    if np.std(p_)==0 or np.std(t_)==0:
        pearson_rows.append({'drug':dn,'n_heldout':int(m_d.sum()),'pearson':np.nan}); continue
    pearson_rows.append({'drug':dn,'n_heldout':int(m_d.sum()),'pearson':float(np.corrcoef(p_,t_)[0,1])})
pd.DataFrame(pearson_rows).to_parquet(OUT/'drug_level_pearson.parquet', index=False)

log('Inference on 20k LUAD cells …')
preds_sc = []
BATCH = 2048
with torch.no_grad():
    for i in range(0, len(Xs_t), BATCH):
        z = enc(Xs_t[i:i+BATCH]); preds_sc.append(dhead(z).cpu().numpy())
preds_sc = np.vstack(preds_sc)
df_sc = pd.DataFrame(preds_sc, columns=agg.index)
df_sc.to_parquet(OUT/'scrna_drug_predictions_zscore_20k.parquet', index=False)
log(f'scRNA drug pred matrix: {df_sc.shape}  saved.')

final = metrics[-1]
summary = {
    'n_drugs': int(Y.shape[1]),
    'n_train_lines': int(len(X_tr)),
    'n_heldout_luad': int(len(X_te)),
    'heldout_lines': HELD_OUT_LUAD,
    'n_scrna_cells_sampled': int(len(Xs)),
    'final_reg_loss': final['reg_loss'],
    'final_dom_loss': final['dom_loss'],
    'mean_drug_pearson_heldout': final['mean_drug_pearson_heldout'],
    'per_epoch': metrics,
}
(OUT/'eval_metrics.json').write_text(json.dumps(summary, indent=2))
log(f'mean drug Pearson on held-out LUAD: {summary["mean_drug_pearson_heldout"]:.3f}')
log('== T2b-LUAD done ==')
