"""
T2b diagnostic — with 3 held-out lines, per-drug Pearson is noise.
Better: per-line Spearman across drugs (drug-ranking quality for each HCC line)
plus sanity check on clinical HCC drugs.
"""
import json, numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

ROOT = Path('/home/holiday01/drug_sc')
PROC = ROOT/'data/processed/hcc_drug'
OUT  = ROOT/'results/t2b'

ck = torch.load(ROOT/'checkpoints/t2b_scdeal.pt', map_location='cpu', weights_only=False)
drugs  = list(ck['drugs'])
genes  = list(ck['genes'])
lines_held = ck['lines_heldout']
drug_mean = np.array(ck['drug_mean']); drug_std = np.array(ck['drug_std'])
gene_mu = np.array(ck['gene_mu']); gene_sd = np.array(ck['gene_sd'])

# Reload model
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self,d_in,d_hid,d_emb,p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in,d_hid), nn.BatchNorm1d(d_hid), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(d_hid,d_emb), nn.BatchNorm1d(d_emb), nn.ReLU(), nn.Dropout(p))
    def forward(self,x): return self.net(x)
class DrugHead(nn.Module):
    def __init__(self,d_emb,n): super().__init__(); self.lin=nn.Linear(d_emb,n)
    def forward(self,z): return self.lin(z)

enc = Encoder(len(genes),1024,256); enc.load_state_dict(ck['enc']); enc.eval()
dhead = DrugHead(256, len(drugs)); dhead.load_state_dict(ck['dhead']); dhead.eval()

# Load bulk expression for held-out
expr_all = pd.read_parquet(PROC/'cellline_expression_panCancer.parquet')
X_he = expr_all.loc[lines_held].values.astype(np.float32)
X_he = np.log1p(X_he)
X_he = (X_he - gene_mu) / gene_sd
with torch.no_grad():
    pred_z = dhead(enc(torch.from_numpy(X_he).float())).numpy()   # (3, D) z-scored
# de-z to raw AUC scale
pred_raw = pred_z * drug_std[None,:] + drug_mean[None,:]

# True held-out AUC (from T2a long table)
long_tab = pd.read_parquet(PROC/'drug_response_long.parquet')
agg = (long_tab.dropna(subset=['auc']).groupby(['drug','ModelID'])['auc'].mean().unstack('ModelID'))
agg = agg.loc[drugs, lines_held]
Y_raw = agg.T.values.astype(np.float32)    # (3, D)
M = ~np.isnan(Y_raw)

# --- per-line across-drug rank correlation ---
lines_meta = pd.read_parquet(PROC/'cellline_meta.parquet').set_index('ModelID')
rows = []
for i, ln in enumerate(lines_held):
    m = M[i]
    if m.sum() < 20:
        rows.append({'line':ln,'cell':'','n':int(m.sum()),'spearman':np.nan,'pearson':np.nan}); continue
    sp = spearmanr(pred_raw[i, m], Y_raw[i, m])[0]
    pr = pearsonr(pred_raw[i, m], Y_raw[i, m])[0]
    rows.append({'line':ln,'cell':lines_meta.loc[ln,'cell_line'],'n':int(m.sum()),
                 'spearman':float(sp),'pearson':float(pr)})
perline = pd.DataFrame(rows)
print('=== Per-held-out-line drug-ranking performance ===')
print(perline.to_string(index=False))
perline.to_parquet(OUT/'per_line_rank_eval.parquet', index=False)

# --- Sanity: clinical HCC drugs ---
clinical = ['sorafenib','lenvatinib','regorafenib','cabozantinib','erlotinib','sunitinib',
            '5-fluorouracil','fluorouracil','cisplatin','oxaliplatin','doxorubicin','gemcitabine','paclitaxel']
drug_idx = {d.lower():i for i,d in enumerate(drugs)}
print('\n=== Clinical HCC drugs — predicted vs true AUC on held-out lines ===')
print(f'{"drug":18s} {"line":14s} {"true_auc":>9s} {"pred_auc":>9s}')
for dn in clinical:
    idx = drug_idx.get(dn)
    if idx is None: continue
    for i, ln in enumerate(lines_held):
        if M[i, idx]:
            print(f'{dn:18s} {lines_meta.loc[ln,"cell_line"]:14s} {Y_raw[i,idx]:9.3f} {pred_raw[i,idx]:9.3f}')

# --- sampled drugs with most extreme predicted kill on HCC scRNA (just peek) ---
pred_sc = pd.read_parquet(OUT/'scrna_drug_predictions_zscore_20k.parquet')
# mean over 20k cells per drug (z-score); more negative z = lower AUC = more kill
mean_z = pred_sc.mean(0).sort_values()
print('\n=== Top 10 predicted most-killing drugs across 20k HCC scRNA cells (low AUC z) ===')
print(mean_z.head(10).to_string())
print('\n=== Bottom 10 (least killing) ===')
print(mean_z.tail(10).to_string())
