"""
T3a — MPNN drug encoder + gene-MLP cell encoder, joint AUC regression.

Single shared head (no per-drug head) -> enables unseen-drug extrapolation.

Eval:
  E1 "line holdout" — same 3 held-out HCC lines as T2b (direct comparison to scDEAL)
  E2 "drug holdout" — 10% drugs randomly removed from training (MPNN-specific strength)

Outputs:
  checkpoints/t3a_mpnn_joint.pt
  results/t3a/eval_metrics.json
  results/t3a/per_line_rank_eval.parquet
  results/t3a/drug_holdout_rank_eval.parquet
"""
from __future__ import annotations
import json, time, math, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool, global_mean_pool, GINEConv
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = Path('/home/holiday01/drug_sc')
PROC = ROOT/'data/processed/hcc_drug'
OUT  = ROOT/'results/t3a'; OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT/'checkpoints'; CKPT.mkdir(parents=True, exist_ok=True)
DEV  = 'cuda'
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

HELD_OUT_HCC = ['ACH-000221','ACH-000480','ACH-000625']
MIN_OBS_PER_DRUG = 30
DRUG_HOLDOUT_FRAC = 0.10
D_HID_CELL = 1024
D_EMB_CELL = 256
D_HID_DRUG = 128
D_EMB_DRUG = 256            # after concat(mean,sum) readout => 2*D_HID_DRUG
N_GNN_LAYERS = 3
N_EPOCHS = 30
BS = 512
LR = 1e-3

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Load tables ----------
expr_all = pd.read_parquet(PROC/'cellline_expression_panCancer.parquet')
long_tab = pd.read_parquet(PROC/'drug_response_long.parquet')
cat = pd.read_parquet(PROC/'drug_catalog.parquet').set_index('drug')
genes = pd.read_parquet(PROC/'gene_universe.parquet')['gene'].tolist()

# keep drugs with ≥30 obs AND valid SMILES
agg = (long_tab.dropna(subset=['auc']).groupby(['drug','ModelID'])['auc'].mean().unstack('ModelID'))
drug_obs = agg.notna().sum(1)
keep = drug_obs[drug_obs>=MIN_OBS_PER_DRUG].index
keep = [d for d in keep if d in cat.index and isinstance(cat.loc[d,'smiles'],str) and cat.loc[d,'smiles']]
agg = agg.loc[keep]
log(f'Drugs with ≥{MIN_OBS_PER_DRUG} obs AND valid SMILES: {len(keep)}')

# align cell lines with expression
common_lines = [c for c in expr_all.index if c in agg.columns]
agg = agg[common_lines]

# per-drug z-score
agg_mean = agg.mean(1); agg_std = agg.std(1).replace(0,np.nan)
agg_z = agg.sub(agg_mean,axis=0).div(agg_std,axis=0)
Y = agg_z.T.values.astype(np.float32)  # (n_line × n_drug)
M = ~np.isnan(Y)
Y = np.nan_to_num(Y,nan=0.0)
log(f'Y {Y.shape}  observed={M.sum():,}')

# ---------- 2. Cell expression tensors ----------
X = np.log1p(expr_all.loc[common_lines].values.astype(np.float32))
g_mu = X.mean(0); g_sd = X.std(0) + 1e-6
X = (X - g_mu) / g_sd
log(f'X {X.shape}')

# ---------- 3. SMILES -> graph ----------
ATOM_LIST=['C','N','O','F','P','S','Cl','Br','I','H','B','Si','Se','As','K','Na','Mg','Ca','*']
HYBRID=[Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]
def atom_feat(a):
    sym = a.GetSymbol(); sym = sym if sym in ATOM_LIST else '*'
    return [
        *[int(sym==s) for s in ATOM_LIST],
        a.GetDegree()/6, a.GetTotalNumHs()/4, a.GetFormalCharge()/5,
        int(a.GetIsAromatic()), int(a.IsInRing()),
        *[int(a.GetHybridization()==h) for h in HYBRID],
    ]
BOND_TYPES=[Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
def bond_feat(b):
    bt = b.GetBondType()
    return [*[int(bt==t) for t in BOND_TYPES], int(b.GetIsConjugated()), int(b.IsInRing())]

D_ATOM = len(ATOM_LIST) + 5 + len(HYBRID)
D_BOND = len(BOND_TYPES) + 2

def smiles_to_data(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0: return None
    x = torch.tensor([atom_feat(a) for a in mol.GetAtoms()], dtype=torch.float)
    src=[]; dst=[]; ea=[]
    for b in mol.GetBonds():
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_feat(b)
        src += [i,j]; dst += [j,i]; ea += [bf, bf]
    if not src:  # single-atom molecule (salts)
        src=[0]; dst=[0]; ea=[[0]*D_BOND]
    return Data(x=x, edge_index=torch.tensor([src,dst],dtype=torch.long),
                edge_attr=torch.tensor(ea,dtype=torch.float))

log('Building drug graphs …')
drug_graphs = {}
for d in agg.index:
    g = smiles_to_data(cat.loc[d,'smiles'])
    if g is not None: drug_graphs[d] = g
log(f'Valid graphs: {len(drug_graphs)} / {len(agg)}')
# restrict agg to drugs with valid graph
agg = agg.loc[list(drug_graphs)]
agg_mean = agg.mean(1); agg_std = agg.std(1).replace(0,np.nan)
agg_z = agg.sub(agg_mean,axis=0).div(agg_std,axis=0)
Y = agg_z.T.values.astype(np.float32); M = ~np.isnan(Y); Y = np.nan_to_num(Y,0.0)
n_drug = Y.shape[1]; n_line = Y.shape[0]
log(f'Final pairs: {M.sum():,}  lines={n_line}  drugs={n_drug}')

# ---------- 4. Splits ----------
held_line_mask = np.array([ln in HELD_OUT_HCC for ln in common_lines])
drugs_all = list(agg.index)
rng = np.random.default_rng(SEED)
ho_drug_idx = set(rng.choice(n_drug, size=int(n_drug*DRUG_HOLDOUT_FRAC), replace=False).tolist())
log(f'Drug holdout: {len(ho_drug_idx)} drugs')

# For training, we use observations NOT in either holdout
train_pairs = []  # (line_idx, drug_idx)
for li in range(n_line):
    if held_line_mask[li]: continue
    for di in range(n_drug):
        if di in ho_drug_idx: continue
        if M[li, di]: train_pairs.append((li, di))
log(f'Train pairs: {len(train_pairs):,}')
train_pairs = np.array(train_pairs, dtype=np.int64)

# ---------- 5. Model ----------
class CellEncoder(nn.Module):
    def __init__(self, d_in, d_hid, d_emb, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in,d_hid), nn.BatchNorm1d(d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid,d_emb), nn.BatchNorm1d(d_emb), nn.ReLU(), nn.Dropout(p))
    def forward(self,x): return self.net(x)

class MPNNEncoder(nn.Module):
    def __init__(self, d_atom, d_bond, d_hid=128, n_layers=3, p=0.1):
        super().__init__()
        self.atom_in = nn.Linear(d_atom, d_hid)
        self.edge_in = nn.Linear(d_bond, d_hid)
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(d_hid, d_hid*2), nn.ReLU(),
                                    nn.Linear(d_hid*2, d_hid)))
            for _ in range(n_layers)])
        self.drop = nn.Dropout(p)
        self.proj = nn.Linear(2*d_hid, 2*d_hid)
    def forward(self, batch):
        x = self.atom_in(batch.x)
        e = self.edge_in(batch.edge_attr)
        for conv in self.convs:
            x = conv(x, batch.edge_index, e)
            x = F.relu(x); x = self.drop(x)
        g = torch.cat([global_mean_pool(x,batch.batch), global_add_pool(x,batch.batch)], dim=-1)
        return self.proj(g)

class JointHead(nn.Module):
    def __init__(self, d_cell, d_drug, d_hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_cell+d_drug, d_hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_hid, d_hid//2), nn.ReLU(),
            nn.Linear(d_hid//2, 1))
    def forward(self, zc, zd): return self.net(torch.cat([zc,zd],-1)).squeeze(-1)

cell_enc = CellEncoder(X.shape[1], D_HID_CELL, D_EMB_CELL).to(DEV)
drug_enc = MPNNEncoder(D_ATOM, D_BOND, D_HID_DRUG, N_GNN_LAYERS).to(DEV)
head     = JointHead(D_EMB_CELL, 2*D_HID_DRUG).to(DEV)
params = list(cell_enc.parameters())+list(drug_enc.parameters())+list(head.parameters())
opt = torch.optim.Adam(params, lr=LR, weight_decay=1e-5)
log(f'Params: cell={sum(p.numel() for p in cell_enc.parameters())/1e6:.2f}M  drug={sum(p.numel() for p in drug_enc.parameters())/1e6:.2f}M  head={sum(p.numel() for p in head.parameters())/1e6:.2f}M')

# Move data to GPU
Xt = torch.from_numpy(X).to(DEV)
Yt = torch.from_numpy(Y).to(DEV)

# Pre-move drug graphs and cache them on GPU
drug_idx_list = drugs_all
drug_data_list = [drug_graphs[d] for d in drug_idx_list]
# Build a ragged list; we batch by gather at train time
def gather_drug_batch(indices):
    batch = Batch.from_data_list([drug_data_list[i] for i in indices]).to(DEV)
    return batch

# ---------- 6. Train ----------
from scipy.stats import spearmanr
def evaluate_line_holdout():
    cell_enc.eval(); drug_enc.eval(); head.eval()
    with torch.no_grad():
        zc_all = cell_enc(Xt[torch.tensor([i for i,h in enumerate(held_line_mask) if h], device=DEV)])
        # batch drugs
        preds = np.zeros((zc_all.shape[0], n_drug), dtype=np.float32)
        BSZ = 256
        for s in range(0, n_drug, BSZ):
            ids = list(range(s, min(s+BSZ, n_drug)))
            zd = drug_enc(gather_drug_batch(ids))
            # broadcast (H × D) pairs
            zc_rep = zc_all.unsqueeze(1).expand(-1,len(ids),-1).reshape(-1,zc_all.shape[1])
            zd_rep = zd.unsqueeze(0).expand(zc_all.shape[0],-1,-1).reshape(-1,zd.shape[1])
            p = head(zc_rep, zd_rep).view(zc_all.shape[0], len(ids)).cpu().numpy()
            preds[:, s:s+len(ids)] = p
    held_ids = [i for i,h in enumerate(held_line_mask) if h]
    rows=[]
    for k, li in enumerate(held_ids):
        m = M[li]
        if m.sum()<20: continue
        s,_ = spearmanr(preds[k,m], Y[li,m])
        rows.append({'line':common_lines[li], 'n':int(m.sum()), 'spearman':float(s)})
    return rows, preds

def evaluate_drug_holdout():
    """Predict AUC for held-out drugs on all non-held-out lines, compute per-drug Spearman over lines."""
    cell_enc.eval(); drug_enc.eval(); head.eval()
    ho_drug_ids = sorted(ho_drug_idx)
    non_held_lines = [i for i,h in enumerate(held_line_mask) if not h]
    with torch.no_grad():
        zc = cell_enc(Xt[torch.tensor(non_held_lines, device=DEV)])
        preds = np.zeros((len(non_held_lines), len(ho_drug_ids)), dtype=np.float32)
        BSZ=256
        for s in range(0, len(ho_drug_ids), BSZ):
            ids = ho_drug_ids[s:s+BSZ]
            zd = drug_enc(gather_drug_batch(ids))
            zc_rep = zc.unsqueeze(1).expand(-1,len(ids),-1).reshape(-1,zc.shape[1])
            zd_rep = zd.unsqueeze(0).expand(zc.shape[0],-1,-1).reshape(-1,zd.shape[1])
            p = head(zc_rep, zd_rep).view(zc.shape[0], len(ids)).cpu().numpy()
            preds[:, s:s+len(ids)] = p
    rows=[]
    for k, di in enumerate(ho_drug_ids):
        m = M[non_held_lines, di]
        if m.sum()<30: continue
        y = Y[non_held_lines, di][m]
        p = preds[m, k]
        if np.std(y)==0 or np.std(p)==0: continue
        s,_ = spearmanr(p, y)
        rows.append({'drug':drugs_all[di], 'n':int(m.sum()), 'spearman':float(s)})
    return rows, preds

for ep in range(N_EPOCHS):
    cell_enc.train(); drug_enc.train(); head.train()
    perm = np.random.permutation(len(train_pairs))
    tot=0.0; cnt=0
    for s in range(0, len(perm), BS):
        idx = train_pairs[perm[s:s+BS]]
        li = torch.from_numpy(idx[:,0]).long().to(DEV)
        di = idx[:,1]
        zc = cell_enc(Xt[li])
        zd = drug_enc(gather_drug_batch(di.tolist()))
        pr = head(zc, zd)
        tgt = Yt[li, torch.from_numpy(di).long().to(DEV)]
        loss = F.mse_loss(pr, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*len(idx); cnt += len(idx)
    line_rows, _ = evaluate_line_holdout()
    mean_sp = float(np.mean([r['spearman'] for r in line_rows])) if line_rows else float('nan')
    detail = ', '.join('{}:{:.2f}'.format(r['line'], r['spearman']) for r in line_rows)
    log(f'ep {ep+1:02d}/{N_EPOCHS}  mse={tot/cnt:.4f}  mean_line_sp={mean_sp:.3f}  [{detail}]')

# ---------- 7. Final eval ----------
line_rows, line_preds = evaluate_line_holdout()
drug_rows, drug_preds = evaluate_drug_holdout()
pd.DataFrame(line_rows).to_parquet(OUT/'per_line_rank_eval.parquet', index=False)
pd.DataFrame(drug_rows).to_parquet(OUT/'drug_holdout_rank_eval.parquet', index=False)

log('\n=== E1 line holdout (3 HCC lines) ===')
for r in line_rows: log(f'  {r["line"]:12s} n={r["n"]:4d}  sp={r["spearman"]:+.3f}')
log(f'  mean = {np.mean([r["spearman"] for r in line_rows]):.3f}')
log('\n=== E2 drug holdout (10% unseen drugs) ===')
if drug_rows:
    sps=[r["spearman"] for r in drug_rows]
    log(f'  n drugs eval = {len(drug_rows)}')
    log(f'  mean Spearman = {np.mean(sps):+.3f}  median = {np.median(sps):+.3f}  frac>0.3 = {(np.array(sps)>0.3).mean():.2f}')

# Save checkpoint
torch.save({'cell_enc':cell_enc.state_dict(),'drug_enc':drug_enc.state_dict(),'head':head.state_dict(),
            'drugs':drugs_all,'lines':common_lines,'g_mu':g_mu,'g_sd':g_sd,
            'drug_mean':agg_mean.values,'drug_std':agg_std.values,
            'held_line_mask':held_line_mask.tolist(),'ho_drug_idx':list(ho_drug_idx),
            'D_ATOM':D_ATOM,'D_BOND':D_BOND},
           CKPT/'t3a_mpnn_joint.pt')

summary = {
    'baseline':'T3a MPNN joint',
    'n_drugs':n_drug, 'n_lines':n_line, 'n_train_pairs':int(len(train_pairs)),
    'E1_line_holdout': line_rows,
    'E1_mean_spearman': float(np.mean([r['spearman'] for r in line_rows])),
    'E2_drug_holdout_mean_spearman': float(np.mean([r['spearman'] for r in drug_rows])) if drug_rows else None,
    'E2_drug_holdout_median_spearman': float(np.median([r['spearman'] for r in drug_rows])) if drug_rows else None,
    'E2_n_drugs_evaluated': len(drug_rows),
}
(OUT/'eval_metrics.json').write_text(json.dumps(summary, indent=2))
log('== T3a done ==')
