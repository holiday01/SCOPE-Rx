"""
T3a v2 — warm-start cell encoder from T2b scDEAL checkpoint (frozen or low-LR),
train MPNN drug encoder + joint head from scratch.

Rationale:
  * T2b cell encoder already learned a good 256-d cell representation under 1806
    drug-specific supervisory signals.
  * T3a v1 failed because the joint-single-head setup gave the cell encoder only
    weak gradient, collapsing to constant output.
  * By freezing cell_enc, gradient flows entirely into the MPNN + joint head,
    isolating whether drug structure helps predictions at all.

Evaluation (identical splits to v1):
  E1 line holdout — 3 HCC lines
  E2 drug holdout — 10% random drugs unseen during training
"""
from __future__ import annotations
import json, time, math, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool, global_mean_pool, GINEConv
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = Path('/home/holiday01/drug_sc')
PROC = ROOT/'data/processed/hcc_drug'
OUT  = ROOT/'results/t3a_v2'; OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT/'checkpoints'; CKPT.mkdir(parents=True, exist_ok=True)
DEV  = 'cuda'
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

HELD_OUT_HCC = ['ACH-000221','ACH-000480','ACH-000625']
MIN_OBS = 30
DRUG_HO = 0.10
D_HID_DRUG = 128
D_EMB_DRUG = 2*D_HID_DRUG
N_LAYERS = 3
EPOCHS = 30
BS = 1024
LR_DRUG = 2e-3
LR_CELL = 1e-4  # small nonzero so cell_enc can still adapt if helpful

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Load data ----------
expr_all = pd.read_parquet(PROC/'cellline_expression_panCancer.parquet')
long_tab = pd.read_parquet(PROC/'drug_response_long.parquet')
cat = pd.read_parquet(PROC/'drug_catalog.parquet').set_index('drug')
genes = pd.read_parquet(PROC/'gene_universe.parquet')['gene'].tolist()

agg = (long_tab.dropna(subset=['auc']).groupby(['drug','ModelID'])['auc'].mean().unstack('ModelID'))
drug_obs = agg.notna().sum(1)
keep = drug_obs[drug_obs>=MIN_OBS].index
keep = [d for d in keep if d in cat.index and isinstance(cat.loc[d,'smiles'],str) and cat.loc[d,'smiles']]
agg = agg.loc[keep]
common_lines = [c for c in expr_all.index if c in agg.columns]
agg = agg[common_lines]
agg_mean = agg.mean(1); agg_std = agg.std(1).replace(0,np.nan)
agg_z = agg.sub(agg_mean,axis=0).div(agg_std,axis=0)
Y = agg_z.T.values.astype(np.float32); M = ~np.isnan(Y); Y = np.nan_to_num(Y,0.0)
log(f'Drugs with ≥{MIN_OBS} obs & SMILES: {len(keep)}  Y {Y.shape}  obs={M.sum():,}')

# ---------- 2. Build drug graphs ----------
ATOM_LIST=['C','N','O','F','P','S','Cl','Br','I','H','B','Si','Se','As','K','Na','Mg','Ca','*']
HYBRID=[Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]
def atom_feat(a):
    sym = a.GetSymbol(); sym = sym if sym in ATOM_LIST else '*'
    return [*[int(sym==s) for s in ATOM_LIST],
            a.GetDegree()/6, a.GetTotalNumHs()/4, a.GetFormalCharge()/5,
            int(a.GetIsAromatic()), int(a.IsInRing()),
            *[int(a.GetHybridization()==h) for h in HYBRID]]
BOND_TYPES=[Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
def bond_feat(b):
    bt=b.GetBondType()
    return [*[int(bt==t) for t in BOND_TYPES], int(b.GetIsConjugated()), int(b.IsInRing())]
D_ATOM=len(ATOM_LIST)+5+len(HYBRID); D_BOND=len(BOND_TYPES)+2

def smi2data(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms()==0: return None
    x = torch.tensor([atom_feat(a) for a in mol.GetAtoms()], dtype=torch.float)
    src=[]; dst=[]; ea=[]
    for b in mol.GetBonds():
        i,j=b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf=bond_feat(b); src+=[i,j]; dst+=[j,i]; ea+=[bf,bf]
    if not src: src=[0]; dst=[0]; ea=[[0]*D_BOND]
    return Data(x=x, edge_index=torch.tensor([src,dst],dtype=torch.long),
                edge_attr=torch.tensor(ea,dtype=torch.float))

log('Building drug graphs …')
graphs=[]; keep2=[]
for d in agg.index:
    g = smi2data(cat.loc[d,'smiles'])
    if g is not None:
        graphs.append(g); keep2.append(d)
log(f'valid graphs: {len(graphs)} / {len(agg)}')
agg = agg.loc[keep2]
agg_mean = agg.mean(1); agg_std = agg.std(1).replace(0,np.nan)
agg_z = agg.sub(agg_mean,axis=0).div(agg_std,axis=0)
Y = agg_z.T.values.astype(np.float32); M = ~np.isnan(Y); Y=np.nan_to_num(Y,0.0)
n_line, n_drug = Y.shape
log(f'Final Y {Y.shape}  obs={M.sum():,}')

# ---------- 3. Warm-start cell encoder ----------
log('Loading T2b scDEAL cell encoder …')
t2b = torch.load(CKPT/'t2b_scdeal.pt', map_location='cpu', weights_only=False)

class CellEncoder(nn.Module):
    def __init__(self,d_in=17460,d_hid=1024,d_emb=256,p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in,d_hid), nn.BatchNorm1d(d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid,d_emb), nn.BatchNorm1d(d_emb), nn.ReLU(), nn.Dropout(p))
    def forward(self,x): return self.net(x)

# T2b enc module was called `Encoder` with .net sub-module — same architecture
cell_enc = CellEncoder(len(genes), 1024, 256).to(DEV)
sd = {k.replace('net.',''):v for k,v in t2b['enc'].items()}
# original uses .net, copy directly
cell_enc.net.load_state_dict({k.replace('net.',''):v for k,v in t2b['enc'].items() if k.startswith('net.')})
log('Cell encoder weights loaded from T2b')

# cell_enc warm-started; we keep params but assign a separate LR
g_mu = np.asarray(t2b['gene_mu']); g_sd = np.asarray(t2b['gene_sd'])

# ---------- 4. Expression tensors ----------
X = np.log1p(expr_all.loc[common_lines].values.astype(np.float32))
X = (X - g_mu) / g_sd
Xt = torch.from_numpy(X).to(DEV)

# ---------- 5. Splits ----------
held_line_mask = np.array([ln in HELD_OUT_HCC for ln in common_lines])
rng = np.random.default_rng(SEED)
ho_drug_idx = set(rng.choice(n_drug, size=int(n_drug*DRUG_HO), replace=False).tolist())
train_pairs=[]
for li in range(n_line):
    if held_line_mask[li]: continue
    for di in range(n_drug):
        if di in ho_drug_idx: continue
        if M[li,di]: train_pairs.append((li,di))
train_pairs = np.asarray(train_pairs, dtype=np.int64)
log(f'train pairs: {len(train_pairs):,}  held-drug={len(ho_drug_idx)}  held-lines={held_line_mask.sum()}')

# ---------- 6. Model ----------
class MPNN(nn.Module):
    def __init__(self,d_atom,d_bond,d_hid=128,n=3,p=0.1):
        super().__init__()
        self.ai = nn.Linear(d_atom,d_hid); self.ei = nn.Linear(d_bond,d_hid)
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(d_hid,d_hid*2),nn.ReLU(),nn.Linear(d_hid*2,d_hid)),
                     train_eps=True)
            for _ in range(n)])
        self.drop=nn.Dropout(p)
        self.proj=nn.Sequential(nn.Linear(2*d_hid,2*d_hid),nn.LayerNorm(2*d_hid))
    def forward(self,b):
        x=self.ai(b.x); e=self.ei(b.edge_attr)
        for c in self.convs:
            x = c(x,b.edge_index,e); x=F.relu(x); x=self.drop(x)
        g = torch.cat([global_mean_pool(x,b.batch), global_add_pool(x,b.batch)],-1)
        return self.proj(g)

class Head(nn.Module):
    def __init__(self,d_c,d_d,d=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_c+d_d, d), nn.LayerNorm(d), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d, d//2), nn.ReLU(),
            nn.Linear(d//2, 1))
    def forward(self,zc,zd): return self.net(torch.cat([zc,zd],-1)).squeeze(-1)

drug_enc = MPNN(D_ATOM, D_BOND, D_HID_DRUG, N_LAYERS).to(DEV)
head = Head(256, D_EMB_DRUG).to(DEV)

log(f'params: cell={sum(p.numel() for p in cell_enc.parameters())/1e6:.2f}M  drug={sum(p.numel() for p in drug_enc.parameters())/1e6:.2f}M  head={sum(p.numel() for p in head.parameters())/1e6:.2f}M')

opt = torch.optim.AdamW([
    {'params': cell_enc.parameters(), 'lr': LR_CELL},
    {'params': drug_enc.parameters(), 'lr': LR_DRUG},
    {'params': head.parameters(),     'lr': LR_DRUG},
], weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

Yt = torch.from_numpy(Y).to(DEV)

def build_batch(ids):
    return Batch.from_data_list([graphs[i] for i in ids]).to(DEV)

from scipy.stats import spearmanr
def eval_lines():
    cell_enc.eval(); drug_enc.eval(); head.eval()
    held_ids = [i for i,h in enumerate(held_line_mask) if h]
    with torch.no_grad():
        zc = cell_enc(Xt[torch.tensor(held_ids,device=DEV)])
        preds = np.zeros((len(held_ids), n_drug), dtype=np.float32)
        for s in range(0, n_drug, 512):
            ids=list(range(s,min(s+512,n_drug)))
            zd = drug_enc(build_batch(ids))
            zc_rep = zc.unsqueeze(1).expand(-1,len(ids),-1).reshape(-1,zc.shape[1])
            zd_rep = zd.unsqueeze(0).expand(len(held_ids),-1,-1).reshape(-1,zd.shape[1])
            p = head(zc_rep,zd_rep).view(len(held_ids),len(ids)).cpu().numpy()
            preds[:,s:s+len(ids)] = p
    rows=[]
    for k,li in enumerate(held_ids):
        m = M[li]
        if m.sum()<20: continue
        s,_ = spearmanr(preds[k,m], Y[li,m])
        rows.append({'line':common_lines[li],'n':int(m.sum()),'spearman':float(s)})
    return rows, preds

def eval_drugs():
    cell_enc.eval(); drug_enc.eval(); head.eval()
    ho = sorted(ho_drug_idx)
    non_held = [i for i,h in enumerate(held_line_mask) if not h]
    with torch.no_grad():
        zc = cell_enc(Xt[torch.tensor(non_held,device=DEV)])
        preds = np.zeros((len(non_held), len(ho)), dtype=np.float32)
        for s in range(0,len(ho),512):
            ids=ho[s:s+512]
            zd = drug_enc(build_batch(ids))
            zc_rep = zc.unsqueeze(1).expand(-1,len(ids),-1).reshape(-1,zc.shape[1])
            zd_rep = zd.unsqueeze(0).expand(len(non_held),-1,-1).reshape(-1,zd.shape[1])
            p = head(zc_rep,zd_rep).view(len(non_held),len(ids)).cpu().numpy()
            preds[:,s:s+len(ids)] = p
    rows=[]
    for k,di in enumerate(ho):
        m = M[non_held,di]
        if m.sum()<30: continue
        y = Y[non_held,di][m]; p = preds[m,k]
        if y.std()==0 or p.std()==0: continue
        s,_ = spearmanr(p,y)
        rows.append({'drug':keep2[di],'n':int(m.sum()),'spearman':float(s)})
    return rows, preds

# ---------- 7. Train ----------
for ep in range(EPOCHS):
    cell_enc.train(); drug_enc.train(); head.train()
    perm = np.random.permutation(len(train_pairs))
    tot=0.0; cnt=0
    for s in range(0,len(perm),BS):
        idx = train_pairs[perm[s:s+BS]]
        li = torch.from_numpy(idx[:,0]).long().to(DEV)
        di = idx[:,1].tolist()
        zc = cell_enc(Xt[li])
        zd = drug_enc(build_batch(di))
        pr = head(zc, zd)
        tgt = Yt[li, torch.tensor(di,device=DEV,dtype=torch.long)]
        loss = F.mse_loss(pr, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*len(idx); cnt += len(idx)
    sched.step()
    rows, _ = eval_lines()
    mean_sp = float(np.mean([r['spearman'] for r in rows])) if rows else float('nan')
    detail = ', '.join('{}:{:.2f}'.format(r['line'], r['spearman']) for r in rows)
    log(f'ep {ep+1:02d}/{EPOCHS}  mse={tot/cnt:.4f}  line_sp̄={mean_sp:.3f}  [{detail}]')

# ---------- 8. Final eval ----------
line_rows,_ = eval_lines()
drug_rows,_ = eval_drugs()
pd.DataFrame(line_rows).to_parquet(OUT/'per_line_rank_eval.parquet', index=False)
pd.DataFrame(drug_rows).to_parquet(OUT/'drug_holdout_rank_eval.parquet', index=False)

log('\n=== E1 line holdout ===')
for r in line_rows: log(f'  {r["line"]:12s} n={r["n"]:4d}  sp={r["spearman"]:+.3f}')
log(f'  mean = {np.mean([r["spearman"] for r in line_rows]):.3f}')
if drug_rows:
    sps=[r["spearman"] for r in drug_rows]
    log(f'\n=== E2 drug holdout ({len(drug_rows)} drugs) ===')
    log(f'  mean Spearman = {np.mean(sps):+.3f}')
    log(f'  median Spearman = {np.median(sps):+.3f}')
    log(f'  frac>0.3 = {(np.array(sps)>0.3).mean():.2f}')
    log(f'  frac>0.5 = {(np.array(sps)>0.5).mean():.2f}')

torch.save({'cell_enc':cell_enc.state_dict(),'drug_enc':drug_enc.state_dict(),'head':head.state_dict(),
            'drugs':keep2,'lines':common_lines,'g_mu':g_mu,'g_sd':g_sd,
            'drug_mean':agg_mean.values,'drug_std':agg_std.values,
            'D_ATOM':D_ATOM,'D_BOND':D_BOND,
            'ho_drug_idx':list(ho_drug_idx),'held_line_mask':held_line_mask.tolist()},
           CKPT/'t3a_v2_mpnn_joint.pt')

(OUT/'eval_metrics.json').write_text(json.dumps({
    'E1_line_holdout': line_rows,
    'E1_mean_spearman': float(np.mean([r['spearman'] for r in line_rows])),
    'E2_drug_holdout_mean': float(np.mean([r['spearman'] for r in drug_rows])) if drug_rows else None,
    'E2_drug_holdout_median': float(np.median([r['spearman'] for r in drug_rows])) if drug_rows else None,
    'E2_n': len(drug_rows),
    'n_drugs': n_drug, 'n_lines': n_line, 'n_train_pairs': int(len(train_pairs)),
}, indent=2))
log('== T3a v2 done ==')
