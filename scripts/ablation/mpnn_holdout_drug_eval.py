"""
MPNN held-out drug evaluation — 5-fold drug-stratified CV.

Trains MPNN(SMILES) + cell-line encoder + AUC head on pan-cancer DepMap
(~1684 cell lines × ~1500 drugs). Each fold holds out ~20% of drugs;
predicts their AUC across cell lines from SMILES alone.

Baselines:
  - ECFP4 (Morgan FP) + Ridge on (cell_line × ECFP4) features per drug
  - Random 256-d drug embedding (architecture preserved, MPNN replaced)
  - Cell-line mean: predict per-cell-line mean from training drugs

Outputs:
  results/ablation_mpnn_holdout/
    cv_results.parquet           per-(fold,drug) Pearson/Spearman
    baseline_results.parquet     same for baselines
    summary.json
    comparison.md
"""
from __future__ import annotations
import json, time, math, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool, global_mean_pool, GINEConv
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge

RDLogger.DisableLog('rdApp.*')

ROOT = Path('/home/holiday01/drug_sc')
PROC = ROOT/'data/processed/hcc_drug'    # use LIHC's gene universe (matches t2b_scdeal.pt warm-start)
OUT  = ROOT/'results/ablation_mpnn_holdout'; OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT/'checkpoints'
DEV  = 'cuda'
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

K_FOLDS = 5
MIN_OBS = 30
D_HID_DRUG = 128
D_EMB_DRUG = 2*D_HID_DRUG  # 256
N_LAYERS = 3
EPOCHS = 18
BS = 1024
LR_DRUG = 2e-3
LR_CELL = 1e-4

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Load data ----------
log('Loading data …')
expr_all = pd.read_parquet(PROC/'cellline_expression_panCancer.parquet')
long_tab = pd.read_parquet(PROC/'drug_response_long.parquet')
cat = pd.read_parquet(PROC/'drug_catalog.parquet').set_index('drug')
genes = pd.read_parquet(PROC/'gene_universe.parquet')['gene'].tolist()

agg = (long_tab.dropna(subset=['auc']).groupby(['drug','ModelID'])['auc']
       .mean().unstack('ModelID'))
drug_obs = agg.notna().sum(1)
keep = drug_obs[drug_obs>=MIN_OBS].index
keep = [d for d in keep if d in cat.index and isinstance(cat.loc[d,'smiles'],str) and cat.loc[d,'smiles']]
agg = agg.loc[keep]
common_lines = [c for c in expr_all.index if c in agg.columns]
agg = agg[common_lines]
agg_mean = agg.mean(1); agg_std = agg.std(1).replace(0,np.nan)
agg_z = agg.sub(agg_mean, axis=0).div(agg_std, axis=0)
Y = agg_z.T.values.astype(np.float32); M = ~np.isnan(Y); Y = np.nan_to_num(Y, 0.0)
log(f'Drugs with ≥{MIN_OBS} obs & SMILES: {len(keep)}  Y {Y.shape}  obs={M.sum():,}')

# ---------- 2. Drug graphs ----------
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
    if mol is None or mol.GetNumAtoms()==0: return None, None
    x = torch.tensor([atom_feat(a) for a in mol.GetAtoms()], dtype=torch.float)
    src=[]; dst=[]; ea=[]
    for b in mol.GetBonds():
        i,j=b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf=bond_feat(b); src+=[i,j]; dst+=[j,i]; ea+=[bf,bf]
    if not src: src=[0]; dst=[0]; ea=[[0]*D_BOND]
    g = Data(x=x, edge_index=torch.tensor([src,dst],dtype=torch.long),
             edge_attr=torch.tensor(ea,dtype=torch.float))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp = np.frombuffer(bytes(fp.ToBitString(),'ascii'), dtype=np.uint8) - ord('0')
    return g, fp.astype(np.float32)

log('Building drug graphs + ECFP4 fingerprints …')
graphs=[]; fps=[]; keep2=[]
for d in agg.index:
    g, fp = smi2data(cat.loc[d,'smiles'])
    if g is not None:
        graphs.append(g); fps.append(fp); keep2.append(d)
fps = np.stack(fps)
log(f'valid: {len(graphs)} / {len(agg)}  ECFP4: {fps.shape}')
agg = agg.loc[keep2]
agg_mean = agg.mean(1); agg_std = agg.std(1).replace(0,np.nan)
agg_z = agg.sub(agg_mean, axis=0).div(agg_std, axis=0)
Y = agg_z.T.values.astype(np.float32); M = ~np.isnan(Y); Y = np.nan_to_num(Y, 0.0)
n_line, n_drug = Y.shape
log(f'Final Y {Y.shape}')

# ---------- 3. Cell-line tensors ----------
X = np.log1p(expr_all.loc[common_lines].values.astype(np.float32))
g_mu = X.mean(0); g_sd = X.std(0)+1e-6
X = (X - g_mu) / g_sd
Xt = torch.from_numpy(X).to(DEV)
log(f'Cell-line tensor: {X.shape}')

# ---------- 4. Models ----------
class CellEncoder(nn.Module):
    def __init__(self,d_in,d_hid=1024,d_emb=256,p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in,d_hid), nn.BatchNorm1d(d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid,d_emb), nn.BatchNorm1d(d_emb), nn.ReLU(), nn.Dropout(p))
    def forward(self,x): return self.net(x)

class MPNN(nn.Module):
    def __init__(self,d_atom,d_bond,d_hid=128,n=3,p=0.1):
        super().__init__()
        self.ai=nn.Linear(d_atom,d_hid); self.ei=nn.Linear(d_bond,d_hid)
        self.convs=nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(d_hid,d_hid*2),nn.ReLU(),nn.Linear(d_hid*2,d_hid)), train_eps=True)
            for _ in range(n)])
        self.drop=nn.Dropout(p)
        self.proj=nn.Sequential(nn.Linear(2*d_hid,2*d_hid), nn.LayerNorm(2*d_hid))
    def forward(self,b):
        x=self.ai(b.x); e=self.ei(b.edge_attr)
        for c in self.convs:
            x = c(x, b.edge_index, e); x = F.relu(x); x = self.drop(x)
        g = torch.cat([global_mean_pool(x,b.batch), global_add_pool(x,b.batch)], -1)
        return self.proj(g)

class Head(nn.Module):
    def __init__(self,d_c,d_d,d=256):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d_c+d_d, d), nn.LayerNorm(d), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d, d//2), nn.ReLU(),
            nn.Linear(d//2, 1))
    def forward(self,zc,zd): return self.net(torch.cat([zc,zd],-1)).squeeze(-1)

def build_batch(ids):
    return Batch.from_data_list([graphs[i] for i in ids]).to(DEV)

# ---------- 5. K-fold drug split ----------
rng = np.random.default_rng(SEED)
order = rng.permutation(n_drug)
folds = np.array_split(order, K_FOLDS)
log(f'\nK-fold drug split: {[len(f) for f in folds]}')

# Collect fold results
all_rows = []
all_top50_overlap = []
fold_details = {}

for fi, ho in enumerate(folds):
    fold_t0 = time.time()
    ho_set = set(ho.tolist())
    log(f'\n----- Fold {fi+1}/{K_FOLDS}  held-out drugs: {len(ho)} -----')

    # Training pairs (all cell lines × non-held drugs with observation)
    train_mask = np.ones(n_drug, dtype=bool); train_mask[list(ho_set)] = False
    train_pairs = []
    for li in range(n_line):
        for di in np.where(M[li])[0]:
            if train_mask[di]:
                train_pairs.append((li, di))
    train_pairs = np.asarray(train_pairs, dtype=np.int64)
    log(f'  train pairs: {len(train_pairs):,}')

    # Models
    cell_enc = CellEncoder(len(genes)).to(DEV)
    drug_enc = MPNN(D_ATOM, D_BOND, D_HID_DRUG, N_LAYERS).to(DEV)
    head = Head(256, D_EMB_DRUG).to(DEV)
    # warm-start cell_enc from T2b (pan-cancer)
    try:
        t2b = torch.load(CKPT/'t2b_scdeal.pt', map_location='cpu', weights_only=False)
        sd = {k:v for k,v in t2b['enc'].items() if k.startswith('net.')}
        cell_enc.net.load_state_dict({k.replace('net.',''):v for k,v in sd.items()})
        log('  cell_enc warm-started from T2b')
    except Exception as e:
        log(f'  warm-start skipped: {e}')

    opt = torch.optim.AdamW([
        {'params':cell_enc.parameters(),'lr':LR_CELL},
        {'params':drug_enc.parameters(),'lr':LR_DRUG},
        {'params':head.parameters(),'lr':LR_DRUG}], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    Yt = torch.from_numpy(Y).to(DEV)

    for ep in range(EPOCHS):
        cell_enc.train(); drug_enc.train(); head.train()
        perm = rng.permutation(len(train_pairs))
        loss_sum = 0; n_seen = 0
        for s in range(0, len(perm), BS):
            ids = train_pairs[perm[s:s+BS]]
            li_b = ids[:,0]; di_b = ids[:,1]
            zc = cell_enc(Xt[torch.from_numpy(li_b).long().to(DEV)])
            zd = drug_enc(build_batch(di_b.tolist()))
            pred = head(zc, zd)
            tgt = Yt[torch.from_numpy(li_b).long().to(DEV), torch.from_numpy(di_b).long().to(DEV)]
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()*len(ids); n_seen += len(ids)
        sched.step()
        if ep % 3 == 0 or ep == EPOCHS-1:
            log(f'  ep {ep+1:02d}/{EPOCHS}  train_mse={loss_sum/n_seen:.4f}  lr={sched.get_last_lr()[0]:.2e}')

    # ---------- Eval on held-out drugs ----------
    cell_enc.eval(); drug_enc.eval(); head.eval()
    ho_list = sorted(ho_set)
    with torch.no_grad():
        zc = cell_enc(Xt)
        preds = np.zeros((n_line, len(ho_list)), dtype=np.float32)
        for s in range(0, len(ho_list), 256):
            ids = ho_list[s:s+256]
            zd = drug_enc(build_batch(ids))
            zc_rep = zc.unsqueeze(1).expand(-1, len(ids), -1).reshape(-1, zc.shape[1])
            zd_rep = zd.unsqueeze(0).expand(n_line, -1, -1).reshape(-1, zd.shape[1])
            p = head(zc_rep, zd_rep).view(n_line, len(ids)).cpu().numpy()
            preds[:, s:s+len(ids)] = p

    # Per held-out drug Pearson + Spearman
    fold_rows = []
    for k, di in enumerate(ho_list):
        m = M[:, di]
        if m.sum() < 20: continue
        y = Y[m, di]; pp = preds[m, k]
        if y.std() < 1e-6 or pp.std() < 1e-6: continue
        r,_ = pearsonr(pp, y); rho,_ = spearmanr(pp, y)
        fold_rows.append({'fold':fi,'drug':keep2[di],'n':int(m.sum()),
                          'pearson':float(r),'spearman':float(rho),'method':'MPNN'})
    log(f'  evaluated {len(fold_rows)} drugs (≥20 obs each, non-degenerate)')
    if fold_rows:
        rs = pd.DataFrame(fold_rows)['pearson']
        log(f'  Fold-{fi+1} mean Pearson r = {rs.mean():.3f}  median = {rs.median():.3f}  '
            f'r>0.3: {(rs>0.3).mean():.1%}')
    all_rows.extend(fold_rows)

    # ---------- ECFP4 + Ridge baseline ----------
    train_drug_idx = np.where(train_mask)[0]
    # For each cell line: fit Ridge(features=ECFP4 of drugs, target=AUC z) on training drugs;
    # predict for held-out drugs.
    ridge_preds = np.zeros((n_line, len(ho_list)), dtype=np.float32)
    for li in range(n_line):
        m = M[li, train_drug_idx]
        if m.sum() < 30:
            ridge_preds[li, :] = 0.0; continue
        Xtr = fps[train_drug_idx[m]]
        ytr = Y[li, train_drug_idx[m]]
        try:
            mdl = Ridge(alpha=1.0).fit(Xtr, ytr)
            ridge_preds[li, :] = mdl.predict(fps[ho_list])
        except Exception:
            ridge_preds[li, :] = 0.0
    rb_rows = []
    for k, di in enumerate(ho_list):
        m = M[:, di]
        if m.sum() < 20: continue
        y = Y[m, di]; pp = ridge_preds[m, k]
        if y.std() < 1e-6 or pp.std() < 1e-6: continue
        r,_ = pearsonr(pp, y); rho,_ = spearmanr(pp, y)
        rb_rows.append({'fold':fi,'drug':keep2[di],'n':int(m.sum()),
                        'pearson':float(r),'spearman':float(rho),'method':'ECFP4+Ridge'})
    if rb_rows:
        rs = pd.DataFrame(rb_rows)['pearson']
        log(f'  ECFP4+Ridge: mean r = {rs.mean():.3f}  median {rs.median():.3f}')
    all_rows.extend(rb_rows)

    # ---------- Random embedding baseline ----------
    # Replace MPNN forward with random fixed vector per drug; refit only the head + cell_enc tail
    rng2 = np.random.default_rng(SEED + fi)
    rand_emb = rng2.standard_normal((n_drug, D_EMB_DRUG)).astype(np.float32)
    rand_emb_t = torch.from_numpy(rand_emb).to(DEV)
    cell_enc_r = CellEncoder(len(genes)).to(DEV)
    head_r = Head(256, D_EMB_DRUG).to(DEV)
    opt_r = torch.optim.AdamW(list(cell_enc_r.parameters())+list(head_r.parameters()),
                              lr=LR_DRUG, weight_decay=1e-5)
    EPOCHS_R = 8
    for ep in range(EPOCHS_R):
        cell_enc_r.train(); head_r.train()
        perm = rng.permutation(len(train_pairs))
        for s in range(0, len(perm), BS):
            ids = train_pairs[perm[s:s+BS]]
            zc = cell_enc_r(Xt[torch.from_numpy(ids[:,0]).long().to(DEV)])
            zd = rand_emb_t[torch.from_numpy(ids[:,1]).long().to(DEV)]
            pred = head_r(zc, zd)
            tgt = Yt[torch.from_numpy(ids[:,0]).long().to(DEV),
                     torch.from_numpy(ids[:,1]).long().to(DEV)]
            loss = F.mse_loss(pred, tgt)
            opt_r.zero_grad(); loss.backward(); opt_r.step()
    cell_enc_r.eval(); head_r.eval()
    rand_preds = np.zeros((n_line, len(ho_list)), dtype=np.float32)
    with torch.no_grad():
        zc = cell_enc_r(Xt)
        for k, di in enumerate(ho_list):
            zd = rand_emb_t[di:di+1].expand(n_line,-1)
            rand_preds[:, k] = head_r(zc, zd).cpu().numpy()
    rd_rows = []
    for k, di in enumerate(ho_list):
        m = M[:, di]
        if m.sum() < 20: continue
        y = Y[m, di]; pp = rand_preds[m, k]
        if y.std() < 1e-6 or pp.std() < 1e-6: continue
        r,_ = pearsonr(pp, y); rho,_ = spearmanr(pp, y)
        rd_rows.append({'fold':fi,'drug':keep2[di],'n':int(m.sum()),
                        'pearson':float(r),'spearman':float(rho),'method':'Random_emb'})
    if rd_rows:
        rs = pd.DataFrame(rd_rows)['pearson']
        log(f'  Random emb: mean r = {rs.mean():.3f}  median {rs.median():.3f}')
    all_rows.extend(rd_rows)

    # ---------- Cell-line mean baseline ----------
    cm_rows = []
    line_train_mean = np.zeros(n_line, dtype=np.float32)
    for li in range(n_line):
        m = M[li, train_drug_idx]
        line_train_mean[li] = Y[li, train_drug_idx[m]].mean() if m.sum()>0 else 0.0
    for k, di in enumerate(ho_list):
        m = M[:, di]
        if m.sum() < 20: continue
        y = Y[m, di]; pp = line_train_mean[m]
        if y.std() < 1e-6 or pp.std() < 1e-6: continue
        r,_ = pearsonr(pp, y); rho,_ = spearmanr(pp, y)
        cm_rows.append({'fold':fi,'drug':keep2[di],'n':int(m.sum()),
                        'pearson':float(r),'spearman':float(rho),'method':'CellLine_mean'})
    if cm_rows:
        rs = pd.DataFrame(cm_rows)['pearson']
        log(f'  Cell-line mean: mean r = {rs.mean():.3f}  median {rs.median():.3f}')
    all_rows.extend(cm_rows)

    log(f'  fold {fi+1} done in {time.time()-fold_t0:.0f}s')
    fold_details[fi] = {'n_held_out':len(ho_list)}

results = pd.DataFrame(all_rows)
results.to_parquet(OUT/'cv_results.parquet', index=False)

# ---------- 6. Summary ----------
summary = {}
for method in ['MPNN','ECFP4+Ridge','Random_emb','CellLine_mean']:
    sub = results[results['method']==method]
    if len(sub)==0: continue
    summary[method] = {
        'n_drug_evals': int(len(sub)),
        'mean_pearson':  float(sub['pearson'].mean()),
        'median_pearson':float(sub['pearson'].median()),
        'q25_pearson':   float(sub['pearson'].quantile(0.25)),
        'q75_pearson':   float(sub['pearson'].quantile(0.75)),
        'mean_spearman': float(sub['spearman'].mean()),
        'frac_r_gt_0.3': float((sub['pearson']>0.3).mean()),
        'frac_r_gt_0.5': float((sub['pearson']>0.5).mean()),
    }
log('\n=== SUMMARY ===')
for m, s in summary.items():
    log(f'{m:<16}  mean r={s["mean_pearson"]:+.3f}  median {s["median_pearson"]:+.3f}  '
        f'IQR=[{s["q25_pearson"]:+.2f},{s["q75_pearson"]:+.2f}]  '
        f'r>0.3: {s["frac_r_gt_0.3"]:.1%}  r>0.5: {s["frac_r_gt_0.5"]:.1%}  n={s["n_drug_evals"]}')

(OUT/'summary.json').write_text(json.dumps(summary, indent=2))

# ---------- 7. Markdown ----------
md = ['# MPNN held-out drug evaluation — 5-fold CV\n',
      '## Setup',
      f'- Pan-cancer DepMap: {n_line} cell lines × {n_drug} drugs (with SMILES, ≥{MIN_OBS} obs)',
      f'- Drug-stratified {K_FOLDS}-fold CV (random seed={SEED})',
      f'- Target: AUC z-score (per-drug standardised across lines)',
      f'- Each fold: train on (~80% drugs × all lines), predict held-out drugs from SMILES alone',
      '',
      '## Results — per-held-out-drug Pearson r distribution\n',
      '| Method | n drugs | mean r | median r | IQR | r>0.3 | r>0.5 |',
      '|---|---:|---:|---:|---|---:|---:|']
for m, s in summary.items():
    md.append(f"| **{m}** | {s['n_drug_evals']} | {s['mean_pearson']:+.3f} | "
              f"{s['median_pearson']:+.3f} | [{s['q25_pearson']:+.2f}, {s['q75_pearson']:+.2f}] | "
              f"{s['frac_r_gt_0.3']:.1%} | {s['frac_r_gt_0.5']:.1%} |")
md += ['',
       '## Interpretation',
       '- **MPNN** must beat **ECFP4+Ridge** to claim graph structure adds signal beyond Morgan FPs.',
       '- **MPNN ≫ Random_emb** confirms the SMILES → drug-emb mapping is the source of generalisation, not the cell encoder alone.',
       '- **CellLine_mean** is the floor — represents the leakage available from average drug effect per cell line.',
       '- Ideal pattern: MPNN > ECFP4+Ridge > Random_emb ≈ CellLine_mean (the latter two near 0).']
(OUT/'comparison.md').write_text('\n'.join(md))
log(f'Report: {OUT/"comparison.md"}')
log('== MPNN holdout eval done ==')
