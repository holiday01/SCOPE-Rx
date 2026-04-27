"""
T2c2 — Scaden-CA-style baseline: deconvolve tumour bulk into cancer-cell-line
composition, then infer drug response = composition · per-line AUC.

Steps:
  1. Simulate pseudobulk from liver cell-line expression via Dirichlet mixing (+ noise)
  2. Train 3-layer MLP (Scaden architecture) to recover composition
  3. Apply to TCGA-LIHC expression → per-patient cell-line composition
  4. Predict drug AUC per patient = composition · AUC_line_drug (masked mean over observed lines)
  5. Eval: Spearman between predicted drug-ranking on LIHC tumour vs pan-cancer mean ranking
          and correlation of predicted sorafenib/lenvatinib with TCGA survival (quick KM)
"""
from __future__ import annotations
import json, time, math, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path('/home/holiday01/drug_sc')
PROC = ROOT/'data/processed/hcc_drug'
TCGA_EXPR = ROOT/'data/TCGA_LIHC/TCGA_LIHC_expression.gz'
TCGA_CLIN = ROOT/'data/TCGA_LIHC/TCGA_LIHC_clinical.tsv'
OUT  = ROOT/'results/t2c2'; OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT/'checkpoints'; CKPT.mkdir(parents=True, exist_ok=True)
DEV = 'cuda'
SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. HCC cell-line expression ----------
expr_liv = pd.read_parquet(PROC/'cellline_expression_liver.parquet')  # liver lines × shared genes
genes = pd.read_parquet(PROC/'gene_universe.parquet')['gene'].tolist()
meta  = pd.read_parquet(PROC/'cellline_meta.parquet').set_index('ModelID')
log(f'Liver cell lines: {expr_liv.shape[0]}  genes: {expr_liv.shape[1]}')

X_liver = np.log1p(expr_liv.values.astype(np.float32))   # (N_lines × N_genes)
line_ids = expr_liv.index.tolist()
N_LINE = X_liver.shape[0]

# ---------- 2. Simulate pseudobulk via Dirichlet ----------
def simulate(N=20000, low_sparsity=True):
    props = np.random.dirichlet(alpha=np.ones(N_LINE)*(0.3 if low_sparsity else 1.0), size=N)
    bulk = props @ X_liver
    # add gaussian noise in log space
    bulk += np.random.normal(0, 0.15, bulk.shape).astype(np.float32)
    return bulk.astype(np.float32), props.astype(np.float32)

log('Simulating pseudobulk …')
Xp_tr, Yp_tr = simulate(N=30000)
Xp_va, Yp_va = simulate(N=3000)
log(f'Simulated train: {Xp_tr.shape}  val: {Xp_va.shape}')

# standardize per gene on training sim
g_mu = Xp_tr.mean(0); g_sd = Xp_tr.std(0) + 1e-6
Xp_tr_s = (Xp_tr - g_mu) / g_sd
Xp_va_s = (Xp_va - g_mu) / g_sd

# ---------- 3. Scaden MLP ----------
class Scaden(nn.Module):
    def __init__(self, d_in, n_line, hidden=(256,128,64), p=0.1):
        super().__init__()
        layers=[]; d=d_in
        for h in hidden:
            layers += [nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_line)
    def forward(self,x):
        z = self.body(x)
        return F.softmax(self.head(z), dim=-1), z

model = Scaden(X_liver.shape[1], N_LINE).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

Xp_tr_t = torch.from_numpy(Xp_tr_s).to(DEV)
Yp_tr_t = torch.from_numpy(Yp_tr).to(DEV)
Xp_va_t = torch.from_numpy(Xp_va_s).to(DEV)
Yp_va_t = torch.from_numpy(Yp_va).to(DEV)

EPOCHS = 30; BS = 512
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(Xp_tr_t), device=DEV)
    tot = 0.0
    for i in range(0, len(perm), BS):
        idx = perm[i:i+BS]
        p,_ = model(Xp_tr_t[idx])
        loss = F.l1_loss(p, Yp_tr_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*len(idx)
    model.eval()
    with torch.no_grad():
        p_va,_ = model(Xp_va_t)
        va_mae = F.l1_loss(p_va, Yp_va_t).item()
        # per-line Pearson averaged
        ppv = p_va.cpu().numpy(); yv = Yp_va.copy()
        perln_r=[]
        for j in range(N_LINE):
            if yv[:,j].std()>1e-6 and ppv[:,j].std()>1e-6:
                perln_r.append(np.corrcoef(ppv[:,j], yv[:,j])[0,1])
        log(f'ep {ep+1:02d}/{EPOCHS}  tr_mae={tot/len(Xp_tr_t):.4f}  va_mae={va_mae:.4f}  mean_perline_r={np.mean(perln_r):.3f}')

# ---------- 4. Apply to TCGA-LIHC ----------
log('Loading TCGA-LIHC expression …')
t_expr = pd.read_csv(TCGA_EXPR, sep='\t', index_col=0, low_memory=False)
log(f'TCGA-LIHC shape raw: {t_expr.shape}')
# TCGA is HTSeq-FPKM (Xena) -> log2(x+1). bring to log1p scale
# detect if already log-scaled by checking max
if t_expr.max().max() < 30:
    log('TCGA expression appears already log-transformed — reversing before log1p')
    t_expr_lin = np.expm1(t_expr.values * np.log(2))  # from log2
    t_expr_lin = np.clip(t_expr_lin, 0, None)
else:
    t_expr_lin = t_expr.values
# map to gene order
t_gene_idx = {g:i for i,g in enumerate(t_expr.index)}
sel = np.array([t_gene_idx[g] if g in t_gene_idx else -1 for g in genes])
valid_mask = sel>=0
log(f'TCGA genes aligned: {valid_mask.sum()}/{len(genes)}')
# samples × genes (only valid genes filled, others zero)
Xt = np.zeros((t_expr.shape[1], len(genes)), dtype=np.float32)
Xt[:, valid_mask] = np.log1p(t_expr_lin[sel[valid_mask], :].T)
Xt_s = (Xt - g_mu) / g_sd
log(f'TCGA processed: {Xt.shape}')

model.eval()
with torch.no_grad():
    prop_t, _ = model(torch.from_numpy(Xt_s).to(DEV))
    prop_t = prop_t.cpu().numpy()
prop_df = pd.DataFrame(prop_t, index=t_expr.columns, columns=line_ids)
log(f'TCGA composition: {prop_df.shape}')
log(f'Avg line proportions in TCGA-LIHC (top 10):')
log(prop_df.mean(0).sort_values(ascending=False).head(10).to_string())

# ---------- 5. Per-patient drug response prediction ----------
long_tab = pd.read_parquet(PROC/'drug_response_long.parquet')
agg = (long_tab.dropna(subset=['auc']).groupby(['drug','ModelID'])['auc'].mean().unstack('ModelID'))
# restrict to liver lines
agg_liv = agg.reindex(columns=line_ids)
# mask-normalized dot: for each sample, sum(prop_i * auc_i) / sum(prop_i_where_auc_exists)
A = agg_liv.values.astype(np.float32)   # drugs × N_LINE
mask = ~np.isnan(A)
A_filled = np.nan_to_num(A, nan=0.0)
log(f'AUC mat: {A.shape}  coverage={mask.mean():.2f}')

P = prop_t                                 # (n_tcga, N_LINE)
denom = P @ mask.T.astype(np.float32)      # (n_tcga, n_drugs) sum of props where auc exists
numer = P @ A_filled.T                     # (n_tcga, n_drugs)
pred  = numer / (denom + 1e-6)
pred[denom<0.05] = np.nan                  # too few supporting lines
pred_df = pd.DataFrame(pred, index=t_expr.columns, columns=agg_liv.index)
pred_df.to_parquet(OUT/'tcga_lihc_predicted_auc.parquet')
log(f'TCGA predicted drug AUC: {pred_df.shape}  fraction NaN={pred_df.isna().mean().mean():.2f}')

# ---------- 6. TCGA survival + sorafenib sanity ----------
log('Loading TCGA clinical …')
clin = pd.read_csv(TCGA_CLIN, sep='\t', low_memory=False)
# figure out survival cols
cols = {c.lower():c for c in clin.columns}
sid_col = cols.get('sampleid') or cols.get('bcr_sample_barcode') or clin.columns[0]
log(f'clinical cols hint: sampleid={sid_col}')
# common Xena fields: OS, OS.time
os_col = next((c for c in clin.columns if c.lower() in ('os','os_status','_os','os_event')), None)
ost_col= next((c for c in clin.columns if c.lower() in ('os.time','os_time','_os_time','os_days')), None)
if os_col is None or ost_col is None:
    # try other candidates
    for c in clin.columns:
        lc=c.lower()
        if os_col is None and ('overall_survival' in lc or 'vital_status' in lc): os_col=c
        if ost_col is None and ('days_to_death' in lc or 'overall_survival_time' in lc or 'os.time' in lc): ost_col=c
log(f'os col={os_col}  os.time col={ost_col}')

km_results={}
if os_col and ost_col:
    from lifelines import CoxPHFitter
    clin_sub = clin.rename(columns={sid_col:'sample', os_col:'event', ost_col:'time'})
    clin_sub = clin_sub[['sample','event','time']]
    # sample IDs in TCGA expression are like TCGA-XX-XXXX-01; clinical sample IDs similar
    pred_df.index.name='sample'
    merged = pred_df.reset_index().merge(clin_sub, on='sample', how='inner')
    log(f'merged survival rows: {len(merged)}')
    for drug in ['sorafenib','lenvatinib','regorafenib','gemcitabine','doxorubicin','paclitaxel']:
        if drug not in pred_df.columns: continue
        d = merged[['event','time',drug]].dropna()
        if len(d)<30: continue
        # convert event/time to numeric
        ev = pd.to_numeric(d['event'], errors='coerce')
        tm = pd.to_numeric(d['time'], errors='coerce')
        valid = ev.notna() & tm.notna() & (tm>0)
        if valid.sum()<30: continue
        dd = pd.DataFrame({'event':ev[valid].astype(int),'time':tm[valid],'x':d.loc[valid,drug].astype(float)})
        try:
            cph = CoxPHFitter(penalizer=0.01).fit(dd, 'time','event')
            hr = float(np.exp(cph.params_['x']))
            p  = float(cph.summary.loc['x','p'])
            km_results[drug]={'hr':hr,'p':p,'n':int(valid.sum())}
        except Exception as e:
            km_results[drug]={'error':str(e)}
    log(f'Cox HR for predicted AUC ~ OS: {json.dumps(km_results, indent=2)}')

# ---------- 7. Summary ----------
torch.save({'model':model.state_dict(),'genes':genes,'line_ids':line_ids,'g_mu':g_mu,'g_sd':g_sd},
           CKPT/'t2c2_scaden.pt')
summary = {
    'baseline':'Scaden-CA',
    'n_liver_lines_as_mixing_basis':N_LINE,
    'tcga_samples':int(prop_df.shape[0]),
    'avg_top_line':prop_df.mean(0).sort_values(ascending=False).head(5).to_dict(),
    'cox_drug_survival':km_results,
}
(OUT/'eval_metrics.json').write_text(json.dumps(summary, indent=2, default=str))
log('== T2c2 done ==')
