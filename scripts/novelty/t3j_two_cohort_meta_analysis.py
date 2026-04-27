"""
T3j — Two-cohort external validation + meta-analysis

Cohorts:
  - TCGA-LIHC                  (training, RNA-seq, n≈423)
  - GSE76427 Yang Singapore    (Illumina HT-12 v4, n=115 tumor)
  - GSE14520 Roessler LCI      (Affymetrix HG-U133A 2.0, n=247 tumor)

Steps:
  1. Process GSE14520 (parse + probe map + clinical merge)
  2. ComBat harmonisation: bring GSE76427 + GSE14520 onto TCGA gene-axis distribution
  3. Apply T3c attention deconvolver to harmonised matrices
  4. Per-prototype Cox in each external cohort (OS + RFS)
  5. Composite TCGA-derived risk score → c-index in each cohort
  6. Fixed-effect meta-analysis combining TCGA + GSE76427 + GSE14520
  7. Concordance summary
"""
from __future__ import annotations
import gzip, json, time, re, math
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from lifelines import CoxPHFitter
from scipy.stats import pearsonr, spearmanr

ROOT = Path('/home/holiday01/drug_sc')
EXT  = ROOT/'data/external_HCC'
T3C  = ROOT/'results/t3c'
T3I  = ROOT/'results/t3i'
OUT  = ROOT/'results/t3j'; OUT.mkdir(parents=True, exist_ok=True)
DEV  = 'cuda'
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ============================================================
# 1. Parse GSE14520 (GPL3921 Affymetrix HT_HG-U133A)
# ============================================================
log('Parsing GSE14520-GPL3921 series matrix …')
sm = EXT/'GSE14520-GPL3921_series_matrix.txt.gz'
sample_ids = []; gsm_ids = []; tissue=[]; individual=[]; data=[]; probes=[]
in_data=False
with gzip.open(sm,'rt') as f:
    for line in f:
        if line.startswith('!Sample_geo_accession'):
            gsm_ids = [x.strip().strip('"') for x in line.split('\t')[1:]]
        elif line.startswith('!Sample_characteristics_ch1'):
            cells = [x.strip().strip('"') for x in line.split('\t')[1:]]
            head = cells[0].split(':')[0].strip().lower() if cells else ''
            vals = [c.split(':',1)[1].strip() if ':' in c else '' for c in cells]
            if 'tissue' in head: tissue = vals
            elif 'individual' in head: individual = vals
        elif line.startswith('!series_matrix_table_begin'):
            in_data=True
        elif line.startswith('!series_matrix_table_end'):
            in_data=False
        elif in_data:
            parts = line.rstrip('\n').split('\t')
            if not parts or parts[0].startswith('!') or parts[0].strip()=='ID_REF': continue
            try:
                vals = [float(x) if x not in ('','NA','null') else np.nan for x in parts[1:]]
            except: continue
            probes.append(parts[0].strip().strip('"'))
            data.append(vals)
expr_g14 = pd.DataFrame(np.array(data,dtype=np.float32), index=probes, columns=gsm_ids)
phen_g14 = pd.DataFrame({'gsm':gsm_ids,'tissue':tissue,'individual':individual}).set_index('gsm')
log(f'GSE14520 matrix: {expr_g14.shape[0]} probes × {expr_g14.shape[1]} samples')
log(f'  tumor: {(phen_g14.tissue.str.contains("Tumor", case=False)).sum()}')

# ============================================================
# 2. Map GPL3921 probes → genes
# ============================================================
log('Loading GPL3921 (HG-U133A 2.0) annotation …')
gpl_path = EXT/'GPL3921_full.soft.gz'
if not gpl_path.exists():
    import subprocess
    subprocess.run(['curl','-sL','--max-time','300','-o',str(gpl_path),
                    'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL3nnn/GPL3921/annot/GPL3921.annot.gz'], check=False)
log(f'GPL3921 annot file size: {gpl_path.stat().st_size/1e6:.1f} MB')
probe_to_gene_g14 = {}
with gzip.open(gpl_path,'rt',errors='replace') as f:
    in_table=False; cols=None; gs_idx=None
    for line in f:
        if line.startswith('!platform_table_begin'):
            in_table=True; continue
        if line.startswith('!platform_table_end'): break
        if not in_table: continue
        if cols is None:
            cols = line.rstrip('\n').split('\t')
            for i,c in enumerate(cols):
                if c.lower() in ('gene symbol','symbol'): gs_idx=i; break
            continue
        parts = line.rstrip('\n').split('\t')
        if len(parts)<=gs_idx: continue
        pid = parts[0].strip()
        gs = parts[gs_idx].split(' /// ')[0].strip()
        if gs and gs not in ('---','-','NA',''):
            probe_to_gene_g14[pid] = gs
log(f'  Probe→gene: {len(probe_to_gene_g14)}')

# Aggregate to gene level
mapped = expr_g14.loc[expr_g14.index.intersection(probe_to_gene_g14)].copy()
mapped['_g'] = mapped.index.map(probe_to_gene_g14)
gene_g14 = mapped.groupby('_g').max()
log(f'  Gene-level: {gene_g14.shape}')

# ============================================================
# 3. Merge clinical
# ============================================================
log('Merging GSE14520 clinical (Roessler 2010 supplement) …')
clin = pd.read_csv(EXT/'GSE14520_Extra_Supplement.txt', sep='\t')
log(f'  Clinical rows: {len(clin)}; cols: {list(clin.columns)[:8]} ...')
# join key: 'Affy_GSM'
clin = clin.dropna(subset=['Affy_GSM']).set_index('Affy_GSM')
# tumor only
tumor_mask_g14 = phen_g14.tissue.str.contains("Tumor", case=False) & ~phen_g14.tissue.str.contains("Non-Tumor", case=False)
phen_g14_t = phen_g14[tumor_mask_g14].copy()
# attach clinical
common = phen_g14_t.index.intersection(clin.index)
log(f'  Tumor with clinical: {len(common)}')
phen_g14_t = phen_g14_t.loc[common]
phen_g14_t['os_event'] = pd.to_numeric(clin.loc[common,'Survival status'], errors='coerce')
phen_g14_t['os_time']  = pd.to_numeric(clin.loc[common,'Survival months'], errors='coerce')*30   # months→days
phen_g14_t['rfs_event']= pd.to_numeric(clin.loc[common,'Recurr status'], errors='coerce')
phen_g14_t['rfs_time'] = pd.to_numeric(clin.loc[common,'Recurr months'], errors='coerce')*30
phen_g14_t['age']      = pd.to_numeric(clin.loc[common,'Age'], errors='coerce')
phen_g14_t['male']     = (clin.loc[common,'Gender'].astype(str).str.upper()=='M').astype(float)
bclc_map = {'0':0,'A':1,'B':2,'C':3,'D':4}
phen_g14_t['bclc_num'] = clin.loc[common,'BCLC staging'].astype(str).str.strip().map(bclc_map)
log(f'  OS available: {phen_g14_t.os_event.notna().sum()}  RFS: {phen_g14_t.rfs_event.notna().sum()}')

# ============================================================
# 4. Build harmonised expression matrix for both cohorts
# ============================================================
proto_expr = pd.read_parquet(T3C/'prototype_expression.parquet')
genes_shared = list(proto_expr.columns)
gene_to_i = {g:i for i,g in enumerate(genes_shared)}

def cohort_to_full(gene_df, sample_ids):
    """gene_df: gene × sample (linear scale).  Returns (samples × |genes_shared|) log1p-CP10K."""
    common = [g for g in genes_shared if g in gene_df.index]
    Xp = gene_df.loc[common, sample_ids].T.values.astype(np.float32)
    if Xp.max() < 30:
        Xp = np.clip(np.expm1(Xp * np.log(2)), 0, None)
    else:
        Xp = np.clip(Xp, 0, None)
    full = np.zeros((len(sample_ids), len(genes_shared)), dtype=np.float32)
    for i,g in enumerate(common):
        full[:, gene_to_i[g]] = Xp[:, i]
    rs = full.sum(1, keepdims=True)+1e-6
    return np.log1p(np.clip(full/rs*1e4, 0, None)).astype(np.float32), common

log('\nBuilding GSE14520 expression matrix …')
g14_samples = phen_g14_t.index.tolist()
X_g14, common_g14 = cohort_to_full(gene_g14, g14_samples)
log(f'  GSE14520: {X_g14.shape}  shared genes used: {len(common_g14)}')

# Reload GSE76427 from previous run output (or re-process)
log('Re-loading GSE76427 …')
sm76 = EXT/'GSE76427_series.txt.gz'
gsm76=[]; tissue76=[]; phen76={}
in_d=False; data76=[]; probes76=[]
phen76_keys = ['patient id','tissue','event_os','duryears_os','event_rfs','duryears_rfs','age','gender','bclc_staging','tnm']
with gzip.open(sm76,'rt') as f:
    for line in f:
        if line.startswith('!Sample_geo_accession'):
            gsm76 = [x.strip().strip('"') for x in line.split('\t')[1:]]
        elif line.startswith('!Sample_characteristics_ch1'):
            cells = [x.strip().strip('"') for x in line.split('\t')[1:]]
            head = cells[0].split(':')[0].strip().lower() if cells else ''
            for k in phen76_keys:
                if head.startswith(k):
                    vals = [c.split(':',1)[1].strip() if ':' in c else '' for c in cells]
                    phen76[k]=vals; break
        elif line.startswith('!series_matrix_table_begin'): in_d=True
        elif line.startswith('!series_matrix_table_end'): in_d=False
        elif in_d:
            parts = line.rstrip('\n').split('\t')
            if not parts or parts[0].startswith('!') or parts[0].strip()=='ID_REF': continue
            try:
                v = [float(x) if x not in ('','NA','null') else np.nan for x in parts[1:]]
            except: continue
            probes76.append(parts[0].strip().strip('"'))
            data76.append(v)
expr76 = pd.DataFrame(np.array(data76,dtype=np.float32), index=probes76, columns=gsm76)
expr76.index = [s.strip().strip('"') for s in expr76.index]
phen76_df = pd.DataFrame(phen76, index=gsm76)
# probe→gene from GPL10558
ptg76 = {}
with gzip.open(EXT/'GPL10558_full.soft.gz','rt',errors='replace') as f:
    in_t=False; cols=None; gi=None
    for line in f:
        if line.startswith('!platform_table_begin'): in_t=True; continue
        if line.startswith('!platform_table_end'): break
        if not in_t: continue
        if cols is None:
            cols = line.rstrip('\n').split('\t')
            for i,c in enumerate(cols):
                if c.lower() in ('gene symbol','symbol'): gi=i; break
            continue
        parts=line.rstrip('\n').split('\t')
        if len(parts)<=gi: continue
        pid=parts[0].strip()
        gs=parts[gi].split(' /// ')[0].strip()
        if gs and gs not in ('---','-','NA',''):
            ptg76[pid]=gs
mapped76 = expr76.loc[expr76.index.intersection(ptg76)].copy()
mapped76['_g'] = mapped76.index.map(ptg76)
gene76 = mapped76.groupby('_g').max()
phen76_t_mask = phen76_df.tissue.astype(str).str.contains('tumor', case=False) & ~phen76_df.tissue.astype(str).str.contains('non-tumor|adjacent', case=False)
g76_samples = phen76_df[phen76_t_mask].index.tolist()
X_g76, common_g76 = cohort_to_full(gene76, g76_samples)
log(f'  GSE76427: {X_g76.shape}  shared genes used: {len(common_g76)}')

# ============================================================
# 5. ComBat harmonisation
# ============================================================
log('\nComBat harmonisation: TCGA + GSE76427 + GSE14520 onto common scale …')
# load TCGA processed once for batch reference
tcga_expr = pd.read_csv(ROOT/'data/TCGA_LIHC/TCGA_LIHC_expression.gz', sep='\t', index_col=0)
tcga_lin = np.clip(np.expm1(tcga_expr.values * np.log(2)), 0, None).astype(np.float32) if tcga_expr.values.max()<30 else tcga_expr.values.astype(np.float32)
tcga_g_to_i = {g:i for i,g in enumerate(tcga_expr.index)}
sel = np.array([tcga_g_to_i.get(g,-1) for g in genes_shared])
valid = sel>=0
T = np.zeros((tcga_expr.shape[1], len(genes_shared)), dtype=np.float32)
T[:, valid] = tcga_lin[sel[valid],:].T
rs = T.sum(1, keepdims=True)+1e-6
X_tcga = np.log1p(np.clip(T/rs*1e4,0,None)).astype(np.float32)
tcga_samples = list(tcga_expr.columns)
log(f'  TCGA matrix: {X_tcga.shape}')

# Stack all
batch = np.array(['TCGA']*X_tcga.shape[0] + ['GSE76427']*X_g76.shape[0] + ['GSE14520']*X_g14.shape[0])
X_all = np.vstack([X_tcga, X_g76, X_g14])
samp_all = tcga_samples + g76_samples + g14_samples
log(f'  Stacked matrix: {X_all.shape}  batches: {pd.Series(batch).value_counts().to_dict()}')

# pycombat needs (genes × samples)
from pycombat import Combat
log('  Running ComBat (this may take ~30s) …')
try:
    cb = Combat()
    Xt = X_all.T.astype(float)        # genes × samples
    # remove zero-variance genes for combat numerical stability
    var = Xt.var(1)
    keep_g = var > 1e-6
    Xt_kept = Xt[keep_g]
    batch_codes = pd.Series(batch).astype('category').cat.codes.values.astype(int)
    Xc = cb.fit_transform(y=Xt_kept.T, b=batch_codes, X=None, C=None)
    Xc = Xc.T
    Xt_full = Xt.copy()
    Xt_full[keep_g] = Xc
    X_harm = Xt_full.T.astype(np.float32)
    log(f'  ComBat done.  Pre-batch mean diff (gene-mean SD across batches): '
        f'before {pd.DataFrame(X_all,index=batch).groupby(level=0).mean().std(axis=0).mean():.3f}, '
        f'after {pd.DataFrame(X_harm,index=batch).groupby(level=0).mean().std(axis=0).mean():.3f}')
except Exception as e:
    log(f'  ComBat failed ({e}) — falling back to per-cohort z-score harmonisation')
    X_harm = np.zeros_like(X_all)
    for b in np.unique(batch):
        m = batch==b
        Xb = X_all[m]
        gm = Xb.mean(0); gs = Xb.std(0)+1e-6
        X_harm[m] = (Xb - gm)/gs
        # rescale to a common (mean=TCGA-mean, std=TCGA-std)
        if b!='TCGA':
            tm = X_all[batch=='TCGA'].mean(0); ts = X_all[batch=='TCGA'].std(0)+1e-6
            X_harm[m] = X_harm[m]*ts + tm

# Split back
X_tcga_h = X_harm[batch=='TCGA']
X_g76_h  = X_harm[batch=='GSE76427']
X_g14_h  = X_harm[batch=='GSE14520']

# ============================================================
# 6. Apply T3c attention deconvolver to harmonised cohorts
# ============================================================
log('\nApplying T3c attention deconvolver on harmonised cohorts …')
ck = torch.load(ROOT/'checkpoints/t3c_attn_deconv.pt', map_location='cpu', weights_only=False)
n_proto = int(ck['n_proto'])
g_mu = np.asarray(ck['g_mu']); g_sd = np.asarray(ck['g_sd'])

class AttnDeconv(nn.Module):
    def __init__(self,d_in,n_proto,d_hid=256,p=0.1):
        super().__init__()
        self.query_enc = nn.Sequential(nn.Linear(d_in,512),nn.LayerNorm(512),nn.ReLU(),nn.Dropout(p),nn.Linear(512,d_hid))
        self.proto_key = nn.Parameter(torch.randn(n_proto,d_hid)*0.02)
        self.temp = nn.Parameter(torch.tensor(1.0))
    def forward(self,x):
        q = self.query_enc(x)
        return F.softmax((q @ self.proto_key.T)/self.temp.clamp(min=0.1), dim=-1)
model = AttnDeconv(X_harm.shape[1], n_proto).to(DEV).eval()
model.load_state_dict(ck['model'])

def deconv(X):
    Xs = (X - g_mu)/g_sd
    with torch.no_grad():
        return model(torch.from_numpy(Xs).to(DEV)).cpu().numpy()

comp_tcga = deconv(X_tcga_h)
comp_g76  = deconv(X_g76_h)
comp_g14  = deconv(X_g14_h)
log(f'  Compositions: TCGA {comp_tcga.shape}  GSE76427 {comp_g76.shape}  GSE14520 {comp_g14.shape}')
log(f'  Entropies: TCGA {(-(comp_tcga*np.log(comp_tcga+1e-9)).sum(1)).mean():.2f}  '
    f'GSE76427 {(-(comp_g76*np.log(comp_g76+1e-9)).sum(1)).mean():.2f}  '
    f'GSE14520 {(-(comp_g14*np.log(comp_g14+1e-9)).sum(1)).mean():.2f}  / max {np.log(n_proto):.2f}')

# Save composition tables
pd.DataFrame(comp_g14, index=g14_samples, columns=[f'proto_{i}' for i in range(n_proto)]).to_parquet(OUT/'gse14520_composition.parquet')
pd.DataFrame(comp_g76, index=g76_samples, columns=[f'proto_{i}' for i in range(n_proto)]).to_parquet(OUT/'gse76427_composition_harmonised.parquet')

# ============================================================
# 7. Per-prototype Cox per cohort
# ============================================================
def cox_runner(comp, phen, ev, tm, covs, cohort):
    rows=[]
    df = pd.DataFrame(comp, index=phen.index, columns=[f'p{i}' for i in range(n_proto)])
    df = df.join(phen[[ev,tm]+covs])
    df = df.dropna(subset=[ev,tm]+covs)
    df = df[df[tm]>0]
    if len(df)<30: return pd.DataFrame()
    for p in range(n_proto):
        c = f'p{p}'
        d = df[[ev,tm,c]+covs].dropna()
        if len(d)<50 or d[c].std()<1e-5: continue
        d['x'] = (d[c]-d[c].mean())/d[c].std()
        try:
            cph = CoxPHFitter(penalizer=0.05).fit(d[[ev,tm,'x']+covs], tm, ev)
            rows.append({'cohort':cohort,'proto':p,
                         'HR':float(np.exp(cph.params_['x'])),
                         'p':float(cph.summary.loc['x','p']),
                         'n':int(len(d))})
        except: pass
    return pd.DataFrame(rows)

# rebuild phen with index = sample id and time/event
phen_g14_t.index.name='gsm'  # already
phen_g14_use = phen_g14_t.copy()

# GSE76427 phen with time/event
def to_num(s): return pd.to_numeric(s, errors='coerce')
phen76_use = phen76_df[phen76_t_mask].copy()
phen76_use['os_event'] = to_num(phen76_use['event_os'])
phen76_use['os_time']  = to_num(phen76_use['duryears_os'])*365
phen76_use['rfs_event']= to_num(phen76_use['event_rfs'])
phen76_use['rfs_time'] = to_num(phen76_use['duryears_rfs'])*365
phen76_use['age']      = to_num(phen76_use['age'])
phen76_use['male']     = (to_num(phen76_use['gender'])==1).astype(float)
phen76_use['bclc_num'] = phen76_use['bclc_staging'].astype(str).str.strip().map({'A':1,'B':2,'C':3,'D':4,'0':0})

cox_g14_os  = cox_runner(comp_g14, phen_g14_use, 'os_event','os_time', ['age','male','bclc_num'], 'GSE14520_OS')
cox_g14_rfs = cox_runner(comp_g14, phen_g14_use, 'rfs_event','rfs_time', ['age','male','bclc_num'],'GSE14520_RFS')
cox_g76_os  = cox_runner(comp_g76, phen76_use,  'os_event','os_time', ['age','male','bclc_num'], 'GSE76427_OS')
cox_g76_rfs = cox_runner(comp_g76, phen76_use,  'rfs_event','rfs_time', ['age','male','bclc_num'], 'GSE76427_RFS')
all_cox = pd.concat([cox_g14_os, cox_g14_rfs, cox_g76_os, cox_g76_rfs])
all_cox.to_parquet(OUT/'all_external_cox.parquet', index=False)
for cc in ['GSE14520_OS','GSE14520_RFS','GSE76427_OS','GSE76427_RFS']:
    sig = (all_cox[(all_cox.cohort==cc) & (all_cox.p<0.05)])
    log(f'  {cc}: {len(sig)} prototypes significant of {(all_cox.cohort==cc).sum()}')

# ============================================================
# 8. Composite TCGA-derived risk score test
# ============================================================
log('\nComposite TCGA-derived risk score (per-prototype log HR weights from TCGA mv) …')
tcga_mv = pd.read_parquet(ROOT/'results/t3h/multivariate_cox.parquet').set_index('proto')
risk_w = np.zeros(n_proto, dtype=np.float32)
for p,r in tcga_mv.iterrows():
    if float(r['p_x_mv']) < 0.1:
        risk_w[int(p)] = float(np.log(r['HR_x_uni']))
log(f'  Using {(risk_w!=0).sum()} TCGA-prognostic prototypes')

def composite_test(comp, phen, ev, tm, covs, label):
    rs = comp.dot(risk_w)
    df = phen.copy()
    df['risk_z'] = (rs - rs.mean())/rs.std()
    d = df[['risk_z',ev,tm]+covs].dropna()
    d = d[d[tm]>0]
    if len(d)<50: return None
    try:
        cph = CoxPHFitter(penalizer=0.05).fit(d[['risk_z',ev,tm]+covs], tm, ev)
        return {'cohort':label, 'n':int(len(d)),
                'HR': float(np.exp(cph.params_['risk_z'])),
                'p':  float(cph.summary.loc['risk_z','p']),
                'c_index': float(cph.concordance_index_)}
    except Exception as e:
        return {'cohort':label,'error':str(e)}

comp_results=[]
df_g14_phen = phen_g14_use.copy(); df_g14_phen.index = g14_samples
df_g76_phen = phen76_use.copy(); df_g76_phen.index = g76_samples
for comp_arr, ph, label_pref in [
    (comp_g14, df_g14_phen, 'GSE14520'),
    (comp_g76, df_g76_phen, 'GSE76427')]:
    for ev,tm,suf in [('os_event','os_time','_OS'),('rfs_event','rfs_time','_RFS')]:
        r = composite_test(comp_arr, ph, ev, tm, ['age','male','bclc_num'], label_pref+suf)
        if r: comp_results.append(r)
comp_df_res = pd.DataFrame(comp_results)
comp_df_res.to_parquet(OUT/'composite_risk_results.parquet', index=False)
log('Composite risk test:')
print(comp_df_res.to_string(index=False))

# ============================================================
# 9. Meta-analysis (fixed-effect) per prototype: TCGA+G76+G14 OS
# ============================================================
log('\nFixed-effect meta-analysis across TCGA-OS + GSE76427-OS + GSE14520-OS …')
# pull TCGA mv
tcga_os = tcga_mv.copy()
tcga_os['cohort']='TCGA_OS'
tcga_os['proto']=tcga_os.index
tcga_os['HR']=tcga_os['HR_x_uni']; tcga_os['p']=tcga_os['p_x_mv']; tcga_os['n']=tcga_os.get('n', 423)
mt = tcga_os[['cohort','proto','HR','p','n']]
g14_os = cox_g14_os[['cohort','proto','HR','p','n']]
g76_os = cox_g76_os[['cohort','proto','HR','p','n']]
agg = pd.concat([mt, g14_os, g76_os])
agg['logHR'] = np.log(agg['HR'].astype(float))
# SE from p-value approximation: |logHR| / Z, Z = norm.ppf(1-p/2)
from scipy.stats import norm
agg['z'] = np.where(agg['p']>0, norm.ppf(1 - agg['p'].clip(1e-300,0.999999)/2), 0)
agg['SE'] = np.where(np.abs(agg['z'])>1e-6, np.abs(agg['logHR'])/np.abs(agg['z']), np.nan)
agg['w']  = 1/(agg['SE']**2)

# fixed-effect per prototype
meta_rows=[]
for p in agg['proto'].unique():
    sub = agg[agg['proto']==p].dropna(subset=['logHR','SE'])
    if len(sub)<2: continue
    w = sub['w'].values
    if w.sum() < 1e-9: continue
    pooled_logHR = (sub['logHR']*sub['w']).sum() / sub['w'].sum()
    pooled_SE = (1/sub['w'].sum())**0.5
    pooled_z  = pooled_logHR / pooled_SE
    pooled_p  = 2*(1 - norm.cdf(abs(pooled_z)))
    # heterogeneity Q
    Q = (sub['w']*(sub['logHR']-pooled_logHR)**2).sum()
    df = len(sub)-1
    meta_rows.append({'proto':int(p), 'n_cohorts':int(len(sub)),
                      'pooled_logHR':float(pooled_logHR),
                      'pooled_HR':float(np.exp(pooled_logHR)),
                      'pooled_p':float(pooled_p),
                      'Q':float(Q), 'df':int(df)})
meta = pd.DataFrame(meta_rows).sort_values('pooled_p')
proto_meta = pd.read_parquet(T3C/'prototype_meta.parquet').set_index('proto')
meta = meta.join(proto_meta[['dominant_cell_type','label']], on='proto')
meta.to_parquet(OUT/'meta_analysis_pooled.parquet', index=False)
log(f'  Prototypes meta-pooled with ≥2 cohorts: {len(meta)}; significant pooled p<0.05: {(meta.pooled_p<0.05).sum()}')
log('Top 10 meta-significant prototypes:')
print(meta.head(10).to_string(index=False))

# ============================================================
# 10. Concordance rates
# ============================================================
log('\nConcordance: TCGA mv-significant 9 prototypes — replication count …')
tcga_sig = tcga_mv[tcga_mv['p_x_mv']<0.05].copy(); tcga_sig['proto']=tcga_sig.index
def replic(co_df, mv_df):
    rep=0
    for p, r in mv_df.iterrows():
        sub = co_df[co_df['proto']==p]
        if len(sub)==0: continue
        if (np.sign(np.log(sub['HR'].iloc[0]))==np.sign(np.log(r['HR_x_uni']))) and (sub['p'].iloc[0]<0.1):
            rep+=1
    return rep
rep_g14_os  = replic(cox_g14_os, tcga_sig)
rep_g14_rfs = replic(cox_g14_rfs, tcga_sig)
rep_g76_os  = replic(cox_g76_os, tcga_sig)
rep_g76_rfs = replic(cox_g76_rfs, tcga_sig)
log(f'  TCGA-sig 9 prototypes replicated (same direction & p<0.1):')
log(f'    GSE14520 OS:  {rep_g14_os}/9')
log(f'    GSE14520 RFS: {rep_g14_rfs}/9')
log(f'    GSE76427 OS:  {rep_g76_os}/9')
log(f'    GSE76427 RFS: {rep_g76_rfs}/9')
log(f'  Replicated in ≥1 external endpoint: {sum(1 for p in tcga_sig.index if any(replic(c, tcga_sig.loc[[p]])>0 for c in [cox_g14_os,cox_g14_rfs,cox_g76_os,cox_g76_rfs]))}/9')

# ============================================================
# 11. Final report
# ============================================================
lines=['# T3j — Two-cohort external validation + ComBat + meta-analysis\n',
       '## Cohorts',
       f'- TCGA-LIHC (training): n=423 RNA-seq',
       f'- GSE76427 Singapore: n=115 tumor (Illumina HT-12 v4)',
       f'- GSE14520 LCI (Roessler 2010): n={len(g14_samples)} tumor (Affymetrix HG-U133A 2.0)',
       '',
       '## Composition entropy (model architecture preserved across cohorts)',
       f'- TCGA: {(-(comp_tcga*np.log(comp_tcga+1e-9)).sum(1)).mean()/np.log(n_proto):.2%} of max',
       f'- GSE76427: {(-(comp_g76*np.log(comp_g76+1e-9)).sum(1)).mean()/np.log(n_proto):.2%}',
       f'- GSE14520: {(-(comp_g14*np.log(comp_g14+1e-9)).sum(1)).mean()/np.log(n_proto):.2%}',
       '',
       '## Composite TCGA-derived risk score (concordance index)',
       '| Cohort | Endpoint | n | HR per SD | p | c-index |',
       '|---|---|---:|---:|---:|---:|',
]
for r in comp_results:
    if 'error' in r: continue
    lines.append(f"| {r['cohort'].split('_')[0]} | {r['cohort'].split('_')[1]} | {r['n']} | {r['HR']:.2f} | {r['p']:.3g} | {r['c_index']:.3f} |")

lines.append('\n## Replication of TCGA mv-significant prototypes (9)\n')
lines.append('| Cohort/endpoint | replicated | rate |')
lines.append('|---|---:|---:|')
for lbl, n in [('GSE14520 OS',rep_g14_os),('GSE14520 RFS',rep_g14_rfs),
               ('GSE76427 OS',rep_g76_os),('GSE76427 RFS',rep_g76_rfs)]:
    lines.append(f'| {lbl} | {n}/9 | {n/9:.1%} |')

lines.append('\n## Top 15 meta-pooled prognostic prototypes (3-cohort fixed-effect)\n')
lines.append('| proto | dominant cell type | pooled HR | pooled p | n cohorts |')
lines.append('|---:|---|---:|---:|---:|')
for _,r in meta.head(15).iterrows():
    lines.append(f"| {int(r['proto'])} | {r['dominant_cell_type']} | {r['pooled_HR']:.2f} | {r['pooled_p']:.3g} | {int(r['n_cohorts'])} |")

(OUT/'comparison_validation_two_cohort.md').write_text('\n'.join(lines))

(OUT/'eval_metrics.json').write_text(json.dumps({
    'cohorts': {'TCGA':int(comp_tcga.shape[0]),
                'GSE76427':int(comp_g76.shape[0]),
                'GSE14520':int(comp_g14.shape[0])},
    'composition_entropy_pct': {
        'TCGA': float((-(comp_tcga*np.log(comp_tcga+1e-9)).sum(1)).mean()/np.log(n_proto)),
        'GSE76427': float((-(comp_g76*np.log(comp_g76+1e-9)).sum(1)).mean()/np.log(n_proto)),
        'GSE14520': float((-(comp_g14*np.log(comp_g14+1e-9)).sum(1)).mean()/np.log(n_proto))},
    'composite_risk_results': comp_results,
    'replication_counts': {
        'GSE14520_OS':rep_g14_os, 'GSE14520_RFS':rep_g14_rfs,
        'GSE76427_OS':rep_g76_os, 'GSE76427_RFS':rep_g76_rfs},
    'meta_significant_prototypes': int((meta.pooled_p<0.05).sum()),
    'meta_total_prototypes': int(len(meta)),
    'meta_top10': meta.head(10).to_dict('records'),
}, indent=2, default=str))
log('== T3j done ==')
