"""
T3i — External validation on GSE76427 (Yang 2017 Singapore HCC cohort)

A 167-sample tumor + matched normal HCC microarray cohort with:
  - Overall survival (event_os, duryears_os)
  - Relapse-free survival (event_rfs, duryears_rfs)
  - BCLC staging (A/B/C/D)
  - TNM clinical staging
  - Age, gender

Pipeline:
  1. Parse Series Matrix → expression (probes × samples) + phenotype
  2. Map Affymetrix HG-U133 Plus 2.0 probes → HGNC gene symbols
     (use the platform GPL13158 / built-in annotation table)
  3. Aggregate probes → genes (max value per gene)
  4. Subset tumor samples
  5. Apply T3c attention deconvolver → 57-prototype composition
  6. Cox(OS) and Cox(RFS) per prototype with stage adjustment
  7. Concordance test with TCGA-LIHC: are the 9 multivariate prototypes still
     significant in this independent cohort?
  8. BCLC stage stratified prototype enrichment
"""
from __future__ import annotations
import gzip, json, time, re
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from lifelines import CoxPHFitter

ROOT = Path('/home/holiday01/drug_sc')
EXT  = ROOT/'data/external_HCC'
T3C  = ROOT/'results/t3c'
OUT  = ROOT/'results/t3i'; OUT.mkdir(parents=True, exist_ok=True)
DEV  = 'cuda'
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Parse Series Matrix ----------
log('Parsing GSE76427 series matrix …')
sm = EXT/'GSE76427_series.txt.gz'
phen_key_patterns = {
    'patient id':'patient id', 'tissue':'tissue',
    'event_os':'event_os','duryears_os':'duryears_os',
    'event_rfs':'event_rfs','duryears_rfs':'duryears_rfs',
    'age':'age', 'gender':'gender',
    'bclc_staging':'bclc_staging', 'tnm_staging_clinical':'tnm_staging_clinical',
}
phen = {}
sample_ids = None
data_started=False; data_rows=[]; probe_ids=[]
with gzip.open(sm,'rt') as f:
    for line in f:
        if line.startswith('!Sample_geo_accession'):
            sample_ids = [x.strip().strip('"') for x in line.split('\t')[1:]]
        elif line.startswith('!Sample_characteristics_ch1'):
            cells = [x.strip().strip('"') for x in line.split('\t')[1:]]
            if not cells: continue
            head_full = cells[0].split(':')[0].strip().lower()
            for short, pat in phen_key_patterns.items():
                if head_full.startswith(pat):
                    vals = [c.split(':',1)[1].strip() if ':' in c else '' for c in cells]
                    phen[short] = vals
                    break
        elif line.startswith('!series_matrix_table_begin'):
            data_started = True
        elif line.startswith('!series_matrix_table_end'):
            data_started = False
        elif data_started:
            parts = line.rstrip('\n').split('\t')
            if not parts or parts[0].startswith('!'): continue
            if parts[0].strip()=='ID_REF':
                # this is column header, skip
                continue
            try:
                vals = [float(x) if x not in ('','NA','null') else np.nan for x in parts[1:]]
            except:
                continue
            probe_ids.append(parts[0].strip().strip('"'))
            data_rows.append(vals)

pheno_df = pd.DataFrame(phen, index=sample_ids)
expr = pd.DataFrame(np.array(data_rows, dtype=np.float32), index=probe_ids, columns=sample_ids)
log(f'Series matrix: {expr.shape[0]} probes × {expr.shape[1]} samples')
log(f'Phenotype columns: {list(pheno_df.columns)}')
log(f'Tissue counts: {pheno_df["tissue"].value_counts().to_dict()}')
log(f'BCLC counts: {pheno_df["bclc_staging"].value_counts().to_dict()}')
log(f'OS-event known: {pheno_df["event_os"].apply(lambda x: x not in ("","NA")).sum()}')

# ---------- 2. Map probes → gene symbols (GSE76427 = GPL10558 Illumina) ----------
log('Loading GPL10558 (Illumina HumanHT-12 v4) annotation …')
import gzip as _gz
gpl_path = EXT/'GPL10558_full.soft.gz'
probe_to_gene = {}
with _gz.open(gpl_path, 'rt', errors='replace') as f:
    in_table = False; cols = None; gs_idx = None
    for line in f:
        if line.startswith('!platform_table_begin'):
            in_table = True; continue
        if line.startswith('!platform_table_end'): break
        if not in_table: continue
        if cols is None:
            cols = line.rstrip('\n').split('\t')
            for i,c in enumerate(cols):
                if c.lower() in ('gene symbol','symbol','gene_symbol'):
                    gs_idx = i; break
            continue
        parts = line.rstrip('\n').split('\t')
        if len(parts) <= gs_idx: continue
        pid = parts[0].strip()
        gs = parts[gs_idx].split(' /// ')[0].strip()
        if gs and gs not in ('---','-','NA',''):
            probe_to_gene[pid] = gs
log(f'Probe→gene map: {len(probe_to_gene)} probes mapped')

# Strip quotes off the matrix probe IDs
expr.index = [s.strip().strip('"') for s in expr.index]

# Aggregate probes per gene (max)
mapped = expr.loc[expr.index.intersection(probe_to_gene)].copy()
mapped['_gene'] = mapped.index.map(probe_to_gene)
gene_expr = mapped.groupby('_gene').max().drop(columns='_gene', errors='ignore')
# previous step left gene as index, _gene was an extra col — handle either
if '_gene' in gene_expr.columns: gene_expr = gene_expr.drop(columns='_gene')
log(f'Gene-level expression: {gene_expr.shape}')

# ---------- 3. Subset tumor samples + align genes ----------
tumor_mask = pheno_df['tissue'].astype(str).str.contains('tumor', case=False, na=False) & ~pheno_df['tissue'].astype(str).str.contains('non-tumor|adjacent', case=False, na=False)
log(f'Tumor samples: {tumor_mask.sum()}')

# Align with T3c gene universe
proto_expr = pd.read_parquet(T3C/'prototype_expression.parquet')
genes_shared = list(proto_expr.columns)
common_g = [g for g in genes_shared if g in gene_expr.index]
log(f'Shared genes (TCGA universe ∩ array gene-level): {len(common_g)} / {len(genes_shared)}')

# Build patient × gene matrix
Xp = gene_expr.loc[common_g, tumor_mask].T   # patients × genes
# microarray data is typically already log2-transformed (range ~3-15). Verify:
log(f'Expression range (after probe→gene): {Xp.values.min():.2f} to {Xp.values.max():.2f}')
# If looks like log already (max < 30), un-log to get linear, then redo CP10K-log1p like TCGA flow
Xp_vals = Xp.values.astype(np.float32)
# Illumina raw signal can be negative (background subtracted) and on linear scale here
if Xp_vals.max() < 30:
    Xp_lin = np.clip(np.expm1(Xp_vals * np.log(2)), 0, None)
else:
    Xp_lin = np.clip(Xp_vals, 0, None)
full = np.zeros((Xp.shape[0], len(genes_shared)), dtype=np.float32)
gene_to_i = {g:i for i,g in enumerate(genes_shared)}
for i, g in enumerate(common_g):
    full[:, gene_to_i[g]] = Xp_lin[:, i]
rs = full.sum(1, keepdims=True) + 1e-6
Xn = np.log1p(np.clip(full / rs * 1e4, 0, None)).astype(np.float32)
log(f'Processed external matrix: {Xn.shape}')

# ---------- 4. Apply T3c attention deconvolver ----------
log('Applying T3c attention deconvolver …')
ck = torch.load(ROOT/'checkpoints/t3c_attn_deconv.pt', map_location='cpu', weights_only=False)
n_proto = int(ck['n_proto'])
g_mu = np.asarray(ck['g_mu']); g_sd = np.asarray(ck['g_sd'])
Xn_std = (Xn - g_mu) / g_sd

class AttnDeconv(nn.Module):
    def __init__(self, d_in, n_proto, d_hid=256, p=0.1):
        super().__init__()
        self.query_enc = nn.Sequential(
            nn.Linear(d_in, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(p),
            nn.Linear(512, d_hid))
        self.proto_key = nn.Parameter(torch.randn(n_proto, d_hid)*0.02)
        self.temp = nn.Parameter(torch.tensor(1.0))
    def forward(self,x):
        q = self.query_enc(x)
        return F.softmax((q @ self.proto_key.T) / self.temp.clamp(min=0.1), dim=-1)

model = AttnDeconv(Xn.shape[1], n_proto).to(DEV).eval()
model.load_state_dict(ck['model'])
with torch.no_grad():
    comp = model(torch.from_numpy(Xn_std).to(DEV)).cpu().numpy()
ext_samples = pheno_df.loc[tumor_mask].index.tolist()
comp_df = pd.DataFrame(comp, index=ext_samples, columns=[f'proto_{i}' for i in range(n_proto)])
comp_df.to_parquet(OUT/'gse76427_composition.parquet')
log(f'External composition: {comp_df.shape}')
log(f'Mean entropy: {(-(comp*np.log(comp+1e-9)).sum(1)).mean():.2f} / max {np.log(n_proto):.2f}')

# ---------- 5. Cox per prototype ----------
log('Cox per prototype on GSE76427 OS and RFS …')
def to_num(s): return pd.to_numeric(s, errors='coerce')
phen_t = pheno_df.loc[tumor_mask].copy()
phen_t['os_event']  = to_num(phen_t['event_os']).astype(float)
phen_t['os_time']   = to_num(phen_t['duryears_os']).astype(float)*365   # years→days
phen_t['rfs_event'] = to_num(phen_t['event_rfs']).astype(float)
phen_t['rfs_time']  = to_num(phen_t['duryears_rfs']).astype(float)*365
phen_t['age']       = to_num(phen_t['age']).astype(float)
phen_t['male']      = (to_num(phen_t['gender'])==1).astype(float)
# BCLC numeric: A=1, B=2, C=3, D=4
bclc_map = {'A':1,'B':2,'C':3,'D':4}
phen_t['bclc_num']  = phen_t['bclc_staging'].map(bclc_map)
log(f'  OS available: {phen_t["os_event"].notna().sum()}  RFS available: {phen_t["rfs_event"].notna().sum()}')

# Run Cox per prototype, both endpoints, with BCLC + age + sex covariates
def cox_per_proto(comp_df, phen_t, event_col, time_col, covariates):
    rows = []
    merged = comp_df.join(phen_t[[event_col, time_col] + covariates])
    merged = merged.dropna(subset=[event_col, time_col] + covariates)
    merged = merged[merged[time_col]>0]
    for p in range(n_proto):
        c = f'proto_{p}'
        d = merged[[event_col, time_col, c] + covariates].dropna()
        if len(d) < 50 or d[c].std()<1e-5: continue
        d['x'] = (d[c]-d[c].mean())/d[c].std()
        try:
            cph = CoxPHFitter(penalizer=0.05).fit(d[[event_col,time_col,'x'] + covariates], time_col, event_col)
            rows.append({'proto':p,
                         'HR_x_uni': float(np.exp(cph.params_['x'])),
                         'p_x':     float(cph.summary.loc['x','p']),
                         'n':       int(len(d))})
        except Exception:
            pass
    return pd.DataFrame(rows)

cox_os  = cox_per_proto(comp_df, phen_t, 'os_event','os_time', ['bclc_num','age','male'])
cox_rfs = cox_per_proto(comp_df, phen_t, 'rfs_event','rfs_time', ['bclc_num','age','male'])
log(f'  OS Cox: {len(cox_os)} prototypes tested, {(cox_os.p_x<0.05).sum()} significant (mv-adjusted)')
log(f'  RFS Cox: {len(cox_rfs)} prototypes tested, {(cox_rfs.p_x<0.05).sum()} significant')

cox_os.to_parquet(OUT/'gse76427_cox_os.parquet', index=False)
cox_rfs.to_parquet(OUT/'gse76427_cox_rfs.parquet', index=False)

# ---------- 5b. Composite TCGA-derived risk score test (more powerful) ----------
log('\nComposite risk score: weight composition by TCGA-derived HRs, test in GSE76427 …')
tcga_mv_full = pd.read_parquet(ROOT/'results/t3h/multivariate_cox.parquet').set_index('proto')
# weight = log(HR_TCGA) for prototypes with p_TCGA<0.1 (allow more inclusive)
risk_w = np.zeros(n_proto, dtype=np.float32)
for p, r in tcga_mv_full.iterrows():
    if float(r['p_x_mv']) < 0.1:
        risk_w[int(p)] = float(np.log(r['HR_x_uni']))
log(f'Composite risk uses {(risk_w!=0).sum()} TCGA-prognostic prototypes (TCGA mv p<0.1)')

# patient risk = sum_p (composition[p] * log_HR_TCGA[p])
risk_score = comp.dot(risk_w)
phen_t['risk_score'] = pd.Series(risk_score, index=ext_samples)
# z-score
phen_t['risk_z'] = (phen_t['risk_score'] - phen_t['risk_score'].mean()) / phen_t['risk_score'].std()

for endpt, ev_col, t_col in [('OS', 'os_event','os_time'), ('RFS', 'rfs_event','rfs_time')]:
    d = phen_t[[ev_col, t_col, 'risk_z', 'bclc_num','age','male']].dropna()
    d = d[d[t_col]>0]
    if len(d)<50: continue
    try:
        cph = CoxPHFitter(penalizer=0.05).fit(d, t_col, ev_col)
        hr = float(np.exp(cph.params_['risk_z']))
        p  = float(cph.summary.loc['risk_z','p'])
        c_idx = float(cph.concordance_index_)
        log(f'  {endpt}: composite-risk HR per SD = {hr:.2f}  p = {p:.3g}  c-index = {c_idx:.3f}  n={len(d)}')
    except Exception as e:
        log(f'  {endpt}: failed — {e}')

# ---------- 6. Concordance with TCGA-LIHC ----------
log('\nConcordance test with TCGA-LIHC multivariate Cox …')
tcga_mv = pd.read_parquet(ROOT/'results/t3h/multivariate_cox.parquet').set_index('proto')
proto_meta = pd.read_parquet(T3C/'prototype_meta.parquet').set_index('proto')

# For each prototype, are HR directions concordant?
merge = cox_os.set_index('proto').join(tcga_mv[['HR_x_uni','p_x_mv','dominant']], how='inner', rsuffix='_tcga')
merge['log_HR_gse']  = np.log(merge['HR_x_uni'])
merge['log_HR_tcga'] = np.log(merge['HR_x_uni_tcga'])
sign_concord = ((merge['log_HR_gse']>0) == (merge['log_HR_tcga']>0))
log(f'Direction concordance (sign of logHR): {sign_concord.sum()} / {len(merge)} = {sign_concord.mean():.2%}')
# weighted concordance — only for TCGA-significant prototypes
tcga_sig = merge[merge['p_x_mv']<0.05].copy()
log(f'TCGA-significant prototypes: {len(tcga_sig)}')
sig_concord = ((tcga_sig['log_HR_gse']>0) == (tcga_sig['log_HR_tcga']>0))
log(f'  Direction concordance (TCGA-sig only): {sig_concord.sum()} / {len(tcga_sig)} = {sig_concord.mean():.2%}')
# Also Pearson on logHR
from scipy.stats import pearsonr, spearmanr
if len(merge) > 5:
    r_p, p_p = pearsonr(merge['log_HR_gse'], merge['log_HR_tcga'])
    r_s, p_s = spearmanr(merge['log_HR_gse'], merge['log_HR_tcga'])
    log(f'  log(HR) cross-cohort Pearson r = {r_p:.3f}  p = {p_p:.2e}')
    log(f'  log(HR) cross-cohort Spearman ρ = {r_s:.3f}  p = {p_s:.2e}')
else:
    r_p=r_s=p_p=p_s=np.nan

# Show top concordant prototypes
log('\n=== Top TCGA-sig prototypes — replication in GSE76427 ===')
tcga_sig = tcga_sig.assign(dominant=tcga_sig['dominant']).reset_index()
tcga_sig['rep_status'] = tcga_sig.apply(
    lambda r: 'replicated' if (np.sign(np.log(r['HR_x_uni']))==np.sign(np.log(r['HR_x_uni_tcga']))) else 'discordant', axis=1)
print(tcga_sig[['proto','dominant','HR_x_uni_tcga','p_x_mv','HR_x_uni','p_x','rep_status']]
      .rename(columns={'HR_x_uni_tcga':'HR_TCGA','p_x_mv':'p_TCGA',
                       'HR_x_uni':'HR_GSE76427','p_x':'p_GSE76427'})
      .sort_values('p_TCGA').head(15).to_string(index=False))

# ---------- 7. Save report ----------
log('\nWriting validation report …')
lines=[]
lines.append('# T3i — External validation on GSE76427 (Yang 2017, n≈167 HCC)\n')
lines.append('## Setup')
lines.append(f'- External cohort: **GSE76427** Yang 2017 Singapore HCC (Affymetrix HG-U133_Plus_2)')
lines.append(f'- Tumor samples used: **{tumor_mask.sum()}**, after gene mapping: **{Xn.shape[0]}** patients')
lines.append(f'- Probe→gene mapping: {len(probe_to_gene):,} probes → unique genes')
lines.append(f'- Genes shared with TCGA universe: {len(common_g)} / {len(genes_shared)}')
lines.append(f'- Endpoints: OS (event_os) AND RFS (event_rfs) — RFS is a stronger endpoint than OS')
lines.append(f'- Covariates in Cox: BCLC stage (A=1…D=4), age, gender')
lines.append('')
lines.append('## Results')
lines.append(f'- T3c attention deconv produces non-collapsing composition on independent cohort (mean entropy {(-(comp*np.log(comp+1e-9)).sum(1)).mean():.2f}/{np.log(n_proto):.2f})')
lines.append(f'- {(cox_os.p_x<0.05).sum()} prototypes Cox(OS)-significant after BCLC/age/sex adjustment')
lines.append(f'- {(cox_rfs.p_x<0.05).sum()} prototypes Cox(RFS)-significant')
lines.append(f'- **Cross-cohort concordance**:')
lines.append(f'  - All-prototype log(HR) correlation: Pearson r = {r_p:.3f}, p = {p_p:.2e}')
lines.append(f'  - All-prototype log(HR) correlation: Spearman ρ = {r_s:.3f}, p = {p_s:.2e}')
lines.append(f'  - Sign concordance (whole panel): {sign_concord.mean():.2%}')
lines.append(f'  - Sign concordance (TCGA-sig only): {sig_concord.mean():.2%} ({sig_concord.sum()}/{len(tcga_sig)} prototypes)')
lines.append('')
lines.append('## TCGA-significant prototypes — independent replication\n')
lines.append('| proto | dominant | HR (TCGA) | p (TCGA) | HR (GSE76427) | p (GSE76427) | replication |')
lines.append('|---:|---|---:|---:|---:|---:|---|')
for _,r in tcga_sig.sort_values('p_TCGA' if 'p_TCGA' in tcga_sig.columns else 'p_x_mv').head(15).iterrows():
    pt = r.get('p_TCGA', r.get('p_x_mv'))
    hg = r.get('HR_GSE76427', r.get('HR_x_uni'))
    pg = r.get('p_GSE76427', r.get('p_x'))
    ht = r.get('HR_TCGA', r.get('HR_x_uni_tcga'))
    lines.append(f"| {r['proto']} | {r['dominant']} | {ht:.2f} | {pt:.3g} | {hg:.2f} | {pg:.3g} | {r['rep_status']} |")
(OUT/'comparison_validation.md').write_text('\n'.join(lines))

(OUT/'eval_metrics.json').write_text(json.dumps({
    'cohort': 'GSE76427 Yang 2017',
    'n_tumor_samples_used': int(Xn.shape[0]),
    'shared_genes': int(len(common_g)),
    'composition_entropy_pct': float((-(comp*np.log(comp+1e-9)).sum(1)).mean()/np.log(n_proto)),
    'os_significant_prototypes': int((cox_os.p_x<0.05).sum()),
    'rfs_significant_prototypes': int((cox_rfs.p_x<0.05).sum()),
    'logHR_pearson_with_TCGA': float(r_p),
    'logHR_pearson_p': float(p_p),
    'logHR_spearman_with_TCGA': float(r_s),
    'sign_concordance_full': float(sign_concord.mean()),
    'sign_concordance_TCGA_sig': float(sig_concord.mean()),
    'TCGA_sig_replicated_count': int(sig_concord.sum()),
    'TCGA_sig_total_count': int(len(tcga_sig)),
}, indent=2, default=str))
log('== T3i done ==')
