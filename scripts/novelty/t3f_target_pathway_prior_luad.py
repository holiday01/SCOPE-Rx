"""
T3f-LUAD — Drug-target pathway prior using TCGA-LUAD Cox.

Same logic as t3f_target_pathway_prior.py but anchored on TCGA-LUAD
(576 patients) and reading from results/t3d_luad/ + results/t3e_luad/.
"""
from __future__ import annotations
import json, time, re
from pathlib import Path
import numpy as np, pandas as pd
import gseapy as gp
from lifelines import CoxPHFitter

ROOT = Path('/home/holiday01/drug_sc')
RAW  = ROOT/'data/drug_sensitivity_raw'
TCGA_EXPR = Path('/mnt/10t/scrna_atac/data/raw/TCGA_LUAD/TCGA_LUAD_expression.gz')
TCGA_CLIN = Path('/mnt/10t/scrna_atac/data/raw/TCGA_LUAD/TCGA_LUAD_clinical.tsv')
T3D  = ROOT/'results/t3d_luad'
T3E  = ROOT/'results/t3e_luad'
OUT  = ROOT/'results/t3f_luad'; OUT.mkdir(parents=True, exist_ok=True)
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Pathway library ----------
log('Loading pathway libraries …')
H = gp.get_library('MSigDB_Hallmark_2020', organism='Human')
K = gp.get_library('KEGG_2021_Human', organism='Human')
R = gp.get_library('Reactome_2022', organism='Human')
pw_map = {f'H:{k}':set(v) for k,v in H.items()}
pw_map.update({f'K:{k}':set(v) for k,v in K.items()})
pw_map.update({f'R:{k}':set(v) for k,v in R.items()})
pw_map = {k:v for k,v in pw_map.items() if 10 <= len(v) <= 500}
log(f'Total pathways kept (10–500 genes): {len(pw_map)}')

# ---------- 2. Drug targets ----------
log('Loading PRISM drug targets …')
meta = pd.read_csv(RAW/'PRISM_secondary_AUC.csv',
                   usecols=['name','target','moa','phase'], low_memory=False).drop_duplicates('name')
meta['drug_lc'] = meta['name'].str.lower()
def parse_targets(t):
    if not isinstance(t, str) or not t.strip(): return set()
    return {x.strip().upper() for x in re.split(r'[,;/|]+', t) if x.strip()}
meta['target_set'] = meta['target'].apply(parse_targets)
drug_targets = dict(zip(meta['drug_lc'], meta['target_set']))
log(f'Drugs with ≥1 target: {sum(1 for v in drug_targets.values() if v)}/{len(drug_targets)}')

# ---------- 3. TCGA-LUAD expression ----------
log('Loading TCGA-LUAD expression …')
t = pd.read_csv(TCGA_EXPR, sep='\t', index_col=0, low_memory=False)
if t.values.max() < 30:
    t_lin = np.clip(np.expm1(t.values * np.log(2)), 0, None).astype(np.float32)
else:
    t_lin = t.values.astype(np.float32)
t_pp = pd.DataFrame(t_lin.T, index=t.columns, columns=t.index)
rs = t_pp.values.sum(1, keepdims=True) + 1e-6
Xp = np.log1p(t_pp.values / rs * 1e4).astype(np.float32)
gmu = Xp.mean(0); gsd = Xp.std(0) + 1e-6
Xpz = (Xp - gmu) / gsd
log(f'TCGA expr: patients={Xp.shape[0]}  genes={Xp.shape[1]}')
gene_to_i = {g:i for i,g in enumerate(t.index)}

# ---------- 4. Pathway activity per patient ----------
log('Computing pathway activity per patient (mean z of member genes) …')
pw_names=[]; pw_act=[]; pw_sizes=[]
for name, genes in pw_map.items():
    idx = [gene_to_i[g] for g in genes if g in gene_to_i]
    if len(idx) < 5: continue
    pw_names.append(name); pw_sizes.append(len(idx))
    pw_act.append(Xpz[:, idx].mean(axis=1))
pw_act = np.stack(pw_act, axis=1)
pw_df = pd.DataFrame(pw_act, index=t.columns, columns=pw_names)
log(f'Pathway activity matrix: {pw_df.shape}')

# ---------- 5. Cox per pathway ----------
log('Loading TCGA-LUAD clinical …')
clin = pd.read_csv(TCGA_CLIN, sep='\t', low_memory=False)
cols = {c.lower():c for c in clin.columns}
sid = cols.get('sampleid') or clin.columns[0]
vs = cols.get('vital_status'); dtd = cols.get('days_to_death'); dtl = cols.get('days_to_last_followup')
clin['event'] = clin[vs].astype(str).str.upper().map({'DEAD':1,'DECEASED':1,'ALIVE':0,'LIVING':0})
dtd_n = pd.to_numeric(clin[dtd], errors='coerce')
dtl_n = pd.to_numeric(clin[dtl], errors='coerce')
clin['time'] = dtd_n.where(clin['event']==1, dtl_n)
surv = clin[[sid,'event','time']].rename(columns={sid:'sample'}).dropna()
merged = pw_df.reset_index().rename(columns={'index':'sample'}).merge(surv, on='sample', how='inner')
merged = merged[merged['time']>0].copy()
log(f'Patients for Cox: {len(merged)}')

cox_rows=[]
for pw in pw_names:
    d = merged[['event','time',pw]].dropna()
    if len(d)<50 or d[pw].std()<1e-6: continue
    d = d.assign(x=(d[pw] - d[pw].mean()) / d[pw].std())
    try:
        cph = CoxPHFitter(penalizer=0.05).fit(d[['event','time','x']], 'time','event')
        cox_rows.append({'pathway':pw,
                         'HR':float(np.exp(cph.params_['x'])),
                         'p':float(cph.summary.loc['x','p']),
                         'n_genes':int(pw_sizes[pw_names.index(pw)])})
    except Exception:
        continue
cox_df = pd.DataFrame(cox_rows)
cox_df['logHR'] = np.log(cox_df['HR'])
cox_df['signed_nlp'] = -np.log10(cox_df['p'].clip(1e-300)) * np.sign(cox_df['logHR'])
cox_df.loc[cox_df['p'] > 0.1, 'signed_nlp'] = 0.0
sig = (cox_df['p']<0.05).sum()
log(f'Pathway Cox: {len(cox_df)} pathways tested, {sig} significant (p<0.05)')
top_pos = cox_df.sort_values('signed_nlp', ascending=False).head(8)
top_neg = cox_df.sort_values('signed_nlp').head(8)
log('Top hazard pathways (bad-prognosis):')
log(top_pos[['pathway','HR','p']].to_string(index=False))
log('Top protective pathways:')
log(top_neg[['pathway','HR','p']].to_string(index=False))
cox_df.to_parquet(OUT/'pathway_cox_tcga_luad.parquet', index=False)

# ---------- 6. Drug-target × pathway coverage ----------
log('Building drug-target × pathway coverage …')
pw_gene_sets = {name: pw_map[name] for name in cox_df['pathway']}
drug_lcs = list(drug_targets.keys())
mat = np.zeros((len(drug_lcs), len(pw_gene_sets)), dtype=np.float32)
pw_list = list(pw_gene_sets.keys())
pw_idx = {n:i for i,n in enumerate(pw_list)}
for i, dlc in enumerate(drug_lcs):
    tgt = drug_targets[dlc]
    if not tgt: continue
    for pn, pg in pw_gene_sets.items():
        inter = tgt & pg
        if inter:
            mat[i, pw_idx[pn]] = len(inter) / max(len(tgt), 1)
cov_df = pd.DataFrame(mat, index=drug_lcs, columns=pw_list)
log(f'Drug-pathway coverage: {cov_df.shape}  non-zero: {(mat>0).sum():,}')
cov_df.to_parquet(OUT/'drug_target_pathway_matrix.parquet')

# ---------- 7. Prior score ----------
log('Computing per-drug target-pathway prior …')
sig_mask = cox_df['p'] < 0.1
sig_pw = cox_df.loc[sig_mask, 'pathway'].tolist()
sig_nlp = cox_df.loc[sig_mask, 'signed_nlp'].values.astype(np.float32)
sig_idx = [pw_idx[p] for p in sig_pw]
sub = mat[:, sig_idx]
S_prior = sub @ sig_nlp
prior_df = pd.DataFrame({'drug_lc':drug_lcs, 'S_prior':S_prior})
prior_df['n_sig_pathways_hit'] = (sub != 0).sum(1)
prior_df.to_parquet(OUT/'drug_pathway_prior_score.parquet', index=False)
log(f'S_prior stats: {prior_df.S_prior.describe().round(2).to_dict()}')
log(f'Drugs hitting ≥1 hazard pathway: {(prior_df.S_prior>0).sum()}')

# ---------- 8. Fuse with T3e soft ----------
log('Fusing with T3e soft_combined …')
soft = pd.read_parquet(T3E/'drug_score_soft_combined.parquet')
soft['drug_lc'] = soft['drug'].str.lower()
final = soft.merge(prior_df, on='drug_lc', how='left').fillna({'S_prior':0.0,'n_sig_pathways_hit':0})
sp = final['S_prior'].values
sp_scale = np.percentile(np.abs(sp[sp!=0]), 95) if (sp!=0).any() else 1.0
final['S_prior_norm'] = np.clip(sp / max(sp_scale, 1e-6), -2.0, 2.0)
def zscale(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std()+1e-6)
final['z_kill'] = zscale(final['score'])
final['z_onc']  = zscale(final['onc_relevance'])
final['z_prior']= zscale(final['S_prior_norm'])
final['score_final'] = 1.0*final['z_kill'] + 0.5*final['z_onc'] + 0.7*final['z_prior']
final = final.sort_values('score_final', ascending=False)
final.to_parquet(OUT/'drug_final_score.parquet', index=False)

# ---------- 9. Report ----------
clinicals = ['erlotinib','gefitinib','afatinib','osimertinib','dacomitinib',
             'crizotinib','alectinib','ceritinib','lorlatinib','brigatinib',
             'pemetrexed','docetaxel','paclitaxel','carboplatin','cisplatin',
             'gemcitabine','vinorelbine','etoposide','trametinib','selumetinib']

lines=['# T3f-LUAD — Final ranking after Patch 1 (target × pathway prior)\n',
       '## Top-25 FINAL (z_kill + 0.5*z_onc + 0.7*z_prior)\n',
       '| # | Drug | Final | Kill | Onc | Prior | #HazPw hit | MOA | Target |',
       '|---:|---|---:|---:|---:|---:|---:|---|---|']
for i, r in enumerate(final.head(25).itertuples(), 1):
    lines.append(f"| {i} | **{r.drug}** | {r.score_final:+.2f} | {r.z_kill:+.2f} | {r.z_onc:+.2f} | {r.z_prior:+.2f} | {int(r.n_sig_pathways_hit)} | {r.moa or ''} | {r.target or ''} |")

lines += ['\n## Clinical LUAD drugs across stages\n',
          '| Drug | Raw (T3d) | Soft (T3e) | **Final (T3f)** |',
          '|---|---:|---:|---:|']
orig_sorted = pd.read_parquet(T3D/'drug_score_cohort.parquet').sort_values('score', ascending=False).reset_index(drop=True)
soft_sorted = soft.sort_values('score_combined', ascending=False).reset_index(drop=True)
fin_sorted  = final.reset_index(drop=True)
for dn in clinicals:
    h1 = orig_sorted[orig_sorted.drug.str.lower()==dn]
    h2 = soft_sorted[soft_sorted.drug.str.lower()==dn]
    h3 = fin_sorted[fin_sorted.drug.str.lower()==dn]
    r1 = int(h1.index[0])+1 if len(h1) else None
    r2 = int(h2.index[0])+1 if len(h2) else None
    r3 = int(h3.index[0])+1 if len(h3) else None
    lines.append(f"| {dn} | {r1 or '—'} | {r2 or '—'} | **{r3 or '—'}** |")

(OUT/'comparison_top20.md').write_text('\n'.join(lines))
log(f'Report: {OUT/"comparison_top20.md"}')

(OUT/'eval_metrics.json').write_text(json.dumps({
    'n_pathways_tested': int(len(cox_df)),
    'n_pathways_sig_p05': int(sig),
    'top_hazard_pathways': top_pos[['pathway','HR','p']].to_dict('records'),
    'top_protective_pathways': top_neg[['pathway','HR','p']].to_dict('records'),
    'n_drugs_hit_hazard_pathway': int((prior_df.S_prior>0).sum()),
    'final_top20': [{'rank':i+1,'drug':r.drug,'score_final':float(r.score_final),
                     'phase':r.phase,'moa':r.moa,'target':r.target}
                    for i,r in enumerate(final.head(20).itertuples())],
}, indent=2, default=str))
log('== T3f-LUAD done ==')
