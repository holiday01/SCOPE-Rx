"""
T3f — Drug-target pathway prior (Patch 1).

For each drug, use its PRISM target column to score a "target-pathway activity hazard":
  1. Build drug → set of HGNC gene symbols.
  2. Build pathway → set of genes (MSigDB Hallmark + KEGG 2021 + Reactome 2022).
  3. For each pathway, compute a GSVA-lite activity per TCGA-LIHC patient
     = mean z-score of pathway member genes on patient log1p-CP10K.
  4. Fit Cox(OS) on each pathway activity → (HR, p).
     Z-score the activity column first so HRs are per-SD (numerically stable).
  5. Per drug: S_prior(d) =
          Σ_pathway   target_coverage(d, pathway) × trust_pw × (-log10(p_pw)) × sign(logHR_pw)
     where target_coverage = fraction of drug targets that appear in the pathway;
     trust_pw = 1 if Cox p<0.1, else 0.
     Rationale: if a drug's targets are enriched in a hazard-driving pathway, reward it.
  6. Final ranking: combine with T3e soft_combined via weighted sum.

Outputs:
  results/t3f/
    pathway_cox_tcga_lihc.parquet   (pathway, HR, p, n_genes)
    drug_target_pathway_matrix.parquet  (drug × pathway coverage)
    drug_pathway_prior_score.parquet    (drug, S_prior)
    drug_final_score.parquet            (drug, S_kill + S_onc + S_prior fused)
    comparison_top20.md
"""
from __future__ import annotations
import json, math, time, re
from pathlib import Path
import numpy as np, pandas as pd
import gseapy as gp
from lifelines import CoxPHFitter

ROOT = Path('/home/holiday01/drug_sc')
RAW  = ROOT/'data/drug_sensitivity_raw'
PROC = ROOT/'data/processed/hcc_drug'
TCGA_EXPR = ROOT/'data/TCGA_LIHC/TCGA_LIHC_expression.gz'
TCGA_CLIN = ROOT/'data/TCGA_LIHC/TCGA_LIHC_clinical.tsv'
T3D  = ROOT/'results/t3d'
T3E  = ROOT/'results/t3e'
OUT  = ROOT/'results/t3f'; OUT.mkdir(parents=True, exist_ok=True)
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Pathway library ----------
log('Loading pathway libraries …')
H = gp.get_library('MSigDB_Hallmark_2020', organism='Human')
K = gp.get_library('KEGG_2021_Human', organism='Human')
R = gp.get_library('Reactome_2022', organism='Human')
pw_map = {f'H:{k}':set(v) for k,v in H.items()}
pw_map.update({f'K:{k}':set(v) for k,v in K.items()})
pw_map.update({f'R:{k}':set(v) for k,v in R.items()})
# Filter tiny / huge pathways
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
n_drugs_with_tgt = sum(1 for v in drug_targets.values() if v)
log(f'Drugs with ≥1 target: {n_drugs_with_tgt}/{len(drug_targets)}')

# ---------- 3. TCGA-LIHC expression (gene × patient log1p) ----------
log('Loading TCGA-LIHC expression …')
t = pd.read_csv(TCGA_EXPR, sep='\t', index_col=0, low_memory=False)
if t.values.max() < 30:
    t_lin = np.clip(np.expm1(t.values * np.log(2)), 0, None).astype(np.float32)
else:
    t_lin = t.values.astype(np.float32)
# genes × patients -> patients × genes
t_pp = pd.DataFrame(t_lin.T, index=t.columns, columns=t.index)
# CP10K-log
rs = t_pp.values.sum(1, keepdims=True) + 1e-6
Xp = np.log1p(t_pp.values / rs * 1e4).astype(np.float32)
# per-gene z-score across patients
gmu = Xp.mean(0); gsd = Xp.std(0) + 1e-6
Xpz = (Xp - gmu) / gsd
log(f'TCGA expr: patients={Xp.shape[0]}  genes={Xp.shape[1]}')
gene_to_i = {g:i for i,g in enumerate(t.index)}

# ---------- 4. Pathway activity per patient (GSVA-lite) ----------
log('Computing pathway activity per patient (mean z of member genes) …')
pw_names = []; pw_act = []; pw_sizes=[]
for name, genes in pw_map.items():
    idx = [gene_to_i[g] for g in genes if g in gene_to_i]
    if len(idx) < 5: continue
    pw_names.append(name); pw_sizes.append(len(idx))
    pw_act.append(Xpz[:, idx].mean(axis=1))
pw_act = np.stack(pw_act, axis=1)   # (patients × pathways)
pw_df = pd.DataFrame(pw_act, index=t.columns, columns=pw_names)
log(f'Pathway activity matrix: {pw_df.shape}')

# ---------- 5. Cox per pathway ----------
log('Loading TCGA-LIHC clinical …')
clin = pd.read_csv(TCGA_CLIN, sep='\t', low_memory=False)
cols = {c.lower():c for c in clin.columns}
sid = cols.get('sampleid') or clin.columns[0]
vs = cols.get('vital_status'); dtd = cols.get('days_to_death'); dtl = cols.get('days_to_last_followup')
clin['event'] = clin[vs].str.upper().map({'DEAD':1,'DECEASED':1,'ALIVE':0,'LIVING':0})
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
    # z-score to stabilise
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
log(f'Pathway Cox done: {len(cox_df)} pathways tested, {sig} significant (p<0.05)')
top_pos = cox_df.sort_values('signed_nlp', ascending=False).head(8)
top_neg = cox_df.sort_values('signed_nlp').head(8)
log('Top hazard-driving pathways (bad-prognosis):')
log(top_pos[['pathway','HR','p']].to_string(index=False))
log('Top protective pathways:')
log(top_neg[['pathway','HR','p']].to_string(index=False))
cox_df.to_parquet(OUT/'pathway_cox_tcga_lihc.parquet', index=False)

# ---------- 6. Drug-target × pathway coverage matrix ----------
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
            mat[i, pw_idx[pn]] = len(inter) / max(len(tgt), 1)  # fraction of drug targets in pathway
cov_df = pd.DataFrame(mat, index=drug_lcs, columns=pw_list)
log(f'Drug-pathway coverage matrix: {cov_df.shape}  non-zero cells: {(mat>0).sum():,}')
cov_df.to_parquet(OUT/'drug_target_pathway_matrix.parquet')

# ---------- 7. Prior score per drug ----------
log('Computing per-drug target-pathway prior score …')
# Only consider significant pathways (p<0.1), signed by logHR direction
sig_mask = cox_df['p'] < 0.1
sig_pw = cox_df.loc[sig_mask, 'pathway'].tolist()
sig_nlp = cox_df.loc[sig_mask, 'signed_nlp'].values.astype(np.float32)
sig_idx = [pw_idx[p] for p in sig_pw]
sub = mat[:, sig_idx]                # (n_drugs × n_sig_pathways)
S_prior = sub @ sig_nlp               # higher = drug targets hazard-driving pathways
prior_df = pd.DataFrame({'drug_lc':drug_lcs, 'S_prior':S_prior})
prior_df['n_sig_pathways_hit'] = (sub != 0).sum(1)
prior_df.to_parquet(OUT/'drug_pathway_prior_score.parquet', index=False)
log(f'S_prior stats: {prior_df.S_prior.describe().round(2).to_dict()}')
log(f'Drugs hitting ≥1 hazard pathway: {(prior_df.S_prior>0).sum()}')

# ---------- 8. Fuse with T3e soft score ----------
log('Fusing with T3e soft_combined score …')
soft = pd.read_parquet(T3E/'drug_score_soft_combined.parquet')
soft['drug_lc'] = soft['drug'].str.lower()
final = soft.merge(prior_df, on='drug_lc', how='left').fillna({'S_prior':0.0,'n_sig_pathways_hit':0})
# normalise S_prior to roughly [-1,1]
sp = final['S_prior'].values
sp_scale = np.percentile(np.abs(sp[sp!=0]), 95) if (sp!=0).any() else 1.0
final['S_prior_norm'] = np.clip(sp / max(sp_scale, 1e-6), -2.0, 2.0)

# final score = 0.5 * soft_combined + 0.5 * (raw score ranked) + weight * prior
# Simpler: additive on a z-scaled basis
def zscale(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std()+1e-6)
final['z_kill'] = zscale(final['score'])         # original kill score (T3d)
final['z_onc']  = zscale(final['onc_relevance']) # oncology relevance
final['z_prior']= zscale(final['S_prior_norm'])  # pathway prior
# weighted sum
final['score_final'] = 1.0*final['z_kill'] + 0.5*final['z_onc'] + 0.7*final['z_prior']
final = final.sort_values('score_final', ascending=False)
final.to_parquet(OUT/'drug_final_score.parquet', index=False)

# ---------- 9. Report ----------
clinicals = ['sorafenib','lenvatinib','regorafenib','cabozantinib','sunitinib',
             'gemcitabine','doxorubicin','paclitaxel','cisplatin','oxaliplatin',
             '5-fluorouracil','erlotinib','lapatinib','afatinib','vincristine']
lines=[]
lines.append('# T3f — Final ranking after Patch 1 (drug-target pathway prior)\n')
lines.append('## Top-20 FINAL (z_kill + 0.5*z_onc + 0.7*z_prior)\n')
lines.append('| # | Drug | Final | Kill | Onc | Prior | #HazPw hit | MOA | Target |')
lines.append('|---:|---|---:|---:|---:|---:|---:|---|---|')
for i, r in enumerate(final.head(25).itertuples(), 1):
    lines.append(f"| {i} | **{r.drug}** | {r.score_final:+.2f} | {r.z_kill:+.2f} | {r.z_onc:+.2f} | {r.z_prior:+.2f} | {int(r.n_sig_pathways_hit)} | {r.moa or ''} | {r.target or ''} |")

lines.append('\n## Clinical HCC drug ranking across all stages\n')
lines.append('| Drug | Raw (T3d) | Soft (T3e) | **Final (T3f)** |')
lines.append('|---|---:|---:|---:|')
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
    lines.append(f"| {dn} | {r1} | {r2} | **{r3}** |")

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
log('== T3f done ==')
