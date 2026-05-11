"""
T3j-LUAD — 4-cohort fixed-effect meta-analysis.

Inputs (already computed):
  results/t3c_luad/cox_per_prototype.parquet                — TCGA-LUAD (n=563, univariate)
  results/t3i_luad/{GSE68465,GSE72094,GSE31210}_cox_os.parquet — adjusted

Output:
  results/t3j_luad/meta_analysis_pooled.parquet
  results/t3j_luad/comparison_meta_luad.md
  results/t3j_luad/eval_metrics.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm

ROOT = Path('/home/holiday01/drug_sc')
T3C  = ROOT/'results/t3c_luad'
T3I  = ROOT/'results/t3i_luad'
OUT  = ROOT/'results/t3j_luad'; OUT.mkdir(parents=True, exist_ok=True)
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

log('Loading per-cohort Cox tables …')
tcga = pd.read_parquet(T3C/'cox_per_prototype.parquet')[['proto','HR','p','n']]
tcga['cohort']='TCGA_LUAD'
g68 = pd.read_parquet(T3I/'GSE68465_cox_os.parquet'); g68['cohort']='GSE68465'
g72 = pd.read_parquet(T3I/'GSE72094_cox_os.parquet'); g72['cohort']='GSE72094'
g31 = pd.read_parquet(T3I/'GSE31210_cox_os.parquet'); g31['cohort']='GSE31210'
agg = pd.concat([tcga, g68[['proto','HR','p','n','cohort']],
                       g72[['proto','HR','p','n','cohort']],
                       g31[['proto','HR','p','n','cohort']]], ignore_index=True)
log(f'Total Cox rows across 4 cohorts: {len(agg)}')
log(agg.groupby('cohort')['proto'].count().to_dict())

agg['logHR'] = np.log(agg['HR'].astype(float).clip(lower=1e-6, upper=1e6))
agg['z']  = norm.ppf(1 - agg['p'].clip(lower=1e-300, upper=0.999999) / 2)
agg['SE'] = np.where(np.abs(agg['z'])>1e-6, np.abs(agg['logHR'])/np.abs(agg['z']), np.nan)
agg['w']  = 1.0 / (agg['SE']**2)

# Fixed-effect pooled per prototype
meta_rows = []
for p in sorted(agg['proto'].unique()):
    sub = agg[agg['proto']==p].dropna(subset=['logHR','SE'])
    if len(sub) < 2: continue
    if sub['w'].sum() < 1e-9: continue
    w = sub['w'].values; lh = sub['logHR'].values
    pooled_lh = (lh*w).sum() / w.sum()
    pooled_se = (1.0/w.sum())**0.5
    pooled_z  = pooled_lh / pooled_se
    pooled_p  = 2*(1 - norm.cdf(abs(pooled_z)))
    Q = (w*(lh - pooled_lh)**2).sum()
    df = len(sub)-1
    I2 = max(0.0, (Q-df)/Q)*100 if Q>0 else 0.0
    meta_rows.append({'proto':int(p),'n_cohorts':int(len(sub)),
                      'pooled_logHR':float(pooled_lh),
                      'pooled_HR':float(np.exp(pooled_lh)),
                      'pooled_SE':float(pooled_se),
                      'pooled_p':float(pooled_p),
                      'Q':float(Q),'df':int(df),'I2_pct':float(I2),
                      'cohorts': ';'.join(sub['cohort'].tolist())})
meta = pd.DataFrame(meta_rows).sort_values('pooled_p')

proto_meta = pd.read_parquet(T3C/'prototype_meta.parquet').set_index('proto')
meta = meta.join(proto_meta[['dominant_cell_type','dominant_sample_type','label','trust_to_depmap']], on='proto')
meta.to_parquet(OUT/'meta_analysis_pooled.parquet', index=False)

n_meta = len(meta)
sig_05  = int((meta.pooled_p<0.05).sum())
sig_01  = int((meta.pooled_p<0.01).sum())
sig_001 = int((meta.pooled_p<0.001).sum())
log(f'\nPrototypes pooled (≥2 cohorts): {n_meta}')
log(f'  pooled p<0.05:  {sig_05}')
log(f'  pooled p<0.01:  {sig_01}')
log(f'  pooled p<0.001: {sig_001}')
log(f'\nMean I^2 (heterogeneity): {meta.I2_pct.mean():.1f}%')
log(f'\nTop 15 meta-pooled prognostic prototypes:')
print(meta.head(15)[['proto','dominant_cell_type','dominant_sample_type','pooled_HR','pooled_p','I2_pct','n_cohorts']].to_string(index=False))

# TCGA-significant replication
tcga_sig_protos = set(tcga[tcga['p']<0.05]['proto'].astype(int))
n_tcga_sig = len(tcga_sig_protos)
log(f'\nTCGA-significant prototypes (p<0.05): {n_tcga_sig}')

def replication_count(ext_df, threshold=0.1):
    rep = 0
    for p in tcga_sig_protos:
        sub_t = tcga[tcga['proto']==p].iloc[0]
        sub_e = ext_df[ext_df['proto']==p]
        if len(sub_e)==0: continue
        if (np.sign(np.log(sub_e['HR'].iloc[0]))==np.sign(np.log(sub_t['HR']))) and (sub_e['p'].iloc[0]<threshold):
            rep += 1
    return rep

rep68 = replication_count(g68)
rep72 = replication_count(g72)
rep31 = replication_count(g31)
log(f'\nTCGA-sig {n_tcga_sig} prototypes — replication (same direction, p<0.1):')
log(f'  GSE68465 OS: {rep68}/{n_tcga_sig}')
log(f'  GSE72094 OS: {rep72}/{n_tcga_sig}')
log(f'  GSE31210 OS: {rep31}/{n_tcga_sig}')

# at least 1 external rep
def any_external_rep(p, threshold=0.1):
    for ext in [g68, g72, g31]:
        sub = ext[ext['proto']==p]
        sub_t = tcga[tcga['proto']==p].iloc[0]
        if len(sub)==0: continue
        if (np.sign(np.log(sub['HR'].iloc[0]))==np.sign(np.log(sub_t['HR']))) and (sub['p'].iloc[0]<threshold):
            return True
    return False
n_any = sum(any_external_rep(p) for p in tcga_sig_protos)
log(f'  ≥1 external endpoint: {n_any}/{n_tcga_sig}')

# Report
lines = ['# T3j-LUAD — 4-cohort fixed-effect meta-analysis\n',
         '## Cohorts',
         '- TCGA-LUAD (training, n≈563)',
         '- GSE68465 Director\'s Challenge (n=442)',
         '- GSE72094 Schabath/Moffitt (n=398)',
         '- GSE31210 Okayama (n=204, stage I-II)',
         '',
         f'## Prototype-level meta-pool (≥2 cohorts):',
         f'- {n_meta} prototypes pooled',
         f'- {sig_05} p<0.05  |  {sig_01} p<0.01  |  {sig_001} p<0.001',
         f'- mean I² heterogeneity: {meta.I2_pct.mean():.1f}%',
         '',
         '## TCGA-significant prototype replication\n',
         f'- TCGA-sig protos (p<0.05): **{n_tcga_sig}**',
         '',
         '| External cohort | Replicated (same direction, p<0.1) |',
         '|---|---:|',
         f'| GSE68465 OS  | {rep68}/{n_tcga_sig} ({rep68/n_tcga_sig:.0%}) |',
         f'| GSE72094 OS  | {rep72}/{n_tcga_sig} ({rep72/n_tcga_sig:.0%}) |',
         f'| GSE31210 OS  | {rep31}/{n_tcga_sig} ({rep31/n_tcga_sig:.0%}) |',
         f'| **≥1 external endpoint** | **{n_any}/{n_tcga_sig} ({n_any/n_tcga_sig:.0%})** |',
         '',
         '## Top-15 meta-pooled prognostic prototypes\n',
         '| Proto | Dominant cell type | T/N | Pooled HR | Pooled p | I² | n_cohorts |',
         '|---:|---|---|---:|---:|---:|---:|']
for _, r in meta.head(15).iterrows():
    lines.append(f"| {int(r['proto'])} | {r['dominant_cell_type']} | {r['dominant_sample_type']} | "
                 f"{r['pooled_HR']:.2f} | {r['pooled_p']:.3g} | {r['I2_pct']:.0f}% | {int(r['n_cohorts'])} |")
(OUT/'comparison_meta_luad.md').write_text('\n'.join(lines))
log(f'Report: {OUT/"comparison_meta_luad.md"}')

(OUT/'eval_metrics.json').write_text(json.dumps({
    'n_pooled_prototypes': n_meta,
    'pooled_p05': sig_05, 'pooled_p01': sig_01, 'pooled_p001': sig_001,
    'mean_I2_pct': float(meta.I2_pct.mean()),
    'tcga_sig_prototypes': n_tcga_sig,
    'replication_GSE68465_OS': rep68,
    'replication_GSE72094_OS': rep72,
    'replication_GSE31210_OS': rep31,
    'replication_any_external': n_any,
    'top10_meta_protos': meta.head(10).to_dict('records'),
}, indent=2, default=str))
log('== T3j-LUAD done ==')
