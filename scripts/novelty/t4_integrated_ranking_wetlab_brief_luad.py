"""
T4-LUAD — Integrated ranking + per-drug wet-lab brief, LUAD edition.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path('/home/holiday01/drug_sc')
PROC = ROOT/'data/processed/luad_drug'
T3C  = ROOT/'results/t3c_luad'
T3D  = ROOT/'results/t3d_luad'
T3F  = ROOT/'results/t3f_luad'
OUT  = ROOT/'results/t4_luad'; OUT.mkdir(parents=True, exist_ok=True)
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

log('Loading T3f-LUAD final score …')
_df = pd.read_parquet(T3F/'drug_final_score.parquet')
drug_score = _df.drop(columns=['score']).rename(columns={'score_final':'score'})
log(f'  {len(drug_score)} drugs')

per_patient = pd.read_parquet(T3D/'drug_score_per_patient.parquet')
proto_meta  = pd.read_parquet(T3C/'prototype_meta.parquet')
proto_expr  = pd.read_parquet(T3C/'prototype_expression.parquet')
cox         = pd.read_parquet(T3C/'cox_per_prototype.parquet')
cat         = pd.read_parquet(PROC/'drug_catalog.parquet').set_index('drug')
luad_expr   = pd.read_parquet(PROC/'cellline_expression_luad.parquet')
luad_meta   = pd.read_parquet(PROC/'cellline_meta.parquet').set_index('ModelID')
genes = list(proto_expr.columns)
n_proto = len(proto_meta)

# Subpopulation DEGs
log('Subpopulation DEGs (filtered for real expression) …')
PB = proto_expr.values.astype(np.float32)
MIN_EXPR = np.log1p(20.0)
gmu = PB.mean(0, keepdims=True); gsd = PB.std(0, keepdims=True)+1e-6
Z = (PB - gmu)/gsd
gene_max = PB.max(0); global_ok = gene_max > MIN_EXPR
log(f'  Genes passing global filter: {global_ok.sum()}/{len(genes)}')
deg_rows = []
for p in range(n_proto):
    mask = global_ok & (PB[p] > MIN_EXPR)
    if mask.sum() < 10: mask = global_ok
    cand = np.where(mask)[0]
    order = cand[np.argsort(-Z[p, cand])[:20]]
    for rnk, g_i in enumerate(order):
        deg_rows.append({'proto':int(p),'rank':rnk+1,'gene':genes[g_i],
                         'zscore':float(Z[p,g_i]),'log1p_cp10k':float(PB[p,g_i])})
deg_df = pd.DataFrame(deg_rows)
deg_df.to_parquet(OUT/'subpopulation_markers.parquet', index=False)

# Closest LUAD cell line per prototype
log('Matching prototypes to closest LUAD cell line …')
Xluad = np.log1p(luad_expr.values.astype(np.float32))
Xluad_z = (Xluad - Xluad.mean(0)) / (Xluad.std(0)+1e-6)
proto_match = {}
for p in range(n_proto):
    z = Z[p]
    cors = [np.corrcoef(z, Xluad_z[i])[0,1] for i in range(Xluad_z.shape[0])]
    best = int(np.argmax(cors))
    proto_match[p] = {'best_model_id': luad_expr.index[best],
                      'best_cell_line': luad_meta.loc[luad_expr.index[best],'cell_line'],
                      'pearson': float(cors[best])}

# Per-patient top-5
log('Per-patient top-5 …')
drugs = list(per_patient.columns)
arr = per_patient.values
order = np.argsort(-arr, axis=1)[:, :5]
pp = []
for i, pid in enumerate(per_patient.index):
    for rnk in range(5):
        di = order[i, rnk]
        pp.append({'patient':pid,'rank':rnk+1,'drug':drugs[di],'score':float(arr[i,di])})
pd.DataFrame(pp).to_parquet(OUT/'per_patient_top5.parquet', index=False)

# Cohort top-20 wet-lab brief
log('Top-20 wet-lab brief …')
top20 = drug_score.head(20).copy()
pred_z = pd.read_parquet(T3D/'predicted_auc_zscore_per_prototype.parquet')
proto_trust = proto_meta['trust_to_depmap'].values

bad_mask = np.zeros(n_proto, dtype=bool)
for _, cr in cox.iterrows():
    if float(cr['p']) < 0.1 and float(cr['HR']) > 1.0:
        bad_mask[int(cr['proto'])] = True
log(f'  Bad-prognosis prototypes (Cox p<0.1 & HR>1): {bad_mask.sum()}')

rows = []
for rank_i, (_, r) in enumerate(top20.iterrows(), 1):
    d = r['drug']
    z = pred_z[d].values
    bad_idx = np.where(bad_mask)[0]
    if len(bad_idx)==0: bad_idx = np.arange(n_proto)
    priority = -z[bad_idx] * (0.3 + 0.7*proto_trust[bad_idx])
    order_p = bad_idx[np.argsort(-priority)[:3]]
    target_protos_str = ', '.join(
        f"#{p}({proto_meta.loc[p,'label']}, trust={proto_trust[p]:.2f}, Z={z[p]:+.2f})"
        for p in order_p)
    markers = ' / '.join(set().union(*[set(deg_df[deg_df['proto']==p]['gene'].head(5).tolist())
                                       for p in order_p]))
    matched_lines = ' / '.join(
        f"{proto_match[int(p)]['best_cell_line']} (r={proto_match[int(p)]['pearson']:.2f})"
        for p in order_p)
    info = cat.loc[d].to_dict() if d in cat.index else {}
    rows.append({'rank':rank_i,'drug':d,'score':float(r['score']),
                 'moa':info.get('moa'),'target':info.get('target'),
                 'phase':info.get('phase'),'smiles':info.get('smiles'),
                 'target_subpopulations':target_protos_str,
                 'facs_markers':markers,
                 'suggested_luad_lines':matched_lines})
brief = pd.DataFrame(rows)
brief.to_csv(OUT/'luad_top20_wetlab_brief.csv', index=False)

md = ['# LUAD Phase-1b wet-lab brief (SCOPE-Rx Top-20)', '',
      '_Ranked by T3f-LUAD final score across TCGA-LUAD (576 patients), validated in 3 external cohorts._',
      '']
for _, r in brief.iterrows():
    md += [f"## {int(r['rank']):>2}. {r['drug']}  — score {r['score']:+.2f}",
           f"- **MOA / target**: {r['moa']}  /  {r['target']}  (phase {r['phase']})",
           f"- **Target subpopulation(s)**: {r['target_subpopulations']}",
           f"- **FACS markers (top DEGs)**: {r['facs_markers']}",
           f"- **Suggested LUAD cell line(s)**: {r['suggested_luad_lines']}",
           f"- **SMILES**: `{r['smiles']}`",
           '']
(OUT/'luad_top20_wetlab_brief.md').write_text('\n'.join(md))
log(f'Markdown: {OUT/"luad_top20_wetlab_brief.md"}')

(OUT/'eval_metrics.json').write_text(json.dumps({
    'n_drugs': int(len(drug_score)),
    'n_patients': int(per_patient.shape[0]),
    'top20_cohort_drugs': brief['drug'].tolist(),
    'n_bad_prognosis_protos': int(bad_mask.sum()),
}, indent=2, default=str))
log('== T4-LUAD done ==')
