"""
T4 — Integrated ranking + per-drug wet-lab brief.

Combines:
  - T3d survival-anchored drug score (seen drugs via scDEAL)
  - T3a MPNN unseen-drug extrapolation (optional augmentation)
  - Per-prototype DEG marker genes (for FACS gating)
  - HCC cell-line matching (closest DepMap line per prototype)
  - PRISM moa/target/phase/smiles

Outputs:
  results/t4/
    hcc_top20_wetlab_brief.md              human-readable for lab partner
    hcc_top20_wetlab_brief.csv             same info machine-readable
    per_patient_top5.parquet               per-TCGA-patient top-5 drug list
    subpopulation_markers.parquet          top DEGs per prognosis-driving prototype
"""
from __future__ import annotations
import json, time, math
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path('/home/holiday01/drug_sc')
PROC = ROOT/'data/processed/hcc_drug'
T3C  = ROOT/'results/t3c'
T3D  = ROOT/'results/t3d'
T3F  = ROOT/'results/t3f'
OUT  = ROOT/'results/t4'; OUT.mkdir(parents=True, exist_ok=True)
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- Load everything — prefer T3f final score if available ----------
log('Loading final scoring (T3f if present, else T3d) …')
if (T3F/'drug_final_score.parquet').exists():
    _df = pd.read_parquet(T3F/'drug_final_score.parquet')
    drug_score = _df.drop(columns=['score']).rename(columns={'score_final':'score'})
    log(f'  using T3f final score (z_kill+0.5*z_onc+0.7*z_prior); {len(drug_score)} drugs')
else:
    drug_score = pd.read_parquet(T3D/'drug_score_cohort.parquet')
    log(f'  using T3d raw score; {len(drug_score)} drugs')
consensus  = pd.read_parquet(T3D/'drug_consensus_rank.parquet')
per_patient = pd.read_parquet(T3D/'drug_score_per_patient.parquet')
top_proto   = pd.read_parquet(T3D/'top_drugs_per_bad_prognosis_prototype.parquet')
proto_meta  = pd.read_parquet(T3C/'prototype_meta.parquet')
proto_expr  = pd.read_parquet(T3C/'prototype_expression.parquet')
comp        = pd.read_parquet(T3C/'tcga_composition.parquet')
cox         = pd.read_parquet(T3C/'cox_per_prototype.parquet')
cat         = pd.read_parquet(PROC/'drug_catalog.parquet').set_index('drug')
liver_expr  = pd.read_parquet(PROC/'cellline_expression_liver.parquet')
liver_meta  = pd.read_parquet(PROC/'cellline_meta.parquet').set_index('ModelID')
genes = list(proto_expr.columns)

# ---------- Subpopulation DEGs (with min-expression filter) ----------
log('Computing top DEGs per prototype (filtered for real expression) …')
PB = proto_expr.values.astype(np.float32)     # 55 × 17460 on log1p(CP10K) scale
# Filter 1: gene must have a meaningful expression (≥ log1p(50) ≈ 3.9) in at least one prototype
MIN_EXPR = np.log1p(20.0)   # ≥20 CP10K in the prototype where it's used
# Filter 2: require the prototype's own expression to exceed MIN_EXPR so we don't highlight near-zero noise
gmu = PB.mean(0, keepdims=True); gsd = PB.std(0, keepdims=True) + 1e-6
Z = (PB - gmu) / gsd
# Mask genes with low expression in the focal prototype OR very low max across panel
gene_max = PB.max(0)
global_ok = gene_max > MIN_EXPR
log(f'  Genes passing global min-expression filter: {global_ok.sum()}/{len(genes)}')
deg_rows=[]
for p in range(len(proto_meta)):
    mask = global_ok & (PB[p] > MIN_EXPR)
    if mask.sum() < 10:   # fall back to global_ok if too few
        mask = global_ok
    candidate_idx = np.where(mask)[0]
    order = candidate_idx[np.argsort(-Z[p, candidate_idx])[:20]]
    for rnk, g_i in enumerate(order):
        deg_rows.append({'proto':int(p),'rank':rnk+1,'gene':genes[g_i],
                         'zscore':float(Z[p,g_i]),
                         'log1p_cp10k':float(PB[p, g_i])})
deg_df = pd.DataFrame(deg_rows)
deg_df.to_parquet(OUT/'subpopulation_markers.parquet', index=False)
marker_summary = deg_df[deg_df['rank']<=5].groupby('proto')['gene'].apply(lambda s: ', '.join(s)).to_dict()

# ---------- Closest HCC cell line per prototype (Pearson on log-TPM) ----------
log('Matching prototypes to closest HCC cell line …')
Xliv = np.log1p(liver_expr.values.astype(np.float32))  # already log-safe
Xliv_z = (Xliv - Xliv.mean(0)) / (Xliv.std(0)+1e-6)
proto_match = {}
for p in range(len(proto_meta)):
    z = Z[p]
    # cell-line side uses its own z-score; compute correlation
    cors = []
    for i in range(Xliv_z.shape[0]):
        cors.append(np.corrcoef(z, Xliv_z[i])[0,1])
    best = int(np.argmax(cors))
    proto_match[p] = {'best_model_id': liver_expr.index[best],
                      'best_cell_line': liver_meta.loc[liver_expr.index[best],'cell_line'],
                      'pearson': float(cors[best])}

# ---------- Per-patient top-5 ----------
log('Per-patient top-5 drug list …')
drugs = list(per_patient.columns)
arr = per_patient.values
order = np.argsort(-arr, axis=1)[:, :5]
pp=[]
for i, pid in enumerate(per_patient.index):
    for rnk in range(5):
        di = order[i, rnk]
        pp.append({'patient':pid, 'rank':rnk+1, 'drug':drugs[di], 'score':float(arr[i,di])})
pd.DataFrame(pp).to_parquet(OUT/'per_patient_top5.parquet', index=False)

# ---------- Cohort top-20 wet-lab brief ----------
log('Assembling wet-lab brief for top-20 cohort drugs …')
top20 = drug_score.head(20).copy()
# which prototypes each drug best kills (most negative pred_auc_z)
pred_z = pd.read_parquet(T3D/'predicted_auc_zscore_per_prototype.parquet')
rows=[]
# Precompute bad-prognosis prototype set (and trust-adjusted priority) once
bad_mask = np.zeros(len(proto_meta), dtype=bool)
proto_trust = proto_meta['trust_to_depmap'].values if 'trust_to_depmap' in proto_meta.columns else np.ones(len(proto_meta))
for _, cr in cox.iterrows():
    if float(cr['p']) < 0.1 and np.log(float(cr['HR']).__gt__(0) or float(cr['HR'])) > 0:
        bad_mask[int(cr['proto'])] = True
for _, r in top20.iterrows():
    d = r['drug']
    z = pred_z[d].values
    bad_idx = np.where(bad_mask)[0]
    if len(bad_idx)==0: bad_idx = np.arange(len(proto_meta))
    # prefer bad-prognosis prototypes where trust_to_depmap is also high (reliable AUC prediction)
    priority = -z[bad_idx] * (0.3 + 0.7 * proto_trust[bad_idx])   # weight by trust
    order_p = bad_idx[np.argsort(-priority)[:3]]
    target_protos_str = ', '.join(
        f"#{p}({proto_meta.loc[p,'label']}, trust={proto_trust[p]:.2f}, Z={z[p]:+.2f})"
        for p in order_p)
    markers = ' / '.join(set().union(*[set(deg_df[deg_df['proto']==p]['gene'].head(5).tolist()) for p in order_p]))
    matched_lines = []
    for p in order_p:
        m = proto_match[int(p)]
        matched_lines.append(f"{m['best_cell_line']} (r={m['pearson']:.2f})")
    info = cat.loc[d].to_dict() if d in cat.index else {}
    rows.append({
        'rank': int(_+1) if isinstance(_,int) else None,
        'drug': d,
        'score': float(r['score']),
        'moa': info.get('moa'),
        'target': info.get('target'),
        'phase': info.get('phase'),
        'smiles': info.get('smiles'),
        'target_subpopulations': target_protos_str,
        'facs_markers': markers,
        'suggested_hcc_lines': ' / '.join(matched_lines),
        'mean_predAUC_bad_protos': float(r.get('mean_predAUC_bad_protos', np.nan)),
    })
brief = pd.DataFrame(rows)
# replace _ column from iteration with real rank
brief['rank'] = np.arange(1, len(brief)+1)
brief.to_csv(OUT/'hcc_top20_wetlab_brief.csv', index=False)
log('Top-20 brief saved.')

# ---------- Human-readable markdown ----------
md = ['# HCC Phase-1 wet-lab brief (SCOPE-Rx Top-20)', '',
      '_Ranked by survival-anchored score across TCGA-LIHC (423 patients) using T3a-d pipeline._',
      '',
      'For each drug below: why the pipeline chose it, which HCC subpopulation is the likely target, which markers the wet lab can use to FACS-sort that subpopulation, and which HCC cell line is the closest in vitro match.',
      '']
for _, r in brief.iterrows():
    md.append(f"## {int(r['rank']):>2}. {r['drug']}  — score {r['score']:.2f}")
    md.append(f"- **MOA / target**: {r['moa']}  /  {r['target']}  (phase {r['phase']})")
    md.append(f"- **Target subpopulation(s)**: {r['target_subpopulations']}")
    md.append(f"- **FACS markers (top DEGs of those subpops)**: {r['facs_markers']}")
    md.append(f"- **Suggested HCC cell line(s)**: {r['suggested_hcc_lines']}")
    md.append(f"- **Mean predicted AUC on bad-prognosis prototypes**: {r['mean_predAUC_bad_protos']:.2f}  (< 0.5 = likely strong kill)")
    md.append(f"- **SMILES**: `{r['smiles']}`")
    md.append('')
(OUT/'hcc_top20_wetlab_brief.md').write_text('\n'.join(md))
log(f'Markdown written: {OUT/"hcc_top20_wetlab_brief.md"}')

# ---------- Summary ----------
(OUT/'eval_metrics.json').write_text(json.dumps({
    'n_drugs': int(len(drug_score)),
    'n_patients': int(per_patient.shape[0]),
    'top20_cohort_drugs': brief['drug'].tolist(),
    'prototypes_used_for_facs': sorted(set(deg_df.loc[deg_df['proto'].isin(
        cox.loc[(cox['p']<0.1) & (np.log(cox['HR'].clip(lower=1e-60))>0),'proto']),
        'proto'].tolist())),
}, indent=2, default=str))
log('== T4 done ==')
