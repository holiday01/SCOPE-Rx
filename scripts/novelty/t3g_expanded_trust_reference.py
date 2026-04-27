"""
T3g — Patch 3: cell-type-matched trust reference (no download needed).

Insight: DepMap already contains ~350 blood / lymphoid / myeloid cell lines that
can serve as in-vitro references for TME cell states (macrophage, T, NK, B).
Previous trust = max correlation with ANY DepMap line penalised TME prototypes
unfairly. Now we compute trust against a cell-type-matched subset:

  Epithelial_tumor   → Liver HCC lines (25)
  M2 / M1 / CD14     → Myeloid + AML lines (~80)
  CD16 / Neutrophil  → Myeloid + AML
  CD4 / CD8 / Memory → T-ALL lines (~26)
  NK / NKT           → NK lines (4) + T-ALL (cross-reactive)
  Memory B / plasma  → B-cell neoplasm lines (~130)
  Fibroblast / VSMC  → Soft-tissue / mesenchymal
  Endothelial        → (no DepMap endothelial; fall back to full-panel max)
  DC                 → Myeloid + AML
  Stem/Progenitor    → AML

Writes back an updated prototype_meta.parquet, then re-runs trust-weighted
scoring in T3d → T3f pipeline for comparison.
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('/home/holiday01/drug_sc')
PROC = ROOT/'data/processed/hcc_drug'
T3C  = ROOT/'results/t3c'
OUT  = ROOT/'results/t3g'; OUT.mkdir(parents=True, exist_ok=True)

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Build cell-type → DepMap subset map ----------
log('Mapping HCC prototype dominant cell-type → DepMap line subset …')
mdl = pd.read_csv(ROOT/'data/drug_sensitivity_raw/Model.csv', low_memory=False)
panC = pd.read_parquet(PROC/'cellline_expression_panCancer.parquet')
have_expr = set(panC.index)

def subset_lines(condition_df):
    return [m for m in condition_df['ModelID'].tolist() if m in have_expr]

# Define subsets
def is_primary(disease):
    return mdl['OncotreePrimaryDisease'].isin(disease) if isinstance(disease, list) else (mdl['OncotreePrimaryDisease']==disease)
def is_lineage(lin):
    return mdl['OncotreeLineage'].isin(lin) if isinstance(lin, list) else (mdl['OncotreeLineage']==lin)
def is_subtype(sub):
    return mdl['OncotreeSubtype'].isin(sub) if isinstance(sub, list) else (mdl['OncotreeSubtype']==sub)

hcc = mdl[is_subtype(['Hepatocellular Carcinoma','Hepatoblastoma'])]
myeloid = mdl[(is_lineage('Myeloid')) | is_primary(['Acute Myeloid Leukemia',
               'AML with Myelodysplasia-Related Changes',
               'Acute Promyelocytic Leukemia',
               'Chronic Myeloid Leukemia, BCR-ABL1+',
               'Myeloproliferative Neoplasms',
               'Acute Leukemias of Ambiguous Lineage'])]
tcell = mdl[is_subtype(['T-Cell Acute Lymphoblastic Leukemia',
                        'Peripheral T-Cell Lymphoma, NOS',
                        'Adult T-Cell Leukemia/Lymphoma',
                        'Cutaneous T-Cell Lymphoma'])]
nk    = mdl[mdl['OncotreeSubtype'].astype(str).str.contains('Natural Killer', case=False, na=False)]
bcell = mdl[is_subtype(['B-Cell Acute Lymphoblastic Leukemia',
                        'Burkitt Lymphoma','Diffuse Large B-Cell Lymphoma, NOS',
                        'Mantle Cell Lymphoma','Marginal Zone Lymphoma',
                        'Plasma Cell Myeloma','Non-Hodgkin Lymphoma','Hodgkin Lymphoma'])]
fibro = mdl[is_primary(['Sarcoma, NOS','Fibrosarcoma','Synovial Sarcoma','Liposarcoma',
                        'Leiomyosarcoma','Undifferentiated Pleomorphic Sarcoma'])]

ref_map = {
    'Epithelial Cells'   : subset_lines(hcc),
    'Epithelial'         : subset_lines(hcc),
    'M2 macrophage'      : subset_lines(myeloid),
    'M1 macrophage'      : subset_lines(myeloid),
    'CD14 macrophage'    : subset_lines(myeloid),
    'CD16 macrophage'    : subset_lines(myeloid),
    'Neutrophil'         : subset_lines(myeloid),
    'monocyte'           : subset_lines(myeloid),
    'DC cell'            : subset_lines(myeloid),
    'cycling T cell'     : subset_lines(tcell),
    'CD4 cell'           : subset_lines(tcell),
    'CD8 T cell'         : subset_lines(tcell),
    'Memory T cell'      : subset_lines(tcell),
    'NK cell / NKT cell' : subset_lines(pd.concat([nk, tcell])),
    'Memory B cell'      : subset_lines(bcell),
    'Fibroblasts'        : subset_lines(fibro),
    'Vascular Smooth Muscle Cells': subset_lines(fibro),
    'Endothelial Cells'  : list(have_expr),   # fall back to full panel
    'Stem and Progenitor Cells': subset_lines(myeloid),
    'cancer cell'        : subset_lines(hcc),
    'Unassigned'         : list(have_expr),
}

for ct, lines in ref_map.items():
    log(f'  {ct:34s} → {len(lines)} DepMap lines')

# ---------- 2. Recompute trust per prototype ----------
log('\nRecomputing per-prototype trust against cell-type-matched reference …')
proto_meta = pd.read_parquet(T3C/'prototype_meta.parquet')
proto_expr = pd.read_parquet(T3C/'prototype_expression.parquet')
PB = proto_expr.values.astype(np.float32)
# cell-line side
Xcl = np.log1p(panC.values.astype(np.float32))
cl_mu = Xcl.mean(0); cl_sd = Xcl.std(0)+1e-6
Xcl_z = (Xcl - cl_mu) / cl_sd
pb_mu = PB.mean(0); pb_sd = PB.std(0)+1e-6
pb_z  = (PB - pb_mu) / pb_sd
pb_n = pb_z / (np.linalg.norm(pb_z, axis=1, keepdims=True)+1e-6)
cl_n = Xcl_z / (np.linalg.norm(Xcl_z, axis=1, keepdims=True)+1e-6)
# full matrix
cor_full = pb_n @ cl_n.T                         # (n_proto × n_cellline)

line_ids = list(panC.index)
line_idx = {m:i for i,m in enumerate(line_ids)}

trust_new = np.zeros(len(proto_meta), dtype=np.float32)
best_line_new = [None]*len(proto_meta)
best_line_old = proto_meta.get('best_cellline', pd.Series(['']*len(proto_meta))).tolist()
trust_old = proto_meta.get('trust_to_depmap', pd.Series([0.0]*len(proto_meta))).values

for i, row in proto_meta.iterrows():
    ct = row['dominant_cell_type']
    refs = ref_map.get(ct, list(have_expr))
    if not refs:
        refs = list(have_expr)
    ref_i = np.array([line_idx[m] for m in refs if m in line_idx])
    cor_sub = cor_full[i, ref_i]
    best = int(np.argmax(cor_sub))
    trust_new[i]   = float(cor_sub[best])
    best_line_new[i] = refs[best]

proto_meta['trust_old'] = trust_old
proto_meta['best_cellline_old'] = best_line_old
proto_meta['trust_to_depmap'] = trust_new
proto_meta['best_cellline'] = best_line_new
# attach human-readable best cell-line name
name_map = dict(zip(mdl['ModelID'], mdl['StrippedCellLineName']))
proto_meta['best_cellline_name'] = [name_map.get(m,'') for m in best_line_new]
proto_meta.to_parquet(T3C/'prototype_meta.parquet', index=False)  # overwrite in place

log('\nTrust before → after (mean by dominant cell type):')
summary = proto_meta.groupby('dominant_cell_type').agg(
    n=('proto','count'),
    trust_before=('trust_old','mean'),
    trust_after=('trust_to_depmap','mean')).round(3).sort_values('n', ascending=False)
print(summary.to_string())
summary.to_parquet(OUT/'trust_before_after_by_celltype.parquet')

# ---------- 3. Re-run T3d + T3f ----------
log('\nRe-running T3d (uses new trust_to_depmap) and T3f (re-fuses pathway prior) …')
import subprocess
PY = str(ROOT/'../miniconda/envs/drug_sc/bin/python')
PY = '/home/holiday01/miniconda/envs/drug_sc/bin/python'
for script in ['scripts/novelty/t3d_signature_reversal_cox.py',
               'scripts/novelty/t3e_oncology_filter.py',
               'scripts/novelty/t3f_target_pathway_prior.py',
               'scripts/novelty/t4_integrated_ranking_wetlab_brief.py']:
    log(f'  Running {script} …')
    r = subprocess.run([PY, script], cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        log(f'  FAIL: {r.stderr[-500:]}')
    else:
        # show last 4 lines
        tail = r.stdout.strip().split('\n')[-4:]
        for t in tail: log(f'    {t}')

(OUT/'eval_metrics.json').write_text(json.dumps({
    'trust_before_mean': float(proto_meta['trust_old'].mean()),
    'trust_after_mean': float(proto_meta['trust_to_depmap'].mean()),
    'n_prototypes_trust_gt_05_before': int((proto_meta['trust_old']>0.5).sum()),
    'n_prototypes_trust_gt_05_after':  int((proto_meta['trust_to_depmap']>0.5).sum()),
    'per_celltype': summary.reset_index().to_dict('records'),
}, indent=2, default=str))
log('== T3g done ==')
