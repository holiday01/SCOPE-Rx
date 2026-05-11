"""
T2a-LUAD — build aligned LUAD drug-response dataset for SCOPE-Rx Phase-1b.

Mirrors t2a_prepare_hcc_drug_data.py but:
  - Tumor scope: OncotreeSubtype ∈ {Lung Adenocarcinoma, Non-Small Cell Lung Cancer}
    (LUSC and SCLC excluded — different transcriptional / clinical entity)
  - scRNA reference: /mnt/10t/scrna_atac/data/processed/LUAD/luad_scrna_annotated.h5ad
  - Output: data/processed/luad_drug/
"""
from __future__ import annotations
import pandas as pd, numpy as np
from pathlib import Path
import re, time

ROOT = Path('/home/holiday01/drug_sc')
RAW  = ROOT / 'data/drug_sensitivity_raw'
SCR  = Path('/mnt/10t/scrna_atac/data/processed/LUAD/luad_scrna_annotated.h5ad')
OUT  = ROOT / 'data/processed/luad_drug'
OUT.mkdir(parents=True, exist_ok=True)

def log(msg): print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

# --- 1. LUAD cell-line roster ---
mdl = pd.read_csv(RAW/'Model.csv', low_memory=False)
LUAD_SUBTYPES = ['Lung Adenocarcinoma', 'Non-Small Cell Lung Cancer']
luad = mdl[mdl['OncotreeSubtype'].isin(LUAD_SUBTYPES)].copy()
luad = luad[['ModelID','StrippedCellLineName','OncotreeSubtype','OncotreePrimaryDisease','SangerModelID','COSMICID','CCLEName']]
luad.columns = ['ModelID','cell_line','subtype','primary_disease','sanger_id','cosmic_id','ccle_name']
log(f'LUAD/NSCLC cell lines: {len(luad)} '
    f'(LUAD={(luad.subtype=="Lung Adenocarcinoma").sum()}, '
    f'NSCLC-NOS={(luad.subtype=="Non-Small Cell Lung Cancer").sum()})')
luad.to_parquet(OUT/'cellline_meta.parquet', index=False)

# --- 2. Cell-line expression ---
log('Reading OmicsExpressionTPM.csv …')
expr = pd.read_csv(RAW/'OmicsExpressionTPM.csv', index_col=0)
expr.columns = [re.sub(r'\s*\(\d+\)$','',c) for c in expr.columns]
log(f'DepMap expression: {expr.shape[0]} lines × {expr.shape[1]} genes')
luad_expr = expr.loc[expr.index.intersection(luad['ModelID'])]
log(f'LUAD lines with expression: {len(luad_expr)}')

# --- 3. scRNA gene universe ---
import anndata as ad
a = ad.read_h5ad(SCR, backed='r')
sc_genes = list(a.var_names)
n_cells, n_genes = a.shape
celltypes = a.obs['celltype'].value_counts().to_dict() if 'celltype' in a.obs.columns else {}
a.file.close()
log(f'scRNA: {n_cells} cells × {n_genes} genes; celltypes: {celltypes}')

common = sorted(set(expr.columns) & set(sc_genes))
log(f'Shared genes bulk ∩ scRNA: {len(common)}')
pd.DataFrame({'gene': common}).to_parquet(OUT/'gene_universe.parquet', index=False)

luad_expr_shared = luad_expr[common]
luad_expr_shared.to_parquet(OUT/'cellline_expression_luad.parquet')
log(f'Saved LUAD expression: {luad_expr_shared.shape}')

expr_shared = expr[common]
expr_shared.to_parquet(OUT/'cellline_expression_panCancer.parquet')
log(f'Saved pan-cancer expression: {expr_shared.shape}')

# --- 4. Drug response (GDSC + PRISM) ---
log('Loading sanger_dose_response …')
sdr = pd.read_csv(RAW/'sanger_dose_response.csv', low_memory=False)
sdr = sdr.rename(columns={'ARXSPAN_ID':'ModelID','DRUG_NAME':'drug','auc':'auc',
                          'log2.ic50':'log2_ic50','DATASET':'source','BROAD_ID':'broad_id'})
sdr = sdr[['source','ModelID','drug','broad_id','auc','log2_ic50','IC50_PUBLISHED','AUC_PUBLISHED']]
sdr = sdr.dropna(subset=['ModelID','drug'])
log(f'GDSC1+2 rows: {len(sdr)}  drugs: {sdr.drug.nunique()}  lines: {sdr.ModelID.nunique()}')

log('Loading PRISM …')
prism = pd.read_csv(RAW/'PRISM_secondary_AUC.csv',
                    usecols=['depmap_id','broad_id','name','auc','ic50','moa','target','smiles','phase'])
prism = prism.rename(columns={'depmap_id':'ModelID','name':'drug'})
prism['source'] = 'PRISM'
log(f'PRISM rows: {len(prism)}  drugs: {prism.drug.nunique()}  lines: {prism.ModelID.nunique()}')

long_gdsc = sdr.copy()
long_gdsc['ic50'] = long_gdsc['log2_ic50']
long_gdsc['moa']=np.nan; long_gdsc['target']=np.nan; long_gdsc['smiles']=np.nan; long_gdsc['phase']=np.nan
long_gdsc = long_gdsc[['source','ModelID','drug','broad_id','auc','ic50','moa','target','smiles','phase']]
long_prism = prism[['source','ModelID','drug','broad_id','auc','ic50','moa','target','smiles','phase']]
long_all = pd.concat([long_gdsc, long_prism], ignore_index=True)
log(f'Unified drug_response_long rows: {len(long_all)}')
long_all.to_parquet(OUT/'drug_response_long.parquet', index=False)

# --- 5. LUAD slice ---
luad_ids = set(luad['ModelID'])
luad_dr = long_all[long_all['ModelID'].isin(luad_ids)]
log(f'LUAD drug-response rows: {len(luad_dr)}')
log(f'  per-source: {luad_dr.source.value_counts().to_dict()}')
log(f'  unique LUAD lines with any drug: {luad_dr.ModelID.nunique()}')
log(f'  unique drugs on LUAD lines: {luad_dr.drug.nunique()}')

cat = (long_all.sort_values(['smiles','moa'], na_position='last')
       .drop_duplicates('drug', keep='first')[['drug','broad_id','moa','target','smiles','phase']])
log(f'Drug catalog rows: {len(cat)}  with SMILES: {cat.smiles.notna().sum()}')
cat.to_parquet(OUT/'drug_catalog.parquet', index=False)

# --- 6. AUC matrices (drug × LUAD line) ---
pivot_gdsc = (long_gdsc[long_gdsc['ModelID'].isin(luad_ids)]
              .groupby(['drug','ModelID'])['auc'].mean().unstack('ModelID'))
pivot_prism = (long_prism[long_prism['ModelID'].isin(luad_ids)]
               .groupby(['drug','ModelID'])['auc'].mean().unstack('ModelID'))
log(f'GDSC LUAD AUC matrix: {pivot_gdsc.shape} (drugs × lines)  '
    f'density={pivot_gdsc.notna().mean().mean():.2f}')
log(f'PRISM LUAD AUC matrix: {pivot_prism.shape}  '
    f'density={pivot_prism.notna().mean().mean():.2f}')
pivot_gdsc.to_parquet(OUT/'auc_matrix_gdsc_luad.parquet')
pivot_prism.to_parquet(OUT/'auc_matrix_prism_luad.parquet')

log('== T2a-LUAD done ==')
print('\nOutputs:')
for p in sorted(OUT.glob('*.parquet')):
    print(f'  {p.name:42s} {p.stat().st_size/1024/1024:7.1f} MB')
