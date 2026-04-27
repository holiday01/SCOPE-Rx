"""
T2a — build aligned HCC drug-response dataset for baselines.

Outputs (all in data/processed/hcc_drug/):
  cellline_meta.parquet       27 liver cell lines (HCC + hepatoblastoma) metadata
  cellline_expression.parquet cell-line × gene TPM (genes intersected with scRNA HVG)
  drug_response_long.parquet  (cell_line, drug, source, auc, ic50) long table
  drug_catalog.parquet        drug-level info with SMILES + MOA + target when available
  gene_universe.parquet       gene symbols used (intersection bulk vs scRNA)
"""
from __future__ import annotations
import pandas as pd, numpy as np
from pathlib import Path
import re, sys, time

ROOT = Path('/home/holiday01/drug_sc')
RAW  = ROOT / 'data/drug_sensitivity_raw'
SCR  = ROOT / 'data/scRNA_HCC_integrated/GSE223204GSE202642GSE162616_bbknn_tumornormal.h5ad'
OUT  = ROOT / 'data/processed/hcc_drug'
OUT.mkdir(parents=True, exist_ok=True)

def log(msg): print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

# --- 1. HCC cell-line roster ---
mdl = pd.read_csv(RAW/'Model.csv', low_memory=False)
liver = mdl[mdl['OncotreeSubtype'].isin(['Hepatocellular Carcinoma','Hepatoblastoma'])].copy()
liver = liver[['ModelID','StrippedCellLineName','OncotreeSubtype','OncotreePrimaryDisease','SangerModelID','COSMICID','CCLEName']]
liver.columns = ['ModelID','cell_line','subtype','primary_disease','sanger_id','cosmic_id','ccle_name']
log(f'Liver cell lines: {len(liver)} (HCC={ (liver.subtype=="Hepatocellular Carcinoma").sum() }, Hepatoblastoma={ (liver.subtype=="Hepatoblastoma").sum() })')
liver.to_parquet(OUT/'cellline_meta.parquet', index=False)

# --- 2. Cell-line expression (all lines, we'll subset later) ---
log('Reading OmicsExpressionTPM.csv …')
expr = pd.read_csv(RAW/'OmicsExpressionTPM.csv', index_col=0)
# rename "GENE (entrez)" -> "GENE"
expr.columns = [re.sub(r'\s*\(\d+\)$','',c) for c in expr.columns]
log(f'DepMap expression: {expr.shape[0]} lines × {expr.shape[1]} genes')
liver_expr = expr.loc[expr.index.intersection(liver['ModelID'])]
log(f'Liver lines with expression: {len(liver_expr)}')

# --- 3. scRNA gene universe (to intersect) ---
import anndata as ad
a = ad.read_h5ad(SCR, backed='r')
sc_genes = list(a.var_names)
a.file.close()
log(f'scRNA genes: {len(sc_genes)}')

common = sorted(set(expr.columns) & set(sc_genes))
log(f'Shared genes bulk ∩ scRNA: {len(common)}')
gu = pd.DataFrame({'gene':common})
gu.to_parquet(OUT/'gene_universe.parquet', index=False)

# keep all liver lines with shared genes (for training we use bulk lines pan-cancer
# but project them onto the shared axis for scRNA transfer)
liver_expr_shared = liver_expr[common]
liver_expr_shared.to_parquet(OUT/'cellline_expression_liver.parquet')
log(f'Saved liver expression: {liver_expr_shared.shape}')

# also export pan-cancer expression on the shared gene set for scDEAL training
expr_shared = expr[common]
expr_shared.to_parquet(OUT/'cellline_expression_panCancer.parquet')
log(f'Saved pan-cancer expression: {expr_shared.shape}')

# --- 4. Drug response (GDSC1+2 via sanger_dose_response, plus PRISM) ---
log('Loading sanger_dose_response …')
sdr = pd.read_csv(RAW/'sanger_dose_response.csv', low_memory=False)
sdr = sdr.rename(columns={'ARXSPAN_ID':'ModelID','DRUG_NAME':'drug','auc':'auc','log2.ic50':'log2_ic50','DATASET':'source','BROAD_ID':'broad_id'})
sdr = sdr[['source','ModelID','drug','broad_id','auc','log2_ic50','IC50_PUBLISHED','AUC_PUBLISHED']]
sdr = sdr.dropna(subset=['ModelID','drug'])
log(f'GDSC1+2 rows: {len(sdr)}  drugs: {sdr.drug.nunique()}  lines: {sdr.ModelID.nunique()}')

log('Loading PRISM …')
prism = pd.read_csv(RAW/'PRISM_secondary_AUC.csv', usecols=['depmap_id','broad_id','name','auc','ic50','moa','target','smiles','phase'])
prism = prism.rename(columns={'depmap_id':'ModelID','name':'drug'})
prism['source'] = 'PRISM'
log(f'PRISM rows: {len(prism)}  drugs: {prism.drug.nunique()}  lines: {prism.ModelID.nunique()}')

# unified long table
long_gdsc = sdr.copy()
long_gdsc['ic50'] = long_gdsc['log2_ic50']  # retain log2 scale from sanger
long_gdsc['moa']=np.nan; long_gdsc['target']=np.nan; long_gdsc['smiles']=np.nan; long_gdsc['phase']=np.nan
long_gdsc = long_gdsc[['source','ModelID','drug','broad_id','auc','ic50','moa','target','smiles','phase']]
long_prism = prism[['source','ModelID','drug','broad_id','auc','ic50','moa','target','smiles','phase']]
long_all = pd.concat([long_gdsc, long_prism], ignore_index=True)
log(f'Unified drug_response_long rows: {len(long_all)}')
long_all.to_parquet(OUT/'drug_response_long.parquet', index=False)

# --- 5. HCC slice ---
hcc_ids = set(liver['ModelID'])
hcc_dr = long_all[long_all['ModelID'].isin(hcc_ids)]
log(f'HCC drug-response rows: {len(hcc_dr)}')
log(f'  per-source: {hcc_dr.source.value_counts().to_dict()}')
log(f'  unique HCC lines with any drug: {hcc_dr.ModelID.nunique()}')
log(f'  unique drugs on HCC lines: {hcc_dr.drug.nunique()}')

# drug catalog (prefer PRISM entries because they carry SMILES/MOA)
cat = (long_all.sort_values(['smiles','moa'], na_position='last')
       .drop_duplicates('drug', keep='first')[['drug','broad_id','moa','target','smiles','phase']])
log(f'Drug catalog rows: {len(cat)}  with SMILES: {cat.smiles.notna().sum()}')
cat.to_parquet(OUT/'drug_catalog.parquet', index=False)

# --- 6. AUC matrices (drug × HCC line) ---
pivot_gdsc = (long_gdsc[long_gdsc['ModelID'].isin(hcc_ids)]
              .groupby(['drug','ModelID'])['auc'].mean().unstack('ModelID'))
pivot_prism = (long_prism[long_prism['ModelID'].isin(hcc_ids)]
               .groupby(['drug','ModelID'])['auc'].mean().unstack('ModelID'))
log(f'GDSC HCC AUC matrix: {pivot_gdsc.shape} (drugs × lines)  density={pivot_gdsc.notna().mean().mean():.2f}')
log(f'PRISM HCC AUC matrix: {pivot_prism.shape}  density={pivot_prism.notna().mean().mean():.2f}')
pivot_gdsc.to_parquet(OUT/'auc_matrix_gdsc_hcc.parquet')
pivot_prism.to_parquet(OUT/'auc_matrix_prism_hcc.parquet')

log('== T2a done ==')
print('\nOutputs:')
for p in sorted(OUT.glob('*.parquet')):
    print(f'  {p.name:42s} {p.stat().st_size/1024/1024:7.1f} MB')
