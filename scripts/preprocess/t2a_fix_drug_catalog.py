"""Patch drug_catalog.parquet to:
1) split comma-duplicated SMILES (PRISM stores same SMILES N times),
2) recover SMILES for GDSC-only drugs by cross-matching on broad_id AND on
   case-insensitive drug name against the PRISM rows,
3) drop drugs that still have no valid RDKit-parsable SMILES.
"""
import pandas as pd, numpy as np
from pathlib import Path
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

PROC = Path('/home/holiday01/drug_sc/data/processed/hcc_drug')
RAW  = Path('/home/holiday01/drug_sc/data/drug_sensitivity_raw')

# raw PRISM for cross-matching
prism = pd.read_csv(RAW/'PRISM_secondary_AUC.csv',
                    usecols=['broad_id','name','moa','target','smiles','phase'])
prism['name_lower'] = prism['name'].astype(str).str.lower()

def first_parsable(smi_str):
    if not isinstance(smi_str,str) or not smi_str.strip(): return None
    # split by comma and take first that is parsable
    for part in smi_str.split(','):
        part = part.strip()
        if not part: continue
        m = Chem.MolFromSmiles(part)
        if m is not None and m.GetNumAtoms()>0:
            return Chem.MolToSmiles(m)
    return None

# build lookup: broad_id -> canonical SMILES, name_lower -> canonical SMILES
broad_lookup={}; name_lookup={}; meta_lookup={}
for _,r in prism.drop_duplicates('broad_id').iterrows():
    canon = first_parsable(r['smiles'])
    if canon is None: continue
    if isinstance(r['broad_id'],str):
        broad_lookup[r['broad_id']] = canon
    if isinstance(r['name_lower'],str):
        name_lookup[r['name_lower']] = canon
    meta_lookup[canon] = {'moa':r['moa'],'target':r['target'],'phase':r['phase']}

old = pd.read_parquet(PROC/'drug_catalog.parquet').drop_duplicates('drug')
old = old.set_index('drug')
print(f'old rows: {len(old)}')
# also collect broad_id map from long table
long_tab = pd.read_parquet(PROC/'drug_response_long.parquet')
drug_broad = (long_tab.dropna(subset=['broad_id']).groupby('drug')['broad_id']
              .agg(lambda s: s.value_counts().index[0] if len(s) else None))

rows=[]
for drug in old.index:
    canon = first_parsable(old.loc[drug,'smiles'])
    if canon is None:
        # try by broad_id
        bid = drug_broad.get(drug)
        if isinstance(bid,str):
            # broad_id can itself be comma-separated from T2a agg
            for b in bid.split(','):
                b = b.strip()
                if b in broad_lookup: canon = broad_lookup[b]; break
    if canon is None:
        # try by lower-case name
        canon = name_lookup.get(str(drug).lower())
    if canon is None:
        continue
    meta = meta_lookup.get(canon, {'moa':None,'target':None,'phase':None})
    rows.append({'drug':drug, 'broad_id':drug_broad.get(drug),
                 'smiles':canon,
                 'moa':meta['moa'],'target':meta['target'],'phase':meta['phase']})
new = pd.DataFrame(rows)
print(f'recovered rows: {len(new)}  (fraction with parsable SMILES)')
new.to_parquet(PROC/'drug_catalog.parquet', index=False)
print('Saved drug_catalog.parquet')
