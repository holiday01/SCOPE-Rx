"""
T3e-LUAD — Oncology / phase / MOA filter (LUAD edition).

Same logic as t3e_oncology_filter.py; reads T3d-LUAD outputs and uses
the LUAD-relevant clinical-drug list for the comparison report.
"""
from __future__ import annotations
import json, time, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path('/home/holiday01/drug_sc')
T3D  = ROOT/'results/t3d_luad'
OUT  = ROOT/'results/t3e_luad'; OUT.mkdir(parents=True, exist_ok=True)
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

CANCER_DRIVERS = set("""
ABL1 AKT1 AKT2 AKT3 ALK APC ARAF ARID1A ARID1B ARID2 ATM ATR AURKA AURKB AXL
BAP1 BCL2 BCL2L1 BCL6 BRAF BRCA1 BRCA2 BRD4 BTK CCND1 CCND2 CCND3 CCNE1
CDK1 CDK2 CDK4 CDK6 CDKN1A CDKN2A CDKN2B CHEK1 CHEK2 CREBBP CSF1R CTNNB1
DDR1 DDR2 DNMT1 DNMT3A DNMT3B DNMT3L E2F1 EED EGFR EP300 EPHA2 EPHA3 EPHB4
ERBB2 ERBB3 ERBB4 ERK1 ERK2 ESR1 ESR2 EZH1 EZH2 FANCA FANCC FANCD2 FAS
FGFR1 FGFR2 FGFR3 FGFR4 FLT1 FLT3 FLT4 FYN GLI1 GLI2 GSK3A GSK3B HDAC1
HDAC2 HDAC3 HDAC6 HDAC8 HER2 HGF HIF1A HIF2A HIF3A HRAS HSP90AA1 HSP90AB1
IDH1 IDH2 IGF1R IGFBP3 IKBKB IKK IKZF1 IKZF3 IL6 IL6R IMPDH1 INSR JAK1 JAK2
JAK3 KDR KIT KMT2A KMT2B KMT2C KMT2D KRAS LCK LMTK3 LTK LYN MAP2K1 MAP2K2
MAP3K1 MAP3K14 MAP4K4 MAPK1 MAPK14 MAPK3 MAPK8 MAPK9 MAX MCL1 MDM2 MDM4 MEK1
MEK2 MET MLL MLL2 MLL3 MLL4 MTOR MYB MYC MYCL MYCN NF1 NF2 NFKB1 NFKB2
NKX2-1 NOTCH1 NOTCH2 NOTCH3 NOTCH4 NPM1 NRAS NTRK1 NTRK2 NTRK3 PALB2 PARP1
PARP2 PARP3 PAX5 PAX8 PBRM1 PDGFRA PDGFRB PDK1 PDPK1 PHF6 PI3K PIK3CA
PIK3CB PIK3CD PIK3CG PIK3R1 PIK3R2 PIM1 PIM2 PIM3 PLK1 PLK2 PLK3 PLK4 POLE
PRKCA PRKCB PRKCD PRKDC PSMB5 PTCH1 PTCH2 PTEN PTK2 PTK2B PTK6 PTPN11
RAC1 RAF1 RB1 RET RHOA ROCK1 ROCK2 ROR1 ROR2 ROS1 RPS6KB1 RUNX1 RUNX2 RXRA
RXRB RXRG SETD2 SF3B1 SHP1 SHP2 SMAD2 SMAD3 SMAD4 SMARCA4 SMARCB1 SMO
SOS1 SRC STAT1 STAT3 STAT5 STAT5A STAT5B STAT6 STK11 SYK TERT TET1 TET2
TGFBR1 TGFBR2 TNK2 TOP1 TOP2A TOP2B TP53 TSC1 TSC2 TYK2 VEGFA VEGFB VEGFC
VEGFR1 VEGFR2 VEGFR3 VHL WEE1 WEE2 WNT WT1 XIAP YES1 ZAP70
ATR ATM RAD51 RAD52 BRD2 BRD3
""".split())

NONSPEC_MOA_PATTERNS = [
    r'\bsurfactant\b', r'\bantisept', r'\bantimicrob', r'\bantifung',
    r'\bantibacter', r'\bantiviral\b(?! hepatitis)', r'\banticonvuls',
    r'\bantidepress', r'\bantipsych', r'\banxiolyt', r'\bgaba\b',
    r'acetylcholin', r'adrenergic receptor', r'antihistamin',
    r'anti-?inflamm', r'antispasmod', r'sodium channel', r'mineralocorticoid',
    r'glucocorticoid', r'benzodiazepin', r'non-?steroidal', r'uric acid',
    r'cox(yc|-)?[12]? inhibitor', r'5-hydroxytryptamine',
    r'calcium channel', r'\bmuscarinic\b', r'antihelminth', r'antimalar',
    r'opioid', r'statin', r'ace inhibitor', r'angiotens',
    r'local anesthet', r'sweetener', r'diuretic', r'bronchodilat',
    r'laxative', r'contracept', r'beta-?adrenergic',
]
NONSPEC_RE = re.compile('|'.join(NONSPEC_MOA_PATTERNS), re.IGNORECASE)
ONC_RE = re.compile(r'oncology|cancer|tumor|tumour|malignan|leukemia|leukaemia|lymphoma|carcinoma|sarcoma|myeloma|melanoma|glioma|glioblastoma|neoplasm|metastas', re.IGNORECASE)

log('Loading PRISM metadata …')
raw = pd.read_csv(ROOT/'data/drug_sensitivity_raw/PRISM_secondary_AUC.csv',
                  usecols=['name','moa','target','phase','indication','disease.area'],
                  low_memory=False).drop_duplicates('name')
raw = raw.rename(columns={'name':'drug','disease.area':'disease_area'})
raw['drug_lc'] = raw['drug'].astype(str).str.lower()

def onc_relevance(row):
    score = 0; flags = []
    da = str(row.get('disease_area') or ''); ind = str(row.get('indication') or '')
    moa = str(row.get('moa') or '');         tgt = str(row.get('target') or '')
    phase = str(row.get('phase') or '')
    if ONC_RE.search(da):  score += 3; flags.append('disease_area_onc')
    if ONC_RE.search(ind): score += 2; flags.append('indication_onc')
    tgt_hits = [t.strip() for t in re.split(r'[,;/\|\s]+', tgt.upper()) if t.strip() and t.strip() in CANCER_DRIVERS]
    if tgt_hits: score += 2; flags.append(f'driver_target({",".join(tgt_hits[:3])})')
    if moa and NONSPEC_RE.search(moa): score -= 3; flags.append('nonspec_moa')
    if phase == 'Preclinical': score -= 2; flags.append('preclin')
    elif phase == 'Launched':   score += 1; flags.append('launched')
    elif phase in ('Phase 3','Phase 2'): score += 0.5
    return pd.Series({'onc_relevance':score, 'onc_flags':';'.join(flags)})

log('Scoring oncology relevance …')
rel = raw.apply(onc_relevance, axis=1)
raw = pd.concat([raw, rel], axis=1)
raw.to_parquet(OUT/'drug_oncology_metadata.parquet', index=False)
log(f'  score ≥ 3: {(raw.onc_relevance>=3).sum()}  |  >0: {(raw.onc_relevance>0).sum()}  |  ≤-3: {(raw.onc_relevance<=-3).sum()}')

cohort = pd.read_parquet(T3D/'drug_score_cohort.parquet')
cohort['drug_lc'] = cohort['drug'].astype(str).str.lower()
merged = cohort.merge(raw[['drug_lc','moa','target','phase','indication','disease_area','onc_relevance','onc_flags']],
                      on='drug_lc', how='left')
log(f'Merge coverage: {merged.onc_relevance.notna().sum()}/{len(merged)}')
merged['onc_relevance'] = merged['onc_relevance'].fillna(0)

strict = merged[(merged.onc_relevance>=1) &
                (merged.phase.isin(['Launched','Phase 3','Phase 2','Phase 1']))].copy()
strict = strict.sort_values('score', ascending=False)
strict.to_parquet(OUT/'drug_score_filtered_strict.parquet', index=False)
log(f'Strict: {len(strict)}/{len(merged)}')

merged['score_rank_frac'] = merged['score'].rank(pct=True)
merged['onc_weight'] = 1.0/(1.0+np.exp(-merged['onc_relevance']/2.0))
merged['score_combined'] = merged['score']*merged['onc_weight'] + 0.5*merged['onc_relevance']
merged = merged.sort_values('score_combined', ascending=False)
merged.to_parquet(OUT/'drug_score_soft_combined.parquet', index=False)

# Report
lines = ['# T3e-LUAD oncology/phase/MOA filter\n', '## Top-20 STRICT (oncology-tagged + Phase 1+)\n',
         '| # | Drug | Score | MOA | Target | Phase | Onc flags |',
         '|---:|---|---:|---|---|---|---|']
for i, r in enumerate(strict.head(20).itertuples(), 1):
    lines.append(f"| {i} | **{r.drug}** | {r.score:.2f} | {r.moa or ''} | {r.target or ''} | {r.phase or ''} | {r.onc_flags or ''} |")

lines += ['\n## Top-20 SOFT-COMBINED\n',
          '| # | Drug | Combined | Orig | Onc-rel | Phase | MOA |',
          '|---:|---|---:|---:|---:|---|---|']
for i, r in enumerate(merged.head(20).itertuples(), 1):
    lines.append(f"| {i} | **{r.drug}** | {r.score_combined:.2f} | {r.score:.2f} | {r.onc_relevance:+.1f} | {r.phase or ''} | {r.moa or ''} |")

clinicals = ['erlotinib','gefitinib','afatinib','osimertinib','dacomitinib',
             'crizotinib','alectinib','ceritinib','lorlatinib','brigatinib',
             'pemetrexed','docetaxel','paclitaxel','carboplatin','cisplatin',
             'gemcitabine','vinorelbine','etoposide','ramucirumab','bevacizumab',
             'nivolumab','pembrolizumab','atezolizumab','sotorasib','adagrasib',
             'trametinib','selumetinib']
lines += ['\n## Clinical LUAD drug ranking — across three lists\n',
          '| Drug | Original rank | Strict rank | Soft rank |',
          '|---|---:|---:|---:|']
orig = cohort.sort_values('score', ascending=False).reset_index(drop=True)
strict_r = strict.reset_index(drop=True)
merged_r = merged.reset_index(drop=True)
for dn in clinicals:
    hit_o = orig[orig.drug.str.lower()==dn]
    hit_s = strict_r[strict_r.drug.str.lower()==dn]
    hit_m = merged_r[merged_r.drug.str.lower()==dn]
    r_o = int(hit_o.index[0])+1 if len(hit_o) else None
    r_s = int(hit_s.index[0])+1 if len(hit_s) else None
    r_m = int(hit_m.index[0])+1 if len(hit_m) else None
    ro = f"{r_o} ({r_o/len(orig)*100:.1f}%)" if r_o else '—'
    rs = f"{r_s} ({r_s/max(len(strict_r),1)*100:.1f}%)" if r_s else '—'
    rm = f"{r_m} ({r_m/len(merged_r)*100:.1f}%)" if r_m else '—'
    lines.append(f"| {dn} | {ro} | {rs} | {rm} |")

(OUT/'comparison_top20.md').write_text('\n'.join(lines))
log(f'Report: {OUT/"comparison_top20.md"}')

(OUT/'eval_metrics.json').write_text(json.dumps({
    'n_drugs_cohort': int(len(merged)),
    'n_strict_retained': int(len(strict)),
    'n_onc_relevance_ge3': int((raw.onc_relevance>=3).sum()),
    'n_nonspec_moa_flagged': int((raw.onc_relevance<=-3).sum()),
    'strict_top20': [{'rank':i+1,'drug':r.drug,'score':float(r.score),
                      'phase':r.phase,'moa':r.moa,'target':r.target}
                     for i,r in enumerate(strict.head(20).itertuples())],
    'soft_top20': [{'rank':i+1,'drug':r.drug,'combined':float(r.score_combined),
                    'onc_rel':float(r.onc_relevance)}
                   for i,r in enumerate(merged.head(20).itertuples())],
}, indent=2, default=str))
log('== T3e-LUAD done ==')
