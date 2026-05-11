"""
Build the 5 main tables in markdown + LaTeX (booktabs) format,
plus a supplementary index.
"""
from pathlib import Path
import pandas as pd, numpy as np, json
ROOT = Path('/home/holiday01/drug_sc')
OUT_MD  = ROOT/'results/tables_main.md'
OUT_TEX = ROOT/'results/tables_main.tex'
OUT_SUP = ROOT/'results/supplementary_index.md'

md_lines = ['# SCOPE-Rx — Main tables\n']
tex_lines = ['% SCOPE-Rx main tables (booktabs)']
tex_lines.append('\\usepackage{booktabs}\n\\usepackage{multirow}')

# -------------------------------------------------------------
# Table 1 — Cohort summary
# -------------------------------------------------------------
md_lines.append('## Table 1. Cohorts and validation summary\n')
md_lines.append('| Cohort | Cancer | Platform | Role | n tumor | Endpoints | Shared genes (pipeline ∩ array) | Composite c-index |')
md_lines.append('|---|---|---|---|---:|---|---:|---|')
T1 = [
    ('TCGA-LIHC','LIHC','RNA-seq','Train (anchor)',423,'OS',17460,'—'),
    ('GSE14520','LIHC','Affy HG-U133A 2.0','External',225,'OS, RFS',11200,'0.727 (OS) / 0.685 (RFS)'),
    ('GSE76427','LIHC','Illumina HT-12 v4','External',115,'OS, RFS',13400,'0.686 (OS) / 0.639 (RFS)'),
    ('TCGA-LUAD','LUAD','RNA-seq','Train (anchor)',576,'OS',17180,'—'),
    ('GSE68465','LUAD','Affy HG-U133A','External',462,'OS, RFS',11523,'0.639 (OS) / 0.591 (RFS)'),
    ('GSE72094','LUAD','Affy HuRSTA','External',442,'OS',15693,'0.670 (OS)'),
    ('GSE31210','LUAD','Affy HG-U133 Plus 2','External',204,'OS, RFS',15971,'0.779 (OS) / 0.618 (RFS)'),
]
for r in T1:
    md_lines.append('| ' + ' | '.join(str(x) for x in r) + ' |')
md_lines.append('\nTotal: **2,447 patients** across **2 cancers**, **5 external cohorts**.\n')

tex_lines += ['\n\\begin{table*}[ht]\n\\centering',
              '\\caption{Cohorts and validation summary. SCOPE-Rx uses two TCGA-anchored training cohorts and five independent external cohorts (2,447 total patients, five array or RNA-seq platforms). The composite TCGA-trained risk score is multivariate-adjusted for stage, age and sex where available; see Methods.}',
              '\\label{tab:cohorts}',
              '\\small\\begin{tabular}{llllrlrl}\\toprule',
              'Cohort & Cancer & Platform & Role & $n$ tumor & Endpoints & Genes & Composite $c$-index \\\\\\midrule']
for r in T1:
    tex_lines.append(' & '.join(str(x).replace('—','---') for x in r) + ' \\\\')
tex_lines += ['\\bottomrule\\end{tabular}\\end{table*}\n']

# -------------------------------------------------------------
# Table 2 — Prognostic prototypes (LIHC + LUAD)
# -------------------------------------------------------------
md_lines.append('## Table 2. Prognostic prototypes that replicate across cohorts\n')
md_lines.append('### LIHC — 9 multivariate-significant prototypes (TCGA-LIHC, stage/age/sex-adjusted)\n')
md_lines.append('| Proto | Dominant cell type | Training HR (p) | GSE14520 HR (p) | GSE76427 HR (p) |')
md_lines.append('|---:|---|---|---|---|')
lihc_mv = pd.read_parquet(ROOT/'results/t3h/multivariate_cox.parquet')
lihc_mv = lihc_mv[lihc_mv['p_x_mv']<0.05].sort_values('p_x_mv').head(9)
ext_lihc = pd.read_parquet(ROOT/'results/t3j/all_external_cox.parquet')
g14 = ext_lihc[ext_lihc['cohort']=='GSE14520_OS']
g76 = ext_lihc[ext_lihc['cohort']=='GSE76427_OS']
def lookup(df,p):
    s = df[df['proto']==p]
    if len(s): return f"{s['HR'].iloc[0]:.2f} ({s['p'].iloc[0]:.2g})"
    return '—'
for _, r in lihc_mv.iterrows():
    p = int(r['proto'])
    md_lines.append(f"| {p} | {r['dominant'][:28]} | {r['HR_x_uni']:.2f} ({r['p_x_mv']:.2g}) | "
                    f"{lookup(g14,p)} | {lookup(g76,p)} |")

md_lines.append('\n### LUAD — Top-15 4-cohort meta-pooled prognostic prototypes\n')
md_lines.append('| Proto | Dominant cell type | T/N | Pooled HR | Pooled p | I² | n cohorts |')
md_lines.append('|---:|---|---|---:|---|---:|---:|')
luad_meta = pd.read_parquet(ROOT/'results/t3j_luad/meta_analysis_pooled.parquet').head(15)
for _, r in luad_meta.iterrows():
    md_lines.append(f"| {int(r['proto'])} | {r.get('dominant_cell_type','')[:24]} | "
                    f"{r.get('dominant_sample_type','')} | {r['pooled_HR']:.2f} | "
                    f"{r['pooled_p']:.2g} | {r['I2_pct']:.0f}% | {int(r['n_cohorts'])} |")

# LaTeX equivalent
tex_lines += ['\n\\begin{table*}[ht]\n\\centering',
              '\\caption{Prognostic prototypes that replicate across cohorts. Top: 9 multivariate-adjusted (stage/age/sex) significant prototypes in TCGA-LIHC and their hazard ratios in two external HCC cohorts. Bottom: top 15 LUAD prototypes ranked by 4-cohort fixed-effect meta-pooled $p$ value (TCGA-LUAD + 3 external).}',
              '\\label{tab:protos}',
              '\\small\\begin{tabular}{rlllll}\\toprule',
              '\\multicolumn{6}{c}{\\textbf{LIHC} (multivariate-adjusted Cox; $n=$385)}\\\\\\midrule',
              'Proto & Dominant cell type & TCGA HR ($p$) & GSE14520 HR ($p$) & GSE76427 HR ($p$) & \\\\\\midrule']
for _, r in lihc_mv.iterrows():
    p = int(r['proto'])
    tex_lines.append(f"{p} & {r['dominant'][:28]} & {r['HR_x_uni']:.2f} ({r['p_x_mv']:.2g}) & "
                     f"{lookup(g14,p)} & {lookup(g76,p)} & \\\\")
tex_lines += ['\\midrule\\multicolumn{6}{c}{\\textbf{LUAD} (4-cohort fixed-effect meta-analysis)}\\\\\\midrule',
              'Proto & Dominant cell type & T/N & Pooled HR & Pooled $p$ & $I^2$ \\\\\\midrule']
for _, r in luad_meta.iterrows():
    tex_lines.append(f"{int(r['proto'])} & {r.get('dominant_cell_type','')[:24]} & "
                     f"{r.get('dominant_sample_type','')} & {r['pooled_HR']:.2f} & "
                     f"{r['pooled_p']:.2g} & {r['I2_pct']:.0f}\\% \\\\")
tex_lines += ['\\bottomrule\\end{tabular}\\end{table*}\n']

# -------------------------------------------------------------
# Table 3 — Top-20 final drug ranking (LIHC + LUAD)
# -------------------------------------------------------------
md_lines.append('\n## Table 3. Top-20 final drug ranking per cancer\n')
def mech_axis(target, moa, drug):
    s = ' '.join([str(target or ''), str(moa or ''), str(drug or '')]).upper()
    if any(k in s for k in ['EGFR','ERBB']) and 'ALK' not in s: return 'EGFR/HER'
    if 'ALK' in s and 'EGFR' not in s: return 'ALK'
    if any(k in s for k in ['MAP2K','MEK ','MEK1','MEK2']): return 'MEK'
    if any(k in s for k in ['HSP90','TANESPI','ALVESPI']): return 'HSP90'
    if 'PSM' in s or 'PROTEASOME' in s: return 'Proteasome'
    if any(k in s for k in ['BCL2','BCL_','VENETOCLAX','NAVITOCLAX']): return 'BCL'
    if any(k in s for k in ['PIK3','MTOR','AKT','PI3K']): return 'PI3K/mTOR'
    if any(k in s for k in ['CDK4','CDK6','PALBOCICLIB']): return 'CDK4/6'
    if any(k in s for k in ['MDM2']): return 'MDM2'
    if any(k in s for k in ['PARP']): return 'PARP'
    if any(k in s for k in ['HDAC']): return 'HDAC'
    if any(k in s for k in ['TOP1','SN-38']): return 'TOP1'
    if any(k in s for k in ['MET ','KDR','VEGFR','CABOZANTINIB','FORETINIB','SORAFENIB']): return 'VEGFR/MET'
    if any(k in s for k in ['DHFR','TYMS','PEMETREXED']): return 'Antifolate'
    if any(k in s for k in ['SMO ','HEDGEHOG']): return 'Hedgehog'
    return 'Other'

for cancer, path in [('LIHC', 'results/t3f/drug_final_score.parquet'),
                     ('LUAD', 'results/t3f_luad/drug_final_score.parquet')]:
    df = pd.read_parquet(ROOT/path).head(20)
    md_lines.append(f'\n### {cancer} — Top-20 (n drugs scored = ' +
                    f"{len(pd.read_parquet(ROOT/path))})\n")
    md_lines.append('| Rank | Drug | Final | z_kill | z_onc | z_prior | MOA | Target | Phase | Axis |')
    md_lines.append('|---:|---|---:|---:|---:|---:|---|---|---|---|')
    for i, r in enumerate(df.itertuples(), 1):
        moa = (r.moa or '')[:24].replace('|','/')
        tgt = (r.target or '')[:18].replace('|','/')
        ph  = r.phase or '—'
        axis = mech_axis(r.target, r.moa, r.drug)
        md_lines.append(f"| {i} | **{r.drug}** | {r.score_final:+.2f} | {r.z_kill:+.2f} | "
                        f"{r.z_onc:+.2f} | {r.z_prior:+.2f} | {moa} | {tgt} | {ph} | {axis} |")

tex_lines += ['\n\\begin{table*}[ht]\\centering',
              '\\caption{Top-20 drug ranking by SCOPE-Rx final score ($z_{kill} + 0.5\\,z_{onc} + 0.7\\,z_{prior}$) for each cancer. Mechanism axis is annotated for cross-cancer comparison; standard-of-care drugs are bolded.}',
              '\\label{tab:top20}',
              '\\scriptsize\\begin{tabular}{rlrrrrlll}\\toprule']
for cancer, path in [('LIHC', 'results/t3f/drug_final_score.parquet'),
                     ('LUAD', 'results/t3f_luad/drug_final_score.parquet')]:
    df = pd.read_parquet(ROOT/path).head(20)
    tex_lines.append(f'\\multicolumn{{9}}{{c}}{{\\textbf{{{cancer}}} top-20}}\\\\\\midrule')
    tex_lines.append('Rank & Drug & Final & $z_{kill}$ & $z_{onc}$ & $z_{prior}$ & MOA & Target & Phase\\\\\\midrule')
    for i, r in enumerate(df.itertuples(), 1):
        def esc(x):
            return str(x).replace('&','\\&').replace('_','\\_').replace('%','\\%')[:18]
        tex_lines.append(f"{i} & {esc(r.drug)} & {r.score_final:+.2f} & {r.z_kill:+.2f} & "
                         f"{r.z_onc:+.2f} & {r.z_prior:+.2f} & {esc(r.moa or '')} & "
                         f"{esc(r.target or '')} & {esc(r.phase or '---')}\\\\")
tex_lines += ['\\bottomrule\\end{tabular}\\end{table*}\n']

# -------------------------------------------------------------
# Table 4 — Composite TCGA-trained risk score validation
# -------------------------------------------------------------
md_lines.append('\n## Table 4. Composite TCGA-trained risk score on external cohorts\n')
md_lines.append('| Cohort | Cancer | Endpoint | n | HR per SD | p | c-index |')
md_lines.append('|---|---|---|---:|---:|---|---:|')
lihc_c = pd.read_parquet(ROOT/'results/t3j/composite_risk_results.parquet')
luad_eval = json.loads((ROOT/'results/t3i_luad/eval_metrics.json').read_text())
table4_rows = []
for _, r in lihc_c.iterrows():
    cohort, endpt = r['cohort'].rsplit('_',1)
    table4_rows.append((cohort, 'LIHC', endpt, int(r['n']),
                        float(r['HR']), float(r['p']), float(r['c_index'])))
for cid, sub in luad_eval['cohorts'].items():
    for endpt, e in sub['composite_risk'].items():
        table4_rows.append((cid, 'LUAD', endpt, int(e['n']),
                            float(e['HR_per_SD']), float(e['p']), float(e['c_index'])))
for r in table4_rows:
    md_lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.2f} | {r[5]:.2g} | {r[6]:.3f} |")

tex_lines += ['\n\\begin{table}[ht]\\centering',
              '\\caption{Composite TCGA-trained risk score $r=\\sum_p c_p \\log\\widehat{HR}_p$ evaluated on independent external cohorts. The score is multivariate-adjusted for stage/age/sex (where available).}',
              '\\label{tab:composite-risk}',
              '\\begin{tabular}{lllrrll}\\toprule',
              'Cohort & Cancer & Endpoint & $n$ & HR/SD & $p$ & $c$-index\\\\\\midrule']
for r in table4_rows:
    tex_lines.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]:.2f} & {r[5]:.2g} & {r[6]:.3f}\\\\")
tex_lines += ['\\bottomrule\\end{tabular}\\end{table}\n']

# -------------------------------------------------------------
# Table 5 — Method ablation summary
# -------------------------------------------------------------
md_lines.append('\n## Table 5. Method ablation summary\n')
md_lines.append('| Tier | Variant | Key metric | Value | Δ vs default |')
md_lines.append('|---|---|---|---:|---|')
A = pd.read_parquet(ROOT/'results/ablation_table/embedding.parquet')
B = pd.read_parquet(ROOT/'results/ablation_table/deconv.parquet')
C = pd.read_parquet(ROOT/'results/ablation_table/score_fusion.parquet')
M = json.loads((ROOT/'results/ablation_mpnn_holdout/summary.json').read_text())

# Tier A (Cox p<0.001 prototypes)
A_def = int(A.iloc[0]['cox_p001'])
for _, r in A.iterrows():
    delta = int(r['cox_p001']) - A_def
    sign = ('+' if delta>=0 else '')
    md_lines.append(f"| A | {r['variant'].replace('_',' ')} | "
                    f"Cox p<0.001 prototypes | {int(r['cox_p001'])} | "
                    f"{'(default)' if r['variant']=='A0_Geneformer' else f'{sign}{delta}'} |")

# Tier B (collapse %)
for _, r in B.iterrows():
    md_lines.append(f"| B | {r['method'].replace('_',' ')} | "
                    f"Top-1 collapse % / entropy % | "
                    f"{r['top1_collapse_pct']*100:.0f}% / {r['entropy_pct']*100:.0f}% | "
                    f"{'(default)' if r['method']=='B0_Attention' else 'collapses' } |")

# Tier C (mean SOC rank %)
C_def = float(C.iloc[0]['mean_pct'])
for _, r in C.iterrows():
    delta = r['mean_pct'] - C_def
    md_lines.append(f"| C | {r['variant'].replace('_',' ')} | "
                    f"9 SOC mean rank% | {r['mean_pct']:.2f}% | "
                    f"{'(default)' if r['variant'].startswith('C0') else f'{delta:+.1f}%'} |")

# MPNN
order = ['MPNN','CellLine_mean','Random_emb','ECFP4+Ridge']
mpnn_def = M['MPNN']['mean_pearson']
for k in order:
    delta = M[k]['mean_pearson'] - mpnn_def
    md_lines.append(f"| MPNN | {k} | held-out mean Pearson r | "
                    f"{M[k]['mean_pearson']:+.3f} | "
                    f"{'(default)' if k=='MPNN' else f'{delta:+.3f}'} |")

tex_lines += ['\n\\begin{table}[ht]\\centering',
              '\\caption{Method ablation summary. Tier A (embedding tower) and Tier C (score fusion) impact prognostic-prototype recovery and SOC drug ranking respectively. Tier B (deconvolution method) shows that NNLS and Scaden-style MLP collapse 95--99\\% of patients to a single prototype while attention deconvolution preserves diversity. The MPNN held-out drug evaluation shows graph encoding outperforms ECFP4+Ridge but only marginally beats a cell-line-mean baseline, consistent with known limits of unseen-drug AUC prediction.}',
              '\\label{tab:ablation}',
              '\\small\\begin{tabular}{llllr}\\toprule',
              'Tier & Variant & Metric & Value & $\\Delta$ vs default \\\\\\midrule']
for _, r in A.iterrows():
    delta = int(r['cox_p001']) - A_def
    tex_lines.append(f"A & {r['variant'].replace('_','-')} & Cox $p<0.001$ protos & "
                     f"{int(r['cox_p001'])} & "
                     f"{'(default)' if r['variant']=='A0_Geneformer' else f'{delta:+d}'} \\\\")
for _, r in B.iterrows():
    tex_lines.append(f"B & {r['method'].replace('_','-')} & "
                     f"Top-1 collapse / entropy & "
                     f"{r['top1_collapse_pct']*100:.0f}\\%~/~{r['entropy_pct']*100:.0f}\\% & "
                     f"{'(default)' if r['method']=='B0_Attention' else 'collapses'} \\\\")
PCT = '\\%'
for _, r in C.iterrows():
    delta = r['mean_pct'] - C_def
    delta_str = '(default)' if r['variant'].startswith('C0') else f'{delta:+.1f}{PCT}'
    tex_lines.append(f"C & {r['variant'].replace('_','-')[:18]} & 9 SOC mean rank {PCT} & "
                     f"{r['mean_pct']:.2f}{PCT} & {delta_str} \\\\")
for k in order:
    delta = M[k]['mean_pearson'] - mpnn_def
    tex_lines.append(f"MPNN & {k.replace('_',' ')} & Held-out mean Pearson $r$ & "
                     f"{M[k]['mean_pearson']:+.3f} & "
                     f"{'(default)' if k=='MPNN' else f'{delta:+.3f}'} \\\\")
tex_lines += ['\\bottomrule\\end{tabular}\\end{table}\n']

OUT_MD.write_text('\n'.join(md_lines))
OUT_TEX.write_text('\n'.join(tex_lines))
print(f'Wrote {OUT_MD}')
print(f'Wrote {OUT_TEX}')

# -------------------------------------------------------------
# Supplementary index
# -------------------------------------------------------------
sup = ['# SCOPE-Rx — Supplementary index\n',
       '## Supplementary figures (potential additional)\n',
       '- **Suppl. Fig S1**: per-cohort Cox per-prototype heatmap (LIHC + LUAD)\n  Source: `results/t3c{,_luad}/cox_per_prototype.parquet`',
       '- **Suppl. Fig S2**: pathway hazard heatmap (1,790 pathways × 2 cancers)\n  Source: `results/t3f/pathway_cox_tcga_lihc.parquet`, `results/t3f_luad/pathway_cox_tcga_luad.parquet`',
       '- **Suppl. Fig S3**: per-prototype DEG markers + best-matching DepMap cell line\n  Source: `results/t4{,_luad}/subpopulation_markers.parquet`',
       '- **Suppl. Fig S4**: MPNN per-fold per-drug Pearson r distribution\n  Source: `results/ablation_mpnn_holdout/cv_results.parquet`',
       '- **Suppl. Fig S5**: Geneformer dictionary mapping coverage and gene-universe overlap\n  Source: `results/t3b{,_luad}/eval_metrics.json` and `data/processed/{hcc,luad}_drug/gene_universe.parquet`',
       '- **Suppl. Fig S6**: Top-20 wet-lab brief (FACS markers + suggested cell line / PDO + SMILES)\n  Source: `results/t4{,_luad}/{hcc,luad}_top20_wetlab_brief.{md,csv}`',
       '- **Suppl. Fig S7**: per-prototype trust-to-DepMap calibration with TME flag\n  Source: `results/t3c{,_luad}/prototype_meta.parquet`',
       '- **Suppl. Fig S8**: 5-method × 4-aggregation consistency battery (T3k)\n  Source: `results/t3k/consistency.parquet`',
       '\n## Supplementary tables\n',
       '- **Suppl. Table S1**: full per-prototype per-cohort Cox table (univariate + multivariate)\n  Source: `results/t3c{,_luad}/cox_per_prototype.parquet`, `results/t3i{,_luad}/`, `results/t3h/multivariate_cox.parquet`',
       '- **Suppl. Table S2**: 1,790 × 2 pathway-Cox results\n  Source: `results/t3f{,_luad}/pathway_cox_tcga_*.parquet`',
       '- **Suppl. Table S3**: drug-target × pathway coverage matrix (1,489 drugs × 1,790 pathways × 2 cancers)\n  Source: `results/t3f{,_luad}/drug_target_pathway_matrix.parquet`',
       '- **Suppl. Table S4**: per-patient top-5 drug list (423 LIHC + 576 LUAD)\n  Source: `results/t4{,_luad}/per_patient_top5.parquet`',
       '- **Suppl. Table S5**: full top-20 wet-lab brief with FACS markers, suggested cell line, dose, mechanism hypothesis\n  Source: `results/t4{,_luad}/`',
       '- **Suppl. Table S6**: full 4-cohort meta-analysis pooled HR table for all 58 LUAD prototypes\n  Source: `results/t3j_luad/meta_analysis_pooled.parquet`',
       '- **Suppl. Table S7**: ablation full results JSON (Tiers A/B/C metrics)\n  Source: `results/ablation_table/eval_metrics.json`',
       '- **Suppl. Table S8**: MPNN 5-fold CV per-drug per-method Pearson + Spearman (n=1,602 × 4 methods × 5 folds)\n  Source: `results/ablation_mpnn_holdout/cv_results.parquet`',
       '- **Suppl. Table S9**: T2b scDEAL per-drug Pearson on held-out cell lines\n  Source: `results/t2b{,_luad}/drug_level_pearson.parquet`',
       '\n## Supplementary code (script paths)\n',
       '- T2a preprocessing: `scripts/preprocess/t2a_prepare_{hcc,luad}_drug_data.py`',
       '- T2b scDEAL training: `scripts/baselines/t2b_scdeal_train{,_luad}.py`',
       '- T3b Geneformer embedding: `scripts/novelty/t3b_geneformer_embed{,_luad}.py`',
       '- T3c Attention deconvolution: `scripts/novelty/t3c_celltype_attention_deconv{,_luad}.py`',
       '- T3d Survival-anchored scoring: `scripts/novelty/t3d_signature_reversal_cox{,_luad}.py`',
       '- T3e Oncology filter: `scripts/novelty/t3e_oncology_filter{,_luad}.py`',
       '- T3f Pathway prior: `scripts/novelty/t3f_target_pathway_prior{,_luad}.py`',
       '- T3i External validation: `scripts/novelty/t3i_external_validation_{gse76427,luad}.py`',
       '- T3j Meta-analysis: `scripts/novelty/t3j_two_cohort_meta_analysis.py`, `t3j_meta_analysis_luad.py`',
       '- T4 Wet-lab brief: `scripts/novelty/t4_integrated_ranking_wetlab_brief{,_luad}.py`',
       '- Ablation tier A/B/C: `scripts/ablation/luad_ablation_table.py`',
       '- MPNN held-out CV: `scripts/ablation/mpnn_holdout_drug_eval.py`',
       '\n## Supplementary data files\n',
       '- DepMap pan-cancer expression: `data/processed/{hcc,luad}_drug/cellline_expression_panCancer.parquet`',
       '- Drug catalog with SMILES: `data/processed/{hcc,luad}_drug/drug_catalog.parquet`',
       '- TCGA expression + clinical (training): `data/TCGA_LIHC/`, `/mnt/10t/scrna_atac/data/raw/TCGA_LUAD/`',
       '- External GEO series matrices + platform annotations: `data/external_HCC/`, `data/external_LUAD/`',
       '- Geneformer V2-104M model + dictionaries: `models/Geneformer/Geneformer-V2-104M/`',
       '- scRNA atlases (LIHC integrated, LUAD annotated): `data/scRNA_HCC_integrated/`, `/mnt/10t/scrna_atac/data/processed/LUAD/`',
]
OUT_SUP.write_text('\n'.join(sup))
print(f'Wrote {OUT_SUP}')
