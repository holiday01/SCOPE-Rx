"""
Cross-check every quantitative claim in narrative.md / SCOPE-Rx_LIHC_vs_LUAD_comparison.md
/ tables_main.md / phase1_final_report.md against the source parquet/JSON files.

Output: results/audit_report.md  with PASS / FAIL flags.
"""
from pathlib import Path
import json
import pandas as pd, numpy as np
ROOT = Path('/home/holiday01/drug_sc')

audit = []   # rows: ('Claim', 'Source', 'Reported', 'Computed', 'Status')

def add(claim, src, reported, computed, ok):
    audit.append((claim, src, str(reported), str(computed), 'PASS' if ok else 'FAIL'))

# ============================================================
# 1. Cohort sizes
# ============================================================
print('--- Cohort sizes ---')

# TCGA-LIHC: t3c tcga_composition shape
lihc_comp = pd.read_parquet(ROOT/'results/t3c/tcga_composition.parquet')
add('TCGA-LIHC composition n', 't3c/tcga_composition.parquet', 423,
    lihc_comp.shape[0], lihc_comp.shape[0] == 423)

# TCGA-LUAD: t3c_luad
luad_comp = pd.read_parquet(ROOT/'results/t3c_luad/tcga_composition.parquet')
add('TCGA-LUAD composition n', 't3c_luad/tcga_composition.parquet', 576,
    luad_comp.shape[0], luad_comp.shape[0] == 576)

# GSE14520
g14_comp = pd.read_parquet(ROOT/'results/t3j/gse14520_composition.parquet')
add('GSE14520 composition n (tumor, Cox-eligible)', 't3j/gse14520_composition.parquet', 225,
    g14_comp.shape[0], g14_comp.shape[0] == 225)

# GSE76427
g76_comp = pd.read_parquet(ROOT/'results/t3i/gse76427_composition.parquet')
add('GSE76427 composition n (tumor)', 't3i/gse76427_composition.parquet', 115,
    g76_comp.shape[0], g76_comp.shape[0] == 115)

# LUAD ext cohorts
for cid, expected in [('GSE68465', 462), ('GSE72094', 442), ('GSE31210', 204)]:
    c = pd.read_parquet(ROOT/f'results/t3i_luad/{cid}_composition.parquet')
    add(f'{cid} composition n (tumor)', f't3i_luad/{cid}_composition.parquet',
        expected, c.shape[0], c.shape[0] == expected)

# Total counts
all_n = 423+576+225+115+462+442+204
add(f'Total patients across all 7 cohorts', '(sum)', 2447, all_n, all_n == 2447)

# ============================================================
# 2. Cox-significant prototypes
# ============================================================
print('--- LIHC mv-Cox sig ---')
lihc_mv = pd.read_parquet(ROOT/'results/t3h/multivariate_cox.parquet')
n_mv05 = (lihc_mv['p_x_mv'] < 0.05).sum()
add('LIHC mv-Cox p<0.05 prototypes', 't3h/multivariate_cox.parquet',
    '9 in narrative', n_mv05, n_mv05 == 9)

print('--- LUAD univariate Cox sig ---')
luad_cox = pd.read_parquet(ROOT/'results/t3c_luad/cox_per_prototype.parquet')
add('LUAD Cox p<0.05 prototypes', 't3c_luad/cox_per_prototype.parquet',
    '15', (luad_cox['p']<0.05).sum(), (luad_cox['p']<0.05).sum() == 15)
add('LUAD Cox p<0.01', '', '8', (luad_cox['p']<0.01).sum(), (luad_cox['p']<0.01).sum() == 8)
add('LUAD Cox p<0.001', '', '4', (luad_cox['p']<0.001).sum(), (luad_cox['p']<0.001).sum() == 4)

# ============================================================
# 3. Composite risk c-index in external cohorts
# ============================================================
print('--- Composite risk LIHC ---')
lihc_c = pd.read_parquet(ROOT/'results/t3j/composite_risk_results.parquet')
g14_os = lihc_c[lihc_c['cohort']=='GSE14520_OS'].iloc[0]
add('GSE14520 OS c-index', 't3j/composite_risk_results.parquet',
    '0.727', f"{g14_os['c_index']:.3f}", abs(g14_os['c_index'] - 0.727) < 0.01)
add('GSE14520 OS HR per SD', '', '1.348',
    f"{g14_os['HR']:.3f}", abs(g14_os['HR'] - 1.348) < 0.01)
add('GSE14520 OS p', '', '0.006',
    f"{g14_os['p']:.3g}", g14_os['p'] < 0.01)

g14_rfs = lihc_c[lihc_c['cohort']=='GSE14520_RFS'].iloc[0]
add('GSE14520 RFS c-index', '', '0.685', f"{g14_rfs['c_index']:.3f}",
    abs(g14_rfs['c_index'] - 0.685) < 0.01)

g76_os = lihc_c[lihc_c['cohort']=='GSE76427_OS'].iloc[0]
add('GSE76427 OS c-index', '', '0.686', f"{g76_os['c_index']:.3f}",
    abs(g76_os['c_index'] - 0.686) < 0.01)

g76_rfs = lihc_c[lihc_c['cohort']=='GSE76427_RFS'].iloc[0]
add('GSE76427 RFS c-index', '', '0.639', f"{g76_rfs['c_index']:.3f}",
    abs(g76_rfs['c_index'] - 0.639) < 0.01)

print('--- Composite risk LUAD ---')
luad_eval = json.loads((ROOT/'results/t3i_luad/eval_metrics.json').read_text())
for cid, endpt, expected_c, expected_p in [
    ('GSE68465','OS', 0.639, 3.08e-6),
    ('GSE68465','RFS', 0.591, 1.26e-4),
    ('GSE72094','OS', 0.670, 1.83e-7),
    ('GSE31210','OS', 0.779, 5e-3),
]:
    e = luad_eval['cohorts'][cid]['composite_risk'][endpt]
    add(f'{cid} {endpt} c-index', f't3i_luad eval', expected_c,
        f"{e['c_index']:.3f}", abs(e['c_index'] - expected_c) < 0.01)
    add(f'{cid} {endpt} p', '', f'{expected_p:.2g}', f"{e['p']:.2g}",
        e['p'] < 0.01 if expected_p < 0.01 else True)

# ============================================================
# 4. 4-cohort meta-analysis pooled significant
# ============================================================
print('--- LUAD 4-cohort meta ---')
meta = pd.read_parquet(ROOT/'results/t3j_luad/meta_analysis_pooled.parquet')
add('LUAD meta n_pooled', 't3j_luad/meta_analysis_pooled.parquet',
    58, len(meta), len(meta) == 58)
add('LUAD meta p<0.05', '', '26', (meta['pooled_p']<0.05).sum(),
    (meta['pooled_p']<0.05).sum() == 26)
add('LUAD meta p<0.01', '', '22', (meta['pooled_p']<0.01).sum(),
    (meta['pooled_p']<0.01).sum() == 22)
add('LUAD meta p<0.001', '', '18', (meta['pooled_p']<0.001).sum(),
    (meta['pooled_p']<0.001).sum() == 18)

# ============================================================
# 5. Top-1 drug for each cancer
# ============================================================
print('--- Top drugs ---')
lihc_final = pd.read_parquet(ROOT/'results/t3f/drug_final_score.parquet').sort_values('score_final', ascending=False)
top1_lihc = lihc_final.iloc[0]['drug']
add('LIHC top-1 final drug', 't3f/drug_final_score.parquet', 'Lapatinib',
    top1_lihc, top1_lihc.lower() == 'lapatinib')

luad_final = pd.read_parquet(ROOT/'results/t3f_luad/drug_final_score.parquet').sort_values('score_final', ascending=False)
top1_luad = luad_final.iloc[0]['drug']
add('LUAD top-1 final drug', 't3f_luad/drug_final_score.parquet', 'GEFITINIB',
    top1_luad, top1_luad.upper() == 'GEFITINIB')

# Verify LUAD SOC drug ranks claimed in narrative
luad_final['drug_lc'] = luad_final['drug'].str.lower()
luad_final = luad_final.reset_index(drop=True)
soc_check = {'gefitinib':1, 'trametinib':4, 'osimertinib':11, 'selumetinib':12,
             'alectinib':13, 'erlotinib':57, 'afatinib':59, 'brigatinib':61,
             'pemetrexed':45}
for d, expected in soc_check.items():
    h = luad_final[luad_final['drug_lc']==d]
    if len(h):
        actual = int(h.index[0]) + 1
        add(f'LUAD {d} final rank', '', expected, actual, actual == expected)
    else:
        add(f'LUAD {d} final rank', '', expected, 'NOT FOUND', False)

# ============================================================
# 6. Ablation results
# ============================================================
print('--- Ablation tier A ---')
A = pd.read_parquet(ROOT/'results/ablation_table/embedding.parquet')
geneformer_p001 = int(A[A['variant']=='A0_Geneformer']['cox_p001'].iloc[0])
random_p001 = int(A[A['variant']=='A3_Random']['cox_p001'].iloc[0])
scvi_p001 = int(A[A['variant']=='A1_scVI']['cox_p001'].iloc[0])
pca_p001 = int(A[A['variant']=='A2_PCA']['cox_p001'].iloc[0])
add('Tier A0 Geneformer p<0.001', 'ablation_table/embedding.parquet',
    '2', geneformer_p001, geneformer_p001 == 2)
add('Tier A1 scVI p<0.001', '', '3', scvi_p001, scvi_p001 == 3)
add('Tier A2 PCA p<0.001', '', '2', pca_p001, pca_p001 == 2)
add('Tier A3 Random p<0.001', '', '0', random_p001, random_p001 == 0)

print('--- Ablation tier B ---')
B = pd.read_parquet(ROOT/'results/ablation_table/deconv.parquet')
attn_collapse = float(B[B['method']=='B0_Attention']['top1_collapse_pct'].iloc[0])
nnls_collapse = float(B[B['method']=='B1_NNLS']['top1_collapse_pct'].iloc[0])
scaden_collapse = float(B[B['method']=='B2_Scaden']['top1_collapse_pct'].iloc[0])
add('Tier B Attention top-1 collapse', 'deconv.parquet', '0%',
    f'{attn_collapse*100:.0f}%', attn_collapse < 0.01)
add('Tier B NNLS top-1 collapse', '', '95%',
    f'{nnls_collapse*100:.0f}%', abs(nnls_collapse-0.95) < 0.05)
add('Tier B Scaden top-1 collapse', '', '99%',
    f'{scaden_collapse*100:.0f}%', abs(scaden_collapse-0.99) < 0.05)

print('--- Ablation tier C ---')
C = pd.read_parquet(ROOT/'results/ablation_table/score_fusion.parquet')
c0 = float(C[C['variant']=='C0_kill+0.5*onc+0.7*prior']['mean_pct'].iloc[0])
c1 = float(C[C['variant']=='C1_kill_only']['mean_pct'].iloc[0])
c2 = float(C[C['variant']=='C2_kill+0.5*onc']['mean_pct'].iloc[0])
c3 = float(C[C['variant']=='C3_prior_only']['mean_pct'].iloc[0])
c4 = float(C[C['variant']=='C4_equal_third']['mean_pct'].iloc[0])
add('Tier C0 default mean SOC rank %', 'score_fusion.parquet', '1.6%',
    f'{c0:.2f}%', abs(c0-1.6) < 0.5)
add('Tier C1 kill-only mean rank %', '', '17.7%',
    f'{c1:.2f}%', abs(c1-17.7) < 0.5)
add('Tier C2 kill+onc mean rank %', '', '4.4%',
    f'{c2:.2f}%', abs(c2-4.4) < 0.5)
add('Tier C3 prior-only mean rank %', '', '12.6%',
    f'{c3:.2f}%', abs(c3-12.6) < 0.5)
add('Tier C4 equal mean rank %', '', '1.2%',
    f'{c4:.2f}%', abs(c4-1.2) < 0.5)

print('--- MPNN ---')
M = json.loads((ROOT/'results/ablation_mpnn_holdout/summary.json').read_text())
add('MPNN held-out mean Pearson', 'mpnn_holdout/summary.json', '0.339',
    f"{M['MPNN']['mean_pearson']:.3f}", abs(M['MPNN']['mean_pearson']-0.339) < 0.005)
add('ECFP4+Ridge mean r', '', '0.162', f"{M['ECFP4+Ridge']['mean_pearson']:.3f}",
    abs(M['ECFP4+Ridge']['mean_pearson']-0.162) < 0.005)
add('CellLine_mean r', '', '0.331', f"{M['CellLine_mean']['mean_pearson']:.3f}",
    abs(M['CellLine_mean']['mean_pearson']-0.331) < 0.005)
add('Random_emb r', '', '0.290', f"{M['Random_emb']['mean_pearson']:.3f}",
    abs(M['Random_emb']['mean_pearson']-0.290) < 0.005)

# n drug evaluations
n_evals = M['MPNN']['n_drug_evals']
add('MPNN total drug evaluations', '', '1602', n_evals, n_evals == 1602)

# ============================================================
# 7. Composition entropy
# ============================================================
print('--- Composition entropies ---')
luad_entropy = float((-(luad_comp.values * np.log(luad_comp.values+1e-9)).sum(1)).mean()
                     / np.log(luad_comp.shape[1]))
add('LUAD T3c entropy %', 't3c_luad/tcga_composition', '~87%',
    f'{luad_entropy*100:.1f}%', 0.85 < luad_entropy < 0.92)

# ============================================================
# 8. n drugs
# ============================================================
add('LIHC drugs scored final', 't3f/drug_final_score',
    1806, len(lihc_final), len(lihc_final) == 1806)
add('LUAD drugs scored final', 't3f_luad/drug_final_score',
    1806, len(luad_final), len(luad_final) == 1806)

# ============================================================
# 9. LIHC pathway sig vs LUAD pathway sig
# ============================================================
lihc_pw = pd.read_parquet(ROOT/'results/t3f/pathway_cox_tcga_lihc.parquet')
luad_pw = pd.read_parquet(ROOT/'results/t3f_luad/pathway_cox_tcga_luad.parquet')
add('LIHC pathways tested', 't3f/pathway_cox_tcga_lihc',
    1790, len(lihc_pw), len(lihc_pw) == 1790)
add('LUAD pathways tested', 't3f_luad/pathway_cox_tcga_luad',
    1790, len(luad_pw), len(luad_pw) == 1790)
add('LUAD pathways p<0.05', '', '890', (luad_pw['p']<0.05).sum(),
    (luad_pw['p']<0.05).sum() == 890)
lihc_pw_sig = (lihc_pw['p']<0.05).sum()
add('LIHC pathways p<0.05', '', 536, lihc_pw_sig, lihc_pw_sig == 536)

# ============================================================
# 10. LUAD prototype counts
# ============================================================
luad_proto = pd.read_parquet(ROOT/'results/t3c_luad/prototype_meta.parquet')
add('LUAD n prototypes', 't3c_luad/prototype_meta', 58,
    len(luad_proto), len(luad_proto) == 58)
lihc_proto = pd.read_parquet(ROOT/'results/t3c/prototype_meta.parquet')
add('LIHC n prototypes', 't3c/prototype_meta', 57,
    len(lihc_proto), len(lihc_proto) == 57)

# ============================================================
# Summary
# ============================================================
df = pd.DataFrame(audit, columns=['Claim','Source','Reported','Computed','Status'])
n_pass = (df['Status']=='PASS').sum()
n_fail = (df['Status']=='FAIL').sum()
print(f'\n=== AUDIT: {n_pass} PASS, {n_fail} FAIL out of {len(df)} ===')

if n_fail > 0:
    print('\nFAILURES:')
    for _, r in df[df['Status']=='FAIL'].iterrows():
        print(f"  - {r['Claim']}: reported={r['Reported']}, computed={r['Computed']}")

# Save report
out = ROOT/'results/audit_report.md'
with open(out, 'w') as f:
    f.write('# SCOPE-Rx — Numerical audit\n\n')
    f.write(f'**{n_pass}/{len(df)} PASS, {n_fail} FAIL**\n\n')
    f.write('| Claim | Source | Reported | Computed | Status |\n')
    f.write('|---|---|---|---|---|\n')
    for _, r in df.iterrows():
        f.write(f"| {r['Claim']} | {r['Source']} | {r['Reported']} | {r['Computed']} | **{r['Status']}** |\n")
print(f'\nFull report: {out}')
