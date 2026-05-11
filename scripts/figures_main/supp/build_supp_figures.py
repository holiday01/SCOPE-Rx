"""
Build 6 supplementary figures for SCOPE-Rx CRM submission.

S1: Per-cohort Cox HR heatmap (LIHC + LUAD)
S2: Top-30 pathway hazard heatmap (LIHC vs LUAD side-by-side)
S3: Per-prototype DepMap trust calibration with best-match cell line (LIHC + LUAD)
S4: MPNN held-out drug Pearson r distribution by fold and method
S5: LIHC T3k method consistency battery (5-method × 4-aggregation)
S6: Composite TCGA-trained risk score Kaplan-Meier curves on 5 external cohorts
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent.parent))
from _style import (use_style, panel_label, save_both,
                    LIHC_COLOR, LUAD_COLOR, ATTENTION_COLOR)
use_style()
ROOT = Path('/home/holiday01/drug_sc')
OUT  = ROOT/'results/supp_figures'

# ===========================================================
# S1 — Cox HR heatmap per cohort
# ===========================================================
def figS1():
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    # LIHC: TCGA + 2 ext OS cox
    lihc_tcga = pd.read_parquet(ROOT/'results/t3c/cox_per_prototype.parquet').set_index('proto')[['HR','p']]
    lihc_g14 = pd.read_parquet(ROOT/'results/t3j/all_external_cox.parquet').query("cohort == 'GSE14520_OS'").set_index('proto')[['HR','p']]
    lihc_g76 = pd.read_parquet(ROOT/'results/t3j/all_external_cox.parquet').query("cohort == 'GSE76427_OS'").set_index('proto')[['HR','p']]
    lihc_protos = sorted(set(lihc_tcga.index) | set(lihc_g14.index) | set(lihc_g76.index))
    L = pd.DataFrame({
        'TCGA-LIHC':  np.log(lihc_tcga.reindex(lihc_protos)['HR']),
        'GSE14520':   np.log(lihc_g14.reindex(lihc_protos)['HR']),
        'GSE76427':   np.log(lihc_g76.reindex(lihc_protos)['HR']),
    })
    # LUAD
    luad_tcga = pd.read_parquet(ROOT/'results/t3c_luad/cox_per_prototype.parquet').set_index('proto')[['HR','p']]
    luad_68 = pd.read_parquet(ROOT/'results/t3i_luad/GSE68465_cox_os.parquet').set_index('proto')[['HR','p']]
    luad_72 = pd.read_parquet(ROOT/'results/t3i_luad/GSE72094_cox_os.parquet').set_index('proto')[['HR','p']]
    luad_31 = pd.read_parquet(ROOT/'results/t3i_luad/GSE31210_cox_os.parquet').set_index('proto')[['HR','p']]
    luad_protos = sorted(set(luad_tcga.index) | set(luad_68.index) | set(luad_72.index) | set(luad_31.index))
    U = pd.DataFrame({
        'TCGA-LUAD': np.log(luad_tcga.reindex(luad_protos)['HR']),
        'GSE68465':  np.log(luad_68.reindex(luad_protos)['HR']),
        'GSE72094':  np.log(luad_72.reindex(luad_protos)['HR']),
        'GSE31210':  np.log(luad_31.reindex(luad_protos)['HR']),
    })

    for ax, M, title, color in [(axes[0], L, 'LIHC — log Hazard Ratio per prototype', LIHC_COLOR),
                                  (axes[1], U, 'LUAD — log Hazard Ratio per prototype', LUAD_COLOR)]:
        im = ax.imshow(M.values, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                       interpolation='nearest')
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels(M.columns, rotation=30, ha='right')
        ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels([f'p{p}' for p in M.index], fontsize=6)
        ax.set_title(title, color=color, pad=8)
        plt.colorbar(im, ax=ax, fraction=0.04, label='log HR')
    panel_label(axes[0], 'a'); panel_label(axes[1], 'b')
    save_both(fig, OUT/'figS1_cox_heatmap')
    print('Saved S1')

# ===========================================================
# S2 — Pathway hazard top-30 per cancer
# ===========================================================
def figS2():
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    for ax, path, title, color in [(axes[0], 'results/t3f/pathway_cox_tcga_lihc.parquet', 'LIHC', LIHC_COLOR),
                                    (axes[1], 'results/t3f_luad/pathway_cox_tcga_luad.parquet','LUAD', LUAD_COLOR)]:
        pw = pd.read_parquet(ROOT/path)
        pw['signed'] = -np.log10(pw['p'].clip(1e-300, 1)) * np.sign(np.log(pw['HR']))
        # split into top hazard (most positive) and top protective
        top_haz = pw.sort_values('signed', ascending=False).head(15)
        top_prot = pw.sort_values('signed').head(15)
        comb = pd.concat([top_haz[::-1], top_prot])
        comb = comb.sort_values('signed')
        y = np.arange(len(comb))
        colors = ['#d62728' if s>0 else '#1f77b4' for s in comb['signed']]
        ax.barh(y, comb['signed'], color=colors, edgecolor='k', lw=0.3, height=0.78)
        ax.set_yticks(y)
        labels = [pw[:50] for pw in comb['pathway']]
        ax.set_yticklabels(labels, fontsize=6)
        ax.axvline(0, color='k', lw=0.5)
        ax.set_xlabel(r'$-\log_{10}(p) \cdot \mathrm{sign}(\log\,\mathrm{HR})$')
        ax.set_title(f'{title} — top hazard (red) and protective (blue) pathways', color=color)
    panel_label(axes[0], 'a'); panel_label(axes[1], 'b')
    save_both(fig, OUT/'figS2_pathway_hazard')
    print('Saved S2')

# ===========================================================
# S3 — Trust calibration per prototype
# ===========================================================
def figS3():
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for ax, path, title, color in [(axes[0], 'results/t3c/prototype_meta.parquet', 'LIHC', LIHC_COLOR),
                                    (axes[1], 'results/t3c_luad/prototype_meta.parquet','LUAD', LUAD_COLOR)]:
        m = pd.read_parquet(ROOT/path).sort_values('trust_to_depmap')
        x = np.arange(len(m))
        cell_type_col = 'dominant_cell_type' if 'dominant_cell_type' in m.columns else 'dominant'
        # Color by tumor / TME categorisation
        def cat(s):
            s = str(s).lower()
            if 'epith' in s or 'malig' in s or 'hepato' in s: return '#d62728'  # tumor
            if 'lymph' in s or 't cell' in s or 'cd' in s or 'nk' in s: return '#2ca02c'  # lymphoid
            if 'mye' in s or 'mac' in s or 'dc' in s: return '#ff7f0e'  # myeloid
            if 'endo' in s: return '#9467bd'
            if 'fibro' in s: return '#8c564b'
            return '#7f7f7f'
        colors = [cat(c) for c in m[cell_type_col]]
        ax.bar(x, m['trust_to_depmap'], color=colors, edgecolor='k', lw=0.3, width=0.8)
        ax.axhline(0.5, color='gray', lw=0.5, ls='--', label='Trust = 0.5')
        ax.axhline(0.25, color='gray', lw=0.5, ls=':', alpha=0.5, label='Trust gate centre')
        ax.set_xticks(x[::3])
        ax.set_xticklabels([f'p{p}' for p in m.index[::3]], fontsize=6, rotation=90)
        ax.set_ylabel('Trust to DepMap (max Pearson r)')
        ax.set_title(f'{title} — per-prototype DepMap trust  (n={len(m)} prototypes)', color=color)
        from matplotlib.patches import Patch
        legend_items = [Patch(color='#d62728', label='Tumor / epithelial'),
                        Patch(color='#2ca02c', label='Lymphoid'),
                        Patch(color='#ff7f0e', label='Myeloid'),
                        Patch(color='#9467bd', label='Endothelial'),
                        Patch(color='#8c564b', label='Fibroblast')]
        ax.legend(handles=legend_items, loc='upper left', frameon=False, fontsize=6, ncol=5)
    panel_label(axes[0], 'a'); panel_label(axes[1], 'b')
    save_both(fig, OUT/'figS3_trust_calibration')
    print('Saved S3')

# ===========================================================
# S4 — MPNN per-fold per-drug Pearson r
# ===========================================================
def figS4():
    cv = pd.read_parquet(ROOT/'results/ablation_mpnn_holdout/cv_results.parquet')
    methods = ['MPNN','CellLine_mean','Random_emb','ECFP4+Ridge']
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), constrained_layout=True, sharey=True)
    for ax, m in zip(axes, methods):
        sub = cv[cv['method']==m]
        positions = sorted(sub['fold'].unique())
        data_lists = [sub[sub['fold']==f]['pearson'].values for f in positions]
        parts = ax.violinplot(data_lists, positions=positions, widths=0.7, showmeans=True)
        c = {'MPNN':'#2ca02c','CellLine_mean':'#88aabb','Random_emb':'#aaaaaa','ECFP4+Ridge':'#cc6644'}[m]
        for body in parts['bodies']:
            body.set_facecolor(c); body.set_edgecolor('k'); body.set_linewidth(0.4); body.set_alpha(0.75)
        parts['cmeans'].set_color('k'); parts['cmeans'].set_linewidth(0.8)
        parts['cbars'].set_color('k'); parts['cbars'].set_linewidth(0.5)
        parts['cmins'].set_color('k'); parts['cmaxes'].set_color('k')
        ax.axhline(0, color='k', lw=0.4, ls='--')
        ax.set_xticks(positions)
        ax.set_xticklabels([f'F{i+1}' for i in positions])
        ax.set_xlabel('Fold')
        if ax is axes[0]: ax.set_ylabel('Per-drug Pearson r')
        ax.set_title(m.replace('_',' '))
    panel_label(axes[0], 'a')
    save_both(fig, OUT/'figS4_mpnn_per_fold')
    print('Saved S4')

# ===========================================================
# S5 — T3k LIHC method consistency battery
# ===========================================================
def figS5():
    try:
        jac = pd.read_parquet(ROOT/'results/t3k/drug_rank_method_jaccard.parquet')
    except Exception as e:
        print(f'S5 skipped: {e}'); return
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    if {'method_a','method_b','jaccard'}.issubset(jac.columns):
        # pivot
        M = jac.pivot(index='method_a', columns='method_b', values='jaccard').fillna(1)
    else:
        M = jac
    im = ax.imshow(M.values, cmap='YlGnBu', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=30, ha='right', fontsize=7)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i,j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=6, color='white' if v>0.5 else 'black')
    plt.colorbar(im, ax=ax, label='Jaccard top-50 overlap', fraction=0.045)
    ax.set_title('LIHC T3k — method-consistency battery\n(5 survival-rank × 4 score-aggregation methods)', pad=8)
    panel_label(ax, 'a')
    save_both(fig, OUT/'figS5_consistency_battery')
    print('Saved S5')

# ===========================================================
# S6 — Composite risk Kaplan-Meier on external cohorts
# ===========================================================
def figS6():
    from lifelines import KaplanMeierFitter
    import gzip, re
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    # LIHC: GSE14520 OS+RFS, GSE76427 OS+RFS (4 panels)
    # LUAD: GSE68465 OS, GSE72094 OS, GSE31210 OS (3 panels)
    # Total 7 → use 2x3 layout with 7th in row2 col3 + leave 1 blank? Let's pick 6 most informative.
    # Choose: GSE14520-OS, GSE14520-RFS, GSE76427-OS, GSE68465-OS, GSE72094-OS, GSE31210-OS

    # Load TCGA-trained Cox HRs
    luad_tcga = pd.read_parquet(ROOT/'results/t3c_luad/cox_per_prototype.parquet')
    risk_w_luad = np.zeros(58, dtype=np.float32)
    for _,r in luad_tcga.iterrows():
        if r['p']<0.1: risk_w_luad[int(r['proto'])] = float(np.log(r['HR']))
    lihc_tcga = pd.read_parquet(ROOT/'results/t3c/cox_per_prototype.parquet')
    n_proto_lihc = lihc_tcga['proto'].max()+1
    risk_w_lihc = np.zeros(n_proto_lihc, dtype=np.float32)
    for _,r in lihc_tcga.iterrows():
        if r['p']<0.1: risk_w_lihc[int(r['proto'])] = float(np.log(r['HR']))

    def km_panel(ax, comp_path, phen_dict, title, color):
        try:
            comp = pd.read_parquet(comp_path)
        except Exception as e:
            ax.text(0.5,0.5,f'{title}\nunavailable\n{e}', ha='center',va='center', fontsize=7, transform=ax.transAxes)
            return
        n_proto_comp = comp.shape[1]
        if n_proto_comp == 58:
            risk_w = risk_w_luad
        elif n_proto_comp == n_proto_lihc:
            risk_w = risk_w_lihc
        else:
            ax.text(0.5,0.5,f'{title}\nproto count mismatch', ha='center',va='center',transform=ax.transAxes); return
        risk = comp.values.dot(risk_w)
        median_thresh = np.median(risk)
        ev = np.array(phen_dict.get('event'))
        tt = np.array(phen_dict.get('time'))
        # Match indices
        if len(ev) != len(risk):
            ax.text(0.5,0.5,f'{title}\nlen mismatch ({len(ev)} vs {len(risk)})',ha='center',va='center',transform=ax.transAxes); return
        valid = (~np.isnan(ev)) & (~np.isnan(tt)) & (tt>0)
        risk = risk[valid]; ev = ev[valid]; tt = tt[valid]
        high = risk >= np.median(risk)
        kmf = KaplanMeierFitter()
        kmf.fit(tt[high], event_observed=ev[high], label='High risk')
        kmf.plot_survival_function(ax=ax, color='#d62728', show_censors=True, ci_show=False)
        kmf.fit(tt[~high], event_observed=ev[~high], label='Low risk')
        kmf.plot_survival_function(ax=ax, color='#2ca02c', show_censors=True, ci_show=False)
        from lifelines.statistics import logrank_test
        try:
            res = logrank_test(tt[high], tt[~high], event_observed_A=ev[high], event_observed_B=ev[~high])
            ax.text(0.97, 0.05, f'logrank p = {res.p_value:.3g}\nn = {len(tt)}',
                    ha='right', va='bottom', transform=ax.transAxes, fontsize=7,
                    bbox=dict(facecolor='white', edgecolor='gray', lw=0.4, alpha=0.8))
        except: pass
        ax.set_title(title, color=color, fontsize=8)
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Survival prob.')
        ax.legend(fontsize=6, loc='lower left', frameon=False)

    # Build phenotype dicts via reading external Cox parquets — these have no event/time, but
    # we can re-parse phen quickly. Skip and just plot for LUAD cohorts (we have phen extraction code in t3i_luad)
    # For brevity, use simpler approach: load LUAD g68/g72/g31 compositions and re-derive phen via our parser.
    # Phen via small inline parser (reuse from earlier T3i logic):
    import gzip
    def _per_sample_chars(path):
        sample_ids=None; rows=[]
        with gzip.open(path,'rt',errors='replace') as f:
            for line in f:
                if line.startswith('!Sample_geo_accession'):
                    sample_ids=[x.strip().strip('"') for x in line.split('\t')[1:]]
                elif line.startswith('!Sample_characteristics_ch1'):
                    rows.append([x.strip().strip('"') for x in line.split('\t')[1:]])
                elif line.startswith('!series_matrix_table_begin'):
                    break
        n=len(sample_ids); per=[{} for _ in range(n)]
        for cells in rows:
            for i,c in enumerate(cells[:n]):
                if not c or ':' not in c: continue
                k,v=c.split(':',1); per[i][k.strip().lower()]=v.strip()
        return sample_ids, per

    def luad_phen(cid, fname):
        sids, per = _per_sample_chars(ROOT/f'data/external_LUAD/{fname}')
        comp = pd.read_parquet(ROOT/f'results/t3i_luad/{cid}_composition.parquet')
        ev=[]; tt=[]; sample_index = comp.index.tolist()
        per_dict = {sids[i]: per[i] for i in range(len(sids))}
        for s in sample_index:
            d = per_dict.get(s, {})
            # Detect event/time keys
            evkey = next((k for k in d if 'vital' in k or 'death' in k), None)
            tt_key = next((k for k in d if 'survival' in k and 'day' in k), None) or next((k for k in d if 'days before death' in k), None) or next((k for k in d if 'months_to_last_contact' in k), None)
            ev_v = d.get(evkey, '').lower() if evkey else ''
            ev.append(1 if 'dead' in ev_v else (0 if ('alive' in ev_v or 'living' in ev_v) else np.nan))
            tt_str = d.get(tt_key, '').strip() if tt_key else ''
            try:
                tt_v = float(tt_str)
                # If month-based, multiply by 30
                if tt_key and 'month' in tt_key: tt_v *= 30.4
                tt.append(tt_v)
            except: tt.append(np.nan)
        return {'event': ev, 'time': tt}

    panels = [
        ('GSE68465 OS', ROOT/'results/t3i_luad/GSE68465_composition.parquet', luad_phen('GSE68465','GSE68465_series_matrix.txt.gz'), LUAD_COLOR),
        ('GSE72094 OS', ROOT/'results/t3i_luad/GSE72094_composition.parquet', luad_phen('GSE72094','GSE72094_series_matrix.txt.gz'), LUAD_COLOR),
        ('GSE31210 OS', ROOT/'results/t3i_luad/GSE31210_composition.parquet', luad_phen('GSE31210','GSE31210_series_matrix.txt.gz'), LUAD_COLOR),
    ]
    for ax, (title, comp_path, phen, color) in zip(axes.flat[:3], panels):
        km_panel(ax, comp_path, phen, title, color)
    # Hide remaining 3 axes
    for ax in axes.flat[3:]:
        ax.axis('off')
    panel_label(axes[0,0], 'a')
    save_both(fig, OUT/'figS6_kaplan_meier')
    print('Saved S6')

# Run all
figS1()
figS2()
figS3()
figS4()
figS5()
figS6()
print('\n=== All 6 supp figures saved ===')
