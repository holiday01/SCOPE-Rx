"""Additional figures for the v2 PPT (external validation + consistency)."""
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from scipy.stats import pearsonr

ROOT = Path('/home/holiday01/drug_sc')
OUT  = ROOT/'results/figures'; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.titleweight':'bold',
                     'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':130,'savefig.dpi':200,'savefig.bbox':'tight'})

# ---------- Fig 11: Meta-analysis forest plot ----------
def fig_meta_forest():
    meta = pd.read_parquet(ROOT/'results/t3j/meta_analysis_pooled.parquet')
    meta = meta.sort_values('pooled_p').head(16).copy()
    meta['label'] = meta.apply(lambda r: f"#{int(r['proto'])}  {r['dominant_cell_type'][:18]}", axis=1)
    meta = meta.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8,6))
    colors = ['#2a9d8f' if h<1 else '#e76f51' for h in meta['pooled_HR']]
    ax.hlines(range(len(meta)), 0.7, meta['pooled_HR'], color='#999', lw=1)
    ax.scatter(meta['pooled_HR'], range(len(meta)), color=colors, s=80, zorder=5)
    ax.axvline(1.0, color='k', lw=0.8, ls='--')
    ax.set_yticks(range(len(meta))); ax.set_yticklabels(meta['label'], fontsize=9)
    for i,(_,r) in enumerate(meta.iterrows()):
        ax.text(r['pooled_HR']+0.01, i, f"p={r['pooled_p']:.1e}  Q={r['Q']:.1f}", va='center', fontsize=8)
    ax.set_xlabel('Pooled Hazard Ratio (3-cohort fixed-effect)')
    ax.set_title('Meta-analysis of TCGA-LIHC + GSE14520 + GSE76427 (n=763)\nTop 16 prognostic prototypes (pooled p<0.05)')
    ax.set_xlim(0.7,1.5)
    plt.tight_layout()
    plt.savefig(OUT/'fig11_meta_forest.png')
    plt.close()

# ---------- Fig 12: KM curves on GSE14520 (composite risk dichotomised) ----------
def fig_km_gse14520():
    g14_comp = pd.read_parquet(ROOT/'results/t3j/gse14520_composition.parquet')
    g14_phen = pd.read_csv(ROOT/'data/external_HCC/GSE14520_Extra_Supplement.txt', sep='\t').set_index('Affy_GSM')
    g14_phen['os_event'] = pd.to_numeric(g14_phen['Survival status'], errors='coerce')
    g14_phen['os_time']  = pd.to_numeric(g14_phen['Survival months'], errors='coerce')*30
    g14_phen['rfs_event']= pd.to_numeric(g14_phen['Recurr status'], errors='coerce')
    g14_phen['rfs_time'] = pd.to_numeric(g14_phen['Recurr months'], errors='coerce')*30
    common = g14_comp.index.intersection(g14_phen.index)
    g14_phen = g14_phen.loc[common]; g14_comp = g14_comp.loc[common]
    tcga_mv = pd.read_parquet(ROOT/'results/t3h/multivariate_cox.parquet').set_index('proto')
    n_proto = g14_comp.shape[1]
    risk_w = np.zeros(n_proto)
    for p, r in tcga_mv.iterrows():
        if float(r['p_x_mv']) < 0.1:
            risk_w[int(p)] = float(np.log(r['HR_x_uni']))
    rs = g14_comp.values.dot(risk_w)
    g14_phen['risk'] = rs
    fig, axes = plt.subplots(1,2, figsize=(11,4.5))
    for ax, ev_col, t_col, ttl in [(axes[0],'os_event','os_time','GSE14520 — Overall Survival'),
                                     (axes[1],'rfs_event','rfs_time','GSE14520 — Recurrence-Free Survival')]:
        d = g14_phen[[ev_col, t_col, 'risk']].dropna()
        d = d[d[t_col]>0]
        med = d['risk'].median(); hi = d['risk']>med
        from lifelines.statistics import logrank_test
        lr = logrank_test(d.loc[hi,t_col], d.loc[~hi,t_col],
                          d.loc[hi,ev_col], d.loc[~hi,ev_col])
        kmf = KaplanMeierFitter()
        kmf.fit(d.loc[hi,t_col]/365, d.loc[hi,ev_col], label=f'High risk (n={hi.sum()})').plot_survival_function(ax=ax, ci_show=False, color='#e76f51')
        kmf.fit(d.loc[~hi,t_col]/365, d.loc[~hi,ev_col], label=f'Low risk (n={(~hi).sum()})').plot_survival_function(ax=ax, ci_show=False, color='#2a9d8f')
        ax.set_title(f'{ttl}\nlog-rank p = {lr.p_value:.2e}')
        ax.set_xlabel('Years'); ax.set_ylabel('Survival probability')
        ax.set_xlim(0,8); ax.set_ylim(0,1.02)
        ax.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT/'fig12_km_gse14520.png')
    plt.close()

# ---------- Fig 13: Cross-cohort log(HR) scatter (TCGA vs GSE14520) ----------
def fig_cross_cohort_scatter():
    tcga = pd.read_parquet(ROOT/'results/t3h/multivariate_cox.parquet').set_index('proto')
    ext = pd.read_parquet(ROOT/'results/t3j/all_external_cox.parquet')
    g14 = ext[ext.cohort=='GSE14520_OS'].set_index('proto')
    g76 = ext[ext.cohort=='GSE76427_OS'].set_index('proto')
    fig, axes = plt.subplots(1,2, figsize=(10,4.5))
    for ax, df_ext, name in [(axes[0], g14, 'GSE14520'), (axes[1], g76, 'GSE76427')]:
        common = tcga.index.intersection(df_ext.index)
        x = np.log(tcga.loc[common,'HR_x_uni'].astype(float))
        y = np.log(df_ext.loc[common,'HR'].astype(float))
        # filter junk
        m = (np.abs(x)<3) & (np.abs(y)<3)
        x, y = x[m], y[m]
        # color by TCGA significance
        sig = tcga.loc[common,'p_x_mv'].astype(float).values[m] < 0.05
        ax.scatter(x[~sig], y[~sig], c='#bbb', s=14, alpha=0.6, label='TCGA p>0.05')
        ax.scatter(x[sig], y[sig], c='#e76f51', s=40, label='TCGA p<0.05', edgecolors='k', linewidths=0.5)
        ax.axhline(0, c='k', lw=0.5); ax.axvline(0, c='k', lw=0.5)
        # diagonal
        lim = max(abs(x).max(), abs(y).max())*1.05
        ax.plot([-lim,lim],[-lim,lim],'--',c='#999',lw=0.7)
        r,p = pearsonr(x,y)
        ax.set_title(f'{name}  vs TCGA-LIHC OS\nlog(HR) Pearson r = {r:.2f}, p = {p:.2e}')
        ax.set_xlabel('log(HR) TCGA-LIHC (mv)')
        ax.set_ylabel(f'log(HR) {name} (OS, mv)')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    plt.tight_layout()
    plt.savefig(OUT/'fig13_cross_cohort_scatter.png')
    plt.close()

# ---------- Fig 14: Method consistency heatmap (survival ranking Jaccard) ----------
def fig_method_jaccard():
    j_surv = pd.read_parquet(ROOT/'results/t3k/prognostic_method_jaccard.parquet')
    j_drug = pd.read_parquet(ROOT/'results/t3k/drug_rank_method_jaccard.parquet')
    fig, axes = plt.subplots(1,2, figsize=(11.5, 4.8))
    sns.heatmap(j_surv.astype(float), annot=True, fmt='.2f', cmap='viridis',
                vmin=0, vmax=1, ax=axes[0], cbar_kws={'label':'Jaccard'},
                annot_kws={'fontsize':10})
    axes[0].set_title('Survival ranking methods\n(top-9 prognostic prototypes Jaccard, TCGA-LIHC)')
    sns.heatmap(j_drug.astype(float), annot=True, fmt='.2f', cmap='viridis',
                vmin=0, vmax=1, ax=axes[1], cbar_kws={'label':'Jaccard'},
                annot_kws={'fontsize':10})
    axes[1].set_title('Drug-score aggregation methods\n(top-20 drugs Jaccard)')
    plt.tight_layout()
    plt.savefig(OUT/'fig14_method_jaccard.png')
    plt.close()

# ---------- Fig 15: Consensus prognostic prototypes votes bar ----------
def fig_consensus_votes():
    v = pd.read_parquet(ROOT/'results/t3k/prognostic_consistency_votes.parquet')
    v = v.sort_values('votes', ascending=False).head(15)
    v['label'] = v.apply(lambda r: f"#{int(r['proto'])}  {str(r['dominant'])[:18]}", axis=1)
    v = v.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7,5.5))
    colors = ['#2a9d8f' if x>=4 else '#e9c46a' if x>=3 else '#bbb' for x in v['votes']]
    ax.barh(range(len(v)), v['votes'], color=colors, edgecolor='k', lw=0.5)
    ax.set_yticks(range(len(v))); ax.set_yticklabels(v['label'], fontsize=9)
    ax.set_xlabel('Number of methods voting prototype into top-9')
    ax.set_xlim(0,5)
    ax.axvline(3, ls='--', color='red', lw=0.8, label='Consensus threshold')
    ax.set_title('Survival method consensus on prognostic prototypes\n(5 methods: Cox uni / Cox mv / KM / RSF / LogReg)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(OUT/'fig15_consensus_votes.png')
    plt.close()

# ---------- Fig 16: Consensus drugs (top-20 across 4 aggregation methods) ----------
def fig_consensus_drugs():
    cd = pd.read_parquet(ROOT/'results/t3k/drug_rank_consistency_votes.parquet')
    cd = cd[cd.votes>=3].sort_values('votes', ascending=False).head(15)
    cd = cd.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5,5.5))
    colors = ['#2a9d8f' if v==4 else '#e9c46a' for v in cd['votes']]
    ax.barh(range(len(cd)), cd['M3a'], color=colors, edgecolor='k', lw=0.5)
    ax.set_yticks(range(len(cd))); ax.set_yticklabels(cd['drug'], fontsize=9)
    ax.set_xlabel('Final score (z_kill + 0.5·z_onc + 0.7·z_prior)')
    ax.set_title('Drugs in top-20 of all 4 score aggregation methods\n(z-sum / rank-mean / Borda / weighted geo-mean)')
    for i,(_,r) in enumerate(cd.iterrows()):
        ax.text(r['M3a']+0.05, i, f"{r['phase'] or ''}", va='center', fontsize=8, color='#333')
    plt.tight_layout()
    plt.savefig(OUT/'fig16_consensus_drugs.png')
    plt.close()

# ---------- Fig 17: Composite risk c-index across cohorts ----------
def fig_cindex_bars():
    res = pd.read_parquet(ROOT/'results/t3j/composite_risk_results.parquet')
    fig, ax = plt.subplots(figsize=(7,4))
    cohorts = res['cohort'].tolist()
    cidx    = res['c_index'].tolist()
    ps      = res['p'].tolist()
    colors = ['#2a9d8f' if p<0.05 else '#e9c46a' if p<0.1 else '#bbb' for p in ps]
    bars = ax.bar(range(len(cohorts)), cidx, color=colors, edgecolor='k', lw=0.5)
    for i,(c,p) in enumerate(zip(cidx,ps)):
        ax.text(i, c+0.005, f'p={p:.3g}\nc={c:.2f}', ha='center', fontsize=8.5)
    ax.axhline(0.5, ls='--', color='k', lw=0.5)
    ax.set_xticks(range(len(cohorts))); ax.set_xticklabels(cohorts, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Concordance index (c-index)')
    ax.set_title('TCGA-derived composite risk score — external validation')
    ax.set_ylim(0.4, 0.8)
    plt.tight_layout()
    plt.savefig(OUT/'fig17_cindex_bars.png')
    plt.close()

# ---------- Fig 18: Deconvolution method comparison ----------
def fig_deconv_compare():
    t = pd.read_parquet(ROOT/'results/t3k/deconv_consistency.parquet')
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    ax = axes[0]
    colors = ['#2a9d8f','#e76f51','#f4a261']
    ax.bar(range(len(t)), t['entropy_pct']*100, color=colors, edgecolor='k')
    ax.set_xticks(range(len(t))); ax.set_xticklabels(t['method'], fontsize=9)
    ax.set_ylabel('Composition entropy (% of max)')
    ax.set_title('Composition entropy by deconvolution method')
    ax.axhline(70, ls='--', color='k', lw=0.5)
    for i,v in enumerate(t['entropy_pct']):
        ax.text(i, v*100+1, f'{v*100:.0f}%', ha='center', fontsize=10)
    ax = axes[1]
    ax.bar(range(len(t)), t['top1_gt_50_pct']*100, color=colors, edgecolor='k')
    ax.set_xticks(range(len(t))); ax.set_xticklabels(t['method'], fontsize=9)
    ax.set_ylabel('% patients with one prototype > 50%')
    ax.set_title('Composition collapse rate (lower = better)')
    for i,v in enumerate(t['top1_gt_50_pct']):
        ax.text(i, v*100+1, f'{v*100:.0f}%', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT/'fig18_deconv_compare.png')
    plt.close()

print('Generating v2 figures …')
for fn in [fig_meta_forest, fig_km_gse14520, fig_cross_cohort_scatter, fig_method_jaccard,
           fig_consensus_votes, fig_consensus_drugs, fig_cindex_bars, fig_deconv_compare]:
    try: fn(); print(f' ✓ {fn.__name__}')
    except Exception as e: print(f' ✗ {fn.__name__} → {e}')
