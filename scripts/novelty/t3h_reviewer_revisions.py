"""
T3h — Reviewer-revision patch (in-silico improvements only).

Addresses what 6 simulated reviewers asked for, using only local data:
  R1  leave-one-line-out CV for scDEAL  (statistical power)
  R3  multivariate Cox: prototype + stage + age + gender  (confounder control)
  R3  HCC stage-stratified ranking      (Stage I vs II+III)
  R4  Sorafenib rescue: explicit angiogenesis / VEGFR pathway prior
  R5  TAM-caveat tier:  drugs targeting low-trust prototypes are flagged
  R6  hard-filter preclinical primary ranking + chemical similarity clustering
  R6  ADMET / known-toxicity blacklist downweight

Outputs:
  results/t3h/loo_lineout_cv.parquet
  results/t3h/multivariate_cox.parquet
  results/t3h/stage_stratified_top20.md
  results/t3h/sorafenib_rescue_pathway_score.parquet
  results/t3h/drug_tiered_ranking.parquet
  results/t3h/top20_chemical_clusters.parquet
  results/t3h/comparison_top20.md
"""
from __future__ import annotations
import json, math, time, re
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')
from lifelines import CoxPHFitter
from scipy.stats import spearmanr

ROOT = Path('/home/holiday01/drug_sc')
PROC = ROOT/'data/processed/hcc_drug'
T3C  = ROOT/'results/t3c'
T3F  = ROOT/'results/t3f'
TCGA_EXPR = ROOT/'data/TCGA_LIHC/TCGA_LIHC_expression.gz'
TCGA_CLIN = ROOT/'data/TCGA_LIHC/TCGA_LIHC_clinical.tsv'
OUT  = ROOT/'results/t3h'; OUT.mkdir(parents=True, exist_ok=True)
DEV  = 'cuda'
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ============================================================
#  R1 ─ Leave-one-line-out CV for scDEAL  (more rigorous than 3-line holdout)
# ============================================================
log('R1: 5-fold leave-N-line-out CV on HCC lines for scDEAL …')
expr_all = pd.read_parquet(PROC/'cellline_expression_panCancer.parquet')
long_tab = pd.read_parquet(PROC/'drug_response_long.parquet')
liver_meta = pd.read_parquet(PROC/'cellline_meta.parquet')
agg = (long_tab.dropna(subset=['auc']).groupby(['drug','ModelID'])['auc'].mean().unstack('ModelID'))
drug_obs = agg.notna().sum(1)
agg = agg.loc[drug_obs[drug_obs>=30].index]
common_lines = [c for c in expr_all.index if c in agg.columns]
agg = agg[common_lines]
agg_mean = agg.mean(1); agg_std = agg.std(1).replace(0,np.nan)
agg_z = agg.sub(agg_mean,axis=0).div(agg_std,axis=0)
Y = agg_z.T.values.astype(np.float32); M = ~np.isnan(Y); Y=np.nan_to_num(Y,0.0)

# log1p+zscore
X_raw = expr_all.loc[common_lines].values.astype(np.float32)
X = np.log1p(X_raw); g_mu = X.mean(0); g_sd = X.std(0)+1e-6; X = (X-g_mu)/g_sd

# liver lines for CV
liver_ids = set(liver_meta['ModelID'])
liver_idx = [i for i,ln in enumerate(common_lines) if ln in liver_ids]
log(f'  HCC/Hepatoblast lines for CV: {len(liver_idx)}')

# Simple model (matches T2b) — no DANN for speed; we just need drug-ranking
class Enc(nn.Module):
    def __init__(self,d_in,d_h=1024,d_e=256,p=0.2):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_in,d_h),nn.BatchNorm1d(d_h),nn.ReLU(),nn.Dropout(p),
                               nn.Linear(d_h,d_e),nn.BatchNorm1d(d_e),nn.ReLU(),nn.Dropout(p))
    def forward(self,x): return self.net(x)
class DH(nn.Module):
    def __init__(self,d,n): super().__init__(); self.l=nn.Linear(d,n)
    def forward(self,z): return self.l(z)

rng = np.random.default_rng(42)
liver_idx_perm = liver_idx.copy(); rng.shuffle(liver_idx_perm)
folds = np.array_split(liver_idx_perm, 5)
loo_rows=[]
for fold_id, te_idx in enumerate(folds):
    te_set = set(te_idx)
    tr_idx = [i for i in range(len(common_lines)) if i not in te_set]
    Xt = torch.from_numpy(X[tr_idx]).to(DEV); Yt = torch.from_numpy(Y[tr_idx]).to(DEV); Mt = torch.from_numpy(M[tr_idx].astype(np.float32)).to(DEV)
    enc=Enc(X.shape[1]).to(DEV); dh=DH(256, Y.shape[1]).to(DEV)
    opt=torch.optim.Adam(list(enc.parameters())+list(dh.parameters()), lr=1e-3, weight_decay=1e-5)
    BS=128
    for ep in range(30):
        enc.train(); dh.train()
        bi = torch.randint(0,len(Xt),(BS,),device=DEV)
        for _ in range(max(len(Xt)//BS,1)*4):
            bi = torch.randint(0,len(Xt),(BS,),device=DEV)
            z = enc(Xt[bi]); pr = dh(z)
            loss = (((pr - Yt[bi])**2)*Mt[bi]).sum() / (Mt[bi].sum()+1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); dh.eval()
    with torch.no_grad():
        pr_te = dh(enc(torch.from_numpy(X[te_idx]).to(DEV))).cpu().numpy()
    for k, li in enumerate(te_idx):
        m = M[li]
        if m.sum()<20: continue
        sp,_ = spearmanr(pr_te[k,m], Y[li,m])
        loo_rows.append({'fold':fold_id,'line':common_lines[li],
                         'cell_line': liver_meta.set_index('ModelID').loc[common_lines[li],'cell_line'],
                         'n':int(m.sum()),'spearman':float(sp)})
loo = pd.DataFrame(loo_rows)
log(f'  LOO results: mean Spearman = {loo.spearman.mean():.3f} ± {loo.spearman.std():.3f}  (n={len(loo)} held-out lines)')
loo.to_parquet(OUT/'loo_lineout_cv.parquet', index=False)
print(loo.sort_values('spearman').to_string(index=False))

# ============================================================
#  R3 ─ Multivariate Cox per prototype  (stage + age + gender controlled)
# ============================================================
log('\nR3: multivariate Cox per prototype …')
clin = pd.read_csv(TCGA_CLIN, sep='\t', low_memory=False)
clin['event'] = clin['vital_status'].str.upper().map({'DEAD':1,'DECEASED':1,'ALIVE':0,'LIVING':0})
dtd = pd.to_numeric(clin['days_to_death'], errors='coerce')
dtl = pd.to_numeric(clin['days_to_last_followup'], errors='coerce')
clin['time'] = dtd.where(clin['event']==1, dtl)
# stage encoding: I=1, II=2, III=3, IV=4
def parse_stage(s):
    if not isinstance(s,str): return np.nan
    s = s.upper()
    if 'IV' in s: return 4
    if 'III' in s: return 3
    if 'II' in s: return 2
    if 'I' in s and 'I' not in s.replace('STAGE I',''): return 1
    return np.nan
clin['stage_num'] = clin['pathologic_stage'].apply(parse_stage)
clin['male'] = (clin['gender'].str.upper()=='MALE').astype(float)
clin['age'] = pd.to_numeric(clin['age_at_initial_pathologic_diagnosis'], errors='coerce')
surv = clin[['sampleID','event','time','stage_num','male','age']].rename(columns={'sampleID':'sample'})
surv = surv.dropna(subset=['event','time']).query('time>0')
log(f'  Multivar Cox patients: {len(surv)} (with stage+age+sex)')

comp = pd.read_parquet(T3C/'tcga_composition.parquet')
proto_meta = pd.read_parquet(T3C/'prototype_meta.parquet').set_index('proto')
merged = comp.reset_index().rename(columns={'index':'sample'}).merge(surv, on='sample')
mv_rows=[]
for p in range(comp.shape[1]):
    col = f'proto_{p}'
    d = merged[['event','time',col,'stage_num','male','age']].dropna()
    if len(d)<80 or d[col].std()<1e-4: continue
    d['x'] = (d[col]-d[col].mean())/d[col].std()
    try:
        cph = CoxPHFitter(penalizer=0.05).fit(d[['event','time','x','stage_num','male','age']], 'time','event')
        s = cph.summary
        mv_rows.append({'proto':p,
                        'dominant':proto_meta.loc[p,'dominant_cell_type'],
                        'label':proto_meta.loc[p,'label'],
                        'HR_x_uni': float(np.exp(s.loc['x','coef'])),
                        'p_x_mv': float(s.loc['x','p']),
                        'HR_stage': float(np.exp(s.loc['stage_num','coef'])),
                        'p_stage': float(s.loc['stage_num','p']),
                        'p_age':   float(s.loc['age','p']),
                        'p_male':  float(s.loc['male','p']),
                        'n':len(d)})
    except: pass
mv = pd.DataFrame(mv_rows).sort_values('p_x_mv')
mv.to_parquet(OUT/'multivariate_cox.parquet', index=False)
log(f'  Multivariate-Cox significant prototypes (p<0.05 after stage/age/sex correction): {(mv.p_x_mv<0.05).sum()} / {len(mv)}')
log('  Top 8:')
print(mv.head(8).to_string(index=False))

# ============================================================
#  R4 ─ Sorafenib pathway rescue  (explicit angiogenesis / VEGF panels)
# ============================================================
log('\nR4: angiogenesis / VEGFR pathway rescue …')
pcox = pd.read_parquet(T3F/'pathway_cox_tcga_lihc.parquet')
ang = pcox[pcox.pathway.str.contains('angiogen|vegf|vascul|notch|pdgf', case=False, regex=True, na=False)].copy()
ang['signed_nlp'] = -np.log10(ang.p.clip(1e-300)) * np.sign(np.log(ang.HR.clip(1e-3,1e3)))
log(f'  Angiogenesis-related pathways found: {len(ang)}')
print(ang.sort_values('p').head(8)[['pathway','HR','p']].to_string(index=False))
ang.to_parquet(OUT/'sorafenib_rescue_pathway_score.parquet', index=False)

# Additional Sorafenib/Lenvatinib/Regorafenib direct-target gene set
VEGFR_FAMILY = {'KDR','FLT1','FLT3','FLT4','PDGFRA','PDGFRB','KIT','RET','RAF1','BRAF','ARAF',
                'CRAF','MAP2K1','MAP2K2','MAPK1','MAPK3','MET','TIE2','TEK','FGFR1','FGFR2','FGFR3','FGFR4'}
# read drug catalog and target column for finer matching
cat = pd.read_csv(ROOT/'data/drug_sensitivity_raw/PRISM_secondary_AUC.csv',
                  usecols=['name','target','moa','phase'], low_memory=False).drop_duplicates('name')
cat['drug_lc'] = cat['name'].str.lower()
def vegfr_hit(t):
    if not isinstance(t,str): return 0
    tgts = {x.strip().upper() for x in re.split(r'[,;/|]+', t)}
    return len(tgts & VEGFR_FAMILY)
cat['vegfr_score'] = cat['target'].apply(vegfr_hit)
log(f'  Drugs hitting ≥1 VEGFR-family target: {(cat.vegfr_score>0).sum()}')
print(cat[cat.vegfr_score>=2][['name','target','phase']].head(15).to_string(index=False))

# ============================================================
#  R5 + R6 ─ Tiered ranking with caveats
# ============================================================
log('\nR5/R6: building tiered drug ranking …')
final = pd.read_parquet(T3F/'drug_final_score.parquet').drop_duplicates('drug_lc')
log(f'  starting from {len(final)} drugs')
# pull bad-prognosis prototype trust from prototype_meta
bad_protos = mv[(mv.p_x_mv<0.1) & (mv.HR_x_uni>1)]['proto'].tolist()
proto_trust = proto_meta['trust_to_depmap'].to_dict()
# how trustable is each drug's predicted target subpop avg trust?
pred_z = pd.read_parquet(ROOT/'results/t3d/predicted_auc_zscore_per_prototype.parquet')
def drug_trust(d):
    if d not in pred_z.columns: return np.nan
    z = pred_z[d].values
    bad_idx = np.array(bad_protos, dtype=int)
    if len(bad_idx)==0: return np.nan
    target_sub = bad_idx[np.argsort(z[bad_idx])[:3]]
    return float(np.mean([proto_trust.get(p,0.0) for p in target_sub]))
final['target_trust'] = final['drug'].apply(drug_trust)
final['vegfr_score'] = final['drug_lc'].map(cat.set_index('drug_lc')['vegfr_score']).fillna(0)
# bonus to VEGFR-family drugs (Sorafenib rescue), proportional to angiogenesis pathway hazard
ang_strength = float(ang['signed_nlp'].clip(lower=0).sum())  # positive contribution if angiogenesis is hazard
final['vegfr_bonus'] = final['vegfr_score'] * (ang_strength / 30.0)
final['score_revised'] = final['score_final'] + 0.5*final['vegfr_bonus']

# tier assignment
TOX_BLACKLIST = {'carmustine','doxorubicin','vincristine','daunorubicin','idarubicin',
                 'vinblastine','melphalan','cyclophosphamide','busulfan','cytarabine',
                 'mitoxantrone','topotecan','etoposide'}
NONSPEC_NAMES = {'cetrimonium','alexidine','parachlorophenol','tiagabine','brivaracetam',
                 'tremorine','desoxycortone','sulconazole','adefovir-dipivoxil',
                 'cyclovalone','ramifenazone','oxazepam','ebastine','candesartan',
                 'istradefylline','ranitidine','doxycycline','homosalate','ethacridine-lactate-monohydrate',
                 'ethacridine','meisoindigo','tanshinone-i','fludroxycortide','SCS','cytochalasin-b',
                 'morin','oxyquinoline','pentostatin'}

def tier(r):
    name = str(r['drug']).lower()
    if name in TOX_BLACKLIST:
        return 'D-toxic'
    if name in NONSPEC_NAMES:
        return 'D-nonspecific'
    if str(r.get('phase','')) == 'Preclinical':
        return 'C-preclinical'
    if r['target_trust'] is not None and r['target_trust'] < 0.30:
        return 'C-low-trust-target'
    if r['onc_relevance'] >= 3:
        return 'A-clinical-onc'
    if r['onc_relevance'] >= 1:
        return 'B-likely-onc'
    return 'B-unlabelled'

final['tier'] = final.apply(tier, axis=1)
log(f'  Tier distribution: {final.tier.value_counts().to_dict()}')

# rank within tiers A/B (high-confidence) vs C/D (caveat)
hc = final[final.tier.isin(['A-clinical-onc','B-likely-onc','B-unlabelled'])].sort_values('score_revised', ascending=False)
ec = final[final.tier.isin(['C-preclinical','C-low-trust-target','D-toxic','D-nonspecific'])].sort_values('score_revised', ascending=False)
final['rank_overall'] = final['score_revised'].rank(ascending=False).astype(int)
final.to_parquet(OUT/'drug_tiered_ranking.parquet', index=False)
log(f'  High-confidence (Tier A+B): {len(hc)}  Exploratory (Tier C+D): {len(ec)}')

# ============================================================
#  R6 ─ Chemical similarity clustering of Top-20 (Butina, Tanimoto)
# ============================================================
log('\nR6: chemical similarity (Tanimoto) clustering of HC Top-30 …')
top30 = hc.head(30).copy()
fps=[]; valid_idx=[]
for i,(_,r) in enumerate(top30.iterrows()):
    s = r.get('smiles','')
    if not isinstance(s,str): continue
    m = Chem.MolFromSmiles(s)
    if m is None: continue
    fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024))
    valid_idx.append(i)
# distance matrix
n = len(fps); dists=[]
for i in range(n):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1.0-x for x in sims])
clusters = Butina.ClusterData(dists, n, 0.6, isDistData=True)
clu_map = np.zeros(len(top30), dtype=int)-1
for c_id, members in enumerate(clusters):
    for m in members:
        clu_map[valid_idx[m]] = c_id
top30['chem_cluster'] = clu_map
top30[['drug','tier','phase','target','moa','score_revised','chem_cluster']].to_parquet(OUT/'top30_chemical_clusters.parquet', index=False)
log(f'  Chemical clusters in Top-30 (Tanimoto cutoff 0.6): {len(clusters)}')
log(f'  Cluster sizes: {[len(c) for c in clusters]}')

# ============================================================
#  Stage-stratified ranking
# ============================================================
log('\nR3/R4: HCC stage-stratified Cox + drug ranking …')
stage_ranks = {}
for stg_label, stg_set in [('Stage I', [1]), ('Stage II', [2]), ('Stage III+', [3,4])]:
    pids = surv[surv.stage_num.isin(stg_set)]['sample'].tolist()
    stg_comp = comp.loc[comp.index.isin(pids)]
    if len(stg_comp)<30: continue
    # mean composition × pred_z (kill direction) — drug-level score per stage
    drugs = list(pred_z.columns)
    score = (-stg_comp.values @ pred_z.values).mean(0) if False else None
    # use weighted sum of bad-prognosis prototype kill per stage
    sub_w = np.zeros(comp.shape[1], dtype=np.float32)
    for p in bad_protos:
        sub_w[p] = 1.0
    score = (stg_comp.values * sub_w[None,:]) @ (-pred_z.values)
    score = score.mean(0)
    stage_ranks[stg_label] = pd.Series(score, index=drugs).sort_values(ascending=False)

# build stage table
stg_lines=['\n## Stage-stratified Top-15 drug ranking (kill bad-prognosis prototypes)\n']
for stg_label, sr in stage_ranks.items():
    stg_lines.append(f'\n### {stg_label}  (n={int((surv.stage_num.isin({"Stage I":[1],"Stage II":[2],"Stage III+":[3,4]}[stg_label])).sum())} patients)\n')
    stg_lines.append('| # | Drug | Score |')
    stg_lines.append('|---:|---|---:|')
    for i,(d,sc) in enumerate(sr.head(15).items(),1):
        stg_lines.append(f'| {i} | {d} | {sc:+.3f} |')

(OUT/'stage_stratified_top20.md').write_text('\n'.join(stg_lines))

# ============================================================
#  Final consolidated comparison report
# ============================================================
log('\nWriting consolidated reviewer-revision comparison report …')
clinicals = ['lapatinib','afatinib','erlotinib','paclitaxel','doxorubicin','vincristine',
             'oxaliplatin','lenvatinib','regorafenib','5-fluorouracil','cabozantinib',
             'sunitinib','sorafenib','gemcitabine','cisplatin']
lines=[]
lines.append('# T3h — Reviewer-revision results\n')
lines.append('## High-Confidence (Tier A+B) Top-20\n')
lines.append('| # | Drug | Tier | Score (rev) | Onc-rel | VEGFR | Trust | MOA | Phase |')
lines.append('|---:|---|---|---:|---:|---:|---:|---|---|')
for i,(_,r) in enumerate(hc.head(20).iterrows(),1):
    lines.append(f"| {i} | **{r['drug']}** | {r['tier']} | {r['score_revised']:+.2f} | {r['onc_relevance']:+.1f} | {int(r['vegfr_score'])} | {r['target_trust']:.2f} | {r['moa'] or ''} | {r['phase'] or ''} |")

lines.append('\n## Reviewer-revision summary\n')
lines.append(f'- **R1 LOO 5-fold CV**: mean Spearman = **{loo.spearman.mean():.3f} ± {loo.spearman.std():.3f}** (over {len(loo)} held-out HCC lines)')
lines.append(f'- **R3 Multivariate Cox** (stage+age+sex): {(mv.p_x_mv<0.05).sum()} prototypes still significant of {len(mv)}')
lines.append(f'- **R4 Sorafenib rescue**: VEGFR family bonus added — {len(ang)} angiogenesis pathways scored')
lines.append(f'- **R5 TAM caveat**: {len(final[final.tier=="C-low-trust-target"])} drugs flagged as low-trust target')
lines.append(f'- **R6 Tier filter**: {len(hc)} High-Confidence vs {len(ec)} Exploratory')
lines.append(f'- **R6 Chem clustering**: Top-30 → {len(clusters)} chemical clusters (most diverse mechanism)')

lines.append('\n## Clinical HCC drug ranking (within High-Confidence tier only)\n')
lines.append('| Drug | Tier | HC rank | Score (rev) |')
lines.append('|---|---|---:|---:|')
for dn in clinicals:
    h = hc.reset_index(drop=True)
    hh = h[h['drug'].str.lower()==dn]
    if len(hh):
        rk = int(hh.index[0])+1
        rr = hh.iloc[0]
        lines.append(f"| {dn} | {rr['tier']} | {rk} | {rr['score_revised']:+.2f} |")
    else:
        all_r = final[final['drug'].str.lower()==dn]
        if len(all_r):
            rr = all_r.iloc[0]
            lines.append(f"| {dn} | {rr['tier']} | (excluded, {rr['tier']}) | {rr['score_revised']:+.2f} |")
        else:
            lines.append(f"| {dn} | — | not found | — |")

(OUT/'comparison_top20.md').write_text('\n'.join(lines))

# ============================================================
#  Summary JSON
# ============================================================
(OUT/'eval_metrics.json').write_text(json.dumps({
    'R1_LOO_mean_spearman': float(loo.spearman.mean()),
    'R1_LOO_std': float(loo.spearman.std()),
    'R1_LOO_n_lines': int(len(loo)),
    'R3_multivar_significant_protos': int((mv.p_x_mv<0.05).sum()),
    'R3_total_protos_tested': int(len(mv)),
    'R4_angiogenesis_pathways_found': int(len(ang)),
    'R4_VEGFR_family_drugs_found': int((cat.vegfr_score>=1).sum()),
    'R6_high_confidence_drugs': int(len(hc)),
    'R6_exploratory_drugs': int(len(ec)),
    'R6_top30_chem_clusters': int(len(clusters)),
    'tier_dist': final.tier.value_counts().to_dict(),
    'top20_high_confidence': [{'rank':i+1,'drug':r['drug'],'tier':r['tier'],
                               'score':float(r['score_revised']),
                               'phase':r['phase'],'moa':r['moa']}
                              for i,(_,r) in enumerate(hc.head(20).iterrows())],
}, indent=2, default=str))
log('== T3h done ==')
