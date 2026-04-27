"""
T3d — Survival-anchored drug scoring.

Pipeline:
  1. For every cell-state prototype (55): use T2b scDEAL encoder+dhead to predict
     per-drug AUC z-score from that prototype's pseudobulk expression.
  2. For every prototype: load Cox(OS) hazard from T3c per-prototype analysis.
     Bad-prognosis direction => we want drugs with LOW predicted AUC (strong kill).
  3. Per-drug prognosis-weighted score:
        S(d) = Σ_p  (max(0, -log10(p_p)) × sign(log HR_p)) × (-predicted_AUC_z_{p,d})
     → higher S(d) = drug reverses the hazard-driving prototypes' signature.
  4. Per-patient ranking = Σ_p  composition_p × (-predicted_AUC_z_{p,d}) × sign(hazard_p)
  5. Clinical sanity: Sorafenib / Lenvatinib / etc. vs top surprises.

Outputs:
  results/t3d/
    predicted_auc_per_prototype.parquet  (55 × n_drugs)
    drug_score_cohort.parquet            (drug-level prognosis-weighted rank)
    drug_score_per_patient.parquet       (423 × n_drugs)
    eval_metrics.json
"""
from __future__ import annotations
import json, time, math
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn

ROOT = Path('/home/holiday01/drug_sc')
PROC = ROOT/'data/processed/hcc_drug'
T3C  = ROOT/'results/t3c'
OUT  = ROOT/'results/t3d'; OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT/'checkpoints'
DEV  = 'cuda'

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Load T2b scDEAL ----------
log('Loading T2b scDEAL …')
t2b = torch.load(CKPT/'t2b_scdeal.pt', map_location='cpu', weights_only=False)
drugs = list(t2b['drugs'])
genes = list(t2b['genes'])
g_mu = np.asarray(t2b['gene_mu']); g_sd = np.asarray(t2b['gene_sd'])
drug_mean = np.asarray(t2b['drug_mean']); drug_std = np.asarray(t2b['drug_std'])

class Encoder(nn.Module):
    def __init__(self,d_in,d_hid,d_emb,p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in,d_hid), nn.BatchNorm1d(d_hid), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(d_hid,d_emb), nn.BatchNorm1d(d_emb), nn.ReLU(), nn.Dropout(p))
    def forward(self,x): return self.net(x)
class DrugHead(nn.Module):
    def __init__(self,d_emb,n): super().__init__(); self.lin=nn.Linear(d_emb,n)
    def forward(self,z): return self.lin(z)

enc = Encoder(len(genes),1024,256).to(DEV); enc.load_state_dict(t2b['enc']); enc.eval()
dhead = DrugHead(256, len(drugs)).to(DEV); dhead.load_state_dict(t2b['dhead']); dhead.eval()
log(f'scDEAL loaded — drugs={len(drugs)}  genes={len(genes)}')

# ---------- 2. Prototype pseudobulk + scDEAL prediction ----------
log('Predicting drug AUC z for each prototype (55) …')
pb = pd.read_parquet(T3C/'prototype_expression.parquet')
assert list(pb.columns) == genes, 'gene order mismatch (T3c built on same gene universe)'
Xp = pb.values.astype(np.float32)   # already log1p-CP10K
Xp_z = (Xp - g_mu) / g_sd
with torch.no_grad():
    pred_z = dhead(enc(torch.from_numpy(Xp_z).to(DEV))).cpu().numpy()  # (55, n_drugs)
pred_auc_raw = pred_z * drug_std[None,:] + drug_mean[None,:]
pd.DataFrame(pred_z, columns=drugs).to_parquet(OUT/'predicted_auc_zscore_per_prototype.parquet')
pd.DataFrame(pred_auc_raw, columns=drugs).to_parquet(OUT/'predicted_auc_raw_per_prototype.parquet')
log(f'pred_z shape {pred_z.shape}  raw_auc range [{pred_auc_raw.min():.2f}, {pred_auc_raw.max():.2f}]')

# ---------- 3. Load prototype Cox results ----------
cox = pd.read_parquet(T3C/'cox_per_prototype.parquet')
proto_meta = pd.read_parquet(T3C/'prototype_meta.parquet')
log(f'Cox rows: {len(cox)}  significant (p<0.05): {(cox["p"]<0.05).sum()}')
# per-SD Cox HR is now numerically sensible (0.5-2)
cox['logHR'] = np.log(cox['HR'].astype(float).clip(lower=1e-6, upper=1e6))
cox['weight_raw'] = (-np.log10(cox['p'].astype(float).clip(lower=1e-300))).clip(upper=10) * np.sign(cox['logHR'])
cox.loc[cox['p'] > 0.1, 'weight_raw'] = 0
# **Asymmetric weighting**: bad-prognosis prototypes full weight; good-prognosis prototypes 0.3×
# so the pipeline prioritises reversing hazard more than preserving protective biology.
cox['weight'] = np.where(cox['weight_raw']>0, cox['weight_raw'], cox['weight_raw']*0.3)

# **Trust weighting**: downweight prototypes whose pseudobulk is far from any DepMap cell line
# (scDEAL AUC predictions extrapolate beyond its training distribution for TAM/T/endothelial).
proto_meta = proto_meta.set_index('proto')
trust_map = proto_meta['trust_to_depmap'].to_dict()
label_map = proto_meta['label'].to_dict()

cox_weights = np.zeros(pred_z.shape[0], dtype=np.float32)
trust_weights = np.ones(pred_z.shape[0], dtype=np.float32)
for _, r in cox.iterrows():
    p = int(r['proto'])
    cox_weights[p]   = r['weight']
    # sigmoid-shaped trust: 0.1 correlation → ~0.2 weight;  0.5 correlation → ~0.9 weight
    t = trust_map.get(p, 0.0)
    trust_weights[p] = 1.0 / (1.0 + math.exp(-(t - 0.25)*8))
# combined per-proto weight
combined_w = cox_weights * trust_weights
log(f'Cox weights (nonzero): {(cox_weights!=0).sum()}  sum abs = {np.abs(cox_weights).sum():.2f}')
log(f'Trust weights range: [{trust_weights.min():.3f}, {trust_weights.max():.3f}]')
log(f'Combined (cox*trust) range: [{combined_w.min():.3f}, {combined_w.max():.3f}]')

# ---------- 4. Cohort-level drug ranking (survival-anchored) ----------
# S(d) = Σ_p  weight_p  *  (-pred_z[p,d])
#   logic:
#     weight_p > 0  => high prototype abundance predicts BAD prognosis (higher risk)
#                      want drugs that KILL that prototype => low pred AUC => -pred_z large
#     weight_p < 0  => high abundance predicts GOOD prognosis => don't kill it => want high AUC
S_drug = combined_w @ (-pred_z)   # (n_drugs,) — uses cox_weights × trust_weights
drug_score = pd.DataFrame({'drug':drugs, 'score':S_drug})
bad = cox_weights > 0
if bad.any():
    drug_score['mean_predAUC_bad_protos'] = pred_auc_raw[bad].mean(0)
    drug_score['mean_predAUC_all_protos'] = pred_auc_raw.mean(0)
# Also a *malignant-hepatocyte-only* score (the prototypes we trust most for direct killing)
mal_mask = np.array([('Epithelial_tumor' in label_map.get(i,'')) for i in range(pred_z.shape[0])])
if mal_mask.any():
    drug_score['score_malignant_only'] = (mal_mask.astype(np.float32) * trust_weights) @ (-pred_z)
drug_score = drug_score.sort_values('score', ascending=False)
drug_score.to_parquet(OUT/'drug_score_cohort.parquet', index=False)
log('Top 20 drugs by survival-anchored score:')
log(drug_score.head(20).to_string())
log('\nBottom 10 (drugs with worst survival score — avoid):')
log(drug_score.tail(10).to_string())

# ---------- 5. Per-patient ranking (composition-weighted + trust + asymmetric Cox) ----------
log('\nComputing per-patient drug scores …')
comp = pd.read_parquet(T3C/'tcga_composition.parquet')  # 423 × 55
C = comp.values.astype(np.float32)
# Per-patient weight per proto = composition × trust × cox_sign_with_asymmetric_penalty
# good-prognosis proto gets 0.3× penalty so we don't over-punish killing protective subpops
prog_w = np.where(cox_weights>0, 1.0, np.where(cox_weights<0, -0.3, 0.0)).astype(np.float32)
pw = prog_w * trust_weights
weighted = C * pw[None,:]         # (423, 55)
S_patient = weighted @ (-pred_z)  # (423, n_drugs)
S_patient_df = pd.DataFrame(S_patient, index=comp.index, columns=drugs)
S_patient_df.to_parquet(OUT/'drug_score_per_patient.parquet')
log(f'Per-patient score: {S_patient_df.shape}')

# Rank consensus across cohort
med_rank = S_patient_df.rank(axis=1, ascending=False).median(0).sort_values()
consensus = pd.DataFrame({'drug':med_rank.index, 'median_rank':med_rank.values})
consensus.to_parquet(OUT/'drug_consensus_rank.parquet', index=False)
log('\nTop 20 drugs by median across-patient rank:')
log(consensus.head(20).to_string())

# ---------- 6. Clinical sanity ----------
clinical = ['sorafenib','lenvatinib','regorafenib','cabozantinib','sunitinib',
            'gemcitabine','doxorubicin','paclitaxel','cisplatin','oxaliplatin',
            '5-fluorouracil','erlotinib','lapatinib','afatinib','vincristine']
drug_idx = {d.lower():i for i,d in enumerate(drugs)}
rows=[]
for dn in clinical:
    i = drug_idx.get(dn)
    if i is None: continue
    r = drug_score[drug_score['drug']==drugs[i]].iloc[0]
    rank = int((drug_score['score'] > r['score']).sum() + 1)
    rows.append({'drug':dn,'score':float(r['score']),'rank':rank, 'percentile':rank/len(drug_score)*100})
clin_df = pd.DataFrame(rows).sort_values('rank')
clin_df.to_parquet(OUT/'clinical_drugs_rank.parquet', index=False)
log('\nClinical HCC drugs — cohort rank (lower = better):')
log(clin_df.to_string())

# ---------- 7. Output per-prototype top drugs (for T4 wet-lab brief) ----------
log('\nPer-prototype top drugs (target subpop) …')
per_proto_top = []
for p in np.where(cox_weights>0)[0]:   # only bad-prognosis prototypes
    ranking = np.argsort(pred_z[p])   # ascending = low AUC = strong kill
    top10 = ranking[:10]
    for rnk, di in enumerate(top10):
        per_proto_top.append({'proto':int(p),'rank':rnk+1,'drug':drugs[di],
                              'pred_auc_z':float(pred_z[p,di]),
                              'pred_auc_raw':float(pred_auc_raw[p,di]),
                              'cox_p':float(cox.loc[cox['proto']==p,'p'].iloc[0]),
                              'cox_logHR':float(cox.loc[cox['proto']==p,'logHR'].iloc[0]),
                              'dominant':str(cox.loc[cox['proto']==p,'dominant'].iloc[0])})
top_df = pd.DataFrame(per_proto_top)
top_df.to_parquet(OUT/'top_drugs_per_bad_prognosis_prototype.parquet', index=False)
log(top_df.head(30).to_string())

# ---------- 8. Summary ----------
(OUT/'eval_metrics.json').write_text(json.dumps({
    'n_drugs': int(len(drugs)),
    'n_prototypes': int(pred_z.shape[0]),
    'n_significant_cox_prototypes': int((cox['p']<0.05).sum()),
    'bad_prognosis_prototypes': int((cox_weights>0).sum()),
    'good_prognosis_prototypes': int((cox_weights<0).sum()),
    'clinical_rank_summary': rows,
    'top10_cohort_drugs': drug_score.head(10)[['drug','score']].to_dict('records'),
    'bottom10_cohort_drugs': drug_score.tail(10)[['drug','score']].to_dict('records'),
}, indent=2, default=str))
log('== T3d done ==')
