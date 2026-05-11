"""
T3d-LUAD — Survival-anchored drug scoring, LUAD edition.

Reads:
  checkpoints/t2b_scdeal_luad.pt        (LUAD-domain-adapted scDEAL)
  results/t3c_luad/{prototype_expression, prototype_meta, tcga_composition,
                    cox_per_prototype}.parquet

Writes:
  results/t3d_luad/{predicted_auc_zscore_per_prototype,
                    predicted_auc_raw_per_prototype,
                    drug_score_cohort, drug_score_per_patient,
                    drug_consensus_rank, clinical_drugs_rank,
                    top_drugs_per_bad_prognosis_prototype, eval_metrics.json}
"""
from __future__ import annotations
import json, time, math
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn

ROOT = Path('/home/holiday01/drug_sc')
PROC = ROOT/'data/processed/luad_drug'
T3C  = ROOT/'results/t3c_luad'
OUT  = ROOT/'results/t3d_luad'; OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT/'checkpoints'
DEV  = 'cuda'

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Load T2b-LUAD scDEAL ----------
log('Loading T2b-LUAD scDEAL …')
t2b = torch.load(CKPT/'t2b_scdeal_luad.pt', map_location='cpu', weights_only=False)
drugs = list(t2b['drugs'])
genes = list(t2b['genes'])
g_mu = np.asarray(t2b['gene_mu']); g_sd = np.asarray(t2b['gene_sd'])
drug_mean = np.asarray(t2b['drug_mean']); drug_std = np.asarray(t2b['drug_std'])

class Encoder(nn.Module):
    def __init__(self, d_in, d_hid, d_emb, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in,d_hid), nn.BatchNorm1d(d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid,d_emb), nn.BatchNorm1d(d_emb), nn.ReLU(), nn.Dropout(p))
    def forward(self, x): return self.net(x)
class DrugHead(nn.Module):
    def __init__(self, d_emb, n): super().__init__(); self.lin = nn.Linear(d_emb, n)
    def forward(self, z): return self.lin(z)

enc = Encoder(len(genes), 1024, 256).to(DEV); enc.load_state_dict(t2b['enc']); enc.eval()
dhead = DrugHead(256, len(drugs)).to(DEV); dhead.load_state_dict(t2b['dhead']); dhead.eval()
log(f'scDEAL — drugs={len(drugs)}  genes={len(genes)}')

# ---------- 2. Prototype pseudobulk → predicted AUC ----------
log('Predicting AUC z per prototype …')
pb = pd.read_parquet(T3C/'prototype_expression.parquet')
assert list(pb.columns) == genes, 'gene order mismatch'
Xp = pb.values.astype(np.float32)
Xp_z = (Xp - g_mu) / g_sd
with torch.no_grad():
    pred_z = dhead(enc(torch.from_numpy(Xp_z).to(DEV))).cpu().numpy()
pred_auc_raw = pred_z * drug_std[None,:] + drug_mean[None,:]
pd.DataFrame(pred_z, columns=drugs).to_parquet(OUT/'predicted_auc_zscore_per_prototype.parquet')
pd.DataFrame(pred_auc_raw, columns=drugs).to_parquet(OUT/'predicted_auc_raw_per_prototype.parquet')
log(f'pred_z shape {pred_z.shape}  raw_auc range [{pred_auc_raw.min():.2f}, {pred_auc_raw.max():.2f}]')

# ---------- 3. Cox + trust → per-prototype weight ----------
cox = pd.read_parquet(T3C/'cox_per_prototype.parquet')
proto_meta = pd.read_parquet(T3C/'prototype_meta.parquet')
log(f'Cox rows: {len(cox)}  significant (p<0.05): {(cox["p"]<0.05).sum()}')
cox['logHR'] = np.log(cox['HR'].astype(float).clip(lower=1e-6, upper=1e6))
cox['weight_raw'] = (-np.log10(cox['p'].astype(float).clip(lower=1e-300))).clip(upper=10) * np.sign(cox['logHR'])
cox.loc[cox['p'] > 0.1, 'weight_raw'] = 0
cox['weight'] = np.where(cox['weight_raw']>0, cox['weight_raw'], cox['weight_raw']*0.3)

proto_meta = proto_meta.set_index('proto')
trust_map = proto_meta['trust_to_depmap'].to_dict()
label_map = proto_meta['label'].to_dict()

cox_weights = np.zeros(pred_z.shape[0], dtype=np.float32)
trust_weights = np.ones(pred_z.shape[0], dtype=np.float32)
for _, r in cox.iterrows():
    p = int(r['proto'])
    cox_weights[p] = r['weight']
    t = trust_map.get(p, 0.0)
    trust_weights[p] = 1.0 / (1.0 + math.exp(-(t - 0.25)*8))
combined_w = cox_weights * trust_weights
log(f'Cox weights nonzero: {(cox_weights!=0).sum()}  Σ|cox|={np.abs(cox_weights).sum():.2f}')
log(f'Trust weights range: [{trust_weights.min():.3f}, {trust_weights.max():.3f}]')
log(f'Combined range: [{combined_w.min():.3f}, {combined_w.max():.3f}]')

# ---------- 4. Cohort drug score ----------
S_drug = combined_w @ (-pred_z)
drug_score = pd.DataFrame({'drug':drugs, 'score':S_drug})
bad = cox_weights > 0
if bad.any():
    drug_score['mean_predAUC_bad_protos'] = pred_auc_raw[bad].mean(0)
    drug_score['mean_predAUC_all_protos'] = pred_auc_raw.mean(0)
mal_mask = np.array([('Epithelial_tumor' in label_map.get(i,'') or 'Malignant' in label_map.get(i,''))
                     for i in range(pred_z.shape[0])])
if mal_mask.any():
    drug_score['score_malignant_only'] = (mal_mask.astype(np.float32) * trust_weights) @ (-pred_z)
drug_score = drug_score.sort_values('score', ascending=False)
drug_score.to_parquet(OUT/'drug_score_cohort.parquet', index=False)
log('Top 20 drugs by survival-anchored score:')
log(drug_score.head(20).to_string())

# ---------- 5. Per-patient ranking ----------
log('\nComputing per-patient drug scores …')
comp = pd.read_parquet(T3C/'tcga_composition.parquet')
C = comp.values.astype(np.float32)
prog_w = np.where(cox_weights>0, 1.0, np.where(cox_weights<0, -0.3, 0.0)).astype(np.float32)
pw = prog_w * trust_weights
weighted = C * pw[None,:]
S_patient = weighted @ (-pred_z)
S_patient_df = pd.DataFrame(S_patient, index=comp.index, columns=drugs)
S_patient_df.to_parquet(OUT/'drug_score_per_patient.parquet')
log(f'Per-patient: {S_patient_df.shape}')

med_rank = S_patient_df.rank(axis=1, ascending=False).median(0).sort_values()
consensus = pd.DataFrame({'drug':med_rank.index, 'median_rank':med_rank.values})
consensus.to_parquet(OUT/'drug_consensus_rank.parquet', index=False)
log('\nTop 20 drugs by median across-patient rank:')
log(consensus.head(20).to_string())

# ---------- 6. Clinical sanity (LUAD-relevant drugs) ----------
clinical_luad = ['erlotinib','gefitinib','afatinib','osimertinib','dacomitinib',
                 'crizotinib','alectinib','ceritinib','lorlatinib','brigatinib',
                 'pemetrexed','docetaxel','paclitaxel','carboplatin','cisplatin',
                 'gemcitabine','vinorelbine','etoposide','ramucirumab','bevacizumab',
                 'nivolumab','pembrolizumab','atezolizumab','sotorasib','adagrasib',
                 'trametinib','selumetinib']
drug_idx = {d.lower():i for i,d in enumerate(drugs)}
rows=[]
for dn in clinical_luad:
    i = drug_idx.get(dn)
    if i is None: continue
    r = drug_score[drug_score['drug']==drugs[i]].iloc[0]
    rank = int((drug_score['score'] > r['score']).sum() + 1)
    rows.append({'drug':dn,'score':float(r['score']),'rank':rank,
                 'percentile':rank/len(drug_score)*100})
clin_df = pd.DataFrame(rows).sort_values('rank')
clin_df.to_parquet(OUT/'clinical_drugs_rank.parquet', index=False)
log('\nClinical LUAD drugs — cohort rank (lower = better):')
log(clin_df.to_string())

# ---------- 7. Per-prototype top drugs ----------
log('\nPer-prototype top drugs (bad-prognosis prototypes only) …')
per_proto_top = []
for p in np.where(cox_weights>0)[0]:
    ranking = np.argsort(pred_z[p])
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
log('== T3d-LUAD done ==')
