"""
T3b-LUAD — Geneformer V2-104M zero-shot embedding of LUAD scRNA cells.

Mirrors t3b_geneformer_embed.py, with two differences vs the LIHC version:
  (1) Source h5ad is /mnt/10t/scrna_atac/data/processed/LUAD/luad_scrna_annotated.h5ad
      which has only 20,000 cells — embed ALL of them, no subsample.
  (2) The h5ad lacks raw counts (a.raw is None, no 'counts' layer); .X is
      log1p(normalize_total(target_sum=1e4)). Since Geneformer's rank is
      monotone in (count / total_count) / gene_median, we compute
      norm = expm1(X) / gene_median  — which produces the identical token
      ordering to the raw-count pathway.
"""
from __future__ import annotations
import json, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
import torch
from transformers import BertModel, BertConfig
from safetensors.torch import load_file
import anndata as ad

ROOT = Path('/home/holiday01/drug_sc')
MODEL_DIR = ROOT/'models/Geneformer/Geneformer-V2-104M'
GF_DICTS  = ROOT/'models/Geneformer/geneformer'
SCR = Path('/mnt/10t/scrna_atac/data/processed/LUAD/luad_scrna_annotated.h5ad')
OUT = ROOT/'results/t3b_luad'; OUT.mkdir(parents=True, exist_ok=True)
DEV = 'cuda'
SEED = 0
SEQ_LEN = 2048
BS = 16
np.random.seed(SEED); torch.manual_seed(SEED)

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Dictionaries ----------
log('Loading Geneformer dictionaries …')
with open(GF_DICTS/'token_dictionary_gc104M.pkl','rb') as f:
    token_dict = pickle.load(f)
with open(GF_DICTS/'gene_median_dictionary_gc104M.pkl','rb') as f:
    gene_median = pickle.load(f)
with open(GF_DICTS/'gene_name_id_dict_gc104M.pkl','rb') as f:
    name_to_id = pickle.load(f)
log(f'  token_dict: {len(token_dict):,}  gene_median: {len(gene_median):,}  name_to_id: {len(name_to_id):,}')
CLS = token_dict.get('<cls>', token_dict.get('[CLS]', token_dict.get('<CLS>', 0)))
PAD = token_dict.get('<pad>', token_dict.get('[PAD]', token_dict.get('<PAD>', 1)))
log(f'  CLS={CLS}  PAD={PAD}')

# ---------- 2. Load model ----------
log('Loading Geneformer V2-104M …')
cfg = BertConfig.from_pretrained(str(MODEL_DIR))
model = BertModel(cfg)
sd = load_file(str(MODEL_DIR/'model.safetensors'))
to_load = {}
for k, v in sd.items():
    nk = k
    if nk.startswith('bert.'): nk = nk[5:]
    if nk.startswith('cls.') or nk.startswith('pooler.'): continue
    to_load[nk] = v
missing, unexpected = model.load_state_dict(to_load, strict=False)
log(f'  missing={len(missing)}  unexpected={len(unexpected)}  hidden={cfg.hidden_size}')
model = model.to(DEV).eval().half()

# ---------- 3. Load scRNA ----------
log('Loading LUAD annotated h5ad …')
a = ad.read_h5ad(SCR, backed='r')
var_names = list(a.var_names)
N = a.n_obs
log(f'  cells={N}  genes={a.n_vars}')

gene_is_usable = np.zeros(len(var_names), dtype=bool)
gene_token_id  = np.full(len(var_names), -1, dtype=np.int64)
gene_med       = np.full(len(var_names), np.nan, dtype=np.float32)
for i, sym in enumerate(var_names):
    ens = name_to_id.get(sym) or name_to_id.get(sym.upper())
    if ens is None or ens not in token_dict or ens not in gene_median: continue
    gene_token_id[i] = token_dict[ens]
    gene_med[i] = gene_median[ens]
    gene_is_usable[i] = True
log(f'  usable genes: {gene_is_usable.sum()} / {len(var_names)}')

usable_gene_idx = np.where(gene_is_usable)[0]
usable_tokens   = gene_token_id[usable_gene_idx]
usable_medians  = gene_med[usable_gene_idx]
sample_idx = np.arange(N)  # embed all cells

# ---------- 4. Tokenize + embed ----------
log('Tokenizing + embedding …')
embeddings = np.zeros((N, cfg.hidden_size), dtype=np.float32)
pooled_cls = np.zeros((N, cfg.hidden_size), dtype=np.float32)

CHUNK = 512
buf_tokens = np.full((BS, SEQ_LEN), PAD, dtype=np.int64)
buf_attmsk = np.zeros((BS, SEQ_LEN), dtype=np.int64)

done = 0
t0 = time.time()
for chunk_start in range(0, N, CHUNK):
    idx_end = min(chunk_start+CHUNK, N)
    block = a.X[chunk_start:idx_end, :][:, usable_gene_idx]
    if hasattr(block, 'toarray'): block = block.toarray()
    block = np.asarray(block, dtype=np.float32)
    # X is log1p(CPMK/1e4); expm1 → counts-per-10k. Rank of (count/total)/median
    # equals rank of expm1(X)/median (same scaling within a cell), so use:
    norm = np.expm1(block) / usable_medians[None, :]

    n = idx_end - chunk_start
    for i in range(n):
        x = norm[i]
        mask = x > 0
        if mask.sum()==0: continue
        order = np.argsort(-x[mask])[:SEQ_LEN-1]
        pos = np.where(mask)[0][order]
        toks = usable_tokens[pos]
        bi = i % BS
        buf_tokens[bi, 0] = CLS
        buf_tokens[bi, 1:1+len(toks)] = toks
        buf_tokens[bi, 1+len(toks):] = PAD
        buf_attmsk[bi, :1+len(toks)] = 1
        buf_attmsk[bi, 1+len(toks):] = 0
        if (i+1) % BS == 0 or (i+1) == n:
            bsz = bi + 1
            t = torch.from_numpy(buf_tokens[:bsz]).to(DEV)
            m = torch.from_numpy(buf_attmsk[:bsz]).to(DEV)
            with torch.inference_mode():
                out = model(input_ids=t, attention_mask=m)
            hid = out.last_hidden_state.float()
            mm = m.float().unsqueeze(-1)
            mean_pool = (hid * mm).sum(1) / mm.sum(1).clamp(min=1.0)
            cls_vec   = hid[:,0,:]
            embeddings[done:done+bsz] = mean_pool.cpu().numpy()
            pooled_cls[done:done+bsz] = cls_vec.cpu().numpy()
            done += bsz
            buf_tokens[:] = PAD; buf_attmsk[:] = 0
    if chunk_start % (CHUNK*4) == 0:
        elapsed = time.time()-t0
        rate = done/max(elapsed,1e-6)
        eta = (N-done)/max(rate,1e-6)
        log(f'  {done}/{N} cells  rate={rate:.1f} cells/s  eta={eta:.0f}s')

obs = a.obs.copy()
a.file.close()

# ---------- 5. Save embeddings + metadata ----------
log('Saving embeddings …')
df_mp = pd.DataFrame(embeddings, columns=[f'e{i}' for i in range(cfg.hidden_size)])
df_mp['cell_global_idx'] = sample_idx
df_mp.to_parquet(OUT/'geneformer_v2_meanpool_luad.parquet', index=False)

df_cls = pd.DataFrame(pooled_cls, columns=[f'e{i}' for i in range(cfg.hidden_size)])
df_cls['cell_global_idx'] = sample_idx
df_cls.to_parquet(OUT/'geneformer_v2_cls_luad.parquet', index=False)

obs['cell_global_idx'] = sample_idx
obs.to_parquet(OUT/'cell_metadata_luad.parquet')

# ---------- 6. Build cell-state prototypes ----------
# Tumor flag: tLung, tL/B, mBrain, mLN, PE → tumor-side; nLung, nLN → normal-side
NORMAL_ORIGINS = {'nLung', 'nLN'}
obs['tn'] = np.where(obs['Sample_Origin'].isin(NORMAL_ORIGINS), 'normal', 'tumor')

# coarse prototype: celltype × tumor/normal
proto_key_coarse = obs['celltype'].astype(str) + '|' + obs['tn']
# fine prototype: Cell_subtype × tumor/normal (drop NaNs)
sub = obs['Cell_subtype'].astype(str)
proto_key_fine = np.where(sub.isin(['nan','None','']), 'NA', sub) + '|' + obs['tn']

def aggregate(keys, label, min_cells=20):
    keys = pd.Series(keys, index=obs.index)
    df = pd.DataFrame(embeddings)
    df['k'] = keys.values
    sizes = df['k'].value_counts()
    keep = sizes[sizes >= min_cells].index
    df = df[df['k'].isin(keep)]
    proto = df.groupby('k').mean().reset_index()
    proto.columns = ['prototype'] + [f'e{i}' for i in range(cfg.hidden_size)]
    proto['n_cells'] = proto['prototype'].map(df['k'].value_counts())
    proto['scope'] = label
    log(f'  {label}: {len(proto)} prototypes (≥{min_cells} cells)')
    return proto

proto_coarse = aggregate(proto_key_coarse, 'coarse', min_cells=20)
proto_fine   = aggregate(proto_key_fine,   'fine',   min_cells=30)
proto = pd.concat([proto_coarse, proto_fine], ignore_index=True)
proto.to_parquet(OUT/'prototype_embeddings_luad.parquet', index=False)
log(f'Total prototypes saved: {len(proto)}')

(OUT/'eval_metrics.json').write_text(json.dumps({
    'n_cells_embedded': int(N),
    'hidden': int(cfg.hidden_size),
    'usable_genes': int(gene_is_usable.sum()),
    'n_prototypes_coarse': int(len(proto_coarse)),
    'n_prototypes_fine': int(len(proto_fine)),
    'total_secs': float(time.time()-t0),
}, indent=2))
log(f'== T3b-LUAD done in {time.time()-t0:.0f}s ==')
