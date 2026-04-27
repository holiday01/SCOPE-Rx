"""
T3b — Geneformer V2-104M zero-shot embedding of HCC scRNA cells.

Custom tokenizer (Geneformer's package has broken imports against transformers 5.5),
wraps the pretrained BERT via huggingface transformers.

Steps:
  1. Load gene_median_dict, token_dict, gene_name_id_dict (gc104M).
  2. For each cell in 20k sampled HCC scRNA cells:
       - normalized expr = (count / n_total) / gene_median_nonzero
       - rank genes descending, take top 2047 + prepend CLS
  3. Forward through BertModel fp16 on GPU.
  4. Mean-pool the final hidden states over non-pad tokens → 768-d embedding.
  5. Save (cells × 768) parquet + attach obs metadata.
"""
from __future__ import annotations
import json, time, pickle, random
from pathlib import Path
import numpy as np, pandas as pd
import torch
from transformers import BertModel, BertConfig
from safetensors.torch import load_file
import anndata as ad

ROOT = Path('/home/holiday01/drug_sc')
MODEL_DIR = ROOT/'models/Geneformer/Geneformer-V2-104M'
GF_DICTS  = ROOT/'models/Geneformer/geneformer'
OUT = ROOT/'results/t3b'; OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT/'checkpoints'
DEV = 'cuda'
SEED = 0
N_CELLS = 20000
SEQ_LEN = 2048
BS = 16
np.random.seed(SEED); torch.manual_seed(SEED)

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# ---------- 1. Dictionaries ----------
log('Loading Geneformer dictionaries …')
with open(GF_DICTS/'token_dictionary_gc104M.pkl','rb') as f:
    token_dict = pickle.load(f)   # ensembl_id → token_id
with open(GF_DICTS/'gene_median_dictionary_gc104M.pkl','rb') as f:
    gene_median = pickle.load(f)  # ensembl_id → median non-zero expr
with open(GF_DICTS/'gene_name_id_dict_gc104M.pkl','rb') as f:
    name_to_id = pickle.load(f)   # gene symbol → ensembl_id

log(f'  token_dict: {len(token_dict):,}  gene_median: {len(gene_median):,}  name_to_id: {len(name_to_id):,}')
# Special tokens
CLS = token_dict.get('<cls>', token_dict.get('[CLS]', token_dict.get('<CLS>', 0)))
PAD = token_dict.get('<pad>', token_dict.get('[PAD]', token_dict.get('<PAD>', 1)))
SEP = token_dict.get('<sep>', token_dict.get('[SEP]', None))
MASK = token_dict.get('<mask>', None)
log(f'  CLS={CLS}  PAD={PAD}  SEP={SEP}  MASK={MASK}')
# peek a few tokens
keys = list(token_dict.keys())[:5]
log(f'  sample tokens: {[(k, token_dict[k]) for k in keys]}')

# ---------- 2. Load model ----------
log('Loading Geneformer V2-104M …')
cfg = BertConfig.from_pretrained(str(MODEL_DIR))
model = BertModel(cfg)
sd = load_file(str(MODEL_DIR/'model.safetensors'))
# Pretrained is a BertForMaskedLM; strip lm-head keys and rename "bert." prefix
to_load = {}
for k, v in sd.items():
    nk = k
    if nk.startswith('bert.'): nk = nk[5:]
    if nk.startswith('cls.'): continue
    if nk.startswith('pooler.'): continue
    to_load[nk] = v
missing, unexpected = model.load_state_dict(to_load, strict=False)
log(f'  missing keys: {len(missing)}  unexpected: {len(unexpected)}')
log(f'  hidden={cfg.hidden_size}  layers={cfg.num_hidden_layers}  max_pos={cfg.max_position_embeddings}')
model = model.to(DEV).eval().half()

# ---------- 3. Load scRNA and sample ----------
log('Loading HCC integrated h5ad …')
a = ad.read_h5ad(ROOT/'data/scRNA_HCC_integrated/GSE223204GSE202642GSE162616_bbknn_tumornormal.h5ad', backed='r')
var_names = list(a.var_names)
log(f'  cells={a.n_obs}  genes={a.n_vars}')

# Which genes map to Geneformer tokens
gene_is_usable = np.zeros(len(var_names), dtype=bool)
gene_ensembl  = [None]*len(var_names)
gene_token_id = np.full(len(var_names), -1, dtype=np.int64)
gene_med      = np.full(len(var_names), np.nan, dtype=np.float32)
for i, sym in enumerate(var_names):
    ens = name_to_id.get(sym) or name_to_id.get(sym.upper())
    if ens is None: continue
    if ens not in token_dict: continue
    if ens not in gene_median: continue
    gene_ensembl[i] = ens
    gene_token_id[i] = token_dict[ens]
    gene_med[i] = gene_median[ens]
    gene_is_usable[i] = True
log(f'  usable genes: {gene_is_usable.sum()} / {len(var_names)}')

sample_idx = np.random.choice(a.n_obs, size=min(N_CELLS,a.n_obs), replace=False); sample_idx.sort()

# ---------- 4. Tokenize + embed ----------
log('Tokenizing + embedding …')
usable_gene_idx = np.where(gene_is_usable)[0]
usable_tokens = gene_token_id[usable_gene_idx]
usable_medians = gene_med[usable_gene_idx]

embeddings = np.zeros((len(sample_idx), cfg.hidden_size), dtype=np.float32)
pooled_cls = np.zeros((len(sample_idx), cfg.hidden_size), dtype=np.float32)

CHUNK = 512
buf_tokens = np.full((BS, SEQ_LEN), PAD, dtype=np.int64)
buf_attmsk = np.zeros((BS, SEQ_LEN), dtype=np.int64)

done = 0
t0 = time.time()
assert 'counts' in a.layers, "Geneformer needs raw counts from a.layers['counts']"
for chunk_start in range(0, len(sample_idx), CHUNK):
    idx = sample_idx[chunk_start:chunk_start+CHUNK]
    # Read RAW counts (a.X is already log-normalised in this h5ad)
    block = a.layers['counts'][idx, :][:, usable_gene_idx]
    if hasattr(block, 'toarray'): block = block.toarray()
    block = np.asarray(block, dtype=np.float32)
    # Normalize per cell then divide by gene median
    n_counts = block.sum(1, keepdims=True) + 1e-6
    norm = block / n_counts      # fractions per cell (sum to 1)
    norm = norm / usable_medians[None, :]
    # Tokenize each cell in the chunk
    for i in range(len(idx)):
        x = norm[i]
        mask = x > 0
        if mask.sum()==0: continue
        order = np.argsort(-x[mask])[:SEQ_LEN-1]  # descending; leave room for CLS
        pos = np.where(mask)[0][order]
        toks = usable_tokens[pos]
        # Prepend CLS
        buf_tokens[i % BS, 0] = CLS
        buf_tokens[i % BS, 1:1+len(toks)] = toks
        buf_tokens[i % BS, 1+len(toks):] = PAD
        buf_attmsk[i % BS, :1+len(toks)] = 1
        buf_attmsk[i % BS, 1+len(toks):] = 0
        if (i+1) % BS == 0 or (i+1) == len(idx):
            bsz = (i % BS) + 1
            t = torch.from_numpy(buf_tokens[:bsz]).to(DEV)
            m = torch.from_numpy(buf_attmsk[:bsz]).to(DEV)
            with torch.inference_mode():
                out = model(input_ids=t, attention_mask=m)
            hid = out.last_hidden_state.float()   # (B,L,H)
            # masked mean pool
            mm = m.float().unsqueeze(-1)
            mean_pool = (hid * mm).sum(1) / mm.sum(1).clamp(min=1.0)
            cls_vec   = hid[:,0,:]
            embeddings[done:done+bsz] = mean_pool.cpu().numpy()
            pooled_cls[done:done+bsz] = cls_vec.cpu().numpy()
            done += bsz
            # reset buffers
            buf_tokens[:] = PAD; buf_attmsk[:] = 0
    if chunk_start % (CHUNK*4) == 0:
        elapsed = time.time()-t0
        rate = done/max(elapsed,1e-6)
        eta = (len(sample_idx)-done)/max(rate,1e-6)
        log(f'  {done}/{len(sample_idx)} cells  rate={rate:.1f} cells/s  eta={eta:.0f}s')

a.file.close()

# ---------- 5. Save ----------
log('Saving embeddings …')
df_mp = pd.DataFrame(embeddings, columns=[f'e{i}' for i in range(cfg.hidden_size)])
df_mp['cell_global_idx'] = sample_idx
df_mp.to_parquet(OUT/'geneformer_v2_meanpool_20k.parquet', index=False)

df_cls = pd.DataFrame(pooled_cls, columns=[f'e{i}' for i in range(cfg.hidden_size)])
df_cls['cell_global_idx'] = sample_idx
df_cls.to_parquet(OUT/'geneformer_v2_cls_20k.parquet', index=False)

# Also attach cell metadata from the h5ad
log('Extracting cell metadata …')
a = ad.read_h5ad(ROOT/'data/scRNA_HCC_integrated/GSE223204GSE202642GSE162616_bbknn_tumornormal.h5ad', backed='r')
obs = a.obs.iloc[sample_idx].copy()
obs['cell_global_idx'] = sample_idx
obs.to_parquet(OUT/'cell_metadata_20k.parquet')
a.file.close()

(OUT/'eval_metrics.json').write_text(json.dumps({
    'n_cells': int(len(sample_idx)),
    'hidden': int(cfg.hidden_size),
    'usable_genes': int(gene_is_usable.sum()),
    'sampled_from': int(a.n_obs if False else 172538),
    'total_secs': float(time.time()-t0),
}, indent=2))
log(f'== T3b done in {time.time()-t0:.0f}s ==')
