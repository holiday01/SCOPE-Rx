"""
T3i-LUAD — External validation across 3 LUAD cohorts.

GSE68465 (n=463, Director's Challenge, GPL96)
GSE72094 (n=443, Schabath/Moffitt, GPL15048)
GSE31210 (n=247, Okayama 2012,    GPL570)

For each cohort:
  1. Parse series matrix → expression (probes × samples) + phenotype
  2. Map probes → gene symbols via platform annotation
  3. Aggregate probes → genes (max), align to LUAD T3c gene universe
  4. Apply T3c-LUAD attention deconvolver → composition
  5. Cox per prototype on OS (and RFS where present), stage-adjusted
  6. Composite TCGA-LUAD-trained risk score → c-index in this cohort
  7. Direction concordance with TCGA-LUAD per-prototype Cox

Final output: comparison_validation_luad.md + per-cohort tables + JSON summary.
"""
from __future__ import annotations
import gzip, json, time, re
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from lifelines import CoxPHFitter
from scipy.stats import pearsonr, spearmanr

ROOT = Path('/home/holiday01/drug_sc')
EXT  = ROOT/'data/external_LUAD'
T3C  = ROOT/'results/t3c_luad'
OUT  = ROOT/'results/t3i_luad'; OUT.mkdir(parents=True, exist_ok=True)
DEV  = 'cuda'
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

# Cohort spec
COHORTS = {
    'GSE68465': {
        'matrix': EXT/'GSE68465_series_matrix.txt.gz',
        'platform': '/mnt/10t/scrna_atac/data/raw/GSE68465/GPL96_full.txt',
        'platform_format': 'soft',          # GEO SOFT format with named cols
        'gene_col_names': ['Gene Symbol','Gene_Symbol','Symbol','GENE_SYMBOL','gene_symbol'],
        'char_map': {
            'vital_status':           'os_event_str',     # row 5
            'months_to_last_contact': 'os_time_months',   # row 12 (months)
            'first_progression':      'rfs_event_str',    # row 9
            'months_to_first_progression':'rfs_time_months', # row 10
            'disease_stage':          'stage_str',        # row 8 — pN/pT
            'sex':                    'sex',
            'age':                    'age',
        },
    },
    'GSE72094': {
        'matrix': EXT/'GSE72094_series_matrix.txt.gz',
        'platform': EXT/'GPL15048_platform_table.tsv',
        'platform_format': 'tsv',
        'gene_col_names': ['GeneSymbol','Gene Symbol','Symbol'],
        'char_map': {
            'vital_status':           'os_event_str',
            'survival_time_in_days':  'os_time_days',
            'Stage':                  'stage_str',
            'gender':                 'sex',
            'age_at_diagnosis':       'age',
        },
    },
    'GSE31210': {
        'matrix': EXT/'GSE31210_series_matrix.txt.gz',
        'platform': '/mnt/10t/scrna_atac/data/raw/GSE39582/GPL570.txt',
        'platform_format': 'soft',
        'gene_col_names': ['Gene Symbol','Gene_Symbol','Symbol'],
        'char_map': {
            'death':                  'os_event_str',
            'days before death':      'os_time_days',
            'relapse':                'rfs_event_str',
            'days before relapse':    'rfs_time_days',
            'pathological stage':     'stage_str',
            'gender':                 'sex',
            'age (years)':            'age',
            'tissue':                 'tissue',
            'exclude for prognosis':  'exclude_flag',
        },
    },
}

# ---------- Probe→gene from platform annotation ----------
def load_probe_to_gene(path, fmt, gene_col_names):
    probe_to_gene = {}
    if fmt == 'soft':
        with open(path,'r',errors='replace') as f:
            in_table=False; cols=None; gs_idx=None
            for line in f:
                if line.startswith('!platform_table_begin') or line.startswith('!Platform_table_begin'):
                    in_table=True; continue
                if line.startswith('!platform_table_end') or line.startswith('!Platform_table_end'):
                    break
                if not in_table: continue
                if cols is None:
                    cols = line.rstrip('\n').split('\t')
                    for n in gene_col_names:
                        if n in cols: gs_idx = cols.index(n); break
                    if gs_idx is None:
                        for i,c in enumerate(cols):
                            if c.lower() in ('gene symbol','gene_symbol','symbol','genesymbol'):
                                gs_idx = i; break
                    continue
                parts = line.rstrip('\n').split('\t')
                if len(parts)<=gs_idx: continue
                pid = parts[0].strip().strip('"')
                gs = parts[gs_idx].split(' /// ')[0].split('///')[0].strip()
                if gs and gs not in ('---','-','NA','','—','‐'):
                    probe_to_gene[pid] = gs
    else:  # tsv
        df = pd.read_csv(path, sep='\t', dtype=str, low_memory=False)
        gs_col = None
        for n in gene_col_names:
            if n in df.columns: gs_col = n; break
        for _, r in df.iterrows():
            pid = str(r.iloc[0]).strip()
            gs  = str(r.get(gs_col,'')).split(' /// ')[0].split('///')[0].strip()
            if pid and gs and gs not in ('---','-','NA','nan','','—','‐'):
                probe_to_gene[pid] = gs
    return probe_to_gene

# ---------- Series matrix parser ----------
def parse_series_matrix(path, char_map):
    """
    Parses GEO series matrix. Some cohorts (e.g. GSE31210) have per-sample
    variable characteristic ordering — so we parse each cell as a 'key: value'
    pair and build a per-sample dict, then map keys to short names.
    """
    sample_ids = None
    char_rows = []   # list of cells lists (rows × samples)
    probes = []; rows = []
    in_data = False
    with gzip.open(path,'rt', errors='replace') as f:
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                sample_ids = [x.strip().strip('"') for x in line.split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1'):
                cells = [x.strip().strip('"') for x in line.split('\t')[1:]]
                char_rows.append(cells)
            elif line.startswith('!series_matrix_table_begin'):
                in_data = True
            elif line.startswith('!series_matrix_table_end'):
                in_data = False
            elif in_data:
                parts = line.rstrip('\n').split('\t')
                if not parts or parts[0].startswith('!'): continue
                p = parts[0].strip().strip('"')
                if p == 'ID_REF': continue
                try:
                    vals = [float(x) if x not in ('','NA','null','NaN') else np.nan for x in parts[1:]]
                except: continue
                probes.append(p); rows.append(vals)

    # Build per-sample dict from all char rows; then map to short names
    n = len(sample_ids)
    per_sample = [dict() for _ in range(n)]
    for cells in char_rows:
        for i, cell in enumerate(cells[:n]):
            if not cell or ':' not in cell: continue
            k, v = cell.split(':', 1)
            per_sample[i][k.strip().lower()] = v.strip()

    phen = {short: [None]*n for short in char_map.values()}
    cm_low = [(k.lower(), short) for k, short in char_map.items()]
    for i, d in enumerate(per_sample):
        for key in d:
            for k_pattern, short in cm_low:
                # Prefer startswith for stricter matching
                if key.startswith(k_pattern):
                    if phen[short][i] is None:
                        phen[short][i] = d[key]
                    break
    pheno_df = pd.DataFrame(phen, index=sample_ids)
    expr = pd.DataFrame(np.array(rows, dtype=np.float32), index=probes, columns=sample_ids)
    return expr, pheno_df

# ---------- Survival parsing helpers ----------
def parse_survival(pheno, cohort_id):
    p = pheno.copy()
    # OS event
    if 'os_event_str' in p.columns:
        ev = p['os_event_str'].astype(str).str.lower()
        p['os_event'] = ev.map(lambda s: 1 if any(k in s for k in ['dead','deceased']) else
                                          (0 if any(k in s for k in ['alive','living','censor']) else np.nan))
    # OS time
    if 'os_time_days' in p.columns:
        p['os_time'] = pd.to_numeric(p['os_time_days'], errors='coerce')
    elif 'os_time_months' in p.columns:
        p['os_time'] = pd.to_numeric(p['os_time_months'], errors='coerce') * 30.4375
    # RFS
    if 'rfs_event_str' in p.columns:
        ev = p['rfs_event_str'].astype(str).str.lower()
        p['rfs_event'] = ev.map(lambda s: 1 if 'relapse' in s or 'progression' in s or 'yes' in s
                                          else (0 if 'no' in s or 'not' in s or 'censor' in s else np.nan))
    if 'rfs_time_days' in p.columns:
        p['rfs_time'] = pd.to_numeric(p['rfs_time_days'], errors='coerce')
    elif 'rfs_time_months' in p.columns:
        p['rfs_time'] = pd.to_numeric(p['rfs_time_months'], errors='coerce') * 30.4375
    p['age'] = pd.to_numeric(p.get('age', pd.Series(dtype=float)), errors='coerce')
    sex = p.get('sex', pd.Series(['']*len(p))).astype(str).str.lower()
    p['male'] = sex.apply(lambda s: 1.0 if s in ('m','male') else (0.0 if s in ('f','female') else np.nan))
    # numeric stage extracted from any of: 'pT2pN0', 'IIB', 'III', 'NA'
    stage = p.get('stage_str', pd.Series(['']*len(p))).astype(str)
    def _stage_num(s):
        s = s.upper()
        if 'IV' in s: return 4
        if 'III' in s: return 3
        if 'II' in s: return 2
        if ' I' in s or s.startswith('I') or 'IA' in s or 'IB' in s: return 1
        # pTNM pT*pN0 fallback
        m = re.search(r'pT(\d)', s)
        if m: return int(m.group(1))
        return np.nan
    p['stage_num'] = stage.apply(_stage_num)
    return p

# ---------- Attention deconvolver class ----------
class AttnDeconv(nn.Module):
    def __init__(self, d_in, n_proto, d_hid=256, p=0.1):
        super().__init__()
        self.query_enc = nn.Sequential(
            nn.Linear(d_in,512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(p),
            nn.Linear(512,d_hid))
        self.proto_key = nn.Parameter(torch.randn(n_proto, d_hid)*0.02)
        self.temp = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        q = self.query_enc(x)
        return F.softmax((q @ self.proto_key.T) / self.temp.clamp(min=0.1), dim=-1)

# ---------- Cox per prototype helper ----------
def cox_per_proto(comp_df, p, ev_col, t_col, n_proto, covariates=()):
    rows = []
    cov_use = [c for c in covariates if c in p.columns and p[c].notna().sum() >= 30]
    merged = comp_df.join(p[[ev_col, t_col] + cov_use], how='inner').dropna(subset=[ev_col, t_col])
    merged = merged[merged[t_col]>0]
    if len(merged) < 30: return pd.DataFrame(), 0, []
    for proto in range(n_proto):
        c = f'proto_{proto}'
        d = merged[[ev_col, t_col, c] + cov_use].dropna()
        if len(d)<30 or d[c].std()<1e-5: continue
        d = d.assign(x=(d[c]-d[c].mean())/d[c].std())
        try:
            cph = CoxPHFitter(penalizer=0.05).fit(d[[ev_col,t_col,'x']+cov_use], t_col, ev_col)
            rows.append({'proto':proto,'HR':float(np.exp(cph.params_['x'])),
                         'p':float(cph.summary.loc['x','p']),'n':int(len(d))})
        except: pass
    return pd.DataFrame(rows), len(merged), cov_use


# =============================================================
# MAIN
# =============================================================
log('Loading T3c-LUAD checkpoint …')
ck = torch.load(ROOT/'checkpoints/t3c_attn_deconv_luad.pt', map_location='cpu', weights_only=False)
n_proto = int(ck['n_proto'])
g_mu = np.asarray(ck['g_mu']); g_sd = np.asarray(ck['g_sd'])
genes_shared = list(ck['genes'])
proto_meta = pd.read_parquet(T3C/'prototype_meta.parquet').set_index('proto')
tcga_cox  = pd.read_parquet(T3C/'cox_per_prototype.parquet')
tcga_cox['logHR'] = np.log(tcga_cox['HR'])

per_cohort_results = {}
all_concordance = []

for cid, spec in COHORTS.items():
    log(f'\n{"="*70}\n  Cohort: {cid}\n{"="*70}')
    log('Parsing series matrix …')
    expr, pheno = parse_series_matrix(spec['matrix'], spec['char_map'])
    log(f'  expr: {expr.shape}  pheno: {pheno.shape}  cols: {list(pheno.columns)}')

    log('Mapping probes → genes …')
    probe_to_gene = load_probe_to_gene(spec['platform'], spec['platform_format'], spec['gene_col_names'])
    log(f'  probe→gene map: {len(probe_to_gene):,}')
    expr.index = [s.strip().strip('"') for s in expr.index]
    mapped = expr.loc[expr.index.intersection(probe_to_gene)].copy()
    mapped['_gene'] = mapped.index.map(probe_to_gene)
    gene_expr = mapped.groupby('_gene').max()
    if '_gene' in gene_expr.columns: gene_expr = gene_expr.drop(columns='_gene')
    log(f'  gene-level expr: {gene_expr.shape}')

    # parse survival
    p_full = parse_survival(pheno, cid)
    n_os = p_full['os_event'].notna().sum() if 'os_event' in p_full.columns else 0
    n_rfs = p_full['rfs_event'].notna().sum() if 'rfs_event' in p_full.columns else 0
    log(f'  OS-events parsed: {n_os}  RFS-events parsed: {n_rfs}')

    # Tumor selection: if tissue column exists, restrict to tumor; else assume all are tumor (cohort spec)
    if 'tissue' in p_full.columns:
        tumor_mask = p_full['tissue'].astype(str).str.contains('tumor|tumour|primary lung', case=False, na=False)
    else:
        tumor_mask = pd.Series(True, index=p_full.index)
    # GSE31210: drop excluded
    if 'exclude_flag' in p_full.columns:
        tumor_mask &= ~p_full['exclude_flag'].astype(str).str.lower().str.startswith('exclude')
    log(f'  Tumor samples kept: {tumor_mask.sum()}')

    # Align genes to T3c-LUAD gene universe
    common_g = [g for g in genes_shared if g in gene_expr.index]
    log(f'  shared genes (LUAD universe ∩ array): {len(common_g)}/{len(genes_shared)}')
    Xp = gene_expr.loc[common_g, tumor_mask].T  # (n_patients × n_common_g)
    Xp_vals = Xp.values.astype(np.float32)
    if Xp_vals.max() < 30:
        Xp_lin = np.clip(np.expm1(Xp_vals * np.log(2)), 0, None)
    else:
        Xp_lin = np.clip(Xp_vals, 0, None)
    full = np.zeros((Xp.shape[0], len(genes_shared)), dtype=np.float32)
    gene_to_i = {g:i for i,g in enumerate(genes_shared)}
    for i, g in enumerate(common_g):
        full[:, gene_to_i[g]] = Xp_lin[:, i]
    rs = full.sum(1, keepdims=True) + 1e-6
    Xn = np.log1p(np.clip(full/rs*1e4, 0, None)).astype(np.float32)
    Xn_std = (Xn - g_mu) / g_sd

    log('  Applying T3c-LUAD attention deconv …')
    model = AttnDeconv(Xn.shape[1], n_proto).to(DEV).eval()
    model.load_state_dict(ck['model'])
    with torch.no_grad():
        comp = model(torch.from_numpy(Xn_std).to(DEV)).cpu().numpy()
    ext_samples = Xp.index.tolist()
    comp_df = pd.DataFrame(comp, index=ext_samples,
                           columns=[f'proto_{i}' for i in range(n_proto)])
    comp_df.to_parquet(OUT/f'{cid}_composition.parquet')
    ent_pct = float((-(comp*np.log(comp+1e-9)).sum(1)).mean()/np.log(n_proto))
    log(f'  composition entropy: {ent_pct:.2%} of max')

    # Cox per prototype
    p_t = p_full.loc[tumor_mask].copy()
    log('  Cox(OS) per prototype (stage/age/sex adjusted) …')
    cox_os, n_os_used, cov_used = cox_per_proto(comp_df, p_t, 'os_event','os_time', n_proto,
                                                covariates=['stage_num','age','male'])
    sig_os_05 = int((cox_os['p']<0.05).sum()) if len(cox_os) else 0
    sig_os_01 = int((cox_os['p']<0.01).sum()) if len(cox_os) else 0
    log(f'    n_used={n_os_used}  covariates={cov_used}  prototypes p<0.05: {sig_os_05}  p<0.01: {sig_os_01}')
    if len(cox_os): cox_os.to_parquet(OUT/f'{cid}_cox_os.parquet', index=False)

    cox_rfs = pd.DataFrame()
    sig_rfs_05 = 0
    if 'rfs_event' in p_t.columns and p_t['rfs_event'].notna().sum() >= 30:
        log('  Cox(RFS) per prototype …')
        cox_rfs, n_rfs_used, _ = cox_per_proto(comp_df, p_t, 'rfs_event','rfs_time', n_proto,
                                               covariates=['stage_num','age','male'])
        sig_rfs_05 = int((cox_rfs['p']<0.05).sum()) if len(cox_rfs) else 0
        log(f'    RFS prototypes p<0.05: {sig_rfs_05}')
        if len(cox_rfs): cox_rfs.to_parquet(OUT/f'{cid}_cox_rfs.parquet', index=False)

    # Composite TCGA-LUAD risk score
    risk_w = np.zeros(n_proto, dtype=np.float32)
    for _, r in tcga_cox.iterrows():
        if float(r['p']) < 0.1:
            risk_w[int(r['proto'])] = float(np.log(r['HR']))
    n_proto_in_risk = int((risk_w!=0).sum())
    risk_score = comp.dot(risk_w)
    p_t['risk_z'] = (risk_score - risk_score.mean()) / (risk_score.std() + 1e-6)
    risk_results = {}
    for endpt, ev_col, t_col in [('OS','os_event','os_time'),('RFS','rfs_event','rfs_time')]:
        if ev_col not in p_t.columns: continue
        cov_avail = [c for c in ['stage_num','age','male'] if c in p_t.columns and p_t[c].notna().sum()>=30]
        d = p_t[[ev_col,t_col,'risk_z']+cov_avail].dropna()
        d = d[d[t_col]>0]
        if len(d) < 50: continue
        try:
            cph = CoxPHFitter(penalizer=0.05).fit(d, t_col, ev_col)
            risk_results[endpt] = {
                'HR_per_SD': float(np.exp(cph.params_['risk_z'])),
                'p': float(cph.summary.loc['risk_z','p']),
                'c_index': float(cph.concordance_index_),
                'n': int(len(d)),
                'covariates': cov_avail,
            }
            log(f'  Composite risk {endpt}: HR/SD = {risk_results[endpt]["HR_per_SD"]:.2f}  '
                f'p = {risk_results[endpt]["p"]:.3g}  c-index = {risk_results[endpt]["c_index"]:.3f}  n={len(d)}')
        except Exception as e:
            log(f'  {endpt} composite risk failed: {e}')

    # Concordance with TCGA-LUAD
    if len(cox_os):
        merged = cox_os.set_index('proto').join(tcga_cox.set_index('proto')[['HR','p','dominant']],
                                                 how='inner', rsuffix='_tcga')
        merged['log_HR_ext']  = np.log(merged['HR'])
        merged['log_HR_tcga'] = np.log(merged['HR_tcga'])
        sign_concord = (np.sign(merged['log_HR_ext']) == np.sign(merged['log_HR_tcga']))
        tcga_sig = merged[merged['p_tcga']<0.05]
        sig_concord = (np.sign(tcga_sig['log_HR_ext']) == np.sign(tcga_sig['log_HR_tcga']))
        merged.to_parquet(OUT/f'{cid}_concordance_table.parquet')
        if len(merged) > 5:
            r_p, p_p = pearsonr(merged['log_HR_ext'], merged['log_HR_tcga'])
            r_s, p_s = spearmanr(merged['log_HR_ext'], merged['log_HR_tcga'])
        else: r_p=p_p=r_s=p_s=np.nan
        log(f'  Concordance: full {sign_concord.mean():.0%}  TCGA-sig {sig_concord.mean():.0%} '
            f'({sig_concord.sum()}/{len(tcga_sig)})  log-HR Pearson r = {r_p:.3f} (p={p_p:.2e})')
    else:
        sign_concord = sig_concord = pd.Series(); r_p=p_p=r_s=p_s=np.nan

    per_cohort_results[cid] = {
        'n_tumor_samples': int(tumor_mask.sum()),
        'shared_genes': len(common_g),
        'composition_entropy_pct': ent_pct,
        'os_p05_count': sig_os_05,
        'os_p01_count': sig_os_01,
        'rfs_p05_count': sig_rfs_05,
        'composite_risk': risk_results,
        'sign_concord_full_pct':  float(sign_concord.mean()) if len(sign_concord) else None,
        'sign_concord_tcga_sig_pct': float(sig_concord.mean()) if len(sig_concord) else None,
        'sign_concord_tcga_sig_count': int(sig_concord.sum()) if len(sig_concord) else 0,
        'sign_concord_tcga_sig_total': int(len(tcga_sig)) if len(cox_os) else 0,
        'logHR_pearson_r': float(r_p) if not np.isnan(r_p) else None,
        'logHR_pearson_p': float(p_p) if not np.isnan(p_p) else None,
        'logHR_spearman_r': float(r_s) if not np.isnan(r_s) else None,
        'covariates_used_OS': cov_used if len(cox_os) else [],
        'n_TCGA_prog_protos_in_risk': n_proto_in_risk,
    }

# ---------- Combined report ----------
log(f'\n{"="*70}\n  Writing combined report\n{"="*70}')
lines = ['# T3i-LUAD — External validation across 3 LUAD cohorts\n',
         '## Setup',
         '- Anchor: TCGA-LUAD (n=576, 58 prototypes; 15 Cox-significant at p<0.05)',
         f'- TCGA-LUAD prog-significant prototypes used in composite risk (p<0.1): {n_proto_in_risk}',
         '- Each cohort: T3c-LUAD attention deconv (frozen) → composition → Cox (stage/age/sex adjusted) per prototype',
         '- Composite risk = composition · log(HR)_TCGA across p<0.1 prototypes\n']

lines += ['## Cohort summary\n',
          '| Cohort | n tumor | shared genes | entropy | OS p<0.05 | OS p<0.01 | RFS p<0.05 | concord (TCGA-sig) | log-HR Pearson r |',
          '|---|---:|---:|---:|---:|---:|---:|---|---:|']
for cid, r in per_cohort_results.items():
    cspct = f"{r['sign_concord_tcga_sig_pct']:.0%} ({r['sign_concord_tcga_sig_count']}/{r['sign_concord_tcga_sig_total']})" if r['sign_concord_tcga_sig_pct'] is not None else '—'
    rho = f"{r['logHR_pearson_r']:.3f}" + (f" (p={r['logHR_pearson_p']:.2e})" if r['logHR_pearson_p'] else '') if r['logHR_pearson_r'] is not None else '—'
    lines.append(f"| {cid} | {r['n_tumor_samples']} | {r['shared_genes']} | {r['composition_entropy_pct']:.0%} | "
                 f"{r['os_p05_count']} | {r['os_p01_count']} | {r['rfs_p05_count']} | {cspct} | {rho} |")

lines += ['\n## Composite TCGA-LUAD-trained risk score (multivariate-adjusted)\n',
          '| Cohort | Endpoint | n | HR per SD | p | c-index |',
          '|---|---|---:|---:|---:|---:|']
for cid, r in per_cohort_results.items():
    for endpt, e in r['composite_risk'].items():
        lines.append(f"| {cid} | {endpt} | {e['n']} | {e['HR_per_SD']:.2f} | {e['p']:.3g} | {e['c_index']:.3f} |")

(OUT/'comparison_validation_luad.md').write_text('\n'.join(lines))
log(f'Report: {OUT/"comparison_validation_luad.md"}')

(OUT/'eval_metrics.json').write_text(json.dumps({
    'cohorts': per_cohort_results,
    'n_TCGA_prog_protos_in_risk': n_proto_in_risk,
}, indent=2, default=str))
log('== T3i-LUAD done ==')
