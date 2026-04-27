# SCOPE-Rx

**A foundation-model attention deconvolution framework empirically validated across three independent patient cohorts for survival-anchored drug repositioning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2Bcu128-red)
![Status](https://img.shields.io/badge/status-Phase--1-green)

---

## Overview

SCOPE-Rx unifies four heterogeneous data domains in a single trustworthy drug-repositioning pipeline:

1. **Single-cell foundation embeddings** (Geneformer V2-104M) used as the geometric basis for bulk-to-prototype attention deconvolution.
2. **Bulk pharmacogenomics** (DepMap + GDSC + PRISM + CTRPv2, 1.13 M drug-response records on 1,684 cell lines).
3. **Patient survival** (TCGA-LIHC training, GSE14520 + GSE76427 external validation; total *n* = 763).
4. **Drug-target × pathway-Cox prior** (1,790 MSigDB / KEGG / Reactome pathways, Cox-anchored on TCGA).

The pipeline produces a **trust-tiered drug ranking** with explicit calibration of which predictions are extrapolated beyond DepMap's reference. Validated on hepatocellular carcinoma; generalisable to other solid tumours via a tissue-matched scRNA reference.

### Key results
- **No composition collapse** in three independent cohorts (entropy 89% of theoretical max), where Scaden-CA collapses to one cell line (91% of patient signal).
- **8 / 9 TCGA-prognostic prototypes replicate** in ≥ 1 external endpoint.
- **Composite TCGA-trained risk score: c-index 0.727 for OS in GSE14520** (*p* = 0.006, multivariate-adjusted).
- **3-cohort fixed-effect meta-analysis**: 16 pooled-significant prognostic prototypes.
- **5+4 method consensus**: 5 survival-ranking methods agree on 9 prototypes, 4 score-aggregation methods agree on 11 mechanism-converged drugs (EGFR/HER, MEK, HSP90, PI3K/AKT, VEGFR).
- **Reproducible end-to-end on a single consumer GPU in <10 min.**

---

## Repository structure

```
SCOPE-Rx/
├── scripts/                 # 24 Python pipelines (T2-T4)
│   ├── preprocess/          # data assembly + drug catalog
│   ├── baselines/           # scDEAL / scPDS / Scaden-CA reproductions
│   ├── novelty/             # T3a-T3k + T4 (core SCOPE-Rx pipeline)
│   └── report/              # figures + PPT + graphical abstract
├── environment.yml          # conda environment definition
├── LICENSE                  # MIT
└── README.md                # this file
```

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/holiday01/SCOPE-Rx.git
cd SCOPE-Rx
```

### 2. Create the conda environment (Python 3.11 + PyTorch 2.10 + CUDA 12.8)
```bash
conda env create -f environment.yml -n scope_rx
conda activate scope_rx
```

> **GPU note:** The pipeline was developed on a single NVIDIA RTX 5070-class card (Blackwell sm_120, 16 GB). For Blackwell GPUs, ensure `torch ≥ 2.7 + cu128`; older Ampere/Turing cards work with cu121 builds.

### 3. Download the Geneformer V2-104M weights (≈ 400 MB)
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ctheodoris/Geneformer',
    local_dir='models/Geneformer',
    allow_patterns=['Geneformer-V2-104M/*','geneformer/*.py','geneformer/*.pkl'])
"
```

### 4. Download public data (not bundled in the repo)
| Resource | Source | Files needed |
|---|---|---|
| TCGA-LIHC bulk RNA-seq + clinical | [UCSC Xena](https://xenabrowser.net) | `TCGA_LIHC_expression.gz`, `TCGA_LIHC_clinical.tsv` |
| DepMap | [depmap.org/portal](https://depmap.org/portal/data_page) | `OmicsExpressionTPM.csv`, `Model.csv`, `PRISM_secondary_AUC.csv`, `sanger_dose_response.csv`, `CTRPv2/` |
| HCC scRNA atlas | NCBI GEO | GSE125449, GSE149614, GSE156625, GSE162616, GSE202642, GSE223204 (or pre-integrated h5ad available on request) |
| GSE14520 (Roessler 2010) | NCBI GEO + supp | `GSE14520-GPL3921_series_matrix.txt.gz` + `GSE14520_Extra_Supplement.txt.gz` |
| GSE76427 (Yang 2017) | NCBI GEO | `GSE76427_series_matrix.txt.gz` |
| Pathway gene sets | gseapy in-app fetch | MSigDB Hallmark, KEGG 2021, Reactome 2022 |

Place data in `data/` with the layout expected by `scripts/preprocess/t2a_prepare_hcc_drug_data.py`. The script auto-creates symlinks if you point it at your downloads.

---

## Reproduction (full pipeline)

```bash
# 1. Data assembly (~ 1 min)
python scripts/preprocess/t2a_prepare_hcc_drug_data.py
python scripts/preprocess/t2a_fix_drug_catalog.py

# 2. Baselines (~ 5 min)
python scripts/baselines/t2b_scdeal_train.py        # scDEAL (LOO Spearman 0.145)
python scripts/baselines/t2b_eval_diagnose.py
python scripts/baselines/t2c_scpds_train.py         # scPDS
python scripts/baselines/t2c2_scaden_ca_train.py    # Scaden-CA (collapses)

# 3. SCOPE-Rx core (~ 5 min)
python scripts/novelty/t3b_geneformer_embed.py             # 768-d zero-shot
python scripts/novelty/t3c_celltype_attention_deconv.py    # 57 prototypes, no collapse
python scripts/novelty/t3d_signature_reversal_cox.py       # Cox + drug scoring
python scripts/novelty/t3e_oncology_filter.py              # PRISM oncology metadata
python scripts/novelty/t3f_target_pathway_prior.py         # 1,790 pathway-Cox priors
python scripts/novelty/t3g_expanded_trust_reference.py     # trust calibration
python scripts/novelty/t3h_reviewer_revisions.py           # multivariate Cox + tier ranking

# 4. External validation + meta-analysis (~ 2 min)
python scripts/novelty/t3i_external_validation_gse76427.py
python scripts/novelty/t3j_two_cohort_meta_analysis.py     # GSE14520 + GSE76427
python scripts/novelty/t3k_consistency_battery.py          # 5+4 method consensus

# 5. Final ranking + wet-lab brief (~ 30 s)
python scripts/novelty/t4_integrated_ranking_wetlab_brief.py

# 6. Figures + PPT + graphical abstract (~ 1 min)
python scripts/report/make_figures.py
python scripts/report/make_figures_v2.py
python scripts/report/make_graphical_abstract.py
python scripts/report/make_ppt_v2.py
```

End-to-end: **~ 13 min** on RTX 5070; **~ 25 min** on RTX 3090; **~ 1 h** on RTX 2080 Ti.

---

## Key result tables (Apache Parquet)

| File | Content |
|---|---|
| `results/t3c/prototype_meta.parquet` | 57 prototypes × (cell type, sample type, tumour fraction, label, trust-to-DepMap, best DepMap line) |
| `results/t3c/prototype_expression.parquet` | 57 × 17,460 prototype pseudobulks |
| `results/t3c/tcga_composition.parquet` | 423 patients × 57 composition |
| `results/t3h/multivariate_cox.parquet` | 9 prognostic prototypes (HR, p, n) |
| `results/t3j/meta_analysis_pooled.parquet` | 16 cross-cohort meta-significant prototypes |
| `results/t3k/prognostic_consistency_votes.parquet` | 5-method survival vote |
| `results/t3k/drug_rank_consistency_votes.parquet` | 4-method drug aggregation vote |
| `results/t4/per_patient_top5.parquet` | 423 × top-5 drug recommendations |
| `results/t4/hcc_top20_wetlab_brief.md` | Wet-lab brief: target subpops, FACS markers, suggested HCC lines, SMILES |

---

## Citation

```bibtex
@article{Chiu2026SCOPERx,
  author  = {Chiu, Yen-Jung},
  title   = {SCOPE-Rx: A foundation-model attention deconvolution framework
             empirically validated across three independent patient cohorts
             for survival-anchored drug repositioning},
  journal = {Methods},
  year    = {2026},
  note    = {Submitted}
}
```

---

## Funding
This research was supported by the National Science and Technology Council, Taiwan (NSTC 113-2221-E-130-005-MY3).

## Contact
**Yen-Jung Chiu** ([d000020163@cgu.edu.tw](mailto:d000020163@cgu.edu.tw))
Department of Biomedical Engineering, Chang Gung University,
259 Wenhua 1st Road, Guishan District, Taoyuan City 33302, Taiwan
Tel: +886-3-2118800 ext. 2626

## License
MIT — see [LICENSE](LICENSE).
