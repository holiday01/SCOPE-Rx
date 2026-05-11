# SCOPE-Rx

**Single-Cell Oncology Prognosis-anchored Exploration of Repositioning** — a trust-tiered attention-deconvolution framework that fuses single-cell foundation embeddings, patient survival, and pharmacogenomic data into a cross-cancer-validated drug-repositioning pipeline.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2Bcu128-red)
![Status](https://img.shields.io/badge/status-Phase--1b-green)

---

## Highlights

- **Cross-cancer validation, no retuning.** Identical pipeline applied to hepatocellular carcinoma (LIHC) and lung adenocarcinoma (LUAD) recovers Lapatinib (EGFR/HER) at rank 1 in LIHC and Gefitinib (EGFR) at rank 1 in LUAD, with all nine evaluated LUAD standard-of-care drugs in the top 61 of 1,806 ranked compounds.
- **Anti-collapse attention deconvolution.** Non-negative least-squares and Scaden-style MLP deconvolution collapse 95% and 99% of TCGA-LUAD patients onto a single prototype; SCOPE-Rx retains 87% composition entropy with 0% collapse.
- **Three-layer scoring composition is individually necessary.** Mean rank percentile of nine LUAD SOC drugs is 1.6% at default weights, 17.7% with kill-only, 12.6% with prior-only.
- **Generalisation to five external cohorts** spanning four array platforms: composite TCGA-trained risk c-index 0.639 to 0.779 (LIHC + LUAD combined).
- **4-cohort fixed-effect meta-analysis (1,684 LUAD patients)**: 18 prototypes pooled-significant at $p<0.001$.
- **End-to-end runtime under 12 min on a single 16 GB consumer GPU** per cancer.

---

## Pipeline overview

```
scRNA atlas ──┐
              │  Geneformer V2-104M zero-shot embed   →  Leiden cell-state prototypes
              ↓
TCGA bulk  ──→  Trust-tiered attention deconvolution  →  per-patient composition × DepMap trust
              ↓
DepMap + ──┐
GDSC +    │  scDEAL drug-AUC regressor                  →  per-prototype kill score
PRISM   ──┘
              ↓
MPNN(SMILES) ──→  drug-target × pathway-Cox prior  →  per-drug pathway-prior score
              ↓
PRISM oncology / phase / MOA filter                     →  per-drug oncology relevance score
              ↓
Final score  S = z(S_kill) + 0.5·z(S_onc) + 0.7·z(S_prior)
              ↓
Per-patient top-5  +  cohort top-K  +  wet-lab brief
```

---

## Repository structure

```
SCOPE-Rx/
├── README.md
├── LICENSE                      (MIT)
├── environment.yml              (conda env: Python 3.11 + PyTorch 2.10 + CUDA 12.8)
├── .gitignore
├── scripts/
│   ├── preprocess/              T2a — bulk drug-response + cell-line + scRNA gene-universe
│   │   ├── t2a_prepare_hcc_drug_data.py
│   │   ├── t2a_prepare_luad_drug_data.py
│   │   └── t2a_fix_drug_catalog.py
│   ├── baselines/               T2b/T2c — scDEAL, scPDS, Scaden-CA training
│   ├── novelty/                 T3a–T3k — core SCOPE-Rx pipeline (MPNN, Geneformer, attention-deconv, scoring)
│   ├── ablation/                Tier A/B/C ablation + MPNN held-out 5-fold CV
│   ├── figures_main/            5 main figures + 6 supplementary figures + audit_numbers.py
│   ├── report/                  PPT and graphical-abstract builders
│   └── download/                Public-data download helpers (TCGA, GEO, DepMap)
├── manuscript/                  paper.tex, supplementary.tex, cover_letter_CRM.tex, refs.bib
├── results/
│   ├── figures_main/            5 main figures + graphical abstract (PNG + PDF)
│   ├── supp_figures/            6 supplementary figures (PNG + PDF)
│   ├── audit_report.md          67/67 numerical assertions verified
│   ├── citation_verification.md CrossRef DOI lookup of every cited reference
│   ├── tables_main.{md,tex}     5 main tables in markdown + LaTeX booktabs
│   ├── supplementary_index.md   Pointer index for 9 supplementary tables
│   ├── target_journals.md       40 SCI journals matched to paper scope
│   ├── expert_reviews.md        5 simulated expert reviews
│   ├── SCOPE-Rx_LIHC_vs_LUAD_comparison.md
│   ├── phase1_final_report.md   Phase-1 (LIHC) summary
│   └── narrative.md             Manuscript writing guide
├── configs/                     YAML configuration files
└── notebooks/                   Exploratory analysis notebooks
```

Large intermediate parquet outputs, model checkpoints, raw scRNA atlases, and DepMap downloads are excluded from this repository; regenerate them by following the **Installation** and **Reproduction** sections.

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/SCOPE-Rx.git
cd SCOPE-Rx
```

### 2. Create the conda environment
```bash
conda env create -f environment.yml -n drug_sc
conda activate drug_sc
```

> **GPU note.** The pipeline was developed on a single NVIDIA RTX 5070-class card (Blackwell sm_120, 16 GB). For Blackwell GPUs, ensure `torch >= 2.7 + cu128`; older Ampere or Turing cards work with `cu121` builds. End-to-end runtime is approximately 12 minutes per cancer.

### 3. Download the Geneformer V2-104M weights (approximately 400 MB)
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ctheodoris/Geneformer',
    local_dir='models/Geneformer',
    allow_patterns=['Geneformer-V2-104M/*','geneformer/*.py','geneformer/*.pkl'])
"
```

### 4. Download public data

| Resource | Source | Files needed |
|---|---|---|
| **TCGA-LIHC** bulk RNA-seq + clinical | [UCSC Xena](https://xenabrowser.net) | `TCGA_LIHC_expression.gz`, `TCGA_LIHC_clinical.tsv` |
| **TCGA-LUAD** bulk RNA-seq + clinical | UCSC Xena | `TCGA_LUAD_expression.gz`, `TCGA_LUAD_clinical.tsv` |
| **DepMap** | [depmap.org/portal](https://depmap.org/portal/data_page) | `OmicsExpressionTPM.csv`, `Model.csv`, `PRISM_secondary_AUC.csv`, `sanger_dose_response.csv` |
| **LIHC scRNA atlas** | NCBI GEO | GSE125449, GSE149614, GSE156625, GSE162616, GSE202642, GSE223204 |
| **LUAD scRNA atlas** | NCBI GEO | GSE131907 (Kim et al., 2020) |
| **LIHC external cohorts** | NCBI GEO | GSE14520-GPL3921, GSE76427 |
| **LUAD external cohorts** | NCBI GEO | GSE68465 (GPL96), GSE72094 (GPL15048), GSE31210 (GPL570) |
| **Pathway gene sets** | `gseapy` in-app fetch | MSigDB Hallmark, KEGG 2021, Reactome 2022 |

Place data in `data/` with the layout expected by the preprocessing scripts (paths are documented at the top of each `scripts/preprocess/t2a_*.py`).

---

## Reproduction

Run the pipeline end-to-end (per cancer):

```bash
# LIHC
python scripts/preprocess/t2a_prepare_hcc_drug_data.py
python scripts/baselines/t2b_scdeal_train.py
python scripts/novelty/t3b_geneformer_embed.py
python scripts/novelty/t3c_celltype_attention_deconv.py
python scripts/novelty/t3d_signature_reversal_cox.py
python scripts/novelty/t3e_oncology_filter.py
python scripts/novelty/t3f_target_pathway_prior.py
python scripts/novelty/t3i_external_validation_gse76427.py
python scripts/novelty/t3j_two_cohort_meta_analysis.py
python scripts/novelty/t4_integrated_ranking_wetlab_brief.py

# LUAD (identical hyperparameters)
python scripts/preprocess/t2a_prepare_luad_drug_data.py
python scripts/baselines/t2b_scdeal_train_luad.py
python scripts/novelty/t3b_geneformer_embed_luad.py
python scripts/novelty/t3c_celltype_attention_deconv_luad.py
python scripts/novelty/t3d_signature_reversal_cox_luad.py
python scripts/novelty/t3e_oncology_filter_luad.py
python scripts/novelty/t3f_target_pathway_prior_luad.py
python scripts/novelty/t3i_external_validation_luad.py
python scripts/novelty/t3j_meta_analysis_luad.py
python scripts/novelty/t4_integrated_ranking_wetlab_brief_luad.py

# Ablation + figures
python scripts/ablation/mpnn_holdout_drug_eval.py
python scripts/ablation/luad_ablation_table.py
python scripts/figures_main/fig1_pipeline.py
python scripts/figures_main/fig2_attention.py
python scripts/figures_main/fig3_replication.py
python scripts/figures_main/fig4_drug_rediscovery.py
python scripts/figures_main/fig5_ablation.py
python scripts/figures_main/supp/build_supp_figures.py
python scripts/figures_main/build_graphical_abstract.py
python scripts/figures_main/build_tables.py
python scripts/figures_main/audit_numbers.py  # 67/67 numerical assertions verified
```

> **Hardcoded paths.** The scripts contain absolute paths (e.g., `/home/<user>/drug_sc/...`, `/mnt/.../TCGA_*/...`) that point to the development environment. Replace these with paths to your local `data/` directory before running. A future release will refactor to use a single `config.yaml` with a base `data_root`.

---

## Cohorts

| Cohort | Cancer | Platform | n tumour | Endpoints | Role |
|---|---|---|---:|---|---|
| TCGA-LIHC | LIHC | RNA-seq | 423 | OS | Training anchor |
| GSE14520 | LIHC | Affy HG-U133A 2.0 | 225 | OS, RFS | External |
| GSE76427 | LIHC | Illumina HT-12 v4 | 115 | OS, RFS | External |
| TCGA-LUAD | LUAD | RNA-seq | 576 | OS | Training anchor |
| GSE68465 | LUAD | Affy HG-U133A | 462 | OS, RFS | External |
| GSE72094 | LUAD | Affy HuRSTA | 442 | OS | External |
| GSE31210 | LUAD | Affy HG-U133 Plus 2 | 204 | OS, RFS | External |
| **Total** | | | **2,447** | | |

---

## Citation

If you use SCOPE-Rx in your work, please cite:

> Chiu, Y.-J. *Trust-tiered attention deconvolution of single-cell foundation embeddings enables cell-state-resolved survival-anchored drug repositioning across solid tumours.* Manuscript under review.

A formal citation block will be added on publication.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contact

Yen-Jung Chiu — Department of Biomedical Engineering, Chang Gung University, Taoyuan, Taiwan — `d000020163 [at] cgu.edu.tw`
