# Kinase_Causal_QSAR

## Pipeline overview
This repository implements a stepwise kinase causal-QSAR pipeline. Each script is a strict continuation of the previous script and records reproducibility artifacts (logs, config snapshots, reports, and structured outputs).

## Prerequisites
- Python 3.10+
- A local ChEMBL SQLite database file

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure
Edit `config.yaml` and set `chembl_sqlite_path` to your local SQLite file. Script-specific paths, endpoint-handling rules, and thresholds for Script-02 are configurable under `script_02`.

---

## Script-01: Diagnostic extraction for human kinase bioactivity data

### Purpose
Run a staged diagnostic extraction to determine exactly where all-kinase row loss happens (endpoint filters, confidence-score filters, kinase classification, or join behavior), while also exporting both broad and strict datasets.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/01_extract_human_kinase_ki.py --mode broad
```
or
```bash
python scripts/01_extract_human_kinase_ki.py --mode strict
```
Optional custom config:
```bash
python scripts/01_extract_human_kinase_ki.py --config /path/to/config.yaml --mode broad
```

### Broad vs strict mode
- `--mode broad`
  - Confidence: `confidence_score IN (8,9)`
  - Endpoints: `standard_type IN ('Ki','IC50','Kd')`
- `--mode strict`
  - Confidence: `confidence_score = 9`
  - Endpoints: `standard_type = 'Ki'`

Both modes are always executed internally, and if non-empty both datasets are saved. `--mode` only controls which query is written to `output_sql_path` and which dataset is written to the legacy `output_csv_path`.

### Diagnostic stages (A-J)
Script-01 logs row counts and unique TID counts for:
- **A**: `assay_type='B'`, `standard_relation='='`, `standard_units='nM'`, `standard_value>0`
- **B**: A + single-protein target type
- **C**: B + `organism='Homo sapiens'`
- **D**: C + `confidence_score IN (8,9)`
- **E**: C + `confidence_score = 9`
- **F**: D + `standard_type IN ('Ki','IC50','Kd')`
- **G**: E + `standard_type='Ki'`
- **H**: independent kinase-target extraction (flexible classification matching across available text/classification columns)
- **I**: F joined to kinase targets
- **J**: G joined to kinase targets

### Outputs
- Broad dataset (if non-empty): `data/raw/chembl_human_kinase_broad_raw.csv`
- Strict dataset (if non-empty): `data/raw/chembl_human_kinase_strict_raw.csv`
- Legacy selected-mode dataset (if non-empty): `data/raw/chembl_human_kinase_ki_raw.csv`
- Diagnostics JSON: `reports/01_extraction_diagnostics.json`
- Stage previews for non-empty stages: `data/raw/debug_stage_A.csv` ... `data/raw/debug_stage_J.csv`
- Kinase target list: `data/raw/debug_kinase_targets.csv`
- SQL used for selected mode: `sql/kinase_ki_extraction.sql`
- Run logs: `logs/extract_human_kinase_ki_YYYYMMDD_HHMMSS.log`

### How to interpret stage counts
- Large drop **C -> D/E**: confidence filtering is the bottleneck.
- Large drop **D -> F** or **E -> G**: endpoint restriction is the bottleneck.
- Large drop **F -> I** or **G -> J**: kinase classification or kinase-join logic is the bottleneck.
- Stage **H** near zero: likely kinase-target identification issue.

### Why zero rows can occur
- Endpoint restriction too narrow (e.g., Ki-only misses IC50/Kd records).
- Confidence restriction too narrow (e.g., score 9-only excludes score 8 records).
- Kinase classification misses valid kinase targets because classification fields vary by ChEMBL build.
- Join mismatch between filtered activities and kinase target IDs.

Note: variant/mutant exclusion is intentionally disabled at this diagnostic stage and should only be reintroduced after base kinase extraction is confirmed non-empty.

---

## Script-02: Curate and aggregate kinase Ki records

### Purpose
Read Script-01 raw output and perform chemical curation + deterministic aggregation to generate publication-ready interim kinase Ki data.

### Required input (from Script-01)
- Script-02 reads **only** `script_02.input_csv_path` from `config.yaml`.
- For Step-2, point `script_02.input_csv_path` to Script-01's strict Ki output: `data/raw/chembl_human_kinase_strict_raw.csv`.
- Do **not** rely on the legacy selected-mode file `data/raw/chembl_human_kinase_ki_raw.csv` for Step-2, because that file may now contain broad mixed endpoints depending on how Script-01 was run.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/02_curate_and_aggregate_kinase_ki.py
```
Optional custom config:
```bash
python scripts/02_curate_and_aggregate_kinase_ki.py --config /path/to/config.yaml
```

### Outputs
- Curated and aggregated long-format dataset: `data/interim/chembl_human_kinase_ki_curated_long.csv`
- Duplicate/replicate aggregation summary: `data/interim/chembl_human_kinase_ki_duplicate_summary.csv`
- Per-kinase record counts: `data/interim/kinase_record_counts.csv`
- Standardized intermediate records (pre-aggregation): `data/interim/chembl_human_kinase_ki_standardized_records.csv`
- Endpoint summary written before any activity transformation: `reports/02_endpoint_summary.csv`
- Curation report: `reports/02_curation_report.json`
- Log file: `logs/02_curate_and_aggregate_kinase_ki_YYYYMMDD_HHMMSS.log`
- Config snapshot: `configs_used/02_curate_and_aggregate_kinase_ki_config.yaml`

### Reproducibility notes
- Script-02 fails clearly if the configured Script-01 output is missing.
- Required columns and `standard_type` values are validated before any activity transformation.
- If the configured input contains only `Ki` in `nM`, Script-02 computes pKi from Ki.
- If the configured input contains mixed endpoint types (for example `Ki`, `IC50`, `Kd`), Script-02 logs endpoint counts, saves `reports/02_endpoint_summary.csv`, and then either stops with a clear error (`endpoint_handling: error`) or filters explicitly according to `script_02.allowed_standard_types` (`endpoint_handling: filter`).
- Filtering counts and provenance metadata are written to `reports/02_curation_report.json`.
- The exact run configuration (with resolved paths and parameters) is written to `configs_used/`.
- Deterministic sorting is used for stable output generation.

### Figure output notes
- Script-02 does not generate figures.
- A fixed manuscript-grade color palette is stored in `config.yaml` for reuse by future plotting scripts.

---

## Script-03: Build the final kinase panel and sparse pKi matrix

### Purpose
Read the Script-02 curated regression-ready long-format dataset, select the final kinase study panel using configurable density thresholds, quantify cross-kinase compound overlap, and export sparse matrix artifacts for downstream causal-QSAR benchmarking.

### Scientific role
Script-03 is the bridge between cleaned continuous `pKi` regression data from Script-02 and later multitask, sparse-matrix, selectivity, and causal-learning stages. This step prepares the final panel and matrix structures only; it does **not** perform any model training, selectivity labeling, causal environment creation, or activity classification.

### Required input (from Script-02)
- `script_03.input_csv_path` in `config.yaml`
- Default expected path: `data/interim/chembl_human_kinase_ki_curated_regression_long.csv`

Script-03 validates that the configured Script-02 output contains the expected aggregated compound, kinase, and `pKi` columns before building any panel outputs.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/03_build_kinase_panel_and_matrix.py
```
Optional custom config:
```bash
python scripts/03_build_kinase_panel_and_matrix.py --config /path/to/config.yaml
```

### Panel selection logic
The final kinase panel is selected entirely from `script_03` thresholds in `config.yaml`.
At minimum, each kinase must satisfy:
- `min_records_per_kinase`
- `min_unique_compounds_per_kinase`

After kinase filtering, compounds can optionally be removed when they are measured against fewer than `min_kinases_per_compound` selected kinases, controlled by `drop_singleton_compounds`.

### Sparse matrix outputs
Script-03 builds two aligned compound × kinase matrices:
- `chembl_human_kinase_pki_matrix.csv`: sparse regression matrix with aggregated `pKi` values and missing entries left as missing (`NaN`/blank)
- `chembl_human_kinase_observation_mask.csv`: binary observation mask with `1` for measured pairs and `0` for missing pairs

No imputation is performed in this step.

### Outputs
- Final long-format panel dataset: `data/processed/chembl_human_kinase_panel_long.csv`
- Sparse compound × kinase pKi matrix: `data/processed/chembl_human_kinase_pki_matrix.csv`
- Observation mask matrix: `data/processed/chembl_human_kinase_observation_mask.csv`
- Final kinase summary table: `data/processed/kinase_panel_summary.csv`
- Final compound summary table: `data/processed/compound_panel_summary.csv`
- Kinase shared-compound overlap matrix: `data/processed/kinase_compound_overlap_matrix.csv`
- Panel report: `reports/03_kinase_panel_report.json`
- Log file: `logs/03_build_kinase_panel_and_matrix_YYYYMMDD_HHMMSS.log`
- Config snapshot: `configs_used/03_build_kinase_panel_and_matrix_config.yaml`
- Optional diagnostics when enabled: dense matrix copy plus Jaccard and overlap-coefficient matrices

### Reproducibility notes
- Script-03 fails clearly if the configured Script-02 input file is missing or lacks required columns.
- All filtering decisions, selected thresholds, overlap diagnostics, matrix dimensions, density statistics, and removed-kinase reasons are captured in the JSON report.
- Deterministic row and column sorting is used for stable output generation.
- The exact resolved configuration is copied to `configs_used/` for auditability.
- This step prepares modeling inputs only and does not fit any machine-learning model.
