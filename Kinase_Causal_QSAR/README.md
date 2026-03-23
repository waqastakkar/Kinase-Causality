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
For compound identity, Script-03 accepts either:
- an explicit `compound_id` column, or
- `standardized_smiles`, which it will reuse as the canonical compound key when `compound_id` is absent.

Script-03 also accepts the current Script-02 aggregated activity column names such as `median_pKi` and `median_ki_nM`, normalizing them internally to the expected `pKi` / `Ki_nM` fields.

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


---

## Script-04: Annotate causal-learning environments for the kinase panel

### Purpose
Read the Script-03 processed kinase regression panel and annotate deterministic compound-, kinase-, source-, and pair-level environments required for downstream causal representation learning, robustness analysis, scaffold/kinase/source-shift benchmarking, and activity-cliff-aware evaluation.

### Scientific role
Script-04 prepares the causal metadata layer used by later steps for environment-aware dataset splitting, invariant learning analyses, robustness testing under scaffold and kinase-family shifts, activity-cliff benchmarking, and later selectivity-margin derivation. This step only creates annotations and diagnostics; it does **not** train any model.

### Required inputs (from Script-03)
Configured under `script_04` in `config.yaml`:
- `data/processed/chembl_human_kinase_panel_long.csv`
- `data/processed/chembl_human_kinase_pki_matrix.csv`
- `data/processed/chembl_human_kinase_observation_mask.csv`
- `data/processed/kinase_panel_summary.csv`
- Optional supplemental provenance metadata via `script_04.supplemental_metadata_path` when assay/document/source columns are not retained upstream

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/04_annotate_environments_for_causal_learning.py
```
Optional custom config:
```bash
python scripts/04_annotate_environments_for_causal_learning.py --config /path/to/config.yaml
```

### Compound environments
Script-04 creates one deterministic annotation row per standardized compound, including:
- Bemis-Murcko scaffold and generic Murcko scaffold
- scaffold frequencies and scaffold-frequency bins
- RDKit-derived descriptors such as heavy-atom count, ring counts, molecular weight, cLogP, H-bond counts, TPSA, rotatable bonds, formal charge, and fraction Csp3
- explicit RDKit parse-success tracking for auditability

### Kinase environments
Script-04 creates one annotation row per retained kinase, including:
- target identifiers and normalized target names
- kinase family / broad group / subfamily metadata when available
- clearly logged fallback family assignment from target-name rules when ChEMBL-style metadata are unavailable
- number of compounds measured, median pKi, pKi spread, and source/document diversity summaries

### Source and provenance environments
When provenance metadata are available, Script-04 summarizes source/document/assay environments by:
- source, document, and assay support counts
- kinase and compound coverage per provenance environment
- source/document frequency bins
- assay diversity within sources

If provenance fields are unavailable, Script-04 continues with explicit `UNAVAILABLE` placeholders and logs the limitation in the JSON report and run log.

### Pair environments
Script-04 creates one row per compound-kinase observation with:
- pKi and compound scaffold context
- kinase family context
- record, assay, document, and source support counts
- multiplicity flags for multi-assay / multi-document / multi-source support
- compound and kinase panel-frequency context features

### Activity-cliff annotations
Script-04 performs kinase-specific activity-cliff analysis using configurable Morgan fingerprints and Tanimoto similarity thresholds. The output flags structurally similar compound pairs with large `pKi` differences and records scaffold agreement diagnostics. Large kinase-specific pair sets are handled with a logged safety limit controlled by config.

### Outputs
- Annotated long-format panel: `data/processed/chembl_human_kinase_panel_annotated_long.csv`
- Compound environments: `data/processed/compound_environment_annotations.csv`
- Kinase environments: `data/processed/kinase_environment_annotations.csv`
- Source environments: `data/processed/source_environment_annotations.csv`
- Pair environments: `data/processed/pair_environment_annotations.csv`
- Activity-cliff annotations: `data/processed/activity_cliff_annotations.csv`
- JSON report: `reports/04_environment_annotation_report.json`
- Log file: `logs/04_annotate_environments_for_causal_learning_YYYYMMDD_HHMMSS.log`
- Config snapshot: `configs_used/04_annotate_environments_for_causal_learning_config.yaml`

### Reproducibility notes
- Script-04 validates required Script-03 inputs and column availability before processing.
- All outputs are written with deterministic sorting for stable downstream use.
- The exact resolved configuration is copied to `configs_used/` when enabled.
- Missing metadata, kinase-family fallback logic, invalid SMILES handling, and skipped activity-cliff computations are recorded in both logs and the JSON report.
- This step prepares causal-learning inputs only and does not fit any machine-learning model.

---

## Script-05: Define selectivity-aware modeling tasks and derived labels

### Purpose
Read the Script-04 annotated kinase panel and convert it into task-ready datasets for later benchmarking. Script-05 preserves the continuous `pKi` regression foundation while deriving selectivity-aware regression targets and optional classification labels.

### Scientific role
Script-05 prepares the task layer required for later multitask `pKi` regression, pairwise selectivity-margin prediction, target-vs-panel selectivity benchmarking, environment-aware causal evaluation, and optional classification baselines. This step defines tasks only; it does **not** train a model or create dataset splits.

### Required input (from Script-04)
Configured under `script_05` in `config.yaml`:
- `data/processed/chembl_human_kinase_panel_annotated_long.csv`
- Optional propagation of activity-cliff flags from `data/processed/activity_cliff_annotations.csv`
- Script-05 validates required compound identity, standardized structure, kinase identity, target-name, and `pKi` columns before generating any output.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/05_define_selectivity_tasks_and_labels.py
```
Optional custom config:
```bash
python scripts/05_define_selectivity_tasks_and_labels.py --config /path/to/config.yaml
```

### Multitask regression task
Script-05 writes one row per observed compound-target pair to:
- `data/processed/task_multitask_regression_long.csv`

This table is the primary downstream regression dataset and retains continuous `pKi` values, optional `Ki_nM`, support/provenance metadata, environment annotations, and propagated activity-cliff flags when available.

### Pairwise selectivity regression task
For compounds measured on at least the configured minimum number of kinases, Script-05 generates within-compound kinase-pair rows with:
- `delta_pKi = pKi_A - pKi_B`
- `abs_delta_pKi`
- directional pair labels when enabled in config

Output:
- `data/processed/task_pairwise_selectivity_regression.csv`

This task supports later selectivity-margin prediction without discarding the underlying continuous activity scale.

### Target-vs-panel selectivity regression task
For each eligible compound-target observation, Script-05 computes:
- target `pKi`
- a config-selected off-target panel reference statistic (for example median, mean, max, or second-best off-target `pKi`)
- `target_vs_panel_delta_pKi = target_pKi - off_target_reference`

Output:
- `data/processed/task_target_vs_panel_selectivity.csv`

This dataset supports one-vs-panel selectivity analyses and later benchmarking against alternative off-target reference definitions.

### Derived classification labels
Script-05 optionally derives publication-traceable classification labels while preserving continuous values in the same output table:
- active vs inactive from `pKi`
- strong binder vs weak binder from `pKi`
- selective / highly selective / non-selective from target-vs-panel `delta_pKi`

Output:
- `data/processed/task_derived_classification_labels.csv`

Thresholds, gray-zone handling, and label provenance columns are recorded directly in the output and JSON report.

### Additional outputs
- Task summary table: `data/processed/task_summary_table.csv`
- JSON report: `reports/05_selectivity_task_report.json`
- Log file: `logs/05_define_selectivity_tasks_and_labels_YYYYMMDD_HHMMSS.log`
- Config snapshot: `configs_used/05_define_selectivity_tasks_and_labels_config.yaml`

### Reproducibility notes
- Script-05 is fully config-driven through `script_05` in `config.yaml`.
- Required files and columns are validated before task generation begins.
- Deterministic sorting is applied to all output tables.
- Thresholds, exclusion rules, gray-zone handling, missing-metadata notes, and label distributions are recorded in the JSON report.
- The exact config used for the run is copied to `configs_used/` when enabled.
- This step defines tasks only and does not fit any machine-learning model.

---

## Script-06: Generate benchmark splits for downstream evaluation

### Purpose
Read the Script-05 task datasets and generate rigorous, reproducible split definitions for later regression, classification, robustness, and causal-QSAR benchmarking. Script-06 defines train/validation/test assignments and subset manifests only; it does **not** train or evaluate any model.

### Scientific role
Script-06 prepares the benchmark split layer needed for:
- baseline model benchmarking
- causal model evaluation
- scaffold generalization testing
- kinase-family transfer testing
- source/environment shift testing
- activity-cliff-aware evaluation
- low-data benchmarking

### Required inputs (from Script-05)
Configured under `script_06` in `config.yaml`:
- `data/processed/task_multitask_regression_long.csv`
- `data/processed/task_pairwise_selectivity_regression.csv`
- `data/processed/task_target_vs_panel_selectivity.csv`
- `data/processed/task_derived_classification_labels.csv`

Optional activity-cliff provenance input:
- `data/processed/activity_cliff_annotations.csv`

Script-06 validates that each task table contains a canonical compound identifier plus the task-specific target columns required for split generation.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/06_generate_benchmark_splits.py
```
Optional custom config:
```bash
python scripts/06_generate_benchmark_splits.py --config /path/to/config.yaml
```

### Split strategies

#### Random split
Script-06 creates deterministic seeded train/validation/test assignments for every task table and also writes seeded k-fold random benchmarking metadata for later cross-validation workflows.

#### Scaffold split
When scaffold metadata are available, Script-06 resolves the first configured scaffold column and assigns whole scaffold groups to train/validation/test to prevent scaffold leakage across partitions.

#### Kinase-family grouped split
When kinase-family annotations are available, Script-06 creates family-held-out assignments so whole kinase families are separated across train/validation/test for transfer and robustness studies.

#### Source/environment grouped split
When provenance/source columns are available, Script-06 creates grouped holdout splits for source, document, or source-frequency environments to support assay/source shift benchmarking.

#### Activity-cliff subsets
When activity-cliff flags are available in the task tables, Script-06 writes manifest files that distinguish cliff-associated and non-cliff observations for later targeted evaluation.

#### Low-data subsets
For multitask regression and classification, Script-06 writes deterministic nested training subsets at configured train sizes while keeping the base validation/test holdout definitions fixed for later few-shot benchmarking.

### Outputs
Script-06 writes a structured split directory:

```text
data/splits/
  split_manifest.csv
  multitask_regression/
  pairwise_selectivity/
  target_vs_panel/
  classification/
```

Within each task-specific split directory, Script-06 writes benchmark artifacts such as:
- row-level split assignment CSVs
- random k-fold assignment and fold-summary CSVs
- grouped split summary CSVs
- activity-cliff subset manifests
- low-data subset manifests when enabled

Top-level outputs:
- Master split manifest: `data/splits/split_manifest.csv`
- JSON report: `reports/06_benchmark_splits_report.json`
- Log file: `logs/06_generate_benchmark_splits_YYYYMMDD_HHMMSS.log`
- Config snapshot: `configs_used/06_generate_benchmark_splits_config.yaml`

### Reproducibility notes
- Script-06 is fully config-driven through `script_06` in `config.yaml`.
- Required input files and required task columns are validated before any split file is written.
- Deterministic sorting and fixed seeded shuffling are used for reproducible split generation.
- Selected grouping columns, split counts, missing-metadata warnings, and subset coverage summaries are recorded in the JSON report.
- The exact config used for the run is copied to `configs_used/` when enabled.
- This step defines splits and subset manifests only and does not fit or evaluate any model.

---

## Script-07: Train classical baseline benchmark models

### Purpose
Script-07 is the non-causal baseline modeling stage for the kinase causal-QSAR benchmark. It trains reproducible classical machine-learning baselines for the Step-05 benchmark tasks using the deterministic split definitions exported by Script-06. These outputs are intended as strong reference baselines for later comparison against graph-learning and causal representation-learning models.

### Scientific scope
Script-07 supports:
- multitask pKi regression via the default `one_model_per_kinase` strategy
- pairwise selectivity-margin regression
- target-vs-panel selectivity regression
- derived binary classification tasks when label columns are available

These are deliberately **non-causal** baselines.

### Required inputs
Configured under `script_07` in `config.yaml`:
- `data/processed/task_multitask_regression_long.csv`
- `data/processed/task_pairwise_selectivity_regression.csv`
- `data/processed/task_target_vs_panel_selectivity.csv`
- `data/processed/task_derived_classification_labels.csv`
- `data/splits/split_manifest.csv`
- row-level split assignment files written by Script-06 and referenced in the manifest

### Descriptor generation
Script-07 uses compound-centric classical descriptors derived from `standardized_smiles`:
- Morgan fingerprints
- RDKit 2D descriptors
- optional environment / provenance categorical features when enabled in config

Descriptor tables are generated once per unique compound, saved when configured, and reused across task assemblies.

### Models trained
Regression baselines:
- Ridge regression
- Random Forest Regressor
- Extra Trees Regressor
- XGBoost Regressor (if installed)
- LightGBM Regressor (if installed)
- SVR with RBF kernel

Classification baselines:
- Logistic Regression
- Random Forest Classifier
- Extra Trees Classifier
- XGBoost Classifier (if installed)
- LightGBM Classifier (if installed)
- SVC with RBF kernel

Script-07 applies deterministic random seeds, uses scaling where appropriate, and supports a small reproducible inner-grid search when enabled.

### Split strategies supported
Script-07 consumes the split strategies defined by Script-06, including:
- random seeded holdout splits
- scaffold holdout splits
- kinase-family grouped holdout splits
- source/environment grouped holdout splits
- any additional manifest-defined holdout variants present in `data/splits/split_manifest.csv`

### Outputs
Primary output roots:
- trained models: `models/classical_baselines/`
- metrics tables: `results/classical_baselines/`
- row-level predictions: `results/classical_baselines/predictions/`
- publication figures: `figures/classical_baselines/`
- JSON run report: `reports/07_classical_baseline_report.json`

Representative metrics outputs:
- `regression_metrics_per_fold.csv`
- `regression_metrics_summary.csv`
- `classification_metrics_per_fold.csv`
- `classification_metrics_summary.csv`
- error-analysis tables for hardest compounds / targets when configured

### Figures
When figure generation is enabled, Script-07 produces manuscript-oriented SVG-first outputs with Times New Roman bold text and a fixed Nature-style palette. Figure source-data CSV files are also exported when applicable.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/07_train_classical_baseline_models.py
```
Optional custom config:
```bash
python scripts/07_train_classical_baseline_models.py --config /path/to/config.yaml
```

### Reproducibility notes
- All key paths and settings are read from `config.yaml`.
- Script-07 saves a config snapshot when enabled.
- Deterministic seeds are fixed across compatible model families.
- Fold-level metrics and row-level predictions are saved for auditing.
- Missing required files, columns, or split assignments trigger clear failures.
- Optional dependencies such as XGBoost and LightGBM are skipped with explicit warnings if unavailable.
---

## Script-08: Graph and deep-learning baseline models

### Purpose
Script-08 trains strong non-causal neural reference baselines for the benchmark tasks defined in Script-05 using the deterministic benchmark splits generated in Script-06. The step covers compound-only graph regression, kinase-aware multitask regression, pairwise selectivity regression, target-vs-panel regression, and optional derived classification tasks.

### Scientific role
These models provide manuscript-grade deep-learning baselines for later comparison against:
- classical baselines from Script-07
- future causal representation-learning models
- robustness analyses under scaffold, kinase-family, and provenance shifts

This step does **not** implement a causal model. It is a neural baseline benchmark only.

### Required inputs
Configured under `script_08` in `config.yaml`:
- `data/processed/task_multitask_regression_long.csv`
- `data/processed/task_pairwise_selectivity_regression.csv`
- `data/processed/task_target_vs_panel_selectivity.csv`
- `data/processed/task_derived_classification_labels.csv`
- `data/splits/split_manifest.csv`
- split assignment files written by Script-06

Optional annotation files are used when available for kinase-family enrichment and downstream reporting consistency:
- `data/processed/compound_environment_annotations.csv`
- `data/processed/kinase_environment_annotations.csv`
- `data/processed/source_environment_annotations.csv`
- `data/processed/activity_cliff_annotations.csv`

### Graph features
Graphs are built from `standardized_smiles` using RDKit and PyTorch Geometric. Node and edge features are config-driven and currently support:
- atom type
- atom degree
- formal charge
- hybridization
- aromaticity
- hydrogen count
- chirality
- bond type
- conjugation
- ring membership
- bond stereochemistry

### Models trained
Configured graph baselines currently support:
- GCN
- GIN
- MPNN
- GAT

Kinase-aware tasks can incorporate:
- target identity embeddings
- kinase-family embeddings
- pair-target embeddings for selectivity tasks

### Split strategies supported
Script-08 consumes the split manifest from Script-06 and trains/evaluates on any supported assignment tables present there, including:
- random splits
- scaffold splits
- kinase-family grouped splits
- source/environment grouped splits
- low-data subsets when generated in Script-06

### Outputs
Primary output locations:
- trained models: `models/deep_baselines/`
- metrics tables: `results/deep_baselines/`
- row-level predictions: `results/deep_baselines/predictions/`
- manuscript-style figures: `figures/deep_baselines/`
- JSON report: `reports/08_deep_baseline_report.json`

At minimum, Script-08 writes:
- `results/deep_baselines/regression_metrics_per_fold.csv`
- `results/deep_baselines/regression_metrics_summary.csv`
- `results/deep_baselines/classification_metrics_per_fold.csv`
- `results/deep_baselines/classification_metrics_summary.csv`
- graph cache metadata and optional error-analysis tables
- per-split prediction tables with observed and predicted values

### Figure outputs
When plotting dependencies are available, Script-08 writes publication-style SVG-first figures and optional PNG/PDF exports for:
- regression model comparisons
- split-strategy comparisons
- classification model comparisons
- observed-vs-predicted scatterplots
- ROC and precision-recall curves for top classification baselines

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/08_train_graph_and_deep_baseline_models.py
```
Optional custom config:
```bash
python scripts/08_train_graph_and_deep_baseline_models.py --config /path/to/config.yaml
```

### Reproducibility notes
- All settings are read from `config.yaml` under `script_08`.
- Deterministic seeds are set for Python, NumPy, and PyTorch where possible.
- The exact config snapshot is copied to `configs_used/` when enabled.
- Split identities are preserved in metrics, predictions, error tables, and the JSON report.
- Graph-construction success/failure metadata are written for auditability.
- Figures use a fixed manuscript-style palette with Times New Roman bold text to stay consistent with downstream publication assets.
- Script-08 fails clearly if required files, columns, PyTorch Geometric, or RDKit are unavailable.


---

## Script-09: Train the main causal environment-aware model

### Purpose
Script-09 is the main methods model of the project. It trains and evaluates the causal, environment-aware graph-learning framework for kinase activity and selectivity tasks using Step-04 environment annotations, Step-05 task tables, and Step-06 benchmark split definitions.

### Scientific role
This step is designed to separate invariant molecular signal from environment-associated spurious signal so later manuscript claims about causal robustness are directly tied to explicit model objectives, diagnostics, and ablations.

### Required inputs (from Steps 04-06)
Configured under `script_09` in `config.yaml`:
- `data/processed/chembl_human_kinase_panel_annotated_long.csv`
- `data/processed/task_multitask_regression_long.csv`
- `data/processed/task_pairwise_selectivity_regression.csv`
- `data/processed/task_target_vs_panel_selectivity.csv`
- `data/processed/task_derived_classification_labels.csv`
- `data/splits/split_manifest.csv`
- Step-06 row-level assignment files referenced by the split manifest

Optional environment-rich inputs that are consumed when present include:
- `data/processed/activity_cliff_annotations.csv`
- `data/processed/compound_environment_annotations.csv`
- `data/processed/kinase_environment_annotations.csv`
- `data/processed/source_environment_annotations.csv`
- `reports/07_classical_baseline_report.json`
- `reports/08_deep_baseline_report.json`

### Supported tasks
- multitask `pKi` regression
- pairwise selectivity-margin regression (`delta_pKi`)
- target-vs-panel selectivity regression (`target_vs_panel_delta_pKi`)
- optional derived binary classification tasks when enabled and sufficiently supported

### Causal objectives supported
- prediction loss for the selected task
- IRM-like invariant risk penalty across resolved environments
- gradient-reversal environment adversarial loss
- supervised environment classification head for diagnostics/ablations
- optional counterfactual consistency placeholder with clean disable behavior
- activity-cliff-aware regularization when cliff flags are available

### Environment types used
Script-09 resolves the first available configured column for each environment family and logs the exact choice:
- scaffold / Murcko scaffold environments
- generic scaffold environments
- kinase-family environments
- source/provenance environments
- activity-cliff flags

If a requested environment is missing, only the dependent objective is disabled; the full script does not fail unless core task requirements are impossible.

### Ablation modes
When `run_core_ablations: true`, Script-09 runs the main model plus configurable ablations such as:
- `no_environment_objectives`
- `no_adversarial_loss`
- `no_invariant_loss`
- `no_activity_cliff_regularization`
- `no_target_embedding`

### Outputs
Structured outputs are written under:
- Models: `models/causal_models/`
- Metrics: `results/causal_models/`
- Predictions and latent/environment exports: `results/causal_models/predictions/`
- Figures: `figures/causal_models/`
- JSON report: `reports/09_causal_model_report.json`

Key tables include:
- `regression_metrics_per_fold.csv`
- `regression_metrics_summary.csv`
- `classification_metrics_per_fold.csv`
- `classification_metrics_summary.csv`
- `ablation_metrics_summary.csv`
- `environment_group_metrics.csv`
- `activity_cliff_metrics.csv`

### Figure outputs
Script-09 generates publication-grade manuscript figures using a fixed Nature-style palette with Times New Roman bold text, exporting SVG as the primary format and optional PNG/PDF copies. Source-data CSV files are saved alongside figure outputs when available.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/09_train_causal_environment_aware_model.py
```
Optional custom config:
```bash
python scripts/09_train_causal_environment_aware_model.py --config /path/to/config.yaml
```

### Reproducibility notes
- All settings are read from `config.yaml` and a config snapshot is saved when enabled.
- The script uses deterministic seeds where feasible for Python, NumPy, and PyTorch.
- Graph construction failures, disabled objectives, skipped splits, and other warnings are written to the log and JSON report.
- Fold-level predictions, metrics, latent embeddings, environment predictions, vocabularies, and model checkpoints can all be saved for full provenance tracking.
- This step is the central causal modeling contribution of the pipeline and is intended for publication-grade benchmarking and robustness analysis.

## Script-10: Global model comparison, robustness analysis, and interpretation

### Purpose

Script-10 integrates the outputs from Steps 07-09 into a unified evaluation layer for manuscript-ready comparison, robustness analysis, and interpretation. It is an evaluation-only step: it does **not** retrain models.

### Scientific role

This step prepares the evidence layer used for:

- cross-family comparison of classical, deep, and causal models
- robustness assessment across split regimes
- activity-cliff sensitivity analysis
- environment-wise error analysis
- kinase-level and compound-level interpretation
- causal ablation interpretation
- low-data and transfer benchmarking
- final figure and table selection for publication

### Required inputs (from Steps 07-09 and earlier annotations)

Configured under `script_10` in `config.yaml`:

- `results/classical_baselines/...` and `reports/07_classical_baseline_report.json`
- `results/deep_baselines/...` and `reports/08_deep_baseline_report.json`
- `results/causal_models/...` and `reports/09_causal_model_report.json`
- `data/processed/chembl_human_kinase_panel_annotated_long.csv`
- optional activity-cliff and environment annotation tables when available
- `data/splits/split_manifest.csv`

### Main outputs

Script-10 writes integrated comparison artifacts under `results/model_comparison/`, including:

- unified per-fold and summary metric tables for regression and classification
- best-model selection tables by task and split strategy
- causal-vs-best-baseline comparison tables
- activity-cliff degradation summaries
- environment-group robustness summaries
- per-kinase, per-compound, and scaffold-level interpretation tables
- ablation drop and rank summaries
- low-data and transfer-gap summaries
- figure source-data tables for reproducible manuscript figures

It also writes figures under `figures/model_comparison/`, a JSON report at `reports/10_model_comparison_and_interpretation_report.json`, a timestamped log under `logs/`, and a config snapshot under `configs_used/`.

### Best-model selection logic

- Regression comparisons use the config-driven primary metric `best_model_selection_metric_regression` (default: `rmse`, lower is better).
- Classification comparisons use `best_model_selection_metric_classification` (default: `roc_auc`, higher is better).
- Model ranking is deterministic and performed within task and split context after schema normalization across earlier steps.
- Causal ablations are compared against the full causal model (`main`) without retraining.

### Activity-cliff and environment analysis

If the required annotations and prediction outputs are available, Script-10 additionally produces:

- activity-cliff vs non-cliff performance tables
- scaffold-group, kinase-family, and source-environment robustness tables
- hardest scaffold groups, hardest kinase families, hardest kinases, and hardest compounds
- per-kinase performance summaries for interpretation and downstream figure preparation

Optional analyses are skipped with explicit logging and warnings in the JSON report when required files are unavailable.

### Run

From the `Kinase_Causal_QSAR` directory:

```bash
python scripts/10_evaluate_compare_and_interpret_models.py --config config.yaml
```

### Reproducibility notes

- Script-10 is fully config-driven and deterministic.
- It saves the exact config snapshot used for the run.
- It records discovered result files, warnings, and summary conclusions in the final JSON report.
- It saves figure source data alongside publication-grade figures using the shared Nature-style palette and Times New Roman bold text.
- It evaluates and compares previously trained models only; it does not modify or retrain earlier model checkpoints.


---

## Script-11: Generate final manuscript figures, tables, and source-data assets

### Purpose
Read the validated benchmarking and interpretation outputs from Steps 07-10 and assemble the final manuscript-ready presentation layer for the kinase causality QSAR study.

This step is strictly presentation-focused:
- it **does not retrain models**
- it **does not rerun benchmark evaluation logic unnecessarily**
- it **does** generate final main figures, supplementary figures, main tables, supplementary tables, source-data CSV files, a manuscript asset manifest, and a JSON report

### Required inputs
Configured under `script_11` in `config.yaml`.

Primary expected Step-10 inputs include files under `results/model_comparison/`, such as:
- `unified_regression_metrics_summary.csv`
- `unified_regression_metrics_per_fold.csv`
- `unified_classification_metrics_summary.csv`
- `unified_ablation_metrics_summary.csv`
- `best_models_by_task.csv`
- `best_models_by_split_strategy.csv`
- `activity_cliff_model_comparison.csv`
- `activity_cliff_degradation_summary.csv`
- `environment_group_metrics.csv`
- `per_kinase_performance_summary.csv`
- `ablation_drop_summary.csv`
- `low_data_performance_summary.csv`
- `low_data_learning_curve_source_data.csv`
- optional transfer-gap and hardest-group summaries when available

Script-11 also records the configured roots for earlier baseline outputs:
- `results/classical_baselines/`
- `results/deep_baselines/`
- `results/causal_models/`

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/11_generate_manuscript_figures_and_tables.py
```
Optional custom config:
```bash
python scripts/11_generate_manuscript_figures_and_tables.py --config /path/to/config.yaml
```

### Outputs
All final manuscript assets are written under `manuscript_outputs/`:
- Main figures: `manuscript_outputs/main_figures/`
- Supplementary figures: `manuscript_outputs/supplementary_figures/`
- Main tables: `manuscript_outputs/main_tables/`
- Supplementary tables: `manuscript_outputs/supplementary_tables/`
- Figure source data: `manuscript_outputs/figure_source_data/`
- Table source data: `manuscript_outputs/table_source_data/`
- Asset manifest: `manuscript_outputs/manuscript_asset_manifest.csv`

Additional reporting outputs:
- JSON report: `reports/11_manuscript_figures_and_tables_report.json`
- Log file: `logs/11_generate_manuscript_figures_and_tables_YYYYMMDDTHHMMSSZ.log`
- Config snapshot: `configs_used/11_generate_manuscript_figures_and_tables_config.yaml`

### Figure naming conventions
Main manuscript figures are written with deterministic manuscript-style names such as:
- `Figure_1.svg`
- `Figure_2.svg`
- ...

Supplementary figures use:
- `Figure_S1.svg`
- `Figure_S2.svg`
- ...

If enabled in config, matching `.png` and `.pdf` copies are also written.

### Table naming conventions
Main manuscript tables use:
- `Table_1.csv`
- `Table_2.csv`
- ...

Supplementary tables use:
- `Table_S1.csv`
- `Table_S2.csv`
- ...

If enabled in config, matching `.xlsx` files are also written.

### Source-data policy
Script-11 writes source-data files for every generated figure and for derived manuscript tables whenever source-data export is enabled.

Examples:
- `manuscript_outputs/figure_source_data/Figure_1_source_data.csv`
- `manuscript_outputs/table_source_data/Table_1_source_data.csv`

These files are intended to reflect the exact transformed values used for plotting or manuscript-table assembly, preserving plotting order and traceability wherever possible.

### Reproducibility and manuscript-style notes
- Script-11 is fully config-driven through `script_11` in `config.yaml`.
- The script validates required inputs before generating each required main asset.
- Missing optional supplementary assets are logged clearly and skipped rather than silently fabricated.
- Matplotlib styling is explicitly controlled to enforce Times New Roman, bold visible text, deterministic ordering, and a fixed Nature-style palette.
- The asset manifest maps each figure/table to its output paths, source-data file, and originating upstream files.
- This step assembles final manuscript assets only and does **not** retrain, resplit, or re-score any model.

---

## Script-12: Package reproducibility and release bundle

### Purpose
Read the validated outputs from Steps 01-11 and assemble the final reproducibility, release, and archival package for the kinase causality QSAR study.

This step is packaging-focused:
- it **does not retrain models**
- it **does not rerun evaluation or regenerate benchmark results unnecessarily**
- it **does** validate required assets, mirror selected outputs into a deterministic release structure, generate a release manifest, write checksums and environment snapshots, and optionally create compressed archives

### Required inputs
Configured under `script_12` in `config.yaml`.

Expected upstream roots may include:
- `data/`
- `models/`
- `results/`
- `figures/`
- `manuscript_outputs/`
- `reports/`
- `logs/`
- `configs_used/`
- `README.md`
- optional dependency files such as `requirements.txt`, `environment.yml`, `pyproject.toml`, and `setup.cfg` when present

Script-12 validates the configured `required_assets` before packaging and fails clearly when mandatory reproducibility assets are missing.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/12_package_reproducibility_and_release.py
```
Optional custom config:
```bash
python scripts/12_package_reproducibility_and_release.py --config /path/to/config.yaml
```

### Outputs
Primary release bundle outputs:
- Release root: `release_package/`
- Release manifest: `release_package/release_manifest.csv`
- Checksum file: `release_package/checksums.txt`
- Environment snapshot: `release_package/environment_snapshot.txt`
- Directory tree snapshot: `release_package/directory_tree.txt`
- Release README: `release_package/README_RELEASE.md`
- Runbook: `release_package/RUNBOOK.md`

Additional reporting outputs:
- JSON report: `reports/12_reproducibility_and_release_report.json`
- Log file: `logs/12_package_reproducibility_and_release_YYYYMMDDTHHMMSSZ.log`
- Config snapshot: `configs_used/12_package_reproducibility_and_release_config.yaml`

Optional archive outputs when enabled:
- `release_archives/kinase_causality_qsar_release.tar.gz`
- `release_archives/kinase_causality_qsar_release.zip`

### Release-package contents
The release bundle mirrors selected project outputs into a deterministic handoff structure, including:
- `data/raw/`, `data/interim/`, `data/processed/`, and `data/splits/`
- `models/` for trained baseline and causal models
- `results/` for benchmark summaries and comparison outputs
- `figures/` for publication-grade benchmarking figures
- `manuscript_outputs/` for final manuscript figures, tables, and source-data assets
- `reports/`, `logs/`, and `configs_used/` for provenance and auditability

### Reproducibility notes
- Script-12 is fully config-driven through `script_12` in `config.yaml`.
- The script uses deterministic file discovery, manifest ordering, checksum ordering, and directory-tree generation.
- It records the exact config used, a packaging log, an environment snapshot, and a machine-readable asset manifest.
- It packages existing project outputs only and does **not** rerun scientific modeling stages.

---

## Script-13A: Prepare and standardize screening libraries

### Purpose
Prepare one or more external or prospective screening libraries for the downstream strategic screening workflow.

This step is preparation-focused:
- it **does standardize and validate raw screening compounds**
- it **does preserve provenance and duplicate traceability**
- it **does write merged, library-specific, QC, manifest, and report assets**
- it **does not score compounds**
- it **does not rank compounds**
- it **does not retrain models**

### Supported input file types
Configured under `script_13a` in `config.yaml`.

Supported library formats include:
- `.csv`
- `.tsv`
- `.txt`
- `.smi` / SMILES flat files
- gzipped flat files when the configured path ends in `.gz`
- `.parquet` when pandas parquet support is available

The script can process multiple screening libraries in one run and resolves the configured file type and delimiter deterministically.

### Required and optional columns
Each configured input library must provide at least one usable SMILES column.

Column resolution is config-driven and uses candidate lists in priority order:
- `smiles_column_candidates`
- `compound_id_column_candidates`
- `extra_metadata_columns`

Typical supported fields include:
- SMILES-like columns such as `smiles`, `SMILES`, `canonical_smiles`, or `standardized_smiles`
- identifier columns such as `compound_id`, `molecule_id`, `catalog_id`, or `ID`
- optional metadata such as vendor/source/library names, price, stock, catalog annotations, and other supplier fields

If no compound identifier column is found, the script generates deterministic library-local identifiers of the form:
- `<library_name>__row_<row_number>`

If no SMILES column can be resolved for a library, the script fails clearly rather than guessing silently.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/13a_prepare_and_standardize_screening_libraries.py
```
Optional custom config:
```bash
python scripts/13a_prepare_and_standardize_screening_libraries.py --config /path/to/config.yaml
```

### Standardization and filtering behavior
Script-13A uses RDKit for deterministic screening-library preparation:
- parses raw SMILES
- removes invalid or empty structures when configured
- standardizes structures with configurable salt removal, largest-fragment retention, normalization, canonicalization, and optional neutralization
- computes simple chemistry QC fields including molecular weight and heavy-atom counts
- filters mixtures, inorganic compounds, out-of-range molecular weights, and overly small fragment-like structures when configured

All removed rows can be written to library-specific failed-row outputs with explicit removal reasons.

### Duplicate handling policy
Duplicate handling is implemented at two levels:

1. **Within-library duplicates**
   - detected on `standardized_smiles`
   - flagged for every duplicate group
   - optionally collapsed to a single retained row when `remove_duplicates_within_library: true`
   - collapsed provenance is preserved through retained-row lineage fields and the provenance table

2. **Across-library duplicates**
   - detected on `standardized_smiles` across all retained libraries
   - annotated with duplicate flags, row counts, and the number of source libraries containing the structure
   - **not removed by default**
   - only collapsed across libraries when `remove_duplicates_across_libraries: true`

This policy preserves shortlist traceability while avoiding silent loss of sourcing information.

### Provenance tracking policy
For each retained screening compound, Script-13A records:
- source library name
- source file path
- original row index
- original compound ID
- original SMILES
- retained standardized SMILES
- within-library duplicate group size
- cross-library duplicate annotations

The provenance table is designed so each retained screening compound can be traced back to the original source row(s) that produced it.

### Outputs
Primary screening-preparation outputs:
- merged screening library: `screening_prepared/merged_screening_library.csv`
- provenance table: `screening_prepared/screening_library_provenance.csv`
- duplicate summary: `screening_prepared/screening_duplicate_summary.csv`
- QC summary: `screening_prepared/screening_qc_summary.csv`
- manifest: `screening_prepared/screening_library_manifest.csv`
- JSON report: `reports/13a_screening_library_preparation_report.json`

Optional library-specific outputs when enabled:
- cleaned per-library tables in `screening_prepared/cleaned_libraries/`
- failed-row tables in `screening_prepared/cleaned_libraries/`
- config snapshot in `configs_used/13a_prepare_and_standardize_screening_libraries_config.yaml`
- timestamped log file in `logs/`

### Reproducibility notes
- Script-13A is fully config-driven through `script_13a` in `config.yaml`.
- Input-library ordering, duplicate retention, output ordering, and manifest generation are deterministic.
- The script records a config snapshot, structured outputs, a machine-readable report, and detailed logging for auditability.
- Removed rows are not silently discarded; they are counted, annotated, and optionally exported with explicit failure reasons.
- This step prepares screening libraries only and does **not** perform screening inference, uncertainty estimation, ranking, or shortlist generation.


---

## Script-13B: Map screening library to model feature space

### Purpose
Map the standardized screening library from Script-13A into deterministic, inference-ready feature spaces that remain aligned with the trained classical, deep/graph, and causal models.

This step is representation-focused:
- it **does generate classical descriptor and fingerprint tables**
- it **does generate graph/deep-model manifests and graph metadata**
- it **does generate environment-style causal covariates and applicability-reference assets**
- it **does validate feature-space consistency against training-time expectations when available**
- it **does not score compounds**
- it **does not rank compounds**
- it **does not retrain any model**

### Required inputs
Configured under `script_13b` in `config.yaml`.

Primary required input from Script-13A:
- `screening_prepared/merged_screening_library.csv`

Expected required columns in the prepared screening library:
- `screening_compound_id`
- `standardized_smiles`

Optional but strongly recommended training-time references:
- `configs_used/07_train_classical_baseline_models_config.yaml`
- `configs_used/08_train_graph_and_deep_baseline_models_config.yaml`
- `configs_used/09_train_causal_environment_aware_model_config.yaml`
- processed training data such as `data/processed/chembl_human_kinase_panel_annotated_long.csv`
- compound / kinase environment annotations from earlier pipeline steps
- previously saved descriptor tables or model feature tables when available

If prior config snapshots cannot be loaded, Script-13B falls back to the `script_13b` section of the current config and records that fallback in logs and the JSON report.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/13b_map_screening_library_to_model_feature_space.py
```
Optional custom config:
```bash
python scripts/13b_map_screening_library_to_model_feature_space.py --config /path/to/config.yaml
```

### Classical feature generation
When `generate_classical_features: true`, Script-13B computes one deterministic feature row per unique `screening_compound_id` using `standardized_smiles`:
- Morgan fingerprints with config-driven radius and bit-length
- RDKit 2D descriptors with deterministic `rdkit_<descriptor_name>` column names
- `rdkit_parse_success` as an audit / QC sentinel
- optional provenance metadata from Script-13A when configured

Where a training-time classical schema can be resolved from prior outputs, the screening feature table is aligned to that schema and the script fails clearly if required feature columns are missing.

Primary output:
- `screening_features/screening_classical_features.csv`

Optional failed-row output:
- `screening_features/failed_classical_feature_rows.csv`

### Graph / deep-model input preparation
When `generate_graph_inputs: true`, Script-13B prepares a graph-ready manifest consistent with Script-08 feature definitions:
- parses `standardized_smiles` with RDKit
- checks the configured atom and bond feature switches
- records graph construction success/failure per compound
- records `number_of_atoms`, `number_of_bonds`, node feature dimensionality, and edge feature dimensionality
- writes a lightweight manifest instead of forcing binary graph serialization

Primary output:
- `screening_features/screening_graph_input_manifest.csv`

Optional failed-row output:
- `screening_features/failed_graph_rows.csv`

### Environment feature generation
When `generate_environment_features: true`, Script-13B computes environment-like and causal covariates for each screening compound, including:
- Murcko scaffold
- generic Murcko scaffold
- molecular weight
- cLogP
- TPSA
- H-bond donor count
- H-bond acceptor count
- rotatable bond count
- heavy atom count
- aromatic ring count
- formal charge
- fraction Csp3

The environment table also preserves available screening-library provenance metadata and can optionally cross-reference earlier environment annotations.

Primary output:
- `screening_features/screening_environment_features.csv`

Optional failed-row output:
- `screening_features/failed_environment_feature_rows.csv`

### Feature-space consistency checks
Script-13B writes explicit feature QC diagnostics to prevent silent inference breakage. At minimum it checks:
- classical fingerprint dimensionality
- RDKit descriptor block availability
- training-schema alignment when a prior schema is available
- graph node and edge feature dimensionality consistency
- required environment-feature availability

Primary QC output:
- `screening_features/screening_feature_qc_summary.csv`

The QC summary records:
- `feature_block`
- `expected_setting`
- `observed_setting`
- `match_flag`
- `severity`
- `notes`

Critical mismatches raise a hard error so later inference stages do not proceed on an invalid feature space.

### Applicability-reference preparation
When `generate_applicability_reference_features: true`, Script-13B prepares basis assets for later novelty and applicability analysis without computing final applicability-domain scores:
- screening-vs-training novelty proxy flags based on standardized SMILES overlap
- optional training reference summary statistics for later downstream screening scripts

Typical outputs include:
- `screening_features/screening_applicability_reference_features.csv`
- `screening_features/training_feature_reference_summary.json`

### Structured outputs
Core output files:
- `screening_features/screening_classical_features.csv`
- `screening_features/screening_graph_input_manifest.csv`
- `screening_features/screening_environment_features.csv`
- `screening_features/screening_feature_qc_summary.csv`
- `screening_features/screening_feature_manifest.csv`
- `reports/13b_screening_feature_mapping_report.json`

Optional provenance / reproducibility outputs:
- failed-row tables in `screening_features/`
- config snapshot in `configs_used/13b_map_screening_library_to_model_feature_space_config.yaml`
- timestamped log file in `logs/`

### Reproducibility notes
- Script-13B is fully config-driven through `script_13b` in `config.yaml`.
- Screening compounds are deterministically ordered by `screening_compound_id` for stable output generation.
- The script preserves the exact mapping between `screening_compound_id`, `standardized_smiles`, and downstream feature assets.
- Config snapshots, detailed logs, manifest entries, QC summaries, and a machine-readable JSON report are written for provenance tracking.
- This step prepares inference-ready features only and does **not** perform model scoring, ranking, or final applicability-domain scoring.


---

## Script-13C: Score the screening library with trained models

### Purpose
Script-13C is the scoring layer for screening inference. It consumes the inference-ready screening assets produced by Script-13B, resolves the best trained models from Steps 07-10, and writes provenance-rich raw prediction tables for downstream consensus scoring, uncertainty estimation, applicability-aware prioritization, strategic ranking, and final shortlist generation. This step generates **raw model scores only**; it does **not** perform final ranking, uncertainty aggregation, applicability scoring, or shortlist bucketing.

### Required inputs
Configured under `script_13c` in `config.yaml`:
- `screening_features/screening_classical_features.csv`
- `screening_features/screening_graph_input_manifest.csv`
- `screening_features/screening_environment_features.csv`
- `screening_features/screening_feature_manifest.csv`
- `results/model_comparison/best_models_by_task.csv` and/or `results/model_comparison/best_models_by_split_strategy.csv`
- trained model artifacts under `models/classical_baselines/`, `models/deep_baselines/`, and `models/causal_models/`

Script-13C validates that the Step-13B screening feature files exist, contain `screening_compound_id` plus `standardized_smiles`, and that selected model artifacts can be traced back to the configured model roots or explicit artifact paths in the model-selection tables.

### Supported task types
- `multitask_regression`: target potency scoring as predicted `pKi`
- `target_vs_panel_regression`: selectivity-oriented scoring as predicted target-vs-panel delta `pKi`
- `pairwise_selectivity_regression`: optional and configuration-gated; requires explicit pair-aware metadata before it can run safely

### Target selection behavior
`script_13c.target_selection_mode` controls how targets are expanded during inference. The default `explicit_list` mode replicates the screening library over the configured `target_chembl_ids`, ensuring one prediction row per compound-target combination for each selected model/task combination.

### Outputs
Script-13C writes structured screening outputs under `screening_scores/` and `reports/`, including:
- `screening_scores/classical_screening_scores.csv`
- `screening_scores/deep_screening_scores.csv`
- `screening_scores/causal_screening_scores.csv`
- `screening_scores/unified_screening_scores.csv`
- `screening_scores/unified_screening_scores_wide.csv`
- `screening_scores/screening_model_metadata.csv`
- `screening_scores/screening_scoring_qc_summary.csv`
- `screening_scores/screening_score_manifest.csv`
- failed-row tables for each enabled model family
- `reports/13c_screening_model_scoring_report.json`

Every prediction row preserves core provenance such as compound identifiers, target identifiers, task name, model family, model name, artifact path, split strategy used for model selection, and prediction value type.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/13c_score_screening_library_with_trained_models.py
```
Optional custom config:
```bash
python scripts/13c_score_screening_library_with_trained_models.py --config /path/to/config.yaml
```

### Reproducibility notes
- Script-13C is fully config-driven and saves an exact config snapshot under `configs_used/` when enabled.
- Outputs are deterministically sorted to preserve stable compound-target-model row ordering.
- The JSON report records the screening feature files, model roots, model-selection tables, task/target coverage, failed-row counts, warnings, and config snapshot reference.
- Classical screening requires exact feature-schema alignment with the saved training artifact metadata.
- Deep and causal screening intentionally require traceable inference artifacts or reconstructable inference bundles; the script fails clearly if only an opaque state dict is available and reproducible inference cannot be validated.


## Step-13D: Strategic screening ranking integration

### Purpose
Script-13D converts the raw screening predictions from Script-13C into a strategic decision layer for screening. It integrates potency-oriented predictions, selectivity-oriented predictions, cross-family consensus/disagreement summaries, uncertainty proxies, applicability-domain readiness proxies, and diversity-preparatory signals into interpretable ranking tables. This step writes strategic scores only and **does not** create the final shortlist buckets; shortlist generation remains deferred to Step-13E.

### Required inputs
Configured under `script_13d` in `config.yaml`:
- Step-13C score files:
  - `screening_scores/classical_screening_scores.csv`
  - `screening_scores/deep_screening_scores.csv`
  - `screening_scores/causal_screening_scores.csv`
  - `screening_scores/unified_screening_scores.csv`
- Step-13B feature files:
  - `screening_features/screening_classical_features.csv`
  - `screening_features/screening_environment_features.csv`
  - Step-13B QC/provenance files may be used as supporting references
- Step-13A screening library file:
  - `screening_prepared/merged_screening_library.csv`
- Optional training/applicability references:
  - `data/processed/chembl_human_kinase_panel_annotated_long.csv`
  - `data/processed/compound_environment_annotations.csv`

Script-13D validates that the unified Step-13C screening score table exists, contains the required compound/model/task/value columns, and includes at least one potency-like prediction type before ranking begins.

### Strategic score components
Script-13D resolves score types explicitly and preserves them separately:
- potency predictions, such as `predicted_pKi`
- selectivity predictions, such as target-vs-panel delta `pKi` or pairwise delta `pKi`
- auxiliary prediction types, if present, without silently merging them into the main potency/selectivity channels

For each compound-target entity, the script summarizes each enabled model family using:
- family mean / median / min / max prediction
- within-family standard deviation
- number of contributing models

The composite strategic score can include:
- potency component
- selectivity component
- uncertainty penalty
- applicability penalty
- diversity bonus placeholder

Weights, source-family preferences, and normalization behavior are all read from `script_13d` in `config.yaml` and are written back to the ranking outputs/report for full interpretability.

### Uncertainty proxy policy
Script-13D computes **uncertainty proxies**, not calibrated posterior uncertainty. These proxies can include:
- cross-model standard deviation
- cross-family disagreement
- within-family disagreement
- disagreement between best causal and best non-causal prediction
- insufficient-model-support flags

All uncertainty-related outputs are labeled as disagreement/support proxies and should be interpreted as practical prioritization aids rather than formal uncertainty quantification.

### Applicability proxy policy
Script-13D computes **applicability proxies**, not exact domain certification. Supported proxy signals include:
- descriptor range violations relative to training references
- descriptor-distance proxies to the training descriptor distribution
- scaffold novelty flags relative to training scaffolds
- physicochemical out-of-range flags

These are intended for triage and auditability. Novelty or range violations must not be interpreted as guaranteed out-of-domain status.

### Output files
Script-13D writes strategic ranking assets under `screening_rankings/` and `reports/`, including:
- `screening_rankings/compound_target_strategic_ranking.csv`
- `screening_rankings/compound_level_strategic_ranking.csv`
- `screening_rankings/screening_consensus_summary.csv`
- `screening_rankings/screening_uncertainty_summary.csv`
- `screening_rankings/screening_applicability_summary.csv`
- `screening_rankings/screening_ranking_manifest.csv`
- optional intermediate component tables and failed-row tables when enabled
- `reports/13d_strategic_screening_ranking_report.json`

The compound-target ranking table preserves component-level raw and normalized values, weight metadata, target-aware ranks, provenance references, and policy notes. The compound-level ranking table rolls up the target-aware outputs to compound-level best/mean strategic scores and supporting summary metrics.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/13d_build_strategic_screening_rankings.py
```
Optional custom config:
```bash
python scripts/13d_build_strategic_screening_rankings.py --config /path/to/config.yaml
```

### Reproducibility notes
- Script-13D is fully config-driven through `script_13d` in `config.yaml`.
- Deterministic sorting/ranking is used to stabilize output ordering and ranking ties.
- The exact config can be snapshotted under `configs_used/` when enabled.
- Ranking outputs preserve the mapping from `screening_compound_id` to raw component values, normalized component values, composite weights, and final strategic scores.
- The JSON report records input paths, component toggles, weight values, warning messages, failed-row counts, uncertainty/applicability summary statistics, and the config snapshot reference.
- This step produces strategic ranking intelligence for downstream triage but does **not** generate the final shortlist buckets yet.


## Step-13E: Final screening shortlist bucket generation

### Purpose
Script-13E converts the Step-13D strategic ranking intelligence into the final actionable screening shortlist. It fills explicit shortlist buckets rather than taking a single top-N slice, so the final candidate panel can balance exploitation, novelty, chemical breadth, and consensus-backed fallback coverage for docking, sourcing, synthesis, and manuscript reporting.

### Required inputs
Configured under `script_13e` in `config.yaml`:
- Step-13D ranking outputs:
  - `screening_rankings/compound_target_strategic_ranking.csv`
  - `screening_rankings/compound_level_strategic_ranking.csv`
  - `screening_rankings/screening_uncertainty_summary.csv`
  - `screening_rankings/screening_applicability_summary.csv`
  - `screening_rankings/screening_consensus_summary.csv`
- Step-13A/13B metadata:
  - `screening_prepared/merged_screening_library.csv`
  - `screening_features/screening_environment_features.csv`
  - `screening_prepared/screening_library_provenance.csv`
- Optional purchasability, pricing, stock, or target metadata may be propagated when present in the prepared screening library/provenance tables.

The script fails clearly if the required ranking files or minimum identifier/score columns are missing, or if no final strategic score is available for shortlist construction.

### Supported shortlist modes
- `compound_level` (default): one final shortlist row per screening compound for purchase or follow-up prioritization.
- `compound_target`: one final shortlist row per compound-target pair for target-specific screening campaigns.

The mode is resolved entirely from `script_13e.shortlist_mode` in `config.yaml`, and target-aware outputs remain available for primary-target exports when target identifiers are present.

### Bucket definitions
Script-13E uses explicit config-driven bucket rules and sizes from `script_13e.bucket_rules` and `script_13e.bucket_sizes`:
- `high_confidence_selective_hits`: exploits the strongest strategic candidates using top-score potency/selectivity support plus low uncertainty/applicability proxy penalties.
- `novel_scaffold_selective_hits`: preserves scaffold novelty while maintaining selective prioritization.
- `diverse_exploratory_hits`: broadens chemistry coverage with rare or underrepresented scaffolds instead of only taking the highest global scores.
- `consensus_supported_fallback_hits`: retains backup candidates with strong cross-model consensus and acceptable proxy-risk profiles.

Eligibility thresholds are applied transparently from config quantiles/flags; Step-13D ranking logic is not recomputed or silently altered.

### Diversity control rules
- Diversity control is scaffold-based by default and is governed by `script_13e.diversity_controls`.
- `max_compounds_per_exact_scaffold` and `max_compounds_per_generic_scaffold` cap shortlist concentration within each bucket.
- When `enforce_scaffold_diversity_within_bucket` is enabled, bucket candidates are filtered deterministically after ranking.
- If `use_fingerprint_diversity_selection` is `false`, the script logs that only scaffold-based diversity was used.

### Deduplication policy
- Cross-bucket deduplication is controlled by `script_13e.deduplicate_across_buckets`.
- Bucket priority order is taken from `script_13e.prioritize_higher_priority_bucket_order`.
- When enabled, compounds are assigned to the highest-priority bucket they qualify for, and rationale outputs record multi-bucket qualification/deduplication traces.
- Deterministic sorting and stable tie-breaking preserve reproducible bucket assignments across reruns with the same inputs.

### Output files
Script-13E writes shortlist assets under `screening_shortlist/` and `reports/`, including:
- `screening_shortlist/final_screening_shortlist.csv`
- `screening_shortlist/final_shortlist_rationale.csv`
- `screening_shortlist/shortlist_bucket_summary.csv`
- `screening_shortlist/shortlist_diversity_summary.csv`
- `screening_shortlist/screening_shortlist_manifest.csv`
- bucket-specific files such as `screening_shortlist/high_confidence_selective_hits.csv` when enabled
- target-specific shortlist files for configured primary targets when enabled
- `reports/13e_screening_shortlist_report.json`

The final shortlist preserves bucket assignment, strategic/component scores, scaffold annotations, consensus and proxy metrics, source-library metadata, and provenance-aware sourcing fields where available.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/13e_generate_screening_shortlist_buckets.py
```
Optional custom config:
```bash
python scripts/13e_generate_screening_shortlist_buckets.py --config /path/to/config.yaml
```

### Reproducibility notes
- Script-13E is fully config-driven through `script_13e` in `config.yaml`.
- The exact config can be snapshotted under `configs_used/` when enabled.
- Output ordering, bucket ranking, deduplication, and scaffold filtering use deterministic sorting and stable tie-breaking.
- Rationale and manifest outputs preserve the exact mapping from `screening_compound_id` (and `target_chembl_id` when applicable) to shortlist-bucket assignment.
- The JSON report records input paths, requested vs achieved bucket sizes, diversity summaries, metadata preference usage, warnings, and the config snapshot reference.
- This step generates the final actionable screening shortlist for downstream docking/refinement review, purchase prioritization, experimental follow-up, and manuscript reporting.

## Step-13F: Screening analysis and publication-grade visualization

### Purpose
Script-13F is the final interpretation layer for the strategic screening workflow. It consumes the fixed outputs of Steps-13A through 13E and generates manuscript-ready chemical-space maps, shortlist interpretation figures, diversity/scaffold summaries, target-wise overlays, score landscapes, source-data files, a screening-analysis manifest, and a provenance-rich JSON report. This step analyzes and visualizes the screening results only; it does **not** retrain any models and does **not** alter shortlist membership.

### Required inputs
Configured under `script_13f` in `config.yaml`:
- Step-13A/13B inputs:
  - `screening_prepared/merged_screening_library.csv`
  - `screening_features/screening_classical_features.csv`
  - `screening_features/screening_environment_features.csv`
- Step-13C/13D inputs:
  - `screening_scores/unified_screening_scores.csv`
  - `screening_rankings/compound_target_strategic_ranking.csv`
  - `screening_rankings/compound_level_strategic_ranking.csv`
- Step-13E inputs:
  - `screening_shortlist/final_screening_shortlist.csv`
  - `screening_shortlist/final_shortlist_rationale.csv`
  - `screening_shortlist/shortlist_bucket_summary.csv`
  - `screening_shortlist/shortlist_diversity_summary.csv`
- Optional training-reference inputs for novelty/interpolation analysis:
  - `data/processed/chembl_human_kinase_panel_annotated_long.csv`
  - `data/processed/compound_environment_annotations.csv`

The script validates required input files and required identifier columns before each major analysis stage and fails clearly when indispensable shortlist/ranking assets are missing.

### Embedding methods and deterministic analysis policy
Script-13F is fully config-driven through `script_13f` in `config.yaml`.
- Embedding inputs are built from Morgan fingerprints when RDKit is available.
- Numeric descriptor fallbacks and optional RDKit descriptor fallbacks are available when configured.
- UMAP, PCA, and optional t-SNE are controlled independently from config.
- Random-state settings are fixed for deterministic reruns whenever the underlying libraries support deterministic behavior.
- Coordinate tables can be saved for each generated embedding, including combined training-vs-screening coordinates when optional training compounds are available.

### Figure outputs
Script-13F writes publication-grade figures under `screening_analysis/figures/`, including configurable subsets of:
- full screening-library chemical-space maps
- shortlist overlay maps
- training-vs-screening comparison maps
- bucket-specific shortlist maps
- potency/selectivity/uncertainty/applicability/final-score landscapes
- shortlist rationale composition plots
- scaffold diversity and source-library contribution plots
- target-specific overlay panels for configured targets

All figures use the shared manuscript style: Times New Roman, bold visible text, Nature-style palette control, SVG-first export, and optional PNG/PDF copies.

### Table outputs
Script-13F writes analysis tables under `screening_analysis/tables/`, including configurable and derived assets such as:
- `screening_umap_coordinates.csv`
- `screening_pca_coordinates.csv`
- `combined_training_screening_umap_coordinates.csv`
- `screening_embedding_summary.csv`
- `shortlist_bucket_statistics.csv`
- `top_compounds_by_bucket.csv`
- `target_specific_shortlist_summary.csv`
- `shortlist_score_distribution_summary.csv`
- `shortlist_novelty_summary.csv`
- `scaffold_frequency_summary.csv`
- `bucket_scaffold_distribution.csv`
- `source_library_contribution_summary.csv`
- `bucket_scaffold_overlap_matrix.csv`

### Source-data and manifest policy
- Every major figure can emit a matching CSV under `screening_analysis/source_data/`.
- Source-data files are intended to match the plotted rows and preserve plotting-relevant columns/order wherever possible.
- `screening_analysis/screening_analysis_manifest.csv` records each figure/table asset together with its source-data reference, panel label, row count, and notes.
- `reports/13f_screening_analysis_report.json` records inputs used, embedding methods, counts of analyzed screening/shortlist/training compounds, generated assets, missing optional analyses, warnings, timestamp, log path, and config-snapshot reference.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/13f_analyze_and_visualize_screening_results.py
```
Optional custom config:
```bash
python scripts/13f_analyze_and_visualize_screening_results.py --config /path/to/config.yaml
```

### Reproducibility notes
- Script-13F is a strict continuation of Steps-13A through 13E and does not modify shortlist membership or upstream ranking outputs.
- The exact config can be snapshotted under `configs_used/13f_analyze_and_visualize_screening_results_config.yaml` when enabled.
- The script writes a dedicated log file, reusable embedding-coordinate tables, source-data CSVs, a manifest, and a JSON report for provenance tracking.
- Plot styling is enforced explicitly rather than relying on matplotlib defaults so reruns remain visually consistent.
- Optional analyses are skipped only with clear warnings in the log/report.

