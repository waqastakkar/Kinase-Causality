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

