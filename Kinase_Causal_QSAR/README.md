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
Edit `config.yaml` and set `chembl_sqlite_path` to your local SQLite file. Script-specific paths and thresholds for Script-02 are also configurable under `script_02`.

---

## Script-01: Extract high-confidence human kinase Ki data

### Purpose
Extract high-confidence human single-protein kinase Ki records from a local ChEMBL SQLite database.

### Run
From `Kinase_Causal_QSAR/`:
```bash
python scripts/01_extract_human_kinase_ki.py
```
Optional custom config:
```bash
python scripts/01_extract_human_kinase_ki.py --config /path/to/config.yaml
```

### Outputs
- Raw extracted data: `data/raw/chembl_human_kinase_ki_raw.csv`
- SQL used for extraction: `sql/kinase_ki_extraction.sql`
- Run logs: `logs/extract_human_kinase_ki_YYYYMMDD_HHMMSS.log`

### Reproducibility notes
- Extraction SQL is saved exactly as executed.
- Logs capture schema decisions and filters.


### Diagnostics / debugging zero-row extractions
Use Script-01 diagnostics to identify which filter stage collapses row counts:

```bash
python scripts/01_extract_human_kinase_ki.py
```

The script logs and prints counts for staged filters:
- **Stage A**: base Ki activity constraints only (`assay_type=B`, `confidence_score=9`, `Ki`, `=`, `nM`, `standard_value>0`)
- **Stage B**: Stage A + single-protein target type
- **Stage C**: Stage B + human organism (`Homo sapiens`)
- **Stage D**: independent count of kinase-classified targets via protein-classification tables
- **Stage E**: Stage C + kinase classification join restriction
- **Stage F**: Stage E + variant/mutant exclusion

Artifacts:
- Diagnostics JSON: `reports/01_extraction_diagnostics.json`
- Stage preview CSVs (for non-empty stages): `data/raw/debug_stage_A.csv`, `..._B.csv`, etc.

To isolate likely causes of zero rows:

Disable kinase restriction:
```bash
python scripts/01_extract_human_kinase_ki.py --disable_kinase_classification
```

Disable variant exclusion:
```bash
python scripts/01_extract_human_kinase_ki.py --disable_variant_filter
```

Disable both (base extraction behavior without kinase/variant constraints):
```bash
python scripts/01_extract_human_kinase_ki.py --disable_kinase_classification --disable_variant_filter
```

Interpretation tip: compare where counts drop sharply (for example C->E indicates kinase classification is too strict; E->F indicates variant filtering is too aggressive).

---

## Script-02: Curate and aggregate kinase Ki records

### Purpose
Read Script-01 raw output and perform chemical curation + deterministic aggregation to generate publication-ready interim kinase Ki data.

### Required input (from Script-01)
- `data/raw/chembl_human_kinase_ki_raw.csv`

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
- Curation report: `reports/02_curation_report.json`
- Log file: `logs/02_curate_and_aggregate_kinase_ki_YYYYMMDD_HHMMSS.log`
- Config snapshot: `configs_used/02_curate_and_aggregate_kinase_ki_config.yaml`

### Reproducibility notes
- Script-02 fails clearly if Script-01 output is missing.
- Required columns are validated before processing.
- Filtering counts and provenance metadata are written to `reports/02_curation_report.json`.
- The exact run configuration (with resolved paths and parameters) is written to `configs_used/`.
- Deterministic sorting is used for stable output generation.

### Figure output notes
- Script-02 does not generate figures.
- A fixed manuscript-grade color palette is stored in `config.yaml` for reuse by future plotting scripts.
