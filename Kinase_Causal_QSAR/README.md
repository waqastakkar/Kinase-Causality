# Kinase_Causal_QSAR

## Extract high-confidence human kinase Ki data from ChEMBL

### Prerequisites
- Python 3.10+
- A local ChEMBL SQLite database file

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure
Edit `config.yaml` and set `chembl_sqlite_path` to your local SQLite file.

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
