# Kinase-Causality

Project code and pipeline documentation live under `Kinase_Causal_QSAR/`.

## Latest pipeline addition
Step-11 (`Kinase_Causal_QSAR/scripts/11_generate_manuscript_figures_and_tables.py`) assembles the final manuscript-grade figures, tables, figure/table source-data files, and manifest/report assets from the validated outputs of Steps 07-10 without retraining or reevaluating models. See `Kinase_Causal_QSAR/README.md` for full usage, required inputs, naming conventions, outputs, and reproducibility details.

Step-12 (`Kinase_Causal_QSAR/scripts/12_package_reproducibility_and_release.py`) packages the validated outputs from Steps 01-11 into `Kinase_Causal_QSAR/release_package/`, writes `release_manifest.csv`, `checksums.txt`, environment and directory snapshots, optionally creates `Kinase_Causal_QSAR/release_archives/`, and records a final reproducibility report without rerunning modeling. See `Kinase_Causal_QSAR/README.md` for the full Step-12 packaging workflow and reproducibility notes.
