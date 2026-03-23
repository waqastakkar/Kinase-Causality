# Kinase-Causality

Project code and pipeline documentation live under `Kinase_Causal_QSAR/`.

## Latest pipeline addition
Step-11 (`Kinase_Causal_QSAR/scripts/11_generate_manuscript_figures_and_tables.py`) assembles the final manuscript-grade figures, tables, figure/table source-data files, and manifest/report assets from the validated outputs of Steps 07-10 without retraining or reevaluating models. See `Kinase_Causal_QSAR/README.md` for full usage, required inputs, naming conventions, outputs, and reproducibility details.

Step-12 (`Kinase_Causal_QSAR/scripts/12_package_reproducibility_and_release.py`) packages the validated outputs from Steps 01-11 into `Kinase_Causal_QSAR/release_package/`, writes `release_manifest.csv`, `checksums.txt`, environment and directory snapshots, optionally creates `Kinase_Causal_QSAR/release_archives/`, and records a final reproducibility report without rerunning modeling. See `Kinase_Causal_QSAR/README.md` for the full Step-12 packaging workflow and reproducibility notes.

Step-13A (`Kinase_Causal_QSAR/scripts/13a_prepare_and_standardize_screening_libraries.py`) begins the strategic screening workflow by standardizing external screening libraries, writing merged and library-specific cleaned outputs, preserving duplicate/provenance mappings, and exporting QC/manifest/report assets without performing inference or ranking. See `Kinase_Causal_QSAR/README.md` for supported input formats, required columns, outputs, and reproducibility notes.

Step-13B (`Kinase_Causal_QSAR/scripts/13b_map_screening_library_to_model_feature_space.py`) continues the screening workflow by aligning the standardized screening library with the trained classical, graph/deep, and causal feature spaces, exporting inference-ready feature tables/manifests, feature-QC diagnostics, and provenance reports without scoring or ranking compounds. See `Kinase_Causal_QSAR/README.md` for required inputs, outputs, QC behavior, and reproducibility notes.


Step-13D (`Kinase_Causal_QSAR/scripts/13d_build_strategic_screening_rankings.py`) integrates Step-13C screening predictions into target-aware and compound-level strategic ranking tables, consensus/disagreement summaries, uncertainty/applicability proxy summaries, diversity-readiness signals, and provenance-rich manifest/report assets without retraining models or creating final shortlist buckets. See `Kinase_Causal_QSAR/README.md` for full Step-13D inputs, score-component policies, outputs, and reproducibility notes.
