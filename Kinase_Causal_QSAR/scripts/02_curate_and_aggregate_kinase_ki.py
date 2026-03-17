#!/usr/bin/env python3
"""Curate and aggregate Script-01 ChEMBL human kinase Ki output.

This script is a strict continuation of Script-01 and expects
`data/raw/chembl_human_kinase_ki_raw.csv` as the primary input.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "RDKit is required for Script-02. Install with `pip install rdkit-pypi`."
    ) from exc


SCRIPT_NAME = "02_curate_and_aggregate_kinase_ki"
RANDOM_SEED = 2025


@dataclass
class AppConfig:
    """Runtime configuration for Script-02."""

    input_csv_path: Path
    curated_long_csv_path: Path
    duplicate_summary_csv_path: Path
    kinase_counts_csv_path: Path
    curation_report_json_path: Path
    configs_used_dir: Path
    logs_dir: Path
    intermediate_standardized_csv_path: Path
    figure_palette: dict[str, str]
    min_heavy_atoms: int = 3

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        script_cfg = raw.get("script_02", {}) if isinstance(raw, dict) else {}

        def resolve(path_like: str | Path) -> Path:
            path = Path(path_like)
            return path if path.is_absolute() else project_root / path

        palette = raw.get(
            "figure_palette",
            {
                "blue": "#386CB0",
                "orange": "#F39C12",
                "green": "#2CA25F",
                "red": "#E74C3C",
                "purple": "#756BB1",
                "gray": "#7F8C8D",
            },
        )

        return AppConfig(
            input_csv_path=resolve(
                script_cfg.get(
                    "input_csv_path", "data/raw/chembl_human_kinase_ki_raw.csv"
                )
            ),
            curated_long_csv_path=resolve(
                script_cfg.get(
                    "curated_long_csv_path",
                    "data/interim/chembl_human_kinase_ki_curated_long.csv",
                )
            ),
            duplicate_summary_csv_path=resolve(
                script_cfg.get(
                    "duplicate_summary_csv_path",
                    "data/interim/chembl_human_kinase_ki_duplicate_summary.csv",
                )
            ),
            kinase_counts_csv_path=resolve(
                script_cfg.get(
                    "kinase_counts_csv_path", "data/interim/kinase_record_counts.csv"
                )
            ),
            curation_report_json_path=resolve(
                script_cfg.get(
                    "curation_report_json_path", "reports/02_curation_report.json"
                )
            ),
            configs_used_dir=resolve(script_cfg.get("configs_used_dir", "configs_used")),
            logs_dir=resolve(script_cfg.get("logs_dir", "logs")),
            intermediate_standardized_csv_path=resolve(
                script_cfg.get(
                    "intermediate_standardized_csv_path",
                    "data/interim/chembl_human_kinase_ki_standardized_records.csv",
                )
            ),
            figure_palette=palette,
            min_heavy_atoms=int(script_cfg.get("min_heavy_atoms", 3)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Curate and aggregate Script-01 human kinase Ki output to a deterministic "
            "publication-ready interim dataset."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file (default: config.yaml).",
    )
    return parser.parse_args()


def load_config(config_path: Path, project_root: Path) -> tuple[AppConfig, Path, dict[str, Any]]:
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Config YAML must contain a top-level mapping/object.")

    return AppConfig.from_dict(raw, project_root), config_path, raw


def ensure_output_dirs(cfg: AppConfig) -> None:
    for path in [
        cfg.curated_long_csv_path,
        cfg.duplicate_summary_csv_path,
        cfg.kinase_counts_csv_path,
        cfg.curation_report_json_path,
        cfg.intermediate_standardized_csv_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    cfg.configs_used_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)


def setup_logging(logs_dir: Path) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{SCRIPT_NAME}_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    return log_file, timestamp


def save_config_snapshot(
    cfg: AppConfig,
    loaded_config_path: Path,
    loaded_raw_config: dict[str, Any],
) -> Path:
    snapshot_path = cfg.configs_used_dir / f"{SCRIPT_NAME}_config.yaml"
    payload = {
        "script": SCRIPT_NAME,
        "random_seed": RANDOM_SEED,
        "loaded_config_path": str(loaded_config_path),
        "resolved_paths": {
            "input_csv_path": str(cfg.input_csv_path),
            "curated_long_csv_path": str(cfg.curated_long_csv_path),
            "duplicate_summary_csv_path": str(cfg.duplicate_summary_csv_path),
            "kinase_counts_csv_path": str(cfg.kinase_counts_csv_path),
            "curation_report_json_path": str(cfg.curation_report_json_path),
            "intermediate_standardized_csv_path": str(
                cfg.intermediate_standardized_csv_path
            ),
            "logs_dir": str(cfg.logs_dir),
            "configs_used_dir": str(cfg.configs_used_dir),
        },
        "parameters": {
            "min_heavy_atoms": cfg.min_heavy_atoms,
            "figure_palette": cfg.figure_palette,
        },
        "source_config": loaded_raw_config,
    }

    with snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    return snapshot_path


def validate_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "Expected Script-01 output is missing: "
            f"{path}. Run scripts/01_extract_human_kinase_ki.py first."
        )


def validate_required_columns(df: pd.DataFrame) -> None:
    required = {
        "compound_chembl_id",
        "canonical_smiles",
        "target_chembl_id",
        "target_name",
        "standard_type",
        "standard_units",
        "standard_value",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required Script-01 columns: {missing}")


def standardize_smiles(smiles: str) -> tuple[str | None, str | None]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "missing_smiles"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"

    try:
        cleaned = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(cleaned)
        if parent is None:
            return None, "no_parent_fragment"

        if parent.GetNumAtoms() == 0:
            return None, "empty_parent_fragment"

        canonical = Chem.MolToSmiles(parent, canonical=True)
        if "." in canonical:
            return None, "mixture_after_standardization"

        return canonical, None
    except Exception:  # noqa: BLE001
        return None, "standardization_failed"


def curate_dataset(df: pd.DataFrame, cfg: AppConfig, logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, int]]:
    counters: dict[str, int] = {"rows_input": int(len(df))}

    working = df.copy()
    working["original_smiles"] = working["canonical_smiles"]

    working["standard_type"] = working["standard_type"].astype(str)
    working["standard_units"] = working["standard_units"].astype(str)

    before_ki_filter = len(working)
    working = working[(working["standard_type"] == "Ki") & (working["standard_units"] == "nM")]
    counters["removed_non_ki_nm"] = int(before_ki_filter - len(working))

    working["ki_nM"] = pd.to_numeric(working["standard_value"], errors="coerce")
    before_ki_valid = len(working)
    working = working[working["ki_nM"].notna() & (working["ki_nM"] > 0)]
    counters["removed_invalid_or_nonpositive_ki"] = int(before_ki_valid - len(working))

    results = working["original_smiles"].map(standardize_smiles)
    working["standardized_smiles"] = [res[0] for res in results]
    working["smiles_failure_reason"] = [res[1] for res in results]

    for reason, count in (
        working["smiles_failure_reason"].fillna("ok").value_counts().to_dict().items()
    ):
        counters[f"smiles_{reason}"] = int(count)

    working = working[working["standardized_smiles"].notna()].copy()

    standardized_mol = working["standardized_smiles"].map(Chem.MolFromSmiles)
    working["heavy_atom_count"] = standardized_mol.map(
        lambda m: m.GetNumHeavyAtoms() if m is not None else 0
    )
    before_heavy = len(working)
    working = working[working["heavy_atom_count"] >= cfg.min_heavy_atoms].copy()
    counters["removed_low_heavy_atom_count"] = int(before_heavy - len(working))

    working["molecular_weight"] = standardized_mol.map(
        lambda m: round(Descriptors.MolWt(m), 3) if m is not None else math.nan
    )
    working["pKi"] = 9.0 - working["ki_nM"].map(math.log10)

    counters["rows_after_curation"] = int(len(working))

    logger.info("Curation counters: %s", counters)
    return working, counters


def aggregate_duplicate_measurements(curated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_keys = ["standardized_smiles", "target_chembl_id", "target_name"]

    agg_map: dict[str, Any] = {
        "ki_nM": ["median", "count"],
        "pKi": ["median", "std"],
        "compound_chembl_id": pd.Series.nunique,
    }

    if "assay_chembl_id" in curated.columns:
        agg_map["assay_chembl_id"] = pd.Series.nunique

    doc_column = None
    for candidate in ["doc_id", "doc_doi", "doc_pubmed_id"]:
        if candidate in curated.columns:
            doc_column = candidate
            agg_map[candidate] = pd.Series.nunique
            break

    grouped = curated.groupby(group_keys, dropna=False).agg(agg_map)
    grouped.columns = [
        "median_ki_nM",
        "n_source_records",
        "median_pKi",
        "pKi_std",
        "n_unique_compound_ids",
        *(
            ["n_unique_assays"] if "assay_chembl_id" in curated.columns else []
        ),
        *(["n_unique_documents"] if doc_column is not None else []),
    ]

    aggregated = grouped.reset_index().sort_values(
        by=["target_chembl_id", "standardized_smiles"], kind="mergesort"
    )
    aggregated["n_source_records"] = aggregated["n_source_records"].astype(int)
    aggregated["is_duplicate_measurement"] = aggregated["n_source_records"] > 1

    duplicate_summary = aggregated[aggregated["is_duplicate_measurement"]].copy()
    duplicate_summary = duplicate_summary.sort_values(
        by=["n_source_records", "target_chembl_id"], ascending=[False, True], kind="mergesort"
    )

    return aggregated, duplicate_summary


def build_kinase_counts(aggregated: pd.DataFrame) -> pd.DataFrame:
    counts = (
        aggregated.groupby(["target_chembl_id", "target_name"], dropna=False)
        .agg(
            n_curated_records=("standardized_smiles", "count"),
            n_unique_compounds=("standardized_smiles", "nunique"),
            median_pKi_across_compounds=("median_pKi", "median"),
            median_ki_nM_across_compounds=("median_ki_nM", "median"),
        )
        .reset_index()
        .sort_values(by="n_curated_records", ascending=False, kind="mergesort")
    )
    return counts


def write_report(
    cfg: AppConfig,
    report: dict[str, Any],
) -> None:
    with cfg.curation_report_json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    args = parse_args()

    try:
        cfg, loaded_cfg_path, loaded_raw_cfg = load_config(args.config, project_root)
        ensure_output_dirs(cfg)
        log_file, timestamp = setup_logging(cfg.logs_dir)
        logger = logging.getLogger(__name__)

        logger.info("Starting %s", SCRIPT_NAME)
        logger.info("Fixed random seed (for traceability): %d", RANDOM_SEED)
        logger.info("Run timestamp: %s", timestamp)

        config_snapshot = save_config_snapshot(cfg, loaded_cfg_path, loaded_raw_cfg)
        logger.info("Saved config snapshot: %s", config_snapshot)

        validate_input_file(cfg.input_csv_path)
        logger.info("Reading Script-01 output: %s", cfg.input_csv_path)
        raw_df = pd.read_csv(cfg.input_csv_path)
        validate_required_columns(raw_df)

        curated_records, counters = curate_dataset(raw_df, cfg, logger)
        curated_records = curated_records.sort_values(
            by=["target_chembl_id", "standardized_smiles", "ki_nM"], kind="mergesort"
        )
        curated_records.to_csv(cfg.intermediate_standardized_csv_path, index=False)
        logger.info(
            "Saved standardized intermediate records: %s",
            cfg.intermediate_standardized_csv_path,
        )

        aggregated, duplicate_summary = aggregate_duplicate_measurements(curated_records)
        kinase_counts = build_kinase_counts(aggregated)

        aggregated.to_csv(cfg.curated_long_csv_path, index=False)
        duplicate_summary.to_csv(cfg.duplicate_summary_csv_path, index=False)
        kinase_counts.to_csv(cfg.kinase_counts_csv_path, index=False)

        report = {
            "script": SCRIPT_NAME,
            "timestamp": timestamp,
            "random_seed": RANDOM_SEED,
            "inputs": {
                "script_01_raw_csv": str(cfg.input_csv_path),
            },
            "outputs": {
                "curated_long_csv": str(cfg.curated_long_csv_path),
                "duplicate_summary_csv": str(cfg.duplicate_summary_csv_path),
                "kinase_counts_csv": str(cfg.kinase_counts_csv_path),
                "intermediate_standardized_csv": str(cfg.intermediate_standardized_csv_path),
                "report_json": str(cfg.curation_report_json_path),
                "config_snapshot_yaml": str(config_snapshot),
                "log_file": str(log_file),
            },
            "record_counts": {
                "rows_input": int(len(raw_df)),
                "rows_after_curation": int(len(curated_records)),
                "rows_after_aggregation": int(len(aggregated)),
                "rows_duplicate_groups": int(len(duplicate_summary)),
                "n_unique_targets_after_aggregation": int(
                    aggregated["target_chembl_id"].nunique(dropna=True)
                ),
                "n_unique_compounds_after_aggregation": int(
                    aggregated["standardized_smiles"].nunique(dropna=True)
                ),
            },
            "filtering_decisions": counters,
            "notes": {
                "continuity": "Script-02 strictly consumes Script-01 raw CSV output.",
                "future_steps": (
                    "No kinase matrix construction, selectivity labeling, or modeling is "
                    "performed in this script."
                ),
            },
        }

        write_report(cfg, report)
        logger.info("Saved curation report: %s", cfg.curation_report_json_path)
        logger.info("Script-02 completed successfully.")
        return 0

    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logging.getLogger(__name__).exception("Script-02 failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
