#!/usr/bin/env python3
"""Annotate Script-03 kinase panel outputs with causal-learning environments.

This script is a strict continuation of Script-03. It creates deterministic
compound-, kinase-, source-, and pair-level environment annotations plus
activity-cliff diagnostics required for downstream causal learning.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, Lipinski, MolSurf, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "RDKit is required for Script-04. Install with `pip install rdkit-pypi`."
    ) from exc


SCRIPT_NAME = "04_annotate_environments_for_causal_learning"
RANDOM_SEED = 2025
REQUIRED_SCRIPT_04_KEYS = {
    "input_long_path",
    "input_matrix_path",
    "input_mask_path",
    "input_kinase_summary_path",
    "output_annotated_long_path",
    "output_compound_env_path",
    "output_kinase_env_path",
    "output_source_env_path",
    "output_pair_env_path",
    "output_activity_cliff_path",
    "output_env_report_path",
    "compute_murcko_scaffolds",
    "compute_generic_murcko_scaffolds",
    "compute_rdkit_descriptors",
    "compute_scaffold_frequency_bins",
    "annotate_kinase_family_from_chembl",
    "kinase_family_fallback_to_name_matching",
    "compute_source_environments",
    "compute_document_frequency_bins",
    "compute_pairwise_activity_cliffs",
    "activity_cliff_similarity_metric",
    "morgan_radius",
    "morgan_nbits",
    "activity_cliff_similarity_threshold",
    "activity_cliff_delta_pki_threshold",
    "save_similarity_diagnostics",
    "max_pairs_for_cliff_analysis_per_kinase",
    "save_config_snapshot",
}
REQUIRED_LONG_COLUMNS = {"compound_id", "standardized_smiles", "target_chembl_id", "target_name", "pKi"}
OPTIONAL_LONG_ALIASES: dict[str, tuple[str, ...]] = {
    "compound_id": (
        "compound_id",
        "compound_chembl_id",
        "molecule_chembl_id",
        "parent_molecule_chembl_id",
    ),
    "standardized_smiles": (
        "standardized_smiles",
        "canonical_smiles_standardized",
        "canonical_smiles",
    ),
    "target_chembl_id": ("target_chembl_id",),
    "target_name": ("target_name", "pref_name"),
    "pKi": ("pKi", "median_pKi", "aggregated_pKi"),
    "assay_chembl_id": ("assay_chembl_id", "assay_id", "assay_chemblid"),
    "doc_id": ("doc_id", "document_chembl_id", "document_id", "doc_chembl_id"),
    "source_id": ("source_id", "src_id", "data_source_id"),
    "source_description": (
        "source_description",
        "src_description",
        "data_source_description",
        "source_name",
    ),
    "unique_assay_count": (
        "unique_assay_count",
        "n_unique_assays",
        "number_of_supporting_assays",
    ),
    "unique_document_count": (
        "unique_document_count",
        "n_unique_documents",
        "number_of_supporting_documents",
    ),
    "source_record_count": (
        "source_record_count",
        "n_source_records",
        "supporting_record_count",
    ),
    "protein_class_desc": ("protein_class_desc", "protein_classification", "target_class"),
    "protein_family": ("protein_family", "kinase_family", "family_name"),
    "protein_subfamily": ("protein_subfamily", "kinase_subfamily", "subfamily_name"),
}
DEFAULT_SOURCE_PLACEHOLDER = "UNAVAILABLE"


def count_heteroatoms(mol: Chem.Mol) -> int:
    """Return the RDKit heteroatom count across supported RDKit versions."""

    heteroatom_fn = getattr(Lipinski, "NumHeteroatoms", None)
    if heteroatom_fn is None:
        heteroatom_fn = getattr(Lipinski, "Heteroatoms", None)
    if heteroatom_fn is None:  # pragma: no cover - defensive compatibility guard
        raise AttributeError(
            "RDKit Lipinski module is missing both NumHeteroatoms and Heteroatoms."
        )
    return int(heteroatom_fn(mol))


@dataclass
class AppConfig:
    input_long_path: Path
    input_matrix_path: Path
    input_mask_path: Path
    input_kinase_summary_path: Path
    output_annotated_long_path: Path
    output_compound_env_path: Path
    output_kinase_env_path: Path
    output_source_env_path: Path
    output_pair_env_path: Path
    output_activity_cliff_path: Path
    output_env_report_path: Path
    compute_murcko_scaffolds: bool
    compute_generic_murcko_scaffolds: bool
    compute_rdkit_descriptors: bool
    compute_scaffold_frequency_bins: bool
    annotate_kinase_family_from_chembl: bool
    kinase_family_fallback_to_name_matching: bool
    compute_source_environments: bool
    compute_document_frequency_bins: bool
    compute_pairwise_activity_cliffs: bool
    activity_cliff_similarity_metric: str
    morgan_radius: int
    morgan_nbits: int
    activity_cliff_similarity_threshold: float
    activity_cliff_delta_pki_threshold: float
    save_similarity_diagnostics: bool
    max_pairs_for_cliff_analysis_per_kinase: int
    save_config_snapshot: bool
    configs_used_dir: Path
    logs_dir: Path
    supplemental_metadata_path: Path | None

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        if not isinstance(raw, dict):
            raise ValueError("Config YAML must contain a top-level mapping/object.")

        script_cfg = raw.get("script_04")
        if not isinstance(script_cfg, dict):
            raise ValueError("Missing required `script_04` section in config.yaml.")

        missing_keys = sorted(REQUIRED_SCRIPT_04_KEYS.difference(script_cfg))
        if missing_keys:
            raise ValueError(
                "Missing required script_04 config values: " + ", ".join(missing_keys)
            )

        def resolve(path_like: str | Path | None) -> Path | None:
            if path_like in (None, ""):
                return None
            path = Path(path_like)
            return path if path.is_absolute() else project_root / path

        def parse_bool(value: Any, key: str) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes"}:
                    return True
                if lowered in {"0", "false", "no"}:
                    return False
            raise ValueError(f"script_04.{key} must be a boolean; got {value!r}.")

        def parse_positive_int(key: str, allow_zero: bool = False) -> int:
            value = script_cfg.get(key)
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"script_04.{key} must be an integer; got {value!r}.") from exc
            minimum = 0 if allow_zero else 1
            if parsed < minimum:
                comparator = ">=" if allow_zero else ">"
                raise ValueError(f"script_04.{key} must be {comparator} {minimum}; got {parsed}.")
            return parsed

        def parse_float_in_range(key: str, minimum: float, maximum: float | None = None) -> float:
            value = script_cfg.get(key)
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"script_04.{key} must be numeric; got {value!r}.") from exc
            if parsed < minimum or (maximum is not None and parsed > maximum):
                if maximum is None:
                    raise ValueError(f"script_04.{key} must be >= {minimum}; got {parsed}.")
                raise ValueError(
                    f"script_04.{key} must be between {minimum} and {maximum}; got {parsed}."
                )
            return parsed

        metric = str(script_cfg["activity_cliff_similarity_metric"]).strip().lower()
        if metric != "tanimoto_morgan":
            raise ValueError(
                "script_04.activity_cliff_similarity_metric currently supports only "
                "'tanimoto_morgan'."
            )

        return AppConfig(
            input_long_path=resolve(script_cfg["input_long_path"]),
            input_matrix_path=resolve(script_cfg["input_matrix_path"]),
            input_mask_path=resolve(script_cfg["input_mask_path"]),
            input_kinase_summary_path=resolve(script_cfg["input_kinase_summary_path"]),
            output_annotated_long_path=resolve(script_cfg["output_annotated_long_path"]),
            output_compound_env_path=resolve(script_cfg["output_compound_env_path"]),
            output_kinase_env_path=resolve(script_cfg["output_kinase_env_path"]),
            output_source_env_path=resolve(script_cfg["output_source_env_path"]),
            output_pair_env_path=resolve(script_cfg["output_pair_env_path"]),
            output_activity_cliff_path=resolve(script_cfg["output_activity_cliff_path"]),
            output_env_report_path=resolve(script_cfg["output_env_report_path"]),
            compute_murcko_scaffolds=parse_bool(
                script_cfg["compute_murcko_scaffolds"], "compute_murcko_scaffolds"
            ),
            compute_generic_murcko_scaffolds=parse_bool(
                script_cfg["compute_generic_murcko_scaffolds"], "compute_generic_murcko_scaffolds"
            ),
            compute_rdkit_descriptors=parse_bool(
                script_cfg["compute_rdkit_descriptors"], "compute_rdkit_descriptors"
            ),
            compute_scaffold_frequency_bins=parse_bool(
                script_cfg["compute_scaffold_frequency_bins"], "compute_scaffold_frequency_bins"
            ),
            annotate_kinase_family_from_chembl=parse_bool(
                script_cfg["annotate_kinase_family_from_chembl"], "annotate_kinase_family_from_chembl"
            ),
            kinase_family_fallback_to_name_matching=parse_bool(
                script_cfg["kinase_family_fallback_to_name_matching"],
                "kinase_family_fallback_to_name_matching",
            ),
            compute_source_environments=parse_bool(
                script_cfg["compute_source_environments"], "compute_source_environments"
            ),
            compute_document_frequency_bins=parse_bool(
                script_cfg["compute_document_frequency_bins"], "compute_document_frequency_bins"
            ),
            compute_pairwise_activity_cliffs=parse_bool(
                script_cfg["compute_pairwise_activity_cliffs"], "compute_pairwise_activity_cliffs"
            ),
            activity_cliff_similarity_metric=metric,
            morgan_radius=parse_positive_int("morgan_radius"),
            morgan_nbits=parse_positive_int("morgan_nbits"),
            activity_cliff_similarity_threshold=parse_float_in_range(
                "activity_cliff_similarity_threshold", 0.0, 1.0
            ),
            activity_cliff_delta_pki_threshold=parse_float_in_range(
                "activity_cliff_delta_pki_threshold", 0.0
            ),
            save_similarity_diagnostics=parse_bool(
                script_cfg["save_similarity_diagnostics"], "save_similarity_diagnostics"
            ),
            max_pairs_for_cliff_analysis_per_kinase=parse_positive_int(
                "max_pairs_for_cliff_analysis_per_kinase", allow_zero=True
            ),
            save_config_snapshot=parse_bool(script_cfg["save_config_snapshot"], "save_config_snapshot"),
            configs_used_dir=resolve(raw.get("script_02", {}).get("configs_used_dir", "configs_used")),
            logs_dir=resolve(raw.get("script_02", {}).get("logs_dir", "logs")),
            supplemental_metadata_path=resolve(script_cfg.get("supplemental_metadata_path")),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate Script-03 kinase panel outputs with compound, kinase, source, "
            "pair, and activity-cliff environment metadata."
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
    return AppConfig.from_dict(raw, project_root), config_path, raw


def ensure_output_dirs(cfg: AppConfig) -> None:
    for path in [
        cfg.output_annotated_long_path,
        cfg.output_compound_env_path,
        cfg.output_kinase_env_path,
        cfg.output_source_env_path,
        cfg.output_pair_env_path,
        cfg.output_activity_cliff_path,
        cfg.output_env_report_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.configs_used_dir.mkdir(parents=True, exist_ok=True)


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


def save_config_snapshot(cfg: AppConfig, loaded_config_path: Path, loaded_raw_config: dict[str, Any]) -> Path:
    snapshot_path = cfg.configs_used_dir / f"{SCRIPT_NAME}_config.yaml"
    payload = {
        "script": SCRIPT_NAME,
        "random_seed": RANDOM_SEED,
        "loaded_config_path": str(loaded_config_path),
        "resolved_paths": {
            "input_long_path": str(cfg.input_long_path),
            "input_matrix_path": str(cfg.input_matrix_path),
            "input_mask_path": str(cfg.input_mask_path),
            "input_kinase_summary_path": str(cfg.input_kinase_summary_path),
            "output_annotated_long_path": str(cfg.output_annotated_long_path),
            "output_compound_env_path": str(cfg.output_compound_env_path),
            "output_kinase_env_path": str(cfg.output_kinase_env_path),
            "output_source_env_path": str(cfg.output_source_env_path),
            "output_pair_env_path": str(cfg.output_pair_env_path),
            "output_activity_cliff_path": str(cfg.output_activity_cliff_path),
            "output_env_report_path": str(cfg.output_env_report_path),
            "configs_used_dir": str(cfg.configs_used_dir),
            "logs_dir": str(cfg.logs_dir),
            "supplemental_metadata_path": str(cfg.supplemental_metadata_path)
            if cfg.supplemental_metadata_path
            else None,
        },
        "parameters": {
            key: getattr(cfg, key)
            for key in [
                "compute_murcko_scaffolds",
                "compute_generic_murcko_scaffolds",
                "compute_rdkit_descriptors",
                "compute_scaffold_frequency_bins",
                "annotate_kinase_family_from_chembl",
                "kinase_family_fallback_to_name_matching",
                "compute_source_environments",
                "compute_document_frequency_bins",
                "compute_pairwise_activity_cliffs",
                "activity_cliff_similarity_metric",
                "morgan_radius",
                "morgan_nbits",
                "activity_cliff_similarity_threshold",
                "activity_cliff_delta_pki_threshold",
                "save_similarity_diagnostics",
                "max_pairs_for_cliff_analysis_per_kinase",
                "save_config_snapshot",
            ]
        },
        "source_config": loaded_raw_config,
    }
    with snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return snapshot_path


def validate_input_paths(cfg: AppConfig) -> None:
    required_paths = {
        "Script-03 panel long CSV": cfg.input_long_path,
        "Script-03 pKi matrix CSV": cfg.input_matrix_path,
        "Script-03 observation mask CSV": cfg.input_mask_path,
        "Script-03 kinase summary CSV": cfg.input_kinase_summary_path,
    }
    missing = [f"{label}: {path}" for label, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Script-04 cannot start because required Script-03 inputs are missing:\n- "
            + "\n- ".join(missing)
        )
    if cfg.supplemental_metadata_path and not cfg.supplemental_metadata_path.exists():
        raise FileNotFoundError(
            "Configured script_04.supplemental_metadata_path does not exist: "
            f"{cfg.supplemental_metadata_path}"
        )


def read_csv_with_logging(path: Path, label: str) -> pd.DataFrame:
    logging.info("Loading %s from %s", label, path)
    return pd.read_csv(path)


def resolve_column_map(df: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in OPTIONAL_LONG_ALIASES.items():
        match = next((candidate for candidate in aliases if candidate in df.columns), None)
        if match is not None:
            resolved[canonical] = match
        elif canonical in REQUIRED_LONG_COLUMNS:
            missing.append(canonical)
    if missing:
        raise ValueError(
            "Missing required Script-03 long-format columns after alias normalization: "
            f"{missing}. Available columns: {sorted(df.columns.tolist())}"
        )
    return resolved


def standardize_internal_columns(long_df: pd.DataFrame, supplemental_df: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    column_map = resolve_column_map(long_df)
    standardized = long_df.rename(columns={v: k for k, v in column_map.items()}).copy()

    for required in REQUIRED_LONG_COLUMNS:
        standardized[required] = standardized[required].astype("string").str.strip() if required != "pKi" else standardized[required]

    for text_required in ["compound_id", "standardized_smiles", "target_chembl_id", "target_name"]:
        if standardized[text_required].isna().any() or (standardized[text_required] == "").any():
            raise ValueError(f"Required identifier column `{text_required}` contains missing/blank values.")

    standardized["pKi"] = pd.to_numeric(standardized["pKi"], errors="coerce")
    if standardized["pKi"].isna().any():
        raise ValueError("Column `pKi` must be numeric and non-null for Script-04.")

    for text_column in [
        "compound_id",
        "standardized_smiles",
        "target_chembl_id",
        "target_name",
        "assay_chembl_id",
        "doc_id",
        "source_id",
        "source_description",
        "protein_class_desc",
        "protein_family",
        "protein_subfamily",
    ]:
        if text_column not in standardized.columns:
            standardized[text_column] = pd.Series(pd.NA, index=standardized.index, dtype="string")
        standardized[text_column] = standardized[text_column].astype("string").str.strip()

    for numeric_column in ["unique_assay_count", "unique_document_count", "source_record_count"]:
        if numeric_column not in standardized.columns:
            standardized[numeric_column] = np.nan
        standardized[numeric_column] = pd.to_numeric(standardized[numeric_column], errors="coerce")

    available_source_columns = standardized[["assay_chembl_id", "doc_id", "source_id", "source_description"]].notna().any()
    if not bool(available_source_columns.any()) and supplemental_df is not None:
        logging.info("Attempting supplemental metadata merge for source/document fields.")
        supp_column_map = resolve_column_map(supplemental_df)
        supp = supplemental_df.rename(columns={v: k for k, v in supp_column_map.items()}).copy()
        join_keys = [key for key in ["compound_id", "target_chembl_id"] if key in supp.columns]
        if len(join_keys) < 2:
            raise ValueError(
                "Supplemental metadata must contain at least `compound_id` and `target_chembl_id` "
                "(or supported aliases) for deterministic merging."
            )
        merge_columns = join_keys + [
            col
            for col in [
                "assay_chembl_id",
                "doc_id",
                "source_id",
                "source_description",
                "protein_class_desc",
                "protein_family",
                "protein_subfamily",
            ]
            if col in supp.columns
        ]
        supp = supp[merge_columns].drop_duplicates().copy()
        standardized = standardized.merge(
            supp,
            on=join_keys,
            how="left",
            suffixes=("", "_supplemental"),
            sort=False,
        )
        for col in [
            "assay_chembl_id",
            "doc_id",
            "source_id",
            "source_description",
            "protein_class_desc",
            "protein_family",
            "protein_subfamily",
        ]:
            supplemental_col = f"{col}_supplemental"
            if supplemental_col in standardized.columns:
                standardized[col] = standardized[col].fillna(standardized[supplemental_col])
                standardized = standardized.drop(columns=[supplemental_col])

    if not bool(standardized[["assay_chembl_id", "doc_id", "source_id", "source_description"]].notna().any().any()):
        warnings.append(
            "Source/document/assay metadata were not available in the Script-03 long-format file "
            "and no usable supplemental metadata were merged. Source-level annotations are retained "
            "with explicit UNAVAILABLE placeholders."
        )

    return standardized, warnings


def validate_panel_alignment(long_df: pd.DataFrame, matrix_df: pd.DataFrame, mask_df: pd.DataFrame, kinase_summary_df: pd.DataFrame) -> None:
    matrix = matrix_df.copy()
    mask = mask_df.copy()
    row_label = matrix.columns[0]
    matrix = matrix.rename(columns={row_label: "standardized_smiles"}) if row_label != "standardized_smiles" else matrix
    row_label_mask = mask.columns[0]
    mask = mask.rename(columns={row_label_mask: "standardized_smiles"}) if row_label_mask != "standardized_smiles" else mask

    if matrix.shape != mask.shape:
        raise ValueError(
            f"Matrix and observation mask must have identical dimensions; got {matrix.shape} vs {mask.shape}."
        )

    matrix_smiles = set(matrix["standardized_smiles"].astype(str))
    mask_smiles = set(mask["standardized_smiles"].astype(str))
    long_smiles = set(long_df["standardized_smiles"].astype(str))
    if matrix_smiles != mask_smiles:
        raise ValueError("Matrix and observation mask contain different compound row identifiers.")
    if not long_smiles.issubset(matrix_smiles):
        raise ValueError("Long-format compounds are not fully represented in the Script-03 pKi matrix.")

    target_columns = set(matrix.columns[1:])
    long_targets = set(long_df["target_chembl_id"].astype(str))
    if not long_targets.issubset(target_columns):
        missing_targets = sorted(long_targets.difference(target_columns))
        raise ValueError(
            "Long-format kinases are not fully represented in the Script-03 pKi matrix columns: "
            f"{missing_targets[:10]}"
        )

    kinase_summary_required = {"target_chembl_id", "target_name"}
    if not kinase_summary_required.issubset(kinase_summary_df.columns):
        raise ValueError(
            "Kinase summary file is missing required columns: "
            f"{sorted(kinase_summary_required.difference(kinase_summary_df.columns))}"
        )


def safe_scaffold(smiles: str, generic: bool = False) -> str | pd.NA:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.NA
    try:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(mol) if generic else MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return pd.NA
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:  # pragma: no cover - defensive RDKit guard
        return pd.NA


def count_aromatic_rings(mol: Chem.Mol) -> int:
    ring_info = mol.GetRingInfo()
    aromatic = 0
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic += 1
    return aromatic


def scaffold_frequency_bin(count: int) -> str:
    if count <= 1:
        return "singleton"
    if count <= 5:
        return "rare"
    if count <= 20:
        return "common"
    return "frequent"


def frequency_bin(count: int) -> str:
    if count <= 1:
        return "singleton"
    if count <= 3:
        return "rare"
    if count <= 10:
        return "common"
    return "frequent"


def annotate_compounds(df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    logging.info("Annotating compound-level environments.")
    warnings: list[str] = []
    compounds = (
        df[["compound_id", "standardized_smiles"]]
        .drop_duplicates()
        .sort_values(["compound_id", "standardized_smiles"], kind="mergesort")
        .reset_index(drop=True)
    )

    compound_records: list[dict[str, Any]] = []
    invalid_smiles_count = 0

    for row in compounds.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.standardized_smiles)
        if mol is None:
            invalid_smiles_count += 1
            record = {
                "compound_id": row.compound_id,
                "standardized_smiles": row.standardized_smiles,
                "rdkit_parse_success": False,
                "murcko_scaffold": pd.NA,
                "generic_murcko_scaffold": pd.NA,
                "heavy_atom_count": np.nan,
                "ring_count": np.nan,
                "aromatic_ring_count": np.nan,
                "molecular_weight": np.nan,
                "clogp": np.nan,
                "hbond_donor_count": np.nan,
                "hbond_acceptor_count": np.nan,
                "tpsa": np.nan,
                "rotatable_bond_count": np.nan,
                "formal_charge": np.nan,
                "fraction_csp3": np.nan,
                "heteroatom_count": np.nan,
                "num_valence_electrons": np.nan,
            }
            compound_records.append(record)
            continue

        record = {
            "compound_id": row.compound_id,
            "standardized_smiles": row.standardized_smiles,
            "rdkit_parse_success": True,
            "murcko_scaffold": safe_scaffold(row.standardized_smiles, generic=False)
            if cfg.compute_murcko_scaffolds
            else pd.NA,
            "generic_murcko_scaffold": safe_scaffold(row.standardized_smiles, generic=True)
            if cfg.compute_generic_murcko_scaffolds
            else pd.NA,
        }
        if cfg.compute_rdkit_descriptors:
            record.update(
                {
                    "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
                    "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
                    "aromatic_ring_count": int(count_aromatic_rings(mol)),
                    "molecular_weight": float(Descriptors.MolWt(mol)),
                    "clogp": float(Descriptors.MolLogP(mol)),
                    "hbond_donor_count": int(Lipinski.NumHDonors(mol)),
                    "hbond_acceptor_count": int(Lipinski.NumHAcceptors(mol)),
                    "tpsa": float(MolSurf.TPSA(mol)),
                    "rotatable_bond_count": int(Lipinski.NumRotatableBonds(mol)),
                    "formal_charge": int(Chem.GetFormalCharge(mol)),
                    "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
                    "heteroatom_count": count_heteroatoms(mol),
                    "num_valence_electrons": float(Descriptors.NumValenceElectrons(mol)),
                }
            )
        compound_records.append(record)

    compound_env = pd.DataFrame(compound_records)

    if cfg.compute_murcko_scaffolds:
        scaffold_counts = (
            compound_env["murcko_scaffold"].fillna("NO_SCAFFOLD").value_counts(dropna=False).to_dict()
        )
        compound_env["scaffold_frequency"] = compound_env["murcko_scaffold"].fillna("NO_SCAFFOLD").map(scaffold_counts)
    else:
        compound_env["scaffold_frequency"] = np.nan

    if cfg.compute_generic_murcko_scaffolds:
        generic_counts = (
            compound_env["generic_murcko_scaffold"].fillna("NO_GENERIC_SCAFFOLD").value_counts(dropna=False).to_dict()
        )
        compound_env["generic_scaffold_frequency"] = compound_env["generic_murcko_scaffold"].fillna("NO_GENERIC_SCAFFOLD").map(generic_counts)
    else:
        compound_env["generic_scaffold_frequency"] = np.nan

    compound_env["scaffold_frequency_bin"] = (
        compound_env["scaffold_frequency"].fillna(0).astype(int).map(scaffold_frequency_bin)
        if cfg.compute_scaffold_frequency_bins
        else pd.Series(pd.NA, index=compound_env.index, dtype="string")
    )
    compound_env["generic_scaffold_frequency_bin"] = (
        compound_env["generic_scaffold_frequency"].fillna(0).astype(int).map(scaffold_frequency_bin)
        if cfg.compute_scaffold_frequency_bins
        else pd.Series(pd.NA, index=compound_env.index, dtype="string")
    )
    compound_env = compound_env.sort_values(["compound_id", "standardized_smiles"], kind="mergesort")

    if invalid_smiles_count > 0:
        warnings.append(
            f"RDKit could not parse {invalid_smiles_count} standardized SMILES strings; descriptor fields were left missing for those compounds."
        )

    summary = {
        "n_compounds": int(len(compound_env)),
        "n_invalid_smiles": int(invalid_smiles_count),
        "n_unique_scaffolds": int(compound_env["murcko_scaffold"].dropna().nunique()),
        "n_unique_generic_scaffolds": int(compound_env["generic_murcko_scaffold"].dropna().nunique()),
        "scaffold_frequency_distribution": {
            str(key): int(value)
            for key, value in compound_env["scaffold_frequency_bin"].fillna("missing").value_counts().to_dict().items()
        },
    }
    return compound_env.reset_index(drop=True), summary, warnings


def normalize_target_name(name: str) -> str:
    return " ".join(str(name).upper().replace("-", " ").replace("/", " ").split())


def infer_kinase_family_from_name(target_name: str) -> tuple[str, str]:
    normalized = normalize_target_name(target_name)
    rules = [
        ("MAPK", "CMGC"),
        ("CDK", "CMGC"),
        ("GSK", "CMGC"),
        ("DYRK", "CMGC"),
        ("CLK", "CMGC"),
        ("ERK", "CMGC"),
        ("JNK", "CMGC"),
        ("P38", "CMGC"),
        ("AKT", "AGC"),
        ("PKC", "AGC"),
        ("PKA", "AGC"),
        ("ROCK", "AGC"),
        ("RSK", "AGC"),
        ("SGK", "AGC"),
        ("ABL", "TK"),
        ("SRC", "TK"),
        ("JAK", "TK"),
        ("EGFR", "TK"),
        ("ERBB", "TK"),
        ("FGFR", "TK"),
        ("MET", "TK"),
        ("RET", "TK"),
        ("FLT", "TK"),
        ("KIT", "TK"),
        ("DDR", "TK"),
        ("TEC", "TK"),
        ("BMX", "TK"),
        ("BTK", "TK"),
        ("SYK", "TK"),
        ("ZAP70", "TK"),
        ("PIK3", "PIKK"),
        ("MTOR", "PIKK"),
        ("ATM", "PIKK"),
        ("ATR", "PIKK"),
        ("DNA PK", "PIKK"),
        ("AUR", "CAMK"),
        ("CAMK", "CAMK"),
        ("DAPK", "CAMK"),
        ("CHEK", "CAMK"),
        ("CHK", "CAMK"),
        ("PIM", "CAMK"),
        ("NUAK", "CAMK"),
        ("MARK", "CAMK"),
        ("MELK", "CAMK"),
        ("STK", "STE"),
        ("RAF", "TKL"),
        ("BRAF", "TKL"),
        ("ALK", "TK"),
        ("IRAK", "TKL"),
        ("MLK", "TKL"),
        ("NEK", "NEK"),
        ("PLK", "OTHER"),
    ]
    for token, family in rules:
        if token in normalized:
            return family, f"name_match:{token}"
    return "Unclassified", "name_match:unclassified"


def _coalesce_duplicate_columns(df: pd.DataFrame, canonical_columns: list[str]) -> pd.DataFrame:
    """Normalize merge-created suffix variants back to canonical column names."""

    normalized = df.copy()
    for column in canonical_columns:
        variants = [candidate for candidate in [f"{column}_x", column, f"{column}_y"] if candidate in normalized.columns]
        if not variants:
            continue
        combined = normalized[variants[0]]
        for variant in variants[1:]:
            combined = combined.combine_first(normalized[variant])
        normalized[column] = combined
        drop_columns = [variant for variant in variants if variant != column]
        if drop_columns:
            normalized = normalized.drop(columns=drop_columns)
    return normalized


def annotate_kinases(df: pd.DataFrame, kinase_summary_df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    logging.info("Annotating kinase-level environments.")
    warnings: list[str] = []
    merged_summary = kinase_summary_df.copy()
    if "target_chembl_id" not in merged_summary.columns or "target_name" not in merged_summary.columns:
        raise ValueError("Kinase summary must contain target_chembl_id and target_name columns.")

    long_family = (
        df.groupby(["target_chembl_id", "target_name"], dropna=False)
        .agg(
            protein_class_desc=("protein_class_desc", lambda s: next((v for v in s.dropna() if str(v).strip()), pd.NA)),
            protein_family=("protein_family", lambda s: next((v for v in s.dropna() if str(v).strip()), pd.NA)),
            protein_subfamily=("protein_subfamily", lambda s: next((v for v in s.dropna() if str(v).strip()), pd.NA)),
            number_of_compounds_measured=("compound_id", "nunique"),
            median_pKi=("pKi", "median"),
            pKi_spread=("pKi", lambda s: float(s.max() - s.min()) if len(s) else 0.0),
            source_diversity=("source_id", lambda s: int(s.dropna().nunique())),
            document_diversity=("doc_id", lambda s: int(s.dropna().nunique())),
            assay_diversity=("assay_chembl_id", lambda s: int(s.dropna().nunique())),
            number_of_records=("target_chembl_id", "size"),
        )
        .reset_index()
    )
    kinase_env = merged_summary.merge(long_family, on=["target_chembl_id", "target_name"], how="left", sort=False)
    kinase_env = _coalesce_duplicate_columns(
        kinase_env,
        [
            "protein_class_desc",
            "protein_family",
            "protein_subfamily",
            "number_of_compounds_measured",
            "median_pKi",
            "pKi_spread",
            "source_diversity",
            "document_diversity",
            "assay_diversity",
            "number_of_records",
        ],
    )
    kinase_env["target_name_normalized"] = kinase_env["target_name"].map(normalize_target_name)

    family_sources = []
    broad_groups = []
    family_labels = []
    subfamily_labels = []
    for row in kinase_env.itertuples(index=False):
        broad = getattr(row, "protein_class_desc", pd.NA) if cfg.annotate_kinase_family_from_chembl else pd.NA
        family = getattr(row, "protein_family", pd.NA) if cfg.annotate_kinase_family_from_chembl else pd.NA
        subfamily = getattr(row, "protein_subfamily", pd.NA) if cfg.annotate_kinase_family_from_chembl else pd.NA
        source = "chembl_metadata" if cfg.annotate_kinase_family_from_chembl else "disabled"

        if (pd.isna(family) or str(family).strip() == "") and cfg.kinase_family_fallback_to_name_matching:
            family, source = infer_kinase_family_from_name(row.target_name)
            if pd.isna(broad) or str(broad).strip() == "":
                broad = family
        elif pd.isna(family) or str(family).strip() == "":
            family = "Unclassified"
            source = "missing"

        if pd.isna(subfamily) or str(subfamily).strip() == "":
            subfamily = family

        broad_groups.append(broad if not pd.isna(broad) and str(broad).strip() else family)
        family_labels.append(family)
        subfamily_labels.append(subfamily)
        family_sources.append(source)

    kinase_env["kinase_family_broad_group"] = broad_groups
    kinase_env["kinase_family"] = family_labels
    kinase_env["kinase_subfamily"] = subfamily_labels
    kinase_env["kinase_family_annotation_source"] = family_sources
    kinase_env = kinase_env.sort_values(["target_chembl_id", "target_name"], kind="mergesort").reset_index(drop=True)

    if (kinase_env["kinase_family_annotation_source"] == "chembl_metadata").sum() == 0:
        warnings.append(
            "No explicit kinase family metadata were available; kinase-family labels were derived entirely from logged name-based fallback rules."
        )

    summary = {
        "n_kinases": int(len(kinase_env)),
        "kinase_family_distribution": {
            str(key): int(value)
            for key, value in kinase_env["kinase_family"].fillna("missing").value_counts().to_dict().items()
        },
        "kinase_family_annotation_sources": {
            str(key): int(value)
            for key, value in kinase_env["kinase_family_annotation_source"].value_counts().to_dict().items()
        },
    }
    return kinase_env, summary, warnings


def build_source_environments(df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, list[str]]:
    logging.info("Annotating source/document/assay environments.")
    warnings: list[str] = []
    working = df.copy()
    for column in ["source_id", "source_description", "doc_id", "assay_chembl_id"]:
        working[column] = working[column].fillna(DEFAULT_SOURCE_PLACEHOLDER)

    if not cfg.compute_source_environments:
        warnings.append("Source environment computation disabled in config; placeholder table will be written.")

    group_cols = ["source_id", "source_description", "doc_id", "assay_chembl_id"]
    source_env = (
        working.groupby(group_cols, dropna=False)
        .agg(
            number_of_records=("compound_id", "size"),
            number_of_kinases=("target_chembl_id", "nunique"),
            number_of_compounds=("compound_id", "nunique"),
            median_pKi=("pKi", "median"),
        )
        .reset_index()
    )
    source_env["source_frequency_bin"] = source_env["number_of_records"].map(frequency_bin)
    source_env["document_frequency_bin"] = (
        source_env.groupby("doc_id")["number_of_records"].transform("sum").astype(int).map(frequency_bin)
        if cfg.compute_document_frequency_bins
        else pd.Series(pd.NA, index=source_env.index, dtype="string")
    )
    source_env["assay_diversity_within_source"] = (
        source_env.groupby(["source_id", "source_description"])["assay_chembl_id"].transform("nunique").astype(int)
    )
    source_env = source_env.sort_values(group_cols, kind="mergesort").reset_index(drop=True)

    if (source_env["source_id"] == DEFAULT_SOURCE_PLACEHOLDER).all():
        warnings.append(
            "All source/document/assay fields were unavailable; source environment outputs use explicit UNAVAILABLE placeholders."
        )

    return source_env, warnings


def build_pair_environments(df: pd.DataFrame, compound_env: pd.DataFrame, kinase_env: pd.DataFrame, matrix_df: pd.DataFrame, mask_df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Annotating pair-level environments.")
    working = df.copy()
    for column in ["assay_chembl_id", "doc_id", "source_id", "source_description"]:
        working[column] = working[column].fillna(DEFAULT_SOURCE_PLACEHOLDER)

    pair_env = (
        working.groupby(["compound_id", "standardized_smiles", "target_chembl_id", "target_name"], dropna=False)
        .agg(
            pKi=("pKi", "median"),
            record_support_count=("compound_id", "size"),
            assay_support_count=("assay_chembl_id", lambda s: int((s[s != DEFAULT_SOURCE_PLACEHOLDER]).nunique())),
            document_support_count=("doc_id", lambda s: int((s[s != DEFAULT_SOURCE_PLACEHOLDER]).nunique())),
            source_support_count=("source_id", lambda s: int((s[s != DEFAULT_SOURCE_PLACEHOLDER]).nunique())),
            multi_assay_support_flag=("assay_chembl_id", lambda s: bool((s[s != DEFAULT_SOURCE_PLACEHOLDER]).nunique() > 1)),
            multi_document_support_flag=("doc_id", lambda s: bool((s[s != DEFAULT_SOURCE_PLACEHOLDER]).nunique() > 1)),
            multi_source_support_flag=("source_id", lambda s: bool((s[s != DEFAULT_SOURCE_PLACEHOLDER]).nunique() > 1)),
            source_multiplicity=("source_id", lambda s: int((s[s != DEFAULT_SOURCE_PLACEHOLDER]).nunique())),
            document_multiplicity=("doc_id", lambda s: int((s[s != DEFAULT_SOURCE_PLACEHOLDER]).nunique())),
        )
        .reset_index()
    )

    compound_frequency = working.groupby("compound_id")["target_chembl_id"].nunique().rename("compound_frequency_across_kinases")
    kinase_frequency = working.groupby("target_chembl_id")["compound_id"].nunique().rename("kinase_frequency_across_compounds")
    density_by_compound = working.groupby("compound_id").size().rename("compound_panel_density_records")
    density_by_kinase = working.groupby("target_chembl_id").size().rename("kinase_panel_density_records")

    matrix = matrix_df.copy()
    matrix_row_label = matrix.columns[0]
    matrix = matrix.rename(columns={matrix_row_label: "standardized_smiles"}) if matrix_row_label != "standardized_smiles" else matrix
    mask = mask_df.copy()
    mask_row_label = mask.columns[0]
    mask = mask.rename(columns={mask_row_label: "standardized_smiles"}) if mask_row_label != "standardized_smiles" else mask
    panel_density = float(mask.drop(columns=["standardized_smiles"]).to_numpy().sum() / (max(mask.shape[0], 1) * max(mask.shape[1] - 1, 1)))

    pair_env = pair_env.merge(
        compound_env[[
            "compound_id",
            "standardized_smiles",
            "murcko_scaffold",
            "generic_murcko_scaffold",
            "scaffold_frequency_bin",
            "generic_scaffold_frequency_bin",
        ]],
        on=["compound_id", "standardized_smiles"],
        how="left",
        sort=False,
    )
    pair_env = pair_env.merge(
        kinase_env[[
            "target_chembl_id",
            "kinase_family_broad_group",
            "kinase_family",
            "kinase_subfamily",
            "kinase_family_annotation_source",
        ]],
        on="target_chembl_id",
        how="left",
        sort=False,
    )
    pair_env = pair_env.merge(compound_frequency, on="compound_id", how="left", sort=False)
    pair_env = pair_env.merge(kinase_frequency, on="target_chembl_id", how="left", sort=False)
    pair_env = pair_env.merge(density_by_compound, on="compound_id", how="left", sort=False)
    pair_env = pair_env.merge(density_by_kinase, on="target_chembl_id", how="left", sort=False)
    pair_env["global_panel_density"] = panel_density
    pair_env = pair_env.sort_values(["compound_id", "target_chembl_id"], kind="mergesort").reset_index(drop=True)
    return pair_env


def fingerprint_from_smiles(smiles: str, radius: int, nbits: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def generate_activity_cliffs(pair_env: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    logging.info("Computing per-kinase activity-cliff annotations.")
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    similarity_values: list[float] = []
    skipped_kinases: list[dict[str, Any]] = []

    if not cfg.compute_pairwise_activity_cliffs:
        return pd.DataFrame(columns=[
            "target_chembl_id",
            "target_name",
            "compound_id_1",
            "compound_id_2",
            "smiles_1",
            "smiles_2",
            "scaffold_1",
            "scaffold_2",
            "generic_scaffold_1",
            "generic_scaffold_2",
            "tanimoto_similarity",
            "pKi_1",
            "pKi_2",
            "delta_pKi",
            "same_scaffold_flag",
            "same_generic_scaffold_flag",
            "activity_cliff_flag",
        ]), {"n_activity_cliff_pairs": 0, "skipped_kinases": []}, ["Activity-cliff computation disabled in config."]

    for (target_chembl_id, target_name), kinase_df in pair_env.groupby(["target_chembl_id", "target_name"], sort=True):
        kinase_df = kinase_df.sort_values(["compound_id", "standardized_smiles"], kind="mergesort").reset_index(drop=True)
        n_compounds = len(kinase_df)
        n_pairs = n_compounds * (n_compounds - 1) // 2
        if cfg.max_pairs_for_cliff_analysis_per_kinase and n_pairs > cfg.max_pairs_for_cliff_analysis_per_kinase:
            message = {
                "target_chembl_id": target_chembl_id,
                "target_name": target_name,
                "candidate_pairs": int(n_pairs),
                "configured_limit": int(cfg.max_pairs_for_cliff_analysis_per_kinase),
            }
            skipped_kinases.append(message)
            warnings.append(
                "Skipped activity-cliff analysis for kinase "
                f"{target_chembl_id} ({target_name}) because candidate pair count {n_pairs} exceeds "
                f"configured limit {cfg.max_pairs_for_cliff_analysis_per_kinase}."
            )
            continue

        fps = [fingerprint_from_smiles(smiles, cfg.morgan_radius, cfg.morgan_nbits) for smiles in kinase_df["standardized_smiles"]]
        for i in range(n_compounds):
            fp_i = fps[i]
            if fp_i is None:
                continue
            for j in range(i + 1, n_compounds):
                fp_j = fps[j]
                if fp_j is None:
                    continue
                similarity = float(DataStructs.TanimotoSimilarity(fp_i, fp_j))
                if cfg.save_similarity_diagnostics:
                    similarity_values.append(similarity)
                pki_1 = float(kinase_df.iloc[i]["pKi"])
                pki_2 = float(kinase_df.iloc[j]["pKi"])
                delta_pki = abs(pki_1 - pki_2)
                flag = bool(
                    similarity >= cfg.activity_cliff_similarity_threshold
                    and delta_pki >= cfg.activity_cliff_delta_pki_threshold
                )
                if flag:
                    records.append(
                        {
                            "target_chembl_id": target_chembl_id,
                            "target_name": target_name,
                            "compound_id_1": kinase_df.iloc[i]["compound_id"],
                            "compound_id_2": kinase_df.iloc[j]["compound_id"],
                            "smiles_1": kinase_df.iloc[i]["standardized_smiles"],
                            "smiles_2": kinase_df.iloc[j]["standardized_smiles"],
                            "scaffold_1": kinase_df.iloc[i]["murcko_scaffold"],
                            "scaffold_2": kinase_df.iloc[j]["murcko_scaffold"],
                            "generic_scaffold_1": kinase_df.iloc[i]["generic_murcko_scaffold"],
                            "generic_scaffold_2": kinase_df.iloc[j]["generic_murcko_scaffold"],
                            "tanimoto_similarity": similarity,
                            "pKi_1": pki_1,
                            "pKi_2": pki_2,
                            "delta_pKi": delta_pki,
                            "same_scaffold_flag": bool(kinase_df.iloc[i]["murcko_scaffold"] == kinase_df.iloc[j]["murcko_scaffold"]),
                            "same_generic_scaffold_flag": bool(
                                kinase_df.iloc[i]["generic_murcko_scaffold"]
                                == kinase_df.iloc[j]["generic_murcko_scaffold"]
                            ),
                            "activity_cliff_flag": True,
                        }
                    )

    activity_cliffs = pd.DataFrame(records)
    if activity_cliffs.empty:
        activity_cliffs = pd.DataFrame(
            columns=[
                "target_chembl_id",
                "target_name",
                "compound_id_1",
                "compound_id_2",
                "smiles_1",
                "smiles_2",
                "scaffold_1",
                "scaffold_2",
                "generic_scaffold_1",
                "generic_scaffold_2",
                "tanimoto_similarity",
                "pKi_1",
                "pKi_2",
                "delta_pKi",
                "same_scaffold_flag",
                "same_generic_scaffold_flag",
                "activity_cliff_flag",
            ]
        )
    else:
        activity_cliffs = activity_cliffs.sort_values(
            ["target_chembl_id", "compound_id_1", "compound_id_2"], kind="mergesort"
        ).reset_index(drop=True)

    summary = {
        "n_activity_cliff_pairs": int(len(activity_cliffs)),
        "thresholds": {
            "similarity_metric": cfg.activity_cliff_similarity_metric,
            "morgan_radius": cfg.morgan_radius,
            "morgan_nbits": cfg.morgan_nbits,
            "activity_cliff_similarity_threshold": cfg.activity_cliff_similarity_threshold,
            "activity_cliff_delta_pki_threshold": cfg.activity_cliff_delta_pki_threshold,
        },
        "similarity_diagnostics": {
            "count": int(len(similarity_values)),
            "mean": float(np.mean(similarity_values)) if similarity_values else None,
            "median": float(np.median(similarity_values)) if similarity_values else None,
            "max": float(np.max(similarity_values)) if similarity_values else None,
        },
        "skipped_kinases": skipped_kinases,
    }
    return activity_cliffs, summary, warnings


def merge_annotated_long(df: pd.DataFrame, compound_env: pd.DataFrame, kinase_env: pd.DataFrame, pair_env: pd.DataFrame) -> pd.DataFrame:
    annotated = df.merge(compound_env, on=["compound_id", "standardized_smiles"], how="left", sort=False)
    annotated = annotated.merge(
        kinase_env[[
            "target_chembl_id",
            "target_name",
            "target_name_normalized",
            "kinase_family_broad_group",
            "kinase_family",
            "kinase_subfamily",
            "kinase_family_annotation_source",
            "number_of_compounds_measured",
            "median_pKi",
            "pKi_spread",
            "source_diversity",
            "document_diversity",
            "assay_diversity",
        ]],
        on=["target_chembl_id", "target_name"],
        how="left",
        sort=False,
        suffixes=("", "_kinase"),
    )
    pair_subset = pair_env.drop(columns=["target_name", "standardized_smiles", "pKi"])
    annotated = annotated.merge(
        pair_subset,
        on=["compound_id", "target_chembl_id"],
        how="left",
        sort=False,
        suffixes=("", "_pair"),
    )
    annotated = annotated.sort_values(["compound_id", "target_chembl_id"], kind="mergesort").reset_index(drop=True)
    return annotated


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def write_outputs(
    cfg: AppConfig,
    annotated_long: pd.DataFrame,
    compound_env: pd.DataFrame,
    kinase_env: pd.DataFrame,
    source_env: pd.DataFrame,
    pair_env: pd.DataFrame,
    activity_cliffs: pd.DataFrame,
    report: dict[str, Any],
) -> None:
    logging.info("Writing Script-04 outputs.")
    annotated_long.to_csv(cfg.output_annotated_long_path, index=False)
    compound_env.to_csv(cfg.output_compound_env_path, index=False)
    kinase_env.to_csv(cfg.output_kinase_env_path, index=False)
    source_env.to_csv(cfg.output_source_env_path, index=False)
    pair_env.to_csv(cfg.output_pair_env_path, index=False)
    activity_cliffs.to_csv(cfg.output_activity_cliff_path, index=False)
    with cfg.output_env_report_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(report), handle, indent=2, sort_keys=False)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg, loaded_config_path, raw_config = load_config(args.config, project_root)
    ensure_output_dirs(cfg)
    log_file, timestamp = setup_logging(cfg.logs_dir)

    logging.info("Starting %s", SCRIPT_NAME)
    logging.info("Using deterministic random seed constant: %s", RANDOM_SEED)

    config_snapshot_path = None
    if cfg.save_config_snapshot:
        config_snapshot_path = save_config_snapshot(cfg, loaded_config_path, raw_config)
        logging.info("Saved config snapshot to %s", config_snapshot_path)

    validate_input_paths(cfg)

    long_df_raw = read_csv_with_logging(cfg.input_long_path, "Script-03 long-format panel")
    matrix_df = read_csv_with_logging(cfg.input_matrix_path, "Script-03 pKi matrix")
    mask_df = read_csv_with_logging(cfg.input_mask_path, "Script-03 observation mask")
    kinase_summary_df = read_csv_with_logging(cfg.input_kinase_summary_path, "Script-03 kinase summary")
    supplemental_df = (
        read_csv_with_logging(cfg.supplemental_metadata_path, "supplemental metadata")
        if cfg.supplemental_metadata_path
        else None
    )

    standardized_long_df, standardization_warnings = standardize_internal_columns(long_df_raw, supplemental_df)
    validate_panel_alignment(standardized_long_df, matrix_df, mask_df, kinase_summary_df)

    compound_env, compound_summary, compound_warnings = annotate_compounds(standardized_long_df, cfg)
    kinase_env, kinase_summary_payload, kinase_warnings = annotate_kinases(standardized_long_df, kinase_summary_df, cfg)
    source_env, source_warnings = build_source_environments(standardized_long_df, cfg)
    pair_env = build_pair_environments(standardized_long_df, compound_env, kinase_env, matrix_df, mask_df)
    activity_cliffs, cliff_summary, cliff_warnings = generate_activity_cliffs(pair_env, cfg)
    annotated_long = merge_annotated_long(standardized_long_df, compound_env, kinase_env, pair_env)

    notes_on_missing_metadata = standardization_warnings + compound_warnings + kinase_warnings + source_warnings + cliff_warnings

    report = {
        "script": SCRIPT_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file_paths": {
            "input_long_path": cfg.input_long_path,
            "input_matrix_path": cfg.input_matrix_path,
            "input_mask_path": cfg.input_mask_path,
            "input_kinase_summary_path": cfg.input_kinase_summary_path,
            "supplemental_metadata_path": cfg.supplemental_metadata_path,
        },
        "output_file_paths": {
            "output_annotated_long_path": cfg.output_annotated_long_path,
            "output_compound_env_path": cfg.output_compound_env_path,
            "output_kinase_env_path": cfg.output_kinase_env_path,
            "output_source_env_path": cfg.output_source_env_path,
            "output_pair_env_path": cfg.output_pair_env_path,
            "output_activity_cliff_path": cfg.output_activity_cliff_path,
            "output_env_report_path": cfg.output_env_report_path,
            "log_file": log_file,
        },
        "total_input_rows": int(len(standardized_long_df)),
        "total_unique_compounds": int(standardized_long_df["compound_id"].nunique()),
        "total_unique_kinases": int(standardized_long_df["target_chembl_id"].nunique()),
        "total_source_entities": {
            "source_id": int(standardized_long_df["source_id"].dropna().nunique()),
            "doc_id": int(standardized_long_df["doc_id"].dropna().nunique()),
            "assay_chembl_id": int(standardized_long_df["assay_chembl_id"].dropna().nunique()),
        },
        "number_of_compounds_successfully_annotated": int(compound_env["rdkit_parse_success"].sum()),
        "number_of_kinases_successfully_annotated": int(len(kinase_env)),
        "number_of_pair_level_rows_annotated": int(len(pair_env)),
        "number_of_unique_scaffolds": compound_summary["n_unique_scaffolds"],
        "number_of_unique_generic_scaffolds": compound_summary["n_unique_generic_scaffolds"],
        "scaffold_frequency_distribution_summary": compound_summary["scaffold_frequency_distribution"],
        "kinase_family_distribution_summary": kinase_summary_payload["kinase_family_distribution"],
        "number_of_activity_cliff_pairs_identified": cliff_summary["n_activity_cliff_pairs"],
        "cliff_analysis_thresholds_used": cliff_summary["thresholds"],
        "notes_on_missing_metadata": notes_on_missing_metadata,
        "config_snapshot_reference": config_snapshot_path,
        "compound_annotation_summary": compound_summary,
        "kinase_annotation_summary": kinase_summary_payload,
        "activity_cliff_summary": cliff_summary,
    }

    write_outputs(
        cfg,
        annotated_long,
        compound_env,
        kinase_env,
        source_env,
        pair_env,
        activity_cliffs,
        report,
    )

    logging.info(
        "Completed %s | rows=%d | compounds=%d | kinases=%d | pair_rows=%d | activity_cliffs=%d",
        SCRIPT_NAME,
        len(annotated_long),
        compound_summary["n_compounds"],
        len(kinase_env),
        len(pair_env),
        cliff_summary["n_activity_cliff_pairs"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
