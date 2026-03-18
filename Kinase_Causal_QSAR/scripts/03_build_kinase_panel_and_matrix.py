#!/usr/bin/env python3
"""Build the final kinase panel and sparse compound-kinase pKi matrix.

This script is a strict continuation of Script-02 and consumes only the
curated long-format CSV configured at `script_03.input_csv_path`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SCRIPT_NAME = "03_build_kinase_panel_and_matrix"
RANDOM_SEED = 2025
REQUIRED_CONFIG_KEYS = {
    "input_csv_path",
    "output_long_path",
    "output_matrix_path",
    "output_mask_path",
    "output_kinase_summary_path",
    "output_overlap_matrix_path",
    "output_compound_summary_path",
    "output_panel_report_path",
    "min_records_per_kinase",
    "min_unique_compounds_per_kinase",
    "min_kinases_per_compound",
    "min_overlap_compounds_between_kinases",
    "drop_singleton_compounds",
    "save_dense_matrix_copy",
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
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
    "Ki_nM": ("Ki_nM", "ki_nM", "median_ki_nM", "aggregated_ki_nM"),
    "source_record_count": (
        "source_record_count",
        "n_source_records",
        "supporting_record_count",
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
}

CANONICAL_COLUMN_ORDER = [
    "compound_id",
    "standardized_smiles",
    "target_chembl_id",
    "target_name",
    "Ki_nM",
    "pKi",
    "source_record_count",
    "unique_assay_count",
    "unique_document_count",
]


def parse_bool(value: Any, key: str) -> bool:
    """Parse a YAML boolean-like value with clear error messaging."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"script_03.{key} must be a boolean; got {value!r}.")


@dataclass
class AppConfig:
    """Runtime configuration for Script-03."""

    input_csv_path: Path
    output_long_path: Path
    output_matrix_path: Path
    output_mask_path: Path
    output_kinase_summary_path: Path
    output_overlap_matrix_path: Path
    output_compound_summary_path: Path
    output_panel_report_path: Path
    min_records_per_kinase: int
    min_unique_compounds_per_kinase: int
    min_kinases_per_compound: int
    min_overlap_compounds_between_kinases: int
    drop_singleton_compounds: bool
    save_dense_matrix_copy: bool
    configs_used_dir: Path
    logs_dir: Path

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        if not isinstance(raw, dict):
            raise ValueError("Config YAML must contain a top-level mapping/object.")

        script_cfg = raw.get("script_03")
        if not isinstance(script_cfg, dict):
            raise ValueError(
                "Missing required `script_03` section in config.yaml for Script-03 execution."
            )

        missing_keys = sorted(REQUIRED_CONFIG_KEYS.difference(script_cfg))
        if missing_keys:
            raise ValueError(
                "Missing required script_03 config values: " + ", ".join(missing_keys)
            )

        def resolve(path_like: str | Path) -> Path:
            path = Path(path_like)
            return path if path.is_absolute() else project_root / path

        def require_non_negative_int(key: str) -> int:
            value = script_cfg.get(key)
            try:
                int_value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"script_03.{key} must be an integer; got {value!r}.") from exc
            if int_value < 0:
                raise ValueError(f"script_03.{key} must be >= 0; got {int_value}.")
            return int_value

        min_records_per_kinase = require_non_negative_int("min_records_per_kinase")
        min_unique_compounds_per_kinase = require_non_negative_int(
            "min_unique_compounds_per_kinase"
        )
        min_kinases_per_compound = require_non_negative_int("min_kinases_per_compound")
        min_overlap = require_non_negative_int("min_overlap_compounds_between_kinases")

        if min_records_per_kinase == 0:
            raise ValueError("script_03.min_records_per_kinase must be > 0.")
        if min_unique_compounds_per_kinase == 0:
            raise ValueError("script_03.min_unique_compounds_per_kinase must be > 0.")

        return AppConfig(
            input_csv_path=resolve(script_cfg["input_csv_path"]),
            output_long_path=resolve(script_cfg["output_long_path"]),
            output_matrix_path=resolve(script_cfg["output_matrix_path"]),
            output_mask_path=resolve(script_cfg["output_mask_path"]),
            output_kinase_summary_path=resolve(script_cfg["output_kinase_summary_path"]),
            output_overlap_matrix_path=resolve(script_cfg["output_overlap_matrix_path"]),
            output_compound_summary_path=resolve(script_cfg["output_compound_summary_path"]),
            output_panel_report_path=resolve(script_cfg["output_panel_report_path"]),
            min_records_per_kinase=min_records_per_kinase,
            min_unique_compounds_per_kinase=min_unique_compounds_per_kinase,
            min_kinases_per_compound=min_kinases_per_compound,
            min_overlap_compounds_between_kinases=min_overlap,
            drop_singleton_compounds=parse_bool(
                script_cfg["drop_singleton_compounds"], "drop_singleton_compounds"
            ),
            save_dense_matrix_copy=parse_bool(
                script_cfg["save_dense_matrix_copy"], "save_dense_matrix_copy"
            ),
            configs_used_dir=resolve(raw.get("script_02", {}).get("configs_used_dir", "configs_used")),
            logs_dir=resolve(raw.get("script_02", {}).get("logs_dir", "logs")),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the final kinase panel and sparse compound-kinase pKi matrix "
            "from Script-02 curated regression-ready output."
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
        cfg.output_long_path,
        cfg.output_matrix_path,
        cfg.output_mask_path,
        cfg.output_kinase_summary_path,
        cfg.output_overlap_matrix_path,
        cfg.output_compound_summary_path,
        cfg.output_panel_report_path,
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
            "output_long_path": str(cfg.output_long_path),
            "output_matrix_path": str(cfg.output_matrix_path),
            "output_mask_path": str(cfg.output_mask_path),
            "output_kinase_summary_path": str(cfg.output_kinase_summary_path),
            "output_overlap_matrix_path": str(cfg.output_overlap_matrix_path),
            "output_compound_summary_path": str(cfg.output_compound_summary_path),
            "output_panel_report_path": str(cfg.output_panel_report_path),
            "logs_dir": str(cfg.logs_dir),
            "configs_used_dir": str(cfg.configs_used_dir),
        },
        "parameters": {
            "min_records_per_kinase": cfg.min_records_per_kinase,
            "min_unique_compounds_per_kinase": cfg.min_unique_compounds_per_kinase,
            "min_kinases_per_compound": cfg.min_kinases_per_compound,
            "min_overlap_compounds_between_kinases": cfg.min_overlap_compounds_between_kinases,
            "drop_singleton_compounds": cfg.drop_singleton_compounds,
            "save_dense_matrix_copy": cfg.save_dense_matrix_copy,
        },
        "source_config": loaded_raw_config,
    }

    with snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    return snapshot_path


def validate_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "Expected Script-02 curated regression-ready CSV is missing: "
            f"{path}. Run scripts/02_curate_and_aggregate_kinase_ki.py first."
        )


def resolve_column_map(df: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for canonical_name, aliases in COLUMN_ALIASES.items():
        match = next((candidate for candidate in aliases if candidate in df.columns), None)
        if match is None:
            if canonical_name in {"unique_assay_count", "unique_document_count"}:
                continue
            missing.append(canonical_name)
        else:
            resolved[canonical_name] = match

    if missing:
        missing_message = ", ".join(missing)
        raise ValueError(
            "Missing required Script-02 columns after alias normalization: "
            f"{missing_message}. Available columns: {sorted(df.columns.tolist())}"
        )

    return resolved


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = resolve_column_map(df)
    rename_map = {source: canonical for canonical, source in column_map.items()}
    working = df.rename(columns=rename_map).copy()

    if "unique_assay_count" not in working.columns:
        working["unique_assay_count"] = np.nan
    if "unique_document_count" not in working.columns:
        working["unique_document_count"] = np.nan

    return working


def validate_curated_dataset(df: pd.DataFrame) -> pd.DataFrame:
    working = canonicalize_columns(df)

    for column in ["compound_id", "standardized_smiles", "target_chembl_id", "target_name"]:
        working[column] = working[column].astype("string").str.strip()
        if working[column].isna().any() or (working[column] == "").any():
            raise ValueError(f"Required identifier column `{column}` contains missing/blank values.")

    working["pKi"] = pd.to_numeric(working["pKi"], errors="coerce")
    if working["pKi"].isna().any():
        raise ValueError(
            "Column `pKi` must be numeric and non-null for all retained Script-02 rows."
        )

    working["Ki_nM"] = pd.to_numeric(working["Ki_nM"], errors="coerce")
    if working["Ki_nM"].isna().any():
        raise ValueError("Column `Ki_nM` must be numeric and non-null for Script-03.")

    for column in ["source_record_count", "unique_assay_count", "unique_document_count"]:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    if working.duplicated(subset=["standardized_smiles", "target_chembl_id"]).any():
        duplicates = int(working.duplicated(subset=["standardized_smiles", "target_chembl_id"]).sum())
        raise ValueError(
            "Script-03 expects one aggregated row per compound-kinase pair, but found "
            f"{duplicates} duplicate pair rows. Review Script-02 aggregation output."
        )

    return working[CANONICAL_COLUMN_ORDER].copy()


def build_kinase_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["target_chembl_id", "target_name"], dropna=False)
        .agg(
            number_of_records=("target_chembl_id", "size"),
            number_of_unique_compounds=("standardized_smiles", "nunique"),
            median_pKi=("pKi", "median"),
            pKi_std=("pKi", "std"),
            number_of_supporting_assays=("unique_assay_count", "sum"),
            number_of_supporting_documents=("unique_document_count", "sum"),
        )
        .reset_index()
    )
    summary["pKi_std"] = summary["pKi_std"].fillna(0.0)
    summary[["number_of_records", "number_of_unique_compounds"]] = summary[[
        "number_of_records",
        "number_of_unique_compounds",
    ]].astype(int)
    summary = summary.sort_values(
        by=["number_of_records", "number_of_unique_compounds", "target_chembl_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return summary


def build_compound_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["compound_id", "standardized_smiles"], dropna=False)
        .agg(
            number_of_kinases_measured=("target_chembl_id", "nunique"),
            number_of_records=("target_chembl_id", "size"),
            median_pKi_across_measured_kinases=("pKi", "median"),
            max_pKi=("pKi", "max"),
            min_pKi=("pKi", "min"),
        )
        .reset_index()
    )
    summary[["number_of_kinases_measured", "number_of_records"]] = summary[[
        "number_of_kinases_measured",
        "number_of_records",
    ]].astype(int)
    summary = summary.sort_values(
        by=["number_of_kinases_measured", "number_of_records", "compound_id", "standardized_smiles"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return summary


def determine_removed_kinases(summary: pd.DataFrame, cfg: AppConfig) -> list[dict[str, Any]]:
    removed: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        reasons: list[str] = []
        if row.number_of_records < cfg.min_records_per_kinase:
            reasons.append(
                f"number_of_records={row.number_of_records} < min_records_per_kinase={cfg.min_records_per_kinase}"
            )
        if row.number_of_unique_compounds < cfg.min_unique_compounds_per_kinase:
            reasons.append(
                "number_of_unique_compounds="
                f"{row.number_of_unique_compounds} < min_unique_compounds_per_kinase="
                f"{cfg.min_unique_compounds_per_kinase}"
            )
        if reasons:
            removed.append(
                {
                    "target_chembl_id": row.target_chembl_id,
                    "target_name": row.target_name,
                    "reasons": reasons,
                }
            )
    return removed


def select_kinase_panel(summary: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    removed = determine_removed_kinases(summary, cfg)
    selected = summary[
        (summary["number_of_records"] >= cfg.min_records_per_kinase)
        & (summary["number_of_unique_compounds"] >= cfg.min_unique_compounds_per_kinase)
    ].copy()
    selected = selected.sort_values(
        by=["number_of_records", "number_of_unique_compounds", "target_chembl_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return selected, removed


def apply_compound_filtering(df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    compound_summary = build_compound_summary(df)
    if cfg.drop_singleton_compounds:
        kept = compound_summary[
            compound_summary["number_of_kinases_measured"] >= cfg.min_kinases_per_compound
        ].copy()
        removed_count = int(len(compound_summary) - len(kept))
        filtered_df = df[df["standardized_smiles"].isin(kept["standardized_smiles"])].copy()
        return filtered_df, kept, removed_count

    return df.copy(), compound_summary, 0


def build_overlap_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    presence = (
        df.assign(observed=1)
        .pivot(index="standardized_smiles", columns="target_chembl_id", values="observed")
        .fillna(0)
        .astype(int)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    raw_overlap = presence.T.dot(presence).astype(int)

    target_sizes = np.diag(raw_overlap.to_numpy())
    denominator_union = target_sizes[:, None] + target_sizes[None, :] - raw_overlap.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.divide(
            raw_overlap.to_numpy(),
            denominator_union,
            out=np.zeros_like(raw_overlap.to_numpy(), dtype=float),
            where=denominator_union > 0,
        )
        overlap_coeff = np.divide(
            raw_overlap.to_numpy(),
            np.minimum(target_sizes[:, None], target_sizes[None, :]),
            out=np.zeros_like(raw_overlap.to_numpy(), dtype=float),
            where=np.minimum(target_sizes[:, None], target_sizes[None, :]) > 0,
        )

    jaccard_df = pd.DataFrame(jaccard, index=raw_overlap.index, columns=raw_overlap.columns)
    overlap_coeff_df = pd.DataFrame(
        overlap_coeff, index=raw_overlap.index, columns=raw_overlap.columns
    )
    return raw_overlap, jaccard_df, overlap_coeff_df


def build_sparse_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = (
        df.pivot(index="standardized_smiles", columns="target_chembl_id", values="pKi")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    mask = matrix.notna().astype(int)
    return matrix, mask


def matrix_density(mask: pd.DataFrame) -> float:
    total_cells = int(mask.shape[0] * mask.shape[1])
    if total_cells == 0:
        return 0.0
    return float(mask.to_numpy().sum() / total_cells)


def build_report(
    cfg: AppConfig,
    timestamp: str,
    config_snapshot: Path,
    raw_df: pd.DataFrame,
    validated_df: pd.DataFrame,
    selected_kinases: pd.DataFrame,
    final_long_df: pd.DataFrame,
    final_compound_summary: pd.DataFrame,
    matrix: pd.DataFrame,
    mask: pd.DataFrame,
    removed_kinases: list[dict[str, Any]],
    singleton_removed_count: int,
    overlap_matrix: pd.DataFrame,
) -> dict[str, Any]:
    density = matrix_density(mask)
    overlap_threshold_pairs = int(
        np.sum(np.triu(overlap_matrix.to_numpy(), k=1) >= cfg.min_overlap_compounds_between_kinases)
    )

    top_by_records = selected_kinases.nlargest(10, "number_of_records")[[
        "target_chembl_id",
        "target_name",
        "number_of_records",
    ]].to_dict(orient="records")
    top_by_compounds = selected_kinases.nlargest(10, "number_of_unique_compounds")[[
        "target_chembl_id",
        "target_name",
        "number_of_unique_compounds",
    ]].to_dict(orient="records")

    return {
        "script": SCRIPT_NAME,
        "timestamp": timestamp,
        "random_seed": RANDOM_SEED,
        "input_file_path": str(cfg.input_csv_path),
        "output_file_paths": {
            "output_long_path": str(cfg.output_long_path),
            "output_matrix_path": str(cfg.output_matrix_path),
            "output_mask_path": str(cfg.output_mask_path),
            "output_kinase_summary_path": str(cfg.output_kinase_summary_path),
            "output_overlap_matrix_path": str(cfg.output_overlap_matrix_path),
            "output_compound_summary_path": str(cfg.output_compound_summary_path),
            "output_panel_report_path": str(cfg.output_panel_report_path),
        },
        "total_input_rows": int(len(raw_df)),
        "total_unique_compounds_before_filtering": int(
            validated_df["standardized_smiles"].nunique(dropna=True)
        ),
        "total_unique_kinases_before_filtering": int(
            validated_df["target_chembl_id"].nunique(dropna=True)
        ),
        "thresholds_used": {
            "min_records_per_kinase": cfg.min_records_per_kinase,
            "min_unique_compounds_per_kinase": cfg.min_unique_compounds_per_kinase,
            "min_kinases_per_compound": cfg.min_kinases_per_compound,
            "min_overlap_compounds_between_kinases": cfg.min_overlap_compounds_between_kinases,
            "drop_singleton_compounds": cfg.drop_singleton_compounds,
        },
        "total_unique_kinases_after_filtering": int(len(selected_kinases)),
        "total_unique_compounds_after_filtering": int(len(final_compound_summary)),
        "total_compound_kinase_pairs_after_filtering": int(len(final_long_df)),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "matrix_density": density,
        "proportion_missing": float(1.0 - density),
        "top_kinases_by_record_count": top_by_records,
        "top_kinases_by_compound_count": top_by_compounds,
        "compounds_removed_due_to_singleton_filtering": singleton_removed_count,
        "list_of_removed_kinases_with_reasons": removed_kinases,
        "pairs_meeting_min_overlap_threshold": overlap_threshold_pairs,
        "config_snapshot_reference": str(config_snapshot),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    args = parse_args()

    try:
        cfg, loaded_config_path, loaded_raw_config = load_config(args.config, project_root)
        ensure_output_dirs(cfg)
        log_file, timestamp = setup_logging(cfg.logs_dir)
        logger = logging.getLogger(__name__)

        logger.info("Starting %s", SCRIPT_NAME)
        logger.info("Fixed random seed (for traceability): %d", RANDOM_SEED)
        logger.info("Run timestamp: %s", timestamp)

        config_snapshot = save_config_snapshot(cfg, loaded_config_path, loaded_raw_config)
        logger.info("Saved config snapshot: %s", config_snapshot)

        validate_input_file(cfg.input_csv_path)
        logger.info("Reading Script-02 curated dataset from %s", cfg.input_csv_path)
        raw_df = pd.read_csv(cfg.input_csv_path)
        validated_df = validate_curated_dataset(raw_df)
        logger.info(
            "Validated curated dataset with %d rows, %d compounds, and %d kinases.",
            len(validated_df),
            validated_df["standardized_smiles"].nunique(dropna=True),
            validated_df["target_chembl_id"].nunique(dropna=True),
        )

        logger.info("Generating kinase-level summary.")
        kinase_summary = build_kinase_summary(validated_df)
        selected_kinases, removed_kinases = select_kinase_panel(kinase_summary, cfg)
        if selected_kinases.empty:
            raise ValueError(
                "No kinases passed the configured Script-03 thresholds. "
                "Lower the thresholds or inspect the Script-02 input coverage."
            )

        logger.info(
            "Selected %d/%d kinases after thresholding.",
            len(selected_kinases),
            len(kinase_summary),
        )
        if removed_kinases:
            logger.info("Removed kinases with reasons: %s", removed_kinases)

        selected_ids = selected_kinases["target_chembl_id"].tolist()
        panel_df = validated_df[validated_df["target_chembl_id"].isin(selected_ids)].copy()
        panel_df = panel_df.sort_values(
            by=["target_chembl_id", "standardized_smiles", "compound_id"],
            kind="mergesort",
        )

        logger.info("Generating compound summary and applying compound-density rules.")
        final_long_df, final_compound_summary, singleton_removed_count = apply_compound_filtering(
            panel_df,
            cfg,
        )
        if final_long_df.empty:
            raise ValueError(
                "Compound filtering removed all rows. Review min_kinases_per_compound or disable "
                "drop_singleton_compounds."
            )

        logger.info(
            "Compound filtering kept %d compounds and removed %d singleton/low-coverage compounds.",
            len(final_compound_summary),
            singleton_removed_count,
        )

        logger.info("Recomputing summaries on final panel dataset.")
        final_long_df = final_long_df[CANONICAL_COLUMN_ORDER].copy()
        final_long_df = final_long_df.sort_values(
            by=["target_chembl_id", "standardized_smiles", "compound_id"],
            kind="mergesort",
        )
        final_kinase_summary = build_kinase_summary(final_long_df)
        final_compound_summary = build_compound_summary(final_long_df)

        logger.info("Quantifying compound overlap between selected kinases.")
        overlap_matrix, jaccard_matrix, overlap_coeff_matrix = build_overlap_outputs(final_long_df)

        logger.info("Building sparse pKi matrix and observation mask.")
        matrix, mask = build_sparse_matrices(final_long_df)
        density = matrix_density(mask)
        logger.info(
            "Matrix shape=%s with density=%.6f and missing proportion=%.6f.",
            matrix.shape,
            density,
            1.0 - density,
        )

        final_kinase_summary.to_csv(cfg.output_kinase_summary_path, index=False)
        final_compound_summary.to_csv(cfg.output_compound_summary_path, index=False)
        final_long_df.to_csv(cfg.output_long_path, index=False)
        matrix.to_csv(cfg.output_matrix_path, index=True)
        mask.to_csv(cfg.output_mask_path, index=True)
        overlap_matrix.to_csv(cfg.output_overlap_matrix_path, index=True)

        if cfg.save_dense_matrix_copy:
            dense_copy_path = cfg.output_matrix_path.with_name(
                f"{cfg.output_matrix_path.stem}_dense{cfg.output_matrix_path.suffix}"
            )
            matrix.reset_index().to_csv(dense_copy_path, index=False)
            logger.info("Saved optional dense matrix copy: %s", dense_copy_path)

        diagnostics_dir = cfg.output_panel_report_path.parent
        jaccard_path = diagnostics_dir / "03_kinase_compound_jaccard_matrix.csv"
        overlap_coeff_path = diagnostics_dir / "03_kinase_compound_overlap_coefficient_matrix.csv"
        jaccard_matrix.to_csv(jaccard_path, index=True)
        overlap_coeff_matrix.to_csv(overlap_coeff_path, index=True)

        report = build_report(
            cfg=cfg,
            timestamp=timestamp,
            config_snapshot=config_snapshot,
            raw_df=raw_df,
            validated_df=validated_df,
            selected_kinases=final_kinase_summary,
            final_long_df=final_long_df,
            final_compound_summary=final_compound_summary,
            matrix=matrix,
            mask=mask,
            removed_kinases=removed_kinases,
            singleton_removed_count=singleton_removed_count,
            overlap_matrix=overlap_matrix,
        )
        report["optional_outputs"] = {
            "dense_matrix_copy": str(dense_copy_path) if cfg.save_dense_matrix_copy else None,
            "jaccard_matrix": str(jaccard_path),
            "overlap_coefficient_matrix": str(overlap_coeff_path),
            "log_file": str(log_file),
        }
        write_json(cfg.output_panel_report_path, report)
        logger.info("Saved Script-03 report: %s", cfg.output_panel_report_path)
        logger.info("Script-03 completed successfully.")
        return 0

    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("Script-03 failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
