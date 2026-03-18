#!/usr/bin/env python3
"""Define selectivity-aware regression and classification task datasets.

This script is a strict continuation of Script-04. It reads the annotated
kinase panel, preserves the core continuous pKi regression view, and derives
selectivity-oriented regression and classification tasks for later benchmarking.
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

SCRIPT_NAME = "05_define_selectivity_tasks_and_labels"
RANDOM_SEED = 2025
REQUIRED_SCRIPT_05_KEYS = {
    "input_annotated_long_path",
    "input_activity_cliff_path",
    "output_regression_long_path",
    "output_pairwise_selectivity_path",
    "output_target_vs_panel_path",
    "output_classification_path",
    "output_task_summary_path",
    "output_report_path",
    "build_multitask_regression_task",
    "build_pairwise_selectivity_task",
    "build_target_vs_panel_task",
    "build_classification_tasks",
    "min_kinases_per_compound_for_selectivity",
    "min_offtargets_per_compound",
    "pairwise_selectivity_mode",
    "pairwise_selectivity_min_delta_pki",
    "pairwise_selectivity_save_directional_pairs",
    "target_vs_panel_reference",
    "target_vs_panel_min_offtargets",
    "classification_active_threshold_pki",
    "classification_inactive_threshold_pki",
    "classification_strong_binder_threshold_pki",
    "classification_weak_binder_threshold_pki",
    "selective_threshold_delta_pki",
    "highly_selective_threshold_delta_pki",
    "gray_zone_policy",
    "save_only_labeled_classification_rows",
    "include_environment_columns_in_outputs",
    "include_activity_cliff_flags_in_outputs",
    "save_config_snapshot",
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
    "n_source_records": ("n_source_records", "source_record_count", "supporting_record_count"),
    "n_unique_assays": ("n_unique_assays", "unique_assay_count", "number_of_supporting_assays"),
    "n_unique_compound_ids": (
        "n_unique_compound_ids",
        "unique_compound_count",
        "number_of_supporting_compounds",
    ),
    "pKi_std": ("pKi_std", "aggregated_pKi_std", "pki_std"),
    "is_duplicate_measurement": ("is_duplicate_measurement", "duplicate_measurement_flag"),
}
CORE_REQUIRED_COLUMNS = {"target_chembl_id", "target_name", "standardized_smiles"}
OPTIONAL_ENVIRONMENT_COLUMNS = [
    "murcko_scaffold",
    "generic_murcko_scaffold",
    "scaffold_frequency",
    "generic_scaffold_frequency",
    "scaffold_frequency_bin",
    "generic_scaffold_frequency_bin",
    "kinase_family",
    "kinase_family_broad_group",
    "kinase_subfamily",
    "kinase_family_annotation_source",
    "source_id",
    "source_description",
    "doc_id",
    "assay_chembl_id",
    "source_frequency_bin",
    "document_frequency_bin",
    "assay_diversity_within_source",
    "record_support_count",
    "assay_support_count",
    "document_support_count",
    "source_support_count",
    "multi_assay_support_flag",
    "multi_document_support_flag",
    "multi_source_support_flag",
    "compound_frequency_across_kinases",
    "kinase_frequency_across_compounds",
]
ACTIVITY_CLIFF_OUTPUT_COLUMNS = [
    "has_activity_cliff_partner_for_target",
    "activity_cliff_partner_count_for_target",
    "max_activity_cliff_delta_pki_for_target",
    "max_activity_cliff_similarity_for_target",
]


@dataclass
class AppConfig:
    input_annotated_long_path: Path
    input_activity_cliff_path: Path | None
    output_regression_long_path: Path
    output_pairwise_selectivity_path: Path
    output_target_vs_panel_path: Path
    output_classification_path: Path
    output_task_summary_path: Path
    output_report_path: Path
    build_multitask_regression_task: bool
    build_pairwise_selectivity_task: bool
    build_target_vs_panel_task: bool
    build_classification_tasks: bool
    min_kinases_per_compound_for_selectivity: int
    min_offtargets_per_compound: int
    pairwise_selectivity_mode: str
    pairwise_selectivity_min_delta_pki: float
    pairwise_selectivity_save_directional_pairs: bool
    target_vs_panel_reference: str
    target_vs_panel_min_offtargets: int
    classification_active_threshold_pki: float
    classification_inactive_threshold_pki: float
    classification_strong_binder_threshold_pki: float
    classification_weak_binder_threshold_pki: float
    selective_threshold_delta_pki: float
    highly_selective_threshold_delta_pki: float
    gray_zone_policy: str
    save_only_labeled_classification_rows: bool
    include_environment_columns_in_outputs: bool
    include_activity_cliff_flags_in_outputs: bool
    save_config_snapshot: bool
    logs_dir: Path
    configs_used_dir: Path

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        if not isinstance(raw, dict):
            raise ValueError("Config YAML must contain a top-level mapping/object.")

        script_cfg = raw.get("script_05")
        if not isinstance(script_cfg, dict):
            raise ValueError("Missing required `script_05` section in config.yaml.")

        missing = sorted(REQUIRED_SCRIPT_05_KEYS.difference(script_cfg))
        if missing:
            raise ValueError("Missing required script_05 config values: " + ", ".join(missing))

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
            raise ValueError(f"script_05.{key} must be a boolean; got {value!r}.")

        def parse_int(key: str, minimum: int = 0) -> int:
            value = script_cfg.get(key)
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"script_05.{key} must be an integer; got {value!r}.") from exc
            if parsed < minimum:
                raise ValueError(f"script_05.{key} must be >= {minimum}; got {parsed}.")
            return parsed

        def parse_float(key: str) -> float:
            value = script_cfg.get(key)
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"script_05.{key} must be numeric; got {value!r}.") from exc

        pairwise_mode = str(script_cfg["pairwise_selectivity_mode"]).strip().lower()
        if pairwise_mode != "all_pairs":
            raise ValueError(
                "script_05.pairwise_selectivity_mode currently supports only 'all_pairs'."
            )

        target_reference = str(script_cfg["target_vs_panel_reference"]).strip().lower()
        allowed_references = {"median_offtarget", "mean_offtarget", "max_offtarget", "second_best_offtarget"}
        if target_reference not in allowed_references:
            raise ValueError(
                "script_05.target_vs_panel_reference must be one of: "
                + ", ".join(sorted(allowed_references))
            )

        gray_zone_policy = str(script_cfg["gray_zone_policy"]).strip().lower()
        if gray_zone_policy not in {"exclude", "unlabeled", "assign_negative"}:
            raise ValueError(
                "script_05.gray_zone_policy must be one of: exclude, unlabeled, assign_negative."
            )

        active_threshold = parse_float("classification_active_threshold_pki")
        inactive_threshold = parse_float("classification_inactive_threshold_pki")
        strong_threshold = parse_float("classification_strong_binder_threshold_pki")
        weak_threshold = parse_float("classification_weak_binder_threshold_pki")
        selective_threshold = parse_float("selective_threshold_delta_pki")
        highly_selective_threshold = parse_float("highly_selective_threshold_delta_pki")

        if active_threshold < inactive_threshold:
            raise ValueError(
                "script_05.classification_active_threshold_pki must be >= classification_inactive_threshold_pki."
            )
        if strong_threshold < weak_threshold:
            raise ValueError(
                "script_05.classification_strong_binder_threshold_pki must be >= classification_weak_binder_threshold_pki."
            )
        if highly_selective_threshold < selective_threshold:
            raise ValueError(
                "script_05.highly_selective_threshold_delta_pki must be >= selective_threshold_delta_pki."
            )

        return AppConfig(
            input_annotated_long_path=resolve(script_cfg["input_annotated_long_path"]),
            input_activity_cliff_path=resolve(script_cfg["input_activity_cliff_path"]),
            output_regression_long_path=resolve(script_cfg["output_regression_long_path"]),
            output_pairwise_selectivity_path=resolve(script_cfg["output_pairwise_selectivity_path"]),
            output_target_vs_panel_path=resolve(script_cfg["output_target_vs_panel_path"]),
            output_classification_path=resolve(script_cfg["output_classification_path"]),
            output_task_summary_path=resolve(script_cfg["output_task_summary_path"]),
            output_report_path=resolve(script_cfg["output_report_path"]),
            build_multitask_regression_task=parse_bool(
                script_cfg["build_multitask_regression_task"], "build_multitask_regression_task"
            ),
            build_pairwise_selectivity_task=parse_bool(
                script_cfg["build_pairwise_selectivity_task"], "build_pairwise_selectivity_task"
            ),
            build_target_vs_panel_task=parse_bool(
                script_cfg["build_target_vs_panel_task"], "build_target_vs_panel_task"
            ),
            build_classification_tasks=parse_bool(
                script_cfg["build_classification_tasks"], "build_classification_tasks"
            ),
            min_kinases_per_compound_for_selectivity=parse_int(
                "min_kinases_per_compound_for_selectivity", minimum=2
            ),
            min_offtargets_per_compound=parse_int("min_offtargets_per_compound", minimum=1),
            pairwise_selectivity_mode=pairwise_mode,
            pairwise_selectivity_min_delta_pki=parse_float("pairwise_selectivity_min_delta_pki"),
            pairwise_selectivity_save_directional_pairs=parse_bool(
                script_cfg["pairwise_selectivity_save_directional_pairs"],
                "pairwise_selectivity_save_directional_pairs",
            ),
            target_vs_panel_reference=target_reference,
            target_vs_panel_min_offtargets=parse_int("target_vs_panel_min_offtargets", minimum=1),
            classification_active_threshold_pki=active_threshold,
            classification_inactive_threshold_pki=inactive_threshold,
            classification_strong_binder_threshold_pki=strong_threshold,
            classification_weak_binder_threshold_pki=weak_threshold,
            selective_threshold_delta_pki=selective_threshold,
            highly_selective_threshold_delta_pki=highly_selective_threshold,
            gray_zone_policy=gray_zone_policy,
            save_only_labeled_classification_rows=parse_bool(
                script_cfg["save_only_labeled_classification_rows"],
                "save_only_labeled_classification_rows",
            ),
            include_environment_columns_in_outputs=parse_bool(
                script_cfg["include_environment_columns_in_outputs"],
                "include_environment_columns_in_outputs",
            ),
            include_activity_cliff_flags_in_outputs=parse_bool(
                script_cfg["include_activity_cliff_flags_in_outputs"],
                "include_activity_cliff_flags_in_outputs",
            ),
            save_config_snapshot=parse_bool(script_cfg["save_config_snapshot"], "save_config_snapshot"),
            logs_dir=resolve(raw.get("script_02", {}).get("logs_dir", "logs")),
            configs_used_dir=resolve(raw.get("script_02", {}).get("configs_used_dir", "configs_used")),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Define multitask regression, selectivity regression, and derived classification tasks from Script-04 outputs."
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
        cfg.output_regression_long_path,
        cfg.output_pairwise_selectivity_path,
        cfg.output_target_vs_panel_path,
        cfg.output_classification_path,
        cfg.output_task_summary_path,
        cfg.output_report_path,
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


def save_config_snapshot(cfg: AppConfig, loaded_config_path: Path) -> Path | None:
    if not cfg.save_config_snapshot:
        logging.info("Config snapshot saving disabled by config.")
        return None
    snapshot_path = cfg.configs_used_dir / f"{SCRIPT_NAME}_config.yaml"
    with loaded_config_path.open("r", encoding="utf-8") as src, snapshot_path.open("w", encoding="utf-8") as dst:
        dst.write(src.read())
    logging.info("Saved config snapshot to %s", snapshot_path)
    return snapshot_path


def resolve_column(df: pd.DataFrame, canonical: str) -> str | None:
    for candidate in COLUMN_ALIASES.get(canonical, (canonical,)):
        if candidate in df.columns:
            return candidate
    return None


def standardize_input_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    warnings: list[str] = []
    mapping: dict[str, str] = {}
    standardized = df.copy()

    compound_col = resolve_column(standardized, "compound_id")
    smiles_col = resolve_column(standardized, "standardized_smiles")
    if compound_col is None and smiles_col is None:
        raise ValueError(
            "Input annotated dataset must contain either `compound_id` or `standardized_smiles` to define compound identity."
        )
    if smiles_col is None:
        raise ValueError("Input annotated dataset is missing required standardized structure column `standardized_smiles`.")
    if compound_col is None:
        standardized["compound_id"] = standardized[smiles_col].astype(str)
        warnings.append(
            "`compound_id` was not present; `standardized_smiles` is being used as the canonical compound identifier."
        )
        mapping["compound_id"] = smiles_col
    elif compound_col != "compound_id":
        standardized = standardized.rename(columns={compound_col: "compound_id"})
        mapping["compound_id"] = compound_col
    else:
        mapping["compound_id"] = "compound_id"

    if smiles_col != "standardized_smiles":
        standardized = standardized.rename(columns={smiles_col: "standardized_smiles"})
    mapping["standardized_smiles"] = smiles_col

    pki_col = resolve_column(standardized, "pKi")
    if pki_col is None:
        raise ValueError("Input annotated dataset must contain either `pKi` or `median_pKi`.")
    if pki_col != "pKi":
        standardized = standardized.rename(columns={pki_col: "pKi"})
    mapping["pKi"] = pki_col

    ki_col = resolve_column(standardized, "Ki_nM")
    if ki_col is not None and ki_col != "Ki_nM":
        standardized = standardized.rename(columns={ki_col: "Ki_nM"})
    if ki_col is not None:
        mapping["Ki_nM"] = ki_col

    for canonical in ["n_source_records", "n_unique_assays", "n_unique_compound_ids", "pKi_std", "is_duplicate_measurement"]:
        col = resolve_column(standardized, canonical)
        if col is not None and col != canonical:
            standardized = standardized.rename(columns={col: canonical})
            mapping[canonical] = col
        elif col is not None:
            mapping[canonical] = canonical

    missing_core = sorted(col for col in CORE_REQUIRED_COLUMNS if col not in standardized.columns)
    if missing_core:
        raise ValueError("Input annotated dataset is missing required columns: " + ", ".join(missing_core))

    standardized["compound_id"] = standardized["compound_id"].astype(str)
    standardized["standardized_smiles"] = standardized["standardized_smiles"].astype(str)
    standardized["target_chembl_id"] = standardized["target_chembl_id"].astype(str)
    standardized["target_name"] = standardized["target_name"].astype(str)
    standardized["pKi"] = pd.to_numeric(standardized["pKi"], errors="coerce")
    if standardized["pKi"].isna().any():
        raise ValueError("Input annotated dataset contains non-numeric or missing values in the normalized `pKi` column.")
    if "Ki_nM" in standardized.columns:
        standardized["Ki_nM"] = pd.to_numeric(standardized["Ki_nM"], errors="coerce")

    return standardized, warnings, mapping


def load_required_dataframe(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} file not found: {path}")
    logging.info("Loading %s from %s", label, path)
    return pd.read_csv(path)


def aggregate_activity_cliff_flags(activity_cliff_df: pd.DataFrame) -> pd.DataFrame:
    if activity_cliff_df.empty:
        return pd.DataFrame(columns=["compound_id", "target_chembl_id", *ACTIVITY_CLIFF_OUTPUT_COLUMNS])

    required_cols = {"target_chembl_id", "compound_id_a", "compound_id_b", "activity_cliff_flag", "delta_pKi", "tanimoto_similarity"}
    missing = sorted(required_cols.difference(activity_cliff_df.columns))
    if missing:
        raise ValueError(
            "Activity-cliff file is missing required columns for propagation: " + ", ".join(missing)
        )

    flagged = activity_cliff_df.copy()
    flagged["activity_cliff_flag"] = flagged["activity_cliff_flag"].fillna(False).astype(bool)
    flagged = flagged[flagged["activity_cliff_flag"]].copy()
    if flagged.empty:
        return pd.DataFrame(columns=["compound_id", "target_chembl_id", *ACTIVITY_CLIFF_OUTPUT_COLUMNS])

    left = flagged[["target_chembl_id", "compound_id_a", "delta_pKi", "tanimoto_similarity"]].rename(
        columns={"compound_id_a": "compound_id"}
    )
    right = flagged[["target_chembl_id", "compound_id_b", "delta_pKi", "tanimoto_similarity"]].rename(
        columns={"compound_id_b": "compound_id"}
    )
    combined = pd.concat([left, right], ignore_index=True)
    combined["abs_delta_pKi"] = combined["delta_pKi"].abs()

    summary = (
        combined.groupby(["compound_id", "target_chembl_id"], dropna=False)
        .agg(
            has_activity_cliff_partner_for_target=("compound_id", lambda s: True),
            activity_cliff_partner_count_for_target=("compound_id", "size"),
            max_activity_cliff_delta_pki_for_target=("abs_delta_pKi", "max"),
            max_activity_cliff_similarity_for_target=("tanimoto_similarity", "max"),
        )
        .reset_index()
    )
    return summary


def select_optional_columns(df: pd.DataFrame, cfg: AppConfig) -> list[str]:
    optional: list[str] = []
    metadata_cols = ["Ki_nM", "n_source_records", "n_unique_assays", "n_unique_compound_ids", "pKi_std", "is_duplicate_measurement"]
    optional.extend([col for col in metadata_cols if col in df.columns])
    if cfg.include_environment_columns_in_outputs:
        optional.extend([col for col in OPTIONAL_ENVIRONMENT_COLUMNS if col in df.columns])
    return list(dict.fromkeys(optional))


def build_multitask_regression_task(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    logging.info("Building multitask regression task table.")
    base_cols = ["compound_id", "standardized_smiles", "target_chembl_id", "target_name", "pKi"]
    selected = base_cols + select_optional_columns(df, cfg)
    regression_df = df[selected].copy()
    regression_df = regression_df.sort_values(
        ["compound_id", "target_chembl_id", "target_name", "standardized_smiles"],
        kind="mergesort",
    ).reset_index(drop=True)
    return regression_df


def summarize_support_row(row: pd.Series) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for column in ["n_source_records", "n_unique_assays", "n_unique_compound_ids", "pKi_std"]:
        if column in row.index:
            summary[column] = row[column]
    return summary


def build_pairwise_selectivity_task(df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    logging.info("Building pairwise selectivity regression task table.")
    eligible_counts = df.groupby("compound_id")["target_chembl_id"].nunique()
    eligible_compounds = eligible_counts[eligible_counts >= cfg.min_kinases_per_compound_for_selectivity].index
    excluded_compounds = sorted(set(df["compound_id"]) - set(eligible_compounds))

    pair_records: list[dict[str, Any]] = []
    group_cols = ["compound_id", "standardized_smiles"]
    optional_cols = select_optional_columns(df, cfg)
    cliff_cols = [col for col in ACTIVITY_CLIFF_OUTPUT_COLUMNS if col in df.columns]

    for (compound_id, smiles), group in (
        df[df["compound_id"].isin(eligible_compounds)]
        .sort_values(["compound_id", "target_chembl_id"], kind="mergesort")
        .groupby(group_cols, dropna=False, sort=True)
    ):
        group = group.reset_index(drop=True)
        if len(group) < cfg.min_kinases_per_compound_for_selectivity:
            continue
        for i in range(len(group) - 1):
            row_a = group.iloc[i]
            for j in range(i + 1, len(group)):
                row_b = group.iloc[j]
                delta = float(row_a["pKi"] - row_b["pKi"])
                abs_delta = abs(delta)
                if abs_delta < cfg.pairwise_selectivity_min_delta_pki:
                    continue
                base_record = {
                    "compound_id": compound_id,
                    "standardized_smiles": smiles,
                    "kinase_a_chembl_id": row_a["target_chembl_id"],
                    "kinase_a_name": row_a["target_name"],
                    "kinase_b_chembl_id": row_b["target_chembl_id"],
                    "kinase_b_name": row_b["target_name"],
                    "pKi_a": float(row_a["pKi"]),
                    "pKi_b": float(row_b["pKi"]),
                    "delta_pKi": delta,
                    "abs_delta_pKi": abs_delta,
                    "pair_label_direction": f"{row_a['target_name']}_vs_{row_b['target_name']}",
                }
                for column in optional_cols:
                    if column.startswith("kinase_"):
                        continue
                    base_record[f"compound_{column}"] = row_a[column]
                for column in ["kinase_family", "kinase_family_broad_group", "kinase_subfamily"]:
                    if column in row_a.index:
                        base_record[f"kinase_a_{column}"] = row_a[column]
                    if column in row_b.index:
                        base_record[f"kinase_b_{column}"] = row_b[column]
                for column in cliff_cols:
                    base_record[f"kinase_a_{column}"] = row_a[column]
                    base_record[f"kinase_b_{column}"] = row_b[column]
                for key, value in summarize_support_row(row_a).items():
                    base_record[f"kinase_a_{key}"] = value
                for key, value in summarize_support_row(row_b).items():
                    base_record[f"kinase_b_{key}"] = value
                pair_records.append(base_record)
                if cfg.pairwise_selectivity_save_directional_pairs:
                    reverse_record = base_record.copy()
                    reverse_record.update(
                        {
                            "kinase_a_chembl_id": row_b["target_chembl_id"],
                            "kinase_a_name": row_b["target_name"],
                            "kinase_b_chembl_id": row_a["target_chembl_id"],
                            "kinase_b_name": row_a["target_name"],
                            "pKi_a": float(row_b["pKi"]),
                            "pKi_b": float(row_a["pKi"]),
                            "delta_pKi": -delta,
                            "abs_delta_pKi": abs_delta,
                            "pair_label_direction": f"{row_b['target_name']}_vs_{row_a['target_name']}",
                        }
                    )
                    for column in ["kinase_family", "kinase_family_broad_group", "kinase_subfamily"]:
                        if column in row_a.index:
                            reverse_record[f"kinase_b_{column}"] = row_a[column]
                        if column in row_b.index:
                            reverse_record[f"kinase_a_{column}"] = row_b[column]
                    for column in cliff_cols:
                        reverse_record[f"kinase_a_{column}"] = row_b[column]
                        reverse_record[f"kinase_b_{column}"] = row_a[column]
                    for key, value in summarize_support_row(row_a).items():
                        reverse_record[f"kinase_b_{key}"] = value
                    for key, value in summarize_support_row(row_b).items():
                        reverse_record[f"kinase_a_{key}"] = value
                    pair_records.append(reverse_record)

    pairwise_df = pd.DataFrame(pair_records)
    if not pairwise_df.empty:
        pairwise_df = pairwise_df.sort_values(
            ["compound_id", "kinase_a_chembl_id", "kinase_b_chembl_id", "pair_label_direction"],
            kind="mergesort",
        ).reset_index(drop=True)

    diagnostics = {
        "eligible_compound_count": int(len(eligible_compounds)),
        "excluded_compound_count": int(len(excluded_compounds)),
        "excluded_compounds": excluded_compounds,
        "min_kinases_per_compound_for_selectivity": cfg.min_kinases_per_compound_for_selectivity,
    }
    return pairwise_df, diagnostics


def compute_reference_statistic(values: pd.Series, reference_name: str) -> tuple[str, float]:
    sorted_values = values.dropna().sort_values(kind="mergesort")
    if sorted_values.empty:
        raise ValueError("Cannot compute off-target reference statistic from an empty series.")
    if reference_name == "median_offtarget":
        return reference_name, float(sorted_values.median())
    if reference_name == "mean_offtarget":
        return reference_name, float(sorted_values.mean())
    if reference_name == "max_offtarget":
        return reference_name, float(sorted_values.max())
    if reference_name == "second_best_offtarget":
        if len(sorted_values) < 2:
            raise ValueError("second_best_offtarget requires at least two off-target observations.")
        descending = sorted_values.sort_values(ascending=False, kind="mergesort").reset_index(drop=True)
        return reference_name, float(descending.iloc[1])
    raise ValueError(f"Unsupported target-vs-panel reference statistic: {reference_name}")


def build_target_vs_panel_task(df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    logging.info("Building target-vs-panel selectivity regression task table.")
    eligible_counts = df.groupby("compound_id")["target_chembl_id"].nunique()
    min_targets_required = max(cfg.target_vs_panel_min_offtargets + 1, cfg.min_offtargets_per_compound + 1)
    eligible_compounds = eligible_counts[eligible_counts >= min_targets_required].index
    excluded_compounds = sorted(set(df["compound_id"]) - set(eligible_compounds))

    records: list[dict[str, Any]] = []
    optional_cols = select_optional_columns(df, cfg)
    for (compound_id, smiles), group in (
        df[df["compound_id"].isin(eligible_compounds)]
        .sort_values(["compound_id", "target_chembl_id"], kind="mergesort")
        .groupby(["compound_id", "standardized_smiles"], dropna=False, sort=True)
    ):
        group = group.reset_index(drop=True)
        if len(group) < min_targets_required:
            continue
        all_pki = group["pKi"]
        for row in group.itertuples(index=False):
            off_targets = group[group["target_chembl_id"] != row.target_chembl_id]
            if len(off_targets) < cfg.target_vs_panel_min_offtargets or len(off_targets) < cfg.min_offtargets_per_compound:
                continue
            stat_name, reference_pki = compute_reference_statistic(off_targets["pKi"], cfg.target_vs_panel_reference)
            record = {
                "compound_id": compound_id,
                "standardized_smiles": smiles,
                "target_chembl_id": row.target_chembl_id,
                "target_name": row.target_name,
                "target_pKi": float(row.pKi),
                "offtarget_count": int(len(off_targets)),
                "offtarget_reference_statistic_name": stat_name,
                "offtarget_reference_pKi": reference_pki,
                "target_vs_panel_delta_pKi": float(row.pKi - reference_pki),
                "max_offtarget_pKi": float(off_targets["pKi"].max()),
                "median_offtarget_pKi": float(off_targets["pKi"].median()),
                "mean_offtarget_pKi": float(off_targets["pKi"].mean()),
                "compound_panel_kinase_count": int(len(group)),
                "compound_panel_pKi_mean": float(all_pki.mean()),
                "compound_panel_pKi_std": float(all_pki.std(ddof=0)),
            }
            for column in optional_cols:
                record[column] = getattr(row, column, np.nan)
            records.append(record)

    target_panel_df = pd.DataFrame(records)
    if not target_panel_df.empty:
        target_panel_df = target_panel_df.sort_values(
            ["compound_id", "target_chembl_id", "target_name"], kind="mergesort"
        ).reset_index(drop=True)

    diagnostics = {
        "eligible_compound_count": int(len(eligible_compounds)),
        "excluded_compound_count": int(len(excluded_compounds)),
        "excluded_compounds": excluded_compounds,
        "target_vs_panel_reference": cfg.target_vs_panel_reference,
        "target_vs_panel_min_offtargets": cfg.target_vs_panel_min_offtargets,
        "min_offtargets_per_compound": cfg.min_offtargets_per_compound,
    }
    return target_panel_df, diagnostics


def apply_binary_threshold(series: pd.Series, positive_threshold: float, negative_threshold: float, gray_zone_policy: str) -> tuple[pd.Series, pd.Series]:
    labels = pd.Series(pd.NA, index=series.index, dtype="object")
    rationale = pd.Series(pd.NA, index=series.index, dtype="object")
    positive_mask = series >= positive_threshold
    negative_mask = series < negative_threshold
    gray_mask = ~(positive_mask | negative_mask)

    labels.loc[positive_mask] = 1
    rationale.loc[positive_mask] = f">= {positive_threshold}"
    labels.loc[negative_mask] = 0
    rationale.loc[negative_mask] = f"< {negative_threshold}"

    if gray_zone_policy == "assign_negative":
        labels.loc[gray_mask] = 0
        rationale.loc[gray_mask] = f"gray_zone_assigned_negative_between_{negative_threshold}_and_{positive_threshold}"
    elif gray_zone_policy == "unlabeled":
        rationale.loc[gray_mask] = f"gray_zone_unlabeled_between_{negative_threshold}_and_{positive_threshold}"
    elif gray_zone_policy == "exclude":
        rationale.loc[gray_mask] = f"gray_zone_excluded_between_{negative_threshold}_and_{positive_threshold}"
    else:  # pragma: no cover - config validation should prevent this
        raise ValueError(f"Unsupported gray_zone_policy: {gray_zone_policy}")
    return labels, rationale


def build_classification_task(regression_df: pd.DataFrame, target_panel_df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    logging.info("Building derived classification task table.")
    classification_df = regression_df.copy()

    active_label, active_rule = apply_binary_threshold(
        classification_df["pKi"],
        cfg.classification_active_threshold_pki,
        cfg.classification_inactive_threshold_pki,
        cfg.gray_zone_policy,
    )
    strong_label, strong_rule = apply_binary_threshold(
        classification_df["pKi"],
        cfg.classification_strong_binder_threshold_pki,
        cfg.classification_weak_binder_threshold_pki,
        cfg.gray_zone_policy,
    )
    classification_df["active_inactive_label"] = active_label
    classification_df["active_inactive_label_name"] = classification_df["active_inactive_label"].map({1: "active", 0: "inactive"})
    classification_df["active_inactive_rule"] = active_rule
    classification_df["strong_weak_label"] = strong_label
    classification_df["strong_weak_label_name"] = classification_df["strong_weak_label"].map({1: "strong_binder", 0: "weak_binder"})
    classification_df["strong_weak_rule"] = strong_rule

    selectivity_cols = [
        "compound_id",
        "standardized_smiles",
        "target_chembl_id",
        "target_name",
        "target_pKi",
        "offtarget_count",
        "offtarget_reference_statistic_name",
        "offtarget_reference_pKi",
        "target_vs_panel_delta_pKi",
        "max_offtarget_pKi",
        "median_offtarget_pKi",
        "mean_offtarget_pKi",
    ]
    selectivity_merge = target_panel_df[selectivity_cols].copy() if not target_panel_df.empty else pd.DataFrame(columns=selectivity_cols)
    classification_df = classification_df.merge(
        selectivity_merge,
        on=["compound_id", "standardized_smiles", "target_chembl_id", "target_name"],
        how="left",
        sort=False,
    )

    selective_label = pd.Series(pd.NA, index=classification_df.index, dtype="object")
    selective_name = pd.Series(pd.NA, index=classification_df.index, dtype="object")
    selective_rule = pd.Series(pd.NA, index=classification_df.index, dtype="object")
    has_delta = classification_df["target_vs_panel_delta_pKi"].notna()
    highly_mask = has_delta & (classification_df["target_vs_panel_delta_pKi"] >= cfg.highly_selective_threshold_delta_pki)
    selective_mask = has_delta & (classification_df["target_vs_panel_delta_pKi"] >= cfg.selective_threshold_delta_pki) & ~highly_mask
    non_selective_mask = has_delta & (classification_df["target_vs_panel_delta_pKi"] < cfg.selective_threshold_delta_pki)

    selective_label.loc[highly_mask] = 1
    selective_name.loc[highly_mask] = "highly_selective"
    selective_rule.loc[highly_mask] = f">= {cfg.highly_selective_threshold_delta_pki}"
    selective_label.loc[selective_mask] = 1
    selective_name.loc[selective_mask] = "selective"
    selective_rule.loc[selective_mask] = (
        f">= {cfg.selective_threshold_delta_pki} and < {cfg.highly_selective_threshold_delta_pki}"
    )
    selective_label.loc[non_selective_mask] = 0
    selective_name.loc[non_selective_mask] = "non_selective"
    selective_rule.loc[non_selective_mask] = f"< {cfg.selective_threshold_delta_pki}"
    selective_rule.loc[~has_delta] = "missing_target_vs_panel_delta_pKi"

    classification_df["selective_label"] = selective_label
    classification_df["selective_label_name"] = selective_name
    classification_df["selective_rule"] = selective_rule

    classification_df["active_threshold_pki"] = cfg.classification_active_threshold_pki
    classification_df["inactive_threshold_pki"] = cfg.classification_inactive_threshold_pki
    classification_df["strong_binder_threshold_pki"] = cfg.classification_strong_binder_threshold_pki
    classification_df["weak_binder_threshold_pki"] = cfg.classification_weak_binder_threshold_pki
    classification_df["selective_threshold_delta_pki"] = cfg.selective_threshold_delta_pki
    classification_df["highly_selective_threshold_delta_pki"] = cfg.highly_selective_threshold_delta_pki
    classification_df["gray_zone_policy"] = cfg.gray_zone_policy

    if cfg.gray_zone_policy == "exclude" and cfg.save_only_labeled_classification_rows:
        label_columns = ["active_inactive_label", "strong_weak_label", "selective_label"]
        classification_df = classification_df[classification_df[label_columns].notna().any(axis=1)].copy()

    classification_df = classification_df.sort_values(
        ["compound_id", "target_chembl_id", "target_name", "standardized_smiles"],
        kind="mergesort",
    ).reset_index(drop=True)

    diagnostics = {
        "gray_zone_policy": cfg.gray_zone_policy,
        "save_only_labeled_classification_rows": cfg.save_only_labeled_classification_rows,
        "active_inactive_counts": value_count_dict(classification_df["active_inactive_label_name"]),
        "strong_weak_counts": value_count_dict(classification_df["strong_weak_label_name"]),
        "selective_counts": value_count_dict(classification_df["selective_label_name"]),
    }
    return classification_df, diagnostics


def describe_numeric(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": int(numeric.notna().sum()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=0)),
    }


def value_count_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.fillna("missing").value_counts(dropna=False).to_dict().items()}


def build_task_summary_table(
    regression_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    target_panel_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []

    def add_summary(task_name: str, df: pd.DataFrame, label_type: str, label_descriptor: str, label_col: str | None = None) -> None:
        summary = {
            "task_name": task_name,
            "number_of_rows": int(len(df)),
            "number_of_unique_compounds": int(df["compound_id"].nunique()) if "compound_id" in df.columns else 0,
            "number_of_unique_kinases": int(
                pd.unique(
                    pd.concat(
                        [
                            df[col]
                            for col in ["target_chembl_id", "kinase_a_chembl_id", "kinase_b_chembl_id"]
                            if col in df.columns
                        ],
                        ignore_index=True,
                    )
                ).size
            )
            if any(col in df.columns for col in ["target_chembl_id", "kinase_a_chembl_id", "kinase_b_chembl_id"])
            else 0,
            "label_type": label_type,
            "label_range_or_classes": label_descriptor,
            "number_missing_labels": int(df[label_col].isna().sum()) if label_col is not None and label_col in df.columns else 0,
            "class_balance": "",
            "summary_statistics": "",
        }
        if label_type == "classification" and label_col is not None and label_col in df.columns:
            summary["class_balance"] = json.dumps(value_count_dict(df[label_col]), sort_keys=True)
        elif label_type == "regression" and label_col is not None and label_col in df.columns:
            summary["summary_statistics"] = json.dumps(describe_numeric(df[label_col]), sort_keys=True)
        summaries.append(summary)

    add_summary("multitask_regression", regression_df, "regression", "continuous pKi", "pKi")
    add_summary("pairwise_selectivity_regression", pairwise_df, "regression", "continuous delta_pKi", "delta_pKi")
    add_summary(
        "target_vs_panel_selectivity_regression",
        target_panel_df,
        "regression",
        "continuous target_vs_panel_delta_pKi",
        "target_vs_panel_delta_pKi",
    )
    add_summary(
        "classification_active_inactive",
        classification_df,
        "classification",
        "active/inactive",
        "active_inactive_label_name",
    )
    add_summary(
        "classification_strong_weak",
        classification_df,
        "classification",
        "strong_binder/weak_binder",
        "strong_weak_label_name",
    )
    add_summary(
        "classification_selective",
        classification_df,
        "classification",
        "selective/highly_selective/non_selective",
        "selective_label_name",
    )

    return pd.DataFrame(summaries).sort_values("task_name", kind="mergesort").reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    logging.info("Writing %s rows to %s", len(df), path)
    df.to_csv(path, index=False)


def build_report(
    cfg: AppConfig,
    input_path: Path,
    activity_cliff_path: Path | None,
    regression_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    target_panel_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    input_df: pd.DataFrame,
    warnings: list[str],
    normalization_mapping: dict[str, str],
    pairwise_diagnostics: dict[str, Any],
    target_panel_diagnostics: dict[str, Any],
    classification_diagnostics: dict[str, Any],
    log_file: Path,
    config_snapshot_path: Path | None,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "script_name": SCRIPT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "input_file_paths": {
            "annotated_long": str(input_path),
            "activity_cliff": str(activity_cliff_path) if activity_cliff_path is not None else None,
        },
        "output_file_paths": {
            "multitask_regression": str(cfg.output_regression_long_path),
            "pairwise_selectivity": str(cfg.output_pairwise_selectivity_path),
            "target_vs_panel": str(cfg.output_target_vs_panel_path),
            "classification": str(cfg.output_classification_path),
            "task_summary": str(cfg.output_task_summary_path),
            "report": str(cfg.output_report_path),
            "log_file": str(log_file),
        },
        "total_input_rows": int(len(input_df)),
        "total_unique_compounds": int(input_df["compound_id"].nunique()),
        "total_unique_kinases": int(input_df["target_chembl_id"].nunique()),
        "total_multitask_regression_rows": int(len(regression_df)),
        "total_pairwise_selectivity_rows": int(len(pairwise_df)),
        "total_target_vs_panel_rows": int(len(target_panel_df)),
        "total_classification_rows": int(len(classification_df)),
        "threshold_values_used": {
            "classification_active_threshold_pki": cfg.classification_active_threshold_pki,
            "classification_inactive_threshold_pki": cfg.classification_inactive_threshold_pki,
            "classification_strong_binder_threshold_pki": cfg.classification_strong_binder_threshold_pki,
            "classification_weak_binder_threshold_pki": cfg.classification_weak_binder_threshold_pki,
            "selective_threshold_delta_pki": cfg.selective_threshold_delta_pki,
            "highly_selective_threshold_delta_pki": cfg.highly_selective_threshold_delta_pki,
            "pairwise_selectivity_min_delta_pki": cfg.pairwise_selectivity_min_delta_pki,
            "target_vs_panel_min_offtargets": cfg.target_vs_panel_min_offtargets,
            "min_kinases_per_compound_for_selectivity": cfg.min_kinases_per_compound_for_selectivity,
            "min_offtargets_per_compound": cfg.min_offtargets_per_compound,
        },
        "label_generation_rules": {
            "multitask_regression": "One row per observed compound-target pKi measurement from Script-04.",
            "pairwise_selectivity": "delta_pKi = pKi_A - pKi_B for all eligible within-compound kinase pairs.",
            "target_vs_panel": f"target_vs_panel_delta_pKi = target_pKi - {cfg.target_vs_panel_reference} off-target reference.",
            "classification": {
                "active_inactive": "Derived from pKi thresholds with gray-zone policy applied.",
                "strong_weak": "Derived from pKi thresholds with gray-zone policy applied.",
                "selective": "Derived from target_vs_panel_delta_pKi thresholds; highly selective is a subset of selective positives.",
            },
        },
        "class_balance_summaries": classification_diagnostics,
        "regression_summary_statistics": {
            "pKi": describe_numeric(regression_df["pKi"]) if "pKi" in regression_df.columns else {},
            "pairwise_delta_pKi": describe_numeric(pairwise_df["delta_pKi"]) if "delta_pKi" in pairwise_df.columns else {},
            "target_vs_panel_delta_pKi": describe_numeric(target_panel_df["target_vs_panel_delta_pKi"]) if "target_vs_panel_delta_pKi" in target_panel_df.columns else {},
        },
        "compounds_excluded_due_to_insufficient_multi_kinase_coverage": pairwise_diagnostics,
        "target_vs_panel_exclusion_summary": target_panel_diagnostics,
        "notes_on_gray_zone_handling": f"gray_zone_policy={cfg.gray_zone_policy}; save_only_labeled_classification_rows={cfg.save_only_labeled_classification_rows}",
        "notes_on_missing_metadata": warnings,
        "input_column_normalization": normalization_mapping,
        "task_summary_preview": summary_df.to_dict(orient="records"),
        "config_snapshot_reference": str(config_snapshot_path) if config_snapshot_path is not None else None,
        "run_timestamp_label": timestamp,
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg, loaded_config_path, loaded_config = load_config(args.config, project_root)
    ensure_output_dirs(cfg)
    log_file, timestamp = setup_logging(cfg.logs_dir)
    logging.info("Starting %s", SCRIPT_NAME)
    logging.info("Deterministic random seed fixed at %s", RANDOM_SEED)

    config_snapshot_path = save_config_snapshot(cfg, loaded_config_path)

    input_df_raw = load_required_dataframe(cfg.input_annotated_long_path, "annotated long-format dataset")
    input_df, normalization_warnings, normalization_mapping = standardize_input_dataframe(input_df_raw)

    warnings = list(normalization_warnings)
    activity_cliff_df = None
    if cfg.include_activity_cliff_flags_in_outputs and cfg.input_activity_cliff_path is not None:
        if cfg.input_activity_cliff_path.exists():
            activity_cliff_df = pd.read_csv(cfg.input_activity_cliff_path)
            cliff_summary = aggregate_activity_cliff_flags(activity_cliff_df)
            logging.info("Merged aggregated activity-cliff flags for %s compound-target rows.", len(cliff_summary))
            input_df = input_df.merge(
                cliff_summary,
                on=["compound_id", "target_chembl_id"],
                how="left",
                sort=False,
            )
        else:
            warning = f"Configured activity-cliff file was not found and cliff flags were skipped: {cfg.input_activity_cliff_path}"
            logging.warning(warning)
            warnings.append(warning)
    elif cfg.include_activity_cliff_flags_in_outputs:
        warning = "Activity-cliff flag propagation was requested but no activity-cliff input path was configured."
        logging.warning(warning)
        warnings.append(warning)

    for column in ACTIVITY_CLIFF_OUTPUT_COLUMNS:
        if column not in input_df.columns:
            input_df[column] = np.nan

    regression_df = pd.DataFrame()
    if cfg.build_multitask_regression_task:
        regression_df = build_multitask_regression_task(input_df, cfg)
        write_csv(regression_df, cfg.output_regression_long_path)
    else:
        logging.info("Multitask regression task generation disabled by config; writing empty placeholder file.")
        regression_df = pd.DataFrame(columns=["compound_id", "standardized_smiles", "target_chembl_id", "target_name", "pKi"])
        write_csv(regression_df, cfg.output_regression_long_path)

    pairwise_df = pd.DataFrame()
    pairwise_diagnostics: dict[str, Any] = {"eligible_compound_count": 0, "excluded_compound_count": 0, "excluded_compounds": []}
    if cfg.build_pairwise_selectivity_task:
        pairwise_df, pairwise_diagnostics = build_pairwise_selectivity_task(input_df, cfg)
        write_csv(pairwise_df, cfg.output_pairwise_selectivity_path)
    else:
        logging.info("Pairwise selectivity task generation disabled by config; writing empty placeholder file.")
        write_csv(pairwise_df, cfg.output_pairwise_selectivity_path)

    target_panel_df = pd.DataFrame()
    target_panel_diagnostics: dict[str, Any] = {"eligible_compound_count": 0, "excluded_compound_count": 0, "excluded_compounds": []}
    if cfg.build_target_vs_panel_task:
        target_panel_df, target_panel_diagnostics = build_target_vs_panel_task(input_df, cfg)
        write_csv(target_panel_df, cfg.output_target_vs_panel_path)
    else:
        logging.info("Target-vs-panel task generation disabled by config; writing empty placeholder file.")
        write_csv(target_panel_df, cfg.output_target_vs_panel_path)

    classification_df = pd.DataFrame()
    classification_diagnostics: dict[str, Any] = {}
    if cfg.build_classification_tasks:
        if regression_df.empty:
            raise ValueError("Classification task generation requires the multitask regression task to be available.")
        classification_df, classification_diagnostics = build_classification_task(regression_df, target_panel_df, cfg)
        write_csv(classification_df, cfg.output_classification_path)
    else:
        logging.info("Classification task generation disabled by config; writing empty placeholder file.")
        write_csv(classification_df, cfg.output_classification_path)

    summary_df = build_task_summary_table(regression_df, pairwise_df, target_panel_df, classification_df)
    write_csv(summary_df, cfg.output_task_summary_path)

    report = build_report(
        cfg=cfg,
        input_path=cfg.input_annotated_long_path,
        activity_cliff_path=cfg.input_activity_cliff_path,
        regression_df=regression_df,
        pairwise_df=pairwise_df,
        target_panel_df=target_panel_df,
        classification_df=classification_df,
        summary_df=summary_df,
        input_df=input_df,
        warnings=warnings,
        normalization_mapping=normalization_mapping,
        pairwise_diagnostics=pairwise_diagnostics,
        target_panel_diagnostics=target_panel_diagnostics,
        classification_diagnostics=classification_diagnostics,
        log_file=log_file,
        config_snapshot_path=config_snapshot_path,
        timestamp=timestamp,
    )
    with cfg.output_report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logging.info("Wrote JSON report to %s", cfg.output_report_path)
    logging.info(
        "Completed %s | regression_rows=%s | pairwise_rows=%s | target_vs_panel_rows=%s | classification_rows=%s",
        SCRIPT_NAME,
        len(regression_df),
        len(pairwise_df),
        len(target_panel_df),
        len(classification_df),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI error handling
        logging.exception("Fatal error during %s execution: %s", SCRIPT_NAME, exc)
        raise
