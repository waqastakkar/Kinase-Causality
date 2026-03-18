#!/usr/bin/env python3
"""Generate rigorous, reproducible benchmark split definitions for downstream tasks.

This script is a strict continuation of Script-05. It reads the task datasets
produced in Step-05 and generates deterministic benchmark split manifests for
random, scaffold, kinase-family, source/environment, activity-cliff, and
low-data evaluation settings.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

SCRIPT_NAME = "06_generate_benchmark_splits"
RANDOM_SEED = 2025

REQUIRED_SCRIPT_06_KEYS = {
    "input_regression_long_path",
    "input_pairwise_selectivity_path",
    "input_target_vs_panel_path",
    "input_classification_path",
    "input_activity_cliff_path",
    "output_split_manifest_path",
    "output_split_report_path",
    "random_seed",
    "n_folds",
    "build_random_split",
    "build_scaffold_split",
    "build_grouped_kinase_family_split",
    "build_source_environment_split",
    "build_activity_cliff_flagged_subsets",
    "build_low_data_subsets",
    "train_fraction",
    "valid_fraction",
    "test_fraction",
    "scaffold_column_candidates",
    "kinase_family_column_candidates",
    "source_environment_column_candidates",
    "activity_cliff_column_candidates",
    "min_examples_per_group",
    "low_data_subset_train_sizes",
    "save_row_level_split_assignments",
    "save_fold_level_split_assignments",
    "save_config_snapshot",
}

TASK_DEFINITIONS: dict[str, dict[str, Any]] = {
    "multitask_regression": {
        "config_key": "input_regression_long_path",
        "subdir": "multitask_regression",
        "required": ["target_chembl_id", "pKi"],
        "kinase_columns": ["target_chembl_id"],
        "default_task_column": "target_name",
        "supports_low_data": True,
        "response_columns": ["pKi"],
    },
    "pairwise_selectivity": {
        "config_key": "input_pairwise_selectivity_path",
        "subdir": "pairwise_selectivity",
        "required": ["kinase_a_chembl_id", "kinase_b_chembl_id", "delta_pKi"],
        "kinase_columns": ["kinase_a_chembl_id", "kinase_b_chembl_id"],
        "supports_low_data": False,
        "response_columns": ["delta_pKi", "abs_delta_pKi"],
    },
    "target_vs_panel": {
        "config_key": "input_target_vs_panel_path",
        "subdir": "target_vs_panel",
        "required": ["target_chembl_id", "target_vs_panel_delta_pKi"],
        "kinase_columns": ["target_chembl_id"],
        "default_task_column": "target_name",
        "supports_low_data": False,
        "response_columns": ["target_vs_panel_delta_pKi", "target_pKi", "offtarget_reference_pKi"],
    },
    "classification": {
        "config_key": "input_classification_path",
        "subdir": "classification",
        "required": [],
        "kinase_columns": ["target_chembl_id"],
        "default_task_column": "target_name",
        "supports_low_data": True,
        "response_columns": ["pKi", "target_vs_panel_delta_pKi"],
    },
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "compound_id": ("compound_id", "standardized_smiles"),
    "standardized_smiles": ("standardized_smiles",),
    "target_chembl_id": ("target_chembl_id",),
    "pKi": ("pKi", "median_pKi", "aggregated_pKi"),
    "delta_pKi": ("delta_pKi", "delta_pki"),
    "target_vs_panel_delta_pKi": ("target_vs_panel_delta_pKi", "target_vs_panel_delta_pki"),
    "active_inactive_label": ("active_inactive_label",),
    "strong_weak_label": ("strong_weak_label",),
    "selective_label": ("selective_label",),
}


@dataclass
class AppConfig:
    input_regression_long_path: Path
    input_pairwise_selectivity_path: Path
    input_target_vs_panel_path: Path
    input_classification_path: Path
    input_activity_cliff_path: Path | None
    output_split_manifest_path: Path
    output_split_report_path: Path
    random_seed: int
    n_folds: int
    build_random_split: bool
    build_scaffold_split: bool
    build_grouped_kinase_family_split: bool
    build_source_environment_split: bool
    build_activity_cliff_flagged_subsets: bool
    build_low_data_subsets: bool
    train_fraction: float
    valid_fraction: float
    test_fraction: float
    scaffold_column_candidates: list[str]
    kinase_family_column_candidates: list[str]
    source_environment_column_candidates: list[str]
    activity_cliff_column_candidates: list[str]
    min_examples_per_group: int
    low_data_subset_train_sizes: list[int]
    save_row_level_split_assignments: bool
    save_fold_level_split_assignments: bool
    save_config_snapshot: bool
    logs_dir: Path
    configs_used_dir: Path
    splits_root: Path

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        if not isinstance(raw, dict):
            raise ValueError("Config YAML must contain a top-level mapping/object.")
        script_cfg = raw.get("script_06")
        if not isinstance(script_cfg, dict):
            raise ValueError("Missing required `script_06` section in config.yaml.")

        missing = sorted(REQUIRED_SCRIPT_06_KEYS.difference(script_cfg))
        if missing:
            raise ValueError("Missing required script_06 config values: " + ", ".join(missing))

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
            raise ValueError(f"script_06.{key} must be a boolean; got {value!r}.")

        def parse_int(key: str, minimum: int = 0) -> int:
            value = script_cfg.get(key)
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"script_06.{key} must be an integer; got {value!r}.") from exc
            if parsed < minimum:
                raise ValueError(f"script_06.{key} must be >= {minimum}; got {parsed}.")
            return parsed

        def parse_float(key: str) -> float:
            value = script_cfg.get(key)
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"script_06.{key} must be numeric; got {value!r}.") from exc

        def parse_string_list(key: str) -> list[str]:
            value = script_cfg.get(key)
            if not isinstance(value, list) or not value:
                raise ValueError(f"script_06.{key} must be a non-empty list.")
            parsed = [str(item).strip() for item in value if str(item).strip()]
            if not parsed:
                raise ValueError(f"script_06.{key} must contain at least one non-empty value.")
            return parsed

        train_fraction = parse_float("train_fraction")
        valid_fraction = parse_float("valid_fraction")
        test_fraction = parse_float("test_fraction")
        total_fraction = train_fraction + valid_fraction + test_fraction
        if not math.isclose(total_fraction, 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError(
                "script_06 train/valid/test fractions must sum to 1.0; "
                f"got {total_fraction:.8f}."
            )

        n_folds = parse_int("n_folds", minimum=2)
        random_seed = parse_int("random_seed", minimum=0)
        low_data_sizes = sorted(set(parse_int_list(script_cfg.get("low_data_subset_train_sizes"), "low_data_subset_train_sizes", minimum=1)))

        output_split_manifest_path = resolve(script_cfg["output_split_manifest_path"])
        if output_split_manifest_path is None:
            raise ValueError("script_06.output_split_manifest_path must not be empty.")
        splits_root = output_split_manifest_path.parent

        return AppConfig(
            input_regression_long_path=resolve(script_cfg["input_regression_long_path"]),
            input_pairwise_selectivity_path=resolve(script_cfg["input_pairwise_selectivity_path"]),
            input_target_vs_panel_path=resolve(script_cfg["input_target_vs_panel_path"]),
            input_classification_path=resolve(script_cfg["input_classification_path"]),
            input_activity_cliff_path=resolve(script_cfg["input_activity_cliff_path"]),
            output_split_manifest_path=output_split_manifest_path,
            output_split_report_path=resolve(script_cfg["output_split_report_path"]),
            random_seed=random_seed,
            n_folds=n_folds,
            build_random_split=parse_bool(script_cfg["build_random_split"], "build_random_split"),
            build_scaffold_split=parse_bool(script_cfg["build_scaffold_split"], "build_scaffold_split"),
            build_grouped_kinase_family_split=parse_bool(
                script_cfg["build_grouped_kinase_family_split"], "build_grouped_kinase_family_split"
            ),
            build_source_environment_split=parse_bool(
                script_cfg["build_source_environment_split"], "build_source_environment_split"
            ),
            build_activity_cliff_flagged_subsets=parse_bool(
                script_cfg["build_activity_cliff_flagged_subsets"], "build_activity_cliff_flagged_subsets"
            ),
            build_low_data_subsets=parse_bool(script_cfg["build_low_data_subsets"], "build_low_data_subsets"),
            train_fraction=train_fraction,
            valid_fraction=valid_fraction,
            test_fraction=test_fraction,
            scaffold_column_candidates=parse_string_list("scaffold_column_candidates"),
            kinase_family_column_candidates=parse_string_list("kinase_family_column_candidates"),
            source_environment_column_candidates=parse_string_list("source_environment_column_candidates"),
            activity_cliff_column_candidates=parse_string_list("activity_cliff_column_candidates"),
            min_examples_per_group=parse_int("min_examples_per_group", minimum=1),
            low_data_subset_train_sizes=low_data_sizes,
            save_row_level_split_assignments=parse_bool(
                script_cfg["save_row_level_split_assignments"], "save_row_level_split_assignments"
            ),
            save_fold_level_split_assignments=parse_bool(
                script_cfg["save_fold_level_split_assignments"], "save_fold_level_split_assignments"
            ),
            save_config_snapshot=parse_bool(script_cfg["save_config_snapshot"], "save_config_snapshot"),
            logs_dir=resolve(raw.get("script_02", {}).get("logs_dir", "logs")),
            configs_used_dir=resolve(raw.get("script_02", {}).get("configs_used_dir", "configs_used")),
            splits_root=splits_root,
        )


def parse_int_list(value: Any, key: str, minimum: int = 0) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"script_06.{key} must be a non-empty list of integers.")
    parsed: list[int] = []
    for item in value:
        try:
            int_value = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"script_06.{key} must contain integers; got {item!r}.") from exc
        if int_value < minimum:
            raise ValueError(f"script_06.{key} values must be >= {minimum}; got {int_value}.")
        parsed.append(int_value)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic benchmark split definitions for Step-05 task tables.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to YAML config file (default: config.yaml).")
    return parser.parse_args()


def load_config(config_path: Path, project_root: Path) -> tuple[AppConfig, Path, dict[str, Any]]:
    resolved = config_path if config_path.is_absolute() else project_root / config_path
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return AppConfig.from_dict(raw, project_root), resolved, raw


def ensure_output_dirs(cfg: AppConfig) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.configs_used_dir.mkdir(parents=True, exist_ok=True)
    cfg.splits_root.mkdir(parents=True, exist_ok=True)
    cfg.output_split_report_path.parent.mkdir(parents=True, exist_ok=True)
    for task in TASK_DEFINITIONS.values():
        (cfg.splits_root / task["subdir"]).mkdir(parents=True, exist_ok=True)


def setup_logging(logs_dir: Path) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{SCRIPT_NAME}_{timestamp}.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
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


def resolve_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def canonicalize_compound_id(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    normalized = df.copy()
    warnings: list[str] = []
    if "compound_id" in normalized.columns:
        normalized["compound_id"] = normalized["compound_id"].astype(str)
        return normalized, warnings
    if "standardized_smiles" in normalized.columns:
        normalized["compound_id"] = normalized["standardized_smiles"].astype(str)
        warnings.append("`compound_id` missing; `standardized_smiles` was used as the canonical compound identifier.")
        return normalized, warnings
    raise ValueError("Dataset must contain either `compound_id` or `standardized_smiles`.")


def standardize_task_dataframe(task_name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], list[str], list[str]]:
    normalized, warnings = canonicalize_compound_id(df)
    column_mapping: dict[str, str] = {"compound_id": "compound_id"}

    if "standardized_smiles" in normalized.columns:
        normalized["standardized_smiles"] = normalized["standardized_smiles"].astype(str)
        column_mapping["standardized_smiles"] = "standardized_smiles"

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in normalized.columns:
            column_mapping.setdefault(canonical, canonical)
            continue
        if canonical == "compound_id":
            continue
        resolved = resolve_first_column(normalized, aliases)
        if resolved is not None and resolved != canonical:
            normalized = normalized.rename(columns={resolved: canonical})
            column_mapping[canonical] = resolved

    if task_name == "classification":
        label_columns = [col for col in ["active_inactive_label", "strong_weak_label", "selective_label"] if col in normalized.columns]
        if not label_columns:
            raise ValueError(
                "Classification task dataset must contain at least one classification label column: "
                "active_inactive_label, strong_weak_label, or selective_label."
            )
    required_columns = ["compound_id", *TASK_DEFINITIONS[task_name]["required"]]
    missing = [column for column in required_columns if column not in normalized.columns]
    if missing:
        raise ValueError(f"{task_name} dataset is missing required columns: {', '.join(missing)}")

    normalized = normalized.reset_index(drop=True).copy()
    normalized["row_index"] = normalized.index.astype(int)
    normalized["row_uid"] = normalized.apply(lambda row: build_row_uid(task_name, row), axis=1)
    return normalized, column_mapping, warnings, infer_label_columns(normalized)


def build_row_uid(task_name: str, row: pd.Series) -> str:
    if task_name == "pairwise_selectivity":
        return "|".join(
            [
                task_name,
                str(row["compound_id"]),
                str(row["kinase_a_chembl_id"]),
                str(row["kinase_b_chembl_id"]),
                str(row["row_index"]),
            ]
        )
    if task_name in {"multitask_regression", "target_vs_panel", "classification"}:
        target = row.get("target_chembl_id", "NA")
        return "|".join([task_name, str(row["compound_id"]), str(target), str(row["row_index"])])
    return "|".join([task_name, str(row["row_index"])])


def infer_label_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in ["active_inactive_label", "strong_weak_label", "selective_label"] if col in df.columns]


def load_required_dataframe(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required {description} not found: {path}")
    logging.info("Loading %s from %s", description, path)
    return pd.read_csv(path)


def stable_order(values: Iterable[Any]) -> list[str]:
    return [str(value) for value in sorted({str(v) for v in values}, key=lambda x: (x,))]


def compute_fraction_counts(total: int, train_fraction: float, valid_fraction: float) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    train = int(math.floor(total * train_fraction))
    valid = int(math.floor(total * valid_fraction))
    if total >= 3:
        train = max(train, 1)
        valid = max(valid, 1)
    if train + valid >= total:
        valid = max(1 if total >= 2 else 0, total - train - 1)
        train = max(1, total - valid - 1) if total >= 3 else max(0, total - valid)
    test = total - train - valid
    if total >= 3 and test <= 0:
        test = 1
        if train >= valid and train > 1:
            train -= 1
        elif valid > 1:
            valid -= 1
        else:
            train = max(train - 1, 1)
    return train, valid, total - train - valid


def assign_labels_from_order(items: list[str], train_fraction: float, valid_fraction: float) -> dict[str, str]:
    train_count, valid_count, test_count = compute_fraction_counts(len(items), train_fraction, valid_fraction)
    assignments: dict[str, str] = {}
    for idx, item in enumerate(items):
        if idx < train_count:
            assignments[item] = "train"
        elif idx < train_count + valid_count:
            assignments[item] = "valid"
        else:
            assignments[item] = "test"
    if len(items) != train_count + valid_count + test_count:
        raise AssertionError("Split counts do not sum to item count.")
    return assignments


def build_random_assignments(df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["row_uid"], kind="mergesort").reset_index(drop=True)
    rng = np.random.default_rng(cfg.random_seed)
    permutation = rng.permutation(len(ordered))
    randomized = ordered.iloc[permutation].reset_index(drop=True)
    row_assignment = assign_labels_from_order(randomized["row_uid"].tolist(), cfg.train_fraction, cfg.valid_fraction)
    random_assignments = ordered[["row_uid"]].copy()
    random_assignments["split_label"] = ordered["row_uid"].map(row_assignment)

    fold_records: list[dict[str, Any]] = []
    fold_assignment_frames: list[pd.DataFrame] = []
    fold_order = randomized["row_uid"].tolist()
    fold_bins = np.array_split(np.arange(len(fold_order)), cfg.n_folds)
    for fold_id, test_idx in enumerate(fold_bins, start=1):
        test_rows = [fold_order[i] for i in test_idx.tolist()]
        remaining = [row_uid for row_uid in fold_order if row_uid not in set(test_rows)]
        valid_target = 0.0 if not remaining else cfg.valid_fraction / (cfg.train_fraction + cfg.valid_fraction)
        valid_count = int(math.floor(len(remaining) * valid_target))
        if len(remaining) >= 2:
            valid_count = max(valid_count, 1)
        if valid_count >= len(remaining):
            valid_count = max(len(remaining) - 1, 0)
        valid_rows = remaining[:valid_count]
        train_rows = remaining[valid_count:]
        fold_records.append(
            {
                "fold_id": fold_id,
                "train_count": int(len(train_rows)),
                "valid_count": int(len(valid_rows)),
                "test_count": int(len(test_rows)),
                "n_rows": int(len(ordered)),
            }
        )
        mask_train = ordered["row_uid"].isin(train_rows)
        mask_valid = ordered["row_uid"].isin(valid_rows)
        mask_test = ordered["row_uid"].isin(test_rows)
        current = pd.DataFrame({
            "row_uid": ordered["row_uid"],
            "fold_id": fold_id,
            "split_label": np.select([mask_train, mask_valid, mask_test], ["train", "valid", "test"], default=pd.NA),
        })
        fold_assignment_frames.append(current)
    fold_assignments = pd.concat(fold_assignment_frames, ignore_index=True)
    return random_assignments, fold_assignments, pd.DataFrame(fold_records)


def deterministic_group_assignments(group_keys: list[str], cfg: AppConfig, seed_offset: int = 0) -> dict[str, str]:
    rng = np.random.default_rng(cfg.random_seed + seed_offset)
    ordered = np.array(sorted(group_keys, key=lambda x: (x,)))
    shuffled = ordered[rng.permutation(len(ordered))].tolist()
    return assign_labels_from_order(shuffled, cfg.train_fraction, cfg.valid_fraction)


def build_group_split(df: pd.DataFrame, grouping_series: pd.Series, cfg: AppConfig, seed_offset: int, grouping_variable: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temp = df[["row_uid"]].copy()
    temp["group_key"] = grouping_series.fillna("MISSING").astype(str)
    group_counts = temp.groupby("group_key", dropna=False).size().rename("row_count").reset_index()
    eligible_group_count = int((group_counts["row_count"] >= cfg.min_examples_per_group).sum())
    if eligible_group_count < 3:
        raise ValueError(
            f"`{grouping_variable}` has only {eligible_group_count} groups with at least "
            f"{cfg.min_examples_per_group} examples; need at least 3 for train/valid/test splits."
        )
    assignments = deterministic_group_assignments(group_counts["group_key"].tolist(), cfg, seed_offset=seed_offset)
    temp["split_label"] = temp["group_key"].map(assignments)
    leakage = temp.groupby("group_key")["split_label"].nunique().max()
    if int(leakage) > 1:
        raise ValueError(f"Detected leakage across grouped split `{grouping_variable}`.")
    summary = group_counts.copy()
    summary["grouping_variable"] = grouping_variable
    summary["split_label"] = summary["group_key"].map(assignments)
    return temp[["row_uid", "split_label"]], pd.DataFrame(), summary


def pick_candidate_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def resolve_kinase_family_grouping(df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.Series, str] | tuple[None, None]:
    if {"kinase_a_kinase_family", "kinase_b_kinase_family"}.issubset(df.columns):
        series = df["kinase_a_kinase_family"].fillna("MISSING") + "__" + df["kinase_b_kinase_family"].fillna("MISSING")
        return series, "kinase_a_kinase_family+kinase_b_kinase_family"
    if {"kinase_a_family", "kinase_b_family"}.issubset(df.columns):
        series = df["kinase_a_family"].fillna("MISSING") + "__" + df["kinase_b_family"].fillna("MISSING")
        return series, "kinase_a_family+kinase_b_family"
    candidate = pick_candidate_column(df, cfg.kinase_family_column_candidates)
    if candidate is None:
        if {"kinase_a_kinase_family", "kinase_b_kinase_family"}.intersection(df.columns):
            parts = []
            for side in ["kinase_a", "kinase_b"]:
                side_candidate = pick_candidate_column(df, [f"{side}_{col}" for col in cfg.kinase_family_column_candidates])
                if side_candidate is not None:
                    parts.append(df[side_candidate].fillna("MISSING").astype(str))
            if len(parts) == 2:
                return parts[0] + "__" + parts[1], "pairwise_kinase_family_candidates"
        return None, None
    return df[candidate].astype(str), candidate


def resolve_source_grouping(df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.Series, str] | tuple[None, None]:
    candidate = pick_candidate_column(df, cfg.source_environment_column_candidates)
    if candidate is None:
        candidate = pick_candidate_column(df, [f"compound_{col}" for col in cfg.source_environment_column_candidates])
    if candidate is None:
        return None, None
    return df[candidate].fillna("MISSING").astype(str), candidate


def resolve_scaffold_grouping(df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.Series, str] | tuple[None, None]:
    candidate = pick_candidate_column(df, cfg.scaffold_column_candidates)
    if candidate is None:
        candidate = pick_candidate_column(df, [f"compound_{col}" for col in cfg.scaffold_column_candidates])
    if candidate is None:
        return None, None
    if candidate in {"murcko_scaffold", "scaffold", "generic_murcko_scaffold", "generic_scaffold"}:
        per_compound = (
            df[["compound_id", candidate]]
            .drop_duplicates()
            .groupby("compound_id", dropna=False)[candidate]
            .agg(lambda values: stable_order(values)[0] if len(set(map(str, values))) >= 1 else "MISSING")
        )
        if per_compound.index.nunique() != df["compound_id"].nunique():
            raise ValueError(f"Could not resolve a unique scaffold mapping for all compounds using column `{candidate}`.")
        return df["compound_id"].map(per_compound).fillna("MISSING").astype(str), candidate
    return df[candidate].fillna("MISSING").astype(str), candidate


def label_activity_cliff_subsets(task_name: str, df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame | None, str | None]:
    cliff_candidate = pick_candidate_column(df, cfg.activity_cliff_column_candidates)
    if cliff_candidate is not None:
        cliff_values = df[cliff_candidate].fillna(0)
        cliff_numeric = pd.to_numeric(cliff_values, errors="coerce").fillna(0)
        cliff_bool = cliff_values.astype(str).str.lower().isin({"1", "true", "yes"}) | cliff_numeric.gt(0)
    elif task_name == "pairwise_selectivity":
        pair_candidates = [
            col for col in df.columns if "activity_cliff_flag" in col or col in {"kinase_a_has_activity_cliff_partner_for_target", "kinase_b_has_activity_cliff_partner_for_target"}
        ]
        if not pair_candidates:
            return None, None
        cliff_bool = pd.Series(False, index=df.index)
        for col in pair_candidates:
            values = df[col].fillna(0)
            cliff_bool = cliff_bool | values.astype(str).str.lower().isin({"1", "true", "yes"})
        cliff_candidate = "+".join(pair_candidates)
    else:
        return None, None
    subset_df = df[["row_uid"]].copy()
    subset_df["subset_name"] = np.where(cliff_bool, "cliff_associated", "non_cliff")
    subset_df["activity_cliff_flag"] = cliff_bool.astype(int)
    return subset_df, cliff_candidate


def summarize_regression_distribution(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for col in columns:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if numeric.empty:
            continue
        summary[col] = {
            "count": int(numeric.shape[0]),
            "mean": float(numeric.mean()),
            "std": float(numeric.std(ddof=0)),
            "min": float(numeric.min()),
            "median": float(numeric.median()),
            "max": float(numeric.max()),
        }
    return summary


def summarize_class_balance(df: pd.DataFrame, label_columns: list[str]) -> dict[str, Any]:
    return {column: {str(k): int(v) for k, v in df[column].value_counts(dropna=False).to_dict().items()} for column in label_columns}


def count_unique_kinases(df: pd.DataFrame, task_name: str) -> int:
    columns = TASK_DEFINITIONS[task_name]["kinase_columns"]
    unique_values: set[str] = set()
    for column in columns:
        if column in df.columns:
            unique_values.update(df[column].dropna().astype(str).tolist())
    return int(len(unique_values))


def build_assignment_output(df: pd.DataFrame, base_assignments: pd.DataFrame, task_name: str, split_strategy: str, split_id: str, fold_id: str | None = None) -> pd.DataFrame:
    merged = df.merge(base_assignments, on="row_uid", how="left", sort=False)
    for column in ["task_name", "split_strategy", "split_id"]:
        if column in merged.columns:
            merged = merged.drop(columns=[column])
    if "fold_id" in merged.columns and fold_id is not None:
        pass
    elif "fold_id" in merged.columns:
        merged = merged.drop(columns=["fold_id"])
    merged.insert(0, "task_name", task_name)
    merged.insert(1, "split_strategy", split_strategy)
    merged.insert(2, "split_id", split_id)
    if "fold_id" not in merged.columns:
        merged.insert(3, "fold_id", fold_id if fold_id is not None else "holdout")
    return merged


def build_low_data_subsets(df: pd.DataFrame, holdout_assignments: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = df[["row_uid"]].merge(holdout_assignments, on="row_uid", how="left", sort=False)
    train_rows = merged[merged["split_label"] == "train"]["row_uid"].sort_values(kind="mergesort").tolist()
    subset_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    for size in cfg.low_data_subset_train_sizes:
        actual_size = min(size, len(train_rows))
        selected = train_rows[:actual_size]
        selected_set = set(selected)
        for row_uid in merged["row_uid"]:
            if row_uid in selected_set:
                subset_label = f"train_subset_{size}"
            else:
                original = merged.loc[merged["row_uid"] == row_uid, "split_label"].iloc[0]
                subset_label = original
            subset_records.append({"row_uid": row_uid, "split_label": subset_label, "subset_train_size": size})
        summary_records.append({"subset_train_size": size, "requested_train_size": int(size), "actual_train_size": int(actual_size), "base_train_size": int(len(train_rows))})
    return pd.DataFrame(subset_records), pd.DataFrame(summary_records)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info("Wrote %s rows to %s", len(df), path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg, loaded_config_path, _ = load_config(args.config, project_root)
    ensure_output_dirs(cfg)
    log_file, timestamp = setup_logging(cfg.logs_dir)
    logging.info("Starting %s", SCRIPT_NAME)
    logging.info("Deterministic random seed fixed at %s", cfg.random_seed)

    config_snapshot_path = save_config_snapshot(cfg, loaded_config_path)
    warnings: list[str] = []
    manifest_records: list[dict[str, Any]] = []
    report_tasks: dict[str, Any] = {}

    for task_name, spec in TASK_DEFINITIONS.items():
        input_path = getattr(cfg, spec["config_key"])
        task_subdir = cfg.splits_root / spec["subdir"]
        raw_df = load_required_dataframe(input_path, f"{task_name} task dataset")
        df, column_mapping, task_warnings, label_columns = standardize_task_dataframe(task_name, raw_df)
        warnings.extend([f"{task_name}: {warning}" for warning in task_warnings])
        logging.info("Validated %s with %s rows, %s unique compounds, %s unique kinases.", task_name, len(df), df["compound_id"].nunique(), count_unique_kinases(df, task_name))

        task_report: dict[str, Any] = {
            "input_dataset_path": str(input_path),
            "n_rows": int(len(df)),
            "n_unique_compounds": int(df["compound_id"].nunique()),
            "n_unique_kinases": count_unique_kinases(df, task_name),
            "column_mapping": column_mapping,
            "label_columns": label_columns,
            "split_outputs": {},
            "selected_grouping_columns": {},
            "regression_response_summary": summarize_regression_distribution(df, spec.get("response_columns", [])),
            "classification_balance_summary": summarize_class_balance(df, label_columns),
        }

        if cfg.build_random_split:
            base_assignments, fold_assignments, fold_summary = build_random_assignments(df, cfg)
            assignment_df = build_assignment_output(df, base_assignments, task_name, "random", "random_seeded_holdout")
            assignment_path = task_subdir / "random_split_assignments.csv"
            if cfg.save_row_level_split_assignments:
                write_csv(assignment_df, assignment_path)
            else:
                assignment_path = Path("")
            fold_path = task_subdir / "random_fold_summary.csv"
            fold_assignment_path = task_subdir / "random_fold_assignments.csv"
            if cfg.save_fold_level_split_assignments:
                write_csv(build_assignment_output(df, fold_assignments, task_name, "random", "random_kfold"), fold_assignment_path)
                write_csv(fold_summary, fold_path)
            else:
                fold_path = Path("")
                fold_assignment_path = Path("")
            split_counts = assignment_df["split_label"].value_counts().to_dict()
            manifest_records.append({
                "task_name": task_name,
                "split_strategy": "random",
                "split_id": "random_seeded_holdout",
                "fold_id": "holdout",
                "input_dataset_path": str(input_path),
                "output_assignment_path": str(assignment_path) if str(assignment_path) else None,
                "grouping_variable_used": "row_uid",
                "number_of_rows": int(len(df)),
                "number_of_unique_compounds": int(df["compound_id"].nunique()),
                "number_of_unique_kinases": count_unique_kinases(df, task_name),
                "train_count": int(split_counts.get("train", 0)),
                "valid_count": int(split_counts.get("valid", 0)),
                "test_count": int(split_counts.get("test", 0)),
                "notes": f"Deterministic row-wise random split using seed {cfg.random_seed}.",
            })
            task_report["split_outputs"]["random"] = {
                "row_assignment_path": str(assignment_path) if str(assignment_path) else None,
                "fold_assignment_path": str(fold_assignment_path) if str(fold_assignment_path) else None,
                "fold_summary_path": str(fold_path) if str(fold_path) else None,
                "counts": {str(k): int(v) for k, v in split_counts.items()},
            }
            base_holdout_assignments = base_assignments
        else:
            base_holdout_assignments = pd.DataFrame(columns=["row_uid", "split_label"])

        if cfg.build_scaffold_split:
            scaffold_series, scaffold_column = resolve_scaffold_grouping(df, cfg)
            if scaffold_series is None:
                warning = f"{task_name}: scaffold split skipped because none of the scaffold column candidates were found."
                warnings.append(warning)
                logging.warning(warning)
            else:
                try:
                    grouped_assignments, _, group_summary = build_group_split(df, scaffold_series, cfg, seed_offset=11, grouping_variable=scaffold_column)
                except ValueError as exc:
                    warning = f"{task_name}: scaffold split skipped because {exc}"
                    warnings.append(warning)
                    logging.warning(warning)
                    grouped_assignments = None
                if grouped_assignments is not None:
                    assignment_df = build_assignment_output(df, grouped_assignments, task_name, "scaffold", "scaffold_group_holdout")
                    assignment_path = task_subdir / "scaffold_split_assignments.csv"
                    group_summary_path = task_subdir / "scaffold_group_summary.csv"
                    if cfg.save_row_level_split_assignments:
                        write_csv(assignment_df, assignment_path)
                    else:
                        assignment_path = Path("")
                    write_csv(group_summary, group_summary_path)
                    split_counts = assignment_df["split_label"].value_counts().to_dict()
                    manifest_records.append({
                        "task_name": task_name,
                        "split_strategy": "scaffold",
                        "split_id": "scaffold_group_holdout",
                        "fold_id": "holdout",
                        "input_dataset_path": str(input_path),
                        "output_assignment_path": str(assignment_path) if str(assignment_path) else None,
                        "grouping_variable_used": scaffold_column,
                        "number_of_rows": int(len(df)),
                        "number_of_unique_compounds": int(df["compound_id"].nunique()),
                        "number_of_unique_kinases": count_unique_kinases(df, task_name),
                        "train_count": int(split_counts.get("train", 0)),
                        "valid_count": int(split_counts.get("valid", 0)),
                        "test_count": int(split_counts.get("test", 0)),
                        "notes": "Whole scaffold groups assigned to train/valid/test with deterministic grouping.",
                    })
                    task_report["selected_grouping_columns"]["scaffold"] = scaffold_column
                    task_report["split_outputs"]["scaffold"] = {
                        "row_assignment_path": str(assignment_path) if str(assignment_path) else None,
                        "group_summary_path": str(group_summary_path),
                        "counts": {str(k): int(v) for k, v in split_counts.items()},
                        "n_scaffold_groups": int(group_summary["group_key"].nunique()),
                    }

        if cfg.build_grouped_kinase_family_split:
            family_series, family_column = resolve_kinase_family_grouping(df, cfg)
            if family_series is None:
                warning = f"{task_name}: kinase-family grouped split skipped because no usable kinase family column was found."
                warnings.append(warning)
                logging.warning(warning)
            else:
                try:
                    grouped_assignments, _, group_summary = build_group_split(df, family_series, cfg, seed_offset=23, grouping_variable=family_column)
                except ValueError as exc:
                    warning = f"{task_name}: kinase-family split skipped because {exc}"
                    warnings.append(warning)
                    logging.warning(warning)
                    grouped_assignments = None
                if grouped_assignments is not None:
                    assignment_df = build_assignment_output(df, grouped_assignments, task_name, "kinase_family_grouped", "kinase_family_holdout")
                    assignment_path = task_subdir / "kinase_family_split_assignments.csv"
                    group_summary_path = task_subdir / "kinase_family_group_summary.csv"
                    if cfg.save_row_level_split_assignments:
                        write_csv(assignment_df, assignment_path)
                    else:
                        assignment_path = Path("")
                    write_csv(group_summary, group_summary_path)
                    split_counts = assignment_df["split_label"].value_counts().to_dict()
                    manifest_records.append({
                        "task_name": task_name,
                        "split_strategy": "kinase_family_grouped",
                        "split_id": "kinase_family_holdout",
                        "fold_id": "holdout",
                        "input_dataset_path": str(input_path),
                        "output_assignment_path": str(assignment_path) if str(assignment_path) else None,
                        "grouping_variable_used": family_column,
                        "number_of_rows": int(len(df)),
                        "number_of_unique_compounds": int(df["compound_id"].nunique()),
                        "number_of_unique_kinases": count_unique_kinases(df, task_name),
                        "train_count": int(split_counts.get("train", 0)),
                        "valid_count": int(split_counts.get("valid", 0)),
                        "test_count": int(split_counts.get("test", 0)),
                        "notes": "Whole kinase-family groups held out for family transfer benchmarking.",
                    })
                    task_report["selected_grouping_columns"]["kinase_family"] = family_column
                    task_report["split_outputs"]["kinase_family_grouped"] = {
                        "row_assignment_path": str(assignment_path) if str(assignment_path) else None,
                        "group_summary_path": str(group_summary_path),
                        "counts": {str(k): int(v) for k, v in split_counts.items()},
                        "n_family_groups": int(group_summary["group_key"].nunique()),
                    }

        if cfg.build_source_environment_split:
            source_series, source_column = resolve_source_grouping(df, cfg)
            if source_series is None:
                warning = f"{task_name}: source/environment grouped split skipped because no configured source column was found."
                warnings.append(warning)
                logging.warning(warning)
            else:
                try:
                    grouped_assignments, _, group_summary = build_group_split(df, source_series, cfg, seed_offset=37, grouping_variable=source_column)
                except ValueError as exc:
                    warning = f"{task_name}: source/environment split skipped because {exc}"
                    warnings.append(warning)
                    logging.warning(warning)
                    grouped_assignments = None
                if grouped_assignments is not None:
                    assignment_df = build_assignment_output(df, grouped_assignments, task_name, "source_environment_grouped", "source_environment_holdout")
                    assignment_path = task_subdir / "source_environment_split_assignments.csv"
                    group_summary_path = task_subdir / "source_environment_group_summary.csv"
                    if cfg.save_row_level_split_assignments:
                        write_csv(assignment_df, assignment_path)
                    else:
                        assignment_path = Path("")
                    write_csv(group_summary, group_summary_path)
                    split_counts = assignment_df["split_label"].value_counts().to_dict()
                    manifest_records.append({
                        "task_name": task_name,
                        "split_strategy": "source_environment_grouped",
                        "split_id": "source_environment_holdout",
                        "fold_id": "holdout",
                        "input_dataset_path": str(input_path),
                        "output_assignment_path": str(assignment_path) if str(assignment_path) else None,
                        "grouping_variable_used": source_column,
                        "number_of_rows": int(len(df)),
                        "number_of_unique_compounds": int(df["compound_id"].nunique()),
                        "number_of_unique_kinases": count_unique_kinases(df, task_name),
                        "train_count": int(split_counts.get("train", 0)),
                        "valid_count": int(split_counts.get("valid", 0)),
                        "test_count": int(split_counts.get("test", 0)),
                        "notes": "Whole provenance/source groups held out for environment-shift benchmarking.",
                    })
                    task_report["selected_grouping_columns"]["source_environment"] = source_column
                    task_report["split_outputs"]["source_environment_grouped"] = {
                        "row_assignment_path": str(assignment_path) if str(assignment_path) else None,
                        "group_summary_path": str(group_summary_path),
                        "counts": {str(k): int(v) for k, v in split_counts.items()},
                        "n_source_groups": int(group_summary["group_key"].nunique()),
                    }

        if cfg.build_activity_cliff_flagged_subsets:
            subset_df, subset_column = label_activity_cliff_subsets(task_name, df, cfg)
            if subset_df is None:
                warning = f"{task_name}: activity-cliff subset manifest skipped because no activity-cliff flag columns were found."
                warnings.append(warning)
                logging.warning(warning)
            else:
                subset_path = task_subdir / "activity_cliff_subset_manifest.csv"
                write_csv(subset_df, subset_path)
                cliff_counts = subset_df["subset_name"].value_counts().to_dict()
                task_report["selected_grouping_columns"]["activity_cliff"] = subset_column
                task_report["split_outputs"]["activity_cliff_subsets"] = {
                    "subset_manifest_path": str(subset_path),
                    "counts": {str(k): int(v) for k, v in cliff_counts.items()},
                }

        if cfg.build_low_data_subsets and spec.get("supports_low_data") and not base_holdout_assignments.empty:
            low_data_df, low_data_summary = build_low_data_subsets(df, base_holdout_assignments, cfg)
            subset_path = task_subdir / "low_data_subset_assignments.csv"
            summary_path = task_subdir / "low_data_subset_summary.csv"
            write_csv(low_data_df, subset_path)
            write_csv(low_data_summary, summary_path)
            task_report["split_outputs"]["low_data_subsets"] = {
                "assignment_path": str(subset_path),
                "summary_path": str(summary_path),
                "subset_counts": low_data_summary.to_dict(orient="records"),
            }

        report_tasks[task_name] = task_report

    manifest_df = pd.DataFrame(manifest_records).sort_values(["task_name", "split_strategy", "split_id"], kind="mergesort").reset_index(drop=True)
    write_csv(manifest_df, cfg.output_split_manifest_path)

    report = {
        "script_name": SCRIPT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": cfg.random_seed,
        "input_file_paths": {spec["config_key"]: str(getattr(cfg, spec["config_key"])) for spec in TASK_DEFINITIONS.values()},
        "optional_input_file_paths": {"input_activity_cliff_path": str(cfg.input_activity_cliff_path) if cfg.input_activity_cliff_path is not None else None},
        "output_file_paths": {
            "split_manifest": str(cfg.output_split_manifest_path),
            "report": str(cfg.output_split_report_path),
            "log_file": str(log_file),
            "config_snapshot": str(config_snapshot_path) if config_snapshot_path is not None else None,
        },
        "split_strategies_attempted": {
            "random": cfg.build_random_split,
            "scaffold": cfg.build_scaffold_split,
            "kinase_family_grouped": cfg.build_grouped_kinase_family_split,
            "source_environment_grouped": cfg.build_source_environment_split,
            "activity_cliff_flagged_subsets": cfg.build_activity_cliff_flagged_subsets,
            "low_data_subsets": cfg.build_low_data_subsets,
        },
        "tasks_processed": list(report_tasks.keys()),
        "task_summaries": report_tasks,
        "scaffold_split_coverage_summary": {task: summary.get("split_outputs", {}).get("scaffold", {}) for task, summary in report_tasks.items()},
        "kinase_family_split_coverage_summary": {task: summary.get("split_outputs", {}).get("kinase_family_grouped", {}) for task, summary in report_tasks.items()},
        "source_environment_split_coverage_summary": {task: summary.get("split_outputs", {}).get("source_environment_grouped", {}) for task, summary in report_tasks.items()},
        "activity_cliff_subset_counts": {task: summary.get("split_outputs", {}).get("activity_cliff_subsets", {}).get("counts", {}) for task, summary in report_tasks.items()},
        "low_data_subset_counts": {task: summary.get("split_outputs", {}).get("low_data_subsets", {}).get("subset_counts", []) for task, summary in report_tasks.items()},
        "warnings": warnings,
        "config_snapshot_reference": str(config_snapshot_path) if config_snapshot_path is not None else None,
    }
    cfg.output_split_report_path.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    logging.info("Wrote JSON report to %s", cfg.output_split_report_path)
    logging.info("Finished %s successfully.", SCRIPT_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
