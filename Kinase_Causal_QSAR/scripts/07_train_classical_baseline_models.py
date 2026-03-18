#!/usr/bin/env python3
"""Train and evaluate classical baseline models for Step-05 tasks using Step-06 splits.

This script is a strict continuation of Script-06. It builds deterministic,
config-driven classical machine-learning baselines for multitask pKi
regression, pairwise selectivity regression, target-vs-panel regression, and
optional derived classification tasks.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None
    LGBMRegressor = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

SCRIPT_NAME = "07_train_classical_baseline_models"
RANDOM_SEED = 2025

REQUIRED_SCRIPT_07_KEYS = {
    "input_regression_long_path",
    "input_pairwise_selectivity_path",
    "input_target_vs_panel_path",
    "input_classification_path",
    "input_split_manifest_path",
    "output_model_root",
    "output_metrics_root",
    "output_predictions_root",
    "output_figures_root",
    "output_report_path",
    "random_seed",
    "n_jobs",
    "run_multitask_regression",
    "run_pairwise_selectivity_regression",
    "run_target_vs_panel_regression",
    "run_classification_tasks",
    "descriptor_type",
    "use_morgan_fingerprints",
    "morgan_radius",
    "morgan_nbits",
    "include_rdkit_2d_descriptors",
    "include_environment_features",
    "regression_models",
    "classification_models",
    "multitask_strategy",
    "pairwise_strategy",
    "target_vs_panel_strategy",
    "save_trained_models",
    "save_feature_tables",
    "save_fold_predictions",
    "save_test_predictions",
    "save_error_tables",
    "save_config_snapshot",
    "make_figures",
    "export_svg",
    "export_png",
    "export_pdf",
    "figure_style",
    "tuning",
}

REGRESSION_TASKS: dict[str, dict[str, Any]] = {
    "multitask_regression": {
        "config_key": "input_regression_long_path",
        "target_column": "pKi",
        "required_columns": ["target_chembl_id", "pKi"],
    },
    "pairwise_selectivity": {
        "config_key": "input_pairwise_selectivity_path",
        "target_column": "delta_pKi",
        "required_columns": ["kinase_a_chembl_id", "kinase_b_chembl_id", "delta_pKi"],
    },
    "target_vs_panel": {
        "config_key": "input_target_vs_panel_path",
        "target_column": "target_vs_panel_delta_pKi",
        "required_columns": ["target_chembl_id", "target_vs_panel_delta_pKi"],
    },
}
CLASSIFICATION_TASK_NAME = "classification"
LABEL_PRIORITY = ["active_inactive_label", "strong_weak_label", "selective_label", "highly_selective_label"]

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "compound_id": ("compound_id", "standardized_smiles"),
    "standardized_smiles": ("standardized_smiles",),
    "target_chembl_id": ("target_chembl_id",),
    "target_name": ("target_name", "pref_name"),
    "pKi": ("pKi", "median_pKi", "aggregated_pKi"),
    "delta_pKi": ("delta_pKi", "delta_pki"),
    "target_vs_panel_delta_pKi": ("target_vs_panel_delta_pKi", "target_vs_panel_delta_pki"),
}

ENVIRONMENT_CANDIDATES = [
    "kinase_family",
    "kinase_family_broad_group",
    "kinase_subfamily",
    "source_id",
    "source_description",
    "doc_id",
    "source_frequency_bin",
    "document_frequency_bin",
    "murcko_scaffold",
    "generic_murcko_scaffold",
    "activity_cliff_flag",
    "has_activity_cliff_partner_for_target",
]


@dataclass
class FigureStyle:
    font_family: str
    bold_text: bool
    output_format_primary: str
    palette_name: str
    dpi_png: int


@dataclass
class TuningConfig:
    use_small_grid_search: bool
    cv_folds_inner: int


@dataclass
class AppConfig:
    input_regression_long_path: Path
    input_pairwise_selectivity_path: Path
    input_target_vs_panel_path: Path
    input_classification_path: Path
    input_split_manifest_path: Path
    output_model_root: Path
    output_metrics_root: Path
    output_predictions_root: Path
    output_figures_root: Path
    output_report_path: Path
    random_seed: int
    n_jobs: int
    run_multitask_regression: bool
    run_pairwise_selectivity_regression: bool
    run_target_vs_panel_regression: bool
    run_classification_tasks: bool
    descriptor_type: str
    use_morgan_fingerprints: bool
    morgan_radius: int
    morgan_nbits: int
    include_rdkit_2d_descriptors: bool
    include_environment_features: bool
    regression_models: list[str]
    classification_models: list[str]
    multitask_strategy: str
    pairwise_strategy: str
    target_vs_panel_strategy: str
    save_trained_models: bool
    save_feature_tables: bool
    save_fold_predictions: bool
    save_test_predictions: bool
    save_error_tables: bool
    save_config_snapshot: bool
    make_figures: bool
    export_svg: bool
    export_png: bool
    export_pdf: bool
    figure_style: FigureStyle
    tuning: TuningConfig
    logs_dir: Path
    configs_used_dir: Path
    report_root: Path

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        if not isinstance(raw, dict):
            raise ValueError("Config YAML must contain a top-level mapping/object.")
        script_cfg = raw.get("script_07")
        if not isinstance(script_cfg, dict):
            raise ValueError("Missing required `script_07` section in config.yaml.")
        missing = sorted(REQUIRED_SCRIPT_07_KEYS.difference(script_cfg))
        if missing:
            raise ValueError("Missing required script_07 config values: " + ", ".join(missing))

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
            raise ValueError(f"script_07.{key} must be a boolean; got {value!r}.")

        def parse_int(key: str, minimum: int | None = None) -> int:
            value = script_cfg.get(key)
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"script_07.{key} must be an integer; got {value!r}.") from exc
            if minimum is not None and parsed < minimum:
                raise ValueError(f"script_07.{key} must be >= {minimum}; got {parsed}.")
            return parsed

        def parse_list(key: str) -> list[str]:
            value = script_cfg.get(key)
            if not isinstance(value, list) or not value:
                raise ValueError(f"script_07.{key} must be a non-empty list.")
            return [str(item).strip() for item in value if str(item).strip()]

        figure_style_raw = script_cfg.get("figure_style")
        if not isinstance(figure_style_raw, dict):
            raise ValueError("script_07.figure_style must be a mapping.")
        tuning_raw = script_cfg.get("tuning")
        if not isinstance(tuning_raw, dict):
            raise ValueError("script_07.tuning must be a mapping.")

        return AppConfig(
            input_regression_long_path=resolve(script_cfg["input_regression_long_path"]),
            input_pairwise_selectivity_path=resolve(script_cfg["input_pairwise_selectivity_path"]),
            input_target_vs_panel_path=resolve(script_cfg["input_target_vs_panel_path"]),
            input_classification_path=resolve(script_cfg["input_classification_path"]),
            input_split_manifest_path=resolve(script_cfg["input_split_manifest_path"]),
            output_model_root=resolve(script_cfg["output_model_root"]),
            output_metrics_root=resolve(script_cfg["output_metrics_root"]),
            output_predictions_root=resolve(script_cfg["output_predictions_root"]),
            output_figures_root=resolve(script_cfg["output_figures_root"]),
            output_report_path=resolve(script_cfg["output_report_path"]),
            random_seed=parse_int("random_seed", minimum=0),
            n_jobs=parse_int("n_jobs"),
            run_multitask_regression=parse_bool(script_cfg["run_multitask_regression"], "run_multitask_regression"),
            run_pairwise_selectivity_regression=parse_bool(script_cfg["run_pairwise_selectivity_regression"], "run_pairwise_selectivity_regression"),
            run_target_vs_panel_regression=parse_bool(script_cfg["run_target_vs_panel_regression"], "run_target_vs_panel_regression"),
            run_classification_tasks=parse_bool(script_cfg["run_classification_tasks"], "run_classification_tasks"),
            descriptor_type=str(script_cfg["descriptor_type"]).strip(),
            use_morgan_fingerprints=parse_bool(script_cfg["use_morgan_fingerprints"], "use_morgan_fingerprints"),
            morgan_radius=parse_int("morgan_radius", minimum=1),
            morgan_nbits=parse_int("morgan_nbits", minimum=8),
            include_rdkit_2d_descriptors=parse_bool(script_cfg["include_rdkit_2d_descriptors"], "include_rdkit_2d_descriptors"),
            include_environment_features=parse_bool(script_cfg["include_environment_features"], "include_environment_features"),
            regression_models=parse_list("regression_models"),
            classification_models=parse_list("classification_models"),
            multitask_strategy=str(script_cfg["multitask_strategy"]).strip(),
            pairwise_strategy=str(script_cfg["pairwise_strategy"]).strip(),
            target_vs_panel_strategy=str(script_cfg["target_vs_panel_strategy"]).strip(),
            save_trained_models=parse_bool(script_cfg["save_trained_models"], "save_trained_models"),
            save_feature_tables=parse_bool(script_cfg["save_feature_tables"], "save_feature_tables"),
            save_fold_predictions=parse_bool(script_cfg["save_fold_predictions"], "save_fold_predictions"),
            save_test_predictions=parse_bool(script_cfg["save_test_predictions"], "save_test_predictions"),
            save_error_tables=parse_bool(script_cfg["save_error_tables"], "save_error_tables"),
            save_config_snapshot=parse_bool(script_cfg["save_config_snapshot"], "save_config_snapshot"),
            make_figures=parse_bool(script_cfg["make_figures"], "make_figures"),
            export_svg=parse_bool(script_cfg["export_svg"], "export_svg"),
            export_png=parse_bool(script_cfg["export_png"], "export_png"),
            export_pdf=parse_bool(script_cfg["export_pdf"], "export_pdf"),
            figure_style=FigureStyle(
                font_family=str(figure_style_raw.get("font_family", "Times New Roman")),
                bold_text=parse_bool(figure_style_raw.get("bold_text", True), "figure_style.bold_text"),
                output_format_primary=str(figure_style_raw.get("output_format_primary", "svg")),
                palette_name=str(figure_style_raw.get("palette_name", "nature_manuscript_palette")),
                dpi_png=int(figure_style_raw.get("dpi_png", 300)),
            ),
            tuning=TuningConfig(
                use_small_grid_search=parse_bool(tuning_raw.get("use_small_grid_search", True), "tuning.use_small_grid_search"),
                cv_folds_inner=int(tuning_raw.get("cv_folds_inner", 3)),
            ),
            logs_dir=resolve(raw.get("script_02", {}).get("logs_dir", "logs")),
            configs_used_dir=resolve(raw.get("script_02", {}).get("configs_used_dir", "configs_used")),
            report_root=resolve("reports"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical baseline models for Step-05 tasks using Step-06 split manifests.")
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
    for path in [cfg.logs_dir, cfg.configs_used_dir, cfg.output_model_root, cfg.output_metrics_root, cfg.output_predictions_root, cfg.output_figures_root, cfg.output_report_path.parent]:
        path.mkdir(parents=True, exist_ok=True)


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
        return None
    snapshot_path = cfg.configs_used_dir / f"{SCRIPT_NAME}_config.yaml"
    snapshot_path.write_text(loaded_config_path.read_text(encoding="utf-8"), encoding="utf-8")
    logging.info("Saved config snapshot to %s", snapshot_path)
    return snapshot_path


def resolve_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def canonicalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    normalized = df.copy()
    mapping: dict[str, str] = {}
    warnings: list[str] = []
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in normalized.columns:
            mapping[canonical] = canonical
            continue
        resolved = resolve_first_column(normalized, aliases)
        if resolved is not None:
            normalized = normalized.rename(columns={resolved: canonical})
            mapping[canonical] = resolved
    if "compound_id" not in normalized.columns:
        raise ValueError("Dataset must contain either `compound_id` or `standardized_smiles`.")
    if mapping.get("compound_id") == "standardized_smiles":
        warnings.append("`compound_id` missing; `standardized_smiles` used as canonical identifier.")
    normalized["compound_id"] = normalized["compound_id"].astype(str)
    if "standardized_smiles" in normalized.columns:
        normalized["standardized_smiles"] = normalized["standardized_smiles"].astype(str)
    return normalized, mapping, warnings


def build_row_uid(task_name: str, row: pd.Series, row_index: int) -> str:
    if task_name == "pairwise_selectivity":
        return "|".join([task_name, str(row["compound_id"]), str(row.get("kinase_a_chembl_id", "NA")), str(row.get("kinase_b_chembl_id", "NA")), str(row_index)])
    if task_name == CLASSIFICATION_TASK_NAME:
        return "|".join([task_name, str(row["compound_id"]), str(row.get("target_chembl_id", "NA")), str(row_index)])
    return "|".join([task_name, str(row["compound_id"]), str(row.get("target_chembl_id", "NA")), str(row_index)])


def standardize_task_dataframe(task_name: str, df: pd.DataFrame, required_columns: list[str]) -> tuple[pd.DataFrame, dict[str, str], list[str], list[str]]:
    normalized, mapping, warnings = canonicalize_columns(df)
    missing = [column for column in required_columns if column not in normalized.columns]
    if missing:
        raise ValueError(f"{task_name} dataset is missing required columns: {', '.join(missing)}")
    normalized = normalized.reset_index(drop=True).copy()
    normalized["row_index"] = normalized.index.astype(int)
    normalized["row_uid"] = [build_row_uid(task_name, row, idx) for idx, row in normalized.iterrows()]
    label_columns = [col for col in LABEL_PRIORITY if col in normalized.columns]
    return normalized, mapping, warnings, label_columns


def load_required_dataframe(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required {description} not found: {path}")
    logging.info("Loading %s from %s", description, path)
    return pd.read_csv(path)


def validate_smiles_column(df: pd.DataFrame, task_name: str) -> None:
    if "standardized_smiles" not in df.columns:
        raise ValueError(f"{task_name} requires `standardized_smiles` for descriptor generation.")


def available_palette() -> list[str]:
    return ["#386CB0", "#F39C12", "#2CA25F", "#E74C3C", "#756BB1", "#7F8C8D"]


def configure_matplotlib(style: FigureStyle) -> None:
    if plt is None:
        return
    plt.rcParams["font.family"] = style.font_family
    plt.rcParams["font.weight"] = "bold" if style.bold_text else "normal"
    plt.rcParams["axes.labelweight"] = "bold" if style.bold_text else "normal"
    plt.rcParams["axes.titleweight"] = "bold" if style.bold_text else "normal"
    plt.rcParams["svg.fonttype"] = "none"


def rdkit_descriptor_names() -> list[str]:
    return [name for name, _ in Descriptors._descList]


def compute_compound_feature_table(compound_df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    validate_smiles_column(compound_df, "descriptor generation")
    unique_compounds = compound_df[["compound_id", "standardized_smiles"]].drop_duplicates().sort_values("compound_id", kind="mergesort")
    descriptor_names = rdkit_descriptor_names() if cfg.include_rdkit_2d_descriptors else []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    invalid_smiles: list[str] = []
    for row in unique_compounds.itertuples(index=False):
        compound_id = str(row.compound_id)
        smiles = str(row.standardized_smiles)
        mol = Chem.MolFromSmiles(smiles)
        record: dict[str, Any] = {"compound_id": compound_id, "standardized_smiles": smiles, "rdkit_parse_success": int(mol is not None)}
        if mol is None:
            invalid_smiles.append(compound_id)
            if cfg.use_morgan_fingerprints:
                for bit_idx in range(cfg.morgan_nbits):
                    record[f"morgan_{bit_idx}"] = np.nan
            for name in descriptor_names:
                record[f"rdkit_{name}"] = np.nan
            records.append(record)
            continue
        if cfg.use_morgan_fingerprints:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, cfg.morgan_radius, nBits=cfg.morgan_nbits)
            arr = np.zeros((cfg.morgan_nbits,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            for bit_idx, value in enumerate(arr.tolist()):
                record[f"morgan_{bit_idx}"] = int(value)
        for name, func in Descriptors._descList:
            if not cfg.include_rdkit_2d_descriptors:
                break
            try:
                record[f"rdkit_{name}"] = float(func(mol))
            except Exception:
                record[f"rdkit_{name}"] = np.nan
        records.append(record)
    features = pd.DataFrame.from_records(records)
    if invalid_smiles:
        warnings.append(f"Descriptor generation encountered {len(invalid_smiles)} invalid molecules; affected rows retain missing descriptor values.")
    metadata = {
        "n_unique_compounds": int(unique_compounds["compound_id"].nunique()),
        "n_feature_columns": int(len([c for c in features.columns if c not in {"compound_id", "standardized_smiles", "rdkit_parse_success"}])),
        "morgan_enabled": cfg.use_morgan_fingerprints,
        "rdkit_2d_enabled": cfg.include_rdkit_2d_descriptors,
        "invalid_compound_count": int(len(invalid_smiles)),
    }
    return features, metadata, warnings


def detect_classification_labels(df: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for col in df.columns:
        if col in LABEL_PRIORITY or col.endswith("_label"):
            labels.append(col)
    stable = []
    for col in LABEL_PRIORITY + sorted(labels):
        if col in df.columns and col not in stable:
            stable.append(col)
    return stable


def load_split_manifest(path: Path) -> pd.DataFrame:
    manifest = load_required_dataframe(path, "split manifest")
    required = {"task_name", "split_strategy", "split_id", "fold_id", "output_assignment_path"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError("Split manifest missing required columns: " + ", ".join(missing))
    return manifest


def load_assignment_table(path_value: str | None, project_root: Path) -> pd.DataFrame | None:
    if path_value in (None, "", np.nan):
        return None
    path = Path(path_value)
    resolved = path if path.is_absolute() else project_root / path
    if not resolved.exists():
        raise FileNotFoundError(f"Split assignment file not found: {resolved}")
    return pd.read_csv(resolved)


def classification_target_ready(series: pd.Series) -> pd.Series:
    mapped = pd.to_numeric(series, errors="coerce")
    if mapped.notna().sum() == 0:
        text = series.astype(str).str.strip().str.lower()
        unique = [value for value in text.dropna().unique().tolist() if value not in {"", "nan", "none"}]
        if len(unique) != 2:
            raise ValueError(f"Classification label column {series.name} must be binary; found values {unique!r}.")
        label_map = {unique[0]: 0, unique[1]: 1}
        mapped = text.map(label_map)
    mapped = mapped.dropna()
    return mapped.astype(int)


def infer_classification_positive_label(series: pd.Series) -> int:
    values = sorted(set(pd.to_numeric(series, errors="coerce").dropna().astype(int).tolist()))
    if values != [0, 1]:
        raise ValueError(f"Classification labels must be binary 0/1 after normalization; got {values}.")
    return 1


def assemble_task_dataset(task_name: str, df: pd.DataFrame, descriptor_table: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    merged = df.merge(descriptor_table, on=["compound_id", "standardized_smiles"], how="left", sort=False)
    numeric_feature_columns = [c for c in descriptor_table.columns if c.startswith("morgan_") or c.startswith("rdkit_") or c == "rdkit_parse_success"]
    categorical_columns: list[str] = []
    if task_name == "pairwise_selectivity":
        pair_cols = [col for col in ["kinase_a_chembl_id", "kinase_b_chembl_id", "kinase_a_target_name", "kinase_b_target_name"] if col in merged.columns]
        categorical_columns.extend(pair_cols)
    elif task_name in {"multitask_regression", "target_vs_panel", CLASSIFICATION_TASK_NAME}:
        for col in ["target_chembl_id", "target_name"]:
            if col in merged.columns:
                categorical_columns.append(col)
    if cfg.include_environment_features:
        categorical_columns.extend([col for col in ENVIRONMENT_CANDIDATES if col in merged.columns])
    ordered_cats: list[str] = []
    for col in categorical_columns:
        if col not in ordered_cats:
            ordered_cats.append(col)
    feature_columns = numeric_feature_columns + ordered_cats
    return merged, feature_columns, numeric_feature_columns, ordered_cats


def build_regression_model_factory(cfg: AppConfig) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    models: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "ridge": (Ridge(random_state=cfg.random_seed), {"model__alpha": [0.1, 1.0, 10.0]}),
        "random_forest": (
            RandomForestRegressor(n_estimators=300, random_state=cfg.random_seed, n_jobs=cfg.n_jobs),
            {"model__max_depth": [None, 12], "model__min_samples_leaf": [1, 3]},
        ),
        "extra_trees": (
            ExtraTreesRegressor(n_estimators=300, random_state=cfg.random_seed, n_jobs=cfg.n_jobs),
            {"model__max_depth": [None, 12], "model__min_samples_leaf": [1, 3]},
        ),
        "svm_rbf": (SVR(kernel="rbf", C=1.0, gamma="scale"), {"model__C": [1.0, 10.0], "model__epsilon": [0.1, 0.2]}),
    }
    if XGBRegressor is not None:
        models["xgboost"] = (
            XGBRegressor(
                random_state=cfg.random_seed,
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                subsample=1.0,
                colsample_bytree=1.0,
                objective="reg:squarederror",
                n_jobs=cfg.n_jobs,
                verbosity=0,
            ),
            {"model__max_depth": [4, 6], "model__n_estimators": [150, 250]},
        )
    if LGBMRegressor is not None:
        models["lightgbm"] = (
            LGBMRegressor(random_state=cfg.random_seed, n_estimators=250, learning_rate=0.05, n_jobs=cfg.n_jobs, verbosity=-1),
            {"model__num_leaves": [31, 63], "model__min_child_samples": [10, 20]},
        )
    return models


def build_classification_model_factory(cfg: AppConfig) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    models: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=cfg.random_seed),
            {"model__C": [0.5, 1.0, 5.0]},
        ),
        "random_forest_classifier": (
            RandomForestClassifier(n_estimators=300, random_state=cfg.random_seed, n_jobs=cfg.n_jobs),
            {"model__max_depth": [None, 12], "model__min_samples_leaf": [1, 3]},
        ),
        "extra_trees_classifier": (
            ExtraTreesClassifier(n_estimators=300, random_state=cfg.random_seed, n_jobs=cfg.n_jobs),
            {"model__max_depth": [None, 12], "model__min_samples_leaf": [1, 3]},
        ),
        "svm_rbf_classifier": (
            SVC(kernel="rbf", C=1.0, probability=True, random_state=cfg.random_seed),
            {"model__C": [1.0, 10.0]},
        ),
    }
    if XGBClassifier is not None:
        models["xgboost_classifier"] = (
            XGBClassifier(
                random_state=cfg.random_seed,
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                subsample=1.0,
                colsample_bytree=1.0,
                eval_metric="logloss",
                n_jobs=cfg.n_jobs,
                verbosity=0,
            ),
            {"model__max_depth": [4, 6], "model__n_estimators": [150, 250]},
        )
    if LGBMClassifier is not None:
        models["lightgbm_classifier"] = (
            LGBMClassifier(random_state=cfg.random_seed, n_estimators=250, learning_rate=0.05, n_jobs=cfg.n_jobs, verbosity=-1),
            {"model__num_leaves": [31, 63], "model__min_child_samples": [10, 20]},
        )
    return models


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def build_pipeline(model_name: str, estimator: Any, numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    scale_numeric = model_name in {"ridge", "svm_rbf", "logistic_regression", "svm_rbf_classifier"}
    preprocessor = build_preprocessor(numeric_columns, categorical_columns, scale_numeric=scale_numeric)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    y_true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    y_pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
    return safe_pearson(y_true_rank, y_pred_rank)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "spearman": safe_spearman(y_true, y_pred),
        "pearson": safe_pearson(y_true, y_pred),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def summarize_metrics(df: pd.DataFrame, metric_columns: list[str], group_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_columns + ["n_evaluations"] + [f"{metric}_mean" for metric in metric_columns])
    summary = df.groupby(group_columns, dropna=False)[metric_columns].agg(["mean", "std", "median"]).reset_index()
    summary.columns = ["_".join([part for part in col if part]).strip("_") for col in summary.columns.to_flat_index()]
    counts = df.groupby(group_columns, dropna=False).size().rename("n_evaluations").reset_index()
    return counts.merge(summary, on=group_columns, how="left", sort=False)


def split_assignment_subsets(assignment_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "split_label" not in assignment_df.columns:
        raise ValueError("Assignment table missing `split_label` column.")
    subsets = {}
    for label in ["train", "valid", "test"]:
        subsets[label] = assignment_df[assignment_df["split_label"] == label].copy()
    return subsets


def fit_model_with_optional_tuning(pipeline: Pipeline, param_grid: dict[str, list[Any]], x_train: pd.DataFrame, y_train: pd.Series, cfg: AppConfig, is_classification: bool) -> tuple[Pipeline, dict[str, Any]]:
    if not cfg.tuning.use_small_grid_search or not param_grid or len(y_train) < max(cfg.tuning.cv_folds_inner * 2, 10):
        fitted = clone(pipeline)
        fitted.fit(x_train, y_train)
        return fitted, {"tuned": False, "selected_params": {}}
    scoring = "neg_root_mean_squared_error" if not is_classification else "roc_auc"
    if is_classification and len(pd.Series(y_train).value_counts()) < 2:
        fitted = clone(pipeline)
        fitted.fit(x_train, y_train)
        return fitted, {"tuned": False, "selected_params": {}, "reason": "single_class_training"}
    search = GridSearchCV(estimator=clone(pipeline), param_grid=param_grid, scoring=scoring, cv=cfg.tuning.cv_folds_inner, n_jobs=cfg.n_jobs, refit=True)
    search.fit(x_train, y_train)
    return search.best_estimator_, {"tuned": True, "selected_params": search.best_params_, "best_score": float(search.best_score_)}


def determine_best_task_summary(metric_summary: pd.DataFrame, task_type: str) -> pd.DataFrame:
    if metric_summary.empty:
        return metric_summary
    score_col = "rmse_mean" if task_type == "regression" else "roc_auc_mean"
    ascending = task_type == "regression"
    subset = metric_summary.sort_values(score_col, ascending=ascending, kind="mergesort").groupby(["task_name", "split_strategy"], dropna=False, as_index=False).first()
    return subset


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info("Wrote %s rows to %s", len(df), path)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)
    logging.info("Saved model artifact to %s", path)


def render_figure(fig: Any, output_base: Path, cfg: AppConfig) -> list[str]:
    paths: list[str] = []
    if plt is None:
        return paths
    if cfg.export_svg:
        svg_path = output_base.with_suffix(".svg")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        paths.append(str(svg_path))
    if cfg.export_png:
        png_path = output_base.with_suffix(".png")
        fig.savefig(png_path, format="png", dpi=cfg.figure_style.dpi_png, bbox_inches="tight")
        paths.append(str(png_path))
    if cfg.export_pdf:
        pdf_path = output_base.with_suffix(".pdf")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        paths.append(str(pdf_path))
    plt.close(fig)
    return paths


def make_bar_figure(summary_df: pd.DataFrame, value_column: str, title: str, ylabel: str, output_base: Path, cfg: AppConfig) -> list[str]:
    if plt is None or summary_df.empty or value_column not in summary_df.columns:
        return []
    configure_matplotlib(cfg.figure_style)
    ordered = summary_df.sort_values(value_column, ascending=(value_column.startswith("rmse")), kind="mergesort")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = available_palette()
    ax.bar(np.arange(len(ordered)), ordered[value_column], color=[colors[i % len(colors)] for i in range(len(ordered))])
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels((ordered["task_name"] + "\n" + ordered["model_name"] + "\n" + ordered["split_strategy"]).tolist(), rotation=45, ha="right", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return render_figure(fig, output_base, cfg)


def make_scatter_figure(pred_df: pd.DataFrame, task_name: str, model_name: str, output_base: Path, cfg: AppConfig) -> list[str]:
    if plt is None or pred_df.empty:
        return []
    configure_matplotlib(cfg.figure_style)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(pred_df["observed"], pred_df["predicted"], alpha=0.5, color=available_palette()[0], edgecolors="none")
    limits = [min(pred_df["observed"].min(), pred_df["predicted"].min()), max(pred_df["observed"].max(), pred_df["predicted"].max())]
    ax.plot(limits, limits, linestyle="--", color=available_palette()[3], linewidth=1.5)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Observed", fontweight="bold")
    ax.set_ylabel("Predicted", fontweight="bold")
    ax.set_title(f"{task_name}: {model_name}", fontweight="bold")
    ax.grid(alpha=0.25)
    return render_figure(fig, output_base, cfg)


def make_roc_pr_figures(pred_df: pd.DataFrame, title_prefix: str, output_base: Path, cfg: AppConfig) -> dict[str, list[str]]:
    if plt is None or pred_df.empty or "score" not in pred_df.columns:
        return {}
    thresholds = np.linspace(0.0, 1.0, 101)
    y_true = pred_df["observed"].to_numpy(dtype=int)
    y_score = pred_df["score"].to_numpy(dtype=float)
    roc_points = []
    pr_points = []
    positives = max((y_true == 1).sum(), 1)
    negatives = max((y_true == 0).sum(), 1)
    for thr in thresholds:
        y_hat = (y_score >= thr).astype(int)
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())
        tn = int(((y_hat == 0) & (y_true == 0)).sum())
        roc_points.append({"fpr": fp / negatives, "tpr": tp / positives})
        precision = tp / max(tp + fp, 1)
        recall = tp / positives
        pr_points.append({"precision": precision, "recall": recall})
    outputs: dict[str, list[str]] = {}
    for kind, x_key, y_key, title, subpath in [
        (roc_points, "fpr", "tpr", f"{title_prefix} ROC", output_base.with_name(output_base.name + "_roc")),
        (pr_points, "recall", "precision", f"{title_prefix} PR", output_base.with_name(output_base.name + "_pr")),
    ]:
        df_points = pd.DataFrame(kind)
        save_dataframe(df_points, subpath.with_suffix(".source_data.csv"))
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot(df_points[x_key], df_points[y_key], color=available_palette()[1], linewidth=2)
        ax.set_xlabel(x_key.upper(), fontweight="bold")
        ax.set_ylabel(y_key.upper(), fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.25)
        outputs[subpath.name] = render_figure(fig, subpath, cfg)
    return outputs


def task_enabled(task_name: str, cfg: AppConfig) -> bool:
    return {
        "multitask_regression": cfg.run_multitask_regression,
        "pairwise_selectivity": cfg.run_pairwise_selectivity_regression,
        "target_vs_panel": cfg.run_target_vs_panel_regression,
        CLASSIFICATION_TASK_NAME: cfg.run_classification_tasks,
    }[task_name]


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg, loaded_config_path, _ = load_config(args.config, project_root)
    ensure_output_dirs(cfg)
    log_file, _ = setup_logging(cfg.logs_dir)
    logging.info("Starting %s", SCRIPT_NAME)
    logging.info("Deterministic random seed fixed at %s", cfg.random_seed)
    np.random.seed(cfg.random_seed)

    config_snapshot_path = save_config_snapshot(cfg, loaded_config_path)
    manifest_df = load_split_manifest(cfg.input_split_manifest_path)

    task_tables: dict[str, pd.DataFrame] = {}
    task_reports: dict[str, Any] = {}
    all_warnings: list[str] = []
    compound_frames: list[pd.DataFrame] = []

    for task_name, spec in REGRESSION_TASKS.items():
        if not task_enabled(task_name, cfg):
            continue
        raw_df = load_required_dataframe(getattr(cfg, spec["config_key"]), f"{task_name} dataset")
        normalized, mapping, warnings, _ = standardize_task_dataframe(task_name, raw_df, spec["required_columns"])
        validate_smiles_column(normalized, task_name)
        task_tables[task_name] = normalized
        compound_frames.append(normalized[["compound_id", "standardized_smiles"]])
        task_reports[task_name] = {"column_mapping": mapping, "warnings": warnings, "n_rows": int(len(normalized)), "n_unique_compounds": int(normalized["compound_id"].nunique())}
        all_warnings.extend([f"{task_name}: {w}" for w in warnings])

    if task_enabled(CLASSIFICATION_TASK_NAME, cfg):
        raw_df = load_required_dataframe(cfg.input_classification_path, "classification dataset")
        normalized, mapping, warnings, label_columns = standardize_task_dataframe(CLASSIFICATION_TASK_NAME, raw_df, [])
        validate_smiles_column(normalized, CLASSIFICATION_TASK_NAME)
        detected_labels = detect_classification_labels(normalized)
        if not detected_labels:
            raise ValueError("Classification dataset does not contain any usable label columns.")
        task_tables[CLASSIFICATION_TASK_NAME] = normalized
        compound_frames.append(normalized[["compound_id", "standardized_smiles"]])
        task_reports[CLASSIFICATION_TASK_NAME] = {
            "column_mapping": mapping,
            "warnings": warnings,
            "label_columns": detected_labels,
            "n_rows": int(len(normalized)),
            "n_unique_compounds": int(normalized["compound_id"].nunique()),
        }
        all_warnings.extend([f"classification: {w}" for w in warnings])
        logging.info("Detected classification labels: %s", ", ".join(detected_labels))

    if not compound_frames:
        raise ValueError("No tasks enabled for Script-07.")

    descriptor_input = pd.concat(compound_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    descriptor_table, descriptor_metadata, descriptor_warnings = compute_compound_feature_table(descriptor_input, cfg)
    all_warnings.extend(descriptor_warnings)
    if cfg.save_feature_tables:
        save_dataframe(descriptor_table, cfg.output_metrics_root / "compound_descriptor_features.csv")

    regression_models = build_regression_model_factory(cfg)
    classification_models = build_classification_model_factory(cfg)

    regression_per_fold: list[dict[str, Any]] = []
    classification_per_fold: list[dict[str, Any]] = []
    regression_predictions: list[pd.DataFrame] = []
    classification_predictions: list[pd.DataFrame] = []
    figure_paths: dict[str, Any] = {}
    skipped_configs: list[dict[str, Any]] = []

    for task_name, df in task_tables.items():
        logging.info("Assembling features for %s", task_name)
        assembled_df, feature_columns, numeric_feature_columns, categorical_columns = assemble_task_dataset(task_name, df, descriptor_table, cfg)
        if cfg.save_feature_tables:
            save_dataframe(assembled_df[["row_uid", "compound_id", *feature_columns]].copy(), cfg.output_metrics_root / task_name / "model_feature_table.csv")
        task_manifest = manifest_df[manifest_df["task_name"] == task_name].copy()
        if task_manifest.empty:
            warning = f"No split manifest entries found for task `{task_name}`; skipping task."
            logging.warning(warning)
            all_warnings.append(warning)
            continue

        if task_name == CLASSIFICATION_TASK_NAME:
            labels_to_run = detect_classification_labels(assembled_df)
            target_specs = [(label, classification_models, cfg.classification_models, True) for label in labels_to_run]
        else:
            target_specs = [(REGRESSION_TASKS[task_name]["target_column"], regression_models, cfg.regression_models, False)]

        for target_column, model_factory, configured_model_names, is_classification in target_specs:
            modeling_df = assembled_df.copy()
            if is_classification:
                label_numeric = pd.to_numeric(modeling_df[target_column], errors="coerce")
                if label_numeric.notna().sum() == 0:
                    text = modeling_df[target_column].astype(str).str.strip().str.lower()
                    unique = [v for v in sorted(text.unique().tolist()) if v not in {"", "nan", "none"}]
                    if len(unique) != 2:
                        warning = f"{task_name}/{target_column}: label normalization failed because values are not binary."
                        logging.warning(warning)
                        all_warnings.append(warning)
                        continue
                    model_map = {unique[0]: 0, unique[1]: 1}
                    modeling_df[target_column] = text.map(model_map)
                else:
                    modeling_df[target_column] = label_numeric
                modeling_df = modeling_df[modeling_df[target_column].isin([0, 1])].copy()
                if modeling_df.empty:
                    warning = f"{task_name}/{target_column}: no labeled rows after binary filtering."
                    logging.warning(warning)
                    all_warnings.append(warning)
                    continue
                modeling_df[target_column] = modeling_df[target_column].astype(int)

            for manifest_row in task_manifest.itertuples(index=False):
                assignment_df = load_assignment_table(getattr(manifest_row, "output_assignment_path", None), project_root)
                if assignment_df is None:
                    skipped_configs.append({"task_name": task_name, "target_label": target_column, "split_strategy": manifest_row.split_strategy, "reason": "missing_assignment_path"})
                    continue
                if "row_uid" not in assignment_df.columns:
                    raise ValueError(f"Assignment table for {task_name}/{manifest_row.split_strategy} lacks `row_uid`.")
                labeled = modeling_df.merge(assignment_df[["row_uid", "split_label"]], on="row_uid", how="inner", sort=False)
                train_df = labeled[labeled["split_label"] == "train"].copy()
                valid_df = labeled[labeled["split_label"] == "valid"].copy()
                test_df = labeled[labeled["split_label"] == "test"].copy()
                if train_df.empty or test_df.empty:
                    warning = f"{task_name}/{target_column}/{manifest_row.split_strategy}: skipped because train or test split is empty."
                    logging.warning(warning)
                    all_warnings.append(warning)
                    skipped_configs.append({"task_name": task_name, "target_label": target_column, "split_strategy": manifest_row.split_strategy, "reason": "empty_train_or_test"})
                    continue
                if is_classification and train_df[target_column].nunique() < 2:
                    warning = f"{task_name}/{target_column}/{manifest_row.split_strategy}: skipped because training data has a single class."
                    logging.warning(warning)
                    all_warnings.append(warning)
                    skipped_configs.append({"task_name": task_name, "target_label": target_column, "split_strategy": manifest_row.split_strategy, "reason": "single_class_train"})
                    continue

                if task_name == "multitask_regression" and cfg.multitask_strategy != "one_model_per_kinase":
                    raise ValueError("Only `one_model_per_kinase` multitask strategy is currently supported.")

                if task_name == "multitask_regression":
                    subgroup_values = sorted(train_df["target_chembl_id"].dropna().astype(str).unique().tolist())
                else:
                    subgroup_values = ["global"]

                for subgroup in subgroup_values:
                    if subgroup == "global":
                        train_subset = train_df
                        valid_subset = valid_df
                        test_subset = test_df
                        subgroup_name = "global"
                    else:
                        train_subset = train_df[train_df["target_chembl_id"].astype(str) == subgroup].copy()
                        valid_subset = valid_df[valid_df["target_chembl_id"].astype(str) == subgroup].copy()
                        test_subset = test_df[test_df["target_chembl_id"].astype(str) == subgroup].copy()
                        subgroup_name = subgroup
                    if len(train_subset) < 10 or test_subset.empty:
                        skipped_configs.append({"task_name": task_name, "target_label": target_column, "split_strategy": manifest_row.split_strategy, "subgroup": subgroup_name, "reason": "insufficient_subgroup_data"})
                        continue
                    x_train = train_subset[feature_columns].copy()
                    y_train = train_subset[target_column].copy()
                    eval_test = test_subset.copy()
                    if not valid_subset.empty:
                        eval_valid = valid_subset.copy()
                    else:
                        eval_valid = pd.DataFrame(columns=test_subset.columns)

                    for model_name in configured_model_names:
                        if model_name not in model_factory:
                            warning = f"Model `{model_name}` unavailable; likely optional dependency missing."
                            logging.warning(warning)
                            all_warnings.append(warning)
                            skipped_configs.append({"task_name": task_name, "target_label": target_column, "model_name": model_name, "split_strategy": manifest_row.split_strategy, "reason": "model_unavailable"})
                            continue
                        estimator, param_grid = model_factory[model_name]
                        pipeline = build_pipeline(model_name, estimator, numeric_feature_columns, categorical_columns)
                        try:
                            fitted_model, tuning_info = fit_model_with_optional_tuning(pipeline, param_grid, x_train, y_train, cfg, is_classification=is_classification)
                        except Exception as exc:
                            warning = f"Training failed for {task_name}/{target_column}/{model_name}/{manifest_row.split_strategy}/{subgroup_name}: {exc}"
                            logging.warning(warning)
                            all_warnings.append(warning)
                            skipped_configs.append({"task_name": task_name, "target_label": target_column, "model_name": model_name, "split_strategy": manifest_row.split_strategy, "subgroup": subgroup_name, "reason": str(exc)})
                            continue

                        eval_sets = [("test", eval_test)]
                        if not eval_valid.empty:
                            eval_sets.append(("valid", eval_valid))
                        for evaluation_split, eval_df in eval_sets:
                            x_eval = eval_df[feature_columns].copy()
                            y_eval = eval_df[target_column].copy()
                            if is_classification:
                                predicted = fitted_model.predict(x_eval)
                                score = None
                                if hasattr(fitted_model, "predict_proba"):
                                    score = fitted_model.predict_proba(x_eval)[:, 1]
                                elif hasattr(fitted_model, "decision_function"):
                                    decision = fitted_model.decision_function(x_eval)
                                    score = 1.0 / (1.0 + np.exp(-decision))
                                metrics = classification_metrics(y_eval.to_numpy(dtype=int), np.asarray(predicted, dtype=int), None if score is None else np.asarray(score, dtype=float))
                                record = {
                                    "task_name": task_name,
                                    "target_label": target_column,
                                    "subtask_id": subgroup_name,
                                    "split_strategy": manifest_row.split_strategy,
                                    "split_id": manifest_row.split_id,
                                    "fold_id": manifest_row.fold_id,
                                    "evaluation_split": evaluation_split,
                                    "model_name": model_name,
                                    "n_train": int(len(train_subset)),
                                    "n_valid": int(len(valid_subset)),
                                    "n_test": int(len(test_subset)),
                                    **metrics,
                                    "tuned": tuning_info.get("tuned", False),
                                    "selected_params": json.dumps(tuning_info.get("selected_params", {}), sort_keys=True),
                                }
                                classification_per_fold.append(record)
                                pred_df = eval_df[["row_uid", "compound_id"]].copy()
                                pred_df["observed"] = y_eval.to_numpy()
                                pred_df["predicted"] = np.asarray(predicted)
                                pred_df["score"] = np.nan if score is None else np.asarray(score)
                                for extra_col in ["target_chembl_id", "target_name", "kinase_a_chembl_id", "kinase_b_chembl_id"]:
                                    if extra_col in eval_df.columns:
                                        pred_df[extra_col] = eval_df[extra_col].values
                                pred_df.insert(0, "task_name", task_name)
                                pred_df.insert(1, "target_label", target_column)
                                pred_df.insert(2, "subtask_id", subgroup_name)
                                pred_df.insert(3, "model_name", model_name)
                                pred_df.insert(4, "split_strategy", manifest_row.split_strategy)
                                pred_df.insert(5, "split_id", manifest_row.split_id)
                                pred_df.insert(6, "fold_id", manifest_row.fold_id)
                                pred_df.insert(7, "evaluation_split", evaluation_split)
                                classification_predictions.append(pred_df)
                            else:
                                predicted = fitted_model.predict(x_eval)
                                metrics = regression_metrics(y_eval.to_numpy(dtype=float), np.asarray(predicted, dtype=float))
                                record = {
                                    "task_name": task_name,
                                    "target_label": target_column,
                                    "subtask_id": subgroup_name,
                                    "split_strategy": manifest_row.split_strategy,
                                    "split_id": manifest_row.split_id,
                                    "fold_id": manifest_row.fold_id,
                                    "evaluation_split": evaluation_split,
                                    "model_name": model_name,
                                    "n_train": int(len(train_subset)),
                                    "n_valid": int(len(valid_subset)),
                                    "n_test": int(len(test_subset)),
                                    **metrics,
                                    "tuned": tuning_info.get("tuned", False),
                                    "selected_params": json.dumps(tuning_info.get("selected_params", {}), sort_keys=True),
                                }
                                regression_per_fold.append(record)
                                pred_df = eval_df[["row_uid", "compound_id"]].copy()
                                pred_df["observed"] = y_eval.to_numpy()
                                pred_df["predicted"] = np.asarray(predicted)
                                pred_df["residual"] = pred_df["observed"] - pred_df["predicted"]
                                pred_df["absolute_error"] = pred_df["residual"].abs()
                                for extra_col in ["target_chembl_id", "target_name", "kinase_a_chembl_id", "kinase_b_chembl_id"]:
                                    if extra_col in eval_df.columns:
                                        pred_df[extra_col] = eval_df[extra_col].values
                                pred_df.insert(0, "task_name", task_name)
                                pred_df.insert(1, "target_label", target_column)
                                pred_df.insert(2, "subtask_id", subgroup_name)
                                pred_df.insert(3, "model_name", model_name)
                                pred_df.insert(4, "split_strategy", manifest_row.split_strategy)
                                pred_df.insert(5, "split_id", manifest_row.split_id)
                                pred_df.insert(6, "fold_id", manifest_row.fold_id)
                                pred_df.insert(7, "evaluation_split", evaluation_split)
                                regression_predictions.append(pred_df)

                        if cfg.save_trained_models:
                            model_path = cfg.output_model_root / task_name / manifest_row.split_strategy / str(manifest_row.fold_id) / subgroup_name / f"{target_column}__{model_name}.pkl"
                            save_pickle({"model": fitted_model, "feature_columns": feature_columns, "numeric_feature_columns": numeric_feature_columns, "categorical_columns": categorical_columns, "tuning": tuning_info}, model_path)

    regression_metrics_df = pd.DataFrame(regression_per_fold)
    classification_metrics_df = pd.DataFrame(classification_per_fold)
    regression_predictions_df = pd.concat(regression_predictions, ignore_index=True) if regression_predictions else pd.DataFrame()
    classification_predictions_df = pd.concat(classification_predictions, ignore_index=True) if classification_predictions else pd.DataFrame()

    if not regression_metrics_df.empty:
        save_dataframe(regression_metrics_df, cfg.output_metrics_root / "regression_metrics_per_fold.csv")
        regression_summary = summarize_metrics(regression_metrics_df[regression_metrics_df["evaluation_split"] == "test"], ["rmse", "mae", "r2", "spearman", "pearson", "median_absolute_error"], ["task_name", "target_label", "split_strategy", "model_name"])
        save_dataframe(regression_summary, cfg.output_metrics_root / "regression_metrics_summary.csv")
        best_regression = determine_best_task_summary(regression_summary, task_type="regression")
        save_dataframe(best_regression, cfg.output_metrics_root / "regression_best_models.csv")
    else:
        regression_summary = pd.DataFrame()
        best_regression = pd.DataFrame()

    if not classification_metrics_df.empty:
        save_dataframe(classification_metrics_df, cfg.output_metrics_root / "classification_metrics_per_fold.csv")
        classification_summary = summarize_metrics(classification_metrics_df[classification_metrics_df["evaluation_split"] == "test"], ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1", "mcc", "precision", "recall"], ["task_name", "target_label", "split_strategy", "model_name"])
        save_dataframe(classification_summary, cfg.output_metrics_root / "classification_metrics_summary.csv")
        best_classification = determine_best_task_summary(classification_summary, task_type="classification")
        save_dataframe(best_classification, cfg.output_metrics_root / "classification_best_models.csv")
    else:
        classification_summary = pd.DataFrame()
        best_classification = pd.DataFrame()

    if cfg.save_test_predictions and not regression_predictions_df.empty:
        save_dataframe(regression_predictions_df, cfg.output_predictions_root / "regression_predictions.csv")
    if cfg.save_test_predictions and not classification_predictions_df.empty:
        save_dataframe(classification_predictions_df, cfg.output_predictions_root / "classification_predictions.csv")

    if cfg.save_error_tables and not regression_predictions_df.empty:
        regression_test_predictions = regression_predictions_df[regression_predictions_df["evaluation_split"] == "test"].copy()
        hardest_compounds = regression_test_predictions.groupby("compound_id", dropna=False)["absolute_error"].mean().sort_values(ascending=False).reset_index().rename(columns={"absolute_error": "mean_absolute_error"})
        save_dataframe(hardest_compounds, cfg.output_metrics_root / "error_analysis_hardest_compounds.csv")
        if "target_chembl_id" in regression_test_predictions.columns:
            hardest_targets = regression_test_predictions.groupby("target_chembl_id", dropna=False)["absolute_error"].mean().sort_values(ascending=False).reset_index().rename(columns={"absolute_error": "mean_absolute_error"})
            save_dataframe(hardest_targets, cfg.output_metrics_root / "error_analysis_hardest_targets.csv")

    if cfg.make_figures:
        if not regression_summary.empty:
            figure_paths["regression_model_comparison"] = make_bar_figure(regression_summary, "rmse_mean", "Regression baseline comparison", "RMSE", cfg.output_figures_root / "regression_model_comparison", cfg)
            save_dataframe(regression_summary, cfg.output_figures_root / "regression_model_comparison.source_data.csv")
        if not classification_summary.empty:
            figure_paths["classification_model_comparison"] = make_bar_figure(classification_summary, "roc_auc_mean", "Classification baseline comparison", "ROC-AUC", cfg.output_figures_root / "classification_model_comparison", cfg)
            save_dataframe(classification_summary, cfg.output_figures_root / "classification_model_comparison.source_data.csv")
        if not regression_predictions_df.empty and not best_regression.empty:
            top_reg = best_regression.iloc[0]
            pred_subset = regression_predictions_df[(regression_predictions_df["task_name"] == top_reg["task_name"]) & (regression_predictions_df["model_name"] == top_reg["model_name"]) & (regression_predictions_df["split_strategy"] == top_reg["split_strategy"]) & (regression_predictions_df["evaluation_split"] == "test")]
            figure_paths["top_regression_scatter"] = make_scatter_figure(pred_subset, str(top_reg["task_name"]), str(top_reg["model_name"]), cfg.output_figures_root / "top_regression_scatter", cfg)
            save_dataframe(pred_subset, cfg.output_figures_root / "top_regression_scatter.source_data.csv")
        if not classification_predictions_df.empty and not best_classification.empty:
            top_cls = best_classification.iloc[0]
            pred_subset = classification_predictions_df[(classification_predictions_df["task_name"] == top_cls["task_name"]) & (classification_predictions_df["target_label"] == top_cls["target_label"]) & (classification_predictions_df["model_name"] == top_cls["model_name"]) & (classification_predictions_df["split_strategy"] == top_cls["split_strategy"]) & (classification_predictions_df["evaluation_split"] == "test")]
            figure_paths["top_classification_curves"] = make_roc_pr_figures(pred_subset, f"{top_cls['task_name']} {top_cls['model_name']}", cfg.output_figures_root / "top_classification", cfg)

    report = {
        "script_name": SCRIPT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file_paths": {
            "input_regression_long_path": str(cfg.input_regression_long_path),
            "input_pairwise_selectivity_path": str(cfg.input_pairwise_selectivity_path),
            "input_target_vs_panel_path": str(cfg.input_target_vs_panel_path),
            "input_classification_path": str(cfg.input_classification_path),
            "input_split_manifest_path": str(cfg.input_split_manifest_path),
        },
        "output_paths": {
            "output_model_root": str(cfg.output_model_root),
            "output_metrics_root": str(cfg.output_metrics_root),
            "output_predictions_root": str(cfg.output_predictions_root),
            "output_figures_root": str(cfg.output_figures_root),
            "output_report_path": str(cfg.output_report_path),
            "log_file": str(log_file),
            "config_snapshot": str(config_snapshot_path) if config_snapshot_path is not None else None,
        },
        "tasks_processed": sorted(task_tables.keys()),
        "models_trained": {
            "regression": [m for m in cfg.regression_models if m in regression_models],
            "classification": [m for m in cfg.classification_models if m in classification_models],
        },
        "split_strategies_processed": sorted(set(manifest_df[manifest_df["task_name"].isin(task_tables.keys())]["split_strategy"].astype(str).tolist())),
        "descriptor_settings": {
            "descriptor_type": cfg.descriptor_type,
            "use_morgan_fingerprints": cfg.use_morgan_fingerprints,
            "morgan_radius": cfg.morgan_radius,
            "morgan_nbits": cfg.morgan_nbits,
            "include_rdkit_2d_descriptors": cfg.include_rdkit_2d_descriptors,
            "include_environment_features": cfg.include_environment_features,
            **descriptor_metadata,
        },
        "feature_dimensions": {task_name: int(len([c for c in assemble_task_dataset(task_name, df, descriptor_table, cfg)[1]])) for task_name, df in task_tables.items()},
        "rows_per_task": {task_name: int(len(df)) for task_name, df in task_tables.items()},
        "compounds_per_task": {task_name: int(df["compound_id"].nunique()) for task_name, df in task_tables.items()},
        "kinases_per_task": {
            task_name: int(
                len(
                    set().union(
                        *[
                            set(df[col].dropna().astype(str).tolist())
                            for col in ["target_chembl_id", "kinase_a_chembl_id", "kinase_b_chembl_id"]
                            if col in df.columns
                        ]
                    )
                )
            )
            for task_name, df in task_tables.items()
        },
        "regression_summary_metrics": [] if regression_summary.empty else regression_summary.to_dict(orient="records"),
        "classification_summary_metrics": [] if classification_summary.empty else classification_summary.to_dict(orient="records"),
        "best_model_per_task": {
            "regression": [] if best_regression.empty else best_regression.to_dict(orient="records"),
            "classification": [] if best_classification.empty else best_classification.to_dict(orient="records"),
        },
        "best_model_per_split_strategy": {
            "regression": [] if regression_summary.empty else regression_summary.sort_values("rmse_mean", kind="mergesort").groupby(["split_strategy"], as_index=False).first().to_dict(orient="records"),
            "classification": [] if classification_summary.empty else classification_summary.sort_values("roc_auc_mean", ascending=False, kind="mergesort").groupby(["split_strategy"], as_index=False).first().to_dict(orient="records"),
        },
        "task_reports": task_reports,
        "figure_outputs": figure_paths,
        "warnings": all_warnings,
        "skipped_configurations": skipped_configs,
    }
    cfg.output_report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logging.info("Wrote report to %s", cfg.output_report_path)
    logging.info("Completed %s", SCRIPT_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
