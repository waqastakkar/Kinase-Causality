#!/usr/bin/env python3
"""Train the main causal environment-aware model for kinase causal-QSAR tasks.

This script is a strict continuation of Steps 04-08. It consumes the annotated
long-format environment table from Step-04, task tables from Step-05, and split
assignments from Step-06 to train a modular graph-based model with optional
causal objectives, environment diagnostics, and ablations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

SCRIPT_NAME = "09_train_causal_environment_aware_model"
DEFAULT_SEED = 2025
CLASSIFICATION_TASK_NAME = "classification"
NATURE_PALETTE = ["#386CB0", "#F39C12", "#2CA25F", "#E74C3C", "#756BB1", "#7F8C8D"]

REQUIRED_SCRIPT_09_KEYS = {
    "input_annotated_long_path",
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
    "device",
    "num_workers",
    "batch_size",
    "max_epochs",
    "early_stopping_patience",
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "run_multitask_regression",
    "run_pairwise_selectivity_regression",
    "run_target_vs_panel_regression",
    "run_classification_tasks",
    "molecular_encoder",
    "hidden_dim",
    "num_message_passing_layers",
    "dropout",
    "use_target_identity_embedding",
    "target_embedding_dim",
    "use_kinase_family_embedding",
    "kinase_family_embedding_dim",
    "environment_columns",
    "causal_objectives",
    "loss_weights",
    "invariant_strategy",
    "adversarial_strategy",
    "environment_pooling_strategy",
    "multitask_strategy",
    "pairwise_strategy",
    "target_vs_panel_strategy",
    "save_trained_models",
    "save_model_checkpoints",
    "save_fold_predictions",
    "save_test_predictions",
    "save_latent_embeddings",
    "save_environment_predictions",
    "save_error_tables",
    "save_ablation_results",
    "save_config_snapshot",
    "run_core_ablations",
    "ablations",
    "make_figures",
    "export_svg",
    "export_png",
    "export_pdf",
    "figure_style",
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "compound_id": ("compound_id", "standardized_smiles"),
    "standardized_smiles": ("standardized_smiles",),
    "pKi": ("pKi", "median_pKi", "aggregated_pKi"),
    "delta_pKi": ("delta_pKi", "delta_pki"),
    "target_vs_panel_delta_pKi": ("target_vs_panel_delta_pKi", "target_vs_panel_delta_pki"),
    "target_chembl_id": ("target_chembl_id",),
    "kinase_a_chembl_id": ("kinase_a_chembl_id",),
    "kinase_b_chembl_id": ("kinase_b_chembl_id",),
    "kinase_family": ("kinase_family", "kinase_family_label", "target_family", "broad_kinase_family"),
    "activity_cliff_flag": ("activity_cliff_flag", "cliff_flag"),
}

REGRESSION_TASKS: dict[str, dict[str, Any]] = {
    "multitask_regression": {
        "config_key": "input_regression_long_path",
        "label_column": "pKi",
        "required_columns": ["standardized_smiles", "target_chembl_id", "pKi"],
        "primary_target_columns": ["target_chembl_id"],
    },
    "pairwise_selectivity": {
        "config_key": "input_pairwise_selectivity_path",
        "label_column": "delta_pKi",
        "required_columns": ["standardized_smiles", "kinase_a_chembl_id", "kinase_b_chembl_id", "delta_pKi"],
        "primary_target_columns": ["kinase_a_chembl_id", "kinase_b_chembl_id"],
    },
    "target_vs_panel": {
        "config_key": "input_target_vs_panel_path",
        "label_column": "target_vs_panel_delta_pKi",
        "required_columns": ["standardized_smiles", "target_chembl_id", "target_vs_panel_delta_pKi"],
        "primary_target_columns": ["target_chembl_id"],
    },
}

CLASSIFICATION_LABEL_PRIORITY = [
    "active_inactive_label",
    "strong_weak_label",
    "selective_label",
    "highly_selective_label",
]


@dataclass
class FigureStyle:
    font_family: str
    bold_text: bool
    output_format_primary: str
    palette_name: str
    dpi_png: int


@dataclass
class ObjectiveConfig:
    use_invariant_loss: bool
    use_environment_adversarial_loss: bool
    use_environment_classification_loss: bool
    use_counterfactual_consistency_loss: bool
    use_activity_cliff_regularization: bool


@dataclass
class AppConfig:
    input_annotated_long_path: Path
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
    device: str
    num_workers: int
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    run_multitask_regression: bool
    run_pairwise_selectivity_regression: bool
    run_target_vs_panel_regression: bool
    run_classification_tasks: bool
    molecular_encoder: str
    hidden_dim: int
    num_message_passing_layers: int
    dropout: float
    use_target_identity_embedding: bool
    target_embedding_dim: int
    use_kinase_family_embedding: bool
    kinase_family_embedding_dim: int
    environment_columns: dict[str, list[str]]
    causal_objectives: ObjectiveConfig
    loss_weights: dict[str, float]
    invariant_strategy: str
    adversarial_strategy: str
    environment_pooling_strategy: str
    multitask_strategy: str
    pairwise_strategy: str
    target_vs_panel_strategy: str
    save_trained_models: bool
    save_model_checkpoints: bool
    save_fold_predictions: bool
    save_test_predictions: bool
    save_latent_embeddings: bool
    save_environment_predictions: bool
    save_error_tables: bool
    save_ablation_results: bool
    save_config_snapshot: bool
    run_core_ablations: bool
    ablations: list[str]
    make_figures: bool
    export_svg: bool
    export_png: bool
    export_pdf: bool
    figure_style: FigureStyle
    logs_dir: Path
    configs_used_dir: Path
    project_root: Path

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        if not isinstance(raw, dict):
            raise ValueError("Config YAML must contain a top-level mapping.")
        section = raw.get("script_09")
        if not isinstance(section, dict):
            raise ValueError("Missing required `script_09` section in config.yaml.")
        missing = sorted(REQUIRED_SCRIPT_09_KEYS.difference(section))
        if missing:
            raise ValueError("Missing required script_09 config values: " + ", ".join(missing))

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
            raise ValueError(f"script_09.{key} must be boolean; got {value!r}.")

        def parse_int(key: str, minimum: int | None = None) -> int:
            value = section.get(key)
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"script_09.{key} must be an integer; got {value!r}.") from exc
            if minimum is not None and parsed < minimum:
                raise ValueError(f"script_09.{key} must be >= {minimum}; got {parsed}.")
            return parsed

        def parse_float(key: str, minimum: float | None = None) -> float:
            value = section.get(key)
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"script_09.{key} must be numeric; got {value!r}.") from exc
            if minimum is not None and parsed < minimum:
                raise ValueError(f"script_09.{key} must be >= {minimum}; got {parsed}.")
            return parsed

        env_columns = section.get("environment_columns")
        if not isinstance(env_columns, dict) or not env_columns:
            raise ValueError("script_09.environment_columns must be a non-empty mapping.")
        env_columns = {
            str(key): [str(item).strip() for item in value if str(item).strip()]
            for key, value in env_columns.items()
            if isinstance(value, list)
        }
        if not env_columns:
            raise ValueError("script_09.environment_columns must contain candidate column lists.")

        objectives_raw = section.get("causal_objectives")
        if not isinstance(objectives_raw, dict):
            raise ValueError("script_09.causal_objectives must be a mapping.")
        figure_raw = section.get("figure_style")
        if not isinstance(figure_raw, dict):
            raise ValueError("script_09.figure_style must be a mapping.")
        loss_weights = section.get("loss_weights")
        if not isinstance(loss_weights, dict) or not loss_weights:
            raise ValueError("script_09.loss_weights must be a non-empty mapping.")
        ablations = section.get("ablations")
        if not isinstance(ablations, list):
            raise ValueError("script_09.ablations must be a list.")

        return AppConfig(
            input_annotated_long_path=resolve(section["input_annotated_long_path"]),
            input_regression_long_path=resolve(section["input_regression_long_path"]),
            input_pairwise_selectivity_path=resolve(section["input_pairwise_selectivity_path"]),
            input_target_vs_panel_path=resolve(section["input_target_vs_panel_path"]),
            input_classification_path=resolve(section["input_classification_path"]),
            input_split_manifest_path=resolve(section["input_split_manifest_path"]),
            output_model_root=resolve(section["output_model_root"]),
            output_metrics_root=resolve(section["output_metrics_root"]),
            output_predictions_root=resolve(section["output_predictions_root"]),
            output_figures_root=resolve(section["output_figures_root"]),
            output_report_path=resolve(section["output_report_path"]),
            random_seed=parse_int("random_seed", minimum=0),
            device=str(section["device"]),
            num_workers=parse_int("num_workers", minimum=0),
            batch_size=parse_int("batch_size", minimum=1),
            max_epochs=parse_int("max_epochs", minimum=1),
            early_stopping_patience=parse_int("early_stopping_patience", minimum=1),
            learning_rate=parse_float("learning_rate", minimum=0.0),
            weight_decay=parse_float("weight_decay", minimum=0.0),
            gradient_clip_norm=parse_float("gradient_clip_norm", minimum=0.0),
            run_multitask_regression=parse_bool(section["run_multitask_regression"], "run_multitask_regression"),
            run_pairwise_selectivity_regression=parse_bool(section["run_pairwise_selectivity_regression"], "run_pairwise_selectivity_regression"),
            run_target_vs_panel_regression=parse_bool(section["run_target_vs_panel_regression"], "run_target_vs_panel_regression"),
            run_classification_tasks=parse_bool(section["run_classification_tasks"], "run_classification_tasks"),
            molecular_encoder=str(section["molecular_encoder"]),
            hidden_dim=parse_int("hidden_dim", minimum=8),
            num_message_passing_layers=parse_int("num_message_passing_layers", minimum=1),
            dropout=parse_float("dropout", minimum=0.0),
            use_target_identity_embedding=parse_bool(section["use_target_identity_embedding"], "use_target_identity_embedding"),
            target_embedding_dim=parse_int("target_embedding_dim", minimum=1),
            use_kinase_family_embedding=parse_bool(section["use_kinase_family_embedding"], "use_kinase_family_embedding"),
            kinase_family_embedding_dim=parse_int("kinase_family_embedding_dim", minimum=1),
            environment_columns=env_columns,
            causal_objectives=ObjectiveConfig(
                use_invariant_loss=parse_bool(objectives_raw.get("use_invariant_loss", False), "causal_objectives.use_invariant_loss"),
                use_environment_adversarial_loss=parse_bool(objectives_raw.get("use_environment_adversarial_loss", False), "causal_objectives.use_environment_adversarial_loss"),
                use_environment_classification_loss=parse_bool(objectives_raw.get("use_environment_classification_loss", False), "causal_objectives.use_environment_classification_loss"),
                use_counterfactual_consistency_loss=parse_bool(objectives_raw.get("use_counterfactual_consistency_loss", False), "causal_objectives.use_counterfactual_consistency_loss"),
                use_activity_cliff_regularization=parse_bool(objectives_raw.get("use_activity_cliff_regularization", False), "causal_objectives.use_activity_cliff_regularization"),
            ),
            loss_weights={str(key): float(value) for key, value in loss_weights.items()},
            invariant_strategy=str(section["invariant_strategy"]),
            adversarial_strategy=str(section["adversarial_strategy"]),
            environment_pooling_strategy=str(section["environment_pooling_strategy"]),
            multitask_strategy=str(section["multitask_strategy"]),
            pairwise_strategy=str(section["pairwise_strategy"]),
            target_vs_panel_strategy=str(section["target_vs_panel_strategy"]),
            save_trained_models=parse_bool(section["save_trained_models"], "save_trained_models"),
            save_model_checkpoints=parse_bool(section["save_model_checkpoints"], "save_model_checkpoints"),
            save_fold_predictions=parse_bool(section["save_fold_predictions"], "save_fold_predictions"),
            save_test_predictions=parse_bool(section["save_test_predictions"], "save_test_predictions"),
            save_latent_embeddings=parse_bool(section["save_latent_embeddings"], "save_latent_embeddings"),
            save_environment_predictions=parse_bool(section["save_environment_predictions"], "save_environment_predictions"),
            save_error_tables=parse_bool(section["save_error_tables"], "save_error_tables"),
            save_ablation_results=parse_bool(section["save_ablation_results"], "save_ablation_results"),
            save_config_snapshot=parse_bool(section["save_config_snapshot"], "save_config_snapshot"),
            run_core_ablations=parse_bool(section["run_core_ablations"], "run_core_ablations"),
            ablations=[str(item) for item in ablations],
            make_figures=parse_bool(section["make_figures"], "make_figures"),
            export_svg=parse_bool(section["export_svg"], "export_svg"),
            export_png=parse_bool(section["export_png"], "export_png"),
            export_pdf=parse_bool(section["export_pdf"], "export_pdf"),
            figure_style=FigureStyle(
                font_family=str(figure_raw.get("font_family", "Times New Roman")),
                bold_text=parse_bool(figure_raw.get("bold_text", True), "figure_style.bold_text"),
                output_format_primary=str(figure_raw.get("output_format_primary", "svg")),
                palette_name=str(figure_raw.get("palette_name", "nature_manuscript_palette")),
                dpi_png=int(figure_raw.get("dpi_png", 300)),
            ),
            logs_dir=resolve(raw.get("logs_dir", "logs")),
            configs_used_dir=resolve(raw.get("configs_used_dir", "configs_used")),
            project_root=project_root,
        )


@dataclass
class TaskContext:
    task_name: str
    label_column: str
    task_type: str
    dataframe: pd.DataFrame
    active_label_name: str | None = None


@dataclass
class EnvironmentResolution:
    resolved_columns: dict[str, str]
    disabled_objectives: list[str]
    warnings: list[str]


class OptionalDependencyError(RuntimeError):
    """Raised when runtime dependencies needed for training are missing."""


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config YAML must contain a top-level mapping.")
    return data


def setup_logging(logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{SCRIPT_NAME}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_path


def save_config_snapshot(cfg: AppConfig, config_path: Path) -> Path | None:
    if not cfg.save_config_snapshot:
        return None
    cfg.configs_used_dir.mkdir(parents=True, exist_ok=True)
    destination = cfg.configs_used_dir / f"{SCRIPT_NAME}_config.yaml"
    destination.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {description} file not found: {path}")


def load_required_dataframe(path: Path, description: str) -> pd.DataFrame:
    require_file(path, description)
    logging.info("Loading %s from %s", description, path)
    return pd.read_csv(path)


def resolve_column(df: pd.DataFrame, logical_name: str, aliases: Sequence[str] | None = None) -> str:
    candidates = aliases if aliases is not None else COLUMN_ALIASES.get(logical_name, (logical_name,))
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Required column `{logical_name}` not found. Checked candidates: {', '.join(candidates)}")


def normalize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    for logical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in renamed.columns:
                renamed = renamed.rename(columns={alias: logical_name})
                break
    if "compound_id" not in renamed.columns:
        if "standardized_smiles" in renamed.columns:
            renamed["compound_id"] = renamed["standardized_smiles"]
        else:
            raise ValueError("Input data require either `compound_id` or `standardized_smiles`.")
    if "standardized_smiles" not in renamed.columns:
        raise ValueError("Graph construction requires `standardized_smiles` but it was not found.")
    return renamed


def ensure_columns(df: pd.DataFrame, required: Iterable[str], description: str) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"{description} missing required columns: {', '.join(missing)}")


def load_split_manifest(path: Path) -> pd.DataFrame:
    manifest = load_required_dataframe(path, "split manifest")
    required = {"task_name", "split_strategy", "split_id", "fold_id", "output_assignment_path"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError("Split manifest missing required columns: " + ", ".join(missing))
    return manifest


def load_assignment_table(path_value: str | None, project_root: Path) -> pd.DataFrame:
    if path_value in (None, "", np.nan):
        raise FileNotFoundError("Split manifest row does not contain an output_assignment_path.")
    path = Path(path_value)
    resolved = path if path.is_absolute() else project_root / path
    return load_required_dataframe(resolved, "split assignment")


def ensure_assignment_columns(df: pd.DataFrame, task_name: str) -> None:
    required = {"row_uid", "split_label", "split_strategy", "split_id", "fold_id"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Assignment file for {task_name} missing columns: {', '.join(missing)}")


def task_enabled(task_name: str, cfg: AppConfig) -> bool:
    return {
        "multitask_regression": cfg.run_multitask_regression,
        "pairwise_selectivity": cfg.run_pairwise_selectivity_regression,
        "target_vs_panel": cfg.run_target_vs_panel_regression,
        "classification": cfg.run_classification_tasks,
    }[task_name]


def load_task_table(task_name: str, cfg: AppConfig) -> TaskContext | list[TaskContext]:
    if task_name == "classification":
        df = normalize_common_columns(load_required_dataframe(cfg.input_classification_path, "classification task table"))
        if "row_uid" not in df.columns:
            raise ValueError("Classification task table must contain `row_uid` from Step-05/06.")
        outputs: list[TaskContext] = []
        for label in CLASSIFICATION_LABEL_PRIORITY:
            if label in df.columns and df[label].notna().sum() > 0:
                label_df = df[df[label].notna()].copy()
                label_df[label] = pd.to_numeric(label_df[label], errors="coerce")
                label_df = label_df[label_df[label].isin([0, 1])].copy()
                if len(label_df) >= 20:
                    outputs.append(TaskContext(task_name, label, "classification", label_df, active_label_name=label))
        if not outputs:
            raise ValueError("Classification requested, but no usable binary label columns were found.")
        return outputs

    spec = REGRESSION_TASKS[task_name]
    input_path = getattr(cfg, spec["config_key"])
    df = normalize_common_columns(load_required_dataframe(input_path, f"{task_name} task table"))
    if "row_uid" not in df.columns:
        raise ValueError(f"{task_name} task table must contain `row_uid` from Step-05/06.")
    ensure_columns(df, spec["required_columns"] + ["row_uid", "compound_id"], f"{task_name} task table")
    df[spec["label_column"]] = pd.to_numeric(df[spec["label_column"]], errors="coerce")
    df = df[df[spec["label_column"]].notna()].copy()
    return TaskContext(task_name, spec["label_column"], "regression", df)


def resolve_environment_columns(df: pd.DataFrame, cfg: AppConfig) -> EnvironmentResolution:
    resolved: dict[str, str] = {}
    warnings_list: list[str] = []
    disabled: list[str] = []
    for env_name, candidates in cfg.environment_columns.items():
        column = next((candidate for candidate in candidates if candidate in df.columns), None)
        if column is None:
            warnings_list.append(f"Environment `{env_name}` unavailable. Checked: {', '.join(candidates)}")
        else:
            resolved[env_name] = column
            logging.info("Resolved environment `%s` to column `%s`", env_name, column)
    if not resolved:
        if cfg.causal_objectives.use_invariant_loss:
            disabled.append("use_invariant_loss")
        if cfg.causal_objectives.use_environment_adversarial_loss:
            disabled.append("use_environment_adversarial_loss")
        if cfg.causal_objectives.use_environment_classification_loss:
            disabled.append("use_environment_classification_loss")
        warnings_list.append("No configured environment columns were available; all environment-dependent objectives will be disabled.")
    if "activity_cliff" not in resolved and cfg.causal_objectives.use_activity_cliff_regularization:
        disabled.append("use_activity_cliff_regularization")
        warnings_list.append("Activity cliff environment unavailable; disabling activity-cliff regularization.")
    return EnvironmentResolution(resolved, sorted(set(disabled)), warnings_list)


def merge_annotated_columns(task_df: pd.DataFrame, annotated_df: pd.DataFrame) -> pd.DataFrame:
    base = annotated_df.copy()
    base = normalize_common_columns(base)
    if "compound_id" not in base.columns:
        base["compound_id"] = base["standardized_smiles"]
    keep = [col for col in base.columns if col not in task_df.columns or col in {"compound_id", "standardized_smiles"}]
    keep = sorted(set(keep + ["compound_id", "standardized_smiles"]))
    annotation_subset = base[keep].drop_duplicates(subset=["compound_id"], keep="first")
    merged = task_df.merge(annotation_subset, on=["compound_id", "standardized_smiles"], how="left", suffixes=("", "_annot"))
    return merged


def import_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset
        from torch_geometric.data import Batch, Data
        from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
    except Exception as exc:
        raise OptionalDependencyError(
            "Step-09 requires PyTorch, PyTorch Geometric, and their dependencies at runtime. "
            f"Import failure: {exc}"
        ) from exc
    try:
        from rdkit import Chem
    except Exception as exc:
        raise OptionalDependencyError(f"Step-09 requires RDKit at runtime. Import failure: {exc}") from exc
    try:
        from scipy import stats
    except Exception as exc:
        raise OptionalDependencyError(f"Step-09 requires SciPy at runtime. Import failure: {exc}") from exc
    return {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Dataset": Dataset,
        "Batch": Batch,
        "Data": Data,
        "GCNConv": GCNConv,
        "GINConv": GINConv,
        "global_mean_pool": global_mean_pool,
        "Chem": Chem,
        "stats": stats,
    }


def atom_features(atom: Any) -> list[float]:
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        int(atom.IsInRing()),
        atom.GetImplicitValence(),
        atom.GetMass() / 100.0,
    ]


def bond_features(bond: Any) -> list[float]:
    bond_type = bond.GetBondType()
    return [
        float(bond_type == bond_type.SINGLE),
        float(bond_type == bond_type.DOUBLE),
        float(bond_type == bond_type.TRIPLE),
        float(bond_type == bond_type.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def build_graph_cache(smiles_list: Sequence[str], deps: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = deps["torch"]
    Data = deps["Data"]
    Chem = deps["Chem"]
    cache: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for smiles in sorted(set(str(value) for value in smiles_list if isinstance(value, str) and value.strip())):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failures[smiles] = "RDKit MolFromSmiles returned None"
            continue
        nodes = [atom_features(atom) for atom in mol.GetAtoms()]
        if not nodes:
            failures[smiles] = "molecule contains zero atoms"
            continue
        edge_index: list[list[int]] = []
        edge_attr: list[list[float]] = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            feats = bond_features(bond)
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([feats, feats])
        if edge_index:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, 6), dtype=torch.float32)
        cache[smiles] = Data(
            x=torch.tensor(nodes, dtype=torch.float32),
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            num_nodes=len(nodes),
        )
    metadata = {
        "num_unique_smiles": len(set(smiles_list)),
        "num_success": len(cache),
        "num_failures": len(failures),
        "failures": failures,
    }
    return cache, metadata


class Vocabulary:
    def __init__(self, values: Sequence[Any], unk_token: str = "<UNK>") -> None:
        cleaned = [str(value) if pd.notna(value) and str(value) != "" else unk_token for value in values]
        unique = sorted(set(cleaned))
        if unk_token not in unique:
            unique.insert(0, unk_token)
        self.unk_token = unk_token
        self.values = unique
        self.mapping = {value: idx for idx, value in enumerate(unique)}

    def encode(self, value: Any) -> int:
        token = str(value) if pd.notna(value) and str(value) != "" else self.unk_token
        return self.mapping.get(token, self.mapping[self.unk_token])

    def to_dict(self) -> dict[str, int]:
        return dict(self.mapping)


class GradientReversalFunction:  # pragma: no cover - tiny autograd wrapper
    @staticmethod
    def apply(x: Any, lambd: float) -> Any:
        import torch

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, inputs: Any, coeff: float) -> Any:
                ctx.coeff = coeff
                return inputs.view_as(inputs)

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> tuple[Any, None]:
                return grad_output.neg() * ctx.coeff, None

        return _Fn.apply(x, lambd)


def make_dataset_class(deps: dict[str, Any]) -> type:
    torch = deps["torch"]
    Dataset = deps["Dataset"]

    class CausalTaskDataset(Dataset):
        def __init__(
            self,
            frame: pd.DataFrame,
            graph_cache: dict[str, Any],
            target_vocab: Vocabulary,
            family_vocab: Vocabulary,
            environment_vocab: Vocabulary | None,
            task_name: str,
            label_column: str,
            task_type: str,
            environment_column: str | None,
        ) -> None:
            self.frame = frame.reset_index(drop=True)
            self.graph_cache = graph_cache
            self.target_vocab = target_vocab
            self.family_vocab = family_vocab
            self.environment_vocab = environment_vocab
            self.task_name = task_name
            self.label_column = label_column
            self.task_type = task_type
            self.environment_column = environment_column

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            row = self.frame.iloc[idx]
            graph = self.graph_cache[row["standardized_smiles"]].clone()
            graph.row_uid = str(row["row_uid"])
            label_value = row[self.label_column]
            if self.task_type == "classification":
                label = torch.tensor(float(label_value), dtype=torch.float32)
            else:
                label = torch.tensor(float(label_value), dtype=torch.float32)
            family_a = row.get("kinase_family", row.get("kinase_family_a", "<UNK>"))
            family_b = row.get("kinase_family_b", family_a)
            sample = {
                "graph": graph,
                "label": label,
                "row_uid": str(row["row_uid"]),
                "compound_id": str(row["compound_id"]),
                "target_id": self.target_vocab.encode(row.get("target_chembl_id", "<UNK>")),
                "target_id_a": self.target_vocab.encode(row.get("kinase_a_chembl_id", row.get("target_chembl_id", "<UNK>"))),
                "target_id_b": self.target_vocab.encode(row.get("kinase_b_chembl_id", row.get("target_chembl_id", "<UNK>"))),
                "family_id": self.family_vocab.encode(family_a),
                "family_id_a": self.family_vocab.encode(family_a),
                "family_id_b": self.family_vocab.encode(family_b),
                "environment_id": -100,
                "environment_value": None,
                "activity_cliff_flag": float(row.get("activity_cliff_flag", 0) or 0),
                "metadata": row.to_dict(),
            }
            if self.environment_column is not None and self.environment_vocab is not None:
                env_value = row.get(self.environment_column)
                sample["environment_id"] = self.environment_vocab.encode(env_value)
                sample["environment_value"] = env_value
            return sample

    return CausalTaskDataset


def make_collate_fn(deps: dict[str, Any]):
    torch = deps["torch"]
    Batch = deps["Batch"]

    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch_graph = Batch.from_data_list([sample["graph"] for sample in samples])
        return {
            "graph": batch_graph,
            "label": torch.stack([sample["label"] for sample in samples]),
            "row_uid": [sample["row_uid"] for sample in samples],
            "compound_id": [sample["compound_id"] for sample in samples],
            "target_id": torch.tensor([sample["target_id"] for sample in samples], dtype=torch.long),
            "target_id_a": torch.tensor([sample["target_id_a"] for sample in samples], dtype=torch.long),
            "target_id_b": torch.tensor([sample["target_id_b"] for sample in samples], dtype=torch.long),
            "family_id": torch.tensor([sample["family_id"] for sample in samples], dtype=torch.long),
            "family_id_a": torch.tensor([sample["family_id_a"] for sample in samples], dtype=torch.long),
            "family_id_b": torch.tensor([sample["family_id_b"] for sample in samples], dtype=torch.long),
            "environment_id": torch.tensor([sample["environment_id"] for sample in samples], dtype=torch.long),
            "activity_cliff_flag": torch.tensor([sample["activity_cliff_flag"] for sample in samples], dtype=torch.float32),
            "metadata": [sample["metadata"] for sample in samples],
        }

    return collate


def make_model_class(deps: dict[str, Any]) -> type:
    torch = deps["torch"]
    nn = deps["nn"]
    GCNConv = deps["GCNConv"]
    GINConv = deps["GINConv"]
    global_mean_pool = deps["global_mean_pool"]

    class MoleculeEncoder(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, encoder_name: str) -> None:
            super().__init__()
            self.encoder_name = encoder_name.lower()
            self.dropout = nn.Dropout(dropout)
            self.input_proj = nn.Linear(in_dim, hidden_dim)
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                if self.encoder_name == "gcn":
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                else:
                    mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                    self.convs.append(GINConv(mlp))
            self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))

        def forward(self, data: Any) -> Any:
            x, edge_index = data.x, data.edge_index
            x = self.input_proj(x)
            for conv, norm in zip(self.convs, self.norms):
                x = conv(x, edge_index)
                x = norm(x)
                x = torch.relu(x)
                x = self.dropout(x)
            return global_mean_pool(x, data.batch)

    class CausalEnvironmentAwareModel(nn.Module):
        def __init__(
            self,
            in_dim: int,
            cfg: AppConfig,
            num_targets: int,
            num_families: int,
            num_envs: int,
            task_name: str,
            task_type: str,
            grl_lambda: float,
        ) -> None:
            super().__init__()
            self.cfg = cfg
            self.task_name = task_name
            self.task_type = task_type
            self.grl_lambda = grl_lambda
            self.encoder = MoleculeEncoder(in_dim, cfg.hidden_dim, cfg.num_message_passing_layers, cfg.dropout, cfg.molecular_encoder)
            representation_dim = cfg.hidden_dim
            self.target_embedding = None
            self.family_embedding = None
            if cfg.use_target_identity_embedding:
                self.target_embedding = nn.Embedding(max(num_targets, 1), cfg.target_embedding_dim)
                representation_dim += cfg.target_embedding_dim
                if task_name == "pairwise_selectivity":
                    representation_dim += cfg.target_embedding_dim
            if cfg.use_kinase_family_embedding:
                self.family_embedding = nn.Embedding(max(num_families, 1), cfg.kinase_family_embedding_dim)
                representation_dim += cfg.kinase_family_embedding_dim
                if task_name == "pairwise_selectivity":
                    representation_dim += cfg.kinase_family_embedding_dim
            self.shared = nn.Sequential(
                nn.Linear(representation_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.ReLU(),
            )
            output_dim = 1
            self.prediction_head = nn.Sequential(
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim // 2, output_dim),
            )
            self.environment_head = None
            self.adversarial_environment_head = None
            if num_envs > 1:
                self.environment_head = nn.Sequential(
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(cfg.hidden_dim // 2, num_envs),
                )
                self.adversarial_environment_head = nn.Sequential(
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(cfg.hidden_dim // 2, num_envs),
                )

        def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
            z = self.encoder(batch["graph"])
            parts = [z]
            if self.target_embedding is not None:
                if self.task_name == "pairwise_selectivity":
                    parts.extend([self.target_embedding(batch["target_id_a"]), self.target_embedding(batch["target_id_b"])])
                else:
                    parts.append(self.target_embedding(batch["target_id"]))
            if self.family_embedding is not None:
                if self.task_name == "pairwise_selectivity":
                    parts.extend([self.family_embedding(batch["family_id_a"]), self.family_embedding(batch["family_id_b"])])
                else:
                    parts.append(self.family_embedding(batch["family_id"]))
            joint = torch.cat(parts, dim=1)
            latent = self.shared(joint)
            predictions = self.prediction_head(latent).squeeze(-1)
            output = {"prediction": predictions, "latent": latent}
            if self.environment_head is not None:
                output["environment_logits"] = self.environment_head(latent)
            if self.adversarial_environment_head is not None:
                reversed_latent = GradientReversalFunction.apply(latent, self.grl_lambda)
                output["environment_adversarial_logits"] = self.adversarial_environment_head(reversed_latent)
            return output

    return CausalEnvironmentAwareModel


def choose_device(requested: str, deps: dict[str, Any]) -> Any:
    torch = deps["torch"]
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_vocabularies(train_df: pd.DataFrame, environment_column: str | None) -> dict[str, Vocabulary | None]:
    target_values = list(train_df.get("target_chembl_id", pd.Series(dtype=object)).fillna("<UNK>"))
    target_values += list(train_df.get("kinase_a_chembl_id", pd.Series(dtype=object)).fillna("<UNK>"))
    target_values += list(train_df.get("kinase_b_chembl_id", pd.Series(dtype=object)).fillna("<UNK>"))
    family_values = list(train_df.get("kinase_family", pd.Series(dtype=object)).fillna("<UNK>"))
    family_values += list(train_df.get("kinase_family_a", pd.Series(dtype=object)).fillna("<UNK>"))
    family_values += list(train_df.get("kinase_family_b", pd.Series(dtype=object)).fillna("<UNK>"))
    env_vocab = Vocabulary(train_df[environment_column].fillna("<UNK>").tolist()) if environment_column else None
    return {
        "target_vocab": Vocabulary(target_values),
        "family_vocab": Vocabulary(family_values),
        "environment_vocab": env_vocab,
    }


def compute_invariant_penalty(prediction: Any, target: Any, environment_id: Any, torch_mod: Any) -> Any:
    mask = environment_id >= 0
    if int(mask.sum().item()) <= 1:
        return torch_mod.tensor(0.0, device=prediction.device)
    errors = (prediction - target).pow(2)
    penalties = []
    for env_value in torch_mod.unique(environment_id[mask]):
        env_mask = environment_id == env_value
        if int(env_mask.sum().item()) < 2:
            continue
        penalties.append(errors[env_mask].mean())
    if len(penalties) <= 1:
        return torch_mod.tensor(0.0, device=prediction.device)
    stacked = torch_mod.stack(penalties)
    return stacked.var(unbiased=False)


def compute_activity_cliff_regularization(prediction: Any, target: Any, cliff_flag: Any, torch_mod: Any) -> Any:
    mask = cliff_flag > 0.5
    if int(mask.sum().item()) == 0:
        return torch_mod.tensor(0.0, device=prediction.device)
    return ((prediction[mask] - target[mask]).abs()).mean()


def compute_metrics(task_type: str, observed: np.ndarray, predicted: np.ndarray, deps: dict[str, Any]) -> dict[str, float]:
    stats = deps["stats"]
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    metrics: dict[str, float] = {}
    if len(observed) == 0:
        return metrics
    if task_type == "regression":
        mse = float(np.mean((observed - predicted) ** 2))
        mae = float(np.mean(np.abs(observed - predicted)))
        rmse = math.sqrt(mse)
        ss_tot = float(np.sum((observed - observed.mean()) ** 2))
        ss_res = float(np.sum((observed - predicted) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        spearman = float(stats.spearmanr(observed, predicted).statistic) if len(observed) > 1 else float("nan")
        pearson = float(stats.pearsonr(observed, predicted).statistic) if len(observed) > 1 else float("nan")
        metrics.update({"RMSE": rmse, "MAE": mae, "R2": r2, "Spearman": spearman, "Pearson": pearson})
    else:
        try:
            from sklearn.metrics import (
                accuracy_score,
                average_precision_score,
                balanced_accuracy_score,
                f1_score,
                matthews_corrcoef,
                roc_auc_score,
            )
        except Exception as exc:
            raise OptionalDependencyError(f"Classification metrics require scikit-learn: {exc}") from exc
        probs = 1.0 / (1.0 + np.exp(-predicted))
        labels = (probs >= 0.5).astype(int)
        metrics.update(
            {
                "ROC_AUC": float(roc_auc_score(observed, probs)) if len(np.unique(observed)) > 1 else float("nan"),
                "PR_AUC": float(average_precision_score(observed, probs)) if len(np.unique(observed)) > 1 else float("nan"),
                "accuracy": float(accuracy_score(observed, labels)),
                "balanced_accuracy": float(balanced_accuracy_score(observed, labels)),
                "F1": float(f1_score(observed, labels, zero_division=0)),
                "MCC": float(matthews_corrcoef(observed, labels)) if len(np.unique(labels)) > 1 else 0.0,
            }
        )
    return metrics


def iterate_batches(model: Any, loader: Any, optimizer: Any | None, device: Any, task_type: str, cfg: AppConfig, deps: dict[str, Any]) -> tuple[float, pd.DataFrame, np.ndarray | None, np.ndarray | None]:
    torch = deps["torch"]
    F = deps["F"]
    total_loss = 0.0
    num_items = 0
    rows: list[dict[str, Any]] = []
    latent_rows: list[np.ndarray] = []
    env_pred_rows: list[np.ndarray] = []
    training = optimizer is not None
    model.train(training)
    for batch in loader:
        batch["graph"] = batch["graph"].to(device)
        for key in ["label", "target_id", "target_id_a", "target_id_b", "family_id", "family_id_a", "family_id_b", "environment_id", "activity_cliff_flag"]:
            batch[key] = batch[key].to(device)
        output = model(batch)
        prediction = output["prediction"]
        target = batch["label"]
        if task_type == "classification":
            prediction_loss = F.binary_cross_entropy_with_logits(prediction, target)
            scored_prediction = prediction.detach().cpu().numpy()
        else:
            prediction_loss = F.smooth_l1_loss(prediction, target)
            scored_prediction = prediction.detach().cpu().numpy()
        loss = cfg.loss_weights.get("prediction_loss", 1.0) * prediction_loss
        if cfg.causal_objectives.use_invariant_loss:
            loss = loss + cfg.loss_weights.get("invariant_loss", 0.0) * compute_invariant_penalty(prediction, target, batch["environment_id"], torch)
        if cfg.causal_objectives.use_environment_classification_loss and "environment_logits" in output:
            mask = batch["environment_id"] >= 0
            if int(mask.sum().item()) > 1:
                env_loss = F.cross_entropy(output["environment_logits"][mask], batch["environment_id"][mask])
                loss = loss + cfg.loss_weights.get("environment_classification_loss", 0.0) * env_loss
                env_pred_rows.append(torch.softmax(output["environment_logits"], dim=1).detach().cpu().numpy())
        if cfg.causal_objectives.use_environment_adversarial_loss and "environment_adversarial_logits" in output:
            mask = batch["environment_id"] >= 0
            if int(mask.sum().item()) > 1:
                adv_loss = F.cross_entropy(output["environment_adversarial_logits"][mask], batch["environment_id"][mask])
                loss = loss + cfg.loss_weights.get("environment_adversarial_loss", 0.0) * adv_loss
        if cfg.causal_objectives.use_activity_cliff_regularization:
            cliff_loss = compute_activity_cliff_regularization(prediction, target, batch["activity_cliff_flag"], torch)
            loss = loss + cfg.loss_weights.get("activity_cliff_regularization_loss", 0.0) * cliff_loss
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            optimizer.step()
        total_loss += float(loss.item()) * len(batch["row_uid"])
        num_items += len(batch["row_uid"])
        latent_rows.append(output["latent"].detach().cpu().numpy())
        pred_np = scored_prediction
        obs_np = target.detach().cpu().numpy()
        env_np = batch["environment_id"].detach().cpu().numpy()
        cliff_np = batch["activity_cliff_flag"].detach().cpu().numpy()
        for idx, row_uid in enumerate(batch["row_uid"]):
            metadata = dict(batch["metadata"][idx])
            metadata.update(
                {
                    "row_uid": row_uid,
                    "compound_id": batch["compound_id"][idx],
                    "observed": float(obs_np[idx]),
                    "predicted": float(pred_np[idx]),
                    "environment_id": int(env_np[idx]),
                    "activity_cliff_flag": float(cliff_np[idx]),
                }
            )
            rows.append(metadata)
    average_loss = total_loss / max(num_items, 1)
    predictions_df = pd.DataFrame(rows)
    latent = np.vstack(latent_rows) if latent_rows else None
    env_preds = np.vstack(env_pred_rows) if env_pred_rows else None
    return average_loss, predictions_df, latent, env_preds


def train_one_configuration(
    task_context: TaskContext,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: AppConfig,
    resolved_environment_column: str | None,
    model_name: str,
    ablation_name: str,
    metadata: dict[str, Any],
    deps: dict[str, Any],
) -> dict[str, Any]:
    torch = deps["torch"]
    DataLoader = __import__("torch.utils.data", fromlist=["DataLoader"]).DataLoader
    DatasetClass = make_dataset_class(deps)
    collate_fn = make_collate_fn(deps)
    ModelClass = make_model_class(deps)
    device = choose_device(cfg.device, deps)
    vocabularies = build_vocabularies(train_df, resolved_environment_column)
    env_vocab = vocabularies["environment_vocab"]
    train_df = train_df[train_df["standardized_smiles"].isin(metadata["graph_cache_keys"])].copy()
    valid_df = valid_df[valid_df["standardized_smiles"].isin(metadata["graph_cache_keys"])].copy()
    test_df = test_df[test_df["standardized_smiles"].isin(metadata["graph_cache_keys"])].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Training or test split became empty after graph filtering.")
    train_dataset = DatasetClass(train_df, metadata["graph_cache"], vocabularies["target_vocab"], vocabularies["family_vocab"], env_vocab, task_context.task_name, task_context.label_column, task_context.task_type, resolved_environment_column)
    valid_dataset = DatasetClass(valid_df, metadata["graph_cache"], vocabularies["target_vocab"], vocabularies["family_vocab"], env_vocab, task_context.task_name, task_context.label_column, task_context.task_type, resolved_environment_column)
    test_dataset = DatasetClass(test_df, metadata["graph_cache"], vocabularies["target_vocab"], vocabularies["family_vocab"], env_vocab, task_context.task_name, task_context.label_column, task_context.task_type, resolved_environment_column)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)
    sample_graph = metadata["graph_cache"][next(iter(metadata["graph_cache_keys"]))]
    model = ModelClass(
        in_dim=int(sample_graph.x.shape[1]),
        cfg=cfg,
        num_targets=len(vocabularies["target_vocab"].mapping),
        num_families=len(vocabularies["family_vocab"].mapping),
        num_envs=len(env_vocab.mapping) if env_vocab is not None else 0,
        task_name=task_context.task_name,
        task_type=task_context.task_type,
        grl_lambda=cfg.loss_weights.get("environment_adversarial_loss", 0.0),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, cfg.max_epochs + 1):
        train_loss, _, _, _ = iterate_batches(model, train_loader, optimizer, device, task_context.task_type, cfg, deps)
        valid_loss, valid_predictions, _, _ = iterate_batches(model, valid_loader, None, device, task_context.task_type, cfg, deps) if len(valid_dataset) > 0 else (train_loss, pd.DataFrame(), None, None)
        logging.info(
            "task=%s ablation=%s split=%s fold=%s epoch=%d train_loss=%.5f valid_loss=%.5f",
            task_context.active_label_name or task_context.task_name,
            ablation_name,
            metadata["split_strategy"],
            metadata["fold_id"],
            epoch,
            train_loss,
            valid_loss,
        )
        metric_for_selection = valid_loss if not valid_predictions.empty else train_loss
        if metric_for_selection + 1e-8 < best_metric:
            best_metric = metric_for_selection
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            if cfg.save_model_checkpoints:
                checkpoint_path = metadata["model_dir"] / f"best_checkpoint_epoch_{epoch}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": epoch, "state_dict": best_state}, checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= cfg.early_stopping_patience:
            logging.info("Early stopping triggered at epoch %d", epoch)
            break
    model.load_state_dict(best_state)
    test_loss, test_predictions, test_latent, environment_outputs = iterate_batches(model, test_loader, None, device, task_context.task_type, cfg, deps)
    test_metrics = compute_metrics(task_context.task_type, test_predictions["observed"].to_numpy(), test_predictions["predicted"].to_numpy(), deps)
    valid_metrics = compute_metrics(task_context.task_type, valid_predictions["observed"].to_numpy(), valid_predictions["predicted"].to_numpy(), deps) if not valid_predictions.empty else {}
    output = {
        "model": model,
        "test_loss": test_loss,
        "test_predictions": test_predictions,
        "test_latent": test_latent,
        "environment_outputs": environment_outputs,
        "test_metrics": test_metrics,
        "valid_metrics": valid_metrics,
        "best_epoch": best_epoch,
        "vocabularies": {key: value.to_dict() if value is not None else None for key, value in vocabularies.items()},
    }
    return output


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return path


def save_predictions(predictions: pd.DataFrame, output_root: Path, metadata: dict[str, Any]) -> Path:
    path = output_root / metadata["task_name"] / metadata["model_name"] / metadata["split_strategy"] / f"{metadata['split_id']}__{metadata['fold_id']}__{metadata['ablation_name']}__test_predictions.csv"
    return save_dataframe(predictions, path)


def save_error_table(predictions: pd.DataFrame, task_type: str, output_root: Path, metadata: dict[str, Any]) -> Path:
    errors = predictions.copy()
    if not errors.empty:
        if task_type == "classification":
            probs = 1.0 / (1.0 + np.exp(-errors["predicted"].to_numpy()))
            errors["absolute_error"] = np.abs(errors["observed"].to_numpy() - probs)
        else:
            errors["absolute_error"] = np.abs(errors["observed"] - errors["predicted"])
        errors = errors.sort_values("absolute_error", ascending=False, kind="mergesort")
    path = output_root / metadata["task_name"] / metadata["model_name"] / metadata["split_strategy"] / f"{metadata['split_id']}__{metadata['fold_id']}__{metadata['ablation_name']}__error_table.csv"
    return save_dataframe(errors, path)


def save_latent_embeddings(latent: np.ndarray | None, predictions: pd.DataFrame, output_root: Path, metadata: dict[str, Any]) -> Path | None:
    if latent is None or predictions.empty:
        return None
    frame = predictions[["row_uid", "compound_id"]].copy()
    for idx in range(latent.shape[1]):
        frame[f"latent_{idx:03d}"] = latent[:, idx]
    path = output_root / metadata["task_name"] / metadata["model_name"] / metadata["split_strategy"] / f"{metadata['split_id']}__{metadata['fold_id']}__{metadata['ablation_name']}__latent_embeddings.csv"
    return save_dataframe(frame, path)


def save_environment_outputs(env_outputs: np.ndarray | None, predictions: pd.DataFrame, output_root: Path, metadata: dict[str, Any]) -> Path | None:
    if env_outputs is None or predictions.empty:
        return None
    frame = predictions[["row_uid", "compound_id", "environment_id"]].copy()
    for idx in range(env_outputs.shape[1]):
        frame[f"environment_prob_{idx:03d}"] = env_outputs[:, idx]
    path = output_root / metadata["task_name"] / metadata["model_name"] / metadata["split_strategy"] / f"{metadata['split_id']}__{metadata['fold_id']}__{metadata['ablation_name']}__environment_predictions.csv"
    return save_dataframe(frame, path)


def summarize_by_environment(predictions: pd.DataFrame, task_type: str, env_column: str | None, deps: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    if env_column is None or env_column not in predictions.columns:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for env_value, group in predictions.groupby(env_column, dropna=False):
        if len(group) < 3:
            continue
        metrics = compute_metrics(task_type, group["observed"].to_numpy(), group["predicted"].to_numpy(), deps)
        row = {"environment_column": env_column, "environment_value": env_value, "n_rows": len(group), **metadata, **metrics}
        records.append(row)
    return pd.DataFrame(records)


def summarize_activity_cliff(predictions: pd.DataFrame, task_type: str, deps: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    if "activity_cliff_flag" not in predictions.columns:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for cliff_value, group in predictions.groupby("activity_cliff_flag"):
        if len(group) < 3:
            continue
        metrics = compute_metrics(task_type, group["observed"].to_numpy(), group["predicted"].to_numpy(), deps)
        records.append({"activity_cliff_flag": cliff_value, "n_rows": len(group), **metadata, **metrics})
    return pd.DataFrame(records)


def aggregate_metrics(df: pd.DataFrame, metric_columns: Sequence[str], group_columns: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(group_columns))
    aggregations = {metric: ["mean", "std"] for metric in metric_columns if metric in df.columns}
    if not aggregations:
        return pd.DataFrame()
    summary = df.groupby(list(group_columns), dropna=False).agg(aggregations)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def apply_ablation(base_cfg: AppConfig, ablation_name: str) -> AppConfig:
    cfg = copy.deepcopy(base_cfg)
    if ablation_name == "no_environment_objectives":
        cfg.causal_objectives.use_invariant_loss = False
        cfg.causal_objectives.use_environment_adversarial_loss = False
        cfg.causal_objectives.use_environment_classification_loss = False
    elif ablation_name == "no_adversarial_loss":
        cfg.causal_objectives.use_environment_adversarial_loss = False
    elif ablation_name == "no_invariant_loss":
        cfg.causal_objectives.use_invariant_loss = False
    elif ablation_name == "no_activity_cliff_regularization":
        cfg.causal_objectives.use_activity_cliff_regularization = False
    elif ablation_name == "no_target_embedding":
        cfg.use_target_identity_embedding = False
    return cfg


def configure_matplotlib(style: FigureStyle) -> tuple[Any, Any] | tuple[None, None]:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return None, None
    plt.rcParams["font.family"] = style.font_family
    plt.rcParams["font.weight"] = "bold" if style.bold_text else "normal"
    plt.rcParams["axes.labelweight"] = "bold" if style.bold_text else "normal"
    plt.rcParams["axes.titleweight"] = "bold" if style.bold_text else "normal"
    sns.set_theme(style="whitegrid")
    return plt, sns


def export_figure(fig: Any, path_root: Path, cfg: AppConfig) -> list[str]:
    outputs: list[str] = []
    path_root.parent.mkdir(parents=True, exist_ok=True)
    primary = path_root.with_suffix(f".{cfg.figure_style.output_format_primary}")
    fig.savefig(primary, bbox_inches="tight", dpi=cfg.figure_style.dpi_png)
    outputs.append(str(primary))
    if cfg.export_png and primary.suffix.lower() != ".png":
        png_path = path_root.with_suffix(".png")
        fig.savefig(png_path, bbox_inches="tight", dpi=cfg.figure_style.dpi_png)
        outputs.append(str(png_path))
    if cfg.export_pdf and primary.suffix.lower() != ".pdf":
        pdf_path = path_root.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        outputs.append(str(pdf_path))
    return outputs


def generate_figures(
    regression_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    predictions_all: pd.DataFrame,
    cfg: AppConfig,
) -> dict[str, list[str]]:
    outputs: dict[str, list[str]] = {}
    if not cfg.make_figures:
        return outputs
    plt, sns = configure_matplotlib(cfg.figure_style)
    if plt is None or sns is None:
        logging.warning("Matplotlib/Seaborn unavailable; skipping figure generation.")
        return outputs
    palette = NATURE_PALETTE
    if not regression_summary.empty and "RMSE_mean" in regression_summary.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_df = regression_summary.copy()
        sns.barplot(data=plot_df, x="split_strategy", y="RMSE_mean", hue="model_name", palette=palette, ax=ax)
        ax.set_title("Causal model performance across split strategies")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Split strategy")
        ax.tick_params(axis="x", rotation=30)
        outputs["split_strategy_performance"] = export_figure(fig, cfg.output_figures_root / "causal_model_performance_by_split", cfg)
        plt.close(fig)
        save_dataframe(plot_df, cfg.output_figures_root / "source_data" / "causal_model_performance_by_split.csv")
    if not ablation_summary.empty and "RMSE_mean" in ablation_summary.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=ablation_summary, x="ablation_name", y="RMSE_mean", hue="task_name", palette=palette, ax=ax)
        ax.set_title("Core causal ablations")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Ablation")
        ax.tick_params(axis="x", rotation=35)
        outputs["ablation_comparison"] = export_figure(fig, cfg.output_figures_root / "causal_ablation_comparison", cfg)
        plt.close(fig)
        save_dataframe(ablation_summary, cfg.output_figures_root / "source_data" / "causal_ablation_comparison.csv")
    regression_predictions = predictions_all[predictions_all["task_type"] == "regression"].copy() if not predictions_all.empty else pd.DataFrame()
    if not regression_predictions.empty:
        top_group = regression_predictions.groupby(["task_name", "split_strategy"], dropna=False).size().sort_values(ascending=False).head(1)
        if not top_group.empty:
            task_name, split_strategy = top_group.index[0]
            subset = regression_predictions[(regression_predictions["task_name"] == task_name) & (regression_predictions["split_strategy"] == split_strategy)]
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.scatterplot(data=subset, x="observed", y="predicted", hue="ablation_name", palette=palette, ax=ax, s=35, alpha=0.7)
            bounds = [float(min(subset["observed"].min(), subset["predicted"].min())), float(max(subset["observed"].max(), subset["predicted"].max()))]
            ax.plot(bounds, bounds, linestyle="--", color="#333333", linewidth=1.2)
            ax.set_title(f"Observed vs predicted: {task_name} / {split_strategy}")
            outputs["observed_vs_predicted"] = export_figure(fig, cfg.output_figures_root / "observed_vs_predicted_top_task", cfg)
            plt.close(fig)
            save_dataframe(subset, cfg.output_figures_root / "source_data" / "observed_vs_predicted_top_task.csv")
    return outputs


def compare_with_previous_reports(cfg: AppConfig) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for step, path in {"step_07": cfg.project_root / "reports/07_classical_baseline_report.json", "step_08": cfg.project_root / "reports/08_deep_baseline_report.json"}.items():
        if path.exists():
            try:
                comparisons[step] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                comparisons[step] = {"warning": f"Failed to parse {path}: {exc}"}
    return comparisons


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config YAML.")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    raw_config = load_yaml_config(config_path)
    cfg = AppConfig.from_dict(raw_config, project_root)
    log_path = setup_logging(cfg.logs_dir)
    set_global_seed(cfg.random_seed)
    config_snapshot_path = save_config_snapshot(cfg, config_path)
    logging.info("Starting %s", SCRIPT_NAME)

    for path, description in [
        (cfg.input_annotated_long_path, "annotated long"),
        (cfg.input_regression_long_path, "multitask regression"),
        (cfg.input_pairwise_selectivity_path, "pairwise selectivity"),
        (cfg.input_target_vs_panel_path, "target-vs-panel regression"),
        (cfg.input_classification_path, "classification task"),
        (cfg.input_split_manifest_path, "split manifest"),
    ]:
        require_file(path, description)

    deps = import_training_dependencies()
    annotated_df = normalize_common_columns(load_required_dataframe(cfg.input_annotated_long_path, "annotated long dataset"))
    split_manifest = load_split_manifest(cfg.input_split_manifest_path)

    task_contexts: list[TaskContext] = []
    for task_name in ["multitask_regression", "pairwise_selectivity", "target_vs_panel", "classification"]:
        if not task_enabled(task_name, cfg):
            continue
        loaded = load_task_table(task_name, cfg)
        if isinstance(loaded, list):
            task_contexts.extend(loaded)
        else:
            task_contexts.append(loaded)

    warnings_list: list[str] = []
    regression_metrics_records: list[dict[str, Any]] = []
    classification_metrics_records: list[dict[str, Any]] = []
    env_group_records: list[pd.DataFrame] = []
    activity_cliff_records: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    best_models: list[dict[str, Any]] = []
    environment_column_usage: dict[str, str] = {}
    graph_metadata_by_task: dict[str, Any] = {}
    ablations_run: list[str] = []

    for task_context in task_contexts:
        task_name = task_context.task_name
        task_df = merge_annotated_columns(task_context.dataframe, annotated_df)
        env_resolution = resolve_environment_columns(task_df, cfg)
        warnings_list.extend(env_resolution.warnings)
        environment_column_usage.update({f"{task_name}:{k}": v for k, v in env_resolution.resolved_columns.items()})
        task_cfg_base = copy.deepcopy(cfg)
        for disabled in env_resolution.disabled_objectives:
            setattr(task_cfg_base.causal_objectives, disabled, False)
        if task_context.task_type == "classification":
            label_name = task_context.active_label_name or task_context.label_column
            logging.info("Processing classification label `%s`", label_name)
        primary_environment_column = next(iter(env_resolution.resolved_columns.values()), None)
        task_df = task_df[task_df["standardized_smiles"].notna()].copy()
        graph_cache, graph_metadata = build_graph_cache(task_df["standardized_smiles"].tolist(), deps)
        graph_metadata_by_task[task_name if task_context.active_label_name is None else f"{task_name}:{task_context.active_label_name}"] = graph_metadata
        if not graph_cache:
            raise ValueError(f"Graph construction failed for every molecule in task {task_name}.")
        task_df = task_df[task_df["standardized_smiles"].isin(graph_cache)].copy()
        task_manifest = split_manifest[split_manifest["task_name"] == task_name].copy()
        if task_manifest.empty:
            warning = f"No split manifest rows found for task `{task_name}`; skipping."
            warnings_list.append(warning)
            logging.warning(warning)
            continue
        ablation_names = ["main"]
        if cfg.run_core_ablations and task_context.task_type == "regression":
            ablation_names.extend(cfg.ablations)
        for ablation_name in ablation_names:
            if ablation_name != "main":
                ablations_run.append(ablation_name)
            task_cfg = task_cfg_base if ablation_name == "main" else apply_ablation(task_cfg_base, ablation_name)
            for manifest_row in task_manifest.itertuples(index=False):
                assignment_df = load_assignment_table(getattr(manifest_row, "output_assignment_path"), project_root)
                ensure_assignment_columns(assignment_df, task_name)
                assignment_df = assignment_df[["row_uid", "split_label", "split_strategy", "split_id", "fold_id"]].drop_duplicates()
                merged = task_df.merge(assignment_df, on="row_uid", how="inner")
                train_df = merged[merged["split_label"] == "train"].copy()
                valid_df = merged[merged["split_label"] == "valid"].copy()
                test_df = merged[merged["split_label"] == "test"].copy()
                if train_df.empty or test_df.empty:
                    warning = f"{task_name}/{manifest_row.split_strategy}/{manifest_row.fold_id}: missing train or test rows; skipping."
                    warnings_list.append(warning)
                    logging.warning(warning)
                    continue
                model_name = f"causal_{task_cfg.molecular_encoder}"
                model_dir = cfg.output_model_root / task_name / model_name / manifest_row.split_strategy / f"{manifest_row.split_id}__{manifest_row.fold_id}__{ablation_name}"
                result = train_one_configuration(
                    task_context=task_context,
                    train_df=train_df,
                    valid_df=valid_df,
                    test_df=test_df,
                    cfg=task_cfg,
                    resolved_environment_column=primary_environment_column,
                    model_name=model_name,
                    ablation_name=ablation_name,
                    metadata={
                        "task_name": task_name,
                        "task_type": task_context.task_type,
                        "label_column": task_context.label_column,
                        "split_strategy": manifest_row.split_strategy,
                        "split_id": manifest_row.split_id,
                        "fold_id": manifest_row.fold_id,
                        "model_name": model_name,
                        "ablation_name": ablation_name,
                        "model_dir": model_dir,
                        "graph_cache": graph_cache,
                        "graph_cache_keys": set(graph_cache.keys()),
                    },
                    deps=deps,
                )
                metric_record = {
                    "task_name": task_name,
                    "task_type": task_context.task_type,
                    "label_column": task_context.label_column,
                    "split_strategy": manifest_row.split_strategy,
                    "split_id": manifest_row.split_id,
                    "fold_id": manifest_row.fold_id,
                    "model_name": model_name,
                    "ablation_name": ablation_name,
                    "best_epoch": result["best_epoch"],
                    "test_loss": result["test_loss"],
                    "environment_column": primary_environment_column,
                    **result["test_metrics"],
                }
                if task_context.task_type == "regression":
                    regression_metrics_records.append(metric_record)
                else:
                    classification_metrics_records.append(metric_record)
                predictions = result["test_predictions"].copy()
                predictions["task_name"] = task_name
                predictions["task_type"] = task_context.task_type
                predictions["model_name"] = model_name
                predictions["ablation_name"] = ablation_name
                predictions["split_strategy"] = manifest_row.split_strategy
                predictions["split_id"] = manifest_row.split_id
                predictions["fold_id"] = manifest_row.fold_id
                if task_context.active_label_name is not None:
                    predictions["label_column"] = task_context.active_label_name
                if cfg.save_test_predictions:
                    save_predictions(predictions, cfg.output_predictions_root, metric_record)
                if cfg.save_error_tables:
                    save_error_table(predictions, task_context.task_type, cfg.output_metrics_root / "error_tables", metric_record)
                if cfg.save_latent_embeddings:
                    save_latent_embeddings(result["test_latent"], predictions, cfg.output_predictions_root, metric_record)
                if cfg.save_environment_predictions:
                    save_environment_outputs(result["environment_outputs"], predictions, cfg.output_predictions_root, metric_record)
                if task_cfg.save_trained_models:
                    model_dir.mkdir(parents=True, exist_ok=True)
                    deps["torch"].save(result["model"].state_dict(), model_dir / "model_state_dict.pt")
                    vocab_path = model_dir / "vocabularies.json"
                    vocab_path.write_text(json.dumps(result["vocabularies"], indent=2), encoding="utf-8")
                env_group_records.append(summarize_by_environment(predictions, task_context.task_type, primary_environment_column, deps, metric_record))
                activity_cliff_records.append(summarize_activity_cliff(predictions, task_context.task_type, deps, metric_record))
                all_predictions.append(predictions)
                best_models.append(metric_record)

    regression_metrics_df = pd.DataFrame(regression_metrics_records)
    classification_metrics_df = pd.DataFrame(classification_metrics_records)
    all_predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    environment_group_metrics_df = pd.concat([frame for frame in env_group_records if not frame.empty], ignore_index=True) if env_group_records else pd.DataFrame()
    activity_cliff_metrics_df = pd.concat([frame for frame in activity_cliff_records if not frame.empty], ignore_index=True) if activity_cliff_records else pd.DataFrame()

    regression_metric_columns = ["RMSE", "MAE", "R2", "Spearman", "Pearson"]
    classification_metric_columns = ["ROC_AUC", "PR_AUC", "accuracy", "balanced_accuracy", "F1", "MCC"]
    regression_summary_df = aggregate_metrics(regression_metrics_df, regression_metric_columns, ["task_name", "model_name", "ablation_name", "split_strategy"])
    classification_summary_df = aggregate_metrics(classification_metrics_df, classification_metric_columns, ["task_name", "model_name", "ablation_name", "split_strategy"])
    ablation_summary_df = aggregate_metrics(regression_metrics_df, regression_metric_columns, ["task_name", "ablation_name"])

    save_dataframe(regression_metrics_df, cfg.output_metrics_root / "regression_metrics_per_fold.csv")
    save_dataframe(regression_summary_df, cfg.output_metrics_root / "regression_metrics_summary.csv")
    save_dataframe(classification_metrics_df, cfg.output_metrics_root / "classification_metrics_per_fold.csv")
    save_dataframe(classification_summary_df, cfg.output_metrics_root / "classification_metrics_summary.csv")
    save_dataframe(ablation_summary_df, cfg.output_metrics_root / "ablation_metrics_summary.csv")
    save_dataframe(environment_group_metrics_df, cfg.output_metrics_root / "environment_group_metrics.csv")
    save_dataframe(activity_cliff_metrics_df, cfg.output_metrics_root / "activity_cliff_metrics.csv")
    if not all_predictions_df.empty:
        save_dataframe(all_predictions_df, cfg.output_predictions_root / "all_test_predictions.csv")

    figure_outputs = generate_figures(regression_summary_df, ablation_summary_df, all_predictions_df, cfg)

    previous_report_comparison = compare_with_previous_reports(cfg)
    report = {
        "script_name": SCRIPT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "annotated_long": str(cfg.input_annotated_long_path),
            "multitask_regression": str(cfg.input_regression_long_path),
            "pairwise_selectivity": str(cfg.input_pairwise_selectivity_path),
            "target_vs_panel": str(cfg.input_target_vs_panel_path),
            "classification": str(cfg.input_classification_path),
            "split_manifest": str(cfg.input_split_manifest_path),
        },
        "output_roots": {
            "model_root": str(cfg.output_model_root),
            "metrics_root": str(cfg.output_metrics_root),
            "predictions_root": str(cfg.output_predictions_root),
            "figures_root": str(cfg.output_figures_root),
            "report_path": str(cfg.output_report_path),
        },
        "tasks_processed": sorted(set(regression_metrics_df.get("task_name", pd.Series(dtype=str)).tolist() + classification_metrics_df.get("task_name", pd.Series(dtype=str)).tolist())),
        "split_strategies_processed": sorted(set(regression_metrics_df.get("split_strategy", pd.Series(dtype=str)).tolist() + classification_metrics_df.get("split_strategy", pd.Series(dtype=str)).tolist())),
        "model_configuration": {
            "molecular_encoder": cfg.molecular_encoder,
            "hidden_dim": cfg.hidden_dim,
            "num_message_passing_layers": cfg.num_message_passing_layers,
            "dropout": cfg.dropout,
            "multitask_strategy": cfg.multitask_strategy,
            "pairwise_strategy": cfg.pairwise_strategy,
            "target_vs_panel_strategy": cfg.target_vs_panel_strategy,
        },
        "active_causal_objectives": asdict(cfg.causal_objectives),
        "environment_columns_used": environment_column_usage,
        "graph_feature_settings_used": {"atom_feature_dim": 8, "bond_feature_dim": 6, "encoder": cfg.molecular_encoder},
        "number_of_unique_compounds_processed": int(all_predictions_df["compound_id"].nunique()) if not all_predictions_df.empty else 0,
        "graph_construction_summary": graph_metadata_by_task,
        "regression_summary_metrics": regression_summary_df.to_dict(orient="records"),
        "classification_summary_metrics": classification_summary_df.to_dict(orient="records"),
        "ablation_summary": ablation_summary_df.to_dict(orient="records"),
        "best_model_per_task": pd.DataFrame(best_models).sort_values("test_loss", ascending=True).groupby("task_name", dropna=False).head(1).to_dict(orient="records") if best_models else [],
        "comparison_summary_previous_steps": previous_report_comparison,
        "warnings": sorted(set(warnings_list)),
        "config_snapshot_reference": str(config_snapshot_path) if config_snapshot_path is not None else None,
        "log_path": str(log_path),
        "figure_outputs": figure_outputs,
        "ablations_run": sorted(set(ablations_run)),
    }
    cfg.output_report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Wrote report to %s", cfg.output_report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
