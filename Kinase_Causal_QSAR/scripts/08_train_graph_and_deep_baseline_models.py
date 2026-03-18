#!/usr/bin/env python3
"""Train and evaluate graph/deep-learning baseline models for Step-05 tasks.

This script is a strict continuation of Script-06 and Script-05. It reads the
benchmark task tables from Script-05, consumes split manifests and assignment
files produced by Script-06, constructs molecular graphs from standardized
SMILES, and trains reproducible graph neural-network baselines for regression
and optional classification tasks.
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
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

SCRIPT_NAME = "08_train_graph_and_deep_baseline_models"
RANDOM_SEED = 2025
CLASSIFICATION_TASK_NAME = "classification"
LABEL_PRIORITY = ["active_inactive_label", "strong_weak_label", "selective_label", "highly_selective_label"]

REQUIRED_SCRIPT_08_KEYS = {
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
    "run_multitask_regression",
    "run_pairwise_selectivity_regression",
    "run_target_vs_panel_regression",
    "run_classification_tasks",
    "molecular_input_representation",
    "graph_models",
    "sequence_or_target_encoding",
    "multitask_strategy",
    "pairwise_strategy",
    "target_vs_panel_strategy",
    "node_features",
    "edge_features",
    "training",
    "save_trained_models",
    "save_model_checkpoints",
    "save_fold_predictions",
    "save_test_predictions",
    "save_latent_embeddings",
    "save_error_tables",
    "save_config_snapshot",
    "make_figures",
    "export_svg",
    "export_png",
    "export_pdf",
    "figure_style",
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "compound_id": ("compound_id", "standardized_smiles"),
    "standardized_smiles": ("standardized_smiles",),
    "target_chembl_id": ("target_chembl_id",),
    "target_name": ("target_name", "pref_name"),
    "pKi": ("pKi", "median_pKi", "aggregated_pKi"),
    "delta_pKi": ("delta_pKi", "delta_pki"),
    "target_vs_panel_delta_pKi": ("target_vs_panel_delta_pKi", "target_vs_panel_delta_pki"),
    "kinase_family": ("kinase_family", "kinase_family_label", "target_family", "broad_kinase_family"),
}

REGRESSION_TASKS: dict[str, dict[str, Any]] = {
    "multitask_regression": {
        "config_key": "input_regression_long_path",
        "target_column": "pKi",
        "required_columns": ["target_chembl_id", "pKi", "standardized_smiles"],
    },
    "pairwise_selectivity": {
        "config_key": "input_pairwise_selectivity_path",
        "target_column": "delta_pKi",
        "required_columns": ["kinase_a_chembl_id", "kinase_b_chembl_id", "delta_pKi", "standardized_smiles"],
    },
    "target_vs_panel": {
        "config_key": "input_target_vs_panel_path",
        "target_column": "target_vs_panel_delta_pKi",
        "required_columns": ["target_chembl_id", "target_vs_panel_delta_pKi", "standardized_smiles"],
    },
}

NATURE_PALETTE = ["#386CB0", "#F39C12", "#2CA25F", "#E74C3C", "#756BB1", "#7F8C8D"]


@dataclass
class FigureStyle:
    font_family: str
    bold_text: bool
    output_format_primary: str
    palette_name: str
    dpi_png: int


@dataclass
class TargetEncodingConfig:
    use_target_identity_embedding: bool
    use_kinase_family_embedding: bool
    use_pair_target_encoding: bool


@dataclass
class TrainingConfig:
    gradient_clip_norm: float
    scheduler: str
    mixed_precision: bool
    save_best_only: bool
    regression_loss: str
    classification_loss: str
    class_weighting: bool


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
    device: str
    num_workers: int
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    run_multitask_regression: bool
    run_pairwise_selectivity_regression: bool
    run_target_vs_panel_regression: bool
    run_classification_tasks: bool
    molecular_input_representation: str
    graph_models: list[str]
    sequence_or_target_encoding: TargetEncodingConfig
    multitask_strategy: str
    pairwise_strategy: str
    target_vs_panel_strategy: str
    node_features: dict[str, bool]
    edge_features: dict[str, bool]
    training: TrainingConfig
    save_trained_models: bool
    save_model_checkpoints: bool
    save_fold_predictions: bool
    save_test_predictions: bool
    save_latent_embeddings: bool
    save_error_tables: bool
    save_config_snapshot: bool
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
            raise ValueError("Config YAML must contain a top-level mapping/object.")
        script_cfg = raw.get("script_08")
        if not isinstance(script_cfg, dict):
            raise ValueError("Missing required `script_08` section in config.yaml.")
        missing = sorted(REQUIRED_SCRIPT_08_KEYS.difference(script_cfg))
        if missing:
            raise ValueError("Missing required script_08 config values: " + ", ".join(missing))

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
            raise ValueError(f"script_08.{key} must be a boolean; got {value!r}.")

        def parse_int(key: str, minimum: int | None = None) -> int:
            value = script_cfg.get(key)
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"script_08.{key} must be an integer; got {value!r}.") from exc
            if minimum is not None and parsed < minimum:
                raise ValueError(f"script_08.{key} must be >= {minimum}; got {parsed}.")
            return parsed

        def parse_float(key: str, minimum: float | None = None) -> float:
            value = script_cfg.get(key)
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"script_08.{key} must be numeric; got {value!r}.") from exc
            if minimum is not None and parsed < minimum:
                raise ValueError(f"script_08.{key} must be >= {minimum}; got {parsed}.")
            return parsed

        def parse_list(key: str) -> list[str]:
            value = script_cfg.get(key)
            if not isinstance(value, list) or not value:
                raise ValueError(f"script_08.{key} must be a non-empty list.")
            parsed = [str(item).strip() for item in value if str(item).strip()]
            if not parsed:
                raise ValueError(f"script_08.{key} must contain non-empty values.")
            return parsed

        figure_style_raw = script_cfg.get("figure_style")
        if not isinstance(figure_style_raw, dict):
            raise ValueError("script_08.figure_style must be a mapping.")
        sequence_raw = script_cfg.get("sequence_or_target_encoding")
        if not isinstance(sequence_raw, dict):
            raise ValueError("script_08.sequence_or_target_encoding must be a mapping.")
        training_raw = script_cfg.get("training")
        if not isinstance(training_raw, dict):
            raise ValueError("script_08.training must be a mapping.")
        node_features = script_cfg.get("node_features")
        if not isinstance(node_features, dict) or not node_features:
            raise ValueError("script_08.node_features must be a non-empty mapping.")
        edge_features = script_cfg.get("edge_features")
        if not isinstance(edge_features, dict) or not edge_features:
            raise ValueError("script_08.edge_features must be a non-empty mapping.")

        molecular_input_representation = str(script_cfg["molecular_input_representation"]).strip().lower()
        if molecular_input_representation != "graph":
            raise ValueError("script_08.molecular_input_representation currently supports only `graph`.")

        graph_models = [item.lower() for item in parse_list("graph_models")]
        allowed_models = {"gcn", "gin", "mpnn", "gat"}
        unknown = sorted(set(graph_models).difference(allowed_models))
        if unknown:
            raise ValueError("Unsupported script_08.graph_models values: " + ", ".join(unknown))

        device = str(script_cfg["device"]).strip().lower()
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("script_08.device must be one of: auto, cpu, cuda, mps.")

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
            device=device,
            num_workers=parse_int("num_workers", minimum=0),
            batch_size=parse_int("batch_size", minimum=1),
            max_epochs=parse_int("max_epochs", minimum=1),
            early_stopping_patience=parse_int("early_stopping_patience", minimum=1),
            learning_rate=parse_float("learning_rate", minimum=0.0),
            weight_decay=parse_float("weight_decay", minimum=0.0),
            run_multitask_regression=parse_bool(script_cfg["run_multitask_regression"], "run_multitask_regression"),
            run_pairwise_selectivity_regression=parse_bool(script_cfg["run_pairwise_selectivity_regression"], "run_pairwise_selectivity_regression"),
            run_target_vs_panel_regression=parse_bool(script_cfg["run_target_vs_panel_regression"], "run_target_vs_panel_regression"),
            run_classification_tasks=parse_bool(script_cfg["run_classification_tasks"], "run_classification_tasks"),
            molecular_input_representation=molecular_input_representation,
            graph_models=graph_models,
            sequence_or_target_encoding=TargetEncodingConfig(
                use_target_identity_embedding=parse_bool(sequence_raw.get("use_target_identity_embedding", True), "sequence_or_target_encoding.use_target_identity_embedding"),
                use_kinase_family_embedding=parse_bool(sequence_raw.get("use_kinase_family_embedding", True), "sequence_or_target_encoding.use_kinase_family_embedding"),
                use_pair_target_encoding=parse_bool(sequence_raw.get("use_pair_target_encoding", True), "sequence_or_target_encoding.use_pair_target_encoding"),
            ),
            multitask_strategy=str(script_cfg["multitask_strategy"]).strip(),
            pairwise_strategy=str(script_cfg["pairwise_strategy"]).strip(),
            target_vs_panel_strategy=str(script_cfg["target_vs_panel_strategy"]).strip(),
            node_features={str(k): parse_bool(v, f"node_features.{k}") for k, v in node_features.items()},
            edge_features={str(k): parse_bool(v, f"edge_features.{k}") for k, v in edge_features.items()},
            training=TrainingConfig(
                gradient_clip_norm=float(training_raw.get("gradient_clip_norm", 5.0)),
                scheduler=str(training_raw.get("scheduler", "reduce_on_plateau")).strip().lower(),
                mixed_precision=parse_bool(training_raw.get("mixed_precision", False), "training.mixed_precision"),
                save_best_only=parse_bool(training_raw.get("save_best_only", True), "training.save_best_only"),
                regression_loss=str(training_raw.get("regression_loss", "mse")).strip().lower(),
                classification_loss=str(training_raw.get("classification_loss", "bce")).strip().lower(),
                class_weighting=parse_bool(training_raw.get("class_weighting", True), "training.class_weighting"),
            ),
            save_trained_models=parse_bool(script_cfg["save_trained_models"], "save_trained_models"),
            save_model_checkpoints=parse_bool(script_cfg["save_model_checkpoints"], "save_model_checkpoints"),
            save_fold_predictions=parse_bool(script_cfg["save_fold_predictions"], "save_fold_predictions"),
            save_test_predictions=parse_bool(script_cfg["save_test_predictions"], "save_test_predictions"),
            save_latent_embeddings=parse_bool(script_cfg["save_latent_embeddings"], "save_latent_embeddings"),
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
            logs_dir=resolve(raw.get("script_02", {}).get("logs_dir", "logs")),
            configs_used_dir=resolve(raw.get("script_02", {}).get("configs_used_dir", "configs_used")),
            project_root=project_root,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train graph/deep baseline models for Step-05 tasks using Step-06 splits.")
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
    for path in [
        cfg.logs_dir,
        cfg.configs_used_dir,
        cfg.output_model_root,
        cfg.output_metrics_root,
        cfg.output_predictions_root,
        cfg.output_figures_root,
        cfg.output_report_path.parent,
    ]:
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


def import_runtime_dependencies() -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.optim.lr_scheduler import ReduceLROnPlateau
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for Script-08. Install torch before running this script."
        ) from exc
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GATConv, GCNConv, GINConv, NNConv, global_add_pool, global_mean_pool
    except Exception as exc:
        raise RuntimeError(
            "PyTorch Geometric is required for Script-08. Install torch-geometric and its dependencies before running this script."
        ) from exc
    try:
        from rdkit import Chem
        from rdkit.Chem import rdchem
    except Exception as exc:
        raise RuntimeError("RDKit is required for graph construction in Script-08.") from exc

    scipy_stats = None
    try:
        from scipy import stats as scipy_stats  # type: ignore[assignment]
    except Exception:
        scipy_stats = None

    plt = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[assignment]
    except Exception:
        plt = None

    return {
        "torch": torch,
        "nn": nn,
        "F": F,
        "ReduceLROnPlateau": ReduceLROnPlateau,
        "Data": Data,
        "DataLoader": DataLoader,
        "GATConv": GATConv,
        "GCNConv": GCNConv,
        "GINConv": GINConv,
        "NNConv": NNConv,
        "global_add_pool": global_add_pool,
        "global_mean_pool": global_mean_pool,
        "Chem": Chem,
        "rdchem": rdchem,
        "scipy_stats": scipy_stats,
        "plt": plt,
    }


def set_global_determinism(seed: int, torch: Any) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        logging.warning("torch.use_deterministic_algorithms(True) could not be fully enabled in this environment.")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(cfg: AppConfig, torch: Any) -> Any:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if cfg.device == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("Config requested MPS but it is unavailable in this environment.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    return "|".join([task_name, str(row["compound_id"]), str(row.get("target_chembl_id", "NA")), str(row_index)])


def standardize_task_dataframe(task_name: str, df: pd.DataFrame, required_columns: list[str]) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    normalized, mapping, warnings = canonicalize_columns(df)
    missing = [column for column in required_columns if column not in normalized.columns]
    if missing:
        raise ValueError(f"{task_name} dataset is missing required columns: {', '.join(missing)}")
    normalized = normalized.reset_index(drop=True).copy()
    normalized["row_index"] = normalized.index.astype(int)
    normalized["row_uid"] = [build_row_uid(task_name, row, idx) for idx, row in normalized.iterrows()]
    return normalized, mapping, warnings


def detect_classification_labels(df: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for col in df.columns:
        if col in LABEL_PRIORITY or col.endswith("_label"):
            labels.append(col)
    stable: list[str] = []
    for col in LABEL_PRIORITY + sorted(labels):
        if col in df.columns and col not in stable:
            stable.append(col)
    return stable


def load_required_dataframe(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required {description} not found: {path}")
    logging.info("Loading %s from %s", description, path)
    return pd.read_csv(path)


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
    if not resolved.exists():
        raise FileNotFoundError(f"Split assignment file not found: {resolved}")
    logging.info("Loading split assignments from %s", resolved)
    return pd.read_csv(resolved)


def ensure_assignment_columns(df: pd.DataFrame, task_name: str) -> None:
    required = {"row_uid", "split_label", "split_strategy", "split_id", "fold_id"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Assignment file for {task_name} missing columns: {', '.join(missing)}")


def configure_matplotlib(style: FigureStyle, plt: Any) -> None:
    if plt is None:
        return
    plt.rcParams["font.family"] = style.font_family
    plt.rcParams["font.weight"] = "bold" if style.bold_text else "normal"
    plt.rcParams["axes.labelweight"] = "bold" if style.bold_text else "normal"
    plt.rcParams["axes.titleweight"] = "bold" if style.bold_text else "normal"
    plt.rcParams["svg.fonttype"] = "none"


class GraphCache:
    """Reusable molecular graph cache keyed by compound_id."""

    def __init__(self, cfg: AppConfig, runtime: dict[str, Any]) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self.cache: dict[str, Any] = {}
        self.metadata_records: list[dict[str, Any]] = []
        self.invalid_compounds: set[str] = set()
        self.node_feature_names: list[str] = []
        self.edge_feature_names: list[str] = []

    def build_for_compounds(self, compound_frame: pd.DataFrame) -> None:
        unique_compounds = compound_frame[["compound_id", "standardized_smiles"]].drop_duplicates().sort_values("compound_id", kind="mergesort")
        logging.info("Constructing graph cache for %s unique compounds.", len(unique_compounds))
        for row in unique_compounds.itertuples(index=False):
            graph = self._smiles_to_graph(str(row.compound_id), str(row.standardized_smiles))
            if graph is not None:
                self.cache[str(row.compound_id)] = graph
        logging.info(
            "Graph construction complete: %s success, %s failure.",
            len(self.cache),
            len(self.invalid_compounds),
        )

    def _smiles_to_graph(self, compound_id: str, smiles: str) -> Any | None:
        Data = self.runtime["Data"]
        Chem = self.runtime["Chem"]
        torch = self.runtime["torch"]
        mol = Chem.MolFromSmiles(smiles)
        parse_success = mol is not None
        if not parse_success or mol is None or mol.GetNumAtoms() == 0:
            self.invalid_compounds.add(compound_id)
            self.metadata_records.append({
                "compound_id": compound_id,
                "standardized_smiles": smiles,
                "rdkit_parse_success": 0,
                "n_atoms": 0,
                "n_bonds": 0,
            })
            return None
        atom_features = [self._atom_features(atom) for atom in mol.GetAtoms()]
        if not self.node_feature_names:
            self.node_feature_names = self._atom_feature_names()
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_pairs: list[list[int]] = []
        edge_features: list[list[float]] = []
        for bond in mol.GetBonds():
            begin_idx = int(bond.GetBeginAtomIdx())
            end_idx = int(bond.GetEndAtomIdx())
            bond_features = self._bond_features(bond)
            edge_pairs.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
            edge_features.extend([bond_features, bond_features])
        if not self.edge_feature_names:
            self.edge_feature_names = self._bond_feature_names()
        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(self.edge_feature_names)), dtype=torch.float)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph.compound_id = compound_id
        graph.standardized_smiles = smiles
        graph.n_atoms = int(mol.GetNumAtoms())
        graph.n_bonds = int(mol.GetNumBonds())
        self.metadata_records.append({
            "compound_id": compound_id,
            "standardized_smiles": smiles,
            "rdkit_parse_success": 1,
            "n_atoms": graph.n_atoms,
            "n_bonds": graph.n_bonds,
        })
        return graph

    def _atom_feature_names(self) -> list[str]:
        names: list[str] = []
        if self.cfg.node_features.get("use_atom_type", True):
            names.extend([f"atom_type_{symbol}" for symbol in ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "other"]])
        if self.cfg.node_features.get("use_degree", True):
            names.append("atom_degree")
        if self.cfg.node_features.get("use_formal_charge", True):
            names.append("formal_charge")
        if self.cfg.node_features.get("use_hybridization", True):
            names.extend([f"hybridization_{name}" for name in ["SP", "SP2", "SP3", "other"]])
        if self.cfg.node_features.get("use_aromaticity", True):
            names.append("is_aromatic")
        if self.cfg.node_features.get("use_num_hs", True):
            names.append("total_num_hs")
        if self.cfg.node_features.get("use_chirality", True):
            names.extend(["chirality_unspecified", "chirality_cw", "chirality_ccw"])
        return names

    def _bond_feature_names(self) -> list[str]:
        names: list[str] = []
        if self.cfg.edge_features.get("use_bond_type", True):
            names.extend([f"bond_type_{name}" for name in ["single", "double", "triple", "aromatic", "other"]])
        if self.cfg.edge_features.get("use_conjugation", True):
            names.append("is_conjugated")
        if self.cfg.edge_features.get("use_ring_status", True):
            names.append("is_in_ring")
        if self.cfg.edge_features.get("use_stereo", True):
            names.extend(["stereo_none", "stereo_any", "stereo_z", "stereo_e", "stereo_other"])
        return names

    def _atom_features(self, atom: Any) -> list[float]:
        rdchem = self.runtime["rdchem"]
        features: list[float] = []
        if self.cfg.node_features.get("use_atom_type", True):
            symbol = atom.GetSymbol()
            symbols = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
            features.extend([1.0 if symbol == item else 0.0 for item in symbols])
            features.append(0.0 if symbol in symbols else 1.0)
        if self.cfg.node_features.get("use_degree", True):
            features.append(float(atom.GetDegree()))
        if self.cfg.node_features.get("use_formal_charge", True):
            features.append(float(atom.GetFormalCharge()))
        if self.cfg.node_features.get("use_hybridization", True):
            hybrid = atom.GetHybridization()
            hybrids = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3]
            features.extend([1.0 if hybrid == item else 0.0 for item in hybrids])
            features.append(0.0 if hybrid in hybrids else 1.0)
        if self.cfg.node_features.get("use_aromaticity", True):
            features.append(float(atom.GetIsAromatic()))
        if self.cfg.node_features.get("use_num_hs", True):
            features.append(float(atom.GetTotalNumHs()))
        if self.cfg.node_features.get("use_chirality", True):
            tag = atom.GetChiralTag()
            features.extend([
                1.0 if tag == rdchem.ChiralType.CHI_UNSPECIFIED else 0.0,
                1.0 if tag == rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0.0,
                1.0 if tag == rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0.0,
            ])
        return features

    def _bond_features(self, bond: Any) -> list[float]:
        rdchem = self.runtime["rdchem"]
        features: list[float] = []
        if self.cfg.edge_features.get("use_bond_type", True):
            bond_type = bond.GetBondType()
            supported = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]
            features.extend([1.0 if bond_type == item else 0.0 for item in supported])
            features.append(0.0 if bond_type in supported else 1.0)
        if self.cfg.edge_features.get("use_conjugation", True):
            features.append(float(bond.GetIsConjugated()))
        if self.cfg.edge_features.get("use_ring_status", True):
            features.append(float(bond.IsInRing()))
        if self.cfg.edge_features.get("use_stereo", True):
            stereo = bond.GetStereo()
            values = [rdchem.BondStereo.STEREONONE, rdchem.BondStereo.STEREOANY, rdchem.BondStereo.STEREOZ, rdchem.BondStereo.STEREOE]
            features.extend([1.0 if stereo == item else 0.0 for item in values])
            features.append(0.0 if stereo in values else 1.0)
        return features


def create_embedding_map(values: Iterable[str]) -> dict[str, int]:
    ordered = sorted({str(value) for value in values if str(value) not in {"", "nan", "None"}})
    return {value: idx + 1 for idx, value in enumerate(ordered)}


def prepare_classification_target(series: pd.Series) -> pd.Series:
    mapped = pd.to_numeric(series, errors="coerce")
    if mapped.notna().sum() == 0:
        text = series.astype(str).str.strip().str.lower()
        values = [item for item in sorted(text.dropna().unique().tolist()) if item not in {"", "nan", "none"}]
        if len(values) != 2:
            raise ValueError(f"Classification label column {series.name} must be binary; found values {values!r}.")
        mapped = text.map({values[0]: 0, values[1]: 1})
    mapped = mapped.dropna().astype(int)
    unique = sorted(mapped.unique().tolist())
    if unique != [0, 1]:
        raise ValueError(f"Classification label column {series.name} must normalize to binary 0/1, found {unique}.")
    return mapped


def apply_embedding_mappings(df: pd.DataFrame, task_name: str, embedding_maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    enriched = df.copy()
    enriched["target_id_idx"] = 0
    enriched["target_family_idx"] = 0
    enriched["kinase_a_idx"] = 0
    enriched["kinase_b_idx"] = 0
    if task_name in {"multitask_regression", "target_vs_panel", CLASSIFICATION_TASK_NAME} and "target_chembl_id" in enriched.columns:
        enriched["target_id_idx"] = enriched["target_chembl_id"].astype(str).map(embedding_maps.get("target_ids", {})).fillna(0).astype(int)
    if "kinase_family" in enriched.columns:
        enriched["target_family_idx"] = enriched["kinase_family"].astype(str).map(embedding_maps.get("kinase_families", {})).fillna(0).astype(int)
    if task_name == "pairwise_selectivity":
        enriched["kinase_a_idx"] = enriched["kinase_a_chembl_id"].astype(str).map(embedding_maps.get("pair_target_ids", {})).fillna(0).astype(int)
        enriched["kinase_b_idx"] = enriched["kinase_b_chembl_id"].astype(str).map(embedding_maps.get("pair_target_ids", {})).fillna(0).astype(int)
    return enriched


def clone_graph(graph: Any) -> Any:
    return copy.deepcopy(graph)


def build_task_data_objects(
    df: pd.DataFrame,
    task_name: str,
    target_column: str,
    graph_cache: GraphCache,
    runtime: dict[str, Any],
    label_name: str | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    torch = runtime["torch"]
    data_objects: list[Any] = []
    skipped_invalid = 0
    skipped_missing_target = 0
    for row in df.itertuples(index=False):
        graph = graph_cache.cache.get(str(row.compound_id))
        if graph is None:
            skipped_invalid += 1
            continue
        target_value = getattr(row, target_column)
        if pd.isna(target_value):
            skipped_missing_target += 1
            continue
        item = clone_graph(graph)
        item.row_uid = str(row.row_uid)
        item.compound_id = str(row.compound_id)
        item.task_name = task_name
        item.label_name = label_name or ""
        item.y = torch.tensor([float(target_value)], dtype=torch.float)
        item.target_id = torch.tensor([int(getattr(row, "target_id_idx", 0))], dtype=torch.long)
        item.target_family_id = torch.tensor([int(getattr(row, "target_family_idx", 0))], dtype=torch.long)
        item.kinase_a_id = torch.tensor([int(getattr(row, "kinase_a_idx", 0))], dtype=torch.long)
        item.kinase_b_id = torch.tensor([int(getattr(row, "kinase_b_idx", 0))], dtype=torch.long)
        if task_name == "pairwise_selectivity":
            item.primary_target_identifier = str(getattr(row, "kinase_a_chembl_id", ""))
            item.secondary_target_identifier = str(getattr(row, "kinase_b_chembl_id", ""))
        else:
            item.primary_target_identifier = str(getattr(row, "target_chembl_id", ""))
            item.secondary_target_identifier = ""
        data_objects.append(item)
    metadata = {
        "n_total_rows": int(len(df)),
        "n_data_objects": int(len(data_objects)),
        "skipped_invalid_graph_rows": int(skipped_invalid),
        "skipped_missing_target_rows": int(skipped_missing_target),
    }
    return data_objects, metadata


def split_data_objects(data_objects: list[Any], split_assignments: pd.DataFrame) -> dict[str, list[Any]]:
    assignment_map = split_assignments.set_index("row_uid")["split_label"].to_dict()
    partitioned = {"train": [], "valid": [], "test": []}
    for item in data_objects:
        label = assignment_map.get(item.row_uid)
        if label in partitioned:
            partitioned[label].append(item)
    return partitioned


def create_data_loader(data: list[Any], batch_size: int, shuffle: bool, runtime: dict[str, Any], cfg: AppConfig) -> Any:
    return runtime["DataLoader"](data, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.num_workers)


def infer_feature_dimensions(graph_cache: GraphCache) -> tuple[int, int]:
    sample = next(iter(graph_cache.cache.values()), None)
    if sample is None:
        raise ValueError("Graph cache is empty; cannot infer feature dimensions.")
    node_dim = int(sample.x.shape[1])
    edge_dim = int(sample.edge_attr.shape[1]) if getattr(sample, "edge_attr", None) is not None and sample.edge_attr.ndim == 2 else 0
    return node_dim, edge_dim


def build_mlp(nn: Any, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 2, dropout: float = 0.1) -> Any:
    layers: list[Any] = []
    current = input_dim
    for _ in range(max(n_layers - 1, 1)):
        layers.append(nn.Linear(current, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def make_model_factory(runtime: dict[str, Any], cfg: AppConfig, metadata: dict[str, Any]) -> dict[str, Any]:
    torch = runtime["torch"]
    nn = runtime["nn"]
    GCNConv = runtime["GCNConv"]
    GINConv = runtime["GINConv"]
    GATConv = runtime["GATConv"]
    NNConv = runtime["NNConv"]
    global_mean_pool = runtime["global_mean_pool"]

    class GraphBaselineModel(nn.Module):
        def __init__(self, model_name: str, task_name: str, output_mode: str) -> None:
            super().__init__()
            self.model_name = model_name
            self.task_name = task_name
            self.output_mode = output_mode
            self.hidden_dim = 128
            self.dropout = 0.2
            self.node_dim = int(metadata["node_feature_dim"])
            self.edge_dim = int(metadata["edge_feature_dim"])
            self.target_vocab = int(metadata.get("target_vocab_size", 0)) + 1
            self.family_vocab = int(metadata.get("family_vocab_size", 0)) + 1
            self.pair_vocab = int(metadata.get("pair_vocab_size", 0)) + 1
            self.embed_dim = 32
            self.use_target_embedding = task_name in {"multitask_regression", "target_vs_panel", CLASSIFICATION_TASK_NAME} and cfg.sequence_or_target_encoding.use_target_identity_embedding
            self.use_family_embedding = task_name in {"multitask_regression", "target_vs_panel", CLASSIFICATION_TASK_NAME} and cfg.sequence_or_target_encoding.use_kinase_family_embedding
            self.use_pair_embedding = task_name == "pairwise_selectivity" and cfg.sequence_or_target_encoding.use_pair_target_encoding

            if model_name == "gcn":
                self.conv1 = GCNConv(self.node_dim, self.hidden_dim)
                self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
                self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim)
            elif model_name == "gin":
                self.conv1 = GINConv(build_mlp(nn, self.node_dim, self.hidden_dim, self.hidden_dim))
                self.conv2 = GINConv(build_mlp(nn, self.hidden_dim, self.hidden_dim, self.hidden_dim))
                self.conv3 = GINConv(build_mlp(nn, self.hidden_dim, self.hidden_dim, self.hidden_dim))
            elif model_name == "gat":
                self.conv1 = GATConv(self.node_dim, self.hidden_dim // 4, heads=4, dropout=self.dropout)
                self.conv2 = GATConv(self.hidden_dim, self.hidden_dim // 4, heads=4, dropout=self.dropout)
                self.conv3 = GATConv(self.hidden_dim, self.hidden_dim // 4, heads=4, dropout=self.dropout)
            elif model_name == "mpnn":
                edge_network = build_mlp(nn, max(self.edge_dim, 1), self.hidden_dim, self.node_dim * self.hidden_dim)
                edge_network_2 = build_mlp(nn, max(self.edge_dim, 1), self.hidden_dim, self.hidden_dim * self.hidden_dim)
                edge_network_3 = build_mlp(nn, max(self.edge_dim, 1), self.hidden_dim, self.hidden_dim * self.hidden_dim)
                self.conv1 = NNConv(self.node_dim, self.hidden_dim, edge_network, aggr="mean")
                self.conv2 = NNConv(self.hidden_dim, self.hidden_dim, edge_network_2, aggr="mean")
                self.conv3 = NNConv(self.hidden_dim, self.hidden_dim, edge_network_3, aggr="mean")
            else:
                raise ValueError(f"Unsupported graph model: {model_name}")

            self.norm1 = nn.BatchNorm1d(self.hidden_dim)
            self.norm2 = nn.BatchNorm1d(self.hidden_dim)
            self.norm3 = nn.BatchNorm1d(self.hidden_dim)
            self.pool = global_mean_pool

            fusion_dim = self.hidden_dim
            if self.use_target_embedding:
                self.target_embedding = nn.Embedding(self.target_vocab, self.embed_dim, padding_idx=0)
                fusion_dim += self.embed_dim
            else:
                self.target_embedding = None
            if self.use_family_embedding:
                self.family_embedding = nn.Embedding(self.family_vocab, self.embed_dim // 2, padding_idx=0)
                fusion_dim += self.embed_dim // 2
            else:
                self.family_embedding = None
            if self.use_pair_embedding:
                self.kinase_embedding = nn.Embedding(self.pair_vocab, self.embed_dim, padding_idx=0)
                fusion_dim += self.embed_dim * 2
            else:
                self.kinase_embedding = None

            self.head = nn.Sequential(
                nn.Linear(fusion_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 1),
            )

        def encode_graph(self, batch: Any) -> Any:
            F = runtime["F"]
            x = self.conv1(batch.x, batch.edge_index, getattr(batch, "edge_attr", None)) if self.model_name == "mpnn" else self.conv1(batch.x, batch.edge_index)
            x = self.norm1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, batch.edge_index, getattr(batch, "edge_attr", None)) if self.model_name == "mpnn" else self.conv2(x, batch.edge_index)
            x = self.norm2(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv3(x, batch.edge_index, getattr(batch, "edge_attr", None)) if self.model_name == "mpnn" else self.conv3(x, batch.edge_index)
            x = self.norm3(x)
            x = F.relu(x)
            pooled = self.pool(x, batch.batch)
            return pooled

        def forward(self, batch: Any) -> tuple[Any, Any]:
            representation = self.encode_graph(batch)
            features = [representation]
            if self.target_embedding is not None:
                features.append(self.target_embedding(batch.target_id.view(-1)))
            if self.family_embedding is not None:
                features.append(self.family_embedding(batch.target_family_id.view(-1)))
            if self.kinase_embedding is not None:
                features.append(self.kinase_embedding(batch.kinase_a_id.view(-1)))
                features.append(self.kinase_embedding(batch.kinase_b_id.view(-1)))
            fused = torch.cat(features, dim=1)
            output = self.head(fused).view(-1)
            return output, representation

    return {name: (lambda model_name=name: GraphBaselineModel(model_name, metadata["task_name"], metadata["output_mode"])) for name in cfg.graph_models}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, scipy_stats: Any | None) -> dict[str, float]:
    if y_true.size == 0:
        return {key: float("nan") for key in ["rmse", "mae", "r2", "pearson", "spearman"]}
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    mae = float(np.mean(np.abs(residuals)))
    if y_true.size > 1 and not np.allclose(y_true, y_true[0]):
        ss_res = float(np.sum(np.square(residuals)))
        ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else float("nan")
        if scipy_stats is not None:
            spearman = float(scipy_stats.spearmanr(y_true, y_pred, nan_policy="omit").correlation)
        else:
            y_true_rank = pd.Series(y_true).rank().to_numpy()
            y_pred_rank = pd.Series(y_pred).rank().to_numpy()
            spearman = float(np.corrcoef(y_true_rank, y_pred_rank)[0, 1]) if np.std(y_pred_rank) > 0 else float("nan")
    else:
        r2 = float("nan")
        pearson = float("nan")
        spearman = float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson, "spearman": spearman}


def binary_classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_score >= 0.5).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = max(len(y_true), 1)
    accuracy = (tp + tn) / total
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    balanced_accuracy = np.nanmean([recall, specificity])
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and (precision + recall) else float("nan")
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else float("nan")

    roc_auc = binary_roc_auc(y_true, y_score)
    pr_auc = binary_pr_auc(y_true, y_score)
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "mcc": float(mcc),
        "precision": float(precision),
        "recall": float(recall),
    }


def binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tpr = [0.0]
    fpr = [0.0]
    tp = fp = 0
    for value in y_sorted:
        if value == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
    return float(np.trapz(np.array(tpr), np.array(fpr)))


def binary_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = np.sum(y_true == 1)
    if positives == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    precisions = [1.0]
    recalls = [0.0]
    tp = fp = 0
    for value in y_sorted:
        if value == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / positives)
    return float(np.trapz(np.array(precisions), np.array(recalls)))


def build_loss_function(task_type: str, cfg: AppConfig, runtime: dict[str, Any], positive_weight: float | None = None) -> Any:
    nn = runtime["nn"]
    torch = runtime["torch"]
    if task_type == "regression":
        if cfg.training.regression_loss == "huber":
            return nn.SmoothL1Loss()
        return nn.MSELoss()
    if positive_weight is not None and cfg.training.class_weighting:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], dtype=torch.float))
    return nn.BCEWithLogitsLoss()


def compute_positive_weight(train_items: list[Any]) -> float | None:
    labels = np.array([float(item.y.view(-1).item()) for item in train_items], dtype=float)
    positives = float(np.sum(labels == 1.0))
    negatives = float(np.sum(labels == 0.0))
    if positives <= 0 or negatives <= 0:
        return None
    return negatives / positives


def batch_loss(model: Any, batch: Any, loss_fn: Any, task_type: str, runtime: dict[str, Any]) -> tuple[Any, Any, Any]:
    output, latent = model(batch)
    target = batch.y.view(-1).float()
    if task_type == "classification":
        loss = loss_fn(output, target)
        prediction = runtime["torch"].sigmoid(output)
    else:
        loss = loss_fn(output, target)
        prediction = output
    return loss, prediction.detach(), latent.detach()


def evaluate_loader(model: Any, loader: Any, device: Any, task_type: str, loss_fn: Any, runtime: dict[str, Any]) -> tuple[dict[str, float], pd.DataFrame, np.ndarray | None]:
    torch = runtime["torch"]
    model.eval()
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    latent_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            loss, prediction, latent = batch_loss(model, batch, loss_fn, task_type, runtime)
            losses.append(float(loss.item()))
            y_true = batch.y.view(-1).detach().cpu().numpy().astype(float)
            y_pred = prediction.view(-1).detach().cpu().numpy().astype(float)
            latent_batches.append(latent.detach().cpu().numpy())
            batch_size = len(y_true)
            row_uids = list(batch.row_uid)
            compound_ids = list(batch.compound_id)
            primary_targets = list(batch.primary_target_identifier)
            secondary_targets = list(batch.secondary_target_identifier)
            task_names = list(batch.task_name)
            label_names = list(batch.label_name)
            for idx in range(batch_size):
                rows.append({
                    "row_uid": row_uids[idx],
                    "compound_id": compound_ids[idx],
                    "primary_target_identifier": primary_targets[idx],
                    "secondary_target_identifier": secondary_targets[idx],
                    "task_name": task_names[idx],
                    "label_name": label_names[idx],
                    "observed": float(y_true[idx]),
                    "predicted": float(y_pred[idx]),
                })
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        metrics = {"loss": float("nan")}
        latent_matrix = None
    else:
        y_true = predictions["observed"].to_numpy(dtype=float)
        y_pred = predictions["predicted"].to_numpy(dtype=float)
        if task_type == "classification":
            metrics = {"loss": float(np.mean(losses)) if losses else float("nan")}
            metrics.update(binary_classification_metrics(y_true, y_pred))
        else:
            metrics = {"loss": float(np.mean(losses)) if losses else float("nan")}
            metrics.update(regression_metrics(y_true, y_pred, runtime.get("scipy_stats")))
        latent_matrix = np.concatenate(latent_batches, axis=0) if latent_batches else None
    return metrics, predictions, latent_matrix


def train_one_model(
    train_items: list[Any],
    valid_items: list[Any],
    test_items: list[Any],
    task_type: str,
    model_name: str,
    model_factory: dict[str, Any],
    cfg: AppConfig,
    runtime: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    torch = runtime["torch"]
    model = model_factory[model_name]().to(device)
    positive_weight = compute_positive_weight(train_items) if task_type == "classification" else None
    loss_fn = build_loss_function(task_type, cfg, runtime, positive_weight=positive_weight)
    if positive_weight is not None and hasattr(loss_fn, "pos_weight"):
        loss_fn = loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.training.scheduler == "reduce_on_plateau":
        scheduler = runtime["ReduceLROnPlateau"](optimizer, mode="min", factor=0.5, patience=max(2, cfg.early_stopping_patience // 3))

    train_loader = create_data_loader(train_items, cfg.batch_size, True, runtime, cfg)
    valid_loader = create_data_loader(valid_items, cfg.batch_size, False, runtime, cfg) if valid_items else None
    test_loader = create_data_loader(test_items, cfg.batch_size, False, runtime, cfg)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.training.mixed_precision and device.type == "cuda"))
    best_state = None
    best_epoch = 0
    best_valid_metric = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            if cfg.training.mixed_precision and device.type == "cuda":
                with torch.cuda.amp.autocast(enabled=True):
                    loss, _, _ = batch_loss(model, batch, loss_fn, task_type, runtime)
                scaler.scale(loss).backward()
                if cfg.training.gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, _, _ = batch_loss(model, batch, loss_fn, task_type, runtime)
                loss.backward()
                if cfg.training.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
                optimizer.step()
            train_losses.append(float(loss.item()))

        train_metrics, _, _ = evaluate_loader(model, train_loader, device, task_type, loss_fn, runtime)
        valid_metrics = {"loss": float("nan")}
        if valid_loader is not None:
            valid_metrics, _, _ = evaluate_loader(model, valid_loader, device, task_type, loss_fn, runtime)
            monitor_value = float(valid_metrics["loss"])
        else:
            monitor_value = float(train_metrics["loss"])
        if scheduler is not None and not math.isnan(monitor_value):
            scheduler.step(monitor_value)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"), "valid_loss": monitor_value})
        logging.info("Epoch %s | model=%s | train_loss=%.6f | valid_loss=%.6f", epoch, model_name, history[-1]["train_loss"], monitor_value)

        if monitor_value < best_valid_metric or best_state is None:
            best_valid_metric = monitor_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.early_stopping_patience:
            logging.info("Early stopping triggered for model=%s after epoch %s.", model_name, epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_train_metrics, train_predictions, train_latent = evaluate_loader(model, train_loader, device, task_type, loss_fn, runtime)
    final_valid_metrics, valid_predictions, valid_latent = (
        evaluate_loader(model, valid_loader, device, task_type, loss_fn, runtime)
        if valid_loader is not None
        else ({"loss": float("nan")}, pd.DataFrame(), None)
    )
    final_test_metrics, test_predictions, test_latent = evaluate_loader(model, test_loader, device, task_type, loss_fn, runtime)
    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_metric,
        "history": pd.DataFrame(history),
        "train_metrics": final_train_metrics,
        "valid_metrics": final_valid_metrics,
        "test_metrics": final_test_metrics,
        "train_predictions": train_predictions,
        "valid_predictions": valid_predictions,
        "test_predictions": test_predictions,
        "train_latent": train_latent,
        "valid_latent": valid_latent,
        "test_latent": test_latent,
    }


def summarize_metrics(df: pd.DataFrame, group_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        columns = group_columns + [f"{metric}_{stat}" for metric in metric_columns for stat in ["mean", "std", "min", "max", "n"]]
        return pd.DataFrame(columns=columns)
    summary = df.groupby(group_columns, dropna=False)[metric_columns].agg(["mean", "std", "min", "max", "count"]).reset_index()
    summary.columns = [
        "_".join([str(part) for part in col if str(part) not in {"", "None"}]).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary = summary.rename(columns={col: col.replace("_count", "_n") for col in summary.columns})
    return summary


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    logging.info("Wrote %s", path)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logging.info("Wrote %s", path)


def save_model_artifacts(model: Any, history: pd.DataFrame, output_dir: Path, cfg: AppConfig, runtime: dict[str, Any]) -> dict[str, str]:
    torch = runtime["torch"]
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    if cfg.save_trained_models:
        model_path = output_dir / "best_model.pt"
        torch.save(model.state_dict(), model_path)
        saved["model_path"] = str(model_path)
    if cfg.save_model_checkpoints:
        history_path = output_dir / "training_history.csv"
        history.to_csv(history_path, index=False)
        saved["history_path"] = str(history_path)
    return saved


def save_prediction_outputs(predictions: pd.DataFrame, split_label: str, metadata: dict[str, Any], output_root: Path) -> Path:
    frame = predictions.copy()
    if frame.empty:
        frame = pd.DataFrame(columns=["row_uid", "compound_id", "primary_target_identifier", "secondary_target_identifier", "task_name", "label_name", "observed", "predicted"])
    for key, value in metadata.items():
        frame[key] = value
    frame["split_label"] = split_label
    path = output_root / metadata["task_name"] / metadata["model_name"] / metadata["split_strategy"] / f"{metadata['split_id']}__{metadata['fold_id']}__{split_label}_predictions.csv"
    write_csv(frame, path)
    return path


def save_error_table(predictions: pd.DataFrame, task_type: str, metadata: dict[str, Any], output_root: Path) -> Path:
    if predictions.empty:
        errors = predictions.copy()
    else:
        errors = predictions.copy()
        if task_type == "classification":
            errors["absolute_error"] = np.abs(errors["observed"] - errors["predicted"])
        else:
            errors["residual"] = errors["observed"] - errors["predicted"]
            errors["absolute_error"] = np.abs(errors["residual"])
        errors = errors.sort_values("absolute_error", ascending=False, kind="mergesort")
    for key, value in metadata.items():
        errors[key] = value
    path = output_root / metadata["task_name"] / metadata["model_name"] / metadata["split_strategy"] / f"{metadata['split_id']}__{metadata['fold_id']}__error_table.csv"
    write_csv(errors, path)
    return path


def save_latent_embeddings(latent: np.ndarray | None, predictions: pd.DataFrame, metadata: dict[str, Any], output_root: Path) -> Path | None:
    if latent is None:
        return None
    frame = predictions[["row_uid", "compound_id", "primary_target_identifier", "secondary_target_identifier"]].copy()
    for idx in range(latent.shape[1]):
        frame[f"latent_{idx:03d}"] = latent[:, idx]
    for key, value in metadata.items():
        frame[key] = value
    path = output_root / metadata["task_name"] / metadata["model_name"] / metadata["split_strategy"] / f"{metadata['split_id']}__{metadata['fold_id']}__latent_embeddings.csv"
    write_csv(frame, path)
    return path


def generate_figures(
    regression_summary: pd.DataFrame,
    classification_summary: pd.DataFrame,
    regression_predictions: pd.DataFrame,
    classification_predictions: pd.DataFrame,
    cfg: AppConfig,
    runtime: dict[str, Any],
) -> list[str]:
    plt = runtime.get("plt")
    if plt is None:
        logging.warning("Matplotlib unavailable; skipping figure generation.")
        return ["Matplotlib unavailable; all figure generation skipped."]
    configure_matplotlib(cfg.figure_style, plt)
    warnings: list[str] = []

    def save_figure(fig: Any, base_path: Path) -> None:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        if cfg.export_svg:
            fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
        if cfg.export_png:
            fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight", dpi=cfg.figure_style.dpi_png)
        if cfg.export_pdf:
            fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    if not regression_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_df = regression_summary.sort_values(["task_name", "rmse_mean"], kind="mergesort")
        ax.bar(np.arange(len(plot_df)), plot_df["rmse_mean"], color=[NATURE_PALETTE[i % len(NATURE_PALETTE)] for i in range(len(plot_df))])
        ax.set_xticks(np.arange(len(plot_df)))
        ax.set_xticklabels((plot_df["task_name"] + "\n" + plot_df["model_name"] + "\n" + plot_df["split_strategy"]).tolist(), rotation=45, ha="right")
        ax.set_ylabel("RMSE")
        ax.set_title("Graph baseline regression performance")
        save_figure(fig, cfg.output_figures_root / "regression_performance_comparison")
        write_csv(plot_df, cfg.output_figures_root / "regression_performance_comparison_source_data.csv")

    if not classification_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_df = classification_summary.sort_values(["task_name", "roc_auc_mean"], ascending=[True, False], kind="mergesort")
        ax.bar(np.arange(len(plot_df)), plot_df["roc_auc_mean"], color=[NATURE_PALETTE[i % len(NATURE_PALETTE)] for i in range(len(plot_df))])
        ax.set_xticks(np.arange(len(plot_df)))
        ax.set_xticklabels((plot_df["task_name"] + "\n" + plot_df["label_name"] + "\n" + plot_df["model_name"]).tolist(), rotation=45, ha="right")
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Graph baseline classification performance")
        save_figure(fig, cfg.output_figures_root / "classification_performance_comparison")
        write_csv(plot_df, cfg.output_figures_root / "classification_performance_comparison_source_data.csv")

    if not regression_predictions.empty:
        top_row = regression_summary.sort_values("rmse_mean", kind="mergesort").head(1)
        if not top_row.empty:
            top = top_row.iloc[0]
            subset = regression_predictions[
                (regression_predictions["task_name"] == top["task_name"]) &
                (regression_predictions["model_name"] == top["model_name"]) &
                (regression_predictions["split_strategy"] == top["split_strategy"])
            ].copy()
            if not subset.empty:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(subset["observed"], subset["predicted"], s=18, alpha=0.7, color=NATURE_PALETTE[0])
                bounds = [min(subset["observed"].min(), subset["predicted"].min()), max(subset["observed"].max(), subset["predicted"].max())]
                ax.plot(bounds, bounds, linestyle="--", color=NATURE_PALETTE[3], linewidth=1.5)
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")
                ax.set_title(f"Observed vs predicted: {top['task_name']} / {top['model_name']}")
                save_figure(fig, cfg.output_figures_root / "top_regression_observed_vs_predicted")
                write_csv(subset, cfg.output_figures_root / "top_regression_observed_vs_predicted_source_data.csv")

    if not classification_predictions.empty and not classification_summary.empty:
        top_row = classification_summary.sort_values("roc_auc_mean", ascending=False, kind="mergesort").head(1)
        if not top_row.empty:
            top = top_row.iloc[0]
            subset = classification_predictions[
                (classification_predictions["task_name"] == top["task_name"]) &
                (classification_predictions["label_name"] == top["label_name"]) &
                (classification_predictions["model_name"] == top["model_name"]) &
                (classification_predictions["split_strategy"] == top["split_strategy"])
            ].copy()
            if subset["observed"].nunique() == 2:
                roc_df = build_curve_points(subset["observed"].to_numpy(dtype=int), subset["predicted"].to_numpy(dtype=float), curve_type="roc")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot(roc_df["x"], roc_df["y"], color=NATURE_PALETTE[0], linewidth=2.0)
                ax.plot([0, 1], [0, 1], linestyle="--", color=NATURE_PALETTE[5])
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("True positive rate")
                ax.set_title(f"ROC curve: {top['label_name']} / {top['model_name']}")
                save_figure(fig, cfg.output_figures_root / "top_classification_roc_curve")
                write_csv(roc_df, cfg.output_figures_root / "top_classification_roc_curve_source_data.csv")
                pr_df = build_curve_points(subset["observed"].to_numpy(dtype=int), subset["predicted"].to_numpy(dtype=float), curve_type="pr")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot(pr_df["x"], pr_df["y"], color=NATURE_PALETTE[1], linewidth=2.0)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title(f"PR curve: {top['label_name']} / {top['model_name']}")
                save_figure(fig, cfg.output_figures_root / "top_classification_pr_curve")
                write_csv(pr_df, cfg.output_figures_root / "top_classification_pr_curve_source_data.csv")
    return warnings


def build_curve_points(y_true: np.ndarray, y_score: np.ndarray, curve_type: str) -> pd.DataFrame:
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = fp = 0
    positives = max(int(np.sum(y_true == 1)), 1)
    negatives = max(int(np.sum(y_true == 0)), 1)
    rows = []
    for value in y_sorted:
        if value == 1:
            tp += 1
        else:
            fp += 1
        if curve_type == "roc":
            rows.append({"x": fp / negatives, "y": tp / positives})
        else:
            precision = tp / max(tp + fp, 1)
            recall = tp / positives
            rows.append({"x": recall, "y": precision})
    return pd.DataFrame(rows)


def task_enabled(task_name: str, cfg: AppConfig) -> bool:
    if task_name == "multitask_regression":
        return cfg.run_multitask_regression
    if task_name == "pairwise_selectivity":
        return cfg.run_pairwise_selectivity_regression
    if task_name == "target_vs_panel":
        return cfg.run_target_vs_panel_regression
    if task_name == CLASSIFICATION_TASK_NAME:
        return cfg.run_classification_tasks
    return False


def add_optional_annotations(project_root: Path, task_name: str, df: pd.DataFrame) -> pd.DataFrame:
    if task_name == "pairwise_selectivity":
        return df
    optional_paths = {
        "compound_environment_annotations": project_root / "data/processed/compound_environment_annotations.csv",
        "kinase_environment_annotations": project_root / "data/processed/kinase_environment_annotations.csv",
    }
    enriched = df.copy()
    if "kinase_family" not in enriched.columns and optional_paths["kinase_environment_annotations"].exists() and "target_chembl_id" in enriched.columns:
        kinase_df = pd.read_csv(optional_paths["kinase_environment_annotations"])
        if "target_chembl_id" in kinase_df.columns:
            selected = [col for col in ["target_chembl_id", "kinase_family", "kinase_family_broad_group", "kinase_subfamily"] if col in kinase_df.columns]
            enriched = enriched.merge(kinase_df[selected].drop_duplicates(), on="target_chembl_id", how="left", sort=False)
    return enriched


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg, loaded_config_path, raw_config = load_config(args.config, project_root)
    ensure_output_dirs(cfg)
    log_file, timestamp = setup_logging(cfg.logs_dir)
    logging.info("Starting %s", SCRIPT_NAME)
    runtime = import_runtime_dependencies()
    set_global_determinism(cfg.random_seed, runtime["torch"])
    device = select_device(cfg, runtime["torch"])
    logging.info("Using device: %s", device)
    config_snapshot_path = save_config_snapshot(cfg, loaded_config_path)

    manifest_df = load_split_manifest(cfg.input_split_manifest_path)
    task_tables: dict[str, pd.DataFrame] = {}
    task_reports: dict[str, Any] = {}
    warnings_list: list[str] = []
    compound_frames: list[pd.DataFrame] = []

    for task_name, spec in REGRESSION_TASKS.items():
        if not task_enabled(task_name, cfg):
            continue
        raw_df = load_required_dataframe(getattr(cfg, spec["config_key"]), f"{task_name} dataset")
        normalized, mapping, local_warnings = standardize_task_dataframe(task_name, raw_df, spec["required_columns"])
        if "standardized_smiles" not in normalized.columns:
            raise ValueError(f"{task_name} requires `standardized_smiles` for graph construction.")
        normalized = add_optional_annotations(project_root, task_name, normalized)
        task_tables[task_name] = normalized
        compound_frames.append(normalized[["compound_id", "standardized_smiles"]])
        task_reports[task_name] = {
            "n_rows": int(len(normalized)),
            "n_unique_compounds": int(normalized["compound_id"].nunique()),
            "n_unique_targets": int(normalized["target_chembl_id"].nunique()) if "target_chembl_id" in normalized.columns else None,
            "column_mapping": mapping,
            "warnings": local_warnings,
        }
        warnings_list.extend([f"{task_name}: {item}" for item in local_warnings])

    classification_labels: list[str] = []
    if task_enabled(CLASSIFICATION_TASK_NAME, cfg):
        raw_df = load_required_dataframe(cfg.input_classification_path, "classification dataset")
        normalized, mapping, local_warnings = standardize_task_dataframe(CLASSIFICATION_TASK_NAME, raw_df, ["standardized_smiles"])
        normalized = add_optional_annotations(project_root, CLASSIFICATION_TASK_NAME, normalized)
        classification_labels = detect_classification_labels(normalized)
        if not classification_labels:
            warnings_list.append("classification: no classification labels detected; task skipped.")
            logging.warning("No classification labels detected; skipping classification tasks.")
        else:
            logging.info("Detected classification labels: %s", ", ".join(classification_labels))
            task_tables[CLASSIFICATION_TASK_NAME] = normalized
            compound_frames.append(normalized[["compound_id", "standardized_smiles"]])
            task_reports[CLASSIFICATION_TASK_NAME] = {
                "n_rows": int(len(normalized)),
                "n_unique_compounds": int(normalized["compound_id"].nunique()),
                "n_unique_targets": int(normalized["target_chembl_id"].nunique()) if "target_chembl_id" in normalized.columns else None,
                "column_mapping": mapping,
                "warnings": local_warnings,
                "label_columns": classification_labels,
            }
            warnings_list.extend([f"classification: {item}" for item in local_warnings])

    if not compound_frames:
        raise ValueError("No tasks enabled or no valid task tables available for Script-08.")

    graph_cache = GraphCache(cfg, runtime)
    graph_cache.build_for_compounds(pd.concat(compound_frames, ignore_index=True).drop_duplicates())
    graph_metadata_path = cfg.output_metrics_root / "graph_cache_metadata.csv"
    write_csv(pd.DataFrame(graph_cache.metadata_records), graph_metadata_path)

    embedding_maps = {
        "target_ids": create_embedding_map(
            pd.concat([task_tables[name]["target_chembl_id"].astype(str) for name in task_tables if name in {"multitask_regression", "target_vs_panel", CLASSIFICATION_TASK_NAME} and "target_chembl_id" in task_tables[name].columns], ignore_index=True)
            if any(name in task_tables and "target_chembl_id" in task_tables[name].columns for name in {"multitask_regression", "target_vs_panel", CLASSIFICATION_TASK_NAME})
            else []
        ),
        "kinase_families": create_embedding_map(
            pd.concat([task_tables[name]["kinase_family"].astype(str) for name in task_tables if "kinase_family" in task_tables[name].columns], ignore_index=True)
            if any("kinase_family" in task_tables[name].columns for name in task_tables)
            else []
        ),
        "pair_target_ids": create_embedding_map(
            pd.concat([task_tables["pairwise_selectivity"]["kinase_a_chembl_id"].astype(str), task_tables["pairwise_selectivity"]["kinase_b_chembl_id"].astype(str)], ignore_index=True)
            if "pairwise_selectivity" in task_tables
            else []
        ),
    }

    node_feature_dim, edge_feature_dim = infer_feature_dimensions(graph_cache)
    all_regression_metrics: list[dict[str, Any]] = []
    all_classification_metrics: list[dict[str, Any]] = []
    saved_prediction_frames_regression: list[pd.DataFrame] = []
    saved_prediction_frames_classification: list[pd.DataFrame] = []
    report_model_records: list[dict[str, Any]] = []

    for task_name, df in task_tables.items():
        task_manifest = manifest_df[manifest_df["task_name"] == task_name].copy()
        if task_manifest.empty:
            warnings_list.append(f"{task_name}: no split-manifest rows found; task skipped.")
            logging.warning("No split-manifest rows found for %s; skipping task.", task_name)
            continue

        if task_name == CLASSIFICATION_TASK_NAME:
            labels_to_run = classification_labels
        else:
            labels_to_run = [REGRESSION_TASKS[task_name]["target_column"]]

        for label_name in labels_to_run:
            if task_name == CLASSIFICATION_TASK_NAME:
                label_series = prepare_classification_target(df[label_name])
                working_df = df.loc[label_series.index].copy()
                working_df[label_name] = label_series.astype(float)
                task_type = "classification"
                target_column = label_name
            else:
                working_df = df.copy()
                task_type = "regression"
                target_column = label_name
            working_df = apply_embedding_mappings(working_df, task_name, embedding_maps)
            data_objects, data_metadata = build_task_data_objects(working_df, task_name, target_column, graph_cache, runtime, label_name if task_name == CLASSIFICATION_TASK_NAME else None)
            if not data_objects:
                warnings_list.append(f"{task_name}/{label_name}: no model-ready rows after graph/label filtering; skipped.")
                logging.warning("No model-ready rows for %s / %s; skipping.", task_name, label_name)
                continue

            model_factory = make_model_factory(
                runtime,
                cfg,
                {
                    "task_name": task_name,
                    "output_mode": task_type,
                    "node_feature_dim": node_feature_dim,
                    "edge_feature_dim": edge_feature_dim,
                    "target_vocab_size": len(embedding_maps["target_ids"]),
                    "family_vocab_size": len(embedding_maps["kinase_families"]),
                    "pair_vocab_size": len(embedding_maps["pair_target_ids"]),
                },
            )

            for manifest_row in task_manifest.itertuples(index=False):
                assignment_df = load_assignment_table(getattr(manifest_row, "output_assignment_path"), project_root)
                ensure_assignment_columns(assignment_df, task_name)
                assignment_df = assignment_df[["row_uid", "split_label", "split_strategy", "split_id", "fold_id"]].drop_duplicates()
                partitions = split_data_objects(data_objects, assignment_df)
                if not partitions["train"] or not partitions["test"]:
                    warning = (
                        f"{task_name}/{label_name}/{manifest_row.split_strategy}/{manifest_row.split_id}: "
                        "missing train or test rows after assignment join; combination skipped."
                    )
                    warnings_list.append(warning)
                    logging.warning(warning)
                    continue
                logging.info(
                    "Training task=%s label=%s split=%s split_id=%s fold=%s | train=%s valid=%s test=%s",
                    task_name,
                    label_name,
                    manifest_row.split_strategy,
                    manifest_row.split_id,
                    manifest_row.fold_id,
                    len(partitions["train"]),
                    len(partitions["valid"]),
                    len(partitions["test"]),
                )
                for model_name in cfg.graph_models:
                    result = train_one_model(
                        train_items=partitions["train"],
                        valid_items=partitions["valid"],
                        test_items=partitions["test"],
                        task_type=task_type,
                        model_name=model_name,
                        model_factory=model_factory,
                        cfg=cfg,
                        runtime=runtime,
                        device=device,
                    )
                    metadata = {
                        "task_name": task_name,
                        "label_name": label_name if task_type == "classification" else "",
                        "model_name": model_name,
                        "split_strategy": str(manifest_row.split_strategy),
                        "split_id": str(manifest_row.split_id),
                        "fold_id": str(manifest_row.fold_id),
                    }
                    model_dir = cfg.output_model_root / task_name / (label_name if task_type == "classification" else "default") / model_name / str(manifest_row.split_strategy) / f"{manifest_row.split_id}__{manifest_row.fold_id}"
                    artifact_paths = save_model_artifacts(result["model"], result["history"], model_dir, cfg, runtime)

                    prediction_root = cfg.output_predictions_root / (label_name if task_type == "classification" else "default")
                    train_prediction_path = save_prediction_outputs(result["train_predictions"], "train", metadata, prediction_root) if cfg.save_fold_predictions else None
                    valid_prediction_path = save_prediction_outputs(result["valid_predictions"], "valid", metadata, prediction_root) if cfg.save_fold_predictions else None
                    test_prediction_path = save_prediction_outputs(result["test_predictions"], "test", metadata, prediction_root) if cfg.save_test_predictions else None
                    error_table_path = save_error_table(result["test_predictions"], task_type, metadata, cfg.output_metrics_root / "error_tables") if cfg.save_error_tables else None
                    latent_path = save_latent_embeddings(result["test_latent"], result["test_predictions"], metadata, cfg.output_metrics_root / "latent_embeddings") if cfg.save_latent_embeddings else None

                    metrics_record = {
                        **metadata,
                        "n_train": len(partitions["train"]),
                        "n_valid": len(partitions["valid"]),
                        "n_test": len(partitions["test"]),
                        "best_epoch": result["best_epoch"],
                        "best_valid_loss": result["best_valid_loss"],
                        **{f"train_{key}": value for key, value in result["train_metrics"].items()},
                        **{f"valid_{key}": value for key, value in result["valid_metrics"].items()},
                        **{f"test_{key}": value for key, value in result["test_metrics"].items()},
                        "model_path": artifact_paths.get("model_path"),
                        "history_path": artifact_paths.get("history_path"),
                        "train_prediction_path": str(train_prediction_path) if train_prediction_path else None,
                        "valid_prediction_path": str(valid_prediction_path) if valid_prediction_path else None,
                        "test_prediction_path": str(test_prediction_path) if test_prediction_path else None,
                        "error_table_path": str(error_table_path) if error_table_path else None,
                        "latent_embedding_path": str(latent_path) if latent_path else None,
                    }
                    if task_type == "classification":
                        all_classification_metrics.append(metrics_record)
                    else:
                        all_regression_metrics.append(metrics_record)
                    if not result["test_predictions"].empty:
                        enriched_predictions = result["test_predictions"].copy()
                        for key, value in metadata.items():
                            enriched_predictions[key] = value
                        if task_type == "classification":
                            saved_prediction_frames_classification.append(enriched_predictions)
                        else:
                            saved_prediction_frames_regression.append(enriched_predictions)
                    report_model_records.append({
                        **metadata,
                        "task_type": task_type,
                        "data_metadata": data_metadata,
                        "artifact_paths": artifact_paths,
                    })

    regression_per_fold = pd.DataFrame(all_regression_metrics)
    classification_per_fold = pd.DataFrame(all_classification_metrics)
    regression_metric_columns = [col for col in ["test_rmse", "test_mae", "test_r2", "test_pearson", "test_spearman"] if col in regression_per_fold.columns]
    classification_metric_columns = [col for col in ["test_roc_auc", "test_pr_auc", "test_accuracy", "test_balanced_accuracy", "test_f1", "test_mcc", "test_precision", "test_recall"] if col in classification_per_fold.columns]

    regression_summary = summarize_metrics(regression_per_fold, ["task_name", "model_name", "split_strategy"], regression_metric_columns)
    regression_summary = regression_summary.rename(columns={
        "test_rmse_mean": "rmse_mean", "test_rmse_std": "rmse_std", "test_rmse_min": "rmse_min", "test_rmse_max": "rmse_max", "test_rmse_n": "rmse_n",
        "test_mae_mean": "mae_mean", "test_mae_std": "mae_std", "test_mae_min": "mae_min", "test_mae_max": "mae_max", "test_mae_n": "mae_n",
        "test_r2_mean": "r2_mean", "test_r2_std": "r2_std", "test_r2_min": "r2_min", "test_r2_max": "r2_max", "test_r2_n": "r2_n",
        "test_pearson_mean": "pearson_mean", "test_pearson_std": "pearson_std", "test_pearson_min": "pearson_min", "test_pearson_max": "pearson_max", "test_pearson_n": "pearson_n",
        "test_spearman_mean": "spearman_mean", "test_spearman_std": "spearman_std", "test_spearman_min": "spearman_min", "test_spearman_max": "spearman_max", "test_spearman_n": "spearman_n",
    })
    classification_summary = summarize_metrics(classification_per_fold, ["task_name", "label_name", "model_name", "split_strategy"], classification_metric_columns)
    classification_summary = classification_summary.rename(columns={
        "test_roc_auc_mean": "roc_auc_mean", "test_roc_auc_std": "roc_auc_std", "test_roc_auc_min": "roc_auc_min", "test_roc_auc_max": "roc_auc_max", "test_roc_auc_n": "roc_auc_n",
        "test_pr_auc_mean": "pr_auc_mean", "test_pr_auc_std": "pr_auc_std", "test_pr_auc_min": "pr_auc_min", "test_pr_auc_max": "pr_auc_max", "test_pr_auc_n": "pr_auc_n",
        "test_accuracy_mean": "accuracy_mean", "test_accuracy_std": "accuracy_std", "test_accuracy_min": "accuracy_min", "test_accuracy_max": "accuracy_max", "test_accuracy_n": "accuracy_n",
        "test_balanced_accuracy_mean": "balanced_accuracy_mean", "test_balanced_accuracy_std": "balanced_accuracy_std", "test_balanced_accuracy_min": "balanced_accuracy_min", "test_balanced_accuracy_max": "balanced_accuracy_max", "test_balanced_accuracy_n": "balanced_accuracy_n",
        "test_f1_mean": "f1_mean", "test_f1_std": "f1_std", "test_f1_min": "f1_min", "test_f1_max": "f1_max", "test_f1_n": "f1_n",
        "test_mcc_mean": "mcc_mean", "test_mcc_std": "mcc_std", "test_mcc_min": "mcc_min", "test_mcc_max": "mcc_max", "test_mcc_n": "mcc_n",
        "test_precision_mean": "precision_mean", "test_precision_std": "precision_std", "test_precision_min": "precision_min", "test_precision_max": "precision_max", "test_precision_n": "precision_n",
        "test_recall_mean": "recall_mean", "test_recall_std": "recall_std", "test_recall_min": "recall_min", "test_recall_max": "recall_max", "test_recall_n": "recall_n",
    })

    write_csv(regression_per_fold, cfg.output_metrics_root / "regression_metrics_per_fold.csv")
    write_csv(regression_summary, cfg.output_metrics_root / "regression_metrics_summary.csv")
    write_csv(classification_per_fold, cfg.output_metrics_root / "classification_metrics_per_fold.csv")
    write_csv(classification_summary, cfg.output_metrics_root / "classification_metrics_summary.csv")

    regression_predictions = pd.concat(saved_prediction_frames_regression, ignore_index=True) if saved_prediction_frames_regression else pd.DataFrame()
    classification_predictions = pd.concat(saved_prediction_frames_classification, ignore_index=True) if saved_prediction_frames_classification else pd.DataFrame()

    figure_warnings: list[str] = []
    if cfg.make_figures:
        figure_warnings = generate_figures(regression_summary, classification_summary, regression_predictions, classification_predictions, cfg, runtime)
        warnings_list.extend(figure_warnings)

    best_regression = []
    if not regression_summary.empty and "rmse_mean" in regression_summary.columns:
        for task_name, frame in regression_summary.groupby("task_name", dropna=False):
            record = frame.sort_values("rmse_mean", kind="mergesort").iloc[0].to_dict()
            best_regression.append(record)
    best_classification = []
    if not classification_summary.empty and "roc_auc_mean" in classification_summary.columns:
        for keys, frame in classification_summary.groupby(["task_name", "label_name"], dropna=False):
            record = frame.sort_values("roc_auc_mean", ascending=False, kind="mergesort").iloc[0].to_dict()
            best_classification.append(record)

    report_payload = {
        "script_name": SCRIPT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "log_file": str(log_file),
        "config_path": str(loaded_config_path),
        "config_snapshot_path": str(config_snapshot_path) if config_snapshot_path else None,
        "input_paths": {
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
        "tasks_processed": sorted(task_tables.keys()),
        "models_trained": cfg.graph_models,
        "split_strategies_processed": sorted(set(manifest_df["split_strategy"].astype(str).tolist())),
        "graph_feature_settings": {
            "node_features": cfg.node_features,
            "edge_features": cfg.edge_features,
            "node_feature_names": graph_cache.node_feature_names,
            "edge_feature_names": graph_cache.edge_feature_names,
            "node_feature_dim": node_feature_dim,
            "edge_feature_dim": edge_feature_dim,
        },
        "graph_construction": {
            "n_unique_compounds_processed": int(pd.concat(compound_frames, ignore_index=True)["compound_id"].nunique()),
            "n_successful_graphs": int(len(graph_cache.cache)),
            "n_failed_graphs": int(len(graph_cache.invalid_compounds)),
            "graph_metadata_path": str(graph_metadata_path),
        },
        "task_reports": task_reports,
        "regression_summary_metrics": regression_summary.to_dict(orient="records"),
        "classification_summary_metrics": classification_summary.to_dict(orient="records"),
        "best_model_per_task": {
            "regression": best_regression,
            "classification": best_classification,
        },
        "best_model_per_split_strategy": {
            "regression": regression_summary.sort_values("rmse_mean", kind="mergesort").groupby("split_strategy", dropna=False).head(1).to_dict(orient="records") if not regression_summary.empty else [],
            "classification": classification_summary.sort_values("roc_auc_mean", ascending=False, kind="mergesort").groupby("split_strategy", dropna=False).head(1).to_dict(orient="records") if not classification_summary.empty else [],
        },
        "model_run_records": report_model_records,
        "warnings": warnings_list,
        "runtime": {
            "device": str(device),
            "random_seed": cfg.random_seed,
            "batch_size": cfg.batch_size,
            "max_epochs": cfg.max_epochs,
            "early_stopping_patience": cfg.early_stopping_patience,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "mixed_precision": cfg.training.mixed_precision,
        },
    }
    write_json(report_payload, cfg.output_report_path)
    logging.info("Completed %s", SCRIPT_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
