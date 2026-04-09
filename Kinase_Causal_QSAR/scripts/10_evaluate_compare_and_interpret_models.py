#!/usr/bin/env python3
"""Integrate, compare, and interpret model outputs across Steps 07-09.

This script is a strict continuation of Steps 07-09 of the kinase causal-QSAR
pipeline. It performs deterministic, config-driven evaluation only: it does not
retrain any models. The script discovers previously generated metrics,
predictions, ablation outputs, and reports; normalizes heterogeneous schemas;
computes unified comparison tables; generates publication-grade summary figures;
and writes a manuscript-oriented JSON report with provenance details.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from pandas.errors import EmptyDataError

SCRIPT_NAME = "10_evaluate_compare_and_interpret_models"
DEFAULT_SEED = 2025
CLASSIFICATION_TASK_NAME = "classification"
NATURE_PALETTE = ["#386CB0", "#F39C12", "#2CA25F", "#E74C3C", "#756BB1", "#7F8C8D"]

REQUIRED_SCRIPT_10_KEYS = {
    "input_classical_results_root",
    "input_deep_results_root",
    "input_causal_results_root",
    "input_annotated_long_path",
    "input_activity_cliff_path",
    "input_kinase_env_path",
    "input_compound_env_path",
    "input_split_manifest_path",
    "output_results_root",
    "output_figures_root",
    "output_report_path",
    "compare_regression_tasks",
    "compare_classification_tasks",
    "compare_split_strategies",
    "compare_low_data_subsets",
    "compare_activity_cliff_performance",
    "compare_environment_group_performance",
    "compare_ablations",
    "best_model_selection_metric_regression",
    "best_model_selection_metric_classification",
    "aggregate_across_folds",
    "aggregate_across_split_ids",
    "make_figures",
    "export_svg",
    "export_png",
    "export_pdf",
    "figure_style",
    "save_source_data_for_figures",
    "save_config_snapshot",
}

REGRESSION_METRICS = ["rmse", "mae", "r2", "spearman", "pearson"]
CLASSIFICATION_METRICS = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1", "mcc"]
LOWER_IS_BETTER = {"rmse", "mae"}
HIGHER_IS_BETTER = {"r2", "spearman", "pearson", "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1", "mcc"}
FULL_CAUSAL_NAMES = {"main", "full", "full_model", "causal_full", "default"}
CORE_ABLATION_ORDER = [
    "main",
    "no_environment_objectives",
    "no_adversarial_loss",
    "no_invariant_loss",
    "no_activity_cliff_regularization",
    "no_target_embedding",
]

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "rmse": ("rmse", "test_rmse", "RMSE"),
    "mae": ("mae", "test_mae", "MAE"),
    "r2": ("r2", "test_r2", "R2"),
    "pearson": ("pearson", "test_pearson", "Pearson"),
    "spearman": ("spearman", "test_spearman", "Spearman"),
    "roc_auc": ("roc_auc", "test_roc_auc", "ROC_AUC", "ROC-AUC"),
    "pr_auc": ("pr_auc", "test_pr_auc", "PR_AUC", "PR-AUC", "average_precision"),
    "accuracy": ("accuracy", "test_accuracy", "Accuracy"),
    "balanced_accuracy": ("balanced_accuracy", "test_balanced_accuracy", "BalancedAccuracy", "balanced_acc"),
    "f1": ("f1", "test_f1", "F1"),
    "mcc": ("mcc", "test_mcc", "MCC"),
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "task_name": ("task_name", "task", "dataset_name"),
    "model_name": ("model_name", "model", "architecture", "estimator_name"),
    "label_name": ("label_name", "target_label", "label_column"),
    "split_strategy": ("split_strategy", "split_type", "split"),
    "split_id": ("split_id", "split_index", "repeat_id"),
    "fold_id": ("fold_id", "fold", "cv_fold"),
    "evaluation_split": ("evaluation_split", "split_label", "partition"),
    "ablation_name": ("ablation_name",),
    "row_uid": ("row_uid",),
    "compound_id": ("compound_id", "molecule_id", "standardized_smiles"),
    "standardized_smiles": ("standardized_smiles", "smiles"),
    "target_chembl_id": ("target_chembl_id", "primary_target_identifier", "target_id"),
    "secondary_target_identifier": ("secondary_target_identifier", "kinase_b_chembl_id"),
    "target_name": ("target_name", "pref_name"),
    "observed": ("observed", "y_true", "actual"),
    "predicted": ("predicted", "y_pred", "prediction"),
    "prediction_score": ("prediction_score", "predicted_probability", "score"),
    "kinase_family": ("kinase_family", "kinase_family_label", "target_family", "broad_kinase_family"),
    "murcko_scaffold": ("murcko_scaffold", "scaffold"),
    "generic_murcko_scaffold": ("generic_murcko_scaffold", "generic_scaffold"),
    "source_id": ("source_id",),
    "source_description": ("source_description",),
    "source_frequency_bin": ("source_frequency_bin",),
    "activity_cliff_flag": ("activity_cliff_flag", "cliff_flag"),
}

ANNOTATION_COLUMNS = [
    "row_uid",
    "compound_id",
    "standardized_smiles",
    "target_chembl_id",
    "target_name",
    "kinase_family",
    "murcko_scaffold",
    "generic_murcko_scaffold",
    "source_id",
    "source_description",
    "source_frequency_bin",
    "activity_cliff_flag",
]


@dataclass
class FigureStyle:
    font_family: str
    bold_text: bool
    output_format_primary: str
    palette_name: str
    dpi_png: int


@dataclass
class AppConfig:
    input_classical_results_root: Path
    input_deep_results_root: Path
    input_causal_results_root: Path
    input_annotated_long_path: Path
    input_activity_cliff_path: Path | None
    input_kinase_env_path: Path | None
    input_compound_env_path: Path | None
    input_split_manifest_path: Path
    output_results_root: Path
    output_figures_root: Path
    output_report_path: Path
    compare_regression_tasks: bool
    compare_classification_tasks: bool
    compare_split_strategies: bool
    compare_low_data_subsets: bool
    compare_activity_cliff_performance: bool
    compare_environment_group_performance: bool
    compare_ablations: bool
    best_model_selection_metric_regression: str
    best_model_selection_metric_classification: str
    aggregate_across_folds: bool
    aggregate_across_split_ids: bool
    make_figures: bool
    export_svg: bool
    export_png: bool
    export_pdf: bool
    figure_style: FigureStyle
    save_source_data_for_figures: bool
    save_config_snapshot: bool
    logs_dir: Path
    configs_used_dir: Path
    project_root: Path

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        if not isinstance(raw, dict):
            raise ValueError("Config YAML must contain a top-level mapping.")
        section = raw.get("script_10")
        if not isinstance(section, dict):
            raise ValueError("Missing required `script_10` section in config.yaml.")
        missing = sorted(REQUIRED_SCRIPT_10_KEYS.difference(section))
        if missing:
            raise ValueError("Missing required script_10 config values: " + ", ".join(missing))

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
            raise ValueError(f"script_10.{key} must be boolean; got {value!r}.")

        figure_raw = section.get("figure_style")
        if not isinstance(figure_raw, dict):
            raise ValueError("script_10.figure_style must be a mapping.")

        regression_metric = str(section["best_model_selection_metric_regression"]).strip().lower()
        classification_metric = str(section["best_model_selection_metric_classification"]).strip().lower()
        if regression_metric not in REGRESSION_METRICS:
            raise ValueError(f"script_10.best_model_selection_metric_regression must be one of {REGRESSION_METRICS}; got {regression_metric!r}.")
        if classification_metric not in CLASSIFICATION_METRICS:
            raise ValueError(f"script_10.best_model_selection_metric_classification must be one of {CLASSIFICATION_METRICS}; got {classification_metric!r}.")

        return AppConfig(
            input_classical_results_root=resolve(section["input_classical_results_root"]),
            input_deep_results_root=resolve(section["input_deep_results_root"]),
            input_causal_results_root=resolve(section["input_causal_results_root"]),
            input_annotated_long_path=resolve(section["input_annotated_long_path"]),
            input_activity_cliff_path=resolve(section.get("input_activity_cliff_path")),
            input_kinase_env_path=resolve(section.get("input_kinase_env_path")),
            input_compound_env_path=resolve(section.get("input_compound_env_path")),
            input_split_manifest_path=resolve(section["input_split_manifest_path"]),
            output_results_root=resolve(section["output_results_root"]),
            output_figures_root=resolve(section["output_figures_root"]),
            output_report_path=resolve(section["output_report_path"]),
            compare_regression_tasks=parse_bool(section["compare_regression_tasks"], "compare_regression_tasks"),
            compare_classification_tasks=parse_bool(section["compare_classification_tasks"], "compare_classification_tasks"),
            compare_split_strategies=parse_bool(section["compare_split_strategies"], "compare_split_strategies"),
            compare_low_data_subsets=parse_bool(section["compare_low_data_subsets"], "compare_low_data_subsets"),
            compare_activity_cliff_performance=parse_bool(section["compare_activity_cliff_performance"], "compare_activity_cliff_performance"),
            compare_environment_group_performance=parse_bool(section["compare_environment_group_performance"], "compare_environment_group_performance"),
            compare_ablations=parse_bool(section["compare_ablations"], "compare_ablations"),
            best_model_selection_metric_regression=regression_metric,
            best_model_selection_metric_classification=classification_metric,
            aggregate_across_folds=parse_bool(section["aggregate_across_folds"], "aggregate_across_folds"),
            aggregate_across_split_ids=parse_bool(section["aggregate_across_split_ids"], "aggregate_across_split_ids"),
            make_figures=parse_bool(section["make_figures"], "make_figures"),
            export_svg=parse_bool(section["export_svg"], "export_svg"),
            export_png=parse_bool(section["export_png"], "export_png"),
            export_pdf=parse_bool(section["export_pdf"], "export_pdf"),
            figure_style=FigureStyle(
                font_family=str(figure_raw.get("font_family", "Times New Roman")),
                bold_text=parse_bool(figure_raw.get("bold_text", True), "figure_style.bold_text"),
                output_format_primary=str(figure_raw.get("output_format_primary", "svg")).lower(),
                palette_name=str(figure_raw.get("palette_name", "nature_manuscript_palette")),
                dpi_png=int(figure_raw.get("dpi_png", 300)),
            ),
            save_source_data_for_figures=parse_bool(section["save_source_data_for_figures"], "save_source_data_for_figures"),
            save_config_snapshot=parse_bool(section["save_config_snapshot"], "save_config_snapshot"),
            logs_dir=project_root / "logs",
            configs_used_dir=project_root / "configs_used",
            project_root=project_root,
        )


@dataclass
class ResultBundle:
    family: str
    root: Path
    report_path: Path | None
    regression_per_fold: pd.DataFrame
    regression_summary: pd.DataFrame
    classification_per_fold: pd.DataFrame
    classification_summary: pd.DataFrame
    ablation_summary: pd.DataFrame
    activity_cliff_metrics: pd.DataFrame
    environment_group_metrics: pd.DataFrame
    predictions: pd.DataFrame
    discovered_files: list[str]
    warnings: list[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML (default: config.yaml relative to project root).")
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must define a top-level mapping: {path}")
    return payload


def configure_logging(cfg: AppConfig) -> Path:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.logs_dir / f"{SCRIPT_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers, force=True)
    return log_path


def set_global_determinism(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_config_snapshot(config_path: Path, raw_config: dict[str, Any], cfg: AppConfig) -> Path | None:
    if not cfg.save_config_snapshot:
        return None
    cfg.configs_used_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = cfg.configs_used_dir / f"{SCRIPT_NAME}_config.yaml"
    snapshot_path.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    logging.info("Saved config snapshot to %s", snapshot_path)
    return snapshot_path


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {description} not found: {path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    logging.info("Wrote %s", path)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logging.info("Wrote %s", path)


def alias_lookup(frame: pd.DataFrame, canonical: str) -> str | None:
    for candidate in COLUMN_ALIASES.get(canonical, (canonical,)):
        if candidate in frame.columns:
            return candidate
    return None


def alias_value(frame: pd.DataFrame, canonical: str, default: Any = np.nan) -> pd.Series:
    column = alias_lookup(frame, canonical)
    if column is None:
        return pd.Series([default] * len(frame), index=frame.index)
    return frame[column]


def metric_column(frame: pd.DataFrame, metric: str) -> str | None:
    for candidate in METRIC_ALIASES.get(metric, (metric,)):
        if candidate in frame.columns:
            return candidate
    return None


def coerce_string_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("string")
    return frame


def normalize_split_strategy(value: Any) -> str:
    text = str(value).strip().lower()
    mapping = {
        "random": "random",
        "random_split": "random",
        "scaffold": "scaffold",
        "scaffold_split": "scaffold",
        "kinase_family": "kinase_family_grouped",
        "kinase_family_grouped": "kinase_family_grouped",
        "family_grouped": "kinase_family_grouped",
        "grouped_family": "kinase_family_grouped",
        "source": "source_environment_grouped",
        "source_environment": "source_environment_grouped",
        "source_environment_grouped": "source_environment_grouped",
        "environment": "source_environment_grouped",
    }
    return mapping.get(text, text)


def normalize_task_name(value: Any, label_name: Any = None) -> str:
    text = str(value).strip().lower()
    if text in {"multitask", "multitask_regression", "pki", "pki_regression"}:
        return "multitask_regression"
    if text in {"pairwise", "pairwise_selectivity", "pairwise_selectivity_regression"}:
        return "pairwise_selectivity"
    if text in {"target_vs_panel", "target_vs_panel_selectivity", "target_vs_panel_regression"}:
        return "target_vs_panel"
    if text in {"classification", "derived_classification", "binary_classification"}:
        suffix = str(label_name).strip() if label_name is not None and not pd.isna(label_name) else "default"
        return f"classification::{suffix}"
    return text


def normalize_ablation_name(value: Any, family: str) -> str:
    if family != "causal":
        return "none"
    text = str(value).strip().lower() if value is not None and not pd.isna(value) else "main"
    return "main" if text in FULL_CAUSAL_NAMES else text


def infer_task_type(task_name: str) -> str:
    return "classification" if str(task_name).startswith("classification::") else "regression"


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def standard_error(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= 1:
        return float("nan")
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def compute_metric_frame(y_true: Sequence[float], y_pred: Sequence[float], task_type: str) -> dict[str, float]:
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[mask]
    predicted = predicted[mask]
    if observed.size == 0:
        return {}
    metrics: dict[str, float] = {}
    if task_type == "regression":
        residual = observed - predicted
        metrics["rmse"] = float(np.sqrt(np.mean(np.square(residual))))
        metrics["mae"] = float(np.mean(np.abs(residual)))
        if observed.size > 1 and not np.allclose(observed, observed[0]):
            ss_res = float(np.sum(np.square(residual)))
            ss_tot = float(np.sum(np.square(observed - observed.mean())))
            metrics["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            try:
                from scipy import stats as scipy_stats  # type: ignore

                pearson = scipy_stats.pearsonr(observed, predicted)
                spearman = scipy_stats.spearmanr(observed, predicted)
                metrics["pearson"] = float(pearson.statistic)
                metrics["spearman"] = float(spearman.statistic)
            except Exception:
                metrics["pearson"] = float(pd.Series(observed).corr(pd.Series(predicted), method="pearson"))
                metrics["spearman"] = float(pd.Series(observed).corr(pd.Series(predicted), method="spearman"))
        else:
            metrics.update({"r2": float("nan"), "pearson": float("nan"), "spearman": float("nan")})
    else:
        y_bin = observed.astype(int)
        score = predicted.astype(float)
        pred_label = (score >= 0.5).astype(int)
        tp = int(np.sum((pred_label == 1) & (y_bin == 1)))
        tn = int(np.sum((pred_label == 0) & (y_bin == 0)))
        fp = int(np.sum((pred_label == 1) & (y_bin == 0)))
        fn = int(np.sum((pred_label == 0) & (y_bin == 1)))
        total = max(len(y_bin), 1)
        metrics["accuracy"] = float((tp + tn) / total)
        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        tnr = tn / (tn + fp) if (tn + fp) else float("nan")
        metrics["balanced_accuracy"] = float(np.nanmean([tpr, tnr]))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        metrics["f1"] = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
        metrics["mcc"] = float(((tp * tn) - (fp * fn)) / denom) if denom else 0.0
        try:
            from scipy import stats as scipy_stats  # noqa: F401
            from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore

            metrics["roc_auc"] = float(roc_auc_score(y_bin, score)) if len(np.unique(y_bin)) > 1 else float("nan")
            metrics["pr_auc"] = float(average_precision_score(y_bin, score)) if len(np.unique(y_bin)) > 1 else float("nan")
        except Exception:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")
    return metrics


def aggregate_metrics(df: pd.DataFrame, group_columns: Sequence[str], metric_columns: Sequence[str]) -> pd.DataFrame:
    metric_columns = [column for column in metric_columns if column in df.columns]
    if df.empty:
        columns = list(group_columns) + [f"{metric}_{suffix}" for metric in metric_columns for suffix in ["mean", "std", "sem", "n"]]
        return pd.DataFrame(columns=columns)
    grouped = df.groupby(list(group_columns), dropna=False, sort=True)
    rows: list[dict[str, Any]] = []
    for keys, frame in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record = {column: key_values[idx] for idx, column in enumerate(group_columns)}
        record["n_rows"] = int(len(frame))
        if "fold_id" in frame.columns:
            record["n_folds"] = int(frame["fold_id"].astype(str).nunique())
        if "split_id" in frame.columns:
            record["n_split_ids"] = int(frame["split_id"].astype(str).nunique())
        for metric in metric_columns:
            values = safe_numeric(frame[metric]).dropna()
            record[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            record[f"{metric}_sem"] = standard_error(values)
            record[f"{metric}_n"] = int(len(values))
        rows.append(record)
    return pd.DataFrame(rows).sort_values(list(group_columns), kind="mergesort").reset_index(drop=True)


def add_schema_defaults(frame: pd.DataFrame, family: str, task_type: str | None = None) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["model_family"] = family
    normalized["task_name"] = [normalize_task_name(task, label) for task, label in zip(alias_value(normalized, "task_name", "unknown"), alias_value(normalized, "label_name", np.nan))]
    normalized["task_type"] = [task_type or infer_task_type(task) for task in normalized["task_name"]]
    normalized["model_name"] = alias_value(normalized, "model_name", family).astype(str)
    normalized["split_strategy"] = alias_value(normalized, "split_strategy", "unknown").map(normalize_split_strategy)
    normalized["split_id"] = alias_value(normalized, "split_id", "unknown").astype(str)
    normalized["fold_id"] = alias_value(normalized, "fold_id", "unknown").astype(str)
    normalized["ablation_name"] = [normalize_ablation_name(value, family) for value in alias_value(normalized, "ablation_name", np.nan)]
    return normalized


def normalize_metrics_per_fold(frame: pd.DataFrame, family: str, task_type: str) -> pd.DataFrame:
    if frame.empty:
        columns = ["model_family", "model_name", "ablation_name", "task_name", "task_type", "split_strategy", "split_id", "fold_id"] + (REGRESSION_METRICS if task_type == "regression" else CLASSIFICATION_METRICS)
        return pd.DataFrame(columns=columns)
    normalized = add_schema_defaults(frame, family, task_type=task_type)
    for metric in (REGRESSION_METRICS if task_type == "regression" else CLASSIFICATION_METRICS):
        column = metric_column(normalized, metric)
        normalized[metric] = safe_numeric(normalized[column]) if column else np.nan
    keep_columns = ["model_family", "model_name", "ablation_name", "task_name", "task_type", "split_strategy", "split_id", "fold_id"] + (REGRESSION_METRICS if task_type == "regression" else CLASSIFICATION_METRICS)
    if "evaluation_split" in normalized.columns:
        normalized = normalized[normalized["evaluation_split"].astype(str).str.lower().isin({"test", "holdout"}) | normalized["evaluation_split"].isna()].copy()
    return normalized[keep_columns].sort_values(["task_name", "split_strategy", "model_family", "model_name", "ablation_name", "split_id", "fold_id"], kind="mergesort").reset_index(drop=True)


def normalize_predictions(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    if frame.empty:
        columns = [
            "model_family", "model_name", "ablation_name", "task_name", "task_type", "split_strategy", "split_id", "fold_id",
            "row_uid", "compound_id", "standardized_smiles", "target_chembl_id", "secondary_target_identifier", "observed", "predicted",
        ]
        return pd.DataFrame(columns=columns)
    normalized = add_schema_defaults(frame, family)
    for column in [
        "row_uid", "compound_id", "standardized_smiles", "target_chembl_id", "secondary_target_identifier", "target_name",
        "kinase_family", "murcko_scaffold", "generic_murcko_scaffold", "source_id", "source_description", "source_frequency_bin", "activity_cliff_flag",
    ]:
        normalized[column] = alias_value(normalized, column, np.nan)
    observed_col = alias_lookup(normalized, "observed")
    predicted_col = alias_lookup(normalized, "predicted") or alias_lookup(normalized, "prediction_score")
    normalized["observed"] = safe_numeric(normalized[observed_col]) if observed_col else np.nan
    normalized["predicted"] = safe_numeric(normalized[predicted_col]) if predicted_col else np.nan
    if "evaluation_split" in normalized.columns:
        mask = normalized["evaluation_split"].astype(str).str.lower().isin({"test", "holdout"})
        if mask.any():
            normalized = normalized[mask].copy()
    return normalized.sort_values(["task_name", "split_strategy", "model_family", "model_name", "ablation_name", "split_id", "fold_id"], kind="mergesort").reset_index(drop=True)


def discover_first_existing(root: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def load_optional_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        logging.warning("Optional CSV %s is empty; treating it as missing.", path)
        return pd.DataFrame()


def read_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("Failed to parse JSON report %s: %s", path, exc)
        return None


def load_result_bundle(family: str, root: Path, expected_report_name: str) -> ResultBundle:
    ensure_exists(root, f"{family} results root")
    discovered_files: list[str] = []
    warnings: list[str] = []

    def load_named_csv(candidates: Sequence[str], required: bool = False) -> pd.DataFrame:
        path = discover_first_existing(root, candidates)
        if path is None:
            if required:
                raise FileNotFoundError(f"Required {family} result table missing under {root}. Expected one of: {', '.join(candidates)}")
            warnings.append(f"Missing optional table for {family}: {', '.join(candidates)}")
            return pd.DataFrame()
        discovered_files.append(str(path))
        logging.info("Loading %s table from %s", family, path)
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            if required:
                raise
            message = f"Optional table for {family} is empty and will be skipped: {path.name}"
            logging.warning(message)
            warnings.append(message)
            return pd.DataFrame()

    regression_per_fold = load_named_csv(["regression_metrics_per_fold.csv"], required=True)
    regression_summary = load_named_csv(["regression_metrics_summary.csv"], required=True)
    classification_per_fold = load_named_csv(["classification_metrics_per_fold.csv"], required=False)
    classification_summary = load_named_csv(["classification_metrics_summary.csv"], required=False)
    ablation_summary = load_named_csv(["ablation_metrics_summary.csv"], required=False)
    activity_cliff_metrics = load_named_csv(["activity_cliff_metrics.csv"], required=False)
    environment_group_metrics = load_named_csv(["environment_group_metrics.csv"], required=False)

    prediction_candidates = [
        root / "predictions" / "all_test_predictions.csv",
        root / "predictions" / "regression_predictions.csv",
        root / "predictions" / "classification_predictions.csv",
    ]
    prediction_frames: list[pd.DataFrame] = []
    seen_paths: set[str] = set()
    for path in prediction_candidates:
        if path.exists():
            discovered_files.append(str(path))
            seen_paths.add(str(path))
            prediction_frames.append(pd.read_csv(path))
    if not prediction_frames:
        for path in sorted((root / "predictions").rglob("*predictions.csv")) if (root / "predictions").exists() else []:
            if str(path) in seen_paths:
                continue
            discovered_files.append(str(path))
            prediction_frames.append(pd.read_csv(path))
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    project_root = root.parent.parent if root.parent.name == "results" else root.parent
    report_path = project_root / "reports" / expected_report_name
    if not report_path.exists():
        fallback = project_root / expected_report_name
        report_path = fallback if fallback.exists() else None
    if report_path is not None:
        discovered_files.append(str(report_path))
    else:
        warnings.append(f"Missing report JSON for {family}: {expected_report_name}")

    return ResultBundle(
        family=family,
        root=root,
        report_path=report_path,
        regression_per_fold=regression_per_fold,
        regression_summary=regression_summary,
        classification_per_fold=classification_per_fold,
        classification_summary=classification_summary,
        ablation_summary=ablation_summary,
        activity_cliff_metrics=activity_cliff_metrics,
        environment_group_metrics=environment_group_metrics,
        predictions=predictions,
        discovered_files=sorted(set(discovered_files)),
        warnings=warnings,
    )


def prepare_annotation_table(cfg: AppConfig) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    ensure_exists(cfg.input_annotated_long_path, "annotated long table")
    annotated = pd.read_csv(cfg.input_annotated_long_path)
    normalized = pd.DataFrame(index=annotated.index)
    for canonical in ANNOTATION_COLUMNS:
        normalized[canonical] = alias_value(annotated, canonical, np.nan)
    normalized = normalized.drop_duplicates().reset_index(drop=True)

    optional_frames: list[pd.DataFrame] = [normalized]
    for extra_path, source_name in [
        (cfg.input_activity_cliff_path, "activity_cliff_annotations"),
        (cfg.input_kinase_env_path, "kinase_environment_annotations"),
        (cfg.input_compound_env_path, "compound_environment_annotations"),
    ]:
        if extra_path and extra_path.exists():
            frame = pd.read_csv(extra_path)
            subset = pd.DataFrame(index=frame.index)
            for canonical in ANNOTATION_COLUMNS:
                subset[canonical] = alias_value(frame, canonical, np.nan)
            subset["annotation_source"] = source_name
            optional_frames.append(subset.drop_duplicates())
        elif extra_path:
            warnings.append(f"Optional annotation file unavailable: {extra_path}")
    merged = pd.concat(optional_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    return merged, warnings


def harmonize_merge_key_types(left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coerce merge keys to a shared nullable/string-friendly dtype.

    Prediction and annotation tables are assembled from heterogeneous CSV files,
    so identifier columns such as ``row_uid`` may arrive as ``object`` in one
    frame and ``float64`` in another when an upstream source contains missing
    values. Pandas refuses to merge those mixed dtypes directly. We normalize
    each participating key column to the pandas ``string`` dtype while
    preserving missing values as ``<NA>``.
    """

    def normalize_identifier(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            return str(int(value)) if float(value).is_integer() else str(value)
        text = str(value).strip()
        try:
            numeric = float(text)
        except ValueError:
            return text
        return str(int(numeric)) if numeric.is_integer() else text

    left_aligned = left.copy()
    right_aligned = right.copy()
    for key in keys:
        left_aligned[key] = left_aligned[key].map(normalize_identifier).astype("string")
        right_aligned[key] = right_aligned[key].map(normalize_identifier).astype("string")
    return left_aligned, right_aligned


def enrich_predictions(predictions: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    result = predictions.copy()
    merge_attempts = [
        ["row_uid"],
        ["compound_id", "target_chembl_id"],
        ["standardized_smiles", "target_chembl_id"],
        ["compound_id"],
        ["standardized_smiles"],
    ]
    remaining_annotation_cols = [col for col in annotations.columns if col not in result.columns]
    if not remaining_annotation_cols:
        return result
    for keys in merge_attempts:
        if all(key in result.columns for key in keys) and all(key in annotations.columns for key in keys):
            subset_cols = keys + [col for col in remaining_annotation_cols if col in annotations.columns]
            subset = annotations[subset_cols].dropna(how="all", subset=[col for col in subset_cols if col not in keys]).drop_duplicates(subset=keys)
            result, subset = harmonize_merge_key_types(result, subset, keys)
            result = result.merge(subset, on=keys, how="left")
            remaining_annotation_cols = [col for col in annotations.columns if col not in result.columns]
            if not remaining_annotation_cols:
                break
    if "absolute_error" not in result.columns:
        result["absolute_error"] = np.where(
            result["observed"].notna() & result["predicted"].notna(),
            np.abs(result["observed"] - result["predicted"]),
            np.nan,
        )
    if "squared_error" not in result.columns:
        result["squared_error"] = np.where(
            result["observed"].notna() & result["predicted"].notna(),
            np.square(result["observed"] - result["predicted"]),
            np.nan,
        )
    return result


def select_best_rows(summary_df: pd.DataFrame, metric: str, group_columns: Sequence[str]) -> pd.DataFrame:
    if summary_df.empty or f"{metric}_mean" not in summary_df.columns:
        return pd.DataFrame(columns=list(group_columns) + summary_df.columns.tolist())
    ascending = metric in LOWER_IS_BETTER
    rows: list[pd.Series] = []
    for _, frame in summary_df.groupby(list(group_columns), dropna=False, sort=True):
        sorted_frame = frame.sort_values([f"{metric}_mean", "model_family", "model_name", "ablation_name"], ascending=[ascending, True, True, True], kind="mergesort")
        rows.append(sorted_frame.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def calculate_relative_improvement(candidate: float, baseline: float, metric: str) -> float:
    if not np.isfinite(candidate) or not np.isfinite(baseline) or baseline == 0:
        return float("nan")
    if metric in LOWER_IS_BETTER:
        return float((baseline - candidate) / abs(baseline) * 100.0)
    return float((candidate - baseline) / abs(baseline) * 100.0)


def rank_models(summary_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary_df.empty or f"{metric}_mean" not in summary_df.columns:
        return pd.DataFrame()
    ascending = metric in LOWER_IS_BETTER
    output = summary_df.copy()
    output["rank_within_task_split"] = output.groupby(["task_name", "split_strategy"], dropna=False)[f"{metric}_mean"].rank(method="dense", ascending=ascending)
    return output.sort_values(["task_name", "split_strategy", "rank_within_task_split", "model_family", "model_name"], kind="mergesort").reset_index(drop=True)


def compare_causal_vs_baseline(summary_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary_df.empty or f"{metric}_mean" not in summary_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (task_name, split_strategy), frame in summary_df.groupby(["task_name", "split_strategy"], dropna=False, sort=True):
        causal_frame = frame[frame["model_family"] == "causal"].copy()
        baseline_frame = frame[frame["model_family"].isin(["classical", "deep"])].copy()
        if causal_frame.empty or baseline_frame.empty:
            continue
        causal_best = select_best_rows(causal_frame, metric, ["task_name", "split_strategy"])
        baseline_best = select_best_rows(baseline_frame, metric, ["task_name", "split_strategy"])
        if causal_best.empty or baseline_best.empty:
            continue
        c = causal_best.iloc[0]
        b = baseline_best.iloc[0]
        metric_diff = float(c[f"{metric}_mean"] - b[f"{metric}_mean"])
        rows.append({
            "task_name": task_name,
            "split_strategy": split_strategy,
            "selection_metric": metric,
            "causal_model_name": c["model_name"],
            "causal_ablation_name": c["ablation_name"],
            "causal_metric_mean": float(c[f"{metric}_mean"]),
            "baseline_family": b["model_family"],
            "baseline_model_name": b["model_name"],
            "baseline_metric_mean": float(b[f"{metric}_mean"]),
            "metric_difference": metric_diff,
            "relative_improvement_percent": calculate_relative_improvement(float(c[f"{metric}_mean"]), float(b[f"{metric}_mean"]), metric),
            "causal_better": bool(metric_diff < 0) if metric in LOWER_IS_BETTER else bool(metric_diff > 0),
        })
    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        return pd.DataFrame(columns=[
            "task_name",
            "split_strategy",
            "selection_metric",
            "causal_model_name",
            "causal_ablation_name",
            "causal_metric_mean",
            "baseline_family",
            "baseline_model_name",
            "baseline_metric_mean",
            "metric_difference",
            "relative_improvement_percent",
            "causal_better",
        ])
    return comparison_df.sort_values(["task_name", "split_strategy"], kind="mergesort").reset_index(drop=True)


def summarize_split_degradation(summary_df: pd.DataFrame, metric: str, reference_split: str = "random") -> pd.DataFrame:
    if summary_df.empty or f"{metric}_mean" not in summary_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    index_columns = ["model_family", "model_name", "ablation_name", "task_name"]
    for keys, frame in summary_df.groupby(index_columns, dropna=False, sort=True):
        ref = frame[frame["split_strategy"] == reference_split]
        if ref.empty:
            continue
        ref_value = float(ref.iloc[0][f"{metric}_mean"])
        for _, row in frame.iterrows():
            if row["split_strategy"] == reference_split:
                continue
            current = float(row[f"{metric}_mean"])
            degradation = current - ref_value
            rows.append({
                **{column: value for column, value in zip(index_columns, keys if isinstance(keys, tuple) else (keys,))},
                "reference_split_strategy": reference_split,
                "comparison_split_strategy": row["split_strategy"],
                "selection_metric": metric,
                "reference_metric_mean": ref_value,
                "comparison_metric_mean": current,
                "absolute_gap": degradation,
                "relative_gap_percent": calculate_relative_improvement(current, ref_value, metric),
            })
    return pd.DataFrame(rows).sort_values(index_columns + ["comparison_split_strategy"], kind="mergesort").reset_index(drop=True)


def summarize_activity_cliff(predictions: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty or "activity_cliff_flag" not in predictions.columns:
        return pd.DataFrame(), pd.DataFrame()
    working = predictions[predictions["task_type"] == "regression"].copy()
    working = working[working["activity_cliff_flag"].notna()].copy()
    if working.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, frame in working.groupby(["model_family", "model_name", "ablation_name", "task_name", "split_strategy", "activity_cliff_flag"], dropna=False, sort=True):
        metrics = compute_metric_frame(frame["observed"], frame["predicted"], "regression")
        rows.append({
            "model_family": keys[0],
            "model_name": keys[1],
            "ablation_name": keys[2],
            "task_name": keys[3],
            "split_strategy": keys[4],
            "activity_cliff_flag": keys[5],
            "n_rows": int(len(frame)),
            **metrics,
        })
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    comparison = pd.DataFrame(rows).sort_values(["task_name", "split_strategy", "model_family", "model_name", "activity_cliff_flag"], kind="mergesort").reset_index(drop=True)
    degradation_rows: list[dict[str, Any]] = []
    for keys, frame in comparison.groupby(["model_family", "model_name", "ablation_name", "task_name", "split_strategy"], dropna=False, sort=True):
        cliff = frame[frame["activity_cliff_flag"].astype(str).isin(["1", "True", "true", "cliff", "yes"])]
        non_cliff = frame[~frame.index.isin(cliff.index)]
        if cliff.empty or non_cliff.empty or f"{metric}" not in cliff.columns or f"{metric}" not in non_cliff.columns:
            continue
        cliff_value = float(cliff.iloc[0][metric])
        non_cliff_value = float(non_cliff.iloc[0][metric])
        degradation_rows.append({
            "model_family": keys[0],
            "model_name": keys[1],
            "ablation_name": keys[2],
            "task_name": keys[3],
            "split_strategy": keys[4],
            "selection_metric": metric,
            "cliff_metric": cliff_value,
            "non_cliff_metric": non_cliff_value,
            "cliff_degradation": cliff_value - non_cliff_value,
            "relative_cliff_degradation_percent": calculate_relative_improvement(cliff_value, non_cliff_value, metric),
        })
    degradation = (
        pd.DataFrame(degradation_rows).sort_values(
            ["task_name", "split_strategy", "model_family", "model_name"],
            kind="mergesort",
        ).reset_index(drop=True)
        if degradation_rows
        else pd.DataFrame()
    )
    return comparison, degradation


def summarize_environment_groups(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    regression = predictions[predictions["task_type"] == "regression"].copy()
    if regression.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    group_specs = {
        "scaffold_group": "murcko_scaffold",
        "generic_scaffold_group": "generic_murcko_scaffold",
        "kinase_family_group": "kinase_family",
        "source_environment_group": "source_description",
        "source_frequency_bin_group": "source_frequency_bin",
    }
    rows: list[dict[str, Any]] = []
    for group_name, column in group_specs.items():
        if column not in regression.columns:
            continue
        subset = regression[regression[column].notna()].copy()
        if subset.empty:
            continue
        for keys, frame in subset.groupby(["model_family", "model_name", "ablation_name", "task_name", "split_strategy", column], dropna=False, sort=True):
            metrics = compute_metric_frame(frame["observed"], frame["predicted"], "regression")
            rows.append({
                "group_type": group_name,
                "group_value": keys[-1],
                "model_family": keys[0],
                "model_name": keys[1],
                "ablation_name": keys[2],
                "task_name": keys[3],
                "split_strategy": keys[4],
                "n_rows": int(len(frame)),
                **metrics,
            })
    env_df = pd.DataFrame(rows).sort_values(["group_type", "task_name", "split_strategy", "model_family", "model_name", "group_value"], kind="mergesort").reset_index(drop=True) if rows else pd.DataFrame()
    hardest_scaffolds = env_df[env_df["group_type"].isin(["scaffold_group", "generic_scaffold_group"])] if not env_df.empty else pd.DataFrame()
    hardest_scaffolds = hardest_scaffolds.sort_values(["rmse", "n_rows"], ascending=[False, False], kind="mergesort").head(50) if not hardest_scaffolds.empty and "rmse" in hardest_scaffolds.columns else pd.DataFrame()
    hardest_kinase_families = env_df[env_df["group_type"] == "kinase_family_group"] if not env_df.empty else pd.DataFrame()
    hardest_kinase_families = hardest_kinase_families.sort_values(["rmse", "n_rows"], ascending=[False, False], kind="mergesort").head(50) if not hardest_kinase_families.empty and "rmse" in hardest_kinase_families.columns else pd.DataFrame()
    source_summary = env_df[env_df["group_type"].isin(["source_environment_group", "source_frequency_bin_group"])] if not env_df.empty else pd.DataFrame()
    return env_df, hardest_scaffolds, hardest_kinase_families, source_summary


def summarize_interpretation_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    regression = predictions[predictions["task_type"] == "regression"].copy()
    if regression.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    per_kinase = pd.DataFrame()
    if "target_chembl_id" in regression.columns:
        kinase_rows: list[dict[str, Any]] = []
        subset = regression[regression["target_chembl_id"].notna()].copy()
        for keys, frame in subset.groupby(["model_family", "model_name", "ablation_name", "task_name", "split_strategy", "target_chembl_id"], dropna=False, sort=True):
            metrics = compute_metric_frame(frame["observed"], frame["predicted"], "regression")
            kinase_rows.append({
                "model_family": keys[0],
                "model_name": keys[1],
                "ablation_name": keys[2],
                "task_name": keys[3],
                "split_strategy": keys[4],
                "target_chembl_id": keys[5],
                "n_rows": int(len(frame)),
                **metrics,
            })
        per_kinase = pd.DataFrame(kinase_rows).sort_values(["task_name", "split_strategy", "rmse"], ascending=[True, True, False], kind="mergesort").reset_index(drop=True) if kinase_rows else pd.DataFrame()

    hardest_kinases = per_kinase.sort_values(["rmse", "n_rows"], ascending=[False, False], kind="mergesort").head(50) if not per_kinase.empty and "rmse" in per_kinase.columns else pd.DataFrame()
    compound_group_cols = [col for col in ["compound_id", "standardized_smiles"] if col in regression.columns]
    if compound_group_cols:
        compound_key = compound_group_cols[0]
        hardest_compounds = regression.groupby(compound_key, dropna=False)["absolute_error"].agg(["mean", "count"]).reset_index().rename(columns={compound_key: "compound_key", "mean": "mean_absolute_error", "count": "n_rows"})
        hardest_compounds = hardest_compounds.sort_values(["mean_absolute_error", "n_rows"], ascending=[False, False], kind="mergesort").head(100).reset_index(drop=True)
    else:
        hardest_compounds = pd.DataFrame()
    scaffold_error_summary = regression.groupby([col for col in ["murcko_scaffold"] if col in regression.columns], dropna=False)["absolute_error"].agg(["mean", "count"]).reset_index() if "murcko_scaffold" in regression.columns else pd.DataFrame()
    if not scaffold_error_summary.empty:
        scaffold_error_summary = scaffold_error_summary.rename(columns={"mean": "mean_absolute_error", "count": "n_rows"}).sort_values(["mean_absolute_error", "n_rows"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    best_predicted_kinases = per_kinase.sort_values(["rmse", "n_rows"], ascending=[True, False], kind="mergesort").head(50) if not per_kinase.empty and "rmse" in per_kinase.columns else pd.DataFrame()
    return per_kinase, hardest_kinases, hardest_compounds, scaffold_error_summary, best_predicted_kinases


def summarize_ablations(summary_df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    causal = summary_df[summary_df["model_family"] == "causal"].copy()
    if causal.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (task_name, split_strategy, model_name), frame in causal.groupby(["task_name", "split_strategy", "model_name"], dropna=False, sort=True):
        main_rows = frame[frame["ablation_name"] == "main"]
        if main_rows.empty:
            continue
        main_value = float(main_rows.iloc[0][f"{metric}_mean"]) if f"{metric}_mean" in main_rows.columns else float("nan")
        for _, row in frame.iterrows():
            if row["ablation_name"] == "main":
                continue
            current = float(row.get(f"{metric}_mean", np.nan))
            rows.append({
                "task_name": task_name,
                "split_strategy": split_strategy,
                "model_name": model_name,
                "ablation_name": row["ablation_name"],
                "selection_metric": metric,
                "full_model_metric_mean": main_value,
                "ablation_metric_mean": current,
                "performance_drop": current - main_value,
                "relative_drop_percent": calculate_relative_improvement(current, main_value, metric),
            })
    drop_df = pd.DataFrame(rows).sort_values(["task_name", "split_strategy", "performance_drop"], ascending=[True, True, False], kind="mergesort").reset_index(drop=True) if rows else pd.DataFrame()
    rank_df = causal.copy()
    if not rank_df.empty and f"{metric}_mean" in rank_df.columns:
        rank_df["ablation_rank"] = rank_df.groupby(["task_name", "split_strategy", "model_name"], dropna=False)[f"{metric}_mean"].rank(method="dense", ascending=(metric in LOWER_IS_BETTER))
        rank_df = rank_df.sort_values(["task_name", "split_strategy", "model_name", "ablation_rank", "ablation_name"], kind="mergesort").reset_index(drop=True)
    return drop_df, rank_df


def summarize_low_data(summary_df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    subset = summary_df[summary_df["split_strategy"].astype(str).str.contains("low", case=False, na=False)].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()
    def extract_size(text: Any) -> float:
        import re
        match = re.search(r"(\d+)", str(text))
        return float(match.group(1)) if match else float("nan")
    subset["train_size"] = subset["split_strategy"].map(extract_size)
    subset = subset.sort_values(["task_name", "train_size", "model_family", "model_name"], kind="mergesort").reset_index(drop=True)
    learning_curve = subset[["task_name", "split_strategy", "train_size", "model_family", "model_name", "ablation_name", f"{metric}_mean"]].copy() if f"{metric}_mean" in subset.columns else pd.DataFrame()
    return subset, learning_curve


def summarize_transfer_gap(summary_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    challenging = summary_df[summary_df["split_strategy"].isin(["kinase_family_grouped", "source_environment_grouped", "scaffold"])]
    return summarize_split_degradation(pd.concat([summary_df[summary_df["split_strategy"] == "random"], challenging], ignore_index=True), metric, reference_split="random")


def paired_statistical_comparison(unified_per_fold: pd.DataFrame, metric: str) -> pd.DataFrame:
    if unified_per_fold.empty or f"{metric}" not in unified_per_fold.columns:
        return pd.DataFrame()
    try:
        from scipy import stats as scipy_stats  # type: ignore
    except Exception:
        logging.warning("SciPy unavailable; skipping statistical comparison support.")
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    pairing_cols = ["task_name", "split_strategy", "split_id", "fold_id"]
    for (task_name, split_strategy), frame in unified_per_fold.groupby(["task_name", "split_strategy"], dropna=False, sort=True):
        pivot = frame.pivot_table(index=pairing_cols, columns=["model_family", "model_name", "ablation_name"], values=metric, aggfunc="first")
        if pivot.shape[1] < 2:
            continue
        columns = list(pivot.columns)
        for idx, first in enumerate(columns):
            for second in columns[idx + 1:]:
                paired = pivot[[first, second]].dropna()
                if len(paired) < 2:
                    continue
                try:
                    stat, pvalue = scipy_stats.ttest_rel(paired[first], paired[second], nan_policy="omit")
                except Exception:
                    stat, pvalue = (float("nan"), float("nan"))
                rows.append({
                    "task_name": task_name,
                    "split_strategy": split_strategy,
                    "metric": metric,
                    "model_a_family": first[0],
                    "model_a_name": first[1],
                    "model_a_ablation": first[2],
                    "model_b_family": second[0],
                    "model_b_name": second[1],
                    "model_b_ablation": second[2],
                    "n_pairs": int(len(paired)),
                    "paired_t_statistic": float(stat),
                    "paired_t_pvalue": float(pvalue),
                    "mean_difference_a_minus_b": float((paired[first] - paired[second]).mean()),
                    "note": "Supportive paired t-test on fold-aligned metrics only; not a sole basis for claims.",
                })
    return pd.DataFrame(rows).sort_values(["task_name", "split_strategy", "metric", "model_a_family", "model_b_family"], kind="mergesort").reset_index(drop=True) if rows else pd.DataFrame()


def configure_matplotlib(cfg: AppConfig) -> tuple[Any | None, Any | None]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.family"] = cfg.figure_style.font_family
        plt.rcParams["font.weight"] = "bold" if cfg.figure_style.bold_text else "normal"
        plt.rcParams["axes.labelweight"] = "bold" if cfg.figure_style.bold_text else "normal"
        plt.rcParams["axes.titleweight"] = "bold" if cfg.figure_style.bold_text else "normal"
        plt.rcParams["svg.fonttype"] = "none"
        return matplotlib, plt
    except Exception as exc:
        logging.warning("Matplotlib unavailable; skipping figure generation: %s", exc)
        return None, None


def export_figure(fig: Any, output_stem: Path, cfg: AppConfig) -> dict[str, str]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    if cfg.export_svg:
        path = output_stem.with_suffix(".svg")
        fig.savefig(path, bbox_inches="tight")
        outputs["svg"] = str(path)
    if cfg.export_png:
        path = output_stem.with_suffix(".png")
        fig.savefig(path, bbox_inches="tight", dpi=cfg.figure_style.dpi_png)
        outputs["png"] = str(path)
    if cfg.export_pdf:
        path = output_stem.with_suffix(".pdf")
        fig.savefig(path, bbox_inches="tight")
        outputs["pdf"] = str(path)
    logging.info("Exported figure variants for %s", output_stem)
    return outputs


def save_figure_source_data(df: pd.DataFrame, path: Path, cfg: AppConfig) -> None:
    if cfg.save_source_data_for_figures:
        write_csv(df, path)


def plot_bar_summary(df: pd.DataFrame, value_col: str, title: str, ylabel: str, hue_col: str, output_stem: Path, cfg: AppConfig, plt: Any) -> dict[str, str] | None:
    if df.empty or value_col not in df.columns:
        return None
    working = df.copy().sort_values(["task_name", value_col], kind="mergesort")
    split_label_col = "split_strategy" if "split_strategy" in working.columns else hue_col if hue_col in working.columns else None
    if split_label_col:
        x_labels = (working["task_name"].astype(str) + "\n" + working[split_label_col].astype(str)).tolist()
    else:
        x_labels = working["task_name"].astype(str).tolist()
    categories = list(dict.fromkeys(working[hue_col].astype(str).tolist()))
    positions = np.arange(len(working))
    fig, ax = plt.subplots(figsize=(max(10, len(working) * 0.55), 6))
    colors = {name: NATURE_PALETTE[idx % len(NATURE_PALETTE)] for idx, name in enumerate(categories)}
    ax.bar(positions, working[value_col].to_numpy(dtype=float), color=[colors[name] for name in working[hue_col].astype(str)])
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontweight="bold" if cfg.figure_style.bold_text else None)
    ax.set_ylabel(ylabel, fontweight="bold" if cfg.figure_style.bold_text else None)
    ax.set_title(title, fontweight="bold" if cfg.figure_style.bold_text else None)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[name]) for name in categories]
    ax.legend(handles, categories, title=hue_col.replace("_", " "))
    fig.tight_layout()
    outputs = export_figure(fig, output_stem, cfg)
    plt.close(fig)
    return outputs


def plot_learning_curve(df: pd.DataFrame, value_col: str, output_stem: Path, cfg: AppConfig, plt: Any) -> dict[str, str] | None:
    if df.empty or "train_size" not in df.columns or value_col not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (family, frame) in enumerate(df.groupby("model_family", dropna=False, sort=True)):
        ordered = frame.sort_values("train_size", kind="mergesort")
        ax.plot(ordered["train_size"], ordered[value_col], marker="o", linewidth=2.0, color=NATURE_PALETTE[idx % len(NATURE_PALETTE)], label=str(family))
    ax.set_xlabel("Training subset size", fontweight="bold" if cfg.figure_style.bold_text else None)
    ax.set_ylabel(value_col.replace("_", " ").upper(), fontweight="bold" if cfg.figure_style.bold_text else None)
    ax.set_title("Low-data learning curve comparison", fontweight="bold" if cfg.figure_style.bold_text else None)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    outputs = export_figure(fig, output_stem, cfg)
    plt.close(fig)
    return outputs


def plot_heatmap(rank_df: pd.DataFrame, output_stem: Path, cfg: AppConfig, plt: Any) -> dict[str, str] | None:
    if rank_df.empty or "rank_within_task_split" not in rank_df.columns:
        return None
    pivot = rank_df.pivot_table(index="task_name", columns=["model_family", "model_name", "ablation_name", "split_strategy"], values="rank_within_task_split", aggfunc="first")
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.6), max(4, pivot.shape[0] * 0.6)))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis_r")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(item) for item in pivot.index], fontweight="bold" if cfg.figure_style.bold_text else None)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(["|".join(map(str, col)) for col in pivot.columns], rotation=90, fontsize=8)
    ax.set_title("Model ranking heatmap across tasks and split strategies", fontweight="bold" if cfg.figure_style.bold_text else None)
    fig.colorbar(image, ax=ax, label="Rank (lower is better)")
    fig.tight_layout()
    outputs = export_figure(fig, output_stem, cfg)
    plt.close(fig)
    return outputs


def generate_figures(
    cfg: AppConfig,
    unified_regression_summary: pd.DataFrame,
    unified_classification_summary: pd.DataFrame,
    best_causal_vs_best_baseline: pd.DataFrame,
    ablation_drop_summary: pd.DataFrame,
    split_degradation: pd.DataFrame,
    low_data_learning_curve: pd.DataFrame,
    activity_cliff_comparison: pd.DataFrame,
    environment_group_metrics: pd.DataFrame,
    per_kinase_summary: pd.DataFrame,
    rank_table: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    figure_outputs: dict[str, dict[str, str]] = {}
    if not cfg.make_figures:
        return figure_outputs
    _, plt = configure_matplotlib(cfg)
    if plt is None:
        return figure_outputs

    source_root = cfg.output_results_root / "figure_source_data"

    maybe = plot_bar_summary(unified_regression_summary, f"{cfg.best_model_selection_metric_regression}_mean", "Overall model-family comparison across regression tasks", cfg.best_model_selection_metric_regression.upper(), "model_family", cfg.output_figures_root / "overall_model_family_regression_comparison", cfg, plt)
    if maybe:
        figure_outputs["overall_model_family_regression_comparison"] = maybe
        save_figure_source_data(unified_regression_summary, source_root / "overall_model_family_regression_comparison.csv", cfg)

    if not unified_regression_summary.empty:
        maybe = plot_bar_summary(unified_regression_summary, f"{cfg.best_model_selection_metric_regression}_mean", "Regression performance across split strategies", cfg.best_model_selection_metric_regression.upper(), "split_strategy", cfg.output_figures_root / "regression_performance_by_split_strategy", cfg, plt)
        if maybe:
            figure_outputs["regression_performance_by_split_strategy"] = maybe
            save_figure_source_data(unified_regression_summary, source_root / "regression_performance_by_split_strategy.csv", cfg)

    if not unified_classification_summary.empty:
        maybe = plot_bar_summary(unified_classification_summary, f"{cfg.best_model_selection_metric_classification}_mean", "Classification performance across split strategies", cfg.best_model_selection_metric_classification.upper(), "split_strategy", cfg.output_figures_root / "classification_performance_by_split_strategy", cfg, plt)
        if maybe:
            figure_outputs["classification_performance_by_split_strategy"] = maybe
            save_figure_source_data(unified_classification_summary, source_root / "classification_performance_by_split_strategy.csv", cfg)

    if not best_causal_vs_best_baseline.empty:
        maybe = plot_bar_summary(best_causal_vs_best_baseline, "relative_improvement_percent", "Causal model vs best baseline", "Relative improvement (%)", "baseline_family", cfg.output_figures_root / "causal_vs_best_baseline", cfg, plt)
        if maybe:
            figure_outputs["causal_vs_best_baseline"] = maybe
            save_figure_source_data(best_causal_vs_best_baseline, source_root / "causal_vs_best_baseline.csv", cfg)

    if not ablation_drop_summary.empty:
        maybe = plot_bar_summary(ablation_drop_summary, "performance_drop", "Causal ablation comparison", "Performance drop", "ablation_name", cfg.output_figures_root / "causal_ablation_comparison", cfg, plt)
        if maybe:
            figure_outputs["causal_ablation_comparison"] = maybe
            save_figure_source_data(ablation_drop_summary, source_root / "causal_ablation_comparison.csv", cfg)

    if not split_degradation.empty:
        maybe = plot_bar_summary(split_degradation, "absolute_gap", "Split robustness degradation from random split", "Absolute gap", "comparison_split_strategy", cfg.output_figures_root / "split_robustness_degradation", cfg, plt)
        if maybe:
            figure_outputs["split_robustness_degradation"] = maybe
            save_figure_source_data(split_degradation, source_root / "split_robustness_degradation.csv", cfg)

    if not low_data_learning_curve.empty:
        maybe = plot_learning_curve(low_data_learning_curve, f"{cfg.best_model_selection_metric_regression}_mean", cfg.output_figures_root / "low_data_learning_curve_comparison", cfg, plt)
        if maybe:
            figure_outputs["low_data_learning_curve_comparison"] = maybe
            save_figure_source_data(low_data_learning_curve, source_root / "low_data_learning_curve_comparison.csv", cfg)

    if not activity_cliff_comparison.empty:
        maybe = plot_bar_summary(activity_cliff_comparison, "rmse", "Activity-cliff performance comparison", "RMSE", "activity_cliff_flag", cfg.output_figures_root / "activity_cliff_performance_comparison", cfg, plt)
        if maybe:
            figure_outputs["activity_cliff_performance_comparison"] = maybe
            save_figure_source_data(activity_cliff_comparison, source_root / "activity_cliff_performance_comparison.csv", cfg)

    if not environment_group_metrics.empty:
        plot_df = environment_group_metrics.groupby(["group_type", "model_family"], dropna=False)["rmse"].mean().reset_index() if "rmse" in environment_group_metrics.columns else pd.DataFrame()
        maybe = plot_bar_summary(plot_df.rename(columns={"group_type": "task_name", "model_family": "model_family", "rmse": "rmse"}).assign(split_strategy="environment_groups"), "rmse", "Environment-group error comparison", "RMSE", "model_family", cfg.output_figures_root / "environment_group_error_comparison", cfg, plt) if not plot_df.empty else None
        if maybe:
            figure_outputs["environment_group_error_comparison"] = maybe
            save_figure_source_data(environment_group_metrics, source_root / "environment_group_error_comparison.csv", cfg)

    if not per_kinase_summary.empty:
        plot_df = per_kinase_summary.groupby("model_family", dropna=False)["rmse"].describe().reset_index() if "rmse" in per_kinase_summary.columns else pd.DataFrame()
        if not plot_df.empty:
            plot_df = plot_df.rename(columns={"50%": "median_rmse"}).assign(task_name="per_kinase_distribution", split_strategy="all", model_family=plot_df["model_family"]) 
            maybe = plot_bar_summary(plot_df.rename(columns={"median_rmse": "rmse"}), "rmse", "Per-kinase top-model performance distribution", "Median RMSE", "model_family", cfg.output_figures_root / "per_kinase_top_model_distribution", cfg, plt)
            if maybe:
                figure_outputs["per_kinase_top_model_distribution"] = maybe
                save_figure_source_data(per_kinase_summary, source_root / "per_kinase_top_model_distribution.csv", cfg)

    maybe = plot_heatmap(rank_table, cfg.output_figures_root / "model_ranking_heatmap", cfg, plt)
    if maybe:
        figure_outputs["model_ranking_heatmap"] = maybe
        save_figure_source_data(rank_table, source_root / "model_ranking_heatmap.csv", cfg)
    return figure_outputs


def build_unified_ablation_summary(causal_ablation_summary: pd.DataFrame) -> pd.DataFrame:
    if causal_ablation_summary.empty:
        return pd.DataFrame(columns=["model_family", "model_name", "ablation_name", "task_name", "split_strategy"])
    summary = causal_ablation_summary.copy()
    summary["model_family"] = "causal"
    summary["ablation_name"] = [normalize_ablation_name(value, "causal") for value in alias_value(summary, "ablation_name", "main")]
    summary["task_name"] = [normalize_task_name(value) for value in alias_value(summary, "task_name", "unknown")]
    summary["model_name"] = alias_value(summary, "model_name", "causal_gnn")
    summary["split_strategy"] = alias_value(summary, "split_strategy", "all").map(normalize_split_strategy)
    return summary.sort_values(["task_name", "split_strategy", "ablation_name"], kind="mergesort").reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    raw_config = load_yaml(config_path)
    cfg = AppConfig.from_dict(raw_config, project_root)
    log_file = configure_logging(cfg)
    set_global_determinism(DEFAULT_SEED)
    config_snapshot_path = save_config_snapshot(config_path, raw_config, cfg)

    logging.info("Starting %s", SCRIPT_NAME)
    ensure_exists(cfg.input_split_manifest_path, "split manifest")
    split_manifest = pd.read_csv(cfg.input_split_manifest_path)
    if "split_strategy" not in split_manifest.columns:
        raise ValueError(f"Split manifest missing required `split_strategy` column: {cfg.input_split_manifest_path}")

    bundles = [
        load_result_bundle("classical", cfg.input_classical_results_root, "07_classical_baseline_report.json"),
        load_result_bundle("deep", cfg.input_deep_results_root, "08_deep_baseline_report.json"),
        load_result_bundle("causal", cfg.input_causal_results_root, "09_causal_model_report.json"),
    ]
    warnings_list = sorted({warning for bundle in bundles for warning in bundle.warnings})

    annotations, annotation_warnings = prepare_annotation_table(cfg)
    warnings_list.extend(annotation_warnings)

    regression_per_fold_all = pd.concat([normalize_metrics_per_fold(bundle.regression_per_fold, bundle.family, "regression") for bundle in bundles], ignore_index=True)
    classification_frames = [frame for frame in (normalize_metrics_per_fold(bundle.classification_per_fold, bundle.family, "classification") for bundle in bundles) if not frame.empty]
    classification_per_fold_all = pd.concat(classification_frames, ignore_index=True) if classification_frames else normalize_metrics_per_fold(pd.DataFrame(), "unknown", "classification")
    regression_predictions_all = pd.concat([normalize_predictions(bundle.predictions, bundle.family) for bundle in bundles], ignore_index=True)
    regression_predictions_all = enrich_predictions(regression_predictions_all, annotations)

    unified_regression_summary = aggregate_metrics(
        regression_per_fold_all,
        ["model_family", "model_name", "ablation_name", "task_name", "split_strategy"],
        REGRESSION_METRICS,
    )
    unified_classification_summary = aggregate_metrics(
        classification_per_fold_all,
        ["model_family", "model_name", "ablation_name", "task_name", "split_strategy"],
        CLASSIFICATION_METRICS,
    )
    unified_ablation_summary = build_unified_ablation_summary(bundles[2].ablation_summary)

    best_models_by_task = pd.concat(
        [
            select_best_rows(unified_regression_summary, cfg.best_model_selection_metric_regression, ["task_name", "model_family"]),
            select_best_rows(unified_classification_summary, cfg.best_model_selection_metric_classification, ["task_name", "model_family"]),
        ],
        ignore_index=True,
    )
    best_models_by_split_strategy = pd.concat(
        [
            select_best_rows(unified_regression_summary, cfg.best_model_selection_metric_regression, ["split_strategy", "model_family"]),
            select_best_rows(unified_classification_summary, cfg.best_model_selection_metric_classification, ["split_strategy", "model_family"]),
        ],
        ignore_index=True,
    )

    best_causal_vs_best_baseline_reg = compare_causal_vs_baseline(unified_regression_summary, cfg.best_model_selection_metric_regression)
    best_causal_vs_best_baseline_cls = compare_causal_vs_baseline(unified_classification_summary, cfg.best_model_selection_metric_classification)
    best_causal_vs_best_baseline = pd.concat([best_causal_vs_best_baseline_reg, best_causal_vs_best_baseline_cls], ignore_index=True)

    regression_rank_table = rank_models(unified_regression_summary, cfg.best_model_selection_metric_regression)
    classification_rank_table = rank_models(unified_classification_summary, cfg.best_model_selection_metric_classification)
    full_rank_table = pd.concat([regression_rank_table, classification_rank_table], ignore_index=True)

    split_degradation_reg = summarize_split_degradation(unified_regression_summary, cfg.best_model_selection_metric_regression)
    split_degradation_cls = summarize_split_degradation(unified_classification_summary, cfg.best_model_selection_metric_classification)
    transfer_gap_summary = pd.concat([
        summarize_transfer_gap(unified_regression_summary, cfg.best_model_selection_metric_regression),
        summarize_transfer_gap(unified_classification_summary, cfg.best_model_selection_metric_classification),
    ], ignore_index=True)

    activity_cliff_model_comparison, activity_cliff_degradation_summary = summarize_activity_cliff(regression_predictions_all, cfg.best_model_selection_metric_regression)
    environment_group_metrics, hardest_scaffold_groups, hardest_kinase_families, source_environment_robustness_summary = summarize_environment_groups(regression_predictions_all)
    per_kinase_summary, hardest_kinases, hardest_compounds, scaffold_error_summary, _best_predicted_kinases = summarize_interpretation_tables(regression_predictions_all)
    ablation_drop_summary, ablation_rank_summary = summarize_ablations(unified_regression_summary, cfg.best_model_selection_metric_regression)
    low_data_performance_summary, low_data_learning_curve_source_data = summarize_low_data(unified_regression_summary, cfg.best_model_selection_metric_regression)

    statistical_comparison_summary = pd.concat([
        paired_statistical_comparison(regression_per_fold_all, cfg.best_model_selection_metric_regression),
        paired_statistical_comparison(classification_per_fold_all, cfg.best_model_selection_metric_classification),
    ], ignore_index=True)

    cfg.output_results_root.mkdir(parents=True, exist_ok=True)
    write_csv(regression_per_fold_all, cfg.output_results_root / "unified_regression_metrics_per_fold.csv")
    write_csv(unified_regression_summary, cfg.output_results_root / "unified_regression_metrics_summary.csv")
    write_csv(classification_per_fold_all, cfg.output_results_root / "unified_classification_metrics_per_fold.csv")
    write_csv(unified_classification_summary, cfg.output_results_root / "unified_classification_metrics_summary.csv")
    write_csv(unified_ablation_summary, cfg.output_results_root / "unified_ablation_metrics_summary.csv")
    write_csv(best_models_by_task, cfg.output_results_root / "best_models_by_task.csv")
    write_csv(best_models_by_split_strategy, cfg.output_results_root / "best_models_by_split_strategy.csv")
    write_csv(best_causal_vs_best_baseline, cfg.output_results_root / "best_causal_vs_best_baseline.csv")
    write_csv(activity_cliff_model_comparison, cfg.output_results_root / "activity_cliff_model_comparison.csv")
    write_csv(activity_cliff_degradation_summary, cfg.output_results_root / "activity_cliff_degradation_summary.csv")
    write_csv(environment_group_metrics, cfg.output_results_root / "environment_group_metrics.csv")
    write_csv(hardest_scaffold_groups, cfg.output_results_root / "hardest_scaffold_groups.csv")
    write_csv(hardest_kinase_families, cfg.output_results_root / "hardest_kinase_families.csv")
    write_csv(per_kinase_summary, cfg.output_results_root / "per_kinase_performance_summary.csv")
    write_csv(hardest_kinases, cfg.output_results_root / "hardest_kinases.csv")
    write_csv(hardest_compounds, cfg.output_results_root / "hardest_compounds.csv")
    write_csv(scaffold_error_summary, cfg.output_results_root / "scaffold_error_summary.csv")
    write_csv(ablation_drop_summary, cfg.output_results_root / "ablation_drop_summary.csv")
    write_csv(ablation_rank_summary, cfg.output_results_root / "ablation_rank_summary.csv")
    write_csv(low_data_performance_summary, cfg.output_results_root / "low_data_performance_summary.csv")
    write_csv(low_data_learning_curve_source_data, cfg.output_results_root / "low_data_learning_curve_source_data.csv")
    write_csv(transfer_gap_summary, cfg.output_results_root / "transfer_gap_summary.csv")
    write_csv(statistical_comparison_summary, cfg.output_results_root / "statistical_comparison_summary.csv")
    write_csv(source_environment_robustness_summary, cfg.output_results_root / "source_environment_robustness_summary.csv")
    write_csv(split_degradation_reg, cfg.output_results_root / "regression_split_degradation_summary.csv")
    write_csv(split_degradation_cls, cfg.output_results_root / "classification_split_degradation_summary.csv")

    figure_outputs = generate_figures(
        cfg,
        unified_regression_summary,
        unified_classification_summary,
        best_causal_vs_best_baseline,
        ablation_drop_summary,
        split_degradation_reg if not split_degradation_reg.empty else split_degradation_cls,
        low_data_learning_curve_source_data,
        activity_cliff_model_comparison,
        environment_group_metrics,
        per_kinase_summary,
        full_rank_table,
    )

    reports = {bundle.family: read_report(bundle.report_path) for bundle in bundles}
    best_models_summary = {
        "regression": select_best_rows(unified_regression_summary, cfg.best_model_selection_metric_regression, ["task_name"]).to_dict(orient="records"),
        "classification": select_best_rows(unified_classification_summary, cfg.best_model_selection_metric_classification, ["task_name"]).to_dict(orient="records"),
    }
    report_payload = {
        "script_name": SCRIPT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "log_file": str(log_file),
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot_path) if config_snapshot_path else None,
        "input_result_roots_used": {
            "classical": str(cfg.input_classical_results_root),
            "deep": str(cfg.input_deep_results_root),
            "causal": str(cfg.input_causal_results_root),
        },
        "input_annotation_paths": {
            "annotated_long": str(cfg.input_annotated_long_path),
            "activity_cliff": str(cfg.input_activity_cliff_path) if cfg.input_activity_cliff_path else None,
            "kinase_environment": str(cfg.input_kinase_env_path) if cfg.input_kinase_env_path else None,
            "compound_environment": str(cfg.input_compound_env_path) if cfg.input_compound_env_path else None,
            "split_manifest": str(cfg.input_split_manifest_path),
        },
        "output_paths": {
            "results_root": str(cfg.output_results_root),
            "figures_root": str(cfg.output_figures_root),
            "report_path": str(cfg.output_report_path),
        },
        "tasks_compared": sorted(set(unified_regression_summary.get("task_name", pd.Series(dtype=str)).astype(str).tolist() + unified_classification_summary.get("task_name", pd.Series(dtype=str)).astype(str).tolist())),
        "model_families_compared": [bundle.family for bundle in bundles],
        "split_strategies_compared": sorted(set(unified_regression_summary.get("split_strategy", pd.Series(dtype=str)).astype(str).tolist() + unified_classification_summary.get("split_strategy", pd.Series(dtype=str)).astype(str).tolist())),
        "number_of_result_files_processed": int(sum(len(bundle.discovered_files) for bundle in bundles)),
        "processed_files_by_family": {bundle.family: bundle.discovered_files for bundle in bundles},
        "source_reports": reports,
        "summary_of_best_models": best_models_summary,
        "summary_of_causal_gains_or_losses_vs_baselines": best_causal_vs_best_baseline.to_dict(orient="records"),
        "ablation_conclusions": ablation_drop_summary.head(20).to_dict(orient="records"),
        "activity_cliff_conclusions": activity_cliff_degradation_summary.head(20).to_dict(orient="records"),
        "environment_robustness_conclusions": source_environment_robustness_summary.head(20).to_dict(orient="records"),
        "low_data_conclusions": low_data_performance_summary.head(20).to_dict(orient="records"),
        "figure_outputs": figure_outputs,
        "warnings": sorted(set(warnings_list)),
    }
    write_json(report_payload, cfg.output_report_path)
    logging.info("Completed %s", SCRIPT_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
