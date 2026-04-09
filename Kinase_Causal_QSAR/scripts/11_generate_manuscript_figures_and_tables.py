#!/usr/bin/env python3
"""Generate final manuscript-grade figures, tables, and source-data assets.

This script is a strict continuation of Steps 01-10 of the kinase causal-QSAR
pipeline. It consumes previously generated comparison, ablation, robustness,
and interpretation outputs and assembles the final publication-ready manuscript
presentation layer. No model training or benchmark recomputation is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

SCRIPT_NAME = "11_generate_manuscript_figures_and_tables"
DEFAULT_SEED = 2025
NATURE_PALETTE = {
    "classical": "#386CB0",
    "deep": "#F39C12",
    "causal": "#2CA25F",
    "ablation": "#E74C3C",
    "neutral": "#7F8C8D",
    "accent": "#756BB1",
}
LOWER_IS_BETTER = {"rmse", "mae"}
HIGHER_IS_BETTER = {"r2", "spearman", "pearson", "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1", "mcc"}

REQUIRED_SCRIPT_11_KEYS = {
    "input_model_comparison_root",
    "input_classical_results_root",
    "input_deep_results_root",
    "input_causal_results_root",
    "output_manuscript_root",
    "output_main_figures_root",
    "output_supplementary_figures_root",
    "output_main_tables_root",
    "output_supplementary_tables_root",
    "output_figure_source_data_root",
    "output_table_source_data_root",
    "output_manifest_path",
    "output_report_path",
    "generate_main_figures",
    "generate_supplementary_figures",
    "generate_main_tables",
    "generate_supplementary_tables",
    "generate_source_data",
    "export_svg",
    "export_png",
    "export_pdf",
    "png_dpi",
    "figure_style",
    "table_style",
    "figure_plan",
    "table_plan",
    "save_config_snapshot",
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "model_family": ("model_family", "family"),
    "model_name": ("model_name", "model"),
    "ablation_name": ("ablation_name",),
    "task_name": ("task_name", "task"),
    "split_strategy": ("split_strategy", "split_type", "split"),
    "metric_name": ("metric_name",),
    "metric_value": ("metric_value",),
    "activity_cliff_group": ("activity_cliff_group", "activity_cliff_flag"),
    "environment_group": ("environment_group", "group_value"),
    "kinase_family": ("kinase_family", "group_value"),
    "scaffold_group": ("scaffold_group", "group_value", "murcko_scaffold"),
    "training_subset_size": ("training_subset_size", "train_size"),
    "target_chembl_id": ("target_chembl_id", "kinase_id"),
}
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "rmse": ("rmse", "rmse_mean", "test_rmse"),
    "mae": ("mae", "mae_mean", "test_mae"),
    "r2": ("r2", "r2_mean", "test_r2"),
    "spearman": ("spearman", "spearman_mean", "test_spearman"),
    "pearson": ("pearson", "pearson_mean", "test_pearson"),
    "roc_auc": ("roc_auc", "roc_auc_mean"),
    "pr_auc": ("pr_auc", "pr_auc_mean"),
}
MAIN_FIGURE_ORDER = [
    "overall_model_comparison",
    "split_strategy_robustness",
    "causal_vs_baselines",
    "ablation_summary",
    "activity_cliff_comparison",
    "low_data_learning_curves",
]
MAIN_FIGURE_REQUIRED = {
    "overall_model_comparison": True,
    "split_strategy_robustness": True,
    "causal_vs_baselines": False,
    "ablation_summary": True,
    "activity_cliff_comparison": True,
    "low_data_learning_curves": False,
}
SUPP_FIGURE_ORDER = [
    "per_kinase_performance_distribution",
    "environment_group_performance",
    "scaffold_gap_analysis",
    "kinase_family_gap_analysis",
    "additional_scatterplots",
    "additional_classification_curves",
]
MAIN_TABLE_ORDER = [
    "overall_model_ranking",
    "best_models_by_task",
    "ablation_effect_summary",
    "activity_cliff_summary",
]
MAIN_TABLE_REQUIRED = {
    "overall_model_ranking": False,
    "best_models_by_task": True,
    "ablation_effect_summary": True,
    "activity_cliff_summary": True,
}
SUPP_TABLE_ORDER = [
    "detailed_regression_metrics",
    "detailed_classification_metrics",
    "environment_group_metrics",
    "per_kinase_metrics",
    "low_data_metrics",
    "transfer_gap_metrics",
]


@dataclass
class FigureStyle:
    font_family: str
    bold_text: bool
    output_format_primary: str
    palette_name: str
    axis_linewidth: float
    tick_linewidth: float
    line_width: float
    marker_size: float
    title_fontsize: int
    label_fontsize: int
    tick_fontsize: int
    legend_fontsize: int


@dataclass
class TableStyle:
    save_csv: bool
    save_xlsx: bool
    round_decimals: int


@dataclass
class AppConfig:
    input_model_comparison_root: Path
    input_classical_results_root: Path
    input_deep_results_root: Path
    input_causal_results_root: Path
    output_manuscript_root: Path
    output_main_figures_root: Path
    output_supplementary_figures_root: Path
    output_main_tables_root: Path
    output_supplementary_tables_root: Path
    output_figure_source_data_root: Path
    output_table_source_data_root: Path
    output_manifest_path: Path
    output_report_path: Path
    generate_main_figures: bool
    generate_supplementary_figures: bool
    generate_main_tables: bool
    generate_supplementary_tables: bool
    generate_source_data: bool
    export_svg: bool
    export_png: bool
    export_pdf: bool
    png_dpi: int
    figure_style: FigureStyle
    table_style: TableStyle
    figure_plan: dict[str, list[str]]
    table_plan: dict[str, list[str]]
    save_config_snapshot: bool
    logs_dir: Path
    configs_used_dir: Path
    project_root: Path

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        if not isinstance(raw, dict):
            raise ValueError("Config YAML must contain a top-level mapping.")
        section = raw.get("script_11")
        if not isinstance(section, dict):
            raise ValueError("Missing required `script_11` section in config.yaml.")
        missing = sorted(REQUIRED_SCRIPT_11_KEYS.difference(section))
        if missing:
            raise ValueError("Missing required script_11 config values: " + ", ".join(missing))

        def resolve(path_like: Any) -> Path:
            path = Path(str(path_like))
            return path if path.is_absolute() else project_root / path

        def parse_bool(value: Any, key: str) -> bool:
            if isinstance(value, bool):
                return value
            raise ValueError(f"script_11.{key} must be boolean; got {value!r}.")

        figure_style_raw = section.get("figure_style")
        table_style_raw = section.get("table_style")
        figure_plan_raw = section.get("figure_plan")
        table_plan_raw = section.get("table_plan")
        if not isinstance(figure_style_raw, dict):
            raise ValueError("script_11.figure_style must be a mapping.")
        if not isinstance(table_style_raw, dict):
            raise ValueError("script_11.table_style must be a mapping.")
        if not isinstance(figure_plan_raw, dict):
            raise ValueError("script_11.figure_plan must be a mapping.")
        if not isinstance(table_plan_raw, dict):
            raise ValueError("script_11.table_plan must be a mapping.")

        return AppConfig(
            input_model_comparison_root=resolve(section["input_model_comparison_root"]),
            input_classical_results_root=resolve(section["input_classical_results_root"]),
            input_deep_results_root=resolve(section["input_deep_results_root"]),
            input_causal_results_root=resolve(section["input_causal_results_root"]),
            output_manuscript_root=resolve(section["output_manuscript_root"]),
            output_main_figures_root=resolve(section["output_main_figures_root"]),
            output_supplementary_figures_root=resolve(section["output_supplementary_figures_root"]),
            output_main_tables_root=resolve(section["output_main_tables_root"]),
            output_supplementary_tables_root=resolve(section["output_supplementary_tables_root"]),
            output_figure_source_data_root=resolve(section["output_figure_source_data_root"]),
            output_table_source_data_root=resolve(section["output_table_source_data_root"]),
            output_manifest_path=resolve(section["output_manifest_path"]),
            output_report_path=resolve(section["output_report_path"]),
            generate_main_figures=parse_bool(section["generate_main_figures"], "generate_main_figures"),
            generate_supplementary_figures=parse_bool(section["generate_supplementary_figures"], "generate_supplementary_figures"),
            generate_main_tables=parse_bool(section["generate_main_tables"], "generate_main_tables"),
            generate_supplementary_tables=parse_bool(section["generate_supplementary_tables"], "generate_supplementary_tables"),
            generate_source_data=parse_bool(section["generate_source_data"], "generate_source_data"),
            export_svg=parse_bool(section["export_svg"], "export_svg"),
            export_png=parse_bool(section["export_png"], "export_png"),
            export_pdf=parse_bool(section["export_pdf"], "export_pdf"),
            png_dpi=int(section["png_dpi"]),
            figure_style=FigureStyle(
                font_family=str(figure_style_raw.get("font_family", "Times New Roman")),
                bold_text=parse_bool(figure_style_raw.get("bold_text", True), "figure_style.bold_text"),
                output_format_primary=str(figure_style_raw.get("output_format_primary", "svg")).lower(),
                palette_name=str(figure_style_raw.get("palette_name", "nature_manuscript_palette")),
                axis_linewidth=float(figure_style_raw.get("axis_linewidth", 1.2)),
                tick_linewidth=float(figure_style_raw.get("tick_linewidth", 1.2)),
                line_width=float(figure_style_raw.get("line_width", 2.0)),
                marker_size=float(figure_style_raw.get("marker_size", 6)),
                title_fontsize=int(figure_style_raw.get("title_fontsize", 14)),
                label_fontsize=int(figure_style_raw.get("label_fontsize", 12)),
                tick_fontsize=int(figure_style_raw.get("tick_fontsize", 10)),
                legend_fontsize=int(figure_style_raw.get("legend_fontsize", 10)),
            ),
            table_style=TableStyle(
                save_csv=parse_bool(table_style_raw.get("save_csv", True), "table_style.save_csv"),
                save_xlsx=parse_bool(table_style_raw.get("save_xlsx", True), "table_style.save_xlsx"),
                round_decimals=int(table_style_raw.get("round_decimals", 3)),
            ),
            figure_plan={k: list(v) for k, v in figure_plan_raw.items()},
            table_plan={k: list(v) for k, v in table_plan_raw.items()},
            save_config_snapshot=parse_bool(section["save_config_snapshot"], "save_config_snapshot"),
            logs_dir=project_root / "logs",
            configs_used_dir=project_root / "configs_used",
            project_root=project_root,
        )


@dataclass
class AssetRecord:
    asset_id: str
    asset_type: str
    display_name: str
    file_path_svg: str
    file_path_png: str
    file_path_pdf: str
    file_path_csv: str
    file_path_xlsx: str
    source_data_path: str
    originating_step_outputs: str
    notes: str


@dataclass
class AssetOutcome:
    asset_name: str
    asset_type: str
    status: str
    outputs: dict[str, str]
    source_files: list[str]
    notes: str


class RequiredAssetError(RuntimeError):
    pass


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    return log_path


def set_global_determinism(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_config_snapshot(raw_config: dict[str, Any], cfg: AppConfig) -> Path | None:
    if not cfg.save_config_snapshot:
        return None
    cfg.configs_used_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.configs_used_dir / f"{SCRIPT_NAME}_config.yaml"
    output_path.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    logging.info("Saved config snapshot to %s", output_path)
    return output_path


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {description} not found: {path}")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logging.info("Wrote %s", path)


def round_numeric(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    rounded = df.copy()
    numeric_columns = rounded.select_dtypes(include=[np.number]).columns
    rounded[numeric_columns] = rounded[numeric_columns].round(decimals)
    return rounded


def write_table_bundle(df: pd.DataFrame, csv_path: Path, xlsx_path: Path | None, cfg: AppConfig) -> dict[str, str]:
    outputs = {"csv": "", "xlsx": ""}
    rounded = round_numeric(df, cfg.table_style.round_decimals)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.table_style.save_csv:
        rounded.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        outputs["csv"] = str(csv_path)
        logging.info("Wrote %s", csv_path)
    if cfg.table_style.save_xlsx and xlsx_path is not None:
        try:
            rounded.to_excel(xlsx_path, index=False)
            outputs["xlsx"] = str(xlsx_path)
            logging.info("Wrote %s", xlsx_path)
        except Exception as exc:
            logging.warning("Failed to write XLSX %s: %s", xlsx_path, exc)
    return outputs


def save_source_data(df: pd.DataFrame, path: Path, cfg: AppConfig) -> str:
    if not cfg.generate_source_data:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    round_numeric(df, cfg.table_style.round_decimals).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    logging.info("Wrote source data %s", path)
    return str(path)


def alias_lookup(frame: pd.DataFrame, canonical: str) -> str | None:
    for candidate in COLUMN_ALIASES.get(canonical, (canonical,)):
        if candidate in frame.columns:
            return candidate
    return None


def metric_lookup(frame: pd.DataFrame, metric: str) -> str | None:
    for candidate in METRIC_ALIASES.get(metric, (metric,)):
        if candidate in frame.columns:
            return candidate
    return None


def alias_series(frame: pd.DataFrame, canonical: str, default: Any = np.nan) -> pd.Series:
    column = alias_lookup(frame, canonical)
    if column is None:
        return pd.Series([default] * len(frame), index=frame.index)
    return frame[column]


def safe_numeric(series: pd.Series | pd.DataFrame | list[Any] | np.ndarray | pd.Index | Any) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        if series.empty:
            return pd.Series(dtype="float64")
        coerced = series.apply(pd.to_numeric, errors="coerce")
        return coerced.bfill(axis=1).iloc[:, 0]

    if isinstance(series, pd.Index):
        series = pd.Series(series, index=series)
    elif isinstance(series, np.ndarray) and series.ndim > 1:
        if series.size == 0:
            return pd.Series(dtype="float64")
        coerced = pd.DataFrame(series).apply(pd.to_numeric, errors="coerce")
        return coerced.bfill(axis=1).iloc[:, 0]
    elif not isinstance(series, pd.Series):
        try:
            series = pd.Series(series)
        except Exception:
            return pd.Series(dtype="float64")

    return pd.to_numeric(series, errors="coerce")


def normalize_split_strategy(value: Any) -> str:
    text = str(value).strip().lower()
    mapping = {
        "random": "random",
        "random_split": "random",
        "scaffold": "scaffold",
        "scaffold_split": "scaffold",
        "kinase_family": "kinase_family_grouped",
        "kinase_family_grouped": "kinase_family_grouped",
        "grouped_kinase_family": "kinase_family_grouped",
        "source": "source_environment_grouped",
        "source_environment": "source_environment_grouped",
        "source_environment_grouped": "source_environment_grouped",
    }
    return mapping.get(text, text)


def infer_primary_metric(task_name: str) -> str:
    return "roc_auc" if str(task_name).startswith("classification") else "rmse"


def normalize_table(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["model_family"] = alias_series(normalized, "model_family", "unknown").astype(str).str.lower()
    normalized["model_name"] = alias_series(normalized, "model_name", "unknown").astype(str)
    normalized["ablation_name"] = alias_series(normalized, "ablation_name", "none").astype(str)
    normalized["task_name"] = alias_series(normalized, "task_name", "unknown").astype(str)
    normalized["split_strategy"] = alias_series(normalized, "split_strategy", "unknown").map(normalize_split_strategy)
    normalized["activity_cliff_group"] = alias_series(normalized, "activity_cliff_group", np.nan)
    normalized["environment_group"] = alias_series(normalized, "environment_group", np.nan)
    normalized["kinase_family"] = alias_series(normalized, "kinase_family", np.nan)
    normalized["scaffold_group"] = alias_series(normalized, "scaffold_group", np.nan)
    normalized["training_subset_size"] = safe_numeric(alias_series(normalized, "training_subset_size", np.nan))
    normalized["target_chembl_id"] = alias_series(normalized, "target_chembl_id", np.nan)
    for metric in METRIC_ALIASES:
        column = metric_lookup(normalized, metric)
        normalized[metric] = safe_numeric(normalized[column]) if column else np.nan
    if "selection_metric" in normalized.columns and "metric_name" not in normalized.columns:
        normalized["metric_name"] = normalized["selection_metric"].astype(str)
    normalized["metric_name"] = alias_series(normalized, "metric_name", np.nan)
    normalized["metric_value"] = safe_numeric(alias_series(normalized, "metric_value", np.nan))
    return normalized


def load_optional_table(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        logging.warning("Optional source table missing for %s: %s", description, path)
        return pd.DataFrame()
    logging.info("Loading %s from %s", description, path)
    try:
        return normalize_table(pd.read_csv(path))
    except EmptyDataError:
        logging.warning("Optional source table is empty for %s: %s", description, path)
        return pd.DataFrame()


def load_inputs(cfg: AppConfig) -> tuple[dict[str, pd.DataFrame], list[str]]:
    ensure_exists(cfg.input_model_comparison_root, "model comparison root")
    warnings: list[str] = []
    root = cfg.input_model_comparison_root
    expected = {
        "unified_regression_metrics_summary": root / "unified_regression_metrics_summary.csv",
        "unified_regression_metrics_per_fold": root / "unified_regression_metrics_per_fold.csv",
        "unified_classification_metrics_summary": root / "unified_classification_metrics_summary.csv",
        "unified_ablation_metrics_summary": root / "unified_ablation_metrics_summary.csv",
        "best_models_by_task": root / "best_models_by_task.csv",
        "best_models_by_split_strategy": root / "best_models_by_split_strategy.csv",
        "activity_cliff_model_comparison": root / "activity_cliff_model_comparison.csv",
        "activity_cliff_degradation_summary": root / "activity_cliff_degradation_summary.csv",
        "environment_group_metrics": root / "environment_group_metrics.csv",
        "hardest_scaffold_groups": root / "hardest_scaffold_groups.csv",
        "hardest_kinase_families": root / "hardest_kinase_families.csv",
        "source_environment_robustness_summary": root / "source_environment_robustness_summary.csv",
        "per_kinase_performance_summary": root / "per_kinase_performance_summary.csv",
        "regression_model_rankings": root / "regression_model_rankings.csv",
        "classification_model_rankings": root / "classification_model_rankings.csv",
        "causal_vs_best_baseline_regression": root / "causal_vs_best_baseline_regression.csv",
        "causal_vs_best_baseline_classification": root / "causal_vs_best_baseline_classification.csv",
        "transfer_gap_summary": root / "transfer_gap_summary.csv",
        "ablation_drop_summary": root / "ablation_drop_summary.csv",
        "ablation_rank_summary": root / "ablation_rank_summary.csv",
        "low_data_performance_summary": root / "low_data_performance_summary.csv",
        "low_data_learning_curve_source_data": root / "low_data_learning_curve_source_data.csv",
    }
    tables = {name: load_optional_table(path, name) for name, path in expected.items()}
    for name, frame in tables.items():
        if frame.empty:
            warnings.append(f"Missing or empty input: {name}")
    return tables, warnings


def configure_matplotlib(cfg: AppConfig) -> None:
    weight = "bold" if cfg.figure_style.bold_text else "normal"
    rcParams.update({
        "font.family": cfg.figure_style.font_family,
        "font.weight": weight,
        "axes.labelweight": weight,
        "axes.titleweight": weight,
        "axes.linewidth": cfg.figure_style.axis_linewidth,
        "xtick.major.width": cfg.figure_style.tick_linewidth,
        "ytick.major.width": cfg.figure_style.tick_linewidth,
        "axes.titlesize": cfg.figure_style.title_fontsize,
        "axes.labelsize": cfg.figure_style.label_fontsize,
        "xtick.labelsize": cfg.figure_style.tick_fontsize,
        "ytick.labelsize": cfg.figure_style.tick_fontsize,
        "legend.fontsize": cfg.figure_style.legend_fontsize,
        "svg.fonttype": "none",
    })


def style_axis(ax: plt.Axes, cfg: AppConfig) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=cfg.figure_style.tick_linewidth)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)


def family_color(name: str) -> str:
    return NATURE_PALETTE.get(str(name).lower(), NATURE_PALETTE["neutral"])


def save_figure(fig: plt.Figure, base_path: Path, cfg: AppConfig) -> dict[str, str]:
    outputs = {"svg": "", "png": "", "pdf": ""}
    base_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.export_svg:
        svg_path = base_path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight")
        outputs["svg"] = str(svg_path)
    if cfg.export_png:
        png_path = base_path.with_suffix(".png")
        fig.savefig(png_path, bbox_inches="tight", dpi=cfg.png_dpi)
        outputs["png"] = str(png_path)
    if cfg.export_pdf:
        pdf_path = base_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        outputs["pdf"] = str(pdf_path)
    plt.close(fig)
    return outputs


def prepare_metric_view(frame: pd.DataFrame, metric: str, value_col: str | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    view = frame.copy()
    if value_col and value_col in view.columns:
        view["plot_metric"] = safe_numeric(view[value_col])
    else:
        column = metric_lookup(view, metric) or f"{metric}_mean"
        if column in view.columns:
            view["plot_metric"] = safe_numeric(view[column])
        elif metric in view.columns:
            view["plot_metric"] = safe_numeric(view[metric])
        else:
            return pd.DataFrame()
    return view[view["plot_metric"].notna()].copy()


def rank_for_display(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ascending = metric in LOWER_IS_BETTER
    return frame.sort_values(["plot_metric", "model_family", "model_name"], ascending=[ascending, True, True], kind="mergesort").reset_index(drop=True)


def required_or_skip(frame: pd.DataFrame, asset_name: str, required: bool, source_name: str) -> pd.DataFrame:
    if not frame.empty:
        return frame
    message = f"Asset `{asset_name}` could not be generated because required input `{source_name}` is missing or empty."
    if required:
        raise RequiredAssetError(message)
    logging.warning(message)
    return pd.DataFrame()


def make_bar_figure(df: pd.DataFrame, x_label: str, y_label: str, title: str, category_col: str, cfg: AppConfig) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    order = df[category_col].astype(str).tolist()
    colors = [family_color(value) for value in order]
    ax.bar(order, df["plot_metric"], color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=30)
    style_axis(ax, cfg)
    return fig


def generate_overall_model_comparison(tables: dict[str, pd.DataFrame], cfg: AppConfig, asset_id: str, display_name: str, required: bool) -> AssetOutcome:
    source = required_or_skip(tables["best_models_by_task"], display_name, required, "best_models_by_task")
    if source.empty:
        return AssetOutcome(display_name, "main_figure", "skipped", {}, ["best_models_by_task"], "Missing best_models_by_task.")
    source = source.copy()
    source["primary_metric"] = source["task_name"].map(infer_primary_metric)
    source["plot_metric"] = [row.get(metric_lookup(source, metric) or metric, np.nan) for metric, (_, row) in zip(source["primary_metric"], source.iterrows())]
    source["plot_metric"] = safe_numeric(source["plot_metric"])
    source = source[source["plot_metric"].notna()].copy()
    source = source.sort_values(["task_name", "plot_metric", "model_family", "model_name"], kind="mergesort").reset_index(drop=True)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    families = ["classical", "deep", "causal"]
    tasks = sorted(source["task_name"].astype(str).unique())
    x = np.arange(len(tasks))
    width = 0.24
    for idx, family in enumerate(families):
        subset = source[source["model_family"] == family].copy()
        subset = subset.drop_duplicates(subset=["task_name", "model_family"]).set_index("task_name").reindex(tasks)
        ax.bar(x + (idx - 1) * width, subset["plot_metric"], width=width, label=family.title(), color=family_color(family), edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_title("Overall model comparison across key tasks")
    ax.set_xlabel("Task")
    ax.set_ylabel("Primary selection metric")
    ax.legend(frameon=False)
    style_axis(ax, cfg)
    base_path = cfg.output_main_figures_root / asset_id
    outputs = save_figure(fig, base_path, cfg)
    source_path = save_source_data(source, cfg.output_figure_source_data_root / f"{asset_id}_source_data.csv", cfg)
    outputs["source_data"] = source_path
    return AssetOutcome(display_name, "main_figure", "generated", outputs, ["best_models_by_task"], "Grouped comparison of best classical, deep, and causal models by task.")


def generate_split_strategy_robustness(tables: dict[str, pd.DataFrame], cfg: AppConfig, asset_id: str, display_name: str, required: bool) -> AssetOutcome:
    source = required_or_skip(tables["best_models_by_split_strategy"], display_name, required, "best_models_by_split_strategy")
    if source.empty:
        return AssetOutcome(display_name, "main_figure", "skipped", {}, ["best_models_by_split_strategy"], "Missing best_models_by_split_strategy.")
    metric = "rmse"
    plot_df = prepare_metric_view(source, metric)
    plot_df = rank_for_display(plot_df, metric)
    fig = make_bar_figure(plot_df, "Best model per split strategy", metric.upper(), "Split-strategy robustness", "split_strategy", cfg)
    base_path = cfg.output_main_figures_root / asset_id
    outputs = save_figure(fig, base_path, cfg)
    outputs["source_data"] = save_source_data(plot_df, cfg.output_figure_source_data_root / f"{asset_id}_source_data.csv", cfg)
    return AssetOutcome(display_name, "main_figure", "generated", outputs, ["best_models_by_split_strategy"], "Best-performing models across benchmark split strategies.")


def generate_causal_vs_baselines(tables: dict[str, pd.DataFrame], cfg: AppConfig, asset_id: str, display_name: str, required: bool) -> AssetOutcome:
    source = required_or_skip(tables["causal_vs_best_baseline_regression"], display_name, required, "causal_vs_best_baseline_regression")
    if source.empty:
        return AssetOutcome(display_name, "main_figure", "skipped", {}, ["causal_vs_best_baseline_regression"], "Missing causal-vs-baseline summary.")
    plot_df = source.copy()
    plot_df["comparison_label"] = plot_df["task_name"].astype(str) + " | " + plot_df["split_strategy"].astype(str)
    plot_df["plot_metric"] = safe_numeric(plot_df["relative_improvement_percent"])
    plot_df = plot_df.sort_values(["plot_metric", "comparison_label"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [family_color("causal") if value >= 0 else NATURE_PALETTE["ablation"] for value in plot_df["plot_metric"]]
    ax.bar(plot_df["comparison_label"], plot_df["plot_metric"], color=colors, edgecolor="black")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Causal model vs top baseline")
    ax.set_xlabel("Task | Split strategy")
    ax.set_ylabel("Relative improvement (%)")
    ax.tick_params(axis="x", rotation=40)
    style_axis(ax, cfg)
    base_path = cfg.output_main_figures_root / asset_id
    outputs = save_figure(fig, base_path, cfg)
    outputs["source_data"] = save_source_data(plot_df, cfg.output_figure_source_data_root / f"{asset_id}_source_data.csv", cfg)
    return AssetOutcome(display_name, "main_figure", "generated", outputs, ["causal_vs_best_baseline_regression"], "Direct causal-versus-top-baseline comparison using Step-10 summary output.")


def generate_ablation_summary(tables: dict[str, pd.DataFrame], cfg: AppConfig, asset_id: str, display_name: str, required: bool) -> AssetOutcome:
    source = required_or_skip(tables["ablation_drop_summary"], display_name, required, "ablation_drop_summary")
    if source.empty:
        return AssetOutcome(display_name, "main_figure", "skipped", {}, ["ablation_drop_summary"], "Missing ablation drop summary.")
    plot_df = prepare_metric_view(source.rename(columns={"performance_drop": "rmse"}), "rmse")
    plot_df = plot_df.sort_values(["plot_metric", "ablation_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    fig = make_bar_figure(plot_df.rename(columns={"ablation_name": "model_family"}), "Ablation", "Performance drop", "Causal ablation summary", "ablation_name", cfg)
    base_path = cfg.output_main_figures_root / asset_id
    outputs = save_figure(fig, base_path, cfg)
    outputs["source_data"] = save_source_data(plot_df, cfg.output_figure_source_data_root / f"{asset_id}_source_data.csv", cfg)
    return AssetOutcome(display_name, "main_figure", "generated", outputs, ["ablation_drop_summary"], "Performance degradation of causal-model ablations relative to the full model.")


def generate_activity_cliff_comparison(tables: dict[str, pd.DataFrame], cfg: AppConfig, asset_id: str, display_name: str, required: bool) -> AssetOutcome:
    source = required_or_skip(tables["activity_cliff_model_comparison"], display_name, required, "activity_cliff_model_comparison")
    if source.empty:
        return AssetOutcome(display_name, "main_figure", "skipped", {}, ["activity_cliff_model_comparison"], "Missing activity cliff comparison.")
    plot_df = prepare_metric_view(source, "rmse")
    plot_df["cliff_label"] = np.where(plot_df["activity_cliff_group"].astype(str).str.lower().isin(["1", "true", "cliff", "yes"]), "Cliff", "Non-cliff")
    grouped = plot_df.groupby(["cliff_label", "model_family"], dropna=False, sort=True)["plot_metric"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Non-cliff", "Cliff"]
    families = ["classical", "deep", "causal"]
    x = np.arange(len(labels))
    width = 0.24
    for idx, family in enumerate(families):
        subset = grouped[grouped["model_family"] == family].set_index("cliff_label").reindex(labels)
        ax.bar(x + (idx - 1) * width, subset["plot_metric"], width=width, label=family.title(), color=family_color(family), edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Activity-cliff comparison")
    ax.set_xlabel("Subset")
    ax.set_ylabel("RMSE")
    ax.legend(frameon=False)
    style_axis(ax, cfg)
    base_path = cfg.output_main_figures_root / asset_id
    outputs = save_figure(fig, base_path, cfg)
    outputs["source_data"] = save_source_data(grouped, cfg.output_figure_source_data_root / f"{asset_id}_source_data.csv", cfg)
    return AssetOutcome(display_name, "main_figure", "generated", outputs, ["activity_cliff_model_comparison"], "RMSE comparison on cliff and non-cliff subsets.")


def generate_low_data_learning_curves(tables: dict[str, pd.DataFrame], cfg: AppConfig, asset_id: str, display_name: str, required: bool) -> AssetOutcome:
    source = required_or_skip(tables["low_data_learning_curve_source_data"], display_name, required, "low_data_learning_curve_source_data")
    if source.empty:
        return AssetOutcome(display_name, "main_figure", "skipped", {}, ["low_data_learning_curve_source_data"], "Missing low-data source table.")
    plot_df = prepare_metric_view(source, "rmse")
    if "training_subset_size" not in plot_df.columns or plot_df["training_subset_size"].isna().all():
        plot_df["training_subset_size"] = safe_numeric(plot_df.get("train_size", pd.Series([np.nan] * len(plot_df))))
    plot_df = plot_df[plot_df["training_subset_size"].notna()].copy()
    plot_df = plot_df.sort_values(["model_family", "training_subset_size", "task_name"], kind="mergesort").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for family in ["classical", "deep", "causal"]:
        subset = plot_df[plot_df["model_family"] == family]
        if subset.empty:
            continue
        grouped = subset.groupby("training_subset_size", dropna=False, sort=True)["plot_metric"].mean().reset_index()
        ax.plot(grouped["training_subset_size"], grouped["plot_metric"], marker="o", linewidth=cfg.figure_style.line_width, markersize=cfg.figure_style.marker_size, color=family_color(family), label=family.title())
    ax.set_title("Low-data learning curves")
    ax.set_xlabel("Training subset size")
    ax.set_ylabel("RMSE")
    ax.legend(frameon=False)
    style_axis(ax, cfg)
    base_path = cfg.output_main_figures_root / asset_id
    outputs = save_figure(fig, base_path, cfg)
    outputs["source_data"] = save_source_data(plot_df, cfg.output_figure_source_data_root / f"{asset_id}_source_data.csv", cfg)
    return AssetOutcome(display_name, "main_figure", "generated", outputs, ["low_data_learning_curve_source_data"], "Learning curves from Step-10 low-data summaries.")


def generate_per_kinase_distribution(tables: dict[str, pd.DataFrame], cfg: AppConfig, asset_id: str, display_name: str) -> AssetOutcome:
    source = tables["per_kinase_performance_summary"]
    if source.empty:
        return AssetOutcome(display_name, "supplementary_figure", "skipped", {}, ["per_kinase_performance_summary"], "Missing per-kinase performance summary.")
    plot_df = prepare_metric_view(source, "rmse")
    fig, ax = plt.subplots(figsize=(10, 6))
    families = [family for family in ["classical", "deep", "causal"] if family in plot_df["model_family"].unique()]
    positions = np.arange(len(families))
    data = [plot_df.loc[plot_df["model_family"] == family, "plot_metric"].dropna().to_numpy() for family in families]
    ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, boxprops={"facecolor": "white", "edgecolor": "black"}, medianprops={"color": NATURE_PALETTE["accent"], "linewidth": 2})
    for pos, family in zip(positions, families):
        ax.scatter(np.full_like(data[pos], pos, dtype=float), data[pos], s=12, color=family_color(family), alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([family.title() for family in families])
    ax.set_title("Per-kinase performance distribution")
    ax.set_xlabel("Model family")
    ax.set_ylabel("RMSE")
    style_axis(ax, cfg)
    base_path = cfg.output_supplementary_figures_root / asset_id
    outputs = save_figure(fig, base_path, cfg)
    outputs["source_data"] = save_source_data(plot_df, cfg.output_figure_source_data_root / f"{asset_id}_source_data.csv", cfg)
    return AssetOutcome(display_name, "supplementary_figure", "generated", outputs, ["per_kinase_performance_summary"], "Distribution of per-kinase regression errors.")


def generate_environment_group_performance(tables: dict[str, pd.DataFrame], cfg: AppConfig, asset_id: str, display_name: str) -> AssetOutcome:
    source = tables["environment_group_metrics"]
    if source.empty:
        return AssetOutcome(display_name, "supplementary_figure", "skipped", {}, ["environment_group_metrics"], "Missing environment-group metrics.")
    plot_df = prepare_metric_view(source, "rmse")
    grouped = plot_df.groupby(["model_family", "group_type"], dropna=False, sort=True)["plot_metric"].mean().reset_index()
    grouped = grouped.sort_values(["group_type", "plot_metric"], kind="mergesort").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    x_labels = grouped["group_type"].astype(str).unique().tolist()
    x = np.arange(len(x_labels))
    width = 0.24
    for idx, family in enumerate(["classical", "deep", "causal"]):
        subset = grouped[grouped["model_family"] == family].set_index("group_type").reindex(x_labels)
        ax.bar(x + (idx - 1) * width, subset["plot_metric"], width=width, label=family.title(), color=family_color(family), edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_title("Environment-group performance")
    ax.set_xlabel("Environment group type")
    ax.set_ylabel("Mean RMSE")
    ax.legend(frameon=False)
    style_axis(ax, cfg)
    base_path = cfg.output_supplementary_figures_root / asset_id
    outputs = save_figure(fig, base_path, cfg)
    outputs["source_data"] = save_source_data(grouped, cfg.output_figure_source_data_root / f"{asset_id}_source_data.csv", cfg)
    return AssetOutcome(display_name, "supplementary_figure", "generated", outputs, ["environment_group_metrics"], "Average error across scaffold, kinase-family, and source environment groups.")


def generate_scaffold_gap_analysis(tables: dict[str, pd.DataFrame], cfg: AppConfig, asset_id: str, display_name: str) -> AssetOutcome:
    source = tables["hardest_scaffold_groups"]
    if source.empty:
        return AssetOutcome(display_name, "supplementary_figure", "skipped", {}, ["hardest_scaffold_groups"], "Missing hardest scaffold groups.")
    plot_df = prepare_metric_view(source, "rmse").head(15)
    plot_df["label"] = plot_df["group_value"].astype(str)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(plot_df["label"], plot_df["plot_metric"], color=NATURE_PALETTE["accent"], edgecolor="black")
    ax.set_title("Scaffold gap analysis")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Scaffold group")
    style_axis(ax, cfg)
    base_path = cfg.output_supplementary_figures_root / asset_id
    outputs = save_figure(fig, base_path, cfg)
    outputs["source_data"] = save_source_data(plot_df, cfg.output_figure_source_data_root / f"{asset_id}_source_data.csv", cfg)
    return AssetOutcome(display_name, "supplementary_figure", "generated", outputs, ["hardest_scaffold_groups"], "Highest-error scaffold groups from Step-10 transfer-gap summaries.")


def generate_kinase_family_gap_analysis(tables: dict[str, pd.DataFrame], cfg: AppConfig, asset_id: str, display_name: str) -> AssetOutcome:
    source = tables["hardest_kinase_families"]
    if source.empty:
        return AssetOutcome(display_name, "supplementary_figure", "skipped", {}, ["hardest_kinase_families"], "Missing hardest kinase families.")
    plot_df = prepare_metric_view(source, "rmse").head(15)
    plot_df["label"] = plot_df["group_value"].astype(str)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(plot_df["label"], plot_df["plot_metric"], color=NATURE_PALETTE["orange"], edgecolor="black")
    ax.set_title("Kinase-family gap analysis")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Kinase family")
    style_axis(ax, cfg)
    base_path = cfg.output_supplementary_figures_root / asset_id
    outputs = save_figure(fig, base_path, cfg)
    outputs["source_data"] = save_source_data(plot_df, cfg.output_figure_source_data_root / f"{asset_id}_source_data.csv", cfg)
    return AssetOutcome(display_name, "supplementary_figure", "generated", outputs, ["hardest_kinase_families"], "Highest-error kinase families from environment-group analysis.")


def generate_additional_scatterplots(tables: dict[str, pd.DataFrame], cfg: AppConfig, asset_id: str, display_name: str) -> AssetOutcome:
    source = tables["unified_regression_metrics_per_fold"]
    if source.empty:
        return AssetOutcome(display_name, "supplementary_figure", "skipped", {}, ["unified_regression_metrics_per_fold"], "Missing regression per-fold metrics.")
    if source["rmse"].isna().all() or source["mae"].isna().all():
        return AssetOutcome(display_name, "supplementary_figure", "skipped", {}, ["unified_regression_metrics_per_fold"], "RMSE/MAE columns unavailable for scatterplot.")
    plot_df = source[["model_family", "model_name", "task_name", "split_strategy", "rmse", "mae"]].dropna().copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    for family in sorted(plot_df["model_family"].astype(str).unique()):
        subset = plot_df[plot_df["model_family"] == family]
        ax.scatter(subset["rmse"], subset["mae"], s=40, color=family_color(family), label=family.title(), alpha=0.8, edgecolors="black", linewidths=0.4)
    ax.set_title("Additional regression scatterplots")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("MAE")
    ax.legend(frameon=False)
    style_axis(ax, cfg)
    base_path = cfg.output_supplementary_figures_root / asset_id
    outputs = save_figure(fig, base_path, cfg)
    outputs["source_data"] = save_source_data(plot_df, cfg.output_figure_source_data_root / f"{asset_id}_source_data.csv", cfg)
    return AssetOutcome(display_name, "supplementary_figure", "generated", outputs, ["unified_regression_metrics_per_fold"], "Scatter view of fold-level RMSE versus MAE.")


def generate_additional_classification_curves(tables: dict[str, pd.DataFrame], cfg: AppConfig, asset_id: str, display_name: str) -> AssetOutcome:
    source = tables["unified_classification_metrics_summary"]
    if source.empty or source["roc_auc"].isna().all() or source["pr_auc"].isna().all():
        return AssetOutcome(display_name, "supplementary_figure", "skipped", {}, ["unified_classification_metrics_summary"], "Classification metrics unavailable for supplementary curve proxy figure.")
    plot_df = source[["model_family", "model_name", "task_name", "split_strategy", "roc_auc", "pr_auc"]].dropna().copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    for family in sorted(plot_df["model_family"].astype(str).unique()):
        subset = plot_df[plot_df["model_family"] == family]
        ax.scatter(subset["roc_auc"], subset["pr_auc"], s=40, color=family_color(family), label=family.title(), alpha=0.8, edgecolors="black", linewidths=0.4)
    ax.set_title("Additional classification summary curves")
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("PR-AUC")
    ax.legend(frameon=False)
    style_axis(ax, cfg)
    base_path = cfg.output_supplementary_figures_root / asset_id
    outputs = save_figure(fig, base_path, cfg)
    outputs["source_data"] = save_source_data(plot_df, cfg.output_figure_source_data_root / f"{asset_id}_source_data.csv", cfg)
    return AssetOutcome(display_name, "supplementary_figure", "generated", outputs, ["unified_classification_metrics_summary"], "Summary ROC-AUC versus PR-AUC comparison when classification outputs exist.")


def derive_overall_model_ranking(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    reg = tables["regression_model_rankings"]
    cls = tables["classification_model_rankings"]
    frames = []
    if not reg.empty:
        reg_frame = reg.copy()
        reg_frame["task_type"] = "regression"
        frames.append(reg_frame)
    if not cls.empty:
        cls_frame = cls.copy()
        cls_frame["task_type"] = "classification"
        frames.append(cls_frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    sort_col = "rank_within_task_split" if "rank_within_task_split" in combined.columns else "model_name"
    return combined.sort_values([sort_col, "task_name", "split_strategy", "model_family", "model_name"], kind="mergesort").reset_index(drop=True)


def derive_activity_cliff_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    source = tables["activity_cliff_degradation_summary"]
    if not source.empty:
        return source.sort_values(["task_name", "split_strategy", "cliff_degradation"], ascending=[True, True, False], kind="mergesort").reset_index(drop=True)
    return tables["activity_cliff_model_comparison"].sort_values(["task_name", "split_strategy", "model_family"], kind="mergesort").reset_index(drop=True) if not tables["activity_cliff_model_comparison"].empty else pd.DataFrame()


def generate_table_asset(df: pd.DataFrame, cfg: AppConfig, asset_id: str, display_name: str, folder: Path, asset_type: str, source_table_path: Path, notes: str, required: bool, source_files: list[str]) -> AssetOutcome:
    if df.empty:
        message = f"Table `{display_name}` could not be generated because its derived dataframe is empty."
        if required:
            raise RequiredAssetError(message)
        logging.warning(message)
        return AssetOutcome(display_name, asset_type, "skipped", {}, source_files, message)
    csv_path = folder / f"{asset_id}.csv"
    xlsx_path = folder / f"{asset_id}.xlsx"
    outputs = write_table_bundle(df, csv_path, xlsx_path, cfg)
    outputs["source_data"] = save_source_data(df, source_table_path, cfg)
    return AssetOutcome(display_name, asset_type, "generated", outputs, source_files, notes)


def build_manifest_row(outcome: AssetOutcome, asset_id: str) -> AssetRecord:
    return AssetRecord(
        asset_id=asset_id,
        asset_type=outcome.asset_type,
        display_name=outcome.asset_name,
        file_path_svg=outcome.outputs.get("svg", ""),
        file_path_png=outcome.outputs.get("png", ""),
        file_path_pdf=outcome.outputs.get("pdf", ""),
        file_path_csv=outcome.outputs.get("csv", ""),
        file_path_xlsx=outcome.outputs.get("xlsx", ""),
        source_data_path=outcome.outputs.get("source_data", ""),
        originating_step_outputs="; ".join(outcome.source_files),
        notes=outcome.notes,
    )


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    raw_config = load_yaml(config_path)
    cfg = AppConfig.from_dict(raw_config, project_root)
    log_path = configure_logging(cfg)
    set_global_determinism()
    configure_matplotlib(cfg)
    config_snapshot = save_config_snapshot(raw_config, cfg)
    tables, load_warnings = load_inputs(cfg)

    for path in [
        cfg.output_manuscript_root,
        cfg.output_main_figures_root,
        cfg.output_supplementary_figures_root,
        cfg.output_main_tables_root,
        cfg.output_supplementary_tables_root,
        cfg.output_figure_source_data_root,
        cfg.output_table_source_data_root,
        cfg.output_manifest_path.parent,
        cfg.output_report_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[AssetRecord] = []
    generated_assets: list[dict[str, Any]] = []
    skipped_assets: list[dict[str, Any]] = []
    warnings_list = list(load_warnings)

    main_figure_generators = {
        "overall_model_comparison": generate_overall_model_comparison,
        "split_strategy_robustness": generate_split_strategy_robustness,
        "causal_vs_baselines": generate_causal_vs_baselines,
        "ablation_summary": generate_ablation_summary,
        "activity_cliff_comparison": generate_activity_cliff_comparison,
        "low_data_learning_curves": generate_low_data_learning_curves,
    }
    supp_figure_generators = {
        "per_kinase_performance_distribution": generate_per_kinase_distribution,
        "environment_group_performance": generate_environment_group_performance,
        "scaffold_gap_analysis": generate_scaffold_gap_analysis,
        "kinase_family_gap_analysis": generate_kinase_family_gap_analysis,
        "additional_scatterplots": generate_additional_scatterplots,
        "additional_classification_curves": generate_additional_classification_curves,
    }

    main_figure_names = cfg.figure_plan.get("main_figures", MAIN_FIGURE_ORDER)
    if cfg.generate_main_figures:
        for idx, figure_name in enumerate(main_figure_names, start=1):
            asset_id = f"Figure_{idx}"
            generator = main_figure_generators.get(figure_name)
            if generator is None:
                raise ValueError(f"Unknown main figure name in config: {figure_name}")
            required = MAIN_FIGURE_REQUIRED.get(figure_name, True)
            outcome = generator(tables, cfg, asset_id, figure_name, required)
            if outcome.status == "generated":
                manifest_rows.append(build_manifest_row(outcome, asset_id))
                generated_assets.append({"asset_id": asset_id, **asdict(outcome)})
            else:
                skipped_assets.append({"asset_id": asset_id, **asdict(outcome)})
                warnings_list.append(outcome.notes)

    supp_figure_names = cfg.figure_plan.get("supplementary_figures", SUPP_FIGURE_ORDER)
    if cfg.generate_supplementary_figures:
        for idx, figure_name in enumerate(supp_figure_names, start=1):
            asset_id = f"Figure_S{idx}"
            generator = supp_figure_generators.get(figure_name)
            if generator is None:
                raise ValueError(f"Unknown supplementary figure name in config: {figure_name}")
            outcome = generator(tables, cfg, asset_id, figure_name)
            if outcome.status == "generated":
                manifest_rows.append(build_manifest_row(outcome, asset_id))
                generated_assets.append({"asset_id": asset_id, **asdict(outcome)})
            else:
                skipped_assets.append({"asset_id": asset_id, **asdict(outcome)})
                warnings_list.append(outcome.notes)

    main_table_sources = {
        "overall_model_ranking": (derive_overall_model_ranking(tables), ["regression_model_rankings", "classification_model_rankings"], "Integrated ranking table spanning regression and classification outputs."),
        "best_models_by_task": (tables["best_models_by_task"], ["best_models_by_task"], "Best-performing model per task and family from Step-10."),
        "ablation_effect_summary": (tables["ablation_drop_summary"], ["ablation_drop_summary"], "Effect size summary for causal-model ablations."),
        "activity_cliff_summary": (derive_activity_cliff_summary(tables), ["activity_cliff_degradation_summary", "activity_cliff_model_comparison"], "Summary of cliff versus non-cliff performance differences."),
    }
    supp_table_sources = {
        "detailed_regression_metrics": (tables["unified_regression_metrics_summary"], ["unified_regression_metrics_summary"], "Unified regression metrics aggregated in Step-10."),
        "detailed_classification_metrics": (tables["unified_classification_metrics_summary"], ["unified_classification_metrics_summary"], "Unified classification metrics aggregated in Step-10."),
        "environment_group_metrics": (tables["environment_group_metrics"], ["environment_group_metrics"], "Detailed environment-group robustness metrics."),
        "per_kinase_metrics": (tables["per_kinase_performance_summary"], ["per_kinase_performance_summary"], "Per-kinase model performance summary."),
        "low_data_metrics": (tables["low_data_performance_summary"], ["low_data_performance_summary"], "Low-data subset summary metrics."),
        "transfer_gap_metrics": (tables["transfer_gap_summary"], ["transfer_gap_summary"], "Transfer-gap metrics across non-random split strategies."),
    }

    if cfg.generate_main_tables:
        for idx, table_name in enumerate(cfg.table_plan.get("main_tables", MAIN_TABLE_ORDER), start=1):
            if table_name not in main_table_sources:
                raise ValueError(f"Unknown main table name in config: {table_name}")
            df, sources, notes = main_table_sources[table_name]
            asset_id = f"Table_{idx}"
            required = MAIN_TABLE_REQUIRED.get(table_name, True)
            outcome = generate_table_asset(df, cfg, asset_id, table_name, cfg.output_main_tables_root, "main_table", cfg.output_table_source_data_root / f"{asset_id}_source_data.csv", notes, required, sources)
            if outcome.status == "generated":
                manifest_rows.append(build_manifest_row(outcome, asset_id))
                generated_assets.append({"asset_id": asset_id, **asdict(outcome)})
            else:
                skipped_assets.append({"asset_id": asset_id, **asdict(outcome)})
                warnings_list.append(outcome.notes)

    if cfg.generate_supplementary_tables:
        for idx, table_name in enumerate(cfg.table_plan.get("supplementary_tables", SUPP_TABLE_ORDER), start=1):
            if table_name not in supp_table_sources:
                raise ValueError(f"Unknown supplementary table name in config: {table_name}")
            df, sources, notes = supp_table_sources[table_name]
            asset_id = f"Table_S{idx}"
            outcome = generate_table_asset(df, cfg, asset_id, table_name, cfg.output_supplementary_tables_root, "supplementary_table", cfg.output_table_source_data_root / f"{asset_id}_source_data.csv", notes, False, sources)
            if outcome.status == "generated":
                manifest_rows.append(build_manifest_row(outcome, asset_id))
                generated_assets.append({"asset_id": asset_id, **asdict(outcome)})
            else:
                skipped_assets.append({"asset_id": asset_id, **asdict(outcome)})
                warnings_list.append(outcome.notes)

    manifest_df = pd.DataFrame([asdict(row) for row in manifest_rows])
    manifest_df.to_csv(cfg.output_manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
    logging.info("Wrote %s", cfg.output_manifest_path)

    report = {
        "script_name": SCRIPT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "log_path": str(log_path),
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot) if config_snapshot else None,
        "input_roots": {
            "model_comparison": str(cfg.input_model_comparison_root),
            "classical": str(cfg.input_classical_results_root),
            "deep": str(cfg.input_deep_results_root),
            "causal": str(cfg.input_causal_results_root),
        },
        "assets_generated": generated_assets,
        "assets_skipped": skipped_assets,
        "counts": {
            "generated_total": len(generated_assets),
            "skipped_total": len(skipped_assets),
            "main_figures_generated": sum(1 for asset in generated_assets if asset["asset_type"] == "main_figure"),
            "supplementary_figures_generated": sum(1 for asset in generated_assets if asset["asset_type"] == "supplementary_figure"),
            "main_tables_generated": sum(1 for asset in generated_assets if asset["asset_type"] == "main_table"),
            "supplementary_tables_generated": sum(1 for asset in generated_assets if asset["asset_type"] == "supplementary_table"),
        },
        "warnings": sorted(set(warnings_list)),
        "loaded_source_tables": sorted(tables.keys()),
        "manifest_path": str(cfg.output_manifest_path),
    }
    write_json(report, cfg.output_report_path)
    logging.info("Step-11 manuscript asset generation completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RequiredAssetError as exc:
        logging.error("Required asset generation failed: %s", exc)
        raise
