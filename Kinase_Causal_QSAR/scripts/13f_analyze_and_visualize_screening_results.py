#!/usr/bin/env python3
"""Analyze and visualize Step-13A–13E screening outputs.

Script-13F is the final interpretation and visualization layer for the kinase
causality QSAR screening workflow. It consumes already-generated screening
library, feature, scoring, ranking, and shortlist assets and produces
publication-grade chemical-space figures, shortlist/diversity summaries,
source-data files, manifests, and a provenance-rich JSON report.

This script does not retrain models, rescore compounds, or alter shortlist
membership.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import Normalize

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    umap = None

try:
    from rdkit import Chem, DataStructs  # type: ignore
    from rdkit.Chem import AllChem, Descriptors  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    DataStructs = None
    AllChem = None
    Descriptors = None

SCRIPT_NAME = "13f_analyze_and_visualize_screening_results"
DEFAULT_SEED = 2025
NATURE_PALETTE = {
    "classical": "#386CB0",
    "deep": "#F39C12",
    "causal": "#2CA25F",
    "training": "#6B6B6B",
    "screening": "#8ECAE6",
    "shortlist": "#D81B60",
    "novel": "#1B9E77",
    "known": "#7570B3",
    "neutral": "#8C8C8C",
    "accent": "#E7298A",
    "bucket_a": "#1B9E77",
    "bucket_b": "#D95F02",
    "bucket_c": "#7570B3",
    "bucket_d": "#E7298A",
}
BUCKET_COLOR_MAP = {
    "high_confidence_selective_hits": NATURE_PALETTE["bucket_a"],
    "novel_scaffold_selective_hits": NATURE_PALETTE["bucket_b"],
    "diverse_exploratory_hits": NATURE_PALETTE["bucket_c"],
    "consensus_supported_fallback_hits": NATURE_PALETTE["bucket_d"],
}
REQUIRED_SCRIPT_KEYS = {
    "input_screening_library_path",
    "input_classical_feature_path",
    "input_environment_feature_path",
    "input_unified_scores_path",
    "input_compound_target_ranking_path",
    "input_compound_summary_ranking_path",
    "input_final_shortlist_path",
    "input_shortlist_rationale_path",
    "input_bucket_summary_path",
    "input_diversity_summary_path",
    "input_training_annotated_long_path",
    "input_training_compound_env_path",
    "output_analysis_root",
    "output_figures_root",
    "output_tables_root",
    "output_source_data_root",
    "output_manifest_path",
    "output_report_path",
    "make_umap",
    "make_pca",
    "make_tsne",
    "embedding_input",
    "umap_settings",
    "pca_settings",
    "visualization_panels",
    "target_specific_targets",
    "save_embedding_coordinates",
    "save_plot_source_data",
    "save_config_snapshot",
    "export_svg",
    "export_png",
    "export_pdf",
    "png_dpi",
    "figure_style",
}
REQUIRED_COLUMNS_BY_INPUT = {
    "screening_library": {"screening_compound_id", "standardized_smiles"},
    "classical_features": {"screening_compound_id"},
    "environment_features": {"screening_compound_id"},
    "unified_scores": {"screening_compound_id", "standardized_smiles"},
    "compound_target_ranking": {"screening_compound_id", "target_chembl_id"},
    "compound_summary_ranking": {"screening_compound_id", "standardized_smiles"},
    "final_shortlist": {"screening_compound_id", "assigned_bucket"},
    "shortlist_rationale": {"screening_compound_id"},
    "bucket_summary": set(),
    "diversity_summary": set(),
}
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "screening_compound_id": ("screening_compound_id", "compound_id", "compound_identifier"),
    "standardized_smiles": ("standardized_smiles", "smiles", "canonical_smiles"),
    "assigned_bucket": ("assigned_bucket", "shortlist_bucket", "bucket_name"),
    "source_library_name": ("source_library_name", "library_name", "source_library"),
    "target_chembl_id": ("target_chembl_id", "chembl_target_id", "kinase_chembl_id"),
    "final_strategic_score": (
        "final_strategic_score",
        "final_compound_level_strategic_score",
        "final_compound_target_strategic_score",
        "compound_level_strategic_score",
    ),
    "potency_component": (
        "potency_component",
        "potency_score",
        "normalized_potency_score",
        "predicted_pki",
        "predicted_pKi",
    ),
    "selectivity_component": (
        "selectivity_component",
        "selectivity_score",
        "normalized_selectivity_score",
        "predicted_target_vs_panel_delta_pki",
        "predicted_target_vs_panel_delta_pKi",
    ),
    "uncertainty_proxy": (
        "uncertainty_proxy",
        "uncertainty_penalty",
        "screening_uncertainty_proxy",
        "family_disagreement_score",
    ),
    "applicability_penalty": (
        "applicability_penalty",
        "applicability_penalty_score",
        "screening_applicability_penalty",
        "descriptor_distance_proxy",
    ),
    "scaffold": ("scaffold", "murcko_scaffold", "exact_scaffold"),
    "generic_scaffold": ("generic_scaffold", "murcko_generic_scaffold", "generic_murcko_scaffold"),
    "scaffold_novelty_flag": (
        "scaffold_novelty_flag",
        "is_scaffold_novel",
        "novel_scaffold_flag",
        "scaffold_is_novel",
    ),
    "shortlist_rank": ("shortlist_rank", "overall_rank", "compound_rank"),
    "selection_rationale": ("selection_rationale", "rationale", "selection_reason"),
}
OPTIONAL_TRAINING_COLUMNS = {
    "standardized_smiles",
    "target_chembl_id",
}


@dataclass(frozen=True)
class EmbeddingInputConfig:
    use_morgan_fingerprints: bool
    fingerprint_radius: int
    fingerprint_nbits: int
    fallback_to_rdkit_descriptors: bool


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class AppConfig:
    input_screening_library_path: Path
    input_classical_feature_path: Path
    input_environment_feature_path: Path
    input_unified_scores_path: Path
    input_compound_target_ranking_path: Path
    input_compound_summary_ranking_path: Path
    input_final_shortlist_path: Path
    input_shortlist_rationale_path: Path
    input_bucket_summary_path: Path
    input_diversity_summary_path: Path
    input_training_annotated_long_path: Path
    input_training_compound_env_path: Path
    output_analysis_root: Path
    output_figures_root: Path
    output_tables_root: Path
    output_source_data_root: Path
    output_manifest_path: Path
    output_report_path: Path
    make_umap: bool
    make_pca: bool
    make_tsne: bool
    embedding_input: EmbeddingInputConfig
    umap_settings: dict[str, Any]
    pca_settings: dict[str, Any]
    visualization_panels: dict[str, bool]
    target_specific_targets: tuple[str, ...]
    save_embedding_coordinates: bool
    save_plot_source_data: bool
    save_config_snapshot: bool
    export_svg: bool
    export_png: bool
    export_pdf: bool
    png_dpi: int
    figure_style: FigureStyle
    project_root: Path
    logs_dir: Path
    configs_used_dir: Path

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        section = raw.get("script_13f")
        if not isinstance(section, dict):
            raise ValueError("Missing required `script_13f` section in config.yaml.")
        missing = sorted(REQUIRED_SCRIPT_KEYS.difference(section))
        if missing:
            raise ValueError("Missing required script_13f config values: " + ", ".join(missing))

        def resolve(path_like: Any) -> Path:
            path = Path(str(path_like))
            return path if path.is_absolute() else project_root / path

        def parse_bool(value: Any, key: str) -> bool:
            if not isinstance(value, bool):
                raise ValueError(f"script_13f.{key} must be boolean; got {value!r}.")
            return value

        def parse_int(value: Any, key: str, minimum: int | None = None) -> int:
            try:
                parsed = int(value)
            except Exception as exc:
                raise ValueError(f"script_13f.{key} must be an integer; got {value!r}.") from exc
            if minimum is not None and parsed < minimum:
                raise ValueError(f"script_13f.{key} must be >= {minimum}; got {parsed}.")
            return parsed

        def parse_float(value: Any, key: str) -> float:
            try:
                return float(value)
            except Exception as exc:
                raise ValueError(f"script_13f.{key} must be numeric; got {value!r}.") from exc

        def parse_str_list(value: Any, key: str) -> tuple[str, ...]:
            if not isinstance(value, list):
                raise ValueError(f"script_13f.{key} must be a list.")
            return tuple(str(item).strip() for item in value if str(item).strip())

        embedding_input = section.get("embedding_input")
        figure_style_raw = section.get("figure_style")
        visualization_panels = section.get("visualization_panels")
        if not isinstance(embedding_input, dict):
            raise ValueError("script_13f.embedding_input must be a mapping.")
        if not isinstance(figure_style_raw, dict):
            raise ValueError("script_13f.figure_style must be a mapping.")
        if not isinstance(visualization_panels, dict):
            raise ValueError("script_13f.visualization_panels must be a mapping.")
        if str(figure_style_raw.get("font_family", "")).strip() != "Times New Roman":
            raise ValueError("script_13f.figure_style.font_family must be exactly `Times New Roman`.")
        if not bool(figure_style_raw.get("bold_text", True)):
            raise ValueError("script_13f.figure_style.bold_text must be true for manuscript style consistency.")

        return AppConfig(
            input_screening_library_path=resolve(section["input_screening_library_path"]),
            input_classical_feature_path=resolve(section["input_classical_feature_path"]),
            input_environment_feature_path=resolve(section["input_environment_feature_path"]),
            input_unified_scores_path=resolve(section["input_unified_scores_path"]),
            input_compound_target_ranking_path=resolve(section["input_compound_target_ranking_path"]),
            input_compound_summary_ranking_path=resolve(section["input_compound_summary_ranking_path"]),
            input_final_shortlist_path=resolve(section["input_final_shortlist_path"]),
            input_shortlist_rationale_path=resolve(section["input_shortlist_rationale_path"]),
            input_bucket_summary_path=resolve(section["input_bucket_summary_path"]),
            input_diversity_summary_path=resolve(section["input_diversity_summary_path"]),
            input_training_annotated_long_path=resolve(section["input_training_annotated_long_path"]),
            input_training_compound_env_path=resolve(section["input_training_compound_env_path"]),
            output_analysis_root=resolve(section["output_analysis_root"]),
            output_figures_root=resolve(section["output_figures_root"]),
            output_tables_root=resolve(section["output_tables_root"]),
            output_source_data_root=resolve(section["output_source_data_root"]),
            output_manifest_path=resolve(section["output_manifest_path"]),
            output_report_path=resolve(section["output_report_path"]),
            make_umap=parse_bool(section["make_umap"], "make_umap"),
            make_pca=parse_bool(section["make_pca"], "make_pca"),
            make_tsne=parse_bool(section["make_tsne"], "make_tsne"),
            embedding_input=EmbeddingInputConfig(
                use_morgan_fingerprints=parse_bool(embedding_input.get("use_morgan_fingerprints", True), "embedding_input.use_morgan_fingerprints"),
                fingerprint_radius=parse_int(embedding_input.get("fingerprint_radius", 2), "embedding_input.fingerprint_radius", minimum=1),
                fingerprint_nbits=parse_int(embedding_input.get("fingerprint_nbits", 2048), "embedding_input.fingerprint_nbits", minimum=32),
                fallback_to_rdkit_descriptors=parse_bool(embedding_input.get("fallback_to_rdkit_descriptors", True), "embedding_input.fallback_to_rdkit_descriptors"),
            ),
            umap_settings=dict(section.get("umap_settings", {})),
            pca_settings=dict(section.get("pca_settings", {})),
            visualization_panels={str(k): parse_bool(v, f"visualization_panels.{k}") for k, v in visualization_panels.items()},
            target_specific_targets=parse_str_list(section["target_specific_targets"], "target_specific_targets"),
            save_embedding_coordinates=parse_bool(section["save_embedding_coordinates"], "save_embedding_coordinates"),
            save_plot_source_data=parse_bool(section["save_plot_source_data"], "save_plot_source_data"),
            save_config_snapshot=parse_bool(section["save_config_snapshot"], "save_config_snapshot"),
            export_svg=parse_bool(section["export_svg"], "export_svg"),
            export_png=parse_bool(section["export_png"], "export_png"),
            export_pdf=parse_bool(section["export_pdf"], "export_pdf"),
            png_dpi=parse_int(section["png_dpi"], "png_dpi", minimum=72),
            figure_style=FigureStyle(
                font_family=str(figure_style_raw.get("font_family", "Times New Roman")),
                bold_text=parse_bool(figure_style_raw.get("bold_text", True), "figure_style.bold_text"),
                output_format_primary=str(figure_style_raw.get("output_format_primary", "svg")).lower(),
                palette_name=str(figure_style_raw.get("palette_name", "nature_manuscript_palette")),
                axis_linewidth=parse_float(figure_style_raw.get("axis_linewidth", 1.2), "figure_style.axis_linewidth"),
                tick_linewidth=parse_float(figure_style_raw.get("tick_linewidth", 1.2), "figure_style.tick_linewidth"),
                line_width=parse_float(figure_style_raw.get("line_width", 2.0), "figure_style.line_width"),
                marker_size=parse_float(figure_style_raw.get("marker_size", 12), "figure_style.marker_size"),
                title_fontsize=parse_int(figure_style_raw.get("title_fontsize", 14), "figure_style.title_fontsize", minimum=1),
                label_fontsize=parse_int(figure_style_raw.get("label_fontsize", 12), "figure_style.label_fontsize", minimum=1),
                tick_fontsize=parse_int(figure_style_raw.get("tick_fontsize", 10), "figure_style.tick_fontsize", minimum=1),
                legend_fontsize=parse_int(figure_style_raw.get("legend_fontsize", 10), "figure_style.legend_fontsize", minimum=1),
            ),
            project_root=project_root,
            logs_dir=project_root / "logs",
            configs_used_dir=project_root / "configs_used",
        )


@dataclass
class ManifestEntry:
    asset_id: str
    asset_type: str
    file_path: str
    source_data_path: str
    related_panel: str
    row_count: int | None
    notes: str


@dataclass
class LoadedInputs:
    screening_library: pd.DataFrame
    classical_features: pd.DataFrame
    environment_features: pd.DataFrame
    unified_scores: pd.DataFrame
    compound_target_ranking: pd.DataFrame
    compound_summary_ranking: pd.DataFrame
    final_shortlist: pd.DataFrame
    shortlist_rationale: pd.DataFrame
    bucket_summary: pd.DataFrame
    diversity_summary: pd.DataFrame
    training_annotated_long: pd.DataFrame
    training_compound_env: pd.DataFrame


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.yaml", help="Path to config.yaml")
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must deserialize to a mapping: {path}")
    return raw


def setup_logging(cfg: AppConfig) -> Path:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.logs_dir / f"{SCRIPT_NAME}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_path


def save_config_snapshot(raw_config: dict[str, Any], cfg: AppConfig) -> Path | None:
    if not cfg.save_config_snapshot:
        return None
    cfg.configs_used_dir.mkdir(parents=True, exist_ok=True)
    output = cfg.configs_used_dir / f"{SCRIPT_NAME}_config.yaml"
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config, handle, sort_keys=False)
    return output


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


def style_axis(ax: plt.Axes, cfg: AppConfig, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=cfg.figure_style.tick_linewidth)
    ax.grid(True, axis=grid_axis, color="#E5E5E5", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def relative_to_project(path: Path, cfg: AppConfig) -> str:
    try:
        return str(path.relative_to(cfg.project_root))
    except Exception:
        return str(path)


def load_required_csv(path: Path, label: str, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Step-13F input missing: {label} -> {path}")
    frame = pd.read_csv(path)
    missing = sorted(col for col in required_columns if col not in frame.columns)
    if missing:
        raise ValueError(f"Input `{label}` is missing required columns: {', '.join(missing)}")
    logging.info("Loaded %s with %d rows and %d columns from %s", label, len(frame), len(frame.columns), path)
    return frame


def load_optional_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        logging.warning("Optional input missing for Step-13F: %s -> %s", label, path)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    logging.info("Loaded optional %s with %d rows and %d columns from %s", label, len(frame), len(frame.columns), path)
    return frame


def load_inputs(cfg: AppConfig) -> LoadedInputs:
    return LoadedInputs(
        screening_library=load_required_csv(cfg.input_screening_library_path, "screening_library", REQUIRED_COLUMNS_BY_INPUT["screening_library"]),
        classical_features=load_required_csv(cfg.input_classical_feature_path, "classical_features", REQUIRED_COLUMNS_BY_INPUT["classical_features"]),
        environment_features=load_required_csv(cfg.input_environment_feature_path, "environment_features", REQUIRED_COLUMNS_BY_INPUT["environment_features"]),
        unified_scores=load_required_csv(cfg.input_unified_scores_path, "unified_scores", REQUIRED_COLUMNS_BY_INPUT["unified_scores"]),
        compound_target_ranking=load_required_csv(cfg.input_compound_target_ranking_path, "compound_target_ranking", REQUIRED_COLUMNS_BY_INPUT["compound_target_ranking"]),
        compound_summary_ranking=load_required_csv(cfg.input_compound_summary_ranking_path, "compound_summary_ranking", REQUIRED_COLUMNS_BY_INPUT["compound_summary_ranking"]),
        final_shortlist=load_required_csv(cfg.input_final_shortlist_path, "final_shortlist", REQUIRED_COLUMNS_BY_INPUT["final_shortlist"]),
        shortlist_rationale=load_required_csv(cfg.input_shortlist_rationale_path, "shortlist_rationale", REQUIRED_COLUMNS_BY_INPUT["shortlist_rationale"]),
        bucket_summary=load_required_csv(cfg.input_bucket_summary_path, "bucket_summary", REQUIRED_COLUMNS_BY_INPUT["bucket_summary"]),
        diversity_summary=load_required_csv(cfg.input_diversity_summary_path, "diversity_summary", REQUIRED_COLUMNS_BY_INPUT["diversity_summary"]),
        training_annotated_long=load_optional_csv(cfg.input_training_annotated_long_path, "training_annotated_long"),
        training_compound_env=load_optional_csv(cfg.input_training_compound_env_path, "training_compound_env"),
    )


def standardize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def find_first_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    alias_lookup = {str(col).strip().lower(): col for col in frame.columns}
    for alias in aliases:
        hit = alias_lookup.get(alias.lower())
        if hit is not None:
            return hit
    return None


def resolve_column(frame: pd.DataFrame, canonical_name: str) -> str | None:
    if canonical_name in frame.columns:
        return canonical_name
    aliases = COLUMN_ALIASES.get(canonical_name, (canonical_name,))
    return find_first_column(frame, aliases)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def coalesce_columns(frame: pd.DataFrame, canonical_name: str, fallback_value: Any = np.nan) -> pd.Series:
    aliases = COLUMN_ALIASES.get(canonical_name, (canonical_name,))
    result = pd.Series([fallback_value] * len(frame), index=frame.index)
    found_any = False
    for alias in aliases:
        if alias in frame.columns:
            column = frame[alias]
            if pd.api.types.is_numeric_dtype(column):
                result = result.where(result.notna(), column)
            else:
                column = column.replace("", np.nan)
                result = result.where(result.notna() & (result != ""), column)
            found_any = True
    if not found_any:
        return result
    return result


def normalize_screening_library(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for key in ["screening_compound_id", "standardized_smiles", "source_library_name", "scaffold", "generic_scaffold"]:
        column = resolve_column(result, key)
        if column is not None:
            result[key] = standardize_text(result[column])
    novelty_col = resolve_column(result, "scaffold_novelty_flag")
    if novelty_col is not None:
        result["scaffold_novelty_flag"] = result[novelty_col].fillna(False).astype(bool)
    else:
        result["scaffold_novelty_flag"] = False
    return result


def prepare_compound_level_dataset(inputs: LoadedInputs) -> pd.DataFrame:
    library = normalize_screening_library(inputs.screening_library)
    compound_summary = inputs.compound_summary_ranking.copy()
    shortlist = inputs.final_shortlist.copy()
    rationale = inputs.shortlist_rationale.copy()
    classical = inputs.classical_features.copy()
    environment = inputs.environment_features.copy()

    for frame in (compound_summary, shortlist, rationale, classical, environment):
        compound_id_col = resolve_column(frame, "screening_compound_id")
        if compound_id_col is None:
            raise ValueError("A required Step-13F input is missing `screening_compound_id` after alias normalization.")
        frame["screening_compound_id"] = standardize_text(frame[compound_id_col])

    standardized_smiles_col = resolve_column(compound_summary, "standardized_smiles")
    if standardized_smiles_col is not None:
        compound_summary["standardized_smiles"] = standardize_text(compound_summary[standardized_smiles_col])

    compound_summary["final_strategic_score"] = safe_numeric(coalesce_columns(compound_summary, "final_strategic_score"))
    compound_summary["potency_component"] = safe_numeric(coalesce_columns(compound_summary, "potency_component"))
    compound_summary["selectivity_component"] = safe_numeric(coalesce_columns(compound_summary, "selectivity_component"))
    compound_summary["uncertainty_proxy"] = safe_numeric(coalesce_columns(compound_summary, "uncertainty_proxy"))
    compound_summary["applicability_penalty"] = safe_numeric(coalesce_columns(compound_summary, "applicability_penalty"))
    scaffold_col = resolve_column(compound_summary, "scaffold")
    generic_scaffold_col = resolve_column(compound_summary, "generic_scaffold")
    novelty_col = resolve_column(compound_summary, "scaffold_novelty_flag")
    if scaffold_col is not None:
        compound_summary["scaffold"] = standardize_text(compound_summary[scaffold_col])
    if generic_scaffold_col is not None:
        compound_summary["generic_scaffold"] = standardize_text(compound_summary[generic_scaffold_col])
    if novelty_col is not None:
        compound_summary["scaffold_novelty_flag"] = compound_summary[novelty_col].fillna(False).astype(bool)

    shortlist["assigned_bucket"] = standardize_text(coalesce_columns(shortlist, "assigned_bucket", fallback_value=""))
    shortlist["selection_rationale"] = standardize_text(coalesce_columns(shortlist, "selection_rationale", fallback_value=""))
    shortlist["shortlist_rank"] = safe_numeric(coalesce_columns(shortlist, "shortlist_rank"))
    shortlist["is_shortlisted"] = True

    rationale["selection_rationale"] = standardize_text(coalesce_columns(rationale, "selection_rationale", fallback_value=""))
    rationale_bucket = resolve_column(rationale, "assigned_bucket")
    if rationale_bucket is not None:
        rationale["assigned_bucket"] = standardize_text(rationale[rationale_bucket])

    dataset = library.merge(compound_summary, on=[c for c in ["screening_compound_id"] if c in library.columns and c in compound_summary.columns], how="left", suffixes=("", "_ranking"))
    if "standardized_smiles" not in dataset.columns:
        dataset["standardized_smiles"] = standardize_text(coalesce_columns(dataset, "standardized_smiles", fallback_value=""))
    shortlist_cols = [col for col in ["screening_compound_id", "assigned_bucket", "selection_rationale", "shortlist_rank", "is_shortlisted"] if col in shortlist.columns]
    dataset = dataset.merge(shortlist[shortlist_cols].drop_duplicates(subset=["screening_compound_id"]), on="screening_compound_id", how="left")
    rationale_cols = [col for col in ["screening_compound_id", "selection_rationale", "assigned_bucket"] if col in rationale.columns]
    rationale_view = rationale[rationale_cols].drop_duplicates(subset=["screening_compound_id"])
    if "selection_rationale" in rationale_view.columns:
        rationale_view = rationale_view.rename(columns={"selection_rationale": "selection_rationale_rationale"})
    if "assigned_bucket" in rationale_view.columns:
        rationale_view = rationale_view.rename(columns={"assigned_bucket": "assigned_bucket_rationale"})
    dataset = dataset.merge(rationale_view, on="screening_compound_id", how="left")
    dataset["is_shortlisted"] = dataset.get("is_shortlisted", pd.Series(False, index=dataset.index)).fillna(False).astype(bool)
    dataset["assigned_bucket"] = standardize_text(dataset.get("assigned_bucket", pd.Series(index=dataset.index, dtype=object))).replace("", np.nan)
    dataset["assigned_bucket"] = dataset["assigned_bucket"].fillna(standardize_text(dataset.get("assigned_bucket_rationale", pd.Series(index=dataset.index, dtype=object))))
    dataset["selection_rationale"] = standardize_text(dataset.get("selection_rationale", pd.Series(index=dataset.index, dtype=object))).replace("", np.nan)
    dataset["selection_rationale"] = dataset["selection_rationale"].fillna(standardize_text(dataset.get("selection_rationale_rationale", pd.Series(index=dataset.index, dtype=object))))

    classical_numeric = classical.select_dtypes(include=[np.number]).copy()
    if not classical_numeric.empty:
        classical_numeric = classical_numeric.add_prefix("classical_feature_")
        classical_numeric["screening_compound_id"] = classical["screening_compound_id"].values
        dataset = dataset.merge(classical_numeric, on="screening_compound_id", how="left")

    environment_numeric = environment.select_dtypes(include=[np.number]).copy()
    if not environment_numeric.empty:
        environment_numeric = environment_numeric.add_prefix("environment_feature_")
        environment_numeric["screening_compound_id"] = environment["screening_compound_id"].values
        dataset = dataset.merge(environment_numeric, on="screening_compound_id", how="left")

    non_null_scores = int(dataset["final_strategic_score"].notna().sum())
    if non_null_scores >= 2:
        dataset["final_score_quantile"] = pd.qcut(dataset["final_strategic_score"].rank(method="first"), q=min(5, non_null_scores), labels=False, duplicates="drop")
        dataset["final_score_quantile"] = dataset["final_score_quantile"].astype("Int64")
    else:
        dataset["final_score_quantile"] = pd.Series([pd.NA] * len(dataset), index=dataset.index, dtype="Int64")
    dataset["shortlist_membership"] = np.where(dataset["is_shortlisted"], "shortlisted", "not_shortlisted")
    dataset["scaffold_novelty_label"] = np.where(dataset.get("scaffold_novelty_flag", pd.Series(False, index=dataset.index)).fillna(False).astype(bool), "novel", "known_or_unspecified")
    return dataset


def prepare_training_compounds(inputs: LoadedInputs) -> pd.DataFrame:
    if inputs.training_annotated_long.empty:
        return pd.DataFrame(columns=["training_compound_id", "standardized_smiles", "target_chembl_id"])
    training = inputs.training_annotated_long.copy()
    smiles_col = resolve_column(training, "standardized_smiles")
    if smiles_col is None:
        raise ValueError("Training-vs-screening comparison requested but training annotated dataset lacks a standardized_smiles column.")
    training["standardized_smiles"] = standardize_text(training[smiles_col])
    if training["standardized_smiles"].eq("").all():
        raise ValueError("Training annotated dataset contains no usable standardized_smiles values for Step-13F.")
    target_col = resolve_column(training, "target_chembl_id")
    if target_col is not None:
        training["target_chembl_id"] = standardize_text(training[target_col])
    else:
        training["target_chembl_id"] = ""
    training_id_col = resolve_column(training, "screening_compound_id") or find_first_column(training, ("compound_chembl_id", "molregno", "compound_id"))
    if training_id_col is not None:
        training["training_compound_id"] = standardize_text(training[training_id_col])
    else:
        training["training_compound_id"] = [f"training_{idx+1}" for idx in range(len(training))]
    training = training[["training_compound_id", "standardized_smiles", "target_chembl_id"]].drop_duplicates().reset_index(drop=True)
    return training


def build_embedding_feature_matrix(compounds: pd.DataFrame, cfg: AppConfig) -> tuple[np.ndarray, str]:
    smiles = standardize_text(compounds["standardized_smiles"]) if "standardized_smiles" in compounds.columns else pd.Series([], dtype=str)
    if cfg.embedding_input.use_morgan_fingerprints and Chem is not None and AllChem is not None and DataStructs is not None and len(smiles) > 0:
        rows: list[np.ndarray] = []
        valid = 0
        for value in smiles:
            arr = np.zeros((cfg.embedding_input.fingerprint_nbits,), dtype=np.float32)
            mol = Chem.MolFromSmiles(value) if value else None
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, cfg.embedding_input.fingerprint_radius, nBits=cfg.embedding_input.fingerprint_nbits)
                DataStructs.ConvertToNumpyArray(fp, arr)
                valid += 1
            rows.append(arr)
        if valid > 0:
            logging.info("Using Morgan fingerprints for embedding input (%d/%d valid SMILES).", valid, len(smiles))
            return np.vstack(rows), "morgan_fingerprint"
        logging.warning("Morgan fingerprint generation yielded zero valid molecules; evaluating fallbacks.")

    numeric_cols = [col for col in compounds.columns if pd.api.types.is_numeric_dtype(compounds[col]) and not col.startswith("embedding_")]
    if numeric_cols:
        matrix = compounds[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if matrix.shape[1] > 0:
            logging.info("Using numeric descriptor fallback for embedding input with %d columns.", matrix.shape[1])
            return matrix, "numeric_feature_fallback"

    if cfg.embedding_input.fallback_to_rdkit_descriptors and Chem is not None and Descriptors is not None and len(smiles) > 0:
        descriptor_names = [
            "MolWt", "MolLogP", "NumHAcceptors", "NumHDonors", "TPSA", "RingCount", "FractionCSP3", "HeavyAtomCount"
        ]
        rows = []
        valid = 0
        for value in smiles:
            mol = Chem.MolFromSmiles(value) if value else None
            if mol is None:
                rows.append(np.zeros((len(descriptor_names),), dtype=np.float32))
                continue
            rows.append(np.array([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.TPSA(mol),
                Descriptors.RingCount(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.HeavyAtomCount(mol),
            ], dtype=np.float32))
            valid += 1
        if valid > 0:
            logging.info("Using RDKit descriptor fallback for embedding input (%d/%d valid SMILES).", valid, len(smiles))
            return np.vstack(rows), "rdkit_descriptor_fallback"

    raise ValueError("No valid embedding input could be generated for Step-13F. Provide usable SMILES for Morgan/RDKit descriptors or numeric feature columns.")


def compute_embedding(name: str, matrix: np.ndarray, cfg: AppConfig) -> pd.DataFrame:
    if matrix.shape[0] == 0:
        return pd.DataFrame()
    if name == "pca":
        n_components = int(cfg.pca_settings.get("n_components", 2))
        model = PCA(n_components=n_components, random_state=DEFAULT_SEED)
        coords = model.fit_transform(matrix)
        return pd.DataFrame(coords[:, :2], columns=["embedding_x", "embedding_y"])
    if name == "umap":
        if umap is None:
            raise RuntimeError("UMAP requested in script_13f.make_umap but umap-learn is not installed.")
        model = umap.UMAP(
            n_neighbors=int(cfg.umap_settings.get("n_neighbors", 25)),
            min_dist=float(cfg.umap_settings.get("min_dist", 0.15)),
            metric=str(cfg.umap_settings.get("metric", "jaccard")),
            random_state=int(cfg.umap_settings.get("random_state", DEFAULT_SEED)),
            transform_seed=int(cfg.umap_settings.get("random_state", DEFAULT_SEED)),
            n_components=2,
        )
        coords = model.fit_transform(matrix)
        return pd.DataFrame(coords[:, :2], columns=["embedding_x", "embedding_y"])
    if name == "tsne":
        perplexity = min(30, max(5, matrix.shape[0] - 1))
        model = TSNE(n_components=2, random_state=DEFAULT_SEED, init="pca", learning_rate="auto", perplexity=perplexity)
        coords = model.fit_transform(matrix)
        return pd.DataFrame(coords[:, :2], columns=["embedding_x", "embedding_y"])
    raise ValueError(f"Unsupported embedding name: {name}")


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logging.info("Wrote table with %d rows to %s", len(frame), path)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
    logging.info("Wrote JSON report to %s", path)


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


def pick_primary_figure_path(output_paths: dict[str, str], cfg: AppConfig) -> str:
    preferred = cfg.figure_style.output_format_primary.lower()
    if output_paths.get(preferred):
        return output_paths[preferred]
    for ext in ("svg", "png", "pdf"):
        if output_paths.get(ext):
            return output_paths[ext]
    return ""


def write_source_data(frame: pd.DataFrame, file_name: str, cfg: AppConfig) -> Path | None:
    if not cfg.save_plot_source_data:
        return None
    path = cfg.output_source_data_root / file_name
    save_table(frame, path)
    return path


def add_manifest_entry(manifest: list[ManifestEntry], asset_id: str, asset_type: str, file_path: Path, source_data_path: Path | None, related_panel: str, row_count: int | None, notes: str, cfg: AppConfig) -> None:
    manifest.append(
        ManifestEntry(
            asset_id=asset_id,
            asset_type=asset_type,
            file_path=relative_to_project(file_path, cfg),
            source_data_path=relative_to_project(source_data_path, cfg) if source_data_path is not None else "",
            related_panel=related_panel,
            row_count=row_count,
            notes=notes,
        )
    )


def categorical_color_map(values: Iterable[str], base_map: dict[str, str] | None = None) -> dict[str, str]:
    categories = [str(v) for v in values if str(v) != ""]
    unique = sorted(set(categories))
    palette_cycle = [
        "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666",
        "#386CB0", "#F39C12", "#2CA25F", "#D81B60",
    ]
    mapping = dict(base_map or {})
    idx = 0
    for category in unique:
        if category not in mapping:
            mapping[category] = palette_cycle[idx % len(palette_cycle)]
            idx += 1
    return mapping


def plot_scatter_categorical(frame: pd.DataFrame, x_col: str, y_col: str, category_col: str, title: str, output_stub: str, cfg: AppConfig, manifest: list[ManifestEntry], asset_id: str, source_file_name: str, notes: str, highlight_shortlist: bool = False) -> None:
    plot_df = frame.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        logging.warning("Skipping figure %s because no coordinates were available.", asset_id)
        return
    plot_df[category_col] = standardize_text(plot_df[category_col])
    color_map = categorical_color_map(plot_df[category_col], BUCKET_COLOR_MAP if category_col == "assigned_bucket" else None)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for category in sorted(plot_df[category_col].fillna("missing").replace("", "missing").unique()):
        subset = plot_df[plot_df[category_col].fillna("missing").replace("", "missing") == category]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=cfg.figure_style.marker_size,
            c=color_map.get(category, NATURE_PALETTE["neutral"]),
            alpha=0.75,
            linewidths=0,
            label=category,
        )
    if highlight_shortlist and "is_shortlisted" in plot_df.columns:
        shortlist_df = plot_df[plot_df["is_shortlisted"].fillna(False)]
        if not shortlist_df.empty:
            ax.scatter(shortlist_df[x_col], shortlist_df[y_col], s=cfg.figure_style.marker_size * 2.2, facecolors="none", edgecolors=NATURE_PALETTE["accent"], linewidths=1.2, label="shortlist_overlay")
    ax.set_title(title)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    style_axis(ax, cfg)
    ax.legend(frameon=False, loc="best")
    output_paths = save_figure(fig, cfg.output_figures_root / output_stub, cfg)
    source_path = write_source_data(plot_df, source_file_name, cfg)
    add_manifest_entry(manifest, asset_id, "figure", Path(pick_primary_figure_path(output_paths, cfg)), source_path, output_stub, len(plot_df), notes, cfg)


def plot_scatter_continuous(frame: pd.DataFrame, x_col: str, y_col: str, value_col: str, title: str, output_stub: str, cfg: AppConfig, manifest: list[ManifestEntry], asset_id: str, source_file_name: str, notes: str) -> None:
    plot_df = frame.dropna(subset=[x_col, y_col]).copy()
    plot_df[value_col] = safe_numeric(plot_df[value_col])
    plot_df = plot_df.dropna(subset=[value_col])
    if plot_df.empty:
        logging.warning("Skipping continuous landscape %s because `%s` had no plottable values.", asset_id, value_col)
        return
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    scatter = ax.scatter(
        plot_df[x_col],
        plot_df[y_col],
        c=plot_df[value_col],
        cmap="viridis",
        norm=Normalize(vmin=float(plot_df[value_col].min()), vmax=float(plot_df[value_col].max())),
        s=cfg.figure_style.marker_size,
        alpha=0.85,
        linewidths=0,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(value_col.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    style_axis(ax, cfg)
    output_paths = save_figure(fig, cfg.output_figures_root / output_stub, cfg)
    source_path = write_source_data(plot_df, source_file_name, cfg)
    add_manifest_entry(manifest, asset_id, "figure", Path(pick_primary_figure_path(output_paths, cfg)), source_path, output_stub, len(plot_df), notes, cfg)


def plot_bar(frame: pd.DataFrame, category_col: str, value_col: str, title: str, output_stub: str, cfg: AppConfig, manifest: list[ManifestEntry], asset_id: str, source_file_name: str, notes: str) -> None:
    plot_df = frame.copy()
    plot_df = plot_df.sort_values(value_col, ascending=False)
    if plot_df.empty:
        logging.warning("Skipping bar plot %s because it has no data.", asset_id)
        return
    color_map = categorical_color_map(plot_df[category_col])
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.bar(plot_df[category_col], plot_df[value_col], color=[color_map.get(str(v), NATURE_PALETTE["neutral"]) for v in plot_df[category_col]])
    ax.set_title(title)
    ax.set_xlabel(category_col.replace("_", " ").title())
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=35)
    style_axis(ax, cfg, grid_axis="y")
    output_paths = save_figure(fig, cfg.output_figures_root / output_stub, cfg)
    source_path = write_source_data(plot_df, source_file_name, cfg)
    add_manifest_entry(manifest, asset_id, "figure", Path(pick_primary_figure_path(output_paths, cfg)), source_path, output_stub, len(plot_df), notes, cfg)


def plot_heatmap(frame: pd.DataFrame, title: str, output_stub: str, cfg: AppConfig, manifest: list[ManifestEntry], asset_id: str, source_file_name: str, notes: str) -> None:
    if frame.empty:
        logging.warning("Skipping heatmap %s because it has no data.", asset_id)
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    matrix = frame.to_numpy(dtype=float)
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(frame.shape[1]), labels=list(frame.columns), rotation=45, ha="right")
    ax.set_yticks(range(frame.shape[0]), labels=list(frame.index))
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    style_axis(ax, cfg, grid_axis="both")
    output_paths = save_figure(fig, cfg.output_figures_root / output_stub, cfg)
    source_path = write_source_data(frame.reset_index().rename(columns={frame.index.name or "index": "row_label"}), source_file_name, cfg)
    add_manifest_entry(manifest, asset_id, "figure", Path(pick_primary_figure_path(output_paths, cfg)), source_path, output_stub, int(frame.shape[0] * frame.shape[1]), notes, cfg)


def summarize_embeddings(coordinates: dict[str, pd.DataFrame], embedding_source: str) -> pd.DataFrame:
    rows = []
    for name, frame in coordinates.items():
        if frame.empty:
            continue
        rows.append({
            "embedding_name": name,
            "row_count": len(frame),
            "x_min": float(frame["embedding_x"].min()),
            "x_max": float(frame["embedding_x"].max()),
            "y_min": float(frame["embedding_y"].min()),
            "y_max": float(frame["embedding_y"].max()),
            "embedding_input_source": embedding_source,
        })
    return pd.DataFrame(rows)


def generate_diversity_tables(dataset: pd.DataFrame, cfg: AppConfig, manifest: list[ManifestEntry]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    shortlist_df = dataset[dataset["is_shortlisted"].fillna(False)].copy()
    if shortlist_df.empty:
        logging.warning("Shortlist is empty in merged Step-13F dataset; diversity tables will be sparse.")
    scaffold_frequency = (
        shortlist_df.assign(scaffold=shortlist_df.get("scaffold", pd.Series(index=shortlist_df.index, dtype=object)).replace("", "unspecified"))
        .groupby(["assigned_bucket", "scaffold"], dropna=False)
        .size()
        .reset_index(name="compound_count")
        .sort_values(["assigned_bucket", "compound_count", "scaffold"], ascending=[True, False, True])
    )
    path = cfg.output_tables_root / "scaffold_frequency_summary.csv"
    save_table(scaffold_frequency, path)
    add_manifest_entry(manifest, "scaffold_frequency_summary", "table", path, None, "scaffold_diversity", len(scaffold_frequency), "Shortlist scaffold frequencies by bucket.", cfg)
    outputs["scaffold_frequency_summary"] = path

    bucket_scaffold_distribution = (
        shortlist_df.groupby("assigned_bucket", dropna=False)
        .agg(
            shortlisted_compounds=("screening_compound_id", "nunique"),
            unique_exact_scaffolds=("scaffold", lambda s: s.replace("", np.nan).dropna().nunique()),
            unique_generic_scaffolds=("generic_scaffold", lambda s: s.replace("", np.nan).dropna().nunique()),
            novel_scaffold_compounds=("scaffold_novelty_flag", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
        )
        .reset_index()
        .sort_values("assigned_bucket")
    )
    path = cfg.output_tables_root / "bucket_scaffold_distribution.csv"
    save_table(bucket_scaffold_distribution, path)
    add_manifest_entry(manifest, "bucket_scaffold_distribution", "table", path, None, "scaffold_diversity", len(bucket_scaffold_distribution), "Bucket-level exact/generic scaffold diversity summary.", cfg)
    outputs["bucket_scaffold_distribution"] = path

    source_library_contribution = (
        shortlist_df.groupby(["assigned_bucket", "source_library_name"], dropna=False)
        .size()
        .reset_index(name="compound_count")
        .sort_values(["assigned_bucket", "compound_count"], ascending=[True, False])
    )
    path = cfg.output_tables_root / "source_library_contribution_summary.csv"
    save_table(source_library_contribution, path)
    add_manifest_entry(manifest, "source_library_contribution_summary", "table", path, None, "source_library_plots", len(source_library_contribution), "Source-library contributions to shortlist buckets.", cfg)
    outputs["source_library_contribution_summary"] = path

    overlap_pivot = pd.crosstab(shortlist_df.get("assigned_bucket", pd.Series(dtype=object)), shortlist_df.get("generic_scaffold", pd.Series(dtype=object)).replace("", "unspecified"))
    if not overlap_pivot.empty:
        binary = (overlap_pivot > 0).astype(int)
        overlap = binary @ binary.T
        overlap.index.name = "assigned_bucket"
        path = cfg.output_tables_root / "bucket_scaffold_overlap_matrix.csv"
        save_table(overlap.reset_index(), path)
        add_manifest_entry(manifest, "bucket_scaffold_overlap_matrix", "table", path, None, "scaffold_diversity", len(overlap), "Generic-scaffold overlap counts across shortlist buckets.", cfg)
        outputs["bucket_scaffold_overlap_matrix"] = path
    return outputs


def generate_summary_tables(dataset: pd.DataFrame, target_ranking: pd.DataFrame, cfg: AppConfig, manifest: list[ManifestEntry]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    shortlist_df = dataset[dataset["is_shortlisted"].fillna(False)].copy()

    bucket_stats = (
        shortlist_df.groupby("assigned_bucket", dropna=False)
        .agg(
            shortlisted_compounds=("screening_compound_id", "nunique"),
            median_final_strategic_score=("final_strategic_score", "median"),
            median_potency_component=("potency_component", "median"),
            median_selectivity_component=("selectivity_component", "median"),
            median_uncertainty_proxy=("uncertainty_proxy", "median"),
            median_applicability_penalty=("applicability_penalty", "median"),
        )
        .reset_index()
        .sort_values("assigned_bucket")
    )
    path = cfg.output_tables_root / "shortlist_bucket_statistics.csv"
    save_table(bucket_stats, path)
    add_manifest_entry(manifest, "shortlist_bucket_statistics", "table", path, None, "shortlist_rationale", len(bucket_stats), "Bucket-level shortlist score summaries.", cfg)
    outputs["shortlist_bucket_statistics"] = path

    top_compounds = shortlist_df.sort_values(["assigned_bucket", "final_strategic_score", "shortlist_rank"], ascending=[True, False, True]).groupby("assigned_bucket", dropna=False).head(15)
    path = cfg.output_tables_root / "top_compounds_by_bucket.csv"
    save_table(top_compounds, path)
    add_manifest_entry(manifest, "top_compounds_by_bucket", "table", path, None, "shortlist_rationale", len(top_compounds), "Top shortlisted compounds per bucket without altering shortlist membership.", cfg)
    outputs["top_compounds_by_bucket"] = path

    score_summary = shortlist_df.groupby("assigned_bucket", dropna=False).agg(
        final_score_min=("final_strategic_score", "min"),
        final_score_median=("final_strategic_score", "median"),
        final_score_max=("final_strategic_score", "max"),
        uncertainty_median=("uncertainty_proxy", "median"),
        applicability_median=("applicability_penalty", "median"),
    ).reset_index()
    path = cfg.output_tables_root / "shortlist_score_distribution_summary.csv"
    save_table(score_summary, path)
    add_manifest_entry(manifest, "shortlist_score_distribution_summary", "table", path, None, "shortlist_rationale", len(score_summary), "Shortlist score distribution summary by bucket.", cfg)
    outputs["shortlist_score_distribution_summary"] = path

    novelty_summary = shortlist_df.groupby("assigned_bucket", dropna=False).agg(
        shortlisted_compounds=("screening_compound_id", "nunique"),
        novel_scaffold_count=("scaffold_novelty_flag", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
    ).reset_index()
    novelty_summary["novel_scaffold_fraction"] = novelty_summary["novel_scaffold_count"] / novelty_summary["shortlisted_compounds"].replace(0, np.nan)
    path = cfg.output_tables_root / "shortlist_novelty_summary.csv"
    save_table(novelty_summary, path)
    add_manifest_entry(manifest, "shortlist_novelty_summary", "table", path, None, "shortlist_rationale", len(novelty_summary), "Novelty summary for shortlisted compounds by bucket.", cfg)
    outputs["shortlist_novelty_summary"] = path

    target_df = target_ranking.copy()
    if not target_df.empty:
        target_id_col = resolve_column(target_df, "target_chembl_id") or "target_chembl_id"
        compound_id_col = resolve_column(target_df, "screening_compound_id") or "screening_compound_id"
        target_df["target_chembl_id"] = standardize_text(target_df[target_id_col])
        target_df["screening_compound_id"] = standardize_text(target_df[compound_id_col])
        target_df["final_strategic_score"] = safe_numeric(coalesce_columns(target_df, "final_strategic_score"))
        target_summary = (
            target_df[target_df["screening_compound_id"].isin(shortlist_df["screening_compound_id"])]
            .groupby("target_chembl_id", dropna=False)
            .agg(
                shortlisted_compounds=("screening_compound_id", "nunique"),
                max_final_strategic_score=("final_strategic_score", "max"),
                median_final_strategic_score=("final_strategic_score", "median"),
            )
            .reset_index()
            .sort_values(["shortlisted_compounds", "max_final_strategic_score"], ascending=[False, False])
        )
    else:
        target_summary = pd.DataFrame(columns=["target_chembl_id", "shortlisted_compounds", "max_final_strategic_score", "median_final_strategic_score"])
    path = cfg.output_tables_root / "target_specific_shortlist_summary.csv"
    save_table(target_summary, path)
    add_manifest_entry(manifest, "target_specific_shortlist_summary", "table", path, None, "target_specific_panels", len(target_summary), "Target-level shortlist summary derived from compound-target strategic rankings.", cfg)
    outputs["target_specific_shortlist_summary"] = path

    return outputs


def generate_embedding_assets(dataset: pd.DataFrame, training_df: pd.DataFrame, cfg: AppConfig, manifest: list[ManifestEntry], warnings: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], str]:
    coordinates_screening: dict[str, pd.DataFrame] = {}
    coordinates_combined: dict[str, pd.DataFrame] = {}
    matrix, embedding_source = build_embedding_feature_matrix(dataset, cfg)
    combined_df = pd.DataFrame()
    combined_matrix = None

    if not training_df.empty:
        training_tagged = training_df.copy()
        training_tagged["dataset_role"] = "training"
        screening_tagged = dataset[["screening_compound_id", "standardized_smiles"]].copy()
        screening_tagged["dataset_role"] = "screening"
        screening_tagged = screening_tagged.rename(columns={"screening_compound_id": "entity_id"})
        training_tagged = training_tagged.rename(columns={"training_compound_id": "entity_id"})
        combined_df = pd.concat([screening_tagged[["entity_id", "standardized_smiles", "dataset_role"]], training_tagged[["entity_id", "standardized_smiles", "dataset_role"]]], ignore_index=True)
        try:
            combined_matrix, _ = build_embedding_feature_matrix(combined_df, cfg)
        except Exception as exc:
            warnings.append(f"Training-vs-screening combined embedding skipped: {exc}")
            logging.warning("Training-vs-screening combined embedding skipped: %s", exc)

    for name, enabled in (("umap", cfg.make_umap), ("pca", cfg.make_pca), ("tsne", cfg.make_tsne)):
        if not enabled:
            continue
        try:
            coords = compute_embedding(name, matrix, cfg)
            coordinates_screening[name] = pd.concat([dataset[["screening_compound_id", "standardized_smiles", "source_library_name", "assigned_bucket", "shortlist_membership", "scaffold_novelty_label", "final_strategic_score", "potency_component", "selectivity_component", "uncertainty_proxy", "applicability_penalty", "is_shortlisted"]].reset_index(drop=True), coords], axis=1)
            if cfg.save_embedding_coordinates:
                path = cfg.output_tables_root / f"screening_{name}_coordinates.csv"
                save_table(coordinates_screening[name], path)
                add_manifest_entry(manifest, f"screening_{name}_coordinates", "table", path, None, f"{name}_coordinates", len(coordinates_screening[name]), f"Screening-library {name.upper()} coordinates using {embedding_source} input.", cfg)
        except Exception as exc:
            warnings.append(f"{name.upper()} embedding for screening library skipped: {exc}")
            logging.warning("%s embedding for screening library skipped: %s", name.upper(), exc)

        if combined_matrix is not None:
            try:
                combined_coords = compute_embedding(name, combined_matrix, cfg)
                combined_frame = pd.concat([combined_df.reset_index(drop=True), combined_coords], axis=1)
                coordinates_combined[name] = combined_frame
                if cfg.save_embedding_coordinates:
                    path = cfg.output_tables_root / f"combined_training_screening_{name}_coordinates.csv"
                    save_table(combined_frame, path)
                    add_manifest_entry(manifest, f"combined_training_screening_{name}_coordinates", "table", path, None, f"combined_{name}_coordinates", len(combined_frame), f"Combined training+screening {name.upper()} coordinates using {embedding_source} input.", cfg)
            except Exception as exc:
                warnings.append(f"Combined training-vs-screening {name.upper()} embedding skipped: {exc}")
                logging.warning("Combined training-vs-screening %s embedding skipped: %s", name.upper(), exc)

    embedding_summary = summarize_embeddings(coordinates_screening, embedding_source)
    path = cfg.output_tables_root / "screening_embedding_summary.csv"
    save_table(embedding_summary, path)
    add_manifest_entry(manifest, "screening_embedding_summary", "table", path, None, "embedding_summary", len(embedding_summary), "Summary statistics for generated screening embeddings.", cfg)
    return coordinates_screening, coordinates_combined, embedding_source


def generate_major_figures(dataset: pd.DataFrame, target_ranking: pd.DataFrame, coordinates: dict[str, pd.DataFrame], combined_coordinates: dict[str, pd.DataFrame], cfg: AppConfig, manifest: list[ManifestEntry], warnings: list[str]) -> None:
    preferred = "umap" if "umap" in coordinates else "pca" if "pca" in coordinates else "tsne" if "tsne" in coordinates else None
    if preferred is None:
        warnings.append("No embedding coordinates were available for Step-13F figure generation.")
        logging.warning("No embedding coordinates available; major maps skipped.")
        return

    emb = coordinates[preferred].copy()
    x_col = "embedding_x"
    y_col = "embedding_y"

    if cfg.visualization_panels.get("full_library_map", False):
        plot_scatter_categorical(emb, x_col, y_col, "source_library_name", f"Screening library {preferred.upper()} map by source library", f"Figure_13F_full_library_map_by_source_{preferred}", cfg, manifest, "Figure_13F_full_library_map_by_source", f"Figure_13F_full_library_map_by_source_{preferred}_source_data.csv", "Full screening library map colored by source library.", highlight_shortlist=False)
        plot_scatter_categorical(emb, x_col, y_col, "scaffold_novelty_label", f"Screening library {preferred.upper()} map by scaffold novelty", f"Figure_13F_full_library_map_by_novelty_{preferred}", cfg, manifest, "Figure_13F_full_library_map_by_novelty", f"Figure_13F_full_library_map_by_novelty_{preferred}_source_data.csv", "Full screening library map colored by scaffold novelty.", highlight_shortlist=False)
        score_quant_df = emb.copy()
        score_quant_df["final_score_quantile_label"] = score_quant_df["final_score_quantile"].astype(str)
        plot_scatter_categorical(score_quant_df, x_col, y_col, "shortlist_membership", f"Screening library {preferred.upper()} shortlist membership map", f"Figure_13F_full_library_map_by_shortlist_{preferred}", cfg, manifest, "Figure_13F_full_library_map_by_shortlist", f"Figure_13F_full_library_map_by_shortlist_{preferred}_source_data.csv", "Full screening library map colored by shortlist membership.", highlight_shortlist=True)
        if "final_strategic_score" in emb.columns:
            plot_scatter_continuous(emb, x_col, y_col, "final_strategic_score", f"Screening library {preferred.upper()} final strategic score landscape", f"Figure_13F_final_score_landscape_{preferred}", cfg, manifest, "Figure_13F_final_score_landscape", f"Figure_13F_final_score_landscape_{preferred}_source_data.csv", "Full screening library map colored by final strategic score.")

    if cfg.visualization_panels.get("shortlist_map", False):
        shortlist_emb = emb.copy()
        plot_scatter_categorical(shortlist_emb, x_col, y_col, "shortlist_membership", f"Shortlist overlay on screening-library {preferred.upper()} map", f"Figure_13F_shortlist_overlay_map_{preferred}", cfg, manifest, "Figure_13F_shortlist_overlay_map", f"Figure_13F_shortlist_overlay_map_{preferred}_source_data.csv", "Full library map with shortlisted compounds highlighted.", highlight_shortlist=True)

    if cfg.visualization_panels.get("bucket_map", False):
        bucket_df = emb[emb["is_shortlisted"].fillna(False)].copy()
        if bucket_df.empty:
            warnings.append("Bucket map skipped because no shortlisted compounds were present in the merged dataset.")
        else:
            plot_scatter_categorical(bucket_df, x_col, y_col, "assigned_bucket", f"Shortlist bucket {preferred.upper()} map", f"Figure_13F_bucket_map_{preferred}", cfg, manifest, "Figure_13F_bucket_map", f"Figure_13F_bucket_map_{preferred}_source_data.csv", "Shortlisted compounds colored by assigned bucket.")

    for flag, column, asset_id, title_stub in [
        ("potency_landscape", "potency_component", "Figure_13F_potency_landscape", "Predicted potency"),
        ("selectivity_landscape", "selectivity_component", "Figure_13F_selectivity_landscape", "Predicted selectivity"),
        ("uncertainty_landscape", "uncertainty_proxy", "Figure_13F_uncertainty_landscape", "Uncertainty proxy"),
        ("applicability_landscape", "applicability_penalty", "Figure_13F_applicability_landscape", "Applicability penalty"),
        ("final_score_landscape", "final_strategic_score", "Figure_13F_final_score_landscape_duplicate", "Final strategic score"),
    ]:
        if cfg.visualization_panels.get(flag, False) and column in emb.columns:
            plot_scatter_continuous(emb, x_col, y_col, column, f"{title_stub} {preferred.upper()} landscape", f"{asset_id}_{preferred}", cfg, manifest, asset_id, f"{asset_id}_{preferred}_source_data.csv", f"{title_stub} landscape across the screening chemical space.")

    if cfg.visualization_panels.get("training_vs_screening_map", False):
        combined_preferred = combined_coordinates.get(preferred)
        if combined_preferred is None or combined_preferred.empty:
            warnings.append("Training-vs-screening comparison requested but combined embedding coordinates were unavailable.")
        else:
            combined_plot = combined_preferred.copy()
            shortlisted_ids = set(dataset.loc[dataset["is_shortlisted"].fillna(False), "screening_compound_id"].astype(str))
            combined_plot["comparison_group"] = np.where(
                combined_plot["dataset_role"] == "training",
                "training_compounds",
                np.where(combined_plot["entity_id"].astype(str).isin(shortlisted_ids), "shortlisted_screening_compounds", "screening_library_compounds"),
            )
            plot_scatter_categorical(combined_plot, x_col, y_col, "comparison_group", f"Training vs screening {preferred.upper()} chemical-space map", f"Figure_13F_training_vs_screening_map_{preferred}", cfg, manifest, "Figure_13F_training_vs_screening_map", f"Figure_13F_training_vs_screening_map_{preferred}_source_data.csv", "Combined training and screening embedding for interpolation/novelty analysis.")

    if cfg.visualization_panels.get("target_specific_panels", False):
        target_df = target_ranking.copy()
        if target_df.empty:
            warnings.append("Target-specific panels requested but compound-target strategic ranking table was empty.")
        else:
            target_id_col = resolve_column(target_df, "target_chembl_id") or "target_chembl_id"
            compound_id_col = resolve_column(target_df, "screening_compound_id") or "screening_compound_id"
            target_df["target_chembl_id"] = standardize_text(target_df[target_id_col])
            target_df["screening_compound_id"] = standardize_text(target_df[compound_id_col])
            target_df["final_strategic_score"] = safe_numeric(coalesce_columns(target_df, "final_strategic_score"))
            target_list = list(cfg.target_specific_targets) or sorted(set(target_df["target_chembl_id"].dropna()))[:6]
            for target in target_list:
                subset_ids = set(target_df.loc[target_df["target_chembl_id"] == target, "screening_compound_id"].astype(str))
                if not subset_ids:
                    warnings.append(f"Target-specific panel skipped for {target}: target absent from compound-target strategic ranking table.")
                    continue
                target_emb = emb.copy()
                target_emb["target_membership"] = np.where(target_emb["screening_compound_id"].astype(str).isin(subset_ids), f"{target}_ranked", "other_screening_compounds")
                plot_scatter_categorical(target_emb, x_col, y_col, "target_membership", f"Target-specific {preferred.upper()} overlay for {target}", f"Figure_13F_target_overlay_{target}_{preferred}", cfg, manifest, f"Figure_13F_target_overlay_{target}", f"Figure_13F_target_overlay_{target}_{preferred}_source_data.csv", f"Target-specific overlay for {target} using compound-target ranking membership.", highlight_shortlist=True)


def generate_shortlist_rationale_figures(dataset: pd.DataFrame, cfg: AppConfig, manifest: list[ManifestEntry], warnings: list[str]) -> None:
    shortlist_df = dataset[dataset["is_shortlisted"].fillna(False)].copy()
    if shortlist_df.empty:
        warnings.append("Shortlist rationale figures skipped because no shortlisted compounds were available in the merged dataset.")
        return
    rationale_counts = shortlist_df["selection_rationale"].fillna("unspecified").replace("", "unspecified").value_counts().rename_axis("selection_rationale").reset_index(name="compound_count")
    plot_bar(rationale_counts, "selection_rationale", "compound_count", "Shortlist rationale composition", "Figure_13F_shortlist_rationale_composition", cfg, manifest, "Figure_13F_shortlist_rationale_composition", "Figure_13F_shortlist_rationale_composition_source_data.csv", "Shortlist composition by recorded rationale text.")

    bucket_counts = shortlist_df["assigned_bucket"].fillna("unspecified").replace("", "unspecified").value_counts().rename_axis("assigned_bucket").reset_index(name="compound_count")
    plot_bar(bucket_counts, "assigned_bucket", "compound_count", "Shortlist bucket composition", "Figure_13F_shortlist_bucket_composition", cfg, manifest, "Figure_13F_shortlist_bucket_composition", "Figure_13F_shortlist_bucket_composition_source_data.csv", "Shortlist composition by assigned bucket.")

    component_frame = shortlist_df[["assigned_bucket", "final_strategic_score", "potency_component", "selectivity_component", "uncertainty_proxy", "applicability_penalty"]].melt(id_vars=["assigned_bucket"], var_name="score_component", value_name="component_value")
    component_summary = component_frame.groupby(["assigned_bucket", "score_component"], dropna=False)["component_value"].median().reset_index()
    save_path = write_source_data(component_summary, "Figure_13F_component_contributions_by_bucket_source_data.csv", cfg)
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    buckets = sorted(component_summary["assigned_bucket"].dropna().unique())
    components = sorted(component_summary["score_component"].dropna().unique())
    width = 0.15
    x = np.arange(len(buckets))
    color_map = categorical_color_map(components)
    for idx, component in enumerate(components):
        subset = component_summary[component_summary["score_component"] == component].set_index("assigned_bucket").reindex(buckets)
        ax.bar(x + idx * width, subset["component_value"].fillna(0.0), width=width, label=component, color=color_map.get(component, NATURE_PALETTE["neutral"]))
    ax.set_xticks(x + width * (len(components) - 1) / 2, labels=buckets, rotation=25, ha="right")
    ax.set_title("Median score-component contributions by shortlist bucket")
    ax.set_xlabel("Assigned bucket")
    ax.set_ylabel("Median component value")
    style_axis(ax, cfg, grid_axis="y")
    ax.legend(frameon=False)
    output_paths = save_figure(fig, cfg.output_figures_root / "Figure_13F_component_contributions_by_bucket", cfg)
    add_manifest_entry(manifest, "Figure_13F_component_contributions_by_bucket", "figure", Path(pick_primary_figure_path(output_paths, cfg)), save_path, "shortlist_rationale", len(component_summary), "Median score-component contributions by shortlist bucket.", cfg)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_config = load_yaml(args.config)
    project_root = args.config.resolve().parent
    cfg = AppConfig.from_dict(raw_config, project_root)
    log_path = setup_logging(cfg)
    configure_matplotlib(cfg)
    config_snapshot = save_config_snapshot(raw_config, cfg)
    np.random.seed(DEFAULT_SEED)

    for path in [cfg.output_analysis_root, cfg.output_figures_root, cfg.output_tables_root, cfg.output_source_data_root, cfg.output_report_path.parent]:
        path.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    manifest_entries: list[ManifestEntry] = []

    logging.info("Starting %s", SCRIPT_NAME)
    inputs = load_inputs(cfg)
    dataset = prepare_compound_level_dataset(inputs)
    if dataset.empty:
        raise ValueError("Merged Step-13F screening dataset is empty after loading required inputs.")
    if dataset["screening_compound_id"].eq("").all():
        raise ValueError("Merged Step-13F screening dataset contains no usable screening_compound_id values.")
    if dataset["standardized_smiles"].eq("").all() and not any(pd.api.types.is_numeric_dtype(dataset[c]) for c in dataset.columns):
        raise ValueError("Merged Step-13F screening dataset has neither usable standardized_smiles values nor numeric feature columns for embedding generation.")

    training_df = prepare_training_compounds(inputs)

    coordinates, combined_coordinates, embedding_source = generate_embedding_assets(dataset, training_df, cfg, manifest_entries, warnings)
    diversity_outputs = generate_diversity_tables(dataset, cfg, manifest_entries)
    summary_outputs = generate_summary_tables(dataset, inputs.compound_target_ranking, cfg, manifest_entries)
    generate_major_figures(dataset, inputs.compound_target_ranking, coordinates, combined_coordinates, cfg, manifest_entries, warnings)
    generate_shortlist_rationale_figures(dataset, cfg, manifest_entries, warnings)

    if cfg.visualization_panels.get("scaffold_diversity_plots", False):
        scaffold_freq = pd.read_csv(diversity_outputs["scaffold_frequency_summary"])
        top_scaffold = scaffold_freq.groupby("scaffold", dropna=False)["compound_count"].sum().reset_index().sort_values("compound_count", ascending=False).head(20)
        plot_bar(top_scaffold, "scaffold", "compound_count", "Top shortlist scaffold frequencies", "Figure_13F_top_shortlist_scaffolds", cfg, manifest_entries, "Figure_13F_top_shortlist_scaffolds", "Figure_13F_top_shortlist_scaffolds_source_data.csv", "Top exact-scaffold frequencies across shortlist buckets.")
        overlap_path = diversity_outputs.get("bucket_scaffold_overlap_matrix")
        if overlap_path is not None and Path(overlap_path).exists():
            overlap_df = pd.read_csv(overlap_path).set_index("assigned_bucket")
            plot_heatmap(overlap_df, "Bucket scaffold overlap heatmap", "Figure_13F_bucket_scaffold_overlap_heatmap", cfg, manifest_entries, "Figure_13F_bucket_scaffold_overlap_heatmap", "Figure_13F_bucket_scaffold_overlap_heatmap_source_data.csv", "Generic-scaffold overlap across shortlist buckets.")

    if cfg.visualization_panels.get("source_library_plots", False):
        source_df = pd.read_csv(diversity_outputs["source_library_contribution_summary"])
        source_totals = source_df.groupby("source_library_name", dropna=False)["compound_count"].sum().reset_index().sort_values("compound_count", ascending=False)
        plot_bar(source_totals, "source_library_name", "compound_count", "Source-library contribution to shortlist", "Figure_13F_source_library_contributions", cfg, manifest_entries, "Figure_13F_source_library_contributions", "Figure_13F_source_library_contributions_source_data.csv", "Source-library contributions across the final shortlist.")

    add_manifest_entry(manifest_entries, "screening_analysis_manifest", "table", cfg.output_manifest_path, None, "manifest", None, "Master manifest for Step-13F assets.", cfg)
    manifest_df = pd.DataFrame(asdict(entry) for entry in manifest_entries)
    save_table(manifest_df, cfg.output_manifest_path)

    report = {
        "script_name": SCRIPT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_files_used": {
            "screening_library": relative_to_project(cfg.input_screening_library_path, cfg),
            "classical_features": relative_to_project(cfg.input_classical_feature_path, cfg),
            "environment_features": relative_to_project(cfg.input_environment_feature_path, cfg),
            "unified_scores": relative_to_project(cfg.input_unified_scores_path, cfg),
            "compound_target_ranking": relative_to_project(cfg.input_compound_target_ranking_path, cfg),
            "compound_summary_ranking": relative_to_project(cfg.input_compound_summary_ranking_path, cfg),
            "final_shortlist": relative_to_project(cfg.input_final_shortlist_path, cfg),
            "shortlist_rationale": relative_to_project(cfg.input_shortlist_rationale_path, cfg),
            "bucket_summary": relative_to_project(cfg.input_bucket_summary_path, cfg),
            "diversity_summary": relative_to_project(cfg.input_diversity_summary_path, cfg),
            "training_annotated_long": relative_to_project(cfg.input_training_annotated_long_path, cfg) if cfg.input_training_annotated_long_path.exists() else "",
            "training_compound_env": relative_to_project(cfg.input_training_compound_env_path, cfg) if cfg.input_training_compound_env_path.exists() else "",
        },
        "embedding_methods_used": sorted(coordinates.keys()),
        "embedding_input_source": embedding_source,
        "number_of_screening_compounds_analyzed": int(dataset["screening_compound_id"].nunique()),
        "number_of_shortlisted_compounds_analyzed": int(dataset.loc[dataset["is_shortlisted"].fillna(False), "screening_compound_id"].nunique()),
        "number_of_training_compounds_included": int(training_df["training_compound_id"].nunique()) if not training_df.empty else 0,
        "figures_generated": manifest_df.loc[manifest_df["asset_type"] == "figure", "file_path"].tolist(),
        "tables_generated": manifest_df.loc[manifest_df["asset_type"] == "table", "file_path"].tolist(),
        "missing_optional_analyses": [message for message in warnings if "skipped" in message.lower() or "unavailable" in message.lower()],
        "warnings": warnings,
        "config_snapshot_reference": relative_to_project(config_snapshot, cfg) if config_snapshot is not None else "",
        "log_path": relative_to_project(log_path, cfg),
    }
    save_json(report, cfg.output_report_path)

    logging.info("Completed %s", SCRIPT_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
