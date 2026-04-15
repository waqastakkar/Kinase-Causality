#!/usr/bin/env python3
"""Score the inference-ready screening library with trained classical, deep, and causal models."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import pickle
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml

SCRIPT_NAME = "13c_score_screening_library_with_trained_models"
SUPPORTED_MODEL_FAMILIES = ("classical", "deep", "causal")
TASK_NAME_ALIASES = {
    "multitask_regression": "multitask_regression",
    "multitask": "multitask_regression",
    "target_vs_panel_regression": "target_vs_panel",
    "target_vs_panel": "target_vs_panel",
    "pairwise_selectivity_regression": "pairwise_selectivity",
    "pairwise_selectivity": "pairwise_selectivity",
}
PREDICTED_VALUE_TYPES = {
    "multitask_regression": "predicted_pKi",
    "target_vs_panel": "predicted_target_vs_panel_delta_pKi",
    "pairwise_selectivity": "predicted_pairwise_delta_pKi",
}
REQUIRED_SCRIPT_13C_KEYS = {
    "input_classical_feature_path",
    "input_graph_manifest_path",
    "input_environment_feature_path",
    "input_feature_manifest_path",
    "input_best_models_by_task_path",
    "input_best_models_by_split_strategy_path",
    "classical_model_root",
    "deep_model_root",
    "causal_model_root",
    "output_scoring_root",
    "output_classical_scores_path",
    "output_deep_scores_path",
    "output_causal_scores_path",
    "output_unified_scores_path",
    "output_manifest_path",
    "output_report_path",
    "score_with_classical_models",
    "score_with_deep_models",
    "score_with_causal_models",
    "scoring_tasks",
    "target_selection_mode",
    "target_chembl_ids",
    "use_best_model_per_task",
    "use_best_model_per_split_strategy",
    "include_supporting_models",
    "max_supporting_models_per_family",
    "save_raw_predictions",
    "save_wide_prediction_tables",
    "save_long_prediction_tables",
    "save_model_metadata_table",
    "save_failed_rows",
    "save_config_snapshot",
    "device",
    "batch_size",
    "num_workers",
    "chunk_size",
}
METADATA_CANDIDATE_NAMES = (
    "inference_metadata.json",
    "model_metadata.json",
    "run_metadata.json",
    "training_summary.json",
    "config.yaml",
    "config_used.yaml",
)
REQUIRED_SCREENING_COLUMNS = {"screening_compound_id", "standardized_smiles"}
OPTIONAL_SCREENING_COLUMNS = ["source_library_name", "target_name", "target_chembl_id"]
TARGET_ANNOTATION_FALLBACK_PATH = Path("data/processed/chembl_human_kinase_panel_annotated_long.csv")
MODEL_SELECTION_COLUMN_CANDIDATES = {
    "model_family": ("model_family", "family"),
    "model_name": ("model_name", "model"),
    "task_name": ("task_name", "task"),
    "split_strategy": ("split_strategy", "split"),
    "selection_metric": ("selection_metric", "metric_name", "metric"),
    "selection_metric_value": ("selection_metric_value", "metric_value", "score"),
    "artifact_path": ("artifact_path", "model_artifact_path", "model_path", "path"),
    "target_label": ("target_label", "label_name", "label_column"),
    "target_chembl_id": ("target_chembl_id", "primary_target_identifier", "target_id"),
    "ablation_name": ("ablation_name",),
}
UNIFIED_COLUMN_ORDER = [
    "screening_compound_id",
    "standardized_smiles",
    "source_library_name",
    "target_chembl_id",
    "target_name",
    "task_name",
    "model_family",
    "model_name",
    "model_artifact_path",
    "split_strategy_used_to_select_model",
    "predicted_value",
    "predicted_value_type",
    "score_type",
    "target_label",
    "selection_criterion",
    "environment_conditioning_metadata",
]


@dataclass(frozen=True)
class ScoringTaskConfig:
    multitask_regression: bool
    target_vs_panel_regression: bool
    pairwise_selectivity_regression: bool


@dataclass(frozen=True)
class AppConfig:
    input_classical_feature_path: Path
    input_graph_manifest_path: Path
    input_environment_feature_path: Path
    input_feature_manifest_path: Path
    input_best_models_by_task_path: Path
    input_best_models_by_split_strategy_path: Path
    classical_model_root: Path
    deep_model_root: Path
    causal_model_root: Path
    output_scoring_root: Path
    output_classical_scores_path: Path
    output_deep_scores_path: Path
    output_causal_scores_path: Path
    output_unified_scores_path: Path
    output_manifest_path: Path
    output_report_path: Path
    score_with_classical_models: bool
    score_with_deep_models: bool
    score_with_causal_models: bool
    scoring_tasks: ScoringTaskConfig
    target_selection_mode: str
    target_chembl_ids: tuple[str, ...]
    use_best_model_per_task: bool
    use_best_model_per_split_strategy: bool
    include_supporting_models: bool
    max_supporting_models_per_family: int
    save_raw_predictions: bool
    save_wide_prediction_tables: bool
    save_long_prediction_tables: bool
    save_model_metadata_table: bool
    save_failed_rows: bool
    save_config_snapshot: bool
    device: str
    batch_size: int
    num_workers: int
    chunk_size: int
    write_every_n_chunks: int
    low_memory: bool
    overwrite_existing_outputs: bool
    allow_full_wide_pivot: bool
    selected_families_to_score: tuple[str, ...]
    skip_unloadable_models: bool
    require_all_models_loadable: bool
    strict_graph_inference_bundle_required: bool
    project_root: Path
    logs_dir: Path
    configs_used_dir: Path

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        section = raw.get("script_13c")
        if not isinstance(section, dict):
            raise ValueError("Missing required `script_13c` section in config.yaml.")
        missing = sorted(REQUIRED_SCRIPT_13C_KEYS.difference(section))
        if missing:
            raise ValueError("Missing required script_13c config values: " + ", ".join(missing))

        def resolve(path_like: Any) -> Path:
            path = Path(str(path_like))
            return path if path.is_absolute() else project_root / path

        def parse_bool(value: Any, key: str) -> bool:
            if not isinstance(value, bool):
                raise ValueError(f"script_13c.{key} must be boolean; got {value!r}.")
            return value

        def parse_int(value: Any, key: str) -> int:
            try:
                return int(value)
            except Exception as exc:
                raise ValueError(f"script_13c.{key} must be an integer; got {value!r}.") from exc

        scoring_tasks = section.get("scoring_tasks")
        if not isinstance(scoring_tasks, dict):
            raise ValueError("script_13c.scoring_tasks must be a mapping.")
        for required_key in ["multitask_regression", "target_vs_panel_regression", "pairwise_selectivity_regression"]:
            if required_key not in scoring_tasks:
                raise ValueError(f"script_13c.scoring_tasks is missing `{required_key}`.")

        targets = section.get("target_chembl_ids")
        if not isinstance(targets, list):
            raise ValueError("script_13c.target_chembl_ids must be a list, even if empty.")

        enabled_from_bools = {
            "classical": parse_bool(section["score_with_classical_models"], "score_with_classical_models"),
            "deep": parse_bool(section["score_with_deep_models"], "score_with_deep_models"),
            "causal": parse_bool(section["score_with_causal_models"], "score_with_causal_models"),
        }
        selected_families_raw = section.get("selected_families_to_score")
        if selected_families_raw is None:
            selected_families = tuple([family for family, enabled in enabled_from_bools.items() if enabled])
        else:
            if not isinstance(selected_families_raw, list):
                raise ValueError("script_13c.selected_families_to_score must be a list when provided.")
            normalized = [str(item).strip().lower() for item in selected_families_raw if str(item).strip()]
            unknown = sorted(set(normalized).difference(SUPPORTED_MODEL_FAMILIES))
            if unknown:
                raise ValueError(f"script_13c.selected_families_to_score contains unsupported families: {unknown}")
            selected_families = tuple(dict.fromkeys(normalized))

        return AppConfig(
            input_classical_feature_path=resolve(section["input_classical_feature_path"]),
            input_graph_manifest_path=resolve(section["input_graph_manifest_path"]),
            input_environment_feature_path=resolve(section["input_environment_feature_path"]),
            input_feature_manifest_path=resolve(section["input_feature_manifest_path"]),
            input_best_models_by_task_path=resolve(section["input_best_models_by_task_path"]),
            input_best_models_by_split_strategy_path=resolve(section["input_best_models_by_split_strategy_path"]),
            classical_model_root=resolve(section["classical_model_root"]),
            deep_model_root=resolve(section["deep_model_root"]),
            causal_model_root=resolve(section["causal_model_root"]),
            output_scoring_root=resolve(section["output_scoring_root"]),
            output_classical_scores_path=resolve(section["output_classical_scores_path"]),
            output_deep_scores_path=resolve(section["output_deep_scores_path"]),
            output_causal_scores_path=resolve(section["output_causal_scores_path"]),
            output_unified_scores_path=resolve(section["output_unified_scores_path"]),
            output_manifest_path=resolve(section["output_manifest_path"]),
            output_report_path=resolve(section["output_report_path"]),
            score_with_classical_models=enabled_from_bools["classical"],
            score_with_deep_models=enabled_from_bools["deep"],
            score_with_causal_models=enabled_from_bools["causal"],
            scoring_tasks=ScoringTaskConfig(
                multitask_regression=parse_bool(scoring_tasks["multitask_regression"], "scoring_tasks.multitask_regression"),
                target_vs_panel_regression=parse_bool(scoring_tasks["target_vs_panel_regression"], "scoring_tasks.target_vs_panel_regression"),
                pairwise_selectivity_regression=parse_bool(scoring_tasks["pairwise_selectivity_regression"], "scoring_tasks.pairwise_selectivity_regression"),
            ),
            target_selection_mode=str(section["target_selection_mode"]).strip(),
            target_chembl_ids=tuple(str(item).strip() for item in targets if str(item).strip()),
            use_best_model_per_task=parse_bool(section["use_best_model_per_task"], "use_best_model_per_task"),
            use_best_model_per_split_strategy=parse_bool(section["use_best_model_per_split_strategy"], "use_best_model_per_split_strategy"),
            include_supporting_models=parse_bool(section["include_supporting_models"], "include_supporting_models"),
            max_supporting_models_per_family=parse_int(section["max_supporting_models_per_family"], "max_supporting_models_per_family"),
            save_raw_predictions=parse_bool(section["save_raw_predictions"], "save_raw_predictions"),
            save_wide_prediction_tables=parse_bool(section["save_wide_prediction_tables"], "save_wide_prediction_tables"),
            save_long_prediction_tables=parse_bool(section["save_long_prediction_tables"], "save_long_prediction_tables"),
            save_model_metadata_table=parse_bool(section["save_model_metadata_table"], "save_model_metadata_table"),
            save_failed_rows=parse_bool(section["save_failed_rows"], "save_failed_rows"),
            save_config_snapshot=parse_bool(section["save_config_snapshot"], "save_config_snapshot"),
            device=str(section["device"]).strip(),
            batch_size=parse_int(section["batch_size"], "batch_size"),
            num_workers=parse_int(section["num_workers"], "num_workers"),
            chunk_size=parse_int(section["chunk_size"], "chunk_size"),
            write_every_n_chunks=max(1, parse_int(section.get("write_every_n_chunks", 1), "write_every_n_chunks")),
            low_memory=parse_bool(section.get("low_memory", True), "low_memory"),
            overwrite_existing_outputs=parse_bool(section.get("overwrite_existing_outputs", False), "overwrite_existing_outputs"),
            allow_full_wide_pivot=parse_bool(section.get("allow_full_wide_pivot", False), "allow_full_wide_pivot"),
            selected_families_to_score=selected_families,
            skip_unloadable_models=parse_bool(section.get("skip_unloadable_models", True), "skip_unloadable_models"),
            require_all_models_loadable=parse_bool(section.get("require_all_models_loadable", False), "require_all_models_loadable"),
            strict_graph_inference_bundle_required=parse_bool(
                section.get("strict_graph_inference_bundle_required", True), "strict_graph_inference_bundle_required"
            ),
            project_root=project_root,
            logs_dir=project_root / "logs",
            configs_used_dir=project_root / "configs_used",
        )


@dataclass
class ModelRecord:
    model_family: str
    model_name: str
    task_name: str
    split_strategy: str
    artifact_path: Path
    source_step: str
    selected_as_best_flag: bool
    selection_criterion: str
    split_strategy_used: str
    target_label: str | None = None
    target_chembl_id: str | None = None
    ablation_name: str | None = None
    supporting_rank: int = 0
    notes: str = ""


@dataclass
class ModelLoadFailure:
    record: ModelRecord
    reason: str
    artifact_diagnostic: str
    reconstruction_stage: str = "unknown"
    exception_type: str = "Exception"
    traceback_summary: str = ""


class ReconstructionError(RuntimeError):
    def __init__(self, code: str, stage: str, *, context: str = "", cause: Exception | None = None):
        message = code if not context else f"{code}::{context}"
        super().__init__(message)
        self.reconstruction_stage = stage
        self.error_code = code
        self.context = context
        if cause is not None:
            self.__cause__ = cause


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to pipeline config YAML.")
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must parse to a mapping: {path}")
    return payload


def setup_logging(cfg: AppConfig) -> Path:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.logs_dir / f"{SCRIPT_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.info("Logging initialized: %s", path)
    return path


def save_config_snapshot(raw_config: dict[str, Any], cfg: AppConfig) -> Path | None:
    if not cfg.save_config_snapshot:
        return None
    cfg.configs_used_dir.mkdir(parents=True, exist_ok=True)
    output = cfg.configs_used_dir / f"{SCRIPT_NAME}_config.yaml"
    output.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    logging.info("Saved config snapshot to %s", output)
    return output


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {description} not found: {path}")


def ensure_columns(df: pd.DataFrame, required: Iterable[str], description: str) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"{description} is missing required columns: {', '.join(missing)}")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info("Wrote %s (%s rows)", path, len(df))


def append_dataframe(df: pd.DataFrame, path: Path, wrote_header: bool) -> bool:
    if df.empty:
        return wrote_header
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not wrote_header, index=False)
    return True


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logging.info("Wrote %s", path)


def alias_column(df: pd.DataFrame, canonical: str) -> str | None:
    for candidate in MODEL_SELECTION_COLUMN_CANDIDATES.get(canonical, (canonical,)):
        if candidate in df.columns:
            return candidate
    return None


def normalize_task_name(value: Any) -> str:
    text = str(value).strip().lower()
    return TASK_NAME_ALIASES.get(text, text)


def normalize_model_selection_table(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["model_family", "model_name", "task_name", "split_strategy", "selection_metric", "selection_metric_value", "artifact_path", "target_label", "target_chembl_id", "ablation_name", "source_table"])
    normalized = pd.DataFrame(index=df.index)
    for canonical in ["model_family", "model_name", "task_name", "split_strategy", "selection_metric", "selection_metric_value", "artifact_path", "target_label", "target_chembl_id", "ablation_name"]:
        column = alias_column(df, canonical)
        normalized[canonical] = df[column] if column else np.nan
    normalized["model_family"] = normalized["model_family"].astype(str).str.strip().str.lower()
    normalized["model_name"] = normalized["model_name"].astype(str).str.strip()
    normalized["task_name"] = normalized["task_name"].map(normalize_task_name)
    normalized["split_strategy"] = normalized["split_strategy"].astype(str).str.strip().fillna("")
    normalized["selection_metric"] = normalized["selection_metric"].astype(str).replace("nan", "")
    normalized["selection_metric_value"] = pd.to_numeric(normalized["selection_metric_value"], errors="coerce")
    normalized["artifact_path"] = normalized["artifact_path"].astype(str).replace("nan", "")
    normalized["target_label"] = normalized["target_label"].replace({np.nan: None})
    normalized["target_chembl_id"] = normalized["target_chembl_id"].replace({np.nan: None})
    normalized["ablation_name"] = normalized["ablation_name"].replace({np.nan: None})
    normalized["source_table"] = table_name
    return normalized


def validate_screening_inputs(cfg: AppConfig) -> None:
    ensure_exists(cfg.input_feature_manifest_path, "screening feature manifest")
    ensure_exists(cfg.input_classical_feature_path, "screening classical feature table")
    ensure_exists(cfg.input_graph_manifest_path, "screening graph manifest")
    ensure_exists(cfg.input_environment_feature_path, "screening environment feature table")
    for path, label in [
        (cfg.input_classical_feature_path, "screening classical feature table"),
        (cfg.input_graph_manifest_path, "screening graph manifest"),
        (cfg.input_environment_feature_path, "screening environment feature table"),
    ]:
        header_df = pd.read_csv(path, nrows=0)
        ensure_columns(header_df, REQUIRED_SCREENING_COLUMNS, label)


def selected_tasks(cfg: AppConfig) -> list[str]:
    tasks: list[str] = []
    if cfg.scoring_tasks.multitask_regression:
        tasks.append("multitask_regression")
    if cfg.scoring_tasks.target_vs_panel_regression:
        tasks.append("target_vs_panel")
    if cfg.scoring_tasks.pairwise_selectivity_regression:
        tasks.append("pairwise_selectivity")
    if not tasks:
        raise ValueError("script_13c must enable at least one scoring task.")
    return tasks


def validate_target_selection(cfg: AppConfig) -> None:
    if cfg.target_selection_mode == "explicit_list" and not cfg.target_chembl_ids:
        raise ValueError("script_13c.target_selection_mode is `explicit_list` but no target_chembl_ids were provided.")


def resolve_artifact_path(raw_path: str, model_family: str, cfg: AppConfig) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    roots = {"classical": cfg.classical_model_root, "deep": cfg.deep_model_root, "causal": cfg.causal_model_root}
    root = roots[model_family]
    for option in [cfg.project_root / candidate, root / candidate, candidate]:
        if option.exists():
            return option
    return cfg.project_root / candidate


def discover_model_artifact(row: pd.Series, model_family: str, cfg: AppConfig) -> Path | None:
    if row.get("artifact_path"):
        resolved = resolve_artifact_path(str(row["artifact_path"]), model_family, cfg)
        if resolved.exists():
            return resolved
    root = {"classical": cfg.classical_model_root, "deep": cfg.deep_model_root, "causal": cfg.causal_model_root}[model_family]
    model_name = str(row.get("model_name", "")).strip()
    task_name = str(row.get("task_name", "")).strip()
    split_strategy = str(row.get("split_strategy", "")).strip()
    target_label = str(row.get("target_label") or "").strip()
    matches: list[Path] = []
    patterns = []
    if model_family == "classical":
        patterns.extend([
            f"{task_name}/**/*{target_label}__{model_name}.pkl" if target_label else f"{task_name}/**/*{model_name}.pkl",
            f"**/{target_label}__{model_name}.pkl" if target_label else f"**/{model_name}.pkl",
        ])
    elif model_family == "deep":
        patterns.extend([f"{task_name}/**/{model_name}/**/best_model.pt", f"**/{model_name}/**/best_model.pt"])
    else:
        patterns.extend([
            f"{task_name}/**/{model_name}/{split_strategy}/**/model_state_dict.pt",
            f"{task_name}/**/{model_name}/**/model_state_dict.pt",
        ])
    for pattern in patterns:
        matches.extend(sorted(root.glob(pattern)))
    return matches[0] if matches else None


def resolve_models(cfg: AppConfig) -> list[ModelRecord]:
    logging.info("Resolving trained models for screening inference.")
    tables: list[pd.DataFrame] = []
    if cfg.use_best_model_per_task:
        ensure_exists(cfg.input_best_models_by_task_path, "best models by task table")
        tables.append(normalize_model_selection_table(pd.read_csv(cfg.input_best_models_by_task_path), "best_models_by_task"))
    if cfg.use_best_model_per_split_strategy:
        ensure_exists(cfg.input_best_models_by_split_strategy_path, "best models by split strategy table")
        tables.append(normalize_model_selection_table(pd.read_csv(cfg.input_best_models_by_split_strategy_path), "best_models_by_split_strategy"))
    if not tables:
        raise ValueError("At least one of use_best_model_per_task or use_best_model_per_split_strategy must be true in script_13c.")

    combined = pd.concat(tables, ignore_index=True)
    combined = combined[combined["task_name"].isin(selected_tasks(cfg))].copy()
    combined = combined[combined["model_family"].isin(cfg.selected_families_to_score)].copy()
    if combined.empty:
        raise ValueError("No candidate models were resolved from the configured model-selection tables for the requested tasks and model families.")

    combined = combined.sort_values(["task_name", "model_family", "selection_metric_value", "model_name", "split_strategy"], ascending=[True] * 5, kind="mergesort").reset_index(drop=True)

    records: list[ModelRecord] = []
    for (task_name, model_family), frame in combined.groupby(["task_name", "model_family"], dropna=False, sort=True):
        primary = frame.head(1)
        supporting = frame.iloc[1 : 1 + (cfg.max_supporting_models_per_family if cfg.include_supporting_models else 0)]
        selected_frame = pd.concat([primary, supporting], ignore_index=True)
        for rank, row in enumerate(selected_frame.itertuples(index=False), start=1):
            row_series = pd.Series(row._asdict())
            artifact_path = discover_model_artifact(row_series, model_family, cfg)
            if artifact_path is None or not artifact_path.exists():
                raise FileNotFoundError(
                    f"Unable to resolve model artifact for family={model_family}, task={task_name}, model={row_series.get('model_name')}."
                )
            records.append(
                ModelRecord(
                    model_family=model_family,
                    model_name=str(row_series.get("model_name")),
                    task_name=str(task_name),
                    split_strategy=str(row_series.get("split_strategy", "")),
                    artifact_path=artifact_path,
                    source_step={"classical": "07", "deep": "08", "causal": "09"}[model_family],
                    selected_as_best_flag=(rank == 1),
                    selection_criterion=str(row_series.get("selection_metric") or row_series.get("source_table") or "configured_best_model_selection"),
                    split_strategy_used=str(row_series.get("split_strategy", "")),
                    target_label=None if pd.isna(row_series.get("target_label")) else str(row_series.get("target_label")),
                    target_chembl_id=None if pd.isna(row_series.get("target_chembl_id")) else str(row_series.get("target_chembl_id")),
                    ablation_name=None if pd.isna(row_series.get("ablation_name")) else str(row_series.get("ablation_name")),
                    supporting_rank=rank,
                )
            )
    return sorted(records, key=lambda rec: (rec.model_family, rec.task_name, rec.supporting_rank, rec.model_name, str(rec.artifact_path)))


def build_target_metadata_map(cfg: AppConfig) -> dict[str, str]:
    mapping: dict[str, str] = {}
    sources = [
        cfg.input_best_models_by_task_path,
        cfg.input_best_models_by_split_strategy_path,
        cfg.project_root / TARGET_ANNOTATION_FALLBACK_PATH,
    ]
    for path in sources:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            logging.warning("Unable to read target metadata source %s: %s", path, exc)
            continue
        if "target_chembl_id" not in frame.columns or "target_name" not in frame.columns:
            continue
        pairs = (
            frame[["target_chembl_id", "target_name"]]
            .dropna()
            .astype(str)
            .assign(target_chembl_id=lambda x: x["target_chembl_id"].str.strip(), target_name=lambda x: x["target_name"].str.strip())
        )
        pairs = pairs[(pairs["target_chembl_id"] != "") & (pairs["target_name"] != "")]
        for row in pairs.drop_duplicates().itertuples(index=False):
            mapping[str(row.target_chembl_id)] = str(row.target_name)
    logging.info("Loaded target metadata map entries: %s", len(mapping))
    return mapping


def build_target_frame(
    base_df: pd.DataFrame, cfg: AppConfig, task_name: str, model_record: ModelRecord, target_name_map: dict[str, str] | None = None
) -> pd.DataFrame:
    if task_name == "pairwise_selectivity":
        raise NotImplementedError("Pairwise selectivity screening requires explicit pair mapping metadata.")
    if cfg.target_selection_mode != "explicit_list":
        raise ValueError(f"Unsupported target_selection_mode for script_13c: {cfg.target_selection_mode}")
    if model_record.target_chembl_id:
        targets = [model_record.target_chembl_id]
    else:
        targets = list(cfg.target_chembl_ids)
    expanded: list[pd.DataFrame] = []
    target_name_map = target_name_map or {}
    for target in targets:
        frame = base_df.copy()
        frame["target_chembl_id"] = target
        frame["target_name"] = target_name_map.get(str(target), str(target))
        expanded.append(frame)
    combined = pd.concat(expanded, ignore_index=True) if expanded else pd.DataFrame(columns=base_df.columns)
    if "target_name" not in combined.columns:
        combined["target_name"] = combined.get("target_chembl_id", "").astype(str)
    return combined


def load_pickle_artifact(path: Path) -> Any:
    loaders = [joblib.load, lambda p: pickle.loads(Path(p).read_bytes())]
    last_exc: Exception | None = None
    for loader in loaders:
        try:
            return loader(path)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Failed to load pickled model artifact {path}: {last_exc}")


def _load_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch is required for deep/causal screening inference but could not be imported: {exc}") from exc
    return torch


def _flatten_metadata(payload: Any, sink: dict[str, Any]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            sink[str(key)] = value


def _read_metadata_file(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".json":
            parsed = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() in {".yaml", ".yml"}:
            parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            return {}
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _discover_metadata_near_artifact(path: Path, max_parent_levels: int = 3) -> dict[str, Any]:
    discovered: dict[str, Any] = {"metadata_sources": []}
    search_dirs = [path.parent]
    current = path.parent
    for _ in range(max_parent_levels):
        current = current.parent
        search_dirs.append(current)
    for directory in search_dirs:
        for candidate_name in METADATA_CANDIDATE_NAMES:
            candidate = directory / candidate_name
            if candidate.exists():
                payload = _read_metadata_file(candidate)
                if payload:
                    _flatten_metadata(payload, discovered)
                    discovered["metadata_sources"].append(str(candidate))
        for candidate in sorted(directory.glob("*.json")) + sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
            if str(candidate) in discovered["metadata_sources"]:
                continue
            payload = _read_metadata_file(candidate)
            if payload:
                _flatten_metadata(payload, discovered)
                discovered["metadata_sources"].append(str(candidate))
    return discovered


def _unwrap_state_dict(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    values = list(payload.values())
    if values and all(hasattr(v, "shape") for v in values):
        return payload
    return None


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "enabled"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _extract_inference_mode(metadata: dict[str, Any]) -> str:
    for key in ("inference_mode", "runtime_inference_mode", "screening_inference_mode"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return ""


def _supports_flat_numeric_tensor_input(metadata: dict[str, Any]) -> bool:
    for key in ("supports_flat_numeric_tensor", "supports_flat_table_tensor", "allow_flat_numeric_tensor"):
        if key in metadata and _is_truthy(metadata.get(key)):
            return True
    return False


def _is_explicitly_screening_ready(metadata: dict[str, Any]) -> bool:
    for key in ("screening_ready", "is_screening_ready", "inference_ready_for_screening"):
        if key in metadata and _is_truthy(metadata.get(key)):
            return True
    return False


def _build_failed_rows_for_scoring_exception(record: ModelRecord, ready: pd.DataFrame, reason: str) -> pd.DataFrame:
    failed_rows = ready[
        ["screening_compound_id", "standardized_smiles"]
        + [c for c in ["source_library_name", "target_chembl_id", "target_name"] if c in ready.columns]
    ].copy()
    failed_rows["failure_reason"] = reason
    failed_rows["model_family"] = record.model_family
    failed_rows["model_name"] = record.model_name
    failed_rows["task_name"] = record.task_name
    return failed_rows


def _load_torch_module_or_bundle(path: Path, torch: Any, record: ModelRecord) -> tuple[Any, dict[str, Any]]:
    bundle_path = path.with_name("inference_bundle.pt")
    metadata = _discover_metadata_near_artifact(path)
    metadata["artifact_path"] = str(path)
    if bundle_path.exists():
        payload = torch.load(bundle_path, map_location="cpu")
        if isinstance(payload, dict) and "model" in payload:
            metadata.update(payload.get("metadata", {}))
            metadata["artifact_kind"] = "inference_bundle"
            return payload["model"], metadata
    payload = torch.load(path, map_location="cpu")
    if hasattr(payload, "eval") and hasattr(payload, "forward"):
        if not _is_explicitly_screening_ready(metadata):
            raise RuntimeError("direct_module_not_screening_ready")
        metadata["artifact_kind"] = "direct_module"
        return payload, metadata
    if isinstance(payload, dict) and "model" in payload and hasattr(payload["model"], "eval"):
        metadata.update(payload.get("metadata", {}))
        if not _is_explicitly_screening_ready(metadata):
            raise RuntimeError("direct_module_not_screening_ready")
        metadata["artifact_kind"] = "dict_with_model"
        return payload["model"], metadata
    if _unwrap_state_dict(payload) is None:
        raise RuntimeError(
            f"Torch artifact {path} is neither direct module, dict-with-model, inference bundle, nor raw state_dict."
        )
    metadata["artifact_kind"] = "raw_state_dict"
    if record.model_family in {"deep", "causal"}:
        raise RuntimeError(
            "raw_graph_checkpoint_without_inference_bundle::"
            "Graph-based deep/causal checkpoint requires original architecture and graph-aware screening inference bundle; flat tensor fallback is not supported."
        )
    raise RuntimeError(f"Raw state_dict unsupported for model family `{record.model_family}` at {path}.")


def _extract_serialized_sklearn_version(artifact: Any) -> str | None:
    if isinstance(artifact, dict):
        for key in ("sklearn_version", "scikit_learn_version", "scikit-learn-version"):
            value = artifact.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        metadata = artifact.get("metadata")
        if isinstance(metadata, dict):
            for key in ("sklearn_version", "scikit_learn_version", "scikit-learn-version"):
                value = metadata.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
    model = artifact.get("model") if isinstance(artifact, dict) else None
    version = getattr(model, "_sklearn_version", None)
    if version is not None and str(version).strip():
        return str(version).strip()
    return None


def _load_script_module(script_path: Path, module_tag: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_tag, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_nearest_config_yaml(path: Path) -> dict[str, Any]:
    for candidate in [path.parent / "config_used.yaml", path.parent / "config.yaml", Path("config.yaml"), Path("Kinase_Causal_QSAR/config.yaml")]:
        if candidate.exists():
            return load_yaml(candidate.resolve())
    raise RuntimeError("missing_config_yaml_for_reconstruction")


def _resolve_runtime_script_path(cfg: AppConfig, script_name: str) -> Path:
    direct_candidates = [
        cfg.project_root / "scripts" / script_name,
        cfg.project_root / "Kinase_Causal_QSAR" / "scripts" / script_name,
        cfg.project_root / script_name,
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            logging.info("Resolved runtime helper script `%s` at %s", script_name, candidate)
            return candidate

    recursive_hits = sorted({path.resolve() for path in cfg.project_root.rglob(script_name)})
    if recursive_hits:
        logging.info("Resolved runtime helper script `%s` via recursive search at %s", script_name, recursive_hits[0])
        return recursive_hits[0]
    raise ReconstructionError("missing_runtime_script", "script import", context=script_name)


def _summarize_traceback(exc: Exception, max_lines: int = 8) -> str:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    if not tb:
        return f"{type(exc).__name__}: {exc}"
    merged = "".join(tb).strip().splitlines()
    return " | ".join(merged[-max_lines:])


def _infer_step08_architecture(model_name: str, state_dict: dict[str, Any]) -> str:
    known = {"gin", "gcn", "gat", "mpnn"}
    lowered_name = str(model_name).strip().lower()
    if lowered_name in known:
        return lowered_name
    keys = set(state_dict)
    if any(".att_src" in key or ".att_dst" in key for key in keys):
        return "gat"
    if any(".nn.0.weight" in key for key in keys):
        if any("edge" in key.lower() for key in keys):
            return "mpnn"
        return "gin"
    if any("edge_network" in key.lower() or "message_mlp" in key.lower() for key in keys):
        return "mpnn"
    if any(".lin.weight" in key for key in keys):
        return "gcn"
    raise ReconstructionError(
        "step08_unmatched_checkpoint_architecture",
        "model factory lookup",
        context=f"model_name={model_name}; sample_keys={sorted(list(keys))[:12]}",
    )


def _infer_step08_dims_from_state_dict(state_dict: dict[str, Any], architecture: str) -> tuple[int, int]:
    def _find_2d_weight(*patterns: str) -> Any:
        for key, value in state_dict.items():
            if not hasattr(value, "shape") or len(getattr(value, "shape", ())) != 2:
                continue
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in patterns):
                return value
        return None

    node_tensor = _find_2d_weight("input_proj.weight", "node_encoder", "conv1.nn.0.weight", "conv1.lin.weight", "conv1.weight")
    if node_tensor is None:
        node_tensor = next(
            (
                value
                for key, value in state_dict.items()
                if hasattr(value, "shape") and len(getattr(value, "shape", ())) == 2 and "weight" in key.lower()
            ),
            None,
        )
    if node_tensor is None:
        raise ReconstructionError("step08_missing_node_projection_weight", "checkpoint read")
    node_dim = int(node_tensor.shape[1])

    edge_dim = 0
    if architecture == "mpnn":
        edge_tensor = _find_2d_weight("edge", "bond", "message")
        if edge_tensor is None:
            edge_dim = node_dim
        else:
            edge_dim = int(edge_tensor.shape[1])
    return node_dim, edge_dim


def _extract_state_dict_from_checkpoint(payload: Any) -> tuple[dict[str, Any] | None, str]:
    if hasattr(payload, "eval") and hasattr(payload, "forward"):
        return None, "full_module"
    if isinstance(payload, dict) and "model" in payload and hasattr(payload["model"], "eval"):
        return None, "dict_with_model"
    state_dict = _unwrap_state_dict(payload)
    if state_dict is not None:
        return state_dict, "raw_state_dict"
    return None, "unknown"


def _discover_vocabularies_json(path: Path) -> dict[str, Any] | None:
    for directory in [path.parent, *path.parents]:
        candidate = directory / "vocabularies.json"
        if candidate.exists():
            try:
                parsed = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return None


def _load_target_family_map(cfg: AppConfig) -> dict[str, str]:
    mapping: dict[str, str] = {}
    candidate_paths = [cfg.project_root / "data/processed/chembl_human_kinase_panel_annotated_long.csv"]
    for path in candidate_paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        family_col = next((c for c in ["kinase_family", "kinase_family_label", "target_family", "broad_kinase_family"] if c in frame.columns), None)
        if family_col is None or "target_chembl_id" not in frame.columns:
            continue
        dedup = frame[["target_chembl_id", family_col]].dropna().drop_duplicates()
        for row in dedup.itertuples(index=False):
            mapping[str(row[0])] = str(row[1])
    return mapping


def _build_vocab_from_values(values: Iterable[Any]) -> dict[str, int]:
    tokens = sorted({str(value).strip() for value in values if str(value).strip() and str(value).strip().lower() != "nan"})
    return {token: idx for idx, token in enumerate(tokens)}


def _reconstruct_step09_vocabularies(cfg: AppConfig, app09: Any) -> dict[str, dict[str, int]]:
    vocab: dict[str, dict[str, int]] = {}
    target_values: list[str] = []
    family_values: list[str] = []
    env_values: list[str] = []

    task_tables = [
        getattr(app09, "input_regression_long_path", None),
        getattr(app09, "input_target_vs_panel_path", None),
        getattr(app09, "input_pairwise_selectivity_path", None),
    ]
    for maybe_path in task_tables:
        if maybe_path is None:
            continue
        table_path = Path(maybe_path)
        if not table_path.exists():
            continue
        frame = pd.read_csv(table_path, low_memory=False)
        for col in ("target_chembl_id", "kinase_a_chembl_id", "kinase_b_chembl_id"):
            if col in frame.columns:
                target_values.extend(frame[col].dropna().astype(str).tolist())
        for col in ("kinase_family", "target_family", "kinase_family_label", "broad_kinase_family"):
            if col in frame.columns:
                family_values.extend(frame[col].dropna().astype(str).tolist())
        for cols in getattr(app09, "environment_columns", {}).values():
            for col in cols:
                if col in frame.columns:
                    env_values.extend(frame[col].dropna().astype(str).tolist())

    annotated_path = cfg.project_root / "data/processed/chembl_human_kinase_panel_annotated_long.csv"
    if annotated_path.exists():
        annotated = pd.read_csv(annotated_path, low_memory=False)
        for col in ("kinase_family", "target_family", "kinase_family_label", "broad_kinase_family"):
            if col in annotated.columns:
                family_values.extend(annotated[col].dropna().astype(str).tolist())
        for cols in getattr(app09, "environment_columns", {}).values():
            for col in cols:
                if col in annotated.columns:
                    env_values.extend(annotated[col].dropna().astype(str).tolist())

    header = pd.read_csv(cfg.input_environment_feature_path, nrows=0).columns.tolist()
    env_feature = pd.read_csv(cfg.input_environment_feature_path, low_memory=False)
    for cols in getattr(app09, "environment_columns", {}).values():
        for col in cols:
            if col in header:
                env_values.extend(env_feature[col].dropna().astype(str).tolist())

    vocab["target_vocab"] = _build_vocab_from_values(target_values)
    vocab["family_vocab"] = _build_vocab_from_values(family_values)
    vocab["environment_vocab"] = _build_vocab_from_values(env_values)
    return vocab


def load_existing_classical_runtime(record: ModelRecord) -> dict[str, Any]:
    artifact = load_pickle_artifact(record.artifact_path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Classical artifact must contain `model`: {record.artifact_path}")
    feature_columns = artifact.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(f"Classical artifact missing non-empty `feature_columns`: {record.artifact_path}")
    return {"runtime_mode": "classical_tabular", "model": artifact["model"], "feature_columns": feature_columns, "metadata": {"serialized_sklearn_version": _extract_serialized_sklearn_version(artifact)}}


def load_existing_deep_runtime(record: ModelRecord, cfg: AppConfig) -> dict[str, Any]:
    try:
        raw_cfg = _read_nearest_config_yaml(record.artifact_path)
    except Exception as exc:
        raise ReconstructionError("missing_config_yaml_for_reconstruction", "config discovery", cause=exc) from exc
    try:
        script08_path = _resolve_runtime_script_path(cfg, "08_train_graph_and_deep_baseline_models.py")
        step08 = _load_script_module(script08_path, "step08_runtime")
    except Exception as exc:
        if isinstance(exc, ReconstructionError):
            raise
        raise ReconstructionError("step08_script_import_failed", "script import", cause=exc) from exc
    if not hasattr(step08, "import_runtime_dependencies"):
        raise ReconstructionError("step08_missing_import_runtime_dependencies", "dependency import")
    if not hasattr(step08, "make_model_factory"):
        raise ReconstructionError("step08_missing_make_model_factory", "model factory lookup")
    try:
        app08 = step08.AppConfig.from_dict(raw_cfg, script08_path.parent.parent)
    except Exception as exc:
        raise ReconstructionError("step08_config_parse_failed", "config discovery", cause=exc) from exc
    try:
        deps = step08.import_runtime_dependencies()
        torch = deps["torch"]
    except Exception as exc:
        raise ReconstructionError("step08_dependency_import_failed", "dependency import", cause=exc) from exc
    try:
        payload = torch.load(record.artifact_path, map_location="cpu")
    except Exception as exc:
        raise ReconstructionError("step08_checkpoint_read_failed", "checkpoint read", cause=exc) from exc
    state_dict, checkpoint_structure = _extract_state_dict_from_checkpoint(payload)
    if checkpoint_structure == "full_module":
        model = payload
        model.eval()
        return {
            "runtime_mode": "graph_batch",
            "torch": torch,
            "deps": deps,
            "model": model,
            "node_features": app08.node_features,
            "edge_features": app08.edge_features,
            "target_map": {},
            "family_map": {},
            "target_family_map": _load_target_family_map(cfg),
            "metadata": {"reconstructed_from": "step08_full_module_checkpoint", "script_path": str(script08_path)},
        }
    if checkpoint_structure == "dict_with_model":
        model = payload["model"]
        model.eval()
        return {
            "runtime_mode": "graph_batch",
            "torch": torch,
            "deps": deps,
            "model": model,
            "node_features": app08.node_features,
            "edge_features": app08.edge_features,
            "target_map": {},
            "family_map": {},
            "target_family_map": _load_target_family_map(cfg),
            "metadata": {"reconstructed_from": "step08_dict_with_model_checkpoint", "script_path": str(script08_path)},
        }
    if state_dict is None:
        raise ReconstructionError("incompatible_checkpoint_keys_for_step08", "state_dict unwrap")
    architecture = _infer_step08_architecture(record.model_name, state_dict)
    node_dim, edge_dim = _infer_step08_dims_from_state_dict(state_dict, architecture)
    target_map: dict[str, int] = {}
    family_map: dict[str, int] = {}
    if app08.sequence_or_target_encoding.use_target_identity_embedding:
        for input_key in ["input_regression_long_path", "input_target_vs_panel_path", "input_pairwise_selectivity_path"]:
            table_path = getattr(app08, input_key, None)
            if table_path and Path(table_path).exists():
                frame = pd.read_csv(table_path, low_memory=False)
                for col in ["target_chembl_id", "kinase_a_chembl_id", "kinase_b_chembl_id"]:
                    if col in frame.columns:
                        target_map.update({value: idx + 1 for idx, value in enumerate(sorted(set(frame[col].dropna().astype(str).tolist())))})
    if app08.sequence_or_target_encoding.use_kinase_family_embedding:
        family_lookup = _load_target_family_map(cfg)
        family_map = {value: idx + 1 for idx, value in enumerate(sorted(set(family_lookup.values())))}
    try:
        model_factory = step08.make_model_factory(
            deps,
            app08,
            {
                "task_name": record.task_name,
                "output_mode": "regression",
                "node_feature_dim": node_dim,
                "edge_feature_dim": edge_dim,
                "target_vocab_size": max(1, len(target_map)),
                "family_vocab_size": max(1, len(family_map)),
                "pair_vocab_size": max(1, len(target_map)),
            },
        )
    except Exception as exc:
        raise ReconstructionError("step08_make_model_factory_failed", "model factory lookup", cause=exc) from exc
    model_ctor = model_factory.get(record.model_name)
    if model_ctor is None:
        raise ReconstructionError("missing_model_class_information", "model factory lookup", context=record.model_name)
    model = model_ctor()
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ReconstructionError("step08_strict_load_state_dict_failed", "strict load_state_dict", cause=exc) from exc
    model.eval()
    return {
        "runtime_mode": "graph_batch",
        "torch": torch,
        "deps": deps,
        "model": model,
        "node_features": app08.node_features,
        "edge_features": app08.edge_features,
        "target_map": target_map,
        "family_map": family_map,
        "target_family_map": _load_target_family_map(cfg),
        "metadata": {
            "reconstructed_from": "step08_checkpoint",
            "checkpoint_structure": checkpoint_structure,
            "architecture": architecture,
            "script_path": str(script08_path),
        },
    }


def load_existing_causal_runtime(record: ModelRecord, cfg: AppConfig) -> dict[str, Any]:
    try:
        raw_cfg = _read_nearest_config_yaml(record.artifact_path)
    except Exception as exc:
        raise ReconstructionError("missing_config_yaml_for_reconstruction", "config discovery", cause=exc) from exc
    try:
        script09_path = _resolve_runtime_script_path(cfg, "09_train_causal_environment_aware_model.py")
        step09 = _load_script_module(script09_path, "step09_runtime")
        app09 = step09.AppConfig.from_dict(raw_cfg, script09_path.parent.parent)
    except Exception as exc:
        if isinstance(exc, ReconstructionError):
            raise
        raise ReconstructionError("step09_script_import_failed", "script import", cause=exc) from exc
    if not hasattr(step09, "import_training_dependencies"):
        raise ReconstructionError("step09_missing_import_training_dependencies", "dependency import")
    if not hasattr(step09, "make_model_class"):
        raise ReconstructionError("step09_missing_make_model_class", "model factory lookup")
    try:
        deps = step09.import_training_dependencies()
        torch = deps["torch"]
    except Exception as exc:
        raise ReconstructionError("step09_dependency_import_failed", "dependency import", cause=exc) from exc
    try:
        payload = torch.load(record.artifact_path, map_location="cpu")
    except Exception as exc:
        raise ReconstructionError("step09_checkpoint_read_failed", "checkpoint read", cause=exc) from exc
    state_dict = _unwrap_state_dict(payload)
    if state_dict is None:
        raise ReconstructionError("incompatible_checkpoint_keys", "state_dict unwrap")
    try:
        ModelClass = step09.make_model_class(deps)
    except Exception as exc:
        raise ReconstructionError("step09_make_model_class_failed", "model factory lookup", cause=exc) from exc
    sample_in_dim = int(state_dict["encoder.input_proj.weight"].shape[1]) if "encoder.input_proj.weight" in state_dict else 8
    vocab = _discover_vocabularies_json(record.artifact_path) or {}
    if not vocab.get("target_vocab") or not vocab.get("family_vocab") or not vocab.get("environment_vocab"):
        reconstructed = _reconstruct_step09_vocabularies(cfg, app09)
        for key in ("target_vocab", "family_vocab", "environment_vocab"):
            if not vocab.get(key):
                vocab[key] = reconstructed.get(key, {})
    if not vocab.get("target_vocab") or not vocab.get("environment_vocab"):
        raise ReconstructionError("missing_target_or_environment_vocabulary", "vocab discovery")
    env_vocab = vocab.get("environment_vocab") or {}
    model = ModelClass(
        in_dim=sample_in_dim,
        cfg=app09,
        num_targets=len(vocab.get("target_vocab") or {}),
        num_families=len(vocab.get("family_vocab") or {}),
        num_envs=len(env_vocab),
        task_name=record.task_name,
        task_type="regression",
        grl_lambda=app09.loss_weights.get("environment_adversarial_loss", 0.0),
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ReconstructionError("step09_strict_load_state_dict_failed", "strict load_state_dict", cause=exc) from exc
    model.eval()
    resolved_env_column = None
    try:
        env_header = pd.read_csv(cfg.input_environment_feature_path, nrows=0).columns
    except Exception as exc:
        raise ReconstructionError("environment_feature_header_read_failed", "environment column discovery", cause=exc) from exc
    for candidates in app09.environment_columns.values():
        for col in candidates:
            if col in env_header:
                resolved_env_column = col
                break
        if resolved_env_column:
            break
    if not resolved_env_column:
        candidate_cols = list(getattr(app09, "environment_columns", {}).keys())
        if candidate_cols:
            resolved_env_column = candidate_cols[0]
        else:
            raise ReconstructionError("missing_environment_vocabulary_or_columns", "environment column discovery")
    return {
        "runtime_mode": "graph_batch_with_environment",
        "torch": torch,
        "deps": deps,
        "model": model,
        "vocab": vocab,
        "environment_column": resolved_env_column,
        "target_family_map": _load_target_family_map(cfg),
        "metadata": {"reconstructed_from": "step09_checkpoint", "script_path": str(script09_path)},
    }


def prepare_model_runtime(record: ModelRecord, cfg: AppConfig) -> dict[str, Any]:
    try:
        if record.model_family == "classical":
            return load_existing_classical_runtime(record)
        if record.model_family == "deep":
            return load_existing_deep_runtime(record, cfg=cfg)
        if record.model_family == "causal":
            return load_existing_causal_runtime(record, cfg=cfg)
        raise ValueError(f"Unsupported model family: {record.model_family}")
    except ReconstructionError:
        raise
    except Exception as exc:
        raise ReconstructionError("model_runtime_assembly_failed", "graph runtime assembly", cause=exc) from exc


def preflight_validate_model_runtimes(cfg: AppConfig, model_records: list[ModelRecord]) -> tuple[list[ModelRecord], dict[int, dict[str, Any]], list[ModelLoadFailure]]:
    loadable_records: list[ModelRecord] = []
    runtimes: dict[int, dict[str, Any]] = {}
    skipped: list[ModelLoadFailure] = []
    serialized_sklearn_versions: set[str] = set()
    for record in model_records:
        try:
            runtime = prepare_model_runtime(record, cfg)
            if record.model_family == "classical":
                version = runtime.get("metadata", {}).get("serialized_sklearn_version")
                if version:
                    serialized_sklearn_versions.add(str(version))
            loadable_records.append(record)
            runtimes[len(loadable_records) - 1] = runtime
        except Exception as exc:
            reason = str(exc)
            stage = getattr(exc, "reconstruction_stage", "unknown")
            exception_type = type(exc).__name__
            tb_summary = _summarize_traceback(exc)
            artifact_kind = "unknown"
            if "direct module" in reason:
                artifact_kind = "direct_module"
            elif "dict-with-model" in reason or "dict with model" in reason:
                artifact_kind = "dict_with_model"
            elif "inference bundle" in reason:
                artifact_kind = "inference_bundle"
            elif "raw_graph_checkpoint_without_inference_bundle" in reason:
                artifact_kind = "raw_graph_checkpoint_without_inference_bundle"
            elif "state_dict" in reason:
                artifact_kind = "raw_state_dict_unsupported"
            elif "unsupported_inference_mode" in reason or "missing_supported_inference_mode_metadata" in reason:
                artifact_kind = "unsupported_inference_mode"
            elif "not_explicitly_supported" in reason or "not_screening_ready" in reason:
                artifact_kind = "not_screening_ready_for_screening"
            skipped.append(
                ModelLoadFailure(
                    record=record,
                    reason=reason,
                    artifact_diagnostic=artifact_kind,
                    reconstruction_stage=stage,
                    exception_type=exception_type,
                    traceback_summary=tb_summary,
                )
            )

    logging.info(
        "Preflight model validation complete. loadable=%s skipped=%s total=%s",
        len(loadable_records),
        len(skipped),
        len(model_records),
    )
    for failure in skipped:
        logging.warning(
            "Skipping model for screening safety: family=%s model=%s task=%s artifact=%s "
            "checkpoint_exists=true valid_screening_inference_artifact=false diagnostic=%s "
            "reconstruction_stage=%s exception_type=%s reason=%s traceback_summary=%s",
            failure.record.model_family,
            failure.record.model_name,
            failure.record.task_name,
            failure.record.artifact_path,
            failure.artifact_diagnostic,
            failure.reconstruction_stage,
            failure.exception_type,
            failure.reason,
            failure.traceback_summary,
        )

    requested_families = set(cfg.selected_families_to_score)
    loadable_families = {record.model_family for record in loadable_records}
    for family in ("deep", "causal"):
        if family in requested_families and family not in loadable_families:
            logging.warning(
                "Configured family %s was requested for screening but no model could be reconstructed from saved artifacts.",
                family,
            )
    try:
        import sklearn  # type: ignore

        current_sklearn = str(getattr(sklearn, "__version__", "unknown"))
    except Exception:
        current_sklearn = "unavailable"
    if serialized_sklearn_versions:
        if current_sklearn not in serialized_sklearn_versions:
            logging.warning(
                "scikit-learn version mismatch detected. runtime_version=%s serialized_versions=%s. "
                "Predictions may be unsafe across sklearn versions.",
                current_sklearn,
                sorted(serialized_sklearn_versions),
            )
        else:
            logging.info("scikit-learn version check passed. runtime_version=%s serialized_version=%s", current_sklearn, current_sklearn)
    else:
        logging.warning(
            "Unable to determine serialized sklearn version from classical artifacts. runtime_version=%s. "
            "Predictions may be unsafe across sklearn versions.",
            current_sklearn,
        )

    if skipped and cfg.require_all_models_loadable:
        summary = "; ".join(
            f"{f.record.model_family}/{f.record.model_name} ({f.artifact_diagnostic}): {f.reason}" for f in skipped[:5]
        )
        raise RuntimeError(f"Model preflight failed because require_all_models_loadable=true. Examples: {summary}")
    if skipped and not cfg.skip_unloadable_models and not cfg.require_all_models_loadable:
        summary = "; ".join(
            f"{f.record.model_family}/{f.record.model_name} ({f.artifact_diagnostic}): {f.reason}" for f in skipped[:5]
        )
        raise RuntimeError(f"Model preflight found unloadable models and skip_unloadable_models=false. Examples: {summary}")
    return loadable_records, runtimes, skipped


def validate_chunk_alignment(chunk_idx: int, left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"Chunk {chunk_idx} misalignment: {left_name} rows={len(left)} != {right_name} rows={len(right)}")
    if left.empty:
        return
    left_key = left[["screening_compound_id", "standardized_smiles"]].astype(str)
    right_key = right[["screening_compound_id", "standardized_smiles"]].astype(str)
    mismatch = (left_key != right_key).any(axis=1)
    if bool(mismatch.any()):
        first = int(np.flatnonzero(mismatch.to_numpy())[0])
        raise ValueError(
            f"Chunk {chunk_idx} misalignment between {left_name} and {right_name} at row offset {first}: "
            f"{left_key.iloc[first].to_dict()} != {right_key.iloc[first].to_dict()}"
        )


def iter_screening_chunks(cfg: AppConfig) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    classical_iter = pd.read_csv(cfg.input_classical_feature_path, chunksize=cfg.chunk_size, low_memory=cfg.low_memory)
    graph_iter = pd.read_csv(cfg.input_graph_manifest_path, chunksize=cfg.chunk_size, low_memory=cfg.low_memory)
    env_iter = pd.read_csv(cfg.input_environment_feature_path, chunksize=cfg.chunk_size, low_memory=False)
    chunk_idx = 0
    while True:
        classical_chunk = next(classical_iter, None)
        graph_chunk = next(graph_iter, None)
        env_chunk = next(env_iter, None)
        if classical_chunk is None and graph_chunk is None and env_chunk is None:
            break
        if classical_chunk is None or graph_chunk is None or env_chunk is None:
            raise ValueError(f"Input chunk count mismatch at chunk index {chunk_idx}.")
        if classical_chunk.empty:
            logging.warning("Skipping empty chunk %s", chunk_idx)
            chunk_idx += 1
            continue
        ensure_columns(classical_chunk, REQUIRED_SCREENING_COLUMNS, "classical chunk")
        ensure_columns(graph_chunk, REQUIRED_SCREENING_COLUMNS, "graph chunk")
        ensure_columns(env_chunk, REQUIRED_SCREENING_COLUMNS, "environment chunk")
        validate_chunk_alignment(chunk_idx, classical_chunk, graph_chunk, "classical", "graph")
        validate_chunk_alignment(chunk_idx, classical_chunk, env_chunk, "classical", "environment")
        yield chunk_idx, classical_chunk, graph_chunk, env_chunk
        chunk_idx += 1


def score_classical_chunk(
    record: ModelRecord, runtime: dict[str, Any], feature_df: pd.DataFrame, cfg: AppConfig, target_name_map: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    scoring_df = build_target_frame(feature_df, cfg, record.task_name, record, target_name_map=target_name_map)
    missing_features = sorted(set(runtime["feature_columns"]).difference(scoring_df.columns))
    if missing_features:
        target_conditioned = [col for col in missing_features if "target" in col.lower() or "chembl" in col.lower()]
        available_sample = sorted(scoring_df.columns.tolist())[:20]
        raise ValueError(
            f"Chunk missing classical feature columns for model {record.model_name}: {missing_features[:25]}; "
            f"looks_target_conditioned={bool(target_conditioned)}; "
            f"target_conditioned_missing={target_conditioned[:10]}; available_column_sample={available_sample}"
        )
    matrix = scoring_df[runtime["feature_columns"]]
    invalid_mask = matrix.isna().any(axis=1)
    failed_rows = scoring_df.loc[invalid_mask, ["screening_compound_id", "standardized_smiles"] + [c for c in ["source_library_name", "target_chembl_id", "target_name"] if c in scoring_df.columns]].copy()
    failed_rows["failure_reason"] = "missing_classical_features"
    failed_rows["model_family"] = record.model_family
    failed_rows["model_name"] = record.model_name
    failed_rows["task_name"] = record.task_name

    ready_df = scoring_df.loc[~invalid_mask].copy()
    if ready_df.empty:
        return pd.DataFrame(), failed_rows, {"attempted": int(len(scoring_df)), "scored": 0, "failed": int(len(failed_rows))}
    predictions = np.asarray(runtime["model"].predict(ready_df[runtime["feature_columns"]]), dtype=float)
    output = ready_df[["screening_compound_id", "standardized_smiles"] + [c for c in ["source_library_name", "target_chembl_id", "target_name"] if c in ready_df.columns]].copy()
    output["task_name"] = record.task_name
    output["model_family"] = record.model_family
    output["model_name"] = record.model_name
    output["model_artifact_path"] = str(record.artifact_path)
    output["split_strategy_used_to_select_model"] = record.split_strategy_used
    output["predicted_value"] = predictions
    output["predicted_value_type"] = PREDICTED_VALUE_TYPES[record.task_name]
    output["score_type"] = output["predicted_value_type"]
    output["target_label"] = record.target_label
    output["selection_criterion"] = record.selection_criterion
    output["environment_conditioning_metadata"] = ""
    return output, failed_rows, {"attempted": int(len(scoring_df)), "scored": int(len(output)), "failed": int(len(failed_rows))}


def score_torch_chunk(
    record: ModelRecord,
    runtime: dict[str, Any],
    graph_df: pd.DataFrame,
    environment_df: pd.DataFrame,
    cfg: AppConfig,
    target_name_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    base = build_target_frame(graph_df, cfg, record.task_name, record, target_name_map=target_name_map)
    merged = base.merge(environment_df, on=["screening_compound_id", "standardized_smiles"], how="left", suffixes=("", "_env"))
    torch = runtime["torch"]
    failed_records: list[dict[str, Any]] = []
    preds: list[float] = []
    kept_rows: list[dict[str, Any]] = []

    if runtime["runtime_mode"] == "graph_batch":
        Data = runtime["deps"]["Data"]
        DataLoader = runtime["deps"]["DataLoader"]
        Chem = runtime["deps"]["Chem"]
        rdchem = runtime["deps"]["rdchem"]
        for row in merged.itertuples(index=False):
            mol = Chem.MolFromSmiles(str(row.standardized_smiles))
            if mol is None or mol.GetNumAtoms() == 0:
                failed_records.append({"screening_compound_id": row.screening_compound_id, "standardized_smiles": row.standardized_smiles, "failure_reason": "missing_graph_feature_metadata"})
                continue
            atom_rows: list[list[float]] = []
            for atom in mol.GetAtoms():
                feats: list[float] = []
                if runtime["node_features"].get("use_atom_type", True):
                    symbols = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
                    feats.extend([1.0 if atom.GetSymbol() == s else 0.0 for s in symbols] + [0.0 if atom.GetSymbol() in symbols else 1.0])
                if runtime["node_features"].get("use_degree", True):
                    feats.append(float(atom.GetDegree()))
                if runtime["node_features"].get("use_formal_charge", True):
                    feats.append(float(atom.GetFormalCharge()))
                if runtime["node_features"].get("use_hybridization", True):
                    hybs = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3]
                    feats.extend([1.0 if atom.GetHybridization() == h else 0.0 for h in hybs] + [0.0 if atom.GetHybridization() in hybs else 1.0])
                if runtime["node_features"].get("use_aromaticity", True):
                    feats.append(float(atom.GetIsAromatic()))
                if runtime["node_features"].get("use_num_hs", True):
                    feats.append(float(atom.GetTotalNumHs()))
                if runtime["node_features"].get("use_chirality", True):
                    feats.extend([
                        1.0 if atom.GetChiralTag() == rdchem.ChiralType.CHI_UNSPECIFIED else 0.0,
                        1.0 if atom.GetChiralTag() == rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0.0,
                        1.0 if atom.GetChiralTag() == rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0.0,
                    ])
                atom_rows.append(feats)
            edge_idx: list[list[int]] = []
            edge_attr: list[list[float]] = []
            for bond in mol.GetBonds():
                bfeats: list[float] = []
                if runtime["edge_features"].get("use_bond_type", True):
                    types = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]
                    bfeats.extend([1.0 if bond.GetBondType() == t else 0.0 for t in types] + [0.0 if bond.GetBondType() in types else 1.0])
                if runtime["edge_features"].get("use_conjugation", True):
                    bfeats.append(float(bond.GetIsConjugated()))
                if runtime["edge_features"].get("use_ring_status", True):
                    bfeats.append(float(bond.IsInRing()))
                if runtime["edge_features"].get("use_stereo", True):
                    stereo = [rdchem.BondStereo.STEREONONE, rdchem.BondStereo.STEREOANY, rdchem.BondStereo.STEREOZ, rdchem.BondStereo.STEREOE]
                    bfeats.extend([1.0 if bond.GetStereo() == t else 0.0 for t in stereo] + [0.0 if bond.GetStereo() in stereo else 1.0])
                i, j = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
                edge_idx.extend([[i, j], [j, i]])
                edge_attr.extend([bfeats, bfeats])
            datum = Data(
                x=torch.tensor(atom_rows, dtype=torch.float32),
                edge_index=torch.tensor(edge_idx, dtype=torch.long).t().contiguous() if edge_idx else torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.empty((0, 0), dtype=torch.float32),
            )
            target_id = runtime["target_map"].get(str(getattr(row, "target_chembl_id", "")), 0)
            family_name = runtime["target_family_map"].get(str(getattr(row, "target_chembl_id", "")), "<UNK>")
            family_id = runtime["family_map"].get(str(family_name), 0)
            datum.target_id = torch.tensor([target_id], dtype=torch.long)
            datum.target_family_id = torch.tensor([family_id], dtype=torch.long)
            datum.kinase_a_id = torch.tensor([target_id], dtype=torch.long)
            datum.kinase_b_id = torch.tensor([target_id], dtype=torch.long)
            datum._meta = {"screening_compound_id": row.screening_compound_id, "standardized_smiles": row.standardized_smiles, "source_library_name": getattr(row, "source_library_name", ""), "target_chembl_id": getattr(row, "target_chembl_id", ""), "target_name": getattr(row, "target_name", "")}
            kept_rows.append(datum._meta)
            preds.append(np.nan)
            if "_data" not in runtime:
                runtime["_data"] = []
            runtime["_data"].append(datum)
        if runtime.get("_data"):
            loader = DataLoader(runtime["_data"], batch_size=max(1, cfg.batch_size), shuffle=False, num_workers=0)
            offset = 0
            with torch.no_grad():
                for batch in loader:
                    out = runtime["model"](batch)
                    out = out[0] if isinstance(out, tuple) else out
                    arr = np.asarray(out.detach().cpu().numpy()).reshape(-1)
                    for idx, value in enumerate(arr):
                        preds[offset + idx] = float(value)
                    offset += len(arr)
        runtime["_data"] = []
    elif runtime["runtime_mode"] == "graph_batch_with_environment":
        deps = runtime["deps"]
        Batch = deps["Batch"]
        Chem = deps["Chem"]
        for row in merged.itertuples(index=False):
            mol = Chem.MolFromSmiles(str(row.standardized_smiles))
            if mol is None:
                failed_records.append({"screening_compound_id": row.screening_compound_id, "standardized_smiles": row.standardized_smiles, "failure_reason": "missing_graph_feature_metadata"})
                continue
            nodes = [[a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(), int(a.GetIsAromatic()), a.GetTotalNumHs(), int(a.IsInRing()), a.GetImplicitValence(), a.GetMass() / 100.0] for a in mol.GetAtoms()]
            if not nodes:
                failed_records.append({"screening_compound_id": row.screening_compound_id, "standardized_smiles": row.standardized_smiles, "failure_reason": "missing_graph_feature_metadata"})
                continue
            edges, attrs = [], []
            for b in mol.GetBonds():
                bf = [float(b.GetBondType() == b.GetBondType().SINGLE), float(b.GetBondType() == b.GetBondType().DOUBLE), float(b.GetBondType() == b.GetBondType().TRIPLE), float(b.GetBondType() == b.GetBondType().AROMATIC), float(b.GetIsConjugated()), float(b.IsInRing())]
                i, j = int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())
                edges.extend([[i, j], [j, i]])
                attrs.extend([bf, bf])
            graph = deps["Data"](
                x=torch.tensor(nodes, dtype=torch.float32),
                edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.tensor(attrs, dtype=torch.float32) if attrs else torch.empty((0, 6), dtype=torch.float32),
            )
            target_vocab = runtime["vocab"].get("target_vocab") or {}
            family_vocab = runtime["vocab"].get("family_vocab") or {}
            env_vocab = runtime["vocab"].get("environment_vocab") or {}
            target_token = str(getattr(row, "target_chembl_id", "<UNK>"))
            family_token = runtime["target_family_map"].get(target_token, "<UNK>")
            env_token = str(getattr(row, runtime["environment_column"], "<UNK>"))
            batch = {
                "graph": Batch.from_data_list([graph]),
                "label": torch.tensor([0.0], dtype=torch.float32),
                "target_id": torch.tensor([target_vocab.get(target_token, target_vocab.get("<UNK>", 0))], dtype=torch.long),
                "target_id_a": torch.tensor([target_vocab.get(target_token, target_vocab.get("<UNK>", 0))], dtype=torch.long),
                "target_id_b": torch.tensor([target_vocab.get(target_token, target_vocab.get("<UNK>", 0))], dtype=torch.long),
                "family_id": torch.tensor([family_vocab.get(family_token, family_vocab.get("<UNK>", 0))], dtype=torch.long),
                "family_id_a": torch.tensor([family_vocab.get(family_token, family_vocab.get("<UNK>", 0))], dtype=torch.long),
                "family_id_b": torch.tensor([family_vocab.get(family_token, family_vocab.get("<UNK>", 0))], dtype=torch.long),
                "environment_id": torch.tensor([env_vocab.get(env_token, env_vocab.get("<UNK>", 0))], dtype=torch.long),
                "activity_cliff_flag": torch.tensor([0], dtype=torch.long),
            }
            with torch.no_grad():
                out = runtime["model"](batch)
            preds.append(float(np.asarray(out["prediction"].detach().cpu().numpy()).reshape(-1)[0]))
            kept_rows.append({"screening_compound_id": row.screening_compound_id, "standardized_smiles": row.standardized_smiles, "source_library_name": getattr(row, "source_library_name", ""), "target_chembl_id": getattr(row, "target_chembl_id", ""), "target_name": getattr(row, "target_name", "")})
    else:
        raise ValueError(f"Unsupported runtime mode: {runtime.get('runtime_mode')}")

    failed_rows = pd.DataFrame(failed_records)
    if not failed_rows.empty:
        failed_rows["model_family"] = record.model_family
        failed_rows["model_name"] = record.model_name
        failed_rows["task_name"] = record.task_name
    if not kept_rows:
        return pd.DataFrame(), failed_rows, {"attempted": int(len(merged)), "scored": 0, "failed": int(len(failed_rows))}
    output = pd.DataFrame(kept_rows)
    output["task_name"] = record.task_name
    output["model_family"] = record.model_family
    output["model_name"] = record.model_name
    output["model_artifact_path"] = str(record.artifact_path)
    output["split_strategy_used_to_select_model"] = record.split_strategy_used
    output["predicted_value"] = np.asarray(preds, dtype=float)
    output["predicted_value_type"] = PREDICTED_VALUE_TYPES[record.task_name]
    output["score_type"] = output["predicted_value_type"]
    output["target_label"] = record.target_label
    output["selection_criterion"] = record.selection_criterion
    output["environment_conditioning_metadata"] = json.dumps(runtime["metadata"], sort_keys=True) if runtime.get("metadata") else ""
    return output, failed_rows, {"attempted": int(len(merged)), "scored": int(len(output)), "failed": int(len(failed_rows))}


def to_unified_columns(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    for col in UNIFIED_COLUMN_ORDER:
        if col not in working.columns:
            working[col] = ""
    remainder = [col for col in working.columns if col not in UNIFIED_COLUMN_ORDER]
    return working[UNIFIED_COLUMN_ORDER + remainder]


def prepare_output_paths(cfg: AppConfig) -> dict[str, Path]:
    paths = {
        "classical_scores": cfg.output_classical_scores_path,
        "deep_scores": cfg.output_deep_scores_path,
        "causal_scores": cfg.output_causal_scores_path,
        "unified_scores": cfg.output_unified_scores_path,
        "failed_classical": cfg.output_scoring_root / "failed_classical_scoring_rows.csv",
        "failed_deep": cfg.output_scoring_root / "failed_deep_scoring_rows.csv",
        "failed_causal": cfg.output_scoring_root / "failed_causal_scoring_rows.csv",
        "qc_summary": cfg.output_scoring_root / "screening_scoring_qc_summary.csv",
        "metadata": cfg.output_scoring_root / "screening_model_metadata.csv",
        "wide": cfg.output_scoring_root / "unified_screening_scores_wide.csv",
        "manifest": cfg.output_manifest_path,
        "report": cfg.output_report_path,
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not cfg.overwrite_existing_outputs:
        raise FileExistsError("Step-13C output files already exist. Set script_13c.overwrite_existing_outputs=true to replace them.")
    if cfg.overwrite_existing_outputs:
        for path in existing:
            path.unlink()
            logging.info("Removed existing output file before run: %s", path)
    return paths


def build_model_metadata_table(records: list[ModelRecord]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "model_family": record.model_family,
            "model_name": record.model_name,
            "task_name": record.task_name,
            "source_step": record.source_step,
            "artifact_path": str(record.artifact_path),
            "selected_as_best_flag": record.selected_as_best_flag,
            "selection_criterion": record.selection_criterion,
            "split_strategy_used": record.split_strategy_used,
            "target_label": record.target_label,
            "target_chembl_id": record.target_chembl_id,
            "ablation_name": record.ablation_name,
            "supporting_rank": record.supporting_rank,
            "notes": record.notes,
        }
        for record in records
    ]).sort_values(["model_family", "task_name", "supporting_rank", "model_name"], kind="mergesort").reset_index(drop=True)


def run_streaming_scoring(cfg: AppConfig, model_records: list[ModelRecord], config_snapshot_path: Path | None) -> None:
    output_paths = prepare_output_paths(cfg)
    target_name_map = build_target_metadata_map(cfg)
    model_records, runtimes, skipped_models = preflight_validate_model_runtimes(cfg, model_records)
    if not model_records:
        raise RuntimeError("No loadable models remained after preflight validation.")

    wrote_header: dict[Path, bool] = defaultdict(bool)
    buffered_frames: dict[Path, list[pd.DataFrame]] = defaultdict(list)

    def flush_path(path: Path) -> None:
        if not buffered_frames[path]:
            return
        non_empty_frames = [frame for frame in buffered_frames[path] if frame is not None and not frame.empty]
        buffered_frames[path] = []
        if not non_empty_frames:
            logging.info("Skipping flush for %s because buffered frames were all empty", path)
            return
        combined = pd.concat(non_empty_frames, ignore_index=True)
        wrote_header[path] = append_dataframe(combined, path, wrote_header[path])

    def flush_all() -> None:
        for path in list(buffered_frames):
            try:
                flush_path(path)
            except Exception:
                logging.exception("Failed to flush buffered frames for %s", path)

    counters: dict[int, dict[str, int]] = {idx: {"attempted": 0, "scored": 0, "failed": 0, "chunks": 0} for idx in range(len(model_records))}
    family_scored_compounds: dict[str, set[str]] = {family: set() for family in SUPPORTED_MODEL_FAMILIES}
    task_coverage: dict[tuple[str, str], int] = defaultdict(int)
    target_coverage: dict[tuple[str, str, str], int] = defaultdict(int)
    total_rows = 0

    for chunk_idx, classical_chunk, graph_chunk, env_chunk in iter_screening_chunks(cfg):
        total_rows += len(classical_chunk)
        logging.info("Processing chunk %s | rows=%s | cumulative_rows=%s", chunk_idx, len(classical_chunk), total_rows)
        for idx, record in enumerate(model_records):
            logging.info("Scoring chunk %s with %s/%s", chunk_idx, record.model_family, record.model_name)
            runtime = runtimes[idx]
            if record.model_family == "classical":
                pred, failed, counts = score_classical_chunk(record, runtime, classical_chunk, cfg, target_name_map)
            else:
                pred, failed, counts = score_torch_chunk(record, runtime, graph_chunk, env_chunk, cfg, target_name_map)

            counters[idx]["attempted"] += counts["attempted"]
            counters[idx]["scored"] += counts["scored"]
            counters[idx]["failed"] += counts["failed"]
            counters[idx]["chunks"] += 1

            family_output = output_paths[f"{record.model_family}_scores"]
            failed_output = output_paths[f"failed_{record.model_family}"]
            if cfg.save_raw_predictions:
                buffered_frames[family_output].append(pred)
                if not pred.empty:
                    logging.info("Buffered %s %s predictions for %s", len(pred), record.model_family, family_output)
            if cfg.save_failed_rows:
                buffered_frames[failed_output].append(failed)

            unified_chunk = to_unified_columns(pred)
            buffered_frames[output_paths["unified_scores"]].append(unified_chunk)
            if not unified_chunk.empty:
                family_scored_compounds[record.model_family].update(unified_chunk["screening_compound_id"].astype(str).tolist())
                task_coverage[(record.task_name, record.model_family)] += len(unified_chunk)
                if "target_chembl_id" in unified_chunk.columns:
                    for key, n in unified_chunk.groupby(["target_chembl_id", "task_name", "model_family"], dropna=False).size().items():
                        target_coverage[(str(key[0]), str(key[1]), str(key[2]))] += int(n)
        if (chunk_idx + 1) % cfg.write_every_n_chunks == 0:
            flush_all()

    flush_all()

    qc_rows = []
    for idx, record in enumerate(model_records):
        c = counters[idx]
        qc_rows.append(
            {
                "model_family": record.model_family,
                "model_name": record.model_name,
                "task_name": record.task_name,
                "target_chembl_id": record.target_chembl_id,
                "number_of_compounds_attempted": c["attempted"],
                "number_of_compounds_scored_successfully": c["scored"],
                "number_of_failed_rows": c["failed"],
                "number_of_processed_chunks": c["chunks"],
                "notes": record.notes,
            }
        )
    qc_df = pd.DataFrame(qc_rows).sort_values(["model_family", "task_name", "model_name"], kind="mergesort").reset_index(drop=True)
    if skipped_models:
        skipped_qc = pd.DataFrame(
            [
                {
                    "model_family": entry.record.model_family,
                    "model_name": entry.record.model_name,
                    "task_name": entry.record.task_name,
                    "target_chembl_id": entry.record.target_chembl_id,
                    "number_of_compounds_attempted": 0,
                    "number_of_compounds_scored_successfully": 0,
                    "number_of_failed_rows": 0,
                    "number_of_processed_chunks": 0,
                    "notes": (
                        f"SKIPPED_UNLOADABLE_MODEL::{entry.artifact_diagnostic}::{entry.reconstruction_stage}"
                        f"::{entry.exception_type}::{entry.reason}"
                    ),
                }
                for entry in skipped_models
            ]
        )
        qc_df = pd.concat([qc_df, skipped_qc], ignore_index=True)
        qc_df = qc_df.sort_values(["model_family", "task_name", "model_name"], kind="mergesort").reset_index(drop=True)
    write_dataframe(qc_df, output_paths["qc_summary"])

    metadata_df = build_model_metadata_table(model_records)
    if skipped_models:
        skipped_metadata = pd.DataFrame(
            [
                {
                    "model_family": entry.record.model_family,
                    "model_name": entry.record.model_name,
                    "task_name": entry.record.task_name,
                    "source_step": entry.record.source_step,
                    "artifact_path": str(entry.record.artifact_path),
                    "selected_as_best_flag": entry.record.selected_as_best_flag,
                    "selection_criterion": entry.record.selection_criterion,
                    "split_strategy_used": entry.record.split_strategy_used,
                    "target_label": entry.record.target_label,
                    "target_chembl_id": entry.record.target_chembl_id,
                    "ablation_name": entry.record.ablation_name,
                    "supporting_rank": entry.record.supporting_rank,
                    "notes": (
                        f"SKIPPED_UNLOADABLE_MODEL::{entry.artifact_diagnostic}::{entry.reconstruction_stage}"
                        f"::{entry.exception_type}::{entry.reason}"
                    ),
                }
                for entry in skipped_models
            ]
        )
        metadata_df = pd.concat([metadata_df, skipped_metadata], ignore_index=True)
        metadata_df = metadata_df.sort_values(["model_family", "task_name", "supporting_rank", "model_name"], kind="mergesort").reset_index(drop=True)
    if cfg.save_model_metadata_table:
        write_dataframe(metadata_df, output_paths["metadata"])

    if cfg.save_wide_prediction_tables:
        if not cfg.allow_full_wide_pivot:
            logging.warning("Skipping wide-table generation to remain memory-safe (allow_full_wide_pivot=false).")
        else:
            unified = pd.read_csv(output_paths["unified_scores"], low_memory=cfg.low_memory)
            wide_key = unified["model_family"].astype(str) + "__" + unified["task_name"].astype(str) + "__" + unified["model_name"].astype(str) + "__" + unified["predicted_value_type"].astype(str)
            unified["wide_column"] = wide_key
            index_cols = [col for col in ["screening_compound_id", "standardized_smiles", "source_library_name", "target_chembl_id", "target_name"] if col in unified.columns]
            wide = unified.pivot_table(index=index_cols, columns="wide_column", values="predicted_value", aggfunc="first").reset_index()
            wide.columns.name = None
            write_dataframe(wide, output_paths["wide"])

    manifest_rows = [
        {"asset_id": "unified_scores", "asset_type": "unified_prediction_table", "file_path": str(output_paths["unified_scores"]), "row_count": int(sum(v["scored"] for v in counters.values())), "notes": "Streaming unified long-format predictions."},
        {"asset_id": "screening_qc_summary", "asset_type": "qc_summary", "file_path": str(output_paths["qc_summary"]), "row_count": int(len(qc_df)), "notes": "Per-model screening scoring QC summary."},
        {"asset_id": "preflight_skipped_models", "asset_type": "preflight_model_validation", "file_path": str(output_paths["qc_summary"]), "row_count": int(len(skipped_models)), "notes": "Count of unloadable models skipped during preflight validation."},
    ]
    for family in SUPPORTED_MODEL_FAMILIES:
        manifest_rows.append({"asset_id": f"{family}_scores", "asset_type": "raw_prediction_table", "file_path": str(output_paths[f"{family}_scores"]), "row_count": int(sum(counters[idx]["scored"] for idx, rec in enumerate(model_records) if rec.model_family == family)), "notes": f"Family-specific predictions for {family}."})
        manifest_rows.append({"asset_id": f"failed_{family}_rows", "asset_type": "failed_rows", "file_path": str(output_paths[f"failed_{family}"]), "row_count": int(sum(counters[idx]["failed"] for idx, rec in enumerate(model_records) if rec.model_family == family)), "notes": f"Failed rows for {family}."})
    if cfg.save_model_metadata_table:
        manifest_rows.append({"asset_id": "model_metadata", "asset_type": "model_metadata", "file_path": str(output_paths["metadata"]), "row_count": int(len(metadata_df)), "notes": "Model provenance metadata."})
    write_dataframe(pd.DataFrame(manifest_rows), output_paths["manifest"])

    report = {
        "script_name": SCRIPT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "chunk_size": cfg.chunk_size,
        "write_every_n_chunks": cfg.write_every_n_chunks,
        "total_screening_rows_processed": total_rows,
        "total_compounds_successfully_scored_by_classical_models": len(family_scored_compounds["classical"]),
        "total_compounds_successfully_scored_by_deep_models": len(family_scored_compounds["deep"]),
        "total_compounds_successfully_scored_by_causal_models": len(family_scored_compounds["causal"]),
        "preflight_model_validation_summary": {
            "loadable_models": len(model_records),
            "skipped_models": len(skipped_models),
            "skipped_model_details": [
                {
                    "model_family": entry.record.model_family,
                    "model_name": entry.record.model_name,
                    "task_name": entry.record.task_name,
                    "artifact_path": str(entry.record.artifact_path),
                    "artifact_diagnostic": entry.artifact_diagnostic,
                    "reconstruction_stage": entry.reconstruction_stage,
                    "exception_type": entry.exception_type,
                    "traceback_summary": entry.traceback_summary,
                    "reason": entry.reason,
                }
                for entry in skipped_models
            ],
        },
        "task_coverage_summary": [
            {"task_name": task, "model_family": family, "n_prediction_rows": n} for (task, family), n in sorted(task_coverage.items())
        ],
        "target_coverage_summary": [
            {"target_chembl_id": t, "task_name": task, "model_family": family, "n_prediction_rows": n} for (t, task, family), n in sorted(target_coverage.items())
        ],
        "config_snapshot_reference": str(config_snapshot_path) if config_snapshot_path else "",
        "output_manifest_path": str(output_paths["manifest"]),
        "qc_summary_path": str(output_paths["qc_summary"]),
    }
    write_json(report, output_paths["report"])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    raw_config = load_yaml(config_path)
    project_root = config_path.parent.resolve()
    cfg = AppConfig.from_dict(raw_config, project_root)
    validate_target_selection(cfg)
    log_path = setup_logging(cfg)
    config_snapshot_path = save_config_snapshot(raw_config, cfg)
    validate_screening_inputs(cfg)
    model_records = resolve_models(cfg)
    run_streaming_scoring(cfg, model_records, config_snapshot_path)
    logging.info("Completed %s successfully.", SCRIPT_NAME)
    logging.info("Log written to %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
