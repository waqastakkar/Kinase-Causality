#!/usr/bin/env python3
"""Score the inference-ready screening library with trained classical, deep, and causal models."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
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


def prepare_model_runtime(record: ModelRecord) -> dict[str, Any]:
    if record.model_family == "classical":
        artifact = load_pickle_artifact(record.artifact_path)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError(f"Classical artifact must contain `model`: {record.artifact_path}")
        feature_columns = artifact.get("feature_columns")
        if not isinstance(feature_columns, list) or not feature_columns:
            raise ValueError(f"Classical artifact missing non-empty `feature_columns`: {record.artifact_path}")
        return {
            "model": artifact["model"],
            "feature_columns": feature_columns,
            "metadata": {"serialized_sklearn_version": _extract_serialized_sklearn_version(artifact)},
        }
    torch = _load_torch()
    model, metadata = _load_torch_module_or_bundle(record.artifact_path, torch, record)
    inference_mode = _extract_inference_mode(metadata)
    if not inference_mode:
        raise RuntimeError("missing_supported_inference_mode_metadata")
    if inference_mode != "flat_numeric_tensor":
        raise RuntimeError(f"unsupported_inference_mode::{inference_mode}")
    if not _supports_flat_numeric_tensor_input(metadata):
        raise RuntimeError("flat_numeric_tensor_not_explicitly_supported")
    model.eval()
    return {"torch": torch, "model": model, "metadata": metadata}


def preflight_validate_model_runtimes(cfg: AppConfig, model_records: list[ModelRecord]) -> tuple[list[ModelRecord], dict[int, dict[str, Any]], list[ModelLoadFailure]]:
    loadable_records: list[ModelRecord] = []
    runtimes: dict[int, dict[str, Any]] = {}
    skipped: list[ModelLoadFailure] = []
    serialized_sklearn_versions: set[str] = set()
    for record in model_records:
        try:
            runtime = prepare_model_runtime(record)
            if record.model_family == "classical":
                version = runtime.get("metadata", {}).get("serialized_sklearn_version")
                if version:
                    serialized_sklearn_versions.add(str(version))
            loadable_records.append(record)
            runtimes[len(loadable_records) - 1] = runtime
        except Exception as exc:
            reason = str(exc)
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
            skipped.append(ModelLoadFailure(record=record, reason=reason, artifact_diagnostic=artifact_kind))

    logging.info(
        "Preflight model validation complete. loadable=%s skipped=%s total=%s",
        len(loadable_records),
        len(skipped),
        len(model_records),
    )
    for failure in skipped:
        logging.warning(
            "Skipping model for screening safety: family=%s model=%s task=%s checkpoint_exists=true "
            "valid_screening_inference_artifact=false diagnostic=%s reason=%s artifact=%s",
            failure.record.model_family,
            failure.record.model_name,
            failure.record.task_name,
            failure.artifact_diagnostic,
            failure.reason,
            failure.record.artifact_path,
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
    metadata = runtime.get("metadata", {})
    inference_mode = _extract_inference_mode(metadata)
    if inference_mode != "flat_numeric_tensor" or not _supports_flat_numeric_tensor_input(metadata):
        raise ValueError(
            f"Unsupported runtime inference mode for {record.model_family}/{record.model_name}: "
            f"mode={inference_mode or 'missing'} supports_flat_numeric_tensor={_supports_flat_numeric_tensor_input(metadata)}"
        )

    base = build_target_frame(graph_df, cfg, record.task_name, record, target_name_map=target_name_map)
    merged = base.merge(environment_df, on=["screening_compound_id", "standardized_smiles"], how="left", suffixes=("", "_env"))
    if record.model_family == "causal":
        missing_env = merged.filter(regex=r"^(env_|environment_|kinase_family|murcko_scaffold)").isna().all(axis=1) if not merged.empty else pd.Series(dtype=bool)
    else:
        missing_env = pd.Series([False] * len(merged), index=merged.index)
    failed_rows = merged.loc[missing_env, ["screening_compound_id", "standardized_smiles"] + [c for c in ["source_library_name", "target_chembl_id", "target_name"] if c in merged.columns]].copy()
    failed_rows["failure_reason"] = "missing_environment_encodings" if record.model_family == "causal" else "graph_input_not_ready"
    failed_rows["model_family"] = record.model_family
    failed_rows["model_name"] = record.model_name
    failed_rows["task_name"] = record.task_name
    ready = merged.loc[~missing_env].copy()
    if ready.empty:
        return pd.DataFrame(), failed_rows, {"attempted": int(len(merged)), "scored": 0, "failed": int(len(failed_rows))}

    numeric_cols = [col for col in ready.columns if col not in {"screening_compound_id", "standardized_smiles", "source_library_name", "target_chembl_id", "target_name"} and pd.api.types.is_numeric_dtype(ready[col])]
    if not numeric_cols:
        raise ValueError(f"No numeric inference features were available for {record.model_family}/{record.model_name} in this chunk.")
    torch = runtime["torch"]
    tensor = torch.tensor(ready[numeric_cols].fillna(0.0).to_numpy(dtype=np.float32))
    with torch.no_grad():
        try:
            output_tensor = runtime["model"](tensor)
        except Exception as exc:
            reason = f"model_forward_failed::{type(exc).__name__}::{exc}"
            failed_all = _build_failed_rows_for_scoring_exception(record, ready, reason)
            return (
                pd.DataFrame(),
                pd.concat([failed_rows, failed_all], ignore_index=True),
                {"attempted": int(len(merged)), "scored": 0, "failed": int(len(failed_rows) + len(failed_all))},
            )
    if isinstance(output_tensor, (list, tuple)):
        output_tensor = output_tensor[0]
    predictions = np.asarray(output_tensor.detach().cpu().numpy()).reshape(-1)

    output = ready[["screening_compound_id", "standardized_smiles"] + [c for c in ["source_library_name", "target_chembl_id", "target_name"] if c in ready.columns]].copy()
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
                    "notes": f"SKIPPED_UNLOADABLE_MODEL::{entry.artifact_diagnostic}::{entry.reason}",
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
                    "notes": f"SKIPPED_UNLOADABLE_MODEL::{entry.artifact_diagnostic}::{entry.reason}",
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
