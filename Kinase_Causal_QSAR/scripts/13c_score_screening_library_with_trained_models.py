#!/usr/bin/env python3
"""Score the inference-ready screening library with trained classical, deep, and causal models.

This script is a strict continuation of the kinase causality QSAR pipeline. It
loads the Step-13B screening feature assets, resolves score-producing models
from Step-10 comparison outputs and/or prior training summaries, performs
inference without retraining, and writes publication-grade prediction tables,
QC summaries, manifests, and a provenance-rich JSON report for downstream
consensus scoring, uncertainty estimation, applicability analysis, and
shortlist generation.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

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
}
REQUIRED_SCREENING_COLUMNS = {"screening_compound_id", "standardized_smiles"}
OPTIONAL_SCREENING_COLUMNS = ["source_library_name", "target_name", "target_chembl_id"]
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
            score_with_classical_models=parse_bool(section["score_with_classical_models"], "score_with_classical_models"),
            score_with_deep_models=parse_bool(section["score_with_deep_models"], "score_with_deep_models"),
            score_with_causal_models=parse_bool(section["score_with_causal_models"], "score_with_causal_models"),
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


def load_screening_inputs(cfg: AppConfig) -> dict[str, pd.DataFrame]:
    logging.info("Loading Step-13B screening inputs.")
    ensure_exists(cfg.input_feature_manifest_path, "screening feature manifest")
    ensure_exists(cfg.input_classical_feature_path, "screening classical feature table")
    ensure_exists(cfg.input_graph_manifest_path, "screening graph manifest")
    ensure_exists(cfg.input_environment_feature_path, "screening environment feature table")

    classical = pd.read_csv(cfg.input_classical_feature_path)
    graph = pd.read_csv(cfg.input_graph_manifest_path)
    environment = pd.read_csv(cfg.input_environment_feature_path)
    manifest = pd.read_csv(cfg.input_feature_manifest_path)

    ensure_columns(classical, REQUIRED_SCREENING_COLUMNS, "screening classical feature table")
    ensure_columns(graph, REQUIRED_SCREENING_COLUMNS, "screening graph manifest")
    ensure_columns(environment, REQUIRED_SCREENING_COLUMNS, "screening environment feature table")

    return {
        "classical": classical.sort_values(["screening_compound_id", "standardized_smiles"], kind="mergesort").reset_index(drop=True),
        "graph": graph.sort_values(["screening_compound_id", "standardized_smiles"], kind="mergesort").reset_index(drop=True),
        "environment": environment.sort_values(["screening_compound_id", "standardized_smiles"], kind="mergesort").reset_index(drop=True),
        "manifest": manifest.sort_values(manifest.columns.tolist(), kind="mergesort").reset_index(drop=True) if not manifest.empty else manifest,
    }


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
    roots = {
        "classical": cfg.classical_model_root,
        "deep": cfg.deep_model_root,
        "causal": cfg.causal_model_root,
    }
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
        patterns.extend([
            f"{task_name}/**/{model_name}/**/best_model.pt",
            f"**/{model_name}/**/best_model.pt",
        ])
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
    enabled_families = {
        "classical": cfg.score_with_classical_models,
        "deep": cfg.score_with_deep_models,
        "causal": cfg.score_with_causal_models,
    }
    combined = combined[combined["model_family"].isin([family for family, enabled in enabled_families.items() if enabled])].copy()
    if combined.empty:
        raise ValueError("No candidate models were resolved from the configured model-selection tables for the requested tasks and model families.")

    sort_cols = ["task_name", "model_family", "selection_metric_value", "model_name", "split_strategy"]
    ascending = [True, True, True, True, True]
    combined = combined.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)

    records: list[ModelRecord] = []
    for (task_name, model_family), frame in combined.groupby(["task_name", "model_family"], dropna=False, sort=True):
        if frame.empty:
            continue
        primary = frame.head(1)
        supporting = frame.iloc[1 : 1 + (cfg.max_supporting_models_per_family if cfg.include_supporting_models else 0)]
        selected_frame = pd.concat([primary, supporting], ignore_index=True)
        for rank, row in enumerate(selected_frame.itertuples(index=False), start=1):
            row_series = pd.Series(row._asdict())
            artifact_path = discover_model_artifact(row_series, model_family, cfg)
            if artifact_path is None or not artifact_path.exists():
                raise FileNotFoundError(
                    f"Unable to resolve model artifact for family={model_family}, task={task_name}, model={row_series.get('model_name')}. "
                    f"Provide artifact_path in the selection table or ensure artifacts exist under the configured model root."
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
    if not records:
        raise ValueError("Model selection resolution produced no usable scoring models.")
    return sorted(records, key=lambda rec: (rec.model_family, rec.task_name, rec.supporting_rank, rec.model_name, str(rec.artifact_path)))


def build_target_frame(base_df: pd.DataFrame, cfg: AppConfig, task_name: str, model_record: ModelRecord) -> pd.DataFrame:
    if task_name == "pairwise_selectivity":
        raise NotImplementedError(
            "Pairwise selectivity screening requires explicit target-pair definitions and model-specific pair mapping metadata; "
            "extend script_13c configuration and model sidecar metadata before enabling this task."
        )
    if cfg.target_selection_mode != "explicit_list":
        raise ValueError(f"Unsupported target_selection_mode for script_13c: {cfg.target_selection_mode}")
    targets = list(cfg.target_chembl_ids)
    expanded: list[pd.DataFrame] = []
    for target in targets:
        frame = base_df.copy()
        frame["target_chembl_id"] = target
        expanded.append(frame)
    combined = pd.concat(expanded, ignore_index=True)
    if "target_name" not in combined.columns:
        combined["target_name"] = np.nan
    if model_record.target_chembl_id:
        combined = combined[combined["target_chembl_id"] == model_record.target_chembl_id].reset_index(drop=True)
    return combined


def load_pickle_artifact(path: Path) -> Any:
    loaders = [joblib.load, lambda p: pickle.loads(Path(p).read_bytes())]
    last_exc: Exception | None = None
    for loader in loaders:
        try:
            return loader(path)
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_exc = exc
    raise RuntimeError(f"Failed to load pickled model artifact {path}: {last_exc}")


def score_classical_model(model_record: ModelRecord, feature_df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    logging.info("Scoring classical model %s for task %s from %s", model_record.model_name, model_record.task_name, model_record.artifact_path)
    artifact = load_pickle_artifact(model_record.artifact_path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Classical artifact must be a dict containing a fitted `model`: {model_record.artifact_path}")
    model = artifact["model"]
    feature_columns = artifact.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(f"Classical artifact is missing non-empty `feature_columns`: {model_record.artifact_path}")
    missing_features = sorted(set(feature_columns).difference(feature_df.columns))
    if missing_features:
        raise ValueError(
            f"Feature-space mismatch for classical scoring model {model_record.model_name}: missing screening feature columns {missing_features[:25]}"
        )
    scoring_df = build_target_frame(feature_df, cfg, model_record.task_name, model_record)
    matrix = scoring_df[feature_columns].copy()
    invalid_mask = matrix.isna().any(axis=1)
    failed_rows = scoring_df.loc[invalid_mask, ["screening_compound_id", "standardized_smiles"] + [c for c in ["source_library_name", "target_chembl_id", "target_name"] if c in scoring_df.columns]].copy()
    failed_rows["failure_reason"] = "missing_classical_features"
    failed_rows["model_family"] = model_record.model_family
    failed_rows["model_name"] = model_record.model_name
    failed_rows["task_name"] = model_record.task_name

    ready_df = scoring_df.loc[~invalid_mask].copy()
    if ready_df.empty:
        return pd.DataFrame(), failed_rows, {"attempted": int(len(scoring_df)), "scored": 0, "failed": int(len(failed_rows))}
    predictions = np.asarray(model.predict(ready_df[feature_columns]), dtype=float)
    output = ready_df[["screening_compound_id", "standardized_smiles"] + [c for c in ["source_library_name", "target_chembl_id", "target_name"] if c in ready_df.columns]].copy()
    output["task_name"] = model_record.task_name
    output["model_family"] = model_record.model_family
    output["model_name"] = model_record.model_name
    output["model_artifact_path"] = str(model_record.artifact_path)
    output["split_strategy_used_to_select_model"] = model_record.split_strategy_used
    output["predicted_value"] = predictions
    output["predicted_value_type"] = PREDICTED_VALUE_TYPES[model_record.task_name]
    output["score_type"] = output["predicted_value_type"]
    output["target_label"] = model_record.target_label
    output["selection_criterion"] = model_record.selection_criterion
    return output, failed_rows, {"attempted": int(len(scoring_df)), "scored": int(len(output)), "failed": int(len(failed_rows))}


def _load_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"PyTorch is required for deep/causal screening inference but could not be imported: {exc}") from exc
    return torch


def _load_torch_module_or_bundle(path: Path, torch: Any) -> tuple[Any, dict[str, Any]]:
    bundle_path = path.with_name("inference_bundle.pt")
    metadata_path = path.with_name("inference_metadata.json")
    if bundle_path.exists():
        payload = torch.load(bundle_path, map_location="cpu")
        if isinstance(payload, dict) and "model" in payload:
            return payload["model"], payload.get("metadata", {})
    payload = torch.load(path, map_location="cpu")
    if hasattr(payload, "eval") and hasattr(payload, "forward"):
        return payload, {}
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "model" in payload and hasattr(payload["model"], "eval"):
        return payload["model"], metadata
    raise RuntimeError(
        f"Torch artifact {path} does not contain a directly loadable model object. "
        "Script-13C supports direct torch modules or sidecar inference bundles/metadata; raw state-dict-only artifacts require a reconstructable inference bundle."
    )


def score_torch_family_model(model_record: ModelRecord, graph_df: pd.DataFrame, environment_df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    logging.info("Scoring %s model %s for task %s from %s", model_record.model_family, model_record.model_name, model_record.task_name, model_record.artifact_path)
    torch = _load_torch()
    model, metadata = _load_torch_module_or_bundle(model_record.artifact_path, torch)
    model.eval()
    base = build_target_frame(graph_df, cfg, model_record.task_name, model_record)
    merged = base.merge(environment_df, on=["screening_compound_id", "standardized_smiles"], how="left", suffixes=("", "_env"))

    if model_record.model_family == "causal":
        missing_env = merged.filter(regex=r"^(env_|environment_|kinase_family|murcko_scaffold)").isna().all(axis=1) if not merged.empty else pd.Series(dtype=bool)
    else:
        missing_env = pd.Series([False] * len(merged), index=merged.index)
    failed_rows = merged.loc[missing_env, ["screening_compound_id", "standardized_smiles"] + [c for c in ["source_library_name", "target_chembl_id", "target_name"] if c in merged.columns]].copy()
    failed_rows["failure_reason"] = "missing_environment_encodings" if model_record.model_family == "causal" else "graph_input_not_ready"
    failed_rows["model_family"] = model_record.model_family
    failed_rows["model_name"] = model_record.model_name
    failed_rows["task_name"] = model_record.task_name

    ready = merged.loc[~missing_env].copy()
    if ready.empty:
        return pd.DataFrame(), failed_rows, {"attempted": int(len(merged)), "scored": 0, "failed": int(len(failed_rows))}

    numeric_cols = [col for col in ready.columns if col not in {"screening_compound_id", "standardized_smiles", "source_library_name", "target_chembl_id", "target_name"} and pd.api.types.is_numeric_dtype(ready[col])]
    if not numeric_cols:
        raise ValueError(
            f"No numeric inference features were available for {model_record.model_family} scoring. "
            "Provide screening graph/environment manifests with precomputed numeric tensors or an inference bundle that declares input preparation."
        )
    tensor = torch.tensor(ready[numeric_cols].fillna(0.0).to_numpy(dtype=np.float32))
    with torch.no_grad():
        output_tensor = model(tensor)
    if isinstance(output_tensor, (list, tuple)):
        output_tensor = output_tensor[0]
    predictions = np.asarray(output_tensor.detach().cpu().numpy()).reshape(-1)

    output = ready[["screening_compound_id", "standardized_smiles"] + [c for c in ["source_library_name", "target_chembl_id", "target_name"] if c in ready.columns]].copy()
    output["task_name"] = model_record.task_name
    output["model_family"] = model_record.model_family
    output["model_name"] = model_record.model_name
    output["model_artifact_path"] = str(model_record.artifact_path)
    output["split_strategy_used_to_select_model"] = model_record.split_strategy_used
    output["predicted_value"] = predictions
    output["predicted_value_type"] = PREDICTED_VALUE_TYPES[model_record.task_name]
    output["score_type"] = output["predicted_value_type"]
    output["target_label"] = model_record.target_label
    output["selection_criterion"] = model_record.selection_criterion
    output["environment_conditioning_metadata"] = json.dumps(metadata, sort_keys=True) if metadata else ""
    return output, failed_rows, {"attempted": int(len(merged)), "scored": int(len(output)), "failed": int(len(failed_rows))}


def score_models(cfg: AppConfig, screening_inputs: dict[str, pd.DataFrame], model_records: list[ModelRecord]) -> dict[str, Any]:
    predictions_by_family: dict[str, list[pd.DataFrame]] = {family: [] for family in SUPPORTED_MODEL_FAMILIES}
    failed_by_family: dict[str, list[pd.DataFrame]] = {family: [] for family in SUPPORTED_MODEL_FAMILIES}
    qc_rows: list[dict[str, Any]] = []
    warnings_list: list[str] = []

    for record in model_records:
        try:
            if record.model_family == "classical":
                predictions, failed_rows, counts = score_classical_model(record, screening_inputs["classical"], cfg)
            elif record.model_family in {"deep", "causal"}:
                predictions, failed_rows, counts = score_torch_family_model(record, screening_inputs["graph"], screening_inputs["environment"], cfg)
            else:  # pragma: no cover - guarded by earlier filtering
                raise ValueError(f"Unsupported model family: {record.model_family}")
        except NotImplementedError as exc:
            logging.warning("Skipping model %s because %s", record.model_name, exc)
            warnings_list.append(f"{record.model_family}/{record.task_name}/{record.model_name}: {exc}")
            failed_rows = build_target_frame(screening_inputs["classical"][["screening_compound_id", "standardized_smiles"] + [c for c in ["source_library_name"] if c in screening_inputs["classical"].columns]].copy(), cfg, record.task_name, record)
            failed_rows = failed_rows[[c for c in ["screening_compound_id", "standardized_smiles", "source_library_name", "target_chembl_id", "target_name"] if c in failed_rows.columns]].copy()
            failed_rows["failure_reason"] = str(exc)
            failed_rows["model_family"] = record.model_family
            failed_rows["model_name"] = record.model_name
            failed_rows["task_name"] = record.task_name
            predictions = pd.DataFrame()
            counts = {"attempted": int(len(failed_rows)), "scored": 0, "failed": int(len(failed_rows))}
        except Exception as exc:
            raise RuntimeError(
                f"Inference failed for model_family={record.model_family}, task={record.task_name}, model={record.model_name}, artifact={record.artifact_path}: {exc}"
            ) from exc

        if not predictions.empty:
            predictions_by_family[record.model_family].append(predictions.sort_values(["screening_compound_id", "target_chembl_id", "model_name"], kind="mergesort", na_position="last").reset_index(drop=True))
        if not failed_rows.empty:
            failed_by_family[record.model_family].append(failed_rows.sort_values(["screening_compound_id"], kind="mergesort").reset_index(drop=True))
        qc_rows.append(
            {
                "model_family": record.model_family,
                "model_name": record.model_name,
                "task_name": record.task_name,
                "target_chembl_id": record.target_chembl_id,
                "number_of_compounds_attempted": counts["attempted"],
                "number_of_compounds_scored_successfully": counts["scored"],
                "number_of_failed_rows": counts["failed"],
                "notes": record.notes,
            }
        )

    return {
        "predictions_by_family": predictions_by_family,
        "failed_by_family": failed_by_family,
        "qc_summary": pd.DataFrame(qc_rows).sort_values(["model_family", "task_name", "model_name"], kind="mergesort").reset_index(drop=True),
        "warnings": warnings_list,
    }


def unify_predictions(predictions_by_family: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    frames = [frame for family_frames in predictions_by_family.values() for frame in family_frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["screening_compound_id", "standardized_smiles", "target_chembl_id", "task_name", "model_family", "model_name", "predicted_value", "predicted_value_type"])
    unified = pd.concat(frames, ignore_index=True)
    ordered_cols = [
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
    existing = [col for col in ordered_cols if col in unified.columns]
    remainder = [col for col in unified.columns if col not in existing]
    return unified[existing + remainder].sort_values(["screening_compound_id", "target_chembl_id", "task_name", "model_family", "model_name"], kind="mergesort", na_position="last").reset_index(drop=True)


def build_wide_prediction_table(unified: pd.DataFrame) -> pd.DataFrame:
    if unified.empty:
        return pd.DataFrame()
    wide_key = unified["model_family"].astype(str) + "__" + unified["task_name"].astype(str) + "__" + unified["model_name"].astype(str) + "__" + unified["predicted_value_type"].astype(str)
    working = unified.copy()
    working["wide_column"] = wide_key
    index_cols = [col for col in ["screening_compound_id", "standardized_smiles", "source_library_name", "target_chembl_id", "target_name"] if col in working.columns]
    wide = working.pivot_table(index=index_cols, columns="wide_column", values="predicted_value", aggfunc="first").reset_index()
    wide.columns.name = None
    return wide.sort_values(index_cols, kind="mergesort", na_position="last").reset_index(drop=True)


def build_model_metadata_table(records: list[ModelRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
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
        ]
    ).sort_values(["model_family", "task_name", "supporting_rank", "model_name"], kind="mergesort").reset_index(drop=True)


def write_outputs(cfg: AppConfig, scoring_outputs: dict[str, Any], model_records: list[ModelRecord], config_snapshot_path: Path | None, screening_inputs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    cfg.output_scoring_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []

    family_output_paths = {
        "classical": cfg.output_classical_scores_path,
        "deep": cfg.output_deep_scores_path,
        "causal": cfg.output_causal_scores_path,
    }
    for family, path in family_output_paths.items():
        frame = pd.concat(scoring_outputs["predictions_by_family"][family], ignore_index=True) if scoring_outputs["predictions_by_family"][family] else pd.DataFrame()
        if cfg.save_raw_predictions:
            write_dataframe(frame, path)
            manifest_rows.append({"asset_id": f"{family}_scores", "asset_type": "raw_prediction_table", "file_path": str(path), "row_count": int(len(frame)), "model_family": family, "task_name": "multiple" if not frame.empty else "", "notes": f"Family-specific screening predictions for {family} models."})
        if cfg.save_failed_rows:
            failed_frame = pd.concat(scoring_outputs["failed_by_family"][family], ignore_index=True) if scoring_outputs["failed_by_family"][family] else pd.DataFrame()
            failed_path = cfg.output_scoring_root / f"failed_{family}_scoring_rows.csv"
            write_dataframe(failed_frame, failed_path)
            manifest_rows.append({"asset_id": f"failed_{family}_rows", "asset_type": "failed_rows", "file_path": str(failed_path), "row_count": int(len(failed_frame)), "model_family": family, "task_name": "multiple" if not failed_frame.empty else "", "notes": f"Rows that could not be scored by {family} models."})

    unified = unify_predictions(scoring_outputs["predictions_by_family"])
    write_dataframe(unified, cfg.output_unified_scores_path)
    manifest_rows.append({"asset_id": "unified_scores", "asset_type": "unified_prediction_table", "file_path": str(cfg.output_unified_scores_path), "row_count": int(len(unified)), "model_family": "multiple", "task_name": "multiple", "notes": "Unified long-format screening predictions across all enabled model families."})

    wide_path = cfg.output_scoring_root / "unified_screening_scores_wide.csv"
    if cfg.save_wide_prediction_tables:
        wide = build_wide_prediction_table(unified)
        write_dataframe(wide, wide_path)
        manifest_rows.append({"asset_id": "unified_scores_wide", "asset_type": "wide_prediction_table", "file_path": str(wide_path), "row_count": int(len(wide)), "model_family": "multiple", "task_name": "multiple", "notes": "Wide-format screening predictions for downstream consensus/ranking convenience."})
    else:
        wide = pd.DataFrame()

    qc_path = cfg.output_scoring_root / "screening_scoring_qc_summary.csv"
    write_dataframe(scoring_outputs["qc_summary"], qc_path)
    manifest_rows.append({"asset_id": "screening_qc_summary", "asset_type": "qc_summary", "file_path": str(qc_path), "row_count": int(len(scoring_outputs["qc_summary"])), "model_family": "multiple", "task_name": "multiple", "notes": "Per-model screening scoring QC summary."})

    metadata_df = build_model_metadata_table(model_records)
    metadata_path = cfg.output_scoring_root / "screening_model_metadata.csv"
    if cfg.save_model_metadata_table:
        write_dataframe(metadata_df, metadata_path)
        manifest_rows.append({"asset_id": "model_metadata", "asset_type": "model_metadata", "file_path": str(metadata_path), "row_count": int(len(metadata_df)), "model_family": "multiple", "task_name": "multiple", "notes": "Metadata and provenance for models used during Step-13C screening inference."})

    write_dataframe(pd.DataFrame(manifest_rows), cfg.output_manifest_path)

    report = {
        "script_name": SCRIPT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_feature_files_used": {
            "classical": str(cfg.input_classical_feature_path),
            "graph_manifest": str(cfg.input_graph_manifest_path),
            "environment": str(cfg.input_environment_feature_path),
            "feature_manifest": str(cfg.input_feature_manifest_path),
        },
        "trained_model_roots_used": {
            "classical": str(cfg.classical_model_root),
            "deep": str(cfg.deep_model_root),
            "causal": str(cfg.causal_model_root),
        },
        "best_model_selection_files_used": {
            "best_models_by_task": str(cfg.input_best_models_by_task_path),
            "best_models_by_split_strategy": str(cfg.input_best_models_by_split_strategy_path),
        },
        "total_screening_compounds_processed": int(screening_inputs["classical"]["screening_compound_id"].nunique()),
        "total_compounds_successfully_scored_by_classical_models": int(unified.loc[unified["model_family"] == "classical", "screening_compound_id"].nunique()) if not unified.empty else 0,
        "total_compounds_successfully_scored_by_deep_models": int(unified.loc[unified["model_family"] == "deep", "screening_compound_id"].nunique()) if not unified.empty else 0,
        "total_compounds_successfully_scored_by_causal_models": int(unified.loc[unified["model_family"] == "causal", "screening_compound_id"].nunique()) if not unified.empty else 0,
        "task_coverage_summary": unified.groupby(["task_name", "model_family"], dropna=False).size().reset_index(name="n_prediction_rows").to_dict(orient="records") if not unified.empty else [],
        "target_coverage_summary": unified.groupby([col for col in ["target_chembl_id", "task_name", "model_family"] if col in unified.columns], dropna=False).size().reset_index(name="n_prediction_rows").to_dict(orient="records") if not unified.empty and "target_chembl_id" in unified.columns else [],
        "model_metadata_summary": metadata_df.to_dict(orient="records"),
        "failed_row_counts": {
            family: int(sum(len(frame) for frame in scoring_outputs["failed_by_family"][family])) for family in SUPPORTED_MODEL_FAMILIES
        },
        "warnings": scoring_outputs["warnings"],
        "config_snapshot_reference": str(config_snapshot_path) if config_snapshot_path else "",
        "output_manifest_path": str(cfg.output_manifest_path),
        "qc_summary_path": str(qc_path),
    }
    write_json(report, cfg.output_report_path)
    return {"unified": unified, "wide": wide, "metadata": metadata_df, "manifest": pd.DataFrame(manifest_rows), "report": report}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    raw_config = load_yaml(config_path)
    project_root = config_path.parent.resolve()
    cfg = AppConfig.from_dict(raw_config, project_root)
    validate_target_selection(cfg)
    log_path = setup_logging(cfg)
    config_snapshot_path = save_config_snapshot(raw_config, cfg)
    logging.info("Starting %s", SCRIPT_NAME)
    screening_inputs = load_screening_inputs(cfg)
    model_records = resolve_models(cfg)
    scoring_outputs = score_models(cfg, screening_inputs, model_records)
    write_outputs(cfg, scoring_outputs, model_records, config_snapshot_path, screening_inputs)
    logging.info("Completed %s successfully.", SCRIPT_NAME)
    logging.info("Log written to %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
