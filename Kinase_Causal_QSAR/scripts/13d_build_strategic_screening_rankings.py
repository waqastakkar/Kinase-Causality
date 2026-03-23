#!/usr/bin/env python3
"""Build strategic multi-criterion screening rankings from Step-13C screening scores.

This script is a strict continuation of the kinase causality QSAR screening
pipeline. It consumes the raw screening predictions emitted by Script-13C,
combines them with Step-13A/13B screening metadata and optional training
references, computes publication-grade consensus/disagreement summaries,
uncertainty proxies, applicability-domain proxy features, diversity-readiness
signals, and interpretable composite strategic ranking scores.

The script is intentionally deterministic, config-driven, provenance-rich, and
explicit that disagreement/applicability outputs are *proxies* rather than
formal calibrated uncertainty or certified domain membership.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

SCRIPT_NAME = "13d_build_strategic_screening_rankings"
REQUIRED_SCRIPT_KEYS = {
    "input_unified_scores_path",
    "input_classical_scores_path",
    "input_deep_scores_path",
    "input_causal_scores_path",
    "input_screening_library_path",
    "input_classical_feature_path",
    "input_environment_feature_path",
    "input_training_annotated_long_path",
    "input_compound_env_path",
    "output_ranking_root",
    "output_compound_target_ranking_path",
    "output_compound_summary_ranking_path",
    "output_uncertainty_summary_path",
    "output_applicability_summary_path",
    "output_consensus_summary_path",
    "output_manifest_path",
    "output_report_path",
    "ranking_mode",
    "primary_target_chembl_ids",
    "score_components",
    "score_weights",
    "potency_source_preference",
    "selectivity_source_preference",
    "uncertainty",
    "applicability",
    "diversity",
    "normalization",
    "save_intermediate_component_tables",
    "save_failed_rows",
    "save_config_snapshot",
}
REQUIRED_UNIFIED_COLUMNS = {
    "screening_compound_id",
    "standardized_smiles",
    "model_family",
    "model_name",
    "task_name",
    "predicted_value",
}
CORE_ID_COLUMNS = ["screening_compound_id", "standardized_smiles", "source_library_name", "target_chembl_id"]
FAMILY_ORDER = ("classical", "deep", "causal")
POTENCY_SCORE_TYPES = {"predicted_pKi", "potency", "predicted_pki"}
SELECTIVITY_SCORE_TYPES = {
    "predicted_target_vs_panel_delta_pKi",
    "predicted_pairwise_delta_pKi",
    "selectivity",
    "predicted_target_vs_panel_delta_pki",
    "predicted_pairwise_delta_pki",
}
NUMERIC_RANGE_FALLBACK = (-10.0, 20.0)


@dataclass(frozen=True)
class ScoreComponentConfig:
    use_potency_score: bool
    use_selectivity_score: bool
    use_uncertainty_penalty: bool
    use_applicability_penalty: bool
    use_diversity_bonus_placeholder: bool


@dataclass(frozen=True)
class ScoreWeightConfig:
    potency_weight: float
    selectivity_weight: float
    uncertainty_penalty_weight: float
    applicability_penalty_weight: float
    diversity_bonus_weight: float


@dataclass(frozen=True)
class UncertaintyConfig:
    use_family_disagreement: bool
    use_within_family_disagreement: bool
    use_cross_model_std: bool
    use_top_model_gap: bool


@dataclass(frozen=True)
class ApplicabilityConfig:
    use_descriptor_distance_proxy: bool
    use_feature_range_violations: bool
    use_scaffold_novelty_flag: bool
    use_out_of_range_physchem_flag: bool


@dataclass(frozen=True)
class DiversityConfig:
    compute_scaffold_groups: bool
    compute_fingerprint_clusters: bool
    diversity_bonus_is_placeholder_only: bool


@dataclass(frozen=True)
class NormalizationConfig:
    method: str
    larger_is_better_for_potency: bool
    larger_is_better_for_selectivity: bool


@dataclass(frozen=True)
class AppConfig:
    input_unified_scores_path: Path
    input_classical_scores_path: Path
    input_deep_scores_path: Path
    input_causal_scores_path: Path
    input_screening_library_path: Path
    input_classical_feature_path: Path
    input_environment_feature_path: Path
    input_training_annotated_long_path: Path
    input_compound_env_path: Path
    output_ranking_root: Path
    output_compound_target_ranking_path: Path
    output_compound_summary_ranking_path: Path
    output_uncertainty_summary_path: Path
    output_applicability_summary_path: Path
    output_consensus_summary_path: Path
    output_manifest_path: Path
    output_report_path: Path
    ranking_mode: str
    primary_target_chembl_ids: tuple[str, ...]
    score_components: ScoreComponentConfig
    score_weights: ScoreWeightConfig
    potency_source_preference: tuple[str, ...]
    selectivity_source_preference: tuple[str, ...]
    uncertainty: UncertaintyConfig
    applicability: ApplicabilityConfig
    diversity: DiversityConfig
    normalization: NormalizationConfig
    save_intermediate_component_tables: bool
    save_failed_rows: bool
    save_config_snapshot: bool
    project_root: Path
    logs_dir: Path
    configs_used_dir: Path

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        section = raw.get("script_13d")
        if not isinstance(section, dict):
            raise ValueError("Missing required `script_13d` section in config.yaml.")
        missing = sorted(REQUIRED_SCRIPT_KEYS.difference(section))
        if missing:
            raise ValueError("Missing required script_13d config values: " + ", ".join(missing))

        def resolve(path_like: Any) -> Path:
            path = Path(str(path_like))
            return path if path.is_absolute() else project_root / path

        def parse_bool(value: Any, key: str) -> bool:
            if not isinstance(value, bool):
                raise ValueError(f"script_13d.{key} must be boolean; got {value!r}.")
            return value

        def parse_float(value: Any, key: str) -> float:
            try:
                return float(value)
            except Exception as exc:
                raise ValueError(f"script_13d.{key} must be numeric; got {value!r}.") from exc

        def parse_str_list(value: Any, key: str) -> tuple[str, ...]:
            if not isinstance(value, list):
                raise ValueError(f"script_13d.{key} must be a list.")
            return tuple(str(item).strip() for item in value if str(item).strip())

        def require_mapping(value: Any, key: str) -> dict[str, Any]:
            if not isinstance(value, dict):
                raise ValueError(f"script_13d.{key} must be a mapping.")
            return value

        component_map = require_mapping(section["score_components"], "score_components")
        weight_map = require_mapping(section["score_weights"], "score_weights")
        uncertainty_map = require_mapping(section["uncertainty"], "uncertainty")
        applicability_map = require_mapping(section["applicability"], "applicability")
        diversity_map = require_mapping(section["diversity"], "diversity")
        normalization_map = require_mapping(section["normalization"], "normalization")

        return AppConfig(
            input_unified_scores_path=resolve(section["input_unified_scores_path"]),
            input_classical_scores_path=resolve(section["input_classical_scores_path"]),
            input_deep_scores_path=resolve(section["input_deep_scores_path"]),
            input_causal_scores_path=resolve(section["input_causal_scores_path"]),
            input_screening_library_path=resolve(section["input_screening_library_path"]),
            input_classical_feature_path=resolve(section["input_classical_feature_path"]),
            input_environment_feature_path=resolve(section["input_environment_feature_path"]),
            input_training_annotated_long_path=resolve(section["input_training_annotated_long_path"]),
            input_compound_env_path=resolve(section["input_compound_env_path"]),
            output_ranking_root=resolve(section["output_ranking_root"]),
            output_compound_target_ranking_path=resolve(section["output_compound_target_ranking_path"]),
            output_compound_summary_ranking_path=resolve(section["output_compound_summary_ranking_path"]),
            output_uncertainty_summary_path=resolve(section["output_uncertainty_summary_path"]),
            output_applicability_summary_path=resolve(section["output_applicability_summary_path"]),
            output_consensus_summary_path=resolve(section["output_consensus_summary_path"]),
            output_manifest_path=resolve(section["output_manifest_path"]),
            output_report_path=resolve(section["output_report_path"]),
            ranking_mode=str(section["ranking_mode"]).strip().lower(),
            primary_target_chembl_ids=parse_str_list(section["primary_target_chembl_ids"], "primary_target_chembl_ids"),
            score_components=ScoreComponentConfig(
                use_potency_score=parse_bool(component_map["use_potency_score"], "score_components.use_potency_score"),
                use_selectivity_score=parse_bool(component_map["use_selectivity_score"], "score_components.use_selectivity_score"),
                use_uncertainty_penalty=parse_bool(component_map["use_uncertainty_penalty"], "score_components.use_uncertainty_penalty"),
                use_applicability_penalty=parse_bool(component_map["use_applicability_penalty"], "score_components.use_applicability_penalty"),
                use_diversity_bonus_placeholder=parse_bool(component_map["use_diversity_bonus_placeholder"], "score_components.use_diversity_bonus_placeholder"),
            ),
            score_weights=ScoreWeightConfig(
                potency_weight=parse_float(weight_map["potency_weight"], "score_weights.potency_weight"),
                selectivity_weight=parse_float(weight_map["selectivity_weight"], "score_weights.selectivity_weight"),
                uncertainty_penalty_weight=parse_float(weight_map["uncertainty_penalty_weight"], "score_weights.uncertainty_penalty_weight"),
                applicability_penalty_weight=parse_float(weight_map["applicability_penalty_weight"], "score_weights.applicability_penalty_weight"),
                diversity_bonus_weight=parse_float(weight_map["diversity_bonus_weight"], "score_weights.diversity_bonus_weight"),
            ),
            potency_source_preference=parse_str_list(section["potency_source_preference"], "potency_source_preference"),
            selectivity_source_preference=parse_str_list(section["selectivity_source_preference"], "selectivity_source_preference"),
            uncertainty=UncertaintyConfig(
                use_family_disagreement=parse_bool(uncertainty_map["use_family_disagreement"], "uncertainty.use_family_disagreement"),
                use_within_family_disagreement=parse_bool(uncertainty_map["use_within_family_disagreement"], "uncertainty.use_within_family_disagreement"),
                use_cross_model_std=parse_bool(uncertainty_map["use_cross_model_std"], "uncertainty.use_cross_model_std"),
                use_top_model_gap=parse_bool(uncertainty_map["use_top_model_gap"], "uncertainty.use_top_model_gap"),
            ),
            applicability=ApplicabilityConfig(
                use_descriptor_distance_proxy=parse_bool(applicability_map["use_descriptor_distance_proxy"], "applicability.use_descriptor_distance_proxy"),
                use_feature_range_violations=parse_bool(applicability_map["use_feature_range_violations"], "applicability.use_feature_range_violations"),
                use_scaffold_novelty_flag=parse_bool(applicability_map["use_scaffold_novelty_flag"], "applicability.use_scaffold_novelty_flag"),
                use_out_of_range_physchem_flag=parse_bool(applicability_map["use_out_of_range_physchem_flag"], "applicability.use_out_of_range_physchem_flag"),
            ),
            diversity=DiversityConfig(
                compute_scaffold_groups=parse_bool(diversity_map["compute_scaffold_groups"], "diversity.compute_scaffold_groups"),
                compute_fingerprint_clusters=parse_bool(diversity_map["compute_fingerprint_clusters"], "diversity.compute_fingerprint_clusters"),
                diversity_bonus_is_placeholder_only=parse_bool(diversity_map["diversity_bonus_is_placeholder_only"], "diversity.diversity_bonus_is_placeholder_only"),
            ),
            normalization=NormalizationConfig(
                method=str(normalization_map["method"]).strip().lower(),
                larger_is_better_for_potency=parse_bool(normalization_map["larger_is_better_for_potency"], "normalization.larger_is_better_for_potency"),
                larger_is_better_for_selectivity=parse_bool(normalization_map["larger_is_better_for_selectivity"], "normalization.larger_is_better_for_selectivity"),
            ),
            save_intermediate_component_tables=parse_bool(section["save_intermediate_component_tables"], "save_intermediate_component_tables"),
            save_failed_rows=parse_bool(section["save_failed_rows"], "save_failed_rows"),
            save_config_snapshot=parse_bool(section["save_config_snapshot"], "save_config_snapshot"),
            project_root=project_root,
            logs_dir=project_root / "logs",
            configs_used_dir=project_root / "configs_used",
        )


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


def normalize_text_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def deterministic_rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(method="first", ascending=False).astype(int)


def robust_rank_normalize(series: pd.Series, *, larger_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = numeric.notna()
    if valid.sum() == 0:
        return result
    if valid.sum() == 1:
        result.loc[valid] = 1.0
        return result
    ranks = numeric.loc[valid].rank(method="average", ascending=not larger_is_better)
    scaled = (ranks - 1.0) / max(float(valid.sum() - 1), 1.0)
    result.loc[valid] = scaled.astype(float)
    return result


def safe_std(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) <= 1:
        return 0.0
    return float(numeric.std(ddof=0))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    alias_map = {
        "compound_id": "screening_compound_id",
        "molecule_id": "screening_compound_id",
        "canonical_smiles": "standardized_smiles",
        "smiles": "standardized_smiles",
        "library_name": "source_library_name",
        "target_id": "target_chembl_id",
        "prediction": "predicted_value",
        "score": "predicted_value",
        "prediction_type": "predicted_value_type",
        "score_type": "predicted_value_type",
    }
    for old, new in alias_map.items():
        if old in renamed.columns and new not in renamed.columns:
            renamed = renamed.rename(columns={old: new})
    return renamed


def infer_predicted_value_type(df: pd.DataFrame) -> pd.Series:
    if "predicted_value_type" in df.columns:
        return normalize_text_series(df["predicted_value_type"])
    task = normalize_text_series(df.get("task_name", pd.Series("", index=df.index)))
    inferred = pd.Series("auxiliary", index=df.index, dtype="string")
    inferred.loc[task.str.contains("multitask|regression", case=False, na=False)] = "predicted_pKi"
    inferred.loc[task.str.contains("target_vs_panel", case=False, na=False)] = "predicted_target_vs_panel_delta_pKi"
    inferred.loc[task.str.contains("pairwise_selectivity", case=False, na=False)] = "predicted_pairwise_delta_pKi"
    return inferred


def resolve_score_category(value_type: str, task_name: str) -> str:
    joined = f"{value_type} {task_name}".lower()
    if any(token.lower() in joined for token in POTENCY_SCORE_TYPES):
        return "potency"
    if any(token.lower() in joined for token in SELECTIVITY_SCORE_TYPES):
        return "selectivity"
    return "auxiliary"


def load_optional_table(path: Path, description: str, warnings_list: list[str]) -> pd.DataFrame:
    if not path.exists():
        warnings_list.append(f"Optional {description} not found: {path}")
        logging.warning("Optional %s not found: %s", description, path)
        return pd.DataFrame()
    logging.info("Loading optional %s from %s", description, path)
    return normalize_columns(pd.read_csv(path))


def load_inputs(cfg: AppConfig) -> tuple[dict[str, pd.DataFrame], list[str]]:
    warnings_list: list[str] = []
    logging.info("Loading required Step-13C/13B/13A inputs for strategic ranking.")
    for path, description in [
        (cfg.input_unified_scores_path, "unified screening scores"),
        (cfg.input_classical_scores_path, "classical screening scores"),
        (cfg.input_deep_scores_path, "deep screening scores"),
        (cfg.input_causal_scores_path, "causal screening scores"),
        (cfg.input_screening_library_path, "merged screening library"),
        (cfg.input_classical_feature_path, "screening classical features"),
        (cfg.input_environment_feature_path, "screening environment features"),
    ]:
        ensure_exists(path, description)

    unified = normalize_columns(pd.read_csv(cfg.input_unified_scores_path))
    ensure_columns(unified, REQUIRED_UNIFIED_COLUMNS, "unified screening scores")
    unified["predicted_value_type"] = infer_predicted_value_type(unified)
    unified["screening_compound_id"] = normalize_text_series(unified["screening_compound_id"])
    unified["standardized_smiles"] = normalize_text_series(unified["standardized_smiles"])
    unified["model_family"] = normalize_text_series(unified["model_family"]).str.lower()
    unified["model_name"] = normalize_text_series(unified["model_name"])
    unified["task_name"] = normalize_text_series(unified["task_name"])
    unified["predicted_value"] = pd.to_numeric(unified["predicted_value"], errors="coerce")
    if "target_chembl_id" not in unified.columns:
        unified["target_chembl_id"] = ""
    if "source_library_name" not in unified.columns:
        unified["source_library_name"] = ""
    unified["target_chembl_id"] = normalize_text_series(unified["target_chembl_id"])
    unified["source_library_name"] = normalize_text_series(unified["source_library_name"])
    unified["score_category"] = [resolve_score_category(vt, task) for vt, task in zip(unified["predicted_value_type"], unified["task_name"])]

    if (unified["score_category"] == "potency").sum() == 0:
        raise ValueError("No potency-like score rows were found in unified screening scores; Step-13D cannot proceed without potency predictions.")
    if cfg.score_components.use_selectivity_score and (unified["score_category"] == "selectivity").sum() == 0:
        raise ValueError(
            "script_13d.score_components.use_selectivity_score is true but no selectivity-like score rows were found in unified screening scores."
        )
    if (unified["score_category"] == "selectivity").sum() == 0:
        warnings_list.append("No selectivity-like score rows detected; selectivity components will be recorded as missing/neutral if disabled in config.")

    inputs = {
        "unified": unified.sort_values(
            ["screening_compound_id", "target_chembl_id", "score_category", "model_family", "model_name", "predicted_value_type"],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True),
        "classical_scores": normalize_columns(pd.read_csv(cfg.input_classical_scores_path)),
        "deep_scores": normalize_columns(pd.read_csv(cfg.input_deep_scores_path)),
        "causal_scores": normalize_columns(pd.read_csv(cfg.input_causal_scores_path)),
        "screening_library": normalize_columns(pd.read_csv(cfg.input_screening_library_path)),
        "classical_features": normalize_columns(pd.read_csv(cfg.input_classical_feature_path)),
        "environment_features": normalize_columns(pd.read_csv(cfg.input_environment_feature_path)),
        "training_annotated": load_optional_table(cfg.input_training_annotated_long_path, "training annotated long table", warnings_list),
        "compound_env": load_optional_table(cfg.input_compound_env_path, "compound environment annotation table", warnings_list),
    }
    return inputs, warnings_list


def build_base_entity_table(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    score_base = inputs["unified"][[col for col in ["screening_compound_id", "standardized_smiles", "source_library_name"] if col in inputs["unified"].columns]].drop_duplicates()
    library = inputs["screening_library"].copy()
    ensure_columns(library, {"screening_compound_id", "standardized_smiles"}, "merged screening library")
    if "source_library_name" not in library.columns:
        library["source_library_name"] = ""
    library = library[[col for col in ["screening_compound_id", "standardized_smiles", "source_library_name"] + [c for c in library.columns if c not in {"screening_compound_id", "standardized_smiles", "source_library_name"}] ]].drop_duplicates(subset=["screening_compound_id", "standardized_smiles"], keep="first")
    base = score_base.merge(library, on=["screening_compound_id", "standardized_smiles", "source_library_name"], how="outer")
    base["screening_compound_id"] = normalize_text_series(base["screening_compound_id"])
    base["standardized_smiles"] = normalize_text_series(base["standardized_smiles"])
    base["source_library_name"] = normalize_text_series(base.get("source_library_name", pd.Series("", index=base.index)))
    return base.sort_values(["screening_compound_id", "standardized_smiles"], kind="mergesort").reset_index(drop=True)


def build_family_summary(unified: pd.DataFrame) -> pd.DataFrame:
    logging.info("Resolving potency/selectivity score types and building family-specific summary tables.")
    group_cols = ["screening_compound_id", "standardized_smiles", "source_library_name", "target_chembl_id", "score_category", "predicted_value_type", "model_family"]
    summary = (
        unified.groupby(group_cols, dropna=False, sort=True)["predicted_value"]
        .agg([("family_mean_prediction", "mean"), ("family_median_prediction", "median"), ("family_min_prediction", "min"), ("family_max_prediction", "max"), ("family_std_prediction", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0)) if len(s) > 1 else 0.0), ("number_of_models_contributing", "count")])
        .reset_index()
    )
    return summary.sort_values(group_cols, kind="mergesort", na_position="last").reset_index(drop=True)


def build_consensus_summary(unified: pd.DataFrame, family_summary: pd.DataFrame) -> pd.DataFrame:
    logging.info("Computing cross-family consensus and disagreement summaries.")
    key_cols = ["screening_compound_id", "standardized_smiles", "source_library_name", "target_chembl_id", "score_category", "predicted_value_type"]
    family_means = family_summary.pivot_table(index=key_cols, columns="model_family", values="family_mean_prediction", aggfunc="first").reset_index()
    family_stds = family_summary.pivot_table(index=key_cols, columns="model_family", values="family_std_prediction", aggfunc="first").reset_index()
    family_means.columns = [col if isinstance(col, str) else col for col in family_means.columns]
    family_stds.columns = [col if isinstance(col, str) else col for col in family_stds.columns]
    for fam in FAMILY_ORDER:
        if fam not in family_means.columns:
            family_means[fam] = np.nan
        if fam not in family_stds.columns:
            family_stds[fam] = np.nan
    consensus = family_means.merge(family_stds, on=key_cols, how="left", suffixes=("", "_within"))
    family_value_cols = list(FAMILY_ORDER)
    consensus["cross_family_mean_prediction"] = consensus[family_value_cols].mean(axis=1, skipna=True)
    consensus["cross_family_median_prediction"] = consensus[family_value_cols].median(axis=1, skipna=True)
    consensus["cross_family_std_prediction"] = consensus[family_value_cols].std(axis=1, ddof=0, skipna=True).fillna(0.0)
    consensus["max_family_disagreement"] = consensus[family_value_cols].max(axis=1, skipna=True) - consensus[family_value_cols].min(axis=1, skipna=True)
    consensus["causal_minus_deep_difference"] = consensus["causal"] - consensus["deep"]
    consensus["causal_minus_classical_difference"] = consensus["causal"] - consensus["classical"]
    consensus["deep_minus_classical_difference"] = consensus["deep"] - consensus["classical"]
    for fam in FAMILY_ORDER:
        source_col = f"{fam}_within"
        if source_col in consensus.columns:
            consensus[f"within_{fam}_std"] = consensus[source_col].fillna(0.0)
            consensus = consensus.drop(columns=[source_col])
        else:
            consensus[f"within_{fam}_std"] = 0.0
        consensus = consensus.rename(columns={fam: f"{fam}_family_mean_prediction"})
    consensus["families_with_support"] = consensus[[f"{fam}_family_mean_prediction" for fam in FAMILY_ORDER]].notna().sum(axis=1)
    return consensus.sort_values(key_cols, kind="mergesort", na_position="last").reset_index(drop=True)


def build_uncertainty_summary(unified: pd.DataFrame, consensus: pd.DataFrame, family_summary: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    logging.info("Constructing uncertainty proxies from disagreement patterns; these are not formal Bayesian uncertainties.")
    key_cols = ["screening_compound_id", "standardized_smiles", "source_library_name", "target_chembl_id", "score_category", "predicted_value_type"]
    model_std = (
        unified.groupby(key_cols, dropna=False, sort=True)["predicted_value"]
        .agg(cross_model_std_proxy=lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0)) if len(s) > 1 else 0.0, number_of_prediction_rows="count")
        .reset_index()
    )
    family_support = family_summary.groupby(key_cols, dropna=False, sort=True)["number_of_models_contributing"].sum().reset_index(name="number_of_models_contributing_total")
    top_by_family = family_summary.copy()
    top_by_family["sort_value"] = top_by_family["family_max_prediction"]
    top_noncausal = (
        top_by_family[top_by_family["model_family"].isin(["classical", "deep"])]
        .groupby(key_cols, dropna=False, sort=True)["sort_value"].max().reset_index(name="best_non_causal_prediction")
    )
    top_causal = top_by_family[top_by_family["model_family"] == "causal"].groupby(key_cols, dropna=False, sort=True)["sort_value"].max().reset_index(name="best_causal_prediction")
    summary = consensus.merge(model_std, on=key_cols, how="left").merge(family_support, on=key_cols, how="left").merge(top_noncausal, on=key_cols, how="left").merge(top_causal, on=key_cols, how="left")
    summary["best_causal_minus_best_non_causal_prediction"] = summary["best_causal_prediction"] - summary["best_non_causal_prediction"]
    within_cols = [f"within_{fam}_std" for fam in FAMILY_ORDER]
    summary["within_family_disagreement_proxy"] = summary[within_cols].fillna(0.0).max(axis=1)
    summary["family_disagreement_proxy"] = summary["max_family_disagreement"].fillna(0.0)
    summary["model_consensus_proxy"] = 1.0 / (1.0 + summary["cross_family_std_prediction"].fillna(0.0))
    summary["insufficient_model_support_flag"] = (summary["number_of_prediction_rows"].fillna(0) < 2).astype(int)
    summary["uncertainty_proxy"] = 0.0
    if cfg.uncertainty.use_cross_model_std:
        summary["uncertainty_proxy"] += summary["cross_model_std_proxy"].fillna(0.0)
    if cfg.uncertainty.use_within_family_disagreement:
        summary["uncertainty_proxy"] += summary["within_family_disagreement_proxy"].fillna(0.0)
    if cfg.uncertainty.use_family_disagreement:
        summary["uncertainty_proxy"] += summary["family_disagreement_proxy"].fillna(0.0)
    if cfg.uncertainty.use_top_model_gap:
        summary["uncertainty_proxy"] += summary["best_causal_minus_best_non_causal_prediction"].abs().fillna(0.0)
    summary["uncertainty_proxy"] += summary["insufficient_model_support_flag"].astype(float)
    summary["uncertainty_proxy_policy_note"] = "Proxy only; derived from model disagreement/support and not a calibrated posterior uncertainty."
    return summary.sort_values(key_cols, kind="mergesort", na_position="last").reset_index(drop=True)


def _extract_scaffold_reference_sets(training_annotated: pd.DataFrame, compound_env: pd.DataFrame) -> tuple[set[str], set[str]]:
    scaffolds: set[str] = set()
    generic_scaffolds: set[str] = set()
    for df in [training_annotated, compound_env]:
        if df.empty:
            continue
        for column, target_set in [("murcko_scaffold", scaffolds), ("generic_murcko_scaffold", generic_scaffolds), ("generic_scaffold", generic_scaffolds), ("scaffold", scaffolds)]:
            if column in df.columns:
                values = df[column].dropna().astype(str).str.strip()
                target_set.update(v for v in values if v)
    return scaffolds, generic_scaffolds


def _select_numeric_descriptor_columns(feature_df: pd.DataFrame) -> list[str]:
    protected = {"screening_compound_id", "standardized_smiles", "source_library_name", "target_chembl_id"}
    return [col for col in feature_df.columns if col not in protected and pd.api.types.is_numeric_dtype(feature_df[col])]


def build_applicability_summary(base_compounds: pd.DataFrame, inputs: dict[str, pd.DataFrame], cfg: AppConfig) -> pd.DataFrame:
    logging.info("Constructing applicability-domain proxy features using screening features and optional training references.")
    screening_features = inputs["classical_features"].copy()
    ensure_columns(screening_features, {"screening_compound_id", "standardized_smiles"}, "screening classical features")
    if "source_library_name" not in screening_features.columns:
        screening_features["source_library_name"] = ""
    env_features = inputs["environment_features"].copy()
    if not env_features.empty:
        for required in ["screening_compound_id", "standardized_smiles"]:
            if required not in env_features.columns:
                raise ValueError(f"screening environment features is missing required columns: {required}")

    training = inputs["training_annotated"].copy()
    compound_env = inputs["compound_env"].copy()
    numeric_screening_cols = _select_numeric_descriptor_columns(screening_features)
    numeric_training_cols = [col for col in numeric_screening_cols if not training.empty and col in training.columns and pd.api.types.is_numeric_dtype(training[col])]
    merged = base_compounds.merge(screening_features, on=["screening_compound_id", "standardized_smiles", "source_library_name"], how="left", suffixes=("", "_classical"))
    if not env_features.empty:
        env_keep = [c for c in env_features.columns if c not in merged.columns or c in {"screening_compound_id", "standardized_smiles"}]
        merged = merged.merge(env_features[env_keep], on=["screening_compound_id", "standardized_smiles"], how="left")

    if numeric_training_cols:
        training_numeric = training[numeric_training_cols].apply(pd.to_numeric, errors="coerce")
        train_median = training_numeric.median(axis=0, skipna=True)
        train_std = training_numeric.std(axis=0, ddof=0, skipna=True).replace(0, np.nan)
        train_min = training_numeric.min(axis=0, skipna=True)
        train_max = training_numeric.max(axis=0, skipna=True)
        screening_numeric = merged[numeric_training_cols].apply(pd.to_numeric, errors="coerce")
        violation_matrix = screening_numeric.lt(train_min, axis=1) | screening_numeric.gt(train_max, axis=1)
        merged["descriptor_range_violation_count"] = violation_matrix.sum(axis=1).astype(float)
        z = (screening_numeric - train_median) / train_std
        merged["descriptor_distance_proxy"] = np.sqrt((z.fillna(0.0) ** 2).mean(axis=1))
    else:
        merged["descriptor_range_violation_count"] = np.nan
        merged["descriptor_distance_proxy"] = np.nan

    murcko_col = first_present(merged, ["murcko_scaffold", "scaffold"])
    generic_col = first_present(merged, ["generic_murcko_scaffold", "generic_scaffold"])
    training_scaffolds, training_generic_scaffolds = _extract_scaffold_reference_sets(training, compound_env)
    if murcko_col:
        merged["murcko_scaffold"] = normalize_text_series(merged[murcko_col])
        merged["scaffold_novelty_flag"] = (~merged["murcko_scaffold"].isin(training_scaffolds) & merged["murcko_scaffold"].ne("")).astype(int)
    else:
        merged["murcko_scaffold"] = ""
        merged["scaffold_novelty_flag"] = np.nan
    if generic_col:
        merged["generic_scaffold"] = normalize_text_series(merged[generic_col])
        merged["generic_scaffold_novelty_flag"] = (~merged["generic_scaffold"].isin(training_generic_scaffolds) & merged["generic_scaffold"].ne("")).astype(int)
    else:
        merged["generic_scaffold"] = ""
        merged["generic_scaffold_novelty_flag"] = np.nan

    physchem_candidates = [col for col in numeric_screening_cols if any(token in col.lower() for token in ["mw", "molecular_weight", "logp", "tpsa", "hba", "hbd", "rot", "ring", "heavy_atom"])]
    training_physchem_cols = [col for col in physchem_candidates if not training.empty and col in training.columns]
    if training_physchem_cols:
        training_phys = training[training_physchem_cols].apply(pd.to_numeric, errors="coerce")
        screen_phys = merged[training_physchem_cols].apply(pd.to_numeric, errors="coerce")
        out_of_range = screen_phys.lt(training_phys.min(axis=0, skipna=True), axis=1) | screen_phys.gt(training_phys.max(axis=0, skipna=True), axis=1)
        merged["out_of_range_physchem_flag"] = out_of_range.any(axis=1).astype(int)
    else:
        merged["out_of_range_physchem_flag"] = np.nan

    merged["feature_input_missing_flag"] = merged[numeric_screening_cols].isna().all(axis=1).astype(int) if numeric_screening_cols else 1
    merged["applicability_proxy_policy_note"] = (
        "Applicability values are proxy indicators based on descriptor ranges/distances and scaffold novelty; they are not formal domain certification."
    )
    merged["applicability_penalty_raw"] = 0.0
    if cfg.applicability.use_feature_range_violations:
        merged["applicability_penalty_raw"] += merged["descriptor_range_violation_count"].fillna(0.0)
    if cfg.applicability.use_descriptor_distance_proxy:
        merged["applicability_penalty_raw"] += merged["descriptor_distance_proxy"].fillna(0.0)
    if cfg.applicability.use_scaffold_novelty_flag:
        merged["applicability_penalty_raw"] += merged["scaffold_novelty_flag"].fillna(0.0)
    if cfg.applicability.use_out_of_range_physchem_flag:
        merged["applicability_penalty_raw"] += merged["out_of_range_physchem_flag"].fillna(0.0)
    keep_cols = [
        "screening_compound_id",
        "standardized_smiles",
        "source_library_name",
        "murcko_scaffold",
        "generic_scaffold",
        "descriptor_range_violation_count",
        "descriptor_distance_proxy",
        "scaffold_novelty_flag",
        "generic_scaffold_novelty_flag",
        "out_of_range_physchem_flag",
        "feature_input_missing_flag",
        "applicability_penalty_raw",
        "applicability_proxy_policy_note",
    ]
    return merged[keep_cols].drop_duplicates(subset=["screening_compound_id", "standardized_smiles", "source_library_name"], keep="first").sort_values(["screening_compound_id", "standardized_smiles"], kind="mergesort").reset_index(drop=True)


def build_diversity_summary(base_compounds: pd.DataFrame, applicability: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    logging.info("Generating diversity-readiness signals for later shortlist generation.")
    diversity = base_compounds[["screening_compound_id", "standardized_smiles", "source_library_name"]].drop_duplicates().merge(
        applicability[["screening_compound_id", "standardized_smiles", "source_library_name", "murcko_scaffold", "generic_scaffold", "scaffold_novelty_flag"]],
        on=["screening_compound_id", "standardized_smiles", "source_library_name"],
        how="left",
    )
    diversity["murcko_scaffold"] = normalize_text_series(diversity.get("murcko_scaffold", pd.Series("", index=diversity.index)))
    diversity["generic_scaffold"] = normalize_text_series(diversity.get("generic_scaffold", pd.Series("", index=diversity.index)))
    scaffold_counts = diversity.groupby("murcko_scaffold", dropna=False, sort=True).size().reset_index(name="screening_scaffold_frequency")
    diversity = diversity.merge(scaffold_counts, on="murcko_scaffold", how="left")
    diversity["screening_scaffold_frequency"] = diversity["screening_scaffold_frequency"].fillna(0).astype(int)
    diversity["scaffold_group_id"] = np.where(diversity["murcko_scaffold"].ne(""), diversity["murcko_scaffold"], diversity["standardized_smiles"])
    if cfg.diversity.compute_fingerprint_clusters:
        diversity["fingerprint_cluster_id"] = diversity["scaffold_group_id"]
    else:
        diversity["fingerprint_cluster_id"] = "not_computed"
    diversity["diversity_bonus_raw"] = 1.0 / diversity["screening_scaffold_frequency"].replace(0, np.nan)
    diversity["diversity_bonus_raw"] = diversity["diversity_bonus_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    diversity["diversity_bonus_policy_note"] = (
        "Diversity component is a preparatory placeholder signal only; final diversity-constrained shortlist bucketing is deferred to Step-13E."
        if cfg.diversity.diversity_bonus_is_placeholder_only
        else "Diversity signal computed for downstream selection support."
    )
    return diversity.sort_values(["screening_compound_id", "standardized_smiles"], kind="mergesort").reset_index(drop=True)


def aggregate_component_from_preference(ranking: pd.DataFrame, prefix: str, preference: tuple[str, ...]) -> tuple[pd.Series, pd.Series]:
    source = pd.Series("", index=ranking.index, dtype="string")
    value = pd.Series(np.nan, index=ranking.index, dtype=float)
    for family in preference:
        candidate_cols = [f"{family}_family_mean_prediction_{prefix}", f"{family}_family_mean_prediction"]
        col = next((candidate for candidate in candidate_cols if candidate in ranking.columns), None)
        if col is None:
            continue
        mask = value.isna() & ranking[col].notna()
        value.loc[mask] = pd.to_numeric(ranking.loc[mask, col], errors="coerce")
        source.loc[mask] = family
    fallback_col = f"cross_family_mean_prediction_{prefix}"
    if fallback_col in ranking.columns:
        mask = value.isna() & ranking[fallback_col].notna()
        value.loc[mask] = pd.to_numeric(ranking.loc[mask, fallback_col], errors="coerce")
        source.loc[mask] = source.loc[mask].replace("", f"cross_family_{prefix}")
    return value, source


def pivot_summary_by_category(df: pd.DataFrame, value_columns: list[str], suffix: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    key_cols = ["screening_compound_id", "standardized_smiles", "source_library_name", "target_chembl_id"]
    for category in ["potency", "selectivity", "auxiliary"]:
        subset = df[df["score_category"] == category].copy()
        if subset.empty:
            continue
        rename_map = {col: f"{col}_{category}{suffix}" for col in value_columns}
        subset = subset[key_cols + value_columns].rename(columns=rename_map)
        frames.append(subset)
    if not frames:
        return pd.DataFrame(columns=key_cols)
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=key_cols, how="outer")
    return out


def build_failed_rows(base_pairs: pd.DataFrame, ranking: pd.DataFrame, applicability: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    logging.info("Collecting failed or incomplete ranking rows.")
    failed_records: list[dict[str, Any]] = []
    rank_lookup = ranking.set_index(["screening_compound_id", "standardized_smiles", "target_chembl_id"], drop=False)
    app_lookup = applicability.set_index(["screening_compound_id", "standardized_smiles"], drop=False)
    for row in base_pairs.itertuples(index=False):
        key = (row.screening_compound_id, row.standardized_smiles, row.target_chembl_id)
        reasons: list[str] = []
        if key not in rank_lookup.index:
            reasons.append("missing_model_scores")
            record = {
                "screening_compound_id": row.screening_compound_id,
                "standardized_smiles": row.standardized_smiles,
                "source_library_name": getattr(row, "source_library_name", ""),
                "target_chembl_id": row.target_chembl_id,
                "failure_reason": ";".join(reasons),
                "model_context": "no_rank_row_created",
            }
            failed_records.append(record)
            continue
        rank_row = rank_lookup.loc[key]
        if isinstance(rank_row, pd.DataFrame):
            rank_row = rank_row.iloc[0]
        if pd.isna(rank_row.get("potency_component_raw")):
            reasons.append("missing_potency_component")
        if cfg.score_components.use_selectivity_score and pd.isna(rank_row.get("selectivity_component_raw")):
            reasons.append("missing_selectivity_component")
        if int(rank_row.get("families_with_support_potency", 0) or 0) < 1:
            reasons.append("incomplete_consensus_support")
        app_key = (row.screening_compound_id, row.standardized_smiles)
        if app_key not in app_lookup.index:
            reasons.append("missing_feature_inputs")
        else:
            app_row = app_lookup.loc[app_key]
            if isinstance(app_row, pd.DataFrame):
                app_row = app_row.iloc[0]
            if int(app_row.get("feature_input_missing_flag", 0) or 0) == 1:
                reasons.append("failed_applicability_calculation")
        if reasons:
            failed_records.append(
                {
                    "screening_compound_id": row.screening_compound_id,
                    "standardized_smiles": row.standardized_smiles,
                    "source_library_name": getattr(row, "source_library_name", ""),
                    "target_chembl_id": row.target_chembl_id,
                    "failure_reason": ";".join(sorted(set(reasons))),
                    "model_context": f"potency_source={rank_row.get('potency_component_source', '')};selectivity_source={rank_row.get('selectivity_component_source', '')}",
                }
            )
    failed = pd.DataFrame(failed_records)
    if failed.empty:
        return pd.DataFrame(columns=["screening_compound_id", "standardized_smiles", "source_library_name", "target_chembl_id", "failure_reason", "model_context"])
    return failed.sort_values(["screening_compound_id", "target_chembl_id", "failure_reason"], kind="mergesort").reset_index(drop=True)


def build_rankings(inputs: dict[str, pd.DataFrame], cfg: AppConfig) -> dict[str, pd.DataFrame]:
    unified = inputs["unified"]
    family_summary = build_family_summary(unified)
    consensus = build_consensus_summary(unified, family_summary)
    uncertainty = build_uncertainty_summary(unified, consensus, family_summary, cfg)
    base_compounds = build_base_entity_table(inputs)
    applicability = build_applicability_summary(base_compounds, inputs, cfg)
    diversity = build_diversity_summary(base_compounds, applicability, cfg)

    key_cols = ["screening_compound_id", "standardized_smiles", "source_library_name", "target_chembl_id"]
    consensus_pivot = pivot_summary_by_category(
        consensus,
        [
            "cross_family_mean_prediction",
            "cross_family_median_prediction",
            "cross_family_std_prediction",
            "max_family_disagreement",
            "causal_minus_deep_difference",
            "causal_minus_classical_difference",
            "deep_minus_classical_difference",
            "within_classical_std",
            "within_deep_std",
            "within_causal_std",
            "families_with_support",
            "classical_family_mean_prediction",
            "deep_family_mean_prediction",
            "causal_family_mean_prediction",
        ],
        suffix="",
    )
    uncertainty_pivot = pivot_summary_by_category(
        uncertainty,
        [
            "cross_model_std_proxy",
            "number_of_prediction_rows",
            "number_of_models_contributing_total",
            "best_non_causal_prediction",
            "best_causal_prediction",
            "best_causal_minus_best_non_causal_prediction",
            "within_family_disagreement_proxy",
            "family_disagreement_proxy",
            "model_consensus_proxy",
            "insufficient_model_support_flag",
            "uncertainty_proxy",
        ],
        suffix="",
    )
    base_pairs = unified[key_cols].drop_duplicates().sort_values(key_cols, kind="mergesort", na_position="last").reset_index(drop=True)
    ranking = base_pairs.merge(consensus_pivot, on=key_cols, how="left").merge(uncertainty_pivot, on=key_cols, how="left")
    ranking = ranking.merge(applicability, on=["screening_compound_id", "standardized_smiles", "source_library_name"], how="left")
    ranking = ranking.merge(diversity, on=["screening_compound_id", "standardized_smiles", "source_library_name"], how="left", suffixes=("", "_diversity"))

    ranking["potency_component_raw"], ranking["potency_component_source"] = aggregate_component_from_preference(ranking, "potency", cfg.potency_source_preference)
    ranking["selectivity_component_raw"], ranking["selectivity_component_source"] = aggregate_component_from_preference(ranking, "selectivity", cfg.selectivity_source_preference)
    ranking["uncertainty_component_raw"] = ranking.get("uncertainty_proxy_potency", pd.Series(np.nan, index=ranking.index)).copy()
    if "uncertainty_proxy_selectivity" in ranking.columns:
        ranking["uncertainty_component_raw"] = pd.concat([ranking["uncertainty_component_raw"], ranking["uncertainty_proxy_selectivity"]], axis=1).max(axis=1, skipna=True)
    ranking["applicability_component_raw"] = ranking["applicability_penalty_raw"]
    ranking["diversity_component_raw"] = ranking["diversity_bonus_raw"]

    ranking["normalized_potency_component"] = robust_rank_normalize(ranking["potency_component_raw"], larger_is_better=cfg.normalization.larger_is_better_for_potency)
    ranking["normalized_selectivity_component"] = robust_rank_normalize(ranking["selectivity_component_raw"], larger_is_better=cfg.normalization.larger_is_better_for_selectivity)
    ranking["normalized_uncertainty_penalty"] = robust_rank_normalize(ranking["uncertainty_component_raw"], larger_is_better=True)
    ranking["normalized_applicability_penalty"] = robust_rank_normalize(ranking["applicability_component_raw"], larger_is_better=True)
    ranking["normalized_diversity_bonus"] = robust_rank_normalize(ranking["diversity_component_raw"], larger_is_better=True)

    ranking["final_strategic_score"] = 0.0
    if cfg.score_components.use_potency_score:
        ranking["final_strategic_score"] += cfg.score_weights.potency_weight * ranking["normalized_potency_component"].fillna(0.0)
    if cfg.score_components.use_selectivity_score:
        ranking["final_strategic_score"] += cfg.score_weights.selectivity_weight * ranking["normalized_selectivity_component"].fillna(0.0)
    if cfg.score_components.use_uncertainty_penalty:
        ranking["final_strategic_score"] -= cfg.score_weights.uncertainty_penalty_weight * ranking["normalized_uncertainty_penalty"].fillna(0.0)
    if cfg.score_components.use_applicability_penalty:
        ranking["final_strategic_score"] -= cfg.score_weights.applicability_penalty_weight * ranking["normalized_applicability_penalty"].fillna(0.0)
    if cfg.score_components.use_diversity_bonus_placeholder:
        ranking["final_strategic_score"] += cfg.score_weights.diversity_bonus_weight * ranking["normalized_diversity_bonus"].fillna(0.0)

    ranking["is_primary_target_flag"] = ranking["target_chembl_id"].isin(cfg.primary_target_chembl_ids).astype(int)
    ranking = ranking.sort_values(["target_chembl_id", "final_strategic_score", "potency_component_raw", "screening_compound_id"], ascending=[True, False, False, True], kind="mergesort").reset_index(drop=True)
    ranking["rank_within_target"] = ranking.groupby("target_chembl_id", dropna=False, sort=False)["final_strategic_score"].rank(method="first", ascending=False).astype(int)
    ranking["global_rank"] = deterministic_rank_desc(ranking["final_strategic_score"])
    ranking["ranking_mode"] = cfg.ranking_mode
    ranking["normalization_method"] = cfg.normalization.method
    ranking["potency_weight"] = cfg.score_weights.potency_weight
    ranking["selectivity_weight"] = cfg.score_weights.selectivity_weight
    ranking["uncertainty_penalty_weight"] = cfg.score_weights.uncertainty_penalty_weight
    ranking["applicability_penalty_weight"] = cfg.score_weights.applicability_penalty_weight
    ranking["diversity_bonus_weight"] = cfg.score_weights.diversity_bonus_weight
    ranking["provenance_score_inputs"] = str(cfg.input_unified_scores_path)
    ranking["uncertainty_policy_note"] = "Uncertainty terms are disagreement/support proxies only; they are not formal Bayesian uncertainty estimates."
    ranking["applicability_policy_note"] = ranking["applicability_proxy_policy_note"]
    ranking["diversity_policy_note"] = ranking["diversity_bonus_policy_note"]

    compound_summary = (
        ranking.groupby(["screening_compound_id", "standardized_smiles", "source_library_name"], dropna=False, sort=True)
        .agg(
            best_target_chembl_id=("target_chembl_id", lambda s: s.iloc[int(np.argmax(ranking.loc[s.index, "final_strategic_score"].to_numpy()))] if len(s) else ""),
            best_target_score=("final_strategic_score", "max"),
            mean_target_score=("final_strategic_score", "mean"),
            best_causal_potency=("causal_family_mean_prediction_potency", "max"),
            best_selectivity_estimate=("selectivity_component_raw", "max"),
            max_uncertainty_penalty=("uncertainty_component_raw", "max"),
            mean_applicability_penalty=("applicability_component_raw", "mean"),
            diversity_bonus_placeholder=("diversity_component_raw", "max"),
            n_targets_ranked=("target_chembl_id", pd.Series.nunique),
            n_primary_targets_ranked=("is_primary_target_flag", "sum"),
            final_compound_level_strategic_score=("final_strategic_score", "max"),
        )
        .reset_index()
    )
    compound_summary = compound_summary.sort_values(["final_compound_level_strategic_score", "best_target_score", "screening_compound_id"], ascending=[False, False, True], kind="mergesort").reset_index(drop=True)
    compound_summary["overall_compound_rank"] = deterministic_rank_desc(compound_summary["final_compound_level_strategic_score"])
    compound_summary = compound_summary.merge(applicability, on=["screening_compound_id", "standardized_smiles", "source_library_name"], how="left")
    compound_summary = compound_summary.merge(diversity, on=["screening_compound_id", "standardized_smiles", "source_library_name"], how="left", suffixes=("", "_diversity"))

    failed_rows = build_failed_rows(base_pairs, ranking, applicability, cfg)
    return {
        "family_summary": family_summary,
        "consensus": consensus,
        "uncertainty": uncertainty,
        "applicability": applicability,
        "diversity": diversity,
        "compound_target_ranking": ranking,
        "compound_summary_ranking": compound_summary,
        "failed_rows": failed_rows,
        "base_pairs": base_pairs,
    }


def build_manifest(cfg: AppConfig, outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    manifest_rows = [
        {
            "asset_id": "compound_target_strategic_ranking",
            "asset_type": "strategic_ranking_table",
            "file_path": str(cfg.output_compound_target_ranking_path),
            "row_count": int(len(outputs["compound_target_ranking"])),
            "ranking_scope": "compound_target",
            "notes": "Composite strategic ranking per compound-target pair.",
        },
        {
            "asset_id": "compound_level_strategic_ranking",
            "asset_type": "strategic_ranking_table",
            "file_path": str(cfg.output_compound_summary_ranking_path),
            "row_count": int(len(outputs["compound_summary_ranking"])),
            "ranking_scope": "compound",
            "notes": "Compound-level rollup derived from target-aware strategic scores.",
        },
        {
            "asset_id": "screening_uncertainty_summary",
            "asset_type": "uncertainty_proxy_table",
            "file_path": str(cfg.output_uncertainty_summary_path),
            "row_count": int(len(outputs["uncertainty"])),
            "ranking_scope": "compound_target_score_category",
            "notes": "Disagreement/support proxy features; not formal calibrated uncertainty.",
        },
        {
            "asset_id": "screening_applicability_summary",
            "asset_type": "applicability_proxy_table",
            "file_path": str(cfg.output_applicability_summary_path),
            "row_count": int(len(outputs["applicability"])),
            "ranking_scope": "compound",
            "notes": "Descriptor/scaffold applicability proxies; not domain certification.",
        },
        {
            "asset_id": "screening_consensus_summary",
            "asset_type": "consensus_summary_table",
            "file_path": str(cfg.output_consensus_summary_path),
            "row_count": int(len(outputs["consensus"])),
            "ranking_scope": "compound_target_score_category",
            "notes": "Cross-family consensus and disagreement summaries.",
        },
    ]
    if cfg.save_intermediate_component_tables:
        manifest_rows.append(
            {
                "asset_id": "family_score_summary",
                "asset_type": "component_intermediate",
                "file_path": str(cfg.output_ranking_root / "family_score_summaries.csv"),
                "row_count": int(len(outputs["family_summary"])),
                "ranking_scope": "compound_target_score_category_family",
                "notes": "Family-specific potency/selectivity summary statistics.",
            }
        )
        manifest_rows.append(
            {
                "asset_id": "diversity_signal_summary",
                "asset_type": "component_intermediate",
                "file_path": str(cfg.output_ranking_root / "screening_diversity_signals.csv"),
                "row_count": int(len(outputs["diversity"])),
                "ranking_scope": "compound",
                "notes": "Preparatory diversity-readiness signals for Step-13E.",
            }
        )
    if cfg.save_failed_rows:
        manifest_rows.append(
            {
                "asset_id": "failed_ranking_rows",
                "asset_type": "failed_rows",
                "file_path": str(cfg.output_ranking_root / "failed_strategic_ranking_rows.csv"),
                "row_count": int(len(outputs["failed_rows"])),
                "ranking_scope": "compound_target",
                "notes": "Rows with missing or incomplete ranking support.",
            }
        )
    return pd.DataFrame(manifest_rows).sort_values(["asset_id"], kind="mergesort").reset_index(drop=True)


def summarize_for_report(outputs: dict[str, pd.DataFrame], cfg: AppConfig, warnings_list: list[str], config_snapshot_path: Path | None) -> dict[str, Any]:
    ranking = outputs["compound_target_ranking"]
    uncertainty = outputs["uncertainty"]
    applicability = outputs["applicability"]
    report = {
        "script_name": SCRIPT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths_used": {
            "unified_scores": str(cfg.input_unified_scores_path),
            "classical_scores": str(cfg.input_classical_scores_path),
            "deep_scores": str(cfg.input_deep_scores_path),
            "causal_scores": str(cfg.input_causal_scores_path),
            "screening_library": str(cfg.input_screening_library_path),
            "classical_features": str(cfg.input_classical_feature_path),
            "environment_features": str(cfg.input_environment_feature_path),
            "training_annotated": str(cfg.input_training_annotated_long_path),
            "compound_environment_annotations": str(cfg.input_compound_env_path),
        },
        "scoring_files_used": [
            str(cfg.input_classical_scores_path),
            str(cfg.input_deep_scores_path),
            str(cfg.input_causal_scores_path),
            str(cfg.input_unified_scores_path),
        ],
        "total_screening_compounds_processed": int(outputs["compound_summary_ranking"]["screening_compound_id"].nunique()) if not outputs["compound_summary_ranking"].empty else 0,
        "total_ranked_compound_target_rows": int(len(outputs["compound_target_ranking"])),
        "total_ranked_compound_level_rows": int(len(outputs["compound_summary_ranking"])),
        "score_components_used": asdict(cfg.score_components),
        "weight_values_used": asdict(cfg.score_weights),
        "uncertainty_proxy_summary": {
            "min_uncertainty_proxy": float(uncertainty["uncertainty_proxy"].min()) if not uncertainty.empty else None,
            "median_uncertainty_proxy": float(uncertainty["uncertainty_proxy"].median()) if not uncertainty.empty else None,
            "max_uncertainty_proxy": float(uncertainty["uncertainty_proxy"].max()) if not uncertainty.empty else None,
            "insufficient_model_support_rows": int(uncertainty["insufficient_model_support_flag"].fillna(0).sum()) if not uncertainty.empty else 0,
        },
        "applicability_proxy_summary": {
            "mean_descriptor_distance_proxy": float(applicability["descriptor_distance_proxy"].dropna().mean()) if applicability["descriptor_distance_proxy"].notna().any() else None,
            "rows_with_scaffold_novelty_flag": int(applicability["scaffold_novelty_flag"].fillna(0).sum()) if "scaffold_novelty_flag" in applicability.columns else 0,
            "rows_with_out_of_range_physchem_flag": int(applicability["out_of_range_physchem_flag"].fillna(0).sum()) if "out_of_range_physchem_flag" in applicability.columns else 0,
        },
        "top_ranked_targets_covered": ranking.groupby("target_chembl_id", dropna=False).size().sort_values(ascending=False).head(10).reset_index(name="n_rows").to_dict(orient="records") if not ranking.empty else [],
        "failed_row_counts": {
            "failed_ranking_rows": int(len(outputs["failed_rows"])),
            "rows_missing_selectivity_component": int(ranking["selectivity_component_raw"].isna().sum()) if "selectivity_component_raw" in ranking.columns else 0,
            "rows_missing_potency_component": int(ranking["potency_component_raw"].isna().sum()) if "potency_component_raw" in ranking.columns else 0,
        },
        "warnings": warnings_list,
        "config_snapshot_reference": str(config_snapshot_path) if config_snapshot_path else "",
    }
    return report


def write_outputs(cfg: AppConfig, outputs: dict[str, pd.DataFrame], report: dict[str, Any]) -> None:
    logging.info("Writing ranking outputs, manifest, and report assets.")
    cfg.output_ranking_root.mkdir(parents=True, exist_ok=True)
    write_dataframe(outputs["compound_target_ranking"], cfg.output_compound_target_ranking_path)
    write_dataframe(outputs["compound_summary_ranking"], cfg.output_compound_summary_ranking_path)
    write_dataframe(outputs["uncertainty"], cfg.output_uncertainty_summary_path)
    write_dataframe(outputs["applicability"], cfg.output_applicability_summary_path)
    write_dataframe(outputs["consensus"], cfg.output_consensus_summary_path)
    if cfg.save_intermediate_component_tables:
        write_dataframe(outputs["family_summary"], cfg.output_ranking_root / "family_score_summaries.csv")
        write_dataframe(outputs["diversity"], cfg.output_ranking_root / "screening_diversity_signals.csv")
    if cfg.save_failed_rows:
        write_dataframe(outputs["failed_rows"], cfg.output_ranking_root / "failed_strategic_ranking_rows.csv")
    manifest = build_manifest(cfg, outputs)
    write_dataframe(manifest, cfg.output_manifest_path)
    write_json(report, cfg.output_report_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    raw_config = load_yaml(config_path)
    project_root = config_path.parent.resolve()
    cfg = AppConfig.from_dict(raw_config, project_root)
    if cfg.ranking_mode not in {"target_aware", "compound_only"}:
        raise ValueError(f"Unsupported script_13d.ranking_mode: {cfg.ranking_mode}")
    if cfg.normalization.method != "robust_rank":
        raise ValueError(f"Unsupported script_13d.normalization.method: {cfg.normalization.method}; only `robust_rank` is currently implemented.")
    log_path = setup_logging(cfg)
    config_snapshot_path = save_config_snapshot(raw_config, cfg)
    logging.info("Starting %s", SCRIPT_NAME)
    inputs, warnings_list = load_inputs(cfg)
    outputs = build_rankings(inputs, cfg)
    report = summarize_for_report(outputs, cfg, warnings_list, config_snapshot_path)
    write_outputs(cfg, outputs, report)
    logging.info("Completed %s successfully.", SCRIPT_NAME)
    logging.info("Log written to %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
