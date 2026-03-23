#!/usr/bin/env python3
"""Generate bucketed actionable screening shortlists from Step-13D rankings.

Script-13E is the final selection layer for the kinase causality QSAR screening
workflow. It consumes the strategic ranking outputs from Script-13D and turns
those ranking tables into deterministic, publication-grade shortlist buckets for
follow-up review, docking/refinement handoff, sourcing, and manuscript-ready
reporting.
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
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

SCRIPT_NAME = "13e_generate_screening_shortlist_buckets"
REQUIRED_SCRIPT_KEYS = {
    "input_compound_target_ranking_path",
    "input_compound_summary_ranking_path",
    "input_uncertainty_summary_path",
    "input_applicability_summary_path",
    "input_consensus_summary_path",
    "input_screening_library_path",
    "input_environment_feature_path",
    "input_provenance_path",
    "output_shortlist_root",
    "output_final_shortlist_path",
    "output_shortlist_rationale_path",
    "output_bucket_summary_path",
    "output_diversity_summary_path",
    "output_manifest_path",
    "output_report_path",
    "shortlist_mode",
    "primary_target_chembl_ids",
    "total_shortlist_size",
    "bucket_sizes",
    "bucket_rules",
    "deduplicate_across_buckets",
    "prioritize_higher_priority_bucket_order",
    "diversity_controls",
    "purchasability_controls",
    "save_bucket_specific_files",
    "save_target_specific_shortlists",
    "save_config_snapshot",
}
REQUIRED_COMPOUND_LEVEL_COLUMNS = {"screening_compound_id", "standardized_smiles", "final_compound_level_strategic_score"}
REQUIRED_COMPOUND_TARGET_COLUMNS = {"screening_compound_id", "standardized_smiles", "target_chembl_id", "final_strategic_score"}
DEFAULT_BUCKETS = (
    "high_confidence_selective_hits",
    "novel_scaffold_selective_hits",
    "diverse_exploratory_hits",
    "consensus_supported_fallback_hits",
)
NUMERIC_FALLBACK_HIGH = 1.0e12
TEXT_EMPTY = ""


@dataclass(frozen=True)
class DiversityControls:
    max_compounds_per_exact_scaffold: int
    max_compounds_per_generic_scaffold: int
    enforce_scaffold_diversity_within_bucket: bool
    use_fingerprint_diversity_selection: bool
    fingerprint_similarity_threshold: float


@dataclass(frozen=True)
class PurchasabilityControls:
    prefer_available_vendor_entries: bool
    prefer_non_missing_supplier_metadata: bool
    prefer_in_stock_if_available: bool


@dataclass(frozen=True)
class AppConfig:
    input_compound_target_ranking_path: Path
    input_compound_summary_ranking_path: Path
    input_uncertainty_summary_path: Path
    input_applicability_summary_path: Path
    input_consensus_summary_path: Path
    input_screening_library_path: Path
    input_environment_feature_path: Path
    input_provenance_path: Path
    output_shortlist_root: Path
    output_final_shortlist_path: Path
    output_shortlist_rationale_path: Path
    output_bucket_summary_path: Path
    output_diversity_summary_path: Path
    output_manifest_path: Path
    output_report_path: Path
    shortlist_mode: str
    primary_target_chembl_ids: tuple[str, ...]
    total_shortlist_size: int
    bucket_sizes: dict[str, int]
    bucket_rules: dict[str, dict[str, Any]]
    deduplicate_across_buckets: bool
    prioritize_higher_priority_bucket_order: tuple[str, ...]
    diversity_controls: DiversityControls
    purchasability_controls: PurchasabilityControls
    save_bucket_specific_files: bool
    save_target_specific_shortlists: bool
    save_config_snapshot: bool
    project_root: Path
    logs_dir: Path
    configs_used_dir: Path

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        section = raw.get("script_13e")
        if not isinstance(section, dict):
            raise ValueError("Missing required `script_13e` section in config.yaml.")
        missing = sorted(REQUIRED_SCRIPT_KEYS.difference(section))
        if missing:
            raise ValueError("Missing required script_13e config values: " + ", ".join(missing))

        def resolve(path_like: Any) -> Path:
            path = Path(str(path_like))
            return path if path.is_absolute() else project_root / path

        def parse_bool(value: Any, key: str) -> bool:
            if not isinstance(value, bool):
                raise ValueError(f"script_13e.{key} must be boolean; got {value!r}.")
            return value

        def parse_int(value: Any, key: str, minimum: int | None = None) -> int:
            try:
                parsed = int(value)
            except Exception as exc:
                raise ValueError(f"script_13e.{key} must be an integer; got {value!r}.") from exc
            if minimum is not None and parsed < minimum:
                raise ValueError(f"script_13e.{key} must be >= {minimum}; got {parsed}.")
            return parsed

        def parse_float(value: Any, key: str) -> float:
            try:
                return float(value)
            except Exception as exc:
                raise ValueError(f"script_13e.{key} must be numeric; got {value!r}.") from exc

        def parse_str_list(value: Any, key: str) -> tuple[str, ...]:
            if not isinstance(value, list):
                raise ValueError(f"script_13e.{key} must be a list.")
            return tuple(str(item).strip() for item in value if str(item).strip())

        def require_mapping(value: Any, key: str) -> dict[str, Any]:
            if not isinstance(value, dict):
                raise ValueError(f"script_13e.{key} must be a mapping.")
            return value

        bucket_sizes = require_mapping(section["bucket_sizes"], "bucket_sizes")
        bucket_rules = require_mapping(section["bucket_rules"], "bucket_rules")
        diversity_map = require_mapping(section["diversity_controls"], "diversity_controls")
        purch_map = require_mapping(section["purchasability_controls"], "purchasability_controls")
        priority_order = parse_str_list(section["prioritize_higher_priority_bucket_order"], "prioritize_higher_priority_bucket_order")
        if not priority_order:
            raise ValueError("script_13e.prioritize_higher_priority_bucket_order must include at least one bucket name.")
        unknown_priorities = [name for name in priority_order if name not in bucket_sizes]
        if unknown_priorities:
            raise ValueError("Bucket priority order contains names missing from bucket_sizes: " + ", ".join(unknown_priorities))
        for bucket in priority_order:
            if bucket not in bucket_rules:
                raise ValueError(f"Missing script_13e.bucket_rules.{bucket} configuration.")
        total_shortlist_size = parse_int(section["total_shortlist_size"], "total_shortlist_size", minimum=1)
        parsed_bucket_sizes = {str(k): parse_int(v, f"bucket_sizes.{k}", minimum=0) for k, v in bucket_sizes.items()}
        if sum(parsed_bucket_sizes.get(name, 0) for name in priority_order) != total_shortlist_size:
            raise ValueError(
                "Sum of configured bucket sizes in script_13e.bucket_sizes for the configured priority order must equal "
                f"script_13e.total_shortlist_size ({total_shortlist_size})."
            )

        shortlist_mode = str(section["shortlist_mode"]).strip().lower()
        if shortlist_mode not in {"compound_level", "compound_target"}:
            raise ValueError("script_13e.shortlist_mode must be either `compound_level` or `compound_target`.")

        return AppConfig(
            input_compound_target_ranking_path=resolve(section["input_compound_target_ranking_path"]),
            input_compound_summary_ranking_path=resolve(section["input_compound_summary_ranking_path"]),
            input_uncertainty_summary_path=resolve(section["input_uncertainty_summary_path"]),
            input_applicability_summary_path=resolve(section["input_applicability_summary_path"]),
            input_consensus_summary_path=resolve(section["input_consensus_summary_path"]),
            input_screening_library_path=resolve(section["input_screening_library_path"]),
            input_environment_feature_path=resolve(section["input_environment_feature_path"]),
            input_provenance_path=resolve(section["input_provenance_path"]),
            output_shortlist_root=resolve(section["output_shortlist_root"]),
            output_final_shortlist_path=resolve(section["output_final_shortlist_path"]),
            output_shortlist_rationale_path=resolve(section["output_shortlist_rationale_path"]),
            output_bucket_summary_path=resolve(section["output_bucket_summary_path"]),
            output_diversity_summary_path=resolve(section["output_diversity_summary_path"]),
            output_manifest_path=resolve(section["output_manifest_path"]),
            output_report_path=resolve(section["output_report_path"]),
            shortlist_mode=shortlist_mode,
            primary_target_chembl_ids=parse_str_list(section["primary_target_chembl_ids"], "primary_target_chembl_ids"),
            total_shortlist_size=total_shortlist_size,
            bucket_sizes=parsed_bucket_sizes,
            bucket_rules={str(k): require_mapping(v, f"bucket_rules.{k}") for k, v in bucket_rules.items()},
            deduplicate_across_buckets=parse_bool(section["deduplicate_across_buckets"], "deduplicate_across_buckets"),
            prioritize_higher_priority_bucket_order=priority_order,
            diversity_controls=DiversityControls(
                max_compounds_per_exact_scaffold=parse_int(diversity_map["max_compounds_per_exact_scaffold"], "diversity_controls.max_compounds_per_exact_scaffold", minimum=1),
                max_compounds_per_generic_scaffold=parse_int(diversity_map["max_compounds_per_generic_scaffold"], "diversity_controls.max_compounds_per_generic_scaffold", minimum=1),
                enforce_scaffold_diversity_within_bucket=parse_bool(diversity_map["enforce_scaffold_diversity_within_bucket"], "diversity_controls.enforce_scaffold_diversity_within_bucket"),
                use_fingerprint_diversity_selection=parse_bool(diversity_map["use_fingerprint_diversity_selection"], "diversity_controls.use_fingerprint_diversity_selection"),
                fingerprint_similarity_threshold=parse_float(diversity_map["fingerprint_similarity_threshold"], "diversity_controls.fingerprint_similarity_threshold"),
            ),
            purchasability_controls=PurchasabilityControls(
                prefer_available_vendor_entries=parse_bool(purch_map["prefer_available_vendor_entries"], "purchasability_controls.prefer_available_vendor_entries"),
                prefer_non_missing_supplier_metadata=parse_bool(purch_map["prefer_non_missing_supplier_metadata"], "purchasability_controls.prefer_non_missing_supplier_metadata"),
                prefer_in_stock_if_available=parse_bool(purch_map["prefer_in_stock_if_available"], "purchasability_controls.prefer_in_stock_if_available"),
            ),
            save_bucket_specific_files=parse_bool(section["save_bucket_specific_files"], "save_bucket_specific_files"),
            save_target_specific_shortlists=parse_bool(section["save_target_specific_shortlists"], "save_target_specific_shortlists"),
            save_config_snapshot=parse_bool(section["save_config_snapshot"], "save_config_snapshot"),
            project_root=project_root,
            logs_dir=project_root / "logs",
            configs_used_dir=project_root / "configs_used",
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.yaml", help="Path to config.yaml")
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must deserialize to a mapping: {path}")
    return data


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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_path = cfg.configs_used_dir / f"{SCRIPT_NAME}_{timestamp}.yaml"
    with snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config, handle, sort_keys=False)
    return snapshot_path


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def require_files(paths: dict[str, Path]) -> None:
    missing = [f"{label}: {path}" for label, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Step-13E input files:\n- " + "\n- ".join(missing))


def normalize_text_series(series: pd.Series) -> pd.Series:
    return series.fillna(TEXT_EMPTY).astype(str).str.strip()


def coalesce_columns(df: pd.DataFrame, candidates: list[str], default: Any = np.nan) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    if len(df.index) == 0:
        return pd.Series(dtype="object")
    return pd.Series([default] * len(df), index=df.index)


def first_present(candidates: list[str], available: pd.Index) -> str | None:
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def validate_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def load_inputs(cfg: AppConfig) -> tuple[dict[str, pd.DataFrame], list[str]]:
    require_files(
        {
            "compound_target_ranking": cfg.input_compound_target_ranking_path,
            "compound_summary_ranking": cfg.input_compound_summary_ranking_path,
            "uncertainty_summary": cfg.input_uncertainty_summary_path,
            "applicability_summary": cfg.input_applicability_summary_path,
            "consensus_summary": cfg.input_consensus_summary_path,
            "screening_library": cfg.input_screening_library_path,
            "environment_features": cfg.input_environment_feature_path,
            "provenance": cfg.input_provenance_path,
        }
    )
    logging.info("Loading Step-13D ranking inputs and Step-13A/B metadata.")
    inputs = {
        "compound_target_ranking": pd.read_csv(cfg.input_compound_target_ranking_path),
        "compound_summary_ranking": pd.read_csv(cfg.input_compound_summary_ranking_path),
        "uncertainty_summary": pd.read_csv(cfg.input_uncertainty_summary_path),
        "applicability_summary": pd.read_csv(cfg.input_applicability_summary_path),
        "consensus_summary": pd.read_csv(cfg.input_consensus_summary_path),
        "screening_library": pd.read_csv(cfg.input_screening_library_path),
        "environment_features": pd.read_csv(cfg.input_environment_feature_path),
        "provenance": pd.read_csv(cfg.input_provenance_path),
    }
    validate_columns(inputs["compound_target_ranking"], REQUIRED_COMPOUND_TARGET_COLUMNS, "Step-13D compound-target ranking")
    validate_columns(inputs["compound_summary_ranking"], REQUIRED_COMPOUND_LEVEL_COLUMNS, "Step-13D compound-level ranking")
    for name in ["uncertainty_summary", "consensus_summary"]:
        validate_columns(inputs[name], {"screening_compound_id", "standardized_smiles", "target_chembl_id"}, name)
    validate_columns(inputs["applicability_summary"], {"screening_compound_id", "standardized_smiles"}, "applicability_summary")
    validate_columns(inputs["screening_library"], {"screening_compound_id", "standardized_smiles"}, "screening_library")
    validate_columns(inputs["provenance"], {"screening_compound_id"}, "screening_library_provenance")
    warnings_list: list[str] = []
    return inputs, warnings_list


def build_entity_table(inputs: dict[str, pd.DataFrame], cfg: AppConfig, warnings_list: list[str]) -> pd.DataFrame:
    mode = cfg.shortlist_mode
    logging.info("Resolving shortlist mode: %s", mode)
    ranking = inputs["compound_summary_ranking"].copy() if mode == "compound_level" else inputs["compound_target_ranking"].copy()

    if mode == "compound_level":
        ranking = ranking.rename(
            columns={
                "final_compound_level_strategic_score": "final_strategic_score",
                "best_selectivity_estimate": "selectivity_component_raw",
                "best_causal_potency": "potency_component_raw",
                "max_uncertainty_penalty": "uncertainty_component_raw",
                "mean_applicability_penalty": "applicability_component_raw",
                "best_target_chembl_id": "target_chembl_id",
            }
        )
        ranking["shortlist_scope"] = "compound_level"
    else:
        ranking["shortlist_scope"] = "compound_target"

    library = inputs["screening_library"].copy()
    applicability = inputs["applicability_summary"].copy()
    provenance = inputs["provenance"].copy()
    env_features = inputs["environment_features"].copy()

    library["source_library_name"] = normalize_text_series(coalesce_columns(library, ["source_library_name", "library_name"], default="unknown_library"))
    library["standardized_smiles"] = normalize_text_series(library["standardized_smiles"])
    applicability["standardized_smiles"] = normalize_text_series(applicability["standardized_smiles"])
    ranking["standardized_smiles"] = normalize_text_series(ranking["standardized_smiles"])
    ranking["source_library_name"] = normalize_text_series(coalesce_columns(ranking, ["source_library_name"], default=""))

    merged = ranking.merge(
        applicability.drop_duplicates(subset=["screening_compound_id", "standardized_smiles"], keep="first"),
        on=["screening_compound_id", "standardized_smiles"],
        how="left",
        suffixes=("", "_app"),
    )
    merged = merged.merge(
        library.drop_duplicates(subset=["screening_compound_id", "standardized_smiles"], keep="first"),
        on=["screening_compound_id", "standardized_smiles"],
        how="left",
        suffixes=("", "_lib"),
    )
    merged = merged.merge(
        provenance.drop_duplicates(subset=["screening_compound_id"], keep="first"),
        on=["screening_compound_id"],
        how="left",
        suffixes=("", "_prov"),
    )
    if "screening_compound_id" in env_features.columns:
        env_subset_cols = [col for col in env_features.columns if col in {"screening_compound_id", "target_name"} or "environment" in col.lower()]
        if env_subset_cols:
            merged = merged.merge(env_features[env_subset_cols].drop_duplicates(subset=["screening_compound_id"], keep="first"), on="screening_compound_id", how="left")

    merged["source_library_name"] = normalize_text_series(coalesce_columns(merged, ["source_library_name", "source_library_name_lib", "source_library_name_prov"], default="unknown_library"))
    merged["target_chembl_id"] = normalize_text_series(coalesce_columns(merged, ["target_chembl_id"], default=""))
    merged["target_name"] = normalize_text_series(coalesce_columns(merged, ["target_name", "target_name_lib"], default=""))
    merged["final_strategic_score"] = pd.to_numeric(coalesce_columns(merged, ["final_strategic_score"]), errors="coerce")
    if merged["final_strategic_score"].isna().all():
        raise ValueError("No final strategic score values were available after loading Step-13D rankings; Step-13E cannot proceed.")

    merged["potency_component_raw"] = pd.to_numeric(coalesce_columns(merged, ["potency_component_raw", "best_causal_potency", "cross_family_mean_prediction_potency", "best_target_score"]), errors="coerce")
    merged["selectivity_component_raw"] = pd.to_numeric(coalesce_columns(merged, ["selectivity_component_raw", "best_selectivity_estimate", "cross_family_mean_prediction_selectivity"]), errors="coerce")
    merged["uncertainty_component_raw"] = pd.to_numeric(coalesce_columns(merged, ["uncertainty_component_raw", "max_uncertainty_penalty", "uncertainty_proxy", "uncertainty_proxy_potency"]), errors="coerce")
    merged["applicability_component_raw"] = pd.to_numeric(coalesce_columns(merged, ["applicability_component_raw", "mean_applicability_penalty", "applicability_penalty_raw"]), errors="coerce")
    merged["murcko_scaffold"] = normalize_text_series(coalesce_columns(merged, ["murcko_scaffold", "scaffold", "exact_scaffold"], default=""))
    merged["generic_scaffold"] = normalize_text_series(coalesce_columns(merged, ["generic_scaffold"], default=""))
    merged["scaffold_novelty_flag"] = pd.to_numeric(coalesce_columns(merged, ["scaffold_novelty_flag", "generic_scaffold_novelty_flag"], default=0), errors="coerce").fillna(0).astype(int)
    merged["consensus_metric"] = pd.to_numeric(coalesce_columns(merged, ["model_consensus_proxy", "cross_family_std_prediction", "families_with_support"], default=np.nan), errors="coerce")
    merged["families_with_support"] = pd.to_numeric(coalesce_columns(merged, ["families_with_support", "families_with_support_potency"], default=np.nan), errors="coerce")

    vendor_cols = [col for col in merged.columns if any(token in col.lower() for token in ["vendor", "supplier", "catalog", "availability", "stock", "price"])]
    merged["has_vendor_metadata"] = merged[vendor_cols].notna().any(axis=1).astype(int) if vendor_cols else 0
    merged["has_supplier_metadata"] = merged[[col for col in merged.columns if any(token in col.lower() for token in ["supplier", "vendor", "catalog"]) ]].notna().any(axis=1).astype(int) if vendor_cols else 0
    stock_cols = [col for col in merged.columns if any(token in col.lower() for token in ["stock", "availability", "available"])]
    if stock_cols:
        stock_frame = merged[stock_cols].astype(str).apply(lambda col: col.str.lower().str.strip())
        merged["in_stock_flag"] = stock_frame.apply(lambda row: int(any(val in {"1", "true", "yes", "y", "in_stock", "available"} for val in row if val not in {"", "nan", "none"})), axis=1)
    else:
        merged["in_stock_flag"] = 0
        warnings_list.append("No stock/availability metadata columns detected; in-stock tie-breaking preferences could not be applied.")

    if cfg.shortlist_mode == "compound_level":
        merged = merged.sort_values(["final_strategic_score", "screening_compound_id"], ascending=[False, True], kind="mergesort")
        merged = merged.drop_duplicates(subset=["screening_compound_id"], keep="first").reset_index(drop=True)
    else:
        merged = merged.sort_values(["final_strategic_score", "screening_compound_id", "target_chembl_id"], ascending=[False, True, True], kind="mergesort").reset_index(drop=True)

    merged["entity_id"] = merged["screening_compound_id"] if cfg.shortlist_mode == "compound_level" else merged["screening_compound_id"] + "__" + merged["target_chembl_id"]
    merged["purchasability_preference_score"] = (
        merged["has_vendor_metadata"] * int(cfg.purchasability_controls.prefer_available_vendor_entries)
        + merged["has_supplier_metadata"] * int(cfg.purchasability_controls.prefer_non_missing_supplier_metadata)
        + merged["in_stock_flag"] * int(cfg.purchasability_controls.prefer_in_stock_if_available)
    )
    merged["scaffold_frequency"] = merged.groupby("murcko_scaffold", dropna=False)["entity_id"].transform("count")
    merged["generic_scaffold_frequency"] = merged.groupby("generic_scaffold", dropna=False)["entity_id"].transform("count")
    merged["rare_scaffold_flag"] = ((merged["scaffold_frequency"] <= 2) | (merged["generic_scaffold_frequency"] <= 2)).astype(int)
    merged["primary_target_match_flag"] = merged["target_chembl_id"].isin(cfg.primary_target_chembl_ids).astype(int)
    return merged.reset_index(drop=True)


def add_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    desc_metrics = ["final_strategic_score", "potency_component_raw", "selectivity_component_raw", "consensus_metric"]
    asc_metrics = ["uncertainty_component_raw", "applicability_component_raw", "scaffold_frequency", "generic_scaffold_frequency"]
    for col in desc_metrics:
        series = pd.to_numeric(out[col], errors="coerce")
        valid = series.notna().sum()
        out[f"{col}_quantile_desc"] = series.rank(method="first", ascending=False, pct=True) if valid else np.nan
    for col in asc_metrics:
        series = pd.to_numeric(out[col], errors="coerce")
        valid = series.notna().sum()
        out[f"{col}_quantile_asc"] = series.rank(method="first", ascending=True, pct=True) if valid else np.nan
    return out


def build_common_sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sort_final_score"] = pd.to_numeric(out["final_strategic_score"], errors="coerce").fillna(-NUMERIC_FALLBACK_HIGH)
    out["sort_potency"] = pd.to_numeric(out["potency_component_raw"], errors="coerce").fillna(-NUMERIC_FALLBACK_HIGH)
    out["sort_selectivity"] = pd.to_numeric(out["selectivity_component_raw"], errors="coerce").fillna(-NUMERIC_FALLBACK_HIGH)
    out["sort_consensus"] = pd.to_numeric(out["consensus_metric"], errors="coerce").fillna(-NUMERIC_FALLBACK_HIGH)
    out["sort_uncertainty"] = pd.to_numeric(out["uncertainty_component_raw"], errors="coerce").fillna(NUMERIC_FALLBACK_HIGH)
    out["sort_applicability"] = pd.to_numeric(out["applicability_component_raw"], errors="coerce").fillna(NUMERIC_FALLBACK_HIGH)
    out["sort_scaffold_frequency"] = pd.to_numeric(out["scaffold_frequency"], errors="coerce").fillna(NUMERIC_FALLBACK_HIGH)
    out["sort_generic_scaffold_frequency"] = pd.to_numeric(out["generic_scaffold_frequency"], errors="coerce").fillna(NUMERIC_FALLBACK_HIGH)
    out["sort_primary_target"] = pd.to_numeric(out["primary_target_match_flag"], errors="coerce").fillna(0)
    out["sort_purchasability"] = pd.to_numeric(out["purchasability_preference_score"], errors="coerce").fillna(0)
    return out


def apply_bucket_rules(df: pd.DataFrame, bucket_name: str, bucket_rules: dict[str, Any], cfg: AppConfig) -> tuple[pd.DataFrame, pd.Series]:
    eligible = pd.Series(True, index=df.index)
    flags: dict[str, pd.Series] = {}

    def quantile_pass(metric_col: str, threshold: float, direction: str) -> pd.Series:
        if direction == "top":
            q_col = f"{metric_col}_quantile_desc"
            return df[q_col].notna() & (df[q_col] <= float(threshold))
        q_col = f"{metric_col}_quantile_asc"
        return df[q_col].notna() & (df[q_col] <= float(threshold))

    if bucket_name == "high_confidence_selective_hits":
        if "require_top_quantile_potency" in bucket_rules:
            flags["top_potency"] = quantile_pass("potency_component_raw", bucket_rules["require_top_quantile_potency"], "top")
            eligible &= flags["top_potency"]
        if "require_top_quantile_selectivity" in bucket_rules:
            flags["top_selectivity"] = quantile_pass("selectivity_component_raw", bucket_rules["require_top_quantile_selectivity"], "top")
            eligible &= flags["top_selectivity"]
        if "max_uncertainty_quantile" in bucket_rules:
            flags["low_uncertainty"] = quantile_pass("uncertainty_component_raw", bucket_rules["max_uncertainty_quantile"], "low")
            eligible &= flags["low_uncertainty"]
        if "max_applicability_penalty_quantile" in bucket_rules:
            flags["low_applicability_penalty"] = quantile_pass("applicability_component_raw", bucket_rules["max_applicability_penalty_quantile"], "low")
            eligible &= flags["low_applicability_penalty"]
    elif bucket_name == "novel_scaffold_selective_hits":
        if bucket_rules.get("require_scaffold_novelty_flag", False):
            flags["novel_scaffold"] = df["scaffold_novelty_flag"].eq(1)
            eligible &= flags["novel_scaffold"]
        if "require_top_quantile_selectivity" in bucket_rules:
            flags["top_selectivity"] = quantile_pass("selectivity_component_raw", bucket_rules["require_top_quantile_selectivity"], "top")
            eligible &= flags["top_selectivity"]
        if "max_uncertainty_quantile" in bucket_rules:
            flags["acceptable_uncertainty"] = quantile_pass("uncertainty_component_raw", bucket_rules["max_uncertainty_quantile"], "low")
            eligible &= flags["acceptable_uncertainty"]
        if "max_applicability_penalty_quantile" in bucket_rules:
            flags["acceptable_applicability_penalty"] = quantile_pass("applicability_component_raw", bucket_rules["max_applicability_penalty_quantile"], "low")
            eligible &= flags["acceptable_applicability_penalty"]
    elif bucket_name == "diverse_exploratory_hits":
        if bucket_rules.get("require_unique_or_rare_scaffold_preference", False):
            flags["rare_scaffold"] = df["rare_scaffold_flag"].eq(1)
            eligible &= flags["rare_scaffold"]
        if "require_minimum_potency_quantile" in bucket_rules:
            flags["minimum_potency"] = quantile_pass("potency_component_raw", bucket_rules["require_minimum_potency_quantile"], "top")
            eligible &= flags["minimum_potency"]
        if not bucket_rules.get("allow_medium_uncertainty", False):
            flags["exploration_uncertainty_control"] = quantile_pass("uncertainty_component_raw", 0.50, "low")
            eligible &= flags["exploration_uncertainty_control"]
    elif bucket_name == "consensus_supported_fallback_hits":
        if bucket_rules.get("require_high_model_consensus", False):
            metric_series = pd.to_numeric(df["consensus_metric"], errors="coerce")
            if metric_series.notna().any():
                threshold = metric_series.quantile(0.80)
                flags["high_model_consensus"] = metric_series >= threshold
            else:
                flags["high_model_consensus"] = pd.Series(False, index=df.index)
            eligible &= flags["high_model_consensus"]
        if "max_uncertainty_quantile" in bucket_rules:
            flags["fallback_uncertainty_ok"] = quantile_pass("uncertainty_component_raw", bucket_rules["max_uncertainty_quantile"], "low")
            eligible &= flags["fallback_uncertainty_ok"]
        if not bucket_rules.get("allow_lower_selectivity_than_primary_bucket", False):
            flags["primary_like_selectivity"] = quantile_pass("selectivity_component_raw", 0.20, "top")
            eligible &= flags["primary_like_selectivity"]

    flags["non_missing_final_score"] = df["final_strategic_score"].notna()
    eligible &= flags["non_missing_final_score"]
    reason_series = pd.Series([";".join(sorted(name for name, mask in flags.items() if bool(mask.iloc[i]))) for i in range(len(df))], index=df.index, dtype="string")
    return df.loc[eligible].copy(), reason_series


def sort_bucket_candidates(df: pd.DataFrame, bucket_name: str) -> pd.DataFrame:
    if bucket_name == "high_confidence_selective_hits":
        sort_cols = ["sort_primary_target", "sort_final_score", "sort_selectivity", "sort_potency", "sort_consensus", "sort_purchasability", "entity_id"]
        ascending = [False, False, False, False, False, False, True]
    elif bucket_name == "novel_scaffold_selective_hits":
        sort_cols = ["scaffold_novelty_flag", "sort_selectivity", "sort_final_score", "sort_potency", "sort_scaffold_frequency", "sort_purchasability", "entity_id"]
        ascending = [False, False, False, False, True, False, True]
    elif bucket_name == "diverse_exploratory_hits":
        sort_cols = ["rare_scaffold_flag", "sort_scaffold_frequency", "sort_generic_scaffold_frequency", "sort_final_score", "sort_potency", "sort_purchasability", "entity_id"]
        ascending = [False, True, True, False, False, False, True]
    else:
        sort_cols = ["sort_consensus", "sort_final_score", "sort_uncertainty", "sort_applicability", "sort_purchasability", "entity_id"]
        ascending = [False, False, True, True, False, True]
    return df.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)


def enforce_diversity(sorted_candidates: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cfg.diversity_controls.enforce_scaffold_diversity_within_bucket:
        log = pd.DataFrame(columns=["entity_id", "diversity_removed_flag", "diversity_removed_reason"])
        return sorted_candidates.copy(), log
    selected_rows: list[pd.Series] = []
    logs: list[dict[str, Any]] = []
    exact_counts: dict[str, int] = {}
    generic_counts: dict[str, int] = {}
    for row in sorted_candidates.itertuples(index=False):
        exact = str(getattr(row, "murcko_scaffold", "") or "")
        generic = str(getattr(row, "generic_scaffold", "") or "")
        exact_count = exact_counts.get(exact, 0)
        generic_count = generic_counts.get(generic, 0)
        if exact and exact_count >= cfg.diversity_controls.max_compounds_per_exact_scaffold:
            logs.append({"entity_id": row.entity_id, "diversity_removed_flag": 1, "diversity_removed_reason": "exact_scaffold_cap"})
            continue
        if generic and generic_count >= cfg.diversity_controls.max_compounds_per_generic_scaffold:
            logs.append({"entity_id": row.entity_id, "diversity_removed_flag": 1, "diversity_removed_reason": "generic_scaffold_cap"})
            continue
        selected_rows.append(pd.Series(row._asdict()))
        if exact:
            exact_counts[exact] = exact_count + 1
        if generic:
            generic_counts[generic] = generic_count + 1
        logs.append({"entity_id": row.entity_id, "diversity_removed_flag": 0, "diversity_removed_reason": "retained"})
    selected = pd.DataFrame(selected_rows) if selected_rows else sorted_candidates.iloc[0:0].copy()
    return selected, pd.DataFrame(logs)


def select_bucket(df: pd.DataFrame, bucket_name: str, cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    bucket_rules = cfg.bucket_rules[bucket_name]
    eligible_df, eligibility_reason_series = apply_bucket_rules(df, bucket_name, bucket_rules, cfg)
    eligible_df = eligible_df.copy()
    eligible_df["eligibility_flags"] = eligibility_reason_series.loc[eligible_df.index].fillna("")
    eligible_count = len(eligible_df)
    logging.info("Bucket %s: %d rows eligible after rule application.", bucket_name, eligible_count)
    ranked = sort_bucket_candidates(eligible_df, bucket_name)
    diversity_filtered, diversity_log = enforce_diversity(ranked, cfg)
    requested = cfg.bucket_sizes[bucket_name]
    selected = diversity_filtered.head(requested).copy()
    selected["assigned_bucket"] = bucket_name
    selected["bucket_priority_rank"] = cfg.prioritize_higher_priority_bucket_order.index(bucket_name) + 1
    selected["diversity_removed_flag"] = 0
    selected["diversity_control_note"] = np.where(
        cfg.diversity_controls.enforce_scaffold_diversity_within_bucket,
        "scaffold_based_diversity_enforced",
        "scaffold_diversity_not_enforced",
    )
    removed_by_diversity = int((diversity_log["diversity_removed_flag"] == 1).sum()) if not diversity_log.empty else 0
    summary = {
        "bucket_name": bucket_name,
        "requested_bucket_size": requested,
        "number_remaining_eligible_after_rules": eligible_count,
        "number_removed_by_diversity_controls": removed_by_diversity,
        "pre_dedup_selected_count": int(len(selected)),
    }
    rationale = selected[["entity_id", "assigned_bucket", "eligibility_flags"]].copy()
    rationale["bucket_specific_rationale"] = selected.apply(lambda row: render_bucket_rationale(row, bucket_name), axis=1)
    return selected, rationale, summary


def render_bucket_rationale(row: pd.Series, bucket_name: str) -> str:
    if bucket_name == "high_confidence_selective_hits":
        return "Selected for strong strategic score, potency/selectivity support, and low proxy penalties."
    if bucket_name == "novel_scaffold_selective_hits":
        return "Selected to preserve scaffold novelty while retaining selective ranking support."
    if bucket_name == "diverse_exploratory_hits":
        return "Selected to broaden shortlist chemistry through rare or underrepresented scaffolds."
    return "Selected as a consensus-supported fallback candidate with acceptable proxy risk."


def deduplicate_buckets(bucket_outputs: dict[str, pd.DataFrame], cfg: AppConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, int]]:
    if not cfg.deduplicate_across_buckets:
        empty = pd.DataFrame(columns=["entity_id", "qualified_bucket_list", "assigned_bucket", "was_deduplicated_to_higher_priority_bucket"])
        return bucket_outputs, empty, {name: 0 for name in bucket_outputs}
    qualified_map: dict[str, list[str]] = {}
    for bucket in cfg.prioritize_higher_priority_bucket_order:
        frame = bucket_outputs.get(bucket, pd.DataFrame())
        for entity_id in frame.get("entity_id", pd.Series(dtype=str)).astype(str):
            qualified_map.setdefault(entity_id, []).append(bucket)
    assigned: set[str] = set()
    dedup_counts: dict[str, int] = {bucket: 0 for bucket in cfg.prioritize_higher_priority_bucket_order}
    dedup_details: list[dict[str, Any]] = []
    final_outputs: dict[str, pd.DataFrame] = {}
    for bucket in cfg.prioritize_higher_priority_bucket_order:
        frame = bucket_outputs.get(bucket, pd.DataFrame()).copy()
        keep_mask = ~frame["entity_id"].isin(assigned) if not frame.empty else pd.Series(dtype=bool)
        removed = int((~keep_mask).sum()) if not frame.empty else 0
        dedup_counts[bucket] = removed
        kept = frame.loc[keep_mask].copy() if not frame.empty else frame
        assigned.update(kept.get("entity_id", pd.Series(dtype=str)).astype(str).tolist())
        final_outputs[bucket] = kept.reset_index(drop=True)
    for entity_id, buckets in sorted(qualified_map.items()):
        assigned_bucket = next((bucket for bucket in cfg.prioritize_higher_priority_bucket_order if bucket in buckets), "")
        dedup_details.append(
            {
                "entity_id": entity_id,
                "qualified_bucket_list": ";".join(buckets),
                "assigned_bucket": assigned_bucket,
                "was_deduplicated_to_higher_priority_bucket": int(len(buckets) > 1),
            }
        )
    return final_outputs, pd.DataFrame(dedup_details), dedup_counts


def combine_buckets(bucket_outputs: dict[str, pd.DataFrame], cfg: AppConfig) -> pd.DataFrame:
    frames = [bucket_outputs[bucket] for bucket in cfg.prioritize_higher_priority_bucket_order if bucket_outputs.get(bucket) is not None and not bucket_outputs[bucket].empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.sort_values(["bucket_priority_rank", "final_strategic_score", "screening_compound_id", "target_chembl_id"], ascending=[True, False, True, True], kind="mergesort").reset_index(drop=True)
    combined["final_shortlist_rank"] = np.arange(1, len(combined) + 1)
    return combined


def build_rationale_table(final_shortlist: pd.DataFrame, bucket_rationales: dict[str, pd.DataFrame], dedup_details: pd.DataFrame) -> pd.DataFrame:
    if final_shortlist.empty:
        return pd.DataFrame()
    rationale = final_shortlist[[
        "entity_id",
        "screening_compound_id",
        "assigned_bucket",
        "final_strategic_score",
        "potency_component_raw",
        "selectivity_component_raw",
        "uncertainty_component_raw",
        "applicability_component_raw",
        "murcko_scaffold",
        "generic_scaffold",
        "scaffold_novelty_flag",
        "consensus_metric",
        "purchasability_preference_score",
        "diversity_control_note",
    ]].copy()
    rationale["metadata_preference_affected_tie_breaking"] = (rationale["purchasability_preference_score"] > 0).astype(int)
    rationale["diversity_control_affected_selection"] = rationale["diversity_control_note"].ne("").astype(int)
    bucket_rationale_df = pd.concat([frame for frame in bucket_rationales.values() if not frame.empty], axis=0, ignore_index=True) if bucket_rationales else pd.DataFrame()
    if not bucket_rationale_df.empty:
        rationale = rationale.merge(bucket_rationale_df, on=["entity_id", "assigned_bucket"], how="left")
    if not dedup_details.empty:
        rationale = rationale.merge(dedup_details, on="entity_id", how="left")
    rationale["qualified_for_multiple_buckets"] = rationale["was_deduplicated_to_higher_priority_bucket"].fillna(0).astype(int)
    rationale["selection_rationale_text"] = rationale.apply(
        lambda row: (
            f"Assigned to {row['assigned_bucket']} with final strategic score {row['final_strategic_score']:.6f}; "
            f"potency={row['potency_component_raw'] if pd.notna(row['potency_component_raw']) else 'NA'}, "
            f"selectivity={row['selectivity_component_raw'] if pd.notna(row['selectivity_component_raw']) else 'NA'}, "
            f"uncertainty_proxy={row['uncertainty_component_raw'] if pd.notna(row['uncertainty_component_raw']) else 'NA'}, "
            f"applicability_penalty={row['applicability_component_raw'] if pd.notna(row['applicability_component_raw']) else 'NA'}."
        ),
        axis=1,
    )
    return rationale.sort_values(["assigned_bucket", "final_strategic_score", "screening_compound_id"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)


def build_diversity_summary(final_shortlist: pd.DataFrame) -> pd.DataFrame:
    if final_shortlist.empty:
        return pd.DataFrame(columns=["bucket_name", "number_of_compounds"])
    rows: list[dict[str, Any]] = []
    for bucket_name, bucket_df in final_shortlist.groupby("assigned_bucket", sort=True):
        scaffold_counts = bucket_df["murcko_scaffold"].fillna("").astype(str).value_counts()
        generic_counts = bucket_df["generic_scaffold"].fillna("").astype(str).value_counts()
        source_counts = bucket_df["source_library_name"].fillna("").astype(str).value_counts()
        rows.append(
            {
                "bucket_name": bucket_name,
                "number_of_compounds": int(bucket_df["screening_compound_id"].nunique()),
                "number_of_rows": int(len(bucket_df)),
                "number_of_unique_exact_scaffolds": int(bucket_df["murcko_scaffold"].replace("", np.nan).nunique(dropna=True)),
                "number_of_unique_generic_scaffolds": int(bucket_df["generic_scaffold"].replace("", np.nan).nunique(dropna=True)),
                "top_exact_scaffold_counts": "; ".join(f"{idx}:{val}" for idx, val in scaffold_counts.head(5).items() if idx),
                "top_generic_scaffold_counts": "; ".join(f"{idx}:{val}" for idx, val in generic_counts.head(5).items() if idx),
                "exact_scaffold_duplication_rate": float(1.0 - (bucket_df["murcko_scaffold"].replace("", np.nan).nunique(dropna=True) / max(len(bucket_df), 1))),
                "generic_scaffold_duplication_rate": float(1.0 - (bucket_df["generic_scaffold"].replace("", np.nan).nunique(dropna=True) / max(len(bucket_df), 1))),
                "source_library_distribution": "; ".join(f"{idx}:{val}" for idx, val in source_counts.head(5).items() if idx),
            }
        )
    overall = final_shortlist.copy()
    rows.append(
        {
            "bucket_name": "ALL_BUCKETS",
            "number_of_compounds": int(overall["screening_compound_id"].nunique()),
            "number_of_rows": int(len(overall)),
            "number_of_unique_exact_scaffolds": int(overall["murcko_scaffold"].replace("", np.nan).nunique(dropna=True)),
            "number_of_unique_generic_scaffolds": int(overall["generic_scaffold"].replace("", np.nan).nunique(dropna=True)),
            "top_exact_scaffold_counts": "; ".join(f"{idx}:{val}" for idx, val in overall["murcko_scaffold"].fillna("").astype(str).value_counts().head(5).items() if idx),
            "top_generic_scaffold_counts": "; ".join(f"{idx}:{val}" for idx, val in overall["generic_scaffold"].fillna("").astype(str).value_counts().head(5).items() if idx),
            "exact_scaffold_duplication_rate": float(1.0 - (overall["murcko_scaffold"].replace("", np.nan).nunique(dropna=True) / max(len(overall), 1))),
            "generic_scaffold_duplication_rate": float(1.0 - (overall["generic_scaffold"].replace("", np.nan).nunique(dropna=True) / max(len(overall), 1))),
            "source_library_distribution": "; ".join(f"{idx}:{val}" for idx, val in overall["source_library_name"].fillna("").astype(str).value_counts().head(10).items() if idx),
        }
    )
    return pd.DataFrame(rows).sort_values("bucket_name", kind="mergesort").reset_index(drop=True)


def build_bucket_summary(bucket_outputs: dict[str, pd.DataFrame], bucket_selection_summaries: dict[str, dict[str, Any]], dedup_counts: dict[str, int], cfg: AppConfig, warnings_list: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket in cfg.prioritize_higher_priority_bucket_order:
        summary = bucket_selection_summaries[bucket]
        final_size = int(len(bucket_outputs.get(bucket, pd.DataFrame())))
        notes = []
        if summary["requested_bucket_size"] > final_size:
            notes.append(f"shortfall={summary['requested_bucket_size'] - final_size}")
            warnings_list.append(f"Bucket `{bucket}` could not be filled to requested size; requested {summary['requested_bucket_size']} and achieved {final_size}.")
        if cfg.diversity_controls.use_fingerprint_diversity_selection is False:
            notes.append("fingerprint_diversity_not_used")
        rows.append(
            {
                "bucket_name": bucket,
                "requested_bucket_size": int(summary["requested_bucket_size"]),
                "final_bucket_size": final_size,
                "number_removed_by_deduplication": int(dedup_counts.get(bucket, 0)),
                "number_removed_by_diversity_controls": int(summary["number_removed_by_diversity_controls"]),
                "number_remaining_eligible_after_rules": int(summary["number_remaining_eligible_after_rules"]),
                "notes": ";".join(notes),
            }
        )
    return pd.DataFrame(rows).sort_values("bucket_name", kind="mergesort").reset_index(drop=True)


def save_target_specific_shortlists(final_shortlist: pd.DataFrame, cfg: AppConfig) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    if not cfg.save_target_specific_shortlists or final_shortlist.empty or "target_chembl_id" not in final_shortlist.columns:
        return manifest_rows
    shortlist_root = cfg.output_shortlist_root
    for target in cfg.primary_target_chembl_ids:
        target_df = final_shortlist[final_shortlist["target_chembl_id"] == target].copy()
        path = shortlist_root / f"target_shortlist_{target}.csv"
        write_dataframe(target_df, path)
        manifest_rows.append(
            {
                "asset_id": f"target_shortlist_{target}",
                "asset_type": "target_specific_shortlist",
                "file_path": str(path),
                "row_count": int(len(target_df)),
                "bucket_name": "ALL_BUCKETS",
                "shortlist_scope": cfg.shortlist_mode,
                "notes": f"Primary target shortlist for {target}.",
            }
        )
    return manifest_rows


def build_manifest(cfg: AppConfig, final_shortlist: pd.DataFrame, rationale: pd.DataFrame, bucket_summary: pd.DataFrame, diversity_summary: pd.DataFrame, bucket_outputs: dict[str, pd.DataFrame], extra_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "asset_id": "final_screening_shortlist",
            "asset_type": "final_shortlist",
            "file_path": str(cfg.output_final_shortlist_path),
            "row_count": int(len(final_shortlist)),
            "bucket_name": "ALL_BUCKETS",
            "shortlist_scope": cfg.shortlist_mode,
            "notes": "Final combined actionable screening shortlist.",
        },
        {
            "asset_id": "final_shortlist_rationale",
            "asset_type": "shortlist_rationale",
            "file_path": str(cfg.output_shortlist_rationale_path),
            "row_count": int(len(rationale)),
            "bucket_name": "ALL_BUCKETS",
            "shortlist_scope": cfg.shortlist_mode,
            "notes": "Structured rationale table describing shortlist inclusion decisions.",
        },
        {
            "asset_id": "shortlist_bucket_summary",
            "asset_type": "qc_summary",
            "file_path": str(cfg.output_bucket_summary_path),
            "row_count": int(len(bucket_summary)),
            "bucket_name": "ALL_BUCKETS",
            "shortlist_scope": cfg.shortlist_mode,
            "notes": "Bucket-level selection and shortfall summary.",
        },
        {
            "asset_id": "shortlist_diversity_summary",
            "asset_type": "diversity_summary",
            "file_path": str(cfg.output_diversity_summary_path),
            "row_count": int(len(diversity_summary)),
            "bucket_name": "ALL_BUCKETS",
            "shortlist_scope": cfg.shortlist_mode,
            "notes": "Scaffold and source-library diversity summary for shortlist outputs.",
        },
    ]
    if cfg.save_bucket_specific_files:
        for bucket_name, frame in bucket_outputs.items():
            bucket_path = cfg.output_shortlist_root / f"{bucket_name}.csv"
            rows.append(
                {
                    "asset_id": bucket_name,
                    "asset_type": "bucket_shortlist",
                    "file_path": str(bucket_path),
                    "row_count": int(len(frame)),
                    "bucket_name": bucket_name,
                    "shortlist_scope": cfg.shortlist_mode,
                    "notes": f"Bucket-specific shortlist for {bucket_name}.",
                }
            )
    rows.extend(extra_rows)
    return pd.DataFrame(rows).sort_values(["asset_id"], kind="mergesort").reset_index(drop=True)


def summarize_report(cfg: AppConfig, final_shortlist: pd.DataFrame, bucket_summary: pd.DataFrame, diversity_summary: pd.DataFrame, warnings_list: list[str], config_snapshot_path: Path | None, bucket_selection_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metadata_summary = {
        "prefer_available_vendor_entries": cfg.purchasability_controls.prefer_available_vendor_entries,
        "prefer_non_missing_supplier_metadata": cfg.purchasability_controls.prefer_non_missing_supplier_metadata,
        "prefer_in_stock_if_available": cfg.purchasability_controls.prefer_in_stock_if_available,
        "rows_with_vendor_metadata": int(final_shortlist.get("has_vendor_metadata", pd.Series(dtype=int)).fillna(0).sum()) if not final_shortlist.empty else 0,
        "rows_with_in_stock_flag": int(final_shortlist.get("in_stock_flag", pd.Series(dtype=int)).fillna(0).sum()) if not final_shortlist.empty else 0,
    }
    return {
        "script_name": SCRIPT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_ranking_files_used": {
            "compound_target_ranking": str(cfg.input_compound_target_ranking_path),
            "compound_summary_ranking": str(cfg.input_compound_summary_ranking_path),
            "uncertainty_summary": str(cfg.input_uncertainty_summary_path),
            "applicability_summary": str(cfg.input_applicability_summary_path),
            "consensus_summary": str(cfg.input_consensus_summary_path),
            "screening_library": str(cfg.input_screening_library_path),
            "environment_features": str(cfg.input_environment_feature_path),
            "provenance": str(cfg.input_provenance_path),
        },
        "shortlist_mode_used": cfg.shortlist_mode,
        "total_requested_shortlist_size": cfg.total_shortlist_size,
        "final_shortlist_size": int(len(final_shortlist)),
        "bucket_sizes_requested": cfg.bucket_sizes,
        "bucket_sizes_achieved": {row["bucket_name"]: int(row["final_bucket_size"]) for row in bucket_summary.to_dict(orient="records")},
        "bucket_rule_eligibility_counts": {bucket: int(summary["number_remaining_eligible_after_rules"]) for bucket, summary in bucket_selection_summaries.items()},
        "diversity_summary": diversity_summary.to_dict(orient="records"),
        "cross_bucket_deduplication_summary": bucket_summary[["bucket_name", "number_removed_by_deduplication"]].to_dict(orient="records") if not bucket_summary.empty else [],
        "metadata_preference_summary": metadata_summary,
        "warnings": warnings_list,
        "config_snapshot_reference": str(config_snapshot_path) if config_snapshot_path else "",
    }


def write_outputs(cfg: AppConfig, final_shortlist: pd.DataFrame, rationale: pd.DataFrame, bucket_summary: pd.DataFrame, diversity_summary: pd.DataFrame, manifest: pd.DataFrame, report: dict[str, Any], bucket_outputs: dict[str, pd.DataFrame]) -> None:
    logging.info("Writing shortlist outputs, manifest, and report.")
    cfg.output_shortlist_root.mkdir(parents=True, exist_ok=True)
    write_dataframe(final_shortlist, cfg.output_final_shortlist_path)
    write_dataframe(rationale, cfg.output_shortlist_rationale_path)
    write_dataframe(bucket_summary, cfg.output_bucket_summary_path)
    write_dataframe(diversity_summary, cfg.output_diversity_summary_path)
    if cfg.save_bucket_specific_files:
        for bucket_name, frame in bucket_outputs.items():
            write_dataframe(frame, cfg.output_shortlist_root / f"{bucket_name}.csv")
    write_dataframe(manifest, cfg.output_manifest_path)
    write_json(report, cfg.output_report_path)


def run_shortlisting(inputs: dict[str, pd.DataFrame], cfg: AppConfig, warnings_list: list[str]) -> dict[str, Any]:
    entity_df = build_common_sort_columns(add_quantiles(build_entity_table(inputs, cfg, warnings_list)))
    if not cfg.diversity_controls.use_fingerprint_diversity_selection:
        logging.info("Fingerprint diversity selection is disabled; using scaffold-based diversity controls only.")
    bucket_candidates: dict[str, pd.DataFrame] = {}
    bucket_rationales: dict[str, pd.DataFrame] = {}
    bucket_selection_summaries: dict[str, dict[str, Any]] = {}
    for bucket in cfg.prioritize_higher_priority_bucket_order:
        logging.info("Applying bucket rule set for %s", bucket)
        selected, rationale, summary = select_bucket(entity_df, bucket, cfg)
        bucket_candidates[bucket] = selected
        bucket_rationales[bucket] = rationale
        bucket_selection_summaries[bucket] = summary
    final_bucket_outputs, dedup_details, dedup_counts = deduplicate_buckets(bucket_candidates, cfg)
    final_shortlist = combine_buckets(final_bucket_outputs, cfg)
    rationale = build_rationale_table(final_shortlist, bucket_rationales, dedup_details)
    diversity_summary = build_diversity_summary(final_shortlist)
    bucket_summary = build_bucket_summary(final_bucket_outputs, bucket_selection_summaries, dedup_counts, cfg, warnings_list)
    extra_manifest_rows = save_target_specific_shortlists(final_shortlist, cfg)
    manifest = build_manifest(cfg, final_shortlist, rationale, bucket_summary, diversity_summary, final_bucket_outputs, extra_manifest_rows)
    return {
        "entity_df": entity_df,
        "bucket_outputs": final_bucket_outputs,
        "bucket_selection_summaries": bucket_selection_summaries,
        "dedup_details": dedup_details,
        "final_shortlist": final_shortlist,
        "rationale": rationale,
        "diversity_summary": diversity_summary,
        "bucket_summary": bucket_summary,
        "manifest": manifest,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    raw_config = load_yaml(config_path)
    project_root = config_path.parent.resolve()
    cfg = AppConfig.from_dict(raw_config, project_root)
    log_path = setup_logging(cfg)
    config_snapshot_path = save_config_snapshot(raw_config, cfg)
    logging.info("Starting %s", SCRIPT_NAME)
    inputs, warnings_list = load_inputs(cfg)
    outputs = run_shortlisting(inputs, cfg, warnings_list)
    report = summarize_report(
        cfg,
        outputs["final_shortlist"],
        outputs["bucket_summary"],
        outputs["diversity_summary"],
        warnings_list,
        config_snapshot_path,
        outputs["bucket_selection_summaries"],
    )
    write_outputs(
        cfg,
        outputs["final_shortlist"],
        outputs["rationale"],
        outputs["bucket_summary"],
        outputs["diversity_summary"],
        outputs["manifest"],
        report,
        outputs["bucket_outputs"],
    )
    logging.info("Completed %s successfully.", SCRIPT_NAME)
    logging.info("Log written to %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
