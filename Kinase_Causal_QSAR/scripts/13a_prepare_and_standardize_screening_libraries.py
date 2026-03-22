#!/usr/bin/env python3
"""Prepare and standardize external screening libraries for downstream inference.

This script is a strict continuation of Steps 01-12 of the kinase causality
QSAR pipeline. It loads one or more raw screening libraries from config-driven
locations, resolves core columns, performs deterministic RDKit-based chemical
standardization, applies configurable QC filters, annotates duplicates and
provenance, and writes publication-grade screening preparation assets. It does
not score compounds, rank compounds, or retrain any model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

SCRIPT_NAME = "13a_prepare_and_standardize_screening_libraries"
REQUIRED_SCRIPT_13A_KEYS = {
    "input_libraries",
    "output_library_root",
    "output_cleaned_root",
    "output_merged_library_path",
    "output_provenance_path",
    "output_duplicate_summary_path",
    "output_qc_summary_path",
    "output_manifest_path",
    "output_report_path",
    "standardization",
    "filtering",
    "provenance",
    "save_library_specific_outputs",
    "save_failed_rows",
    "save_config_snapshot",
}
REQUIRED_STANDARDIZATION_KEYS = {
    "remove_salts",
    "keep_largest_fragment",
    "normalize_molecules",
    "canonicalize_smiles",
    "neutralize_if_possible",
    "preserve_original_smiles",
}
REQUIRED_FILTERING_KEYS = {
    "remove_invalid_smiles",
    "remove_mixtures",
    "remove_inorganic",
    "remove_empty_smiles",
    "remove_duplicates_within_library",
    "remove_duplicates_across_libraries",
    "min_heavy_atoms",
    "max_molecular_weight",
    "min_molecular_weight",
}
REQUIRED_PROVENANCE_KEYS = {
    "track_input_library_name",
    "track_original_row_index",
    "track_original_compound_id",
    "track_original_smiles",
    "track_source_file_path",
}
INTERNAL_COLUMN_ORDER = [
    "screening_compound_id",
    "source_library_name",
    "source_file_path",
    "original_row_index",
    "original_compound_id",
    "original_smiles",
    "standardized_smiles",
    "library_specific_duplicate_flag",
    "cross_library_duplicate_flag",
    "within_library_duplicate_count",
    "cross_library_duplicate_count",
    "number_of_libraries_containing_this_structure",
    "molecular_weight",
    "heavy_atom_count",
    "provenance_record_count",
    "collapsed_original_row_indices",
    "collapsed_original_compound_ids",
    "collapsed_original_smiles",
]


@dataclass
class LibraryConfig:
    name: str
    path: Path
    file_type: str
    smiles_column_candidates: list[str]
    compound_id_column_candidates: list[str]
    extra_metadata_columns: list[str] = field(default_factory=list)


@dataclass
class StandardizationConfig:
    remove_salts: bool
    keep_largest_fragment: bool
    normalize_molecules: bool
    canonicalize_smiles: bool
    neutralize_if_possible: bool
    preserve_original_smiles: bool


@dataclass
class FilteringConfig:
    remove_invalid_smiles: bool
    remove_mixtures: bool
    remove_inorganic: bool
    remove_empty_smiles: bool
    remove_duplicates_within_library: bool
    remove_duplicates_across_libraries: bool
    min_heavy_atoms: int
    max_molecular_weight: float
    min_molecular_weight: float


@dataclass
class ProvenanceConfig:
    track_input_library_name: bool
    track_original_row_index: bool
    track_original_compound_id: bool
    track_original_smiles: bool
    track_source_file_path: bool


@dataclass
class AppConfig:
    input_libraries: list[LibraryConfig]
    output_library_root: Path
    output_cleaned_root: Path
    output_merged_library_path: Path
    output_provenance_path: Path
    output_duplicate_summary_path: Path
    output_qc_summary_path: Path
    output_manifest_path: Path
    output_report_path: Path
    standardization: StandardizationConfig
    filtering: FilteringConfig
    provenance: ProvenanceConfig
    save_library_specific_outputs: bool
    save_failed_rows: bool
    save_config_snapshot: bool
    project_root: Path
    logs_dir: Path
    configs_used_dir: Path

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        section = raw.get("script_13a")
        if not isinstance(section, dict):
            raise ValueError("Missing required `script_13a` section in config.yaml.")
        missing = sorted(REQUIRED_SCRIPT_13A_KEYS.difference(section))
        if missing:
            raise ValueError("Missing required script_13a config values: " + ", ".join(missing))

        def resolve(path_like: Any) -> Path:
            path = Path(str(path_like))
            return path if path.is_absolute() else project_root / path

        def parse_bool(value: Any, key: str) -> bool:
            if isinstance(value, bool):
                return value
            raise ValueError(f"script_13a.{key} must be boolean; got {value!r}.")

        libs_raw = section["input_libraries"]
        if not isinstance(libs_raw, list) or not libs_raw:
            raise ValueError("script_13a.input_libraries must be a non-empty list.")

        libraries: list[LibraryConfig] = []
        seen_names: set[str] = set()
        for idx, item in enumerate(libs_raw, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"script_13a.input_libraries[{idx}] must be a mapping.")
            for key in ["name", "path", "file_type", "smiles_column_candidates", "compound_id_column_candidates", "extra_metadata_columns"]:
                if key not in item:
                    raise ValueError(f"Missing script_13a.input_libraries[{idx}].{key}.")
            name = str(item["name"]).strip()
            if not name:
                raise ValueError(f"script_13a.input_libraries[{idx}].name must be non-empty.")
            if name in seen_names:
                raise ValueError(f"Duplicate screening library name in config: {name}")
            seen_names.add(name)
            smiles_candidates = [str(value).strip() for value in item["smiles_column_candidates"] if str(value).strip()]
            compound_id_candidates = [str(value).strip() for value in item["compound_id_column_candidates"] if str(value).strip()]
            extra_cols = [str(value).strip() for value in item["extra_metadata_columns"] if str(value).strip()]
            if not smiles_candidates:
                raise ValueError(f"script_13a.input_libraries[{idx}].smiles_column_candidates must be non-empty.")
            libraries.append(
                LibraryConfig(
                    name=name,
                    path=resolve(item["path"]),
                    file_type=str(item["file_type"]).strip().lower(),
                    smiles_column_candidates=smiles_candidates,
                    compound_id_column_candidates=compound_id_candidates,
                    extra_metadata_columns=extra_cols,
                )
            )

        std_raw = section["standardization"]
        filt_raw = section["filtering"]
        prov_raw = section["provenance"]
        if not isinstance(std_raw, dict):
            raise ValueError("script_13a.standardization must be a mapping.")
        if not isinstance(filt_raw, dict):
            raise ValueError("script_13a.filtering must be a mapping.")
        if not isinstance(prov_raw, dict):
            raise ValueError("script_13a.provenance must be a mapping.")

        missing_std = sorted(REQUIRED_STANDARDIZATION_KEYS.difference(std_raw))
        missing_filt = sorted(REQUIRED_FILTERING_KEYS.difference(filt_raw))
        missing_prov = sorted(REQUIRED_PROVENANCE_KEYS.difference(prov_raw))
        if missing_std:
            raise ValueError("Missing required script_13a.standardization values: " + ", ".join(missing_std))
        if missing_filt:
            raise ValueError("Missing required script_13a.filtering values: " + ", ".join(missing_filt))
        if missing_prov:
            raise ValueError("Missing required script_13a.provenance values: " + ", ".join(missing_prov))

        return AppConfig(
            input_libraries=libraries,
            output_library_root=resolve(section["output_library_root"]),
            output_cleaned_root=resolve(section["output_cleaned_root"]),
            output_merged_library_path=resolve(section["output_merged_library_path"]),
            output_provenance_path=resolve(section["output_provenance_path"]),
            output_duplicate_summary_path=resolve(section["output_duplicate_summary_path"]),
            output_qc_summary_path=resolve(section["output_qc_summary_path"]),
            output_manifest_path=resolve(section["output_manifest_path"]),
            output_report_path=resolve(section["output_report_path"]),
            standardization=StandardizationConfig(**{key: parse_bool(std_raw[key], f"standardization.{key}") for key in REQUIRED_STANDARDIZATION_KEYS}),
            filtering=FilteringConfig(
                remove_invalid_smiles=parse_bool(filt_raw["remove_invalid_smiles"], "filtering.remove_invalid_smiles"),
                remove_mixtures=parse_bool(filt_raw["remove_mixtures"], "filtering.remove_mixtures"),
                remove_inorganic=parse_bool(filt_raw["remove_inorganic"], "filtering.remove_inorganic"),
                remove_empty_smiles=parse_bool(filt_raw["remove_empty_smiles"], "filtering.remove_empty_smiles"),
                remove_duplicates_within_library=parse_bool(filt_raw["remove_duplicates_within_library"], "filtering.remove_duplicates_within_library"),
                remove_duplicates_across_libraries=parse_bool(filt_raw["remove_duplicates_across_libraries"], "filtering.remove_duplicates_across_libraries"),
                min_heavy_atoms=int(filt_raw["min_heavy_atoms"]),
                max_molecular_weight=float(filt_raw["max_molecular_weight"]),
                min_molecular_weight=float(filt_raw["min_molecular_weight"]),
            ),
            provenance=ProvenanceConfig(**{key: parse_bool(prov_raw[key], f"provenance.{key}") for key in REQUIRED_PROVENANCE_KEYS}),
            save_library_specific_outputs=parse_bool(section["save_library_specific_outputs"], "save_library_specific_outputs"),
            save_failed_rows=parse_bool(section["save_failed_rows"], "save_failed_rows"),
            save_config_snapshot=parse_bool(section["save_config_snapshot"], "save_config_snapshot"),
            project_root=project_root,
            logs_dir=project_root / "logs",
            configs_used_dir=project_root / "configs_used",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML relative to the project root (default: config.yaml).")
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must define a top-level mapping: {path}")
    return payload


def configure_logging(project_root: Path) -> Path:
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{SCRIPT_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    return log_path


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_config_snapshot(raw_config: dict[str, Any], cfg: AppConfig) -> Path | None:
    if not cfg.save_config_snapshot:
        return None
    ensure_directory(cfg.configs_used_dir)
    output_path = cfg.configs_used_dir / f"{SCRIPT_NAME}_config.yaml"
    output_path.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    logging.info("Saved config snapshot to %s", output_path)
    return output_path


def sanitize_column_name(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "unnamed_column"
    return text


def make_unique_column_names(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    output: list[str] = []
    for column in columns:
        base = sanitize_column_name(column)
        count = counts.get(base, 0)
        counts[base] = count + 1
        output.append(base if count == 0 else f"{base}__dup_{count}")
    return output


def detect_delimiter(path: Path) -> str:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(4096)
    if not sample.strip():
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except csv.Error:
        return "\t" if "\t" in sample else ","


def read_smi_file(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            smiles = tokens[0] if tokens else ""
            compound_id = tokens[1] if len(tokens) > 1 else ""
            molecule_name = " ".join(tokens[2:]) if len(tokens) > 2 else ""
            rows.append(
                {
                    "smiles": smiles,
                    "compound_id": compound_id,
                    "molecule_name": molecule_name,
                    "_smi_line_number": line_number,
                }
            )
    return pd.DataFrame(rows)


def read_library_table(library_cfg: LibraryConfig) -> pd.DataFrame:
    path = library_cfg.path
    if not path.exists():
        raise FileNotFoundError(f"Configured screening library file does not exist: {path}")
    file_type = library_cfg.file_type.lower()
    compression = "gzip" if path.suffix.lower() == ".gz" else "infer"
    logging.info("Loading library %s from %s as %s", library_cfg.name, path, file_type)

    if file_type == "parquet":
        frame = pd.read_parquet(path)
    elif file_type in {"smi", "smiles"}:
        frame = read_smi_file(path)
    elif file_type in {"csv", "tsv", "txt", "gz"}:
        delimiter = detect_delimiter(path)
        frame = pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=False, compression=compression)
    else:
        delimiter = detect_delimiter(path)
        frame = pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=False, compression=compression)

    frame = frame.copy()
    frame.columns = make_unique_column_names([str(column) for column in frame.columns])
    for column in frame.columns:
        frame[column] = frame[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    logging.info("Loaded %d rows and %d columns for library %s", len(frame), len(frame.columns), library_cfg.name)
    return frame


def resolve_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized_lookup = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        match = normalized_lookup.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def initialize_library_frame(frame: pd.DataFrame, library_cfg: LibraryConfig) -> tuple[pd.DataFrame, list[str], list[str]]:
    smiles_column = resolve_column(frame.columns.tolist(), library_cfg.smiles_column_candidates)
    if smiles_column is None:
        raise ValueError(
            f"Unable to identify a SMILES column for library {library_cfg.name}. "
            f"Configured candidates were {library_cfg.smiles_column_candidates}; available columns are {frame.columns.tolist()}."
        )
    compound_id_column = resolve_column(frame.columns.tolist(), library_cfg.compound_id_column_candidates)
    missing_metadata: list[str] = []
    retained_metadata_columns: list[str] = []
    for candidate in library_cfg.extra_metadata_columns:
        resolved = resolve_column(frame.columns.tolist(), [candidate])
        if resolved is None:
            missing_metadata.append(candidate)
        elif resolved not in retained_metadata_columns:
            retained_metadata_columns.append(resolved)

    working = frame.copy().reset_index(drop=True)
    working["source_library_name"] = library_cfg.name
    working["source_file_path"] = str(library_cfg.path)
    working["original_row_index"] = (working.index + 1).astype(int)
    working["original_smiles"] = working[smiles_column].astype(str).str.strip()

    if compound_id_column is None:
        logging.warning("No configured compound ID column found for library %s; generating deterministic IDs.", library_cfg.name)
        generated = working["original_row_index"].map(lambda row_idx: f"{library_cfg.name}__row_{row_idx}")
        working["original_compound_id"] = generated
    else:
        original_ids = working[compound_id_column].astype(str).str.strip()
        generated = working["original_row_index"].map(lambda row_idx: f"{library_cfg.name}__row_{row_idx}")
        working["original_compound_id"] = original_ids.where(original_ids != "", generated)

    working["screening_compound_id"] = working.apply(
        lambda row: f"{library_cfg.name}__{row['original_compound_id']}__row_{int(row['original_row_index'])}", axis=1
    )

    base_columns = [
        "screening_compound_id",
        "source_library_name",
        "source_file_path",
        "original_row_index",
        "original_compound_id",
        "original_smiles",
    ]
    selected_columns = base_columns + retained_metadata_columns
    working = working[selected_columns].copy()
    logging.info(
        "Resolved library %s core columns: smiles=%s, compound_id=%s, retained_metadata=%s",
        library_cfg.name,
        smiles_column,
        compound_id_column,
        retained_metadata_columns,
    )
    if missing_metadata:
        logging.warning("Library %s is missing optional metadata columns: %s", library_cfg.name, missing_metadata)
    return working, retained_metadata_columns, missing_metadata


def build_standardizer(cfg: AppConfig) -> dict[str, Any]:
    return {
        "fragment_parent": rdMolStandardize.FragmentParent,
        "normalizer": rdMolStandardize.Normalizer(),
        "uncharger": rdMolStandardize.Uncharger(),
        "largest_fragment": rdMolStandardize.LargestFragmentChooser(preferOrganic=True),
    }


def contains_mixture(raw_smiles: str, mol: Chem.Mol) -> bool:
    if "." in (raw_smiles or ""):
        return True
    return len(Chem.GetMolFrags(mol)) > 1


def is_inorganic(mol: Chem.Mol) -> bool:
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    if not atomic_numbers:
        return True
    return not any(number == 6 for number in atomic_numbers)


def standardize_smiles(raw_smiles: str, app_cfg: AppConfig, standardizers: dict[str, Any]) -> dict[str, Any]:
    smiles = (raw_smiles or "").strip()
    if not smiles:
        return {"status": "failed", "reason": "missing_smiles"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"status": "failed", "reason": "invalid_smiles"}
    try:
        processed = Chem.Mol(mol)
        if app_cfg.standardization.remove_salts:
            processed = standardizers["fragment_parent"](processed)
        if app_cfg.standardization.keep_largest_fragment:
            processed = standardizers["largest_fragment"].choose(processed)
        if app_cfg.standardization.normalize_molecules:
            processed = standardizers["normalizer"].normalize(processed)
        if app_cfg.standardization.neutralize_if_possible:
            processed = standardizers["uncharger"].uncharge(processed)
        Chem.SanitizeMol(processed)
        standardized_smiles = Chem.MolToSmiles(processed, canonical=app_cfg.standardization.canonicalize_smiles)
        return {
            "status": "ok",
            "reason": "retained",
            "standardized_smiles": standardized_smiles,
            "molecular_weight": float(Descriptors.MolWt(processed)),
            "heavy_atom_count": int(processed.GetNumHeavyAtoms()),
            "mixture_flag": contains_mixture(smiles, processed),
            "inorganic_flag": is_inorganic(processed),
        }
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return {"status": "failed", "reason": f"standardization_error:{exc.__class__.__name__}"}


def standardize_library(working: pd.DataFrame, library_cfg: LibraryConfig, app_cfg: AppConfig, standardizers: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in working.to_dict(orient="records"):
        standardized = standardize_smiles(str(row.get("original_smiles", "")), app_cfg, standardizers)
        combined = dict(row)
        combined.update(standardized)
        records.append(combined)
    frame = pd.DataFrame(records)
    if frame.empty:
        for column in ["status", "reason", "standardized_smiles", "molecular_weight", "heavy_atom_count", "mixture_flag", "inorganic_flag"]:
            frame[column] = pd.Series(dtype="object")
    logging.info("Completed RDKit standardization for library %s (%d rows)", library_cfg.name, len(frame))
    return frame


def split_failed_and_retained(frame: pd.DataFrame, app_cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    failed_mask = frame["status"] != "ok"
    failed = frame.loc[failed_mask].copy()
    retained = frame.loc[~failed_mask].copy()

    def move_rows(reason_mask: pd.Series, reason: str) -> None:
        nonlocal failed, retained
        if not bool(reason_mask.any()):
            return
        moved = retained.loc[reason_mask].copy()
        moved["reason"] = reason
        failed = pd.concat([failed, moved], ignore_index=True)
        retained = retained.loc[~reason_mask].copy()

    if app_cfg.filtering.remove_empty_smiles:
        move_rows(retained["original_smiles"].astype(str).str.strip() == "", "missing_smiles")
    if app_cfg.filtering.remove_invalid_smiles:
        move_rows(retained["standardized_smiles"].astype(str).str.strip() == "", "invalid_smiles")
    if app_cfg.filtering.remove_mixtures:
        move_rows(retained["mixture_flag"].eq(True), "mixture_removed")
    if app_cfg.filtering.remove_inorganic:
        move_rows(retained["inorganic_flag"].eq(True), "inorganic_removed")

    min_mw = app_cfg.filtering.min_molecular_weight
    max_mw = app_cfg.filtering.max_molecular_weight
    move_rows((retained["molecular_weight"] < min_mw) | (retained["molecular_weight"] > max_mw), "molecular_weight_filter")
    move_rows(retained["heavy_atom_count"] < app_cfg.filtering.min_heavy_atoms, "heavy_atom_filter")

    failed = failed.sort_values(["source_library_name", "original_row_index", "screening_compound_id"], kind="mergesort").reset_index(drop=True)
    retained = retained.sort_values(["source_library_name", "original_row_index", "screening_compound_id"], kind="mergesort").reset_index(drop=True)
    return retained, failed


def collapse_within_library_duplicates(retained: pd.DataFrame, remove_duplicates: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if retained.empty:
        provenance = retained.copy()
        return retained.copy(), provenance

    ordered = retained.sort_values(["source_library_name", "standardized_smiles", "original_row_index", "screening_compound_id"], kind="mergesort").copy()
    grouped_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    for (library_name, standardized_smiles), group in ordered.groupby(["source_library_name", "standardized_smiles"], sort=True, dropna=False):
        group = group.sort_values(["original_row_index", "screening_compound_id"], kind="mergesort").copy()
        retained_row = group.iloc[0].copy()
        duplicate_count = int(len(group))
        retained_row["library_specific_duplicate_flag"] = duplicate_count > 1
        retained_row["within_library_duplicate_count"] = duplicate_count
        retained_row["collapsed_original_row_indices"] = "|".join(str(value) for value in group["original_row_index"].tolist())
        retained_row["collapsed_original_compound_ids"] = "|".join(str(value) for value in group["original_compound_id"].tolist())
        retained_row["collapsed_original_smiles"] = "|".join(str(value) for value in group["original_smiles"].tolist())
        retained_row["provenance_record_count"] = duplicate_count
        grouped_rows.append(retained_row.to_dict())

        for _, source_row in group.iterrows():
            provenance_rows.append(
                {
                    "retained_screening_compound_id": retained_row["screening_compound_id"] if remove_duplicates else source_row["screening_compound_id"],
                    "retained_standardized_smiles": standardized_smiles,
                    "source_library_name": library_name,
                    "source_file_path": source_row["source_file_path"],
                    "original_row_index": source_row["original_row_index"],
                    "original_compound_id": source_row["original_compound_id"],
                    "original_smiles": source_row["original_smiles"],
                    "within_library_duplicate_group_size": duplicate_count,
                    "library_specific_duplicate_flag": duplicate_count > 1,
                }
            )

    collapsed = pd.DataFrame(grouped_rows).sort_values(["source_library_name", "original_row_index", "screening_compound_id"], kind="mergesort").reset_index(drop=True)
    if not remove_duplicates:
        expanded = ordered.merge(
            collapsed[
                [
                    "source_library_name",
                    "standardized_smiles",
                    "library_specific_duplicate_flag",
                    "within_library_duplicate_count",
                    "collapsed_original_row_indices",
                    "collapsed_original_compound_ids",
                    "collapsed_original_smiles",
                    "provenance_record_count",
                ]
            ],
            on=["source_library_name", "standardized_smiles"],
            how="left",
        )
        expanded = expanded.sort_values(["source_library_name", "original_row_index", "screening_compound_id"], kind="mergesort").reset_index(drop=True)
        collapsed = expanded
    provenance = pd.DataFrame(provenance_rows).sort_values(["source_library_name", "original_row_index", "retained_screening_compound_id"], kind="mergesort").reset_index(drop=True)
    return collapsed, provenance


def annotate_cross_library_duplicates(merged: pd.DataFrame, remove_across: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if merged.empty:
        empty = merged.copy()
        return empty, pd.DataFrame(), pd.DataFrame()

    ordered = merged.sort_values(["standardized_smiles", "source_library_name", "original_row_index", "screening_compound_id"], kind="mergesort").copy()
    counts = ordered.groupby("standardized_smiles", sort=True).agg(
        cross_library_duplicate_count=("screening_compound_id", "size"),
        number_of_libraries_containing_this_structure=("source_library_name", "nunique"),
    ).reset_index()
    ordered = ordered.merge(counts, on="standardized_smiles", how="left")
    ordered["cross_library_duplicate_flag"] = ordered["number_of_libraries_containing_this_structure"] > 1

    duplicate_summary = ordered.groupby(["standardized_smiles", "source_library_name"], sort=True).agg(
        number_of_rows_collapsed_within_library=("within_library_duplicate_count", "first"),
        number_of_libraries_containing_this_structure=("number_of_libraries_containing_this_structure", "first"),
        cross_library_duplicate_flag=("cross_library_duplicate_flag", "first"),
        retained_screening_compound_id=("screening_compound_id", "first"),
    ).reset_index().sort_values(["standardized_smiles", "source_library_name"], kind="mergesort")

    cross_library_provenance = ordered[[
        "screening_compound_id",
        "standardized_smiles",
        "source_library_name",
        "cross_library_duplicate_flag",
        "cross_library_duplicate_count",
        "number_of_libraries_containing_this_structure",
    ]].copy().rename(columns={"screening_compound_id": "retained_screening_compound_id"})

    if remove_across:
        ordered = ordered.sort_values(["standardized_smiles", "source_library_name", "original_row_index", "screening_compound_id"], kind="mergesort")
        ordered = ordered.drop_duplicates(subset=["standardized_smiles"], keep="first").reset_index(drop=True)

    ordered = ordered.sort_values(["source_library_name", "original_row_index", "screening_compound_id"], kind="mergesort").reset_index(drop=True)
    return ordered, duplicate_summary, cross_library_provenance.drop_duplicates().reset_index(drop=True)


def choose_output_columns(frame: pd.DataFrame) -> list[str]:
    metadata_columns = [column for column in frame.columns if column not in INTERNAL_COLUMN_ORDER and column not in {"status", "reason", "mixture_flag", "inorganic_flag"}]
    ordered = [column for column in INTERNAL_COLUMN_ORDER if column in frame.columns]
    return ordered + sorted(metadata_columns)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    ensure_directory(path.parent)
    frame.to_csv(path, index=False)
    logging.info("Wrote %d rows to %s", len(frame), path)


def display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def build_qc_row(library_name: str, input_row_count: int, standardized_frame: pd.DataFrame, failed_frame: pd.DataFrame, retained_frame: pd.DataFrame) -> dict[str, Any]:
    reason_counts = failed_frame["reason"].value_counts().to_dict() if not failed_frame.empty else {}
    return {
        "library_name": library_name,
        "input_row_count": int(input_row_count),
        "valid_smiles_count": int((standardized_frame["status"] == "ok").sum()) if not standardized_frame.empty else 0,
        "invalid_smiles_count": int(reason_counts.get("invalid_smiles", 0)),
        "removed_empty_smiles_count": int(reason_counts.get("missing_smiles", 0)),
        "removed_mixture_count": int(reason_counts.get("mixture_removed", 0)),
        "removed_inorganic_count": int(reason_counts.get("inorganic_removed", 0)),
        "removed_by_mw_count": int(reason_counts.get("molecular_weight_filter", 0)),
        "removed_by_heavy_atom_count": int(reason_counts.get("heavy_atom_filter", 0)),
        "within_library_duplicate_count": int(max(retained_frame["within_library_duplicate_count"].sum() - len(retained_frame), 0)) if not retained_frame.empty else 0,
        "retained_row_count": int(len(retained_frame)),
        "unique_standardized_smiles_count": int(retained_frame["standardized_smiles"].nunique()) if not retained_frame.empty else 0,
    }


def build_manifest(records: list[dict[str, Any]]) -> pd.DataFrame:
    manifest = pd.DataFrame(records)
    if manifest.empty:
        return pd.DataFrame(columns=["asset_id", "asset_type", "file_path", "library_name", "row_count", "notes"])
    return manifest.sort_values(["asset_type", "library_name", "file_path", "asset_id"], kind="mergesort").reset_index(drop=True)


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logging.info("Wrote JSON report to %s", path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    raw_config = load_yaml(config_path)
    cfg = AppConfig.from_dict(raw_config, project_root)
    log_path = configure_logging(cfg.project_root)
    logging.info("Starting %s", SCRIPT_NAME)

    ensure_directory(cfg.output_library_root)
    ensure_directory(cfg.output_cleaned_root)
    ensure_directory(cfg.output_report_path.parent)
    config_snapshot_path = save_config_snapshot(raw_config, cfg)
    standardizers = build_standardizer(cfg)

    all_retained_frames: list[pd.DataFrame] = []
    all_provenance_frames: list[pd.DataFrame] = []
    manifest_records: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    missing_optional_metadata: dict[str, list[str]] = {}

    for library_cfg in cfg.input_libraries:
        raw_frame = read_library_table(library_cfg)
        initialized_frame, retained_metadata_columns, missing_metadata = initialize_library_frame(raw_frame, library_cfg)
        missing_optional_metadata[library_cfg.name] = missing_metadata
        standardized_frame = standardize_library(initialized_frame, library_cfg, cfg, standardizers)
        retained_frame, failed_frame = split_failed_and_retained(standardized_frame, cfg)
        collapsed_frame, provenance_frame = collapse_within_library_duplicates(
            retained_frame,
            remove_duplicates=cfg.filtering.remove_duplicates_within_library,
        )

        if cfg.save_library_specific_outputs:
            cleaned_path = cfg.output_cleaned_root / f"{library_cfg.name}_cleaned.csv"
            cleaned_output = collapsed_frame.reindex(columns=choose_output_columns(collapsed_frame))
            write_table(cleaned_output, cleaned_path)
            manifest_records.append(
                {
                    "asset_id": f"cleaned_library::{library_cfg.name}",
                    "asset_type": "cleaned_library",
                    "file_path": display_path(cleaned_path, cfg.project_root),
                    "library_name": library_cfg.name,
                    "row_count": int(len(cleaned_output)),
                    "notes": "Library-specific standardized and de-duplicated screening library.",
                }
            )

        if cfg.save_failed_rows:
            failed_path = cfg.output_cleaned_root / f"{library_cfg.name}_failed_rows.csv"
            failed_output = failed_frame.copy()
            failed_columns = choose_output_columns(failed_output) + [column for column in ["reason", "status"] if column in failed_output.columns and column not in choose_output_columns(failed_output)]
            write_table(failed_output.reindex(columns=failed_columns), failed_path)
            manifest_records.append(
                {
                    "asset_id": f"failed_rows::{library_cfg.name}",
                    "asset_type": "failed_rows",
                    "file_path": display_path(failed_path, cfg.project_root),
                    "library_name": library_cfg.name,
                    "row_count": int(len(failed_output)),
                    "notes": "Rows removed during screening-library preparation, with removal reasons.",
                }
            )

        qc_rows.append(build_qc_row(library_cfg.name, len(raw_frame), standardized_frame, failed_frame, collapsed_frame))
        all_retained_frames.append(collapsed_frame)
        all_provenance_frames.append(provenance_frame)
        if missing_metadata:
            warnings.append(f"Library {library_cfg.name} missing optional metadata columns: {missing_metadata}")
        if retained_metadata_columns:
            logging.info("Library %s retained metadata columns: %s", library_cfg.name, retained_metadata_columns)

    merged_input = pd.concat(all_retained_frames, ignore_index=True) if all_retained_frames else pd.DataFrame()
    merged_library, duplicate_summary, cross_library_provenance = annotate_cross_library_duplicates(
        merged_input,
        remove_across=cfg.filtering.remove_duplicates_across_libraries,
    )

    provenance_table = pd.concat(all_provenance_frames, ignore_index=True) if all_provenance_frames else pd.DataFrame()
    if not provenance_table.empty:
        provenance_table = provenance_table.merge(
            cross_library_provenance.rename(columns={"standardized_smiles": "retained_standardized_smiles"}),
            on=["retained_screening_compound_id", "retained_standardized_smiles", "source_library_name"],
            how="left",
        )
        provenance_table = provenance_table.sort_values(["source_library_name", "original_row_index", "retained_screening_compound_id"], kind="mergesort").reset_index(drop=True)

    merged_output = merged_library.reindex(columns=choose_output_columns(merged_library))
    write_table(merged_output, cfg.output_merged_library_path)
    write_table(provenance_table, cfg.output_provenance_path)
    write_table(duplicate_summary, cfg.output_duplicate_summary_path)

    qc_rows.append(
        {
            "library_name": "__merged__",
            "input_row_count": int(sum(row["input_row_count"] for row in qc_rows)),
            "valid_smiles_count": int(sum(row["valid_smiles_count"] for row in qc_rows)),
            "invalid_smiles_count": int(sum(row["invalid_smiles_count"] for row in qc_rows)),
            "removed_empty_smiles_count": int(sum(row["removed_empty_smiles_count"] for row in qc_rows)),
            "removed_mixture_count": int(sum(row["removed_mixture_count"] for row in qc_rows)),
            "removed_inorganic_count": int(sum(row["removed_inorganic_count"] for row in qc_rows)),
            "removed_by_mw_count": int(sum(row["removed_by_mw_count"] for row in qc_rows)),
            "removed_by_heavy_atom_count": int(sum(row["removed_by_heavy_atom_count"] for row in qc_rows)),
            "within_library_duplicate_count": int(sum(row["within_library_duplicate_count"] for row in qc_rows)),
            "retained_row_count": int(len(merged_output)),
            "unique_standardized_smiles_count": int(merged_output["standardized_smiles"].nunique()) if not merged_output.empty else 0,
        }
    )
    qc_summary = pd.DataFrame(qc_rows).sort_values(["library_name"], kind="mergesort").reset_index(drop=True)
    write_table(qc_summary, cfg.output_qc_summary_path)

    manifest_records.extend(
        [
            {
                "asset_id": "merged_screening_library",
                "asset_type": "merged_library",
                "file_path": display_path(cfg.output_merged_library_path, cfg.project_root),
                "library_name": "",
                "row_count": int(len(merged_output)),
                "notes": "Merged retained screening library prepared for later feature mapping and inference.",
            },
            {
                "asset_id": "screening_library_provenance",
                "asset_type": "provenance",
                "file_path": display_path(cfg.output_provenance_path, cfg.project_root),
                "library_name": "",
                "row_count": int(len(provenance_table)),
                "notes": "Original-to-retained provenance mappings for all retained screening compounds.",
            },
            {
                "asset_id": "screening_duplicate_summary",
                "asset_type": "duplicate_summary",
                "file_path": display_path(cfg.output_duplicate_summary_path, cfg.project_root),
                "library_name": "",
                "row_count": int(len(duplicate_summary)),
                "notes": "Duplicate annotations across and within screening libraries.",
            },
            {
                "asset_id": "screening_qc_summary",
                "asset_type": "qc_summary",
                "file_path": display_path(cfg.output_qc_summary_path, cfg.project_root),
                "library_name": "",
                "row_count": int(len(qc_summary)),
                "notes": "Library-level and merged chemistry QC summary statistics.",
            },
        ]
    )

    manifest = build_manifest(manifest_records)
    write_table(manifest, cfg.output_manifest_path)

    manifest = pd.concat(
        [
            manifest,
            pd.DataFrame(
                [
                    {
                        "asset_id": "screening_manifest",
                        "asset_type": "manifest",
                        "file_path": display_path(cfg.output_manifest_path, cfg.project_root),
                        "library_name": "",
                        "row_count": int(len(manifest)),
                        "notes": "Manifest of Step-13A screening-library preparation assets.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    write_table(manifest.sort_values(["asset_type", "file_path"], kind="mergesort").reset_index(drop=True), cfg.output_manifest_path)

    duplicate_stats = {
        "within_library_duplicate_rows_collapsed": int(sum(row["within_library_duplicate_count"] for row in qc_rows if row["library_name"] != "__merged__")),
        "cross_library_duplicate_retained_rows": int(merged_output["cross_library_duplicate_flag"].fillna(False).sum()) if not merged_output.empty else 0,
        "cross_library_unique_structures_with_duplicates": int(duplicate_summary["cross_library_duplicate_flag"].fillna(False).sum()) if not duplicate_summary.empty else 0,
        "cross_library_duplicates_removed": bool(cfg.filtering.remove_duplicates_across_libraries),
    }
    report_payload = {
        "script_name": SCRIPT_NAME,
        "input_libraries_processed": [library.name for library in cfg.input_libraries],
        "source_paths": {library.name: str(library.path) for library in cfg.input_libraries},
        "total_input_rows_across_libraries": int(sum(row["input_row_count"] for row in qc_rows if row["library_name"] != "__merged__")),
        "total_retained_rows_across_libraries": int(len(merged_output)),
        "total_unique_standardized_compounds": int(merged_output["standardized_smiles"].nunique()) if not merged_output.empty else 0,
        "duplicate_statistics": duplicate_stats,
        "qc_summaries": qc_summary.to_dict(orient="records"),
        "missing_optional_metadata_fields": missing_optional_metadata,
        "warnings": warnings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_snapshot_reference": "" if config_snapshot_path is None else str(config_snapshot_path),
        "log_path": str(log_path),
        "manifest_path": str(cfg.output_manifest_path),
        "outputs": {
            "merged_library": str(cfg.output_merged_library_path),
            "provenance": str(cfg.output_provenance_path),
            "duplicate_summary": str(cfg.output_duplicate_summary_path),
            "qc_summary": str(cfg.output_qc_summary_path),
            "manifest": str(cfg.output_manifest_path),
        },
    }
    write_json(report_payload, cfg.output_report_path)
    logging.info("Completed %s successfully with %d retained rows.", SCRIPT_NAME, len(merged_output))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI safety guard
        logging.exception("%s failed: %s", SCRIPT_NAME, exc)
        raise
