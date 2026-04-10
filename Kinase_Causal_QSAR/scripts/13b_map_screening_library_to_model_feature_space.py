#!/usr/bin/env python3
"""Map prepared screening compounds into training-consistent model feature spaces.

This script is a strict continuation of the kinase causality QSAR pipeline.
It loads the standardized screening library created by Script-13A, resolves
training-time feature-generation settings from Steps 07-09 when available,
generates deterministic classical / graph / environment feature assets,
performs feature-space QC checks, writes structured manifests and reports, and
prepares inference-ready screening inputs without scoring, ranking, or
retraining models.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

SCRIPT_NAME = "13b_map_screening_library_to_model_feature_space"
REQUIRED_SCRIPT_13B_KEYS = {
    "input_screening_library_path",
    "input_step07_config_path",
    "input_step08_config_path",
    "input_step09_config_path",
    "input_training_annotated_long_path",
    "input_compound_env_path",
    "input_kinase_env_path",
    "output_feature_root",
    "output_classical_feature_path",
    "output_graph_manifest_path",
    "output_environment_feature_path",
    "output_feature_qc_summary_path",
    "output_manifest_path",
    "output_report_path",
    "generate_classical_features",
    "generate_graph_inputs",
    "generate_environment_features",
    "generate_applicability_reference_features",
    "classical_features",
    "graph_features",
    "environment_features",
    "applicability_reference",
    "save_failed_rows",
    "save_config_snapshot",
    "chunk_size",
}
REQUIRED_CLASSICAL_KEYS = {
    "use_morgan_fingerprints",
    "morgan_radius",
    "morgan_nbits",
    "include_rdkit_2d_descriptors",
}
REQUIRED_GRAPH_KEYS = {
    "use_atom_type",
    "use_degree",
    "use_formal_charge",
    "use_hybridization",
    "use_aromaticity",
    "use_num_hs",
    "use_chirality",
    "use_bond_type",
    "use_conjugation",
    "use_ring_status",
    "use_stereo",
}
REQUIRED_ENVIRONMENT_KEYS = {
    "compute_murcko_scaffolds",
    "compute_generic_murcko_scaffolds",
    "map_kinase_family_vocabularies",
    "preserve_source_library_metadata",
}
REQUIRED_APPLICABILITY_KEYS = {
    "compare_to_training_compounds",
    "compute_basic_distance_proxies",
    "save_training_feature_reference_summary",
}
REQUIRED_SCREENING_COLUMNS = {"screening_compound_id", "standardized_smiles"}
OPTIONAL_METADATA_COLUMNS = [
    "original_smiles",
    "source_library_name",
    "source_file_path",
    "original_compound_id",
    "original_row_index",
]
DEFAULT_FAIL_SEVERITIES = {"critical", "error"}


@dataclass(frozen=True)
class ClassicalFeatureConfig:
    use_morgan_fingerprints: bool
    morgan_radius: int
    morgan_nbits: int
    include_rdkit_2d_descriptors: bool


@dataclass(frozen=True)
class GraphFeatureConfig:
    use_atom_type: bool
    use_degree: bool
    use_formal_charge: bool
    use_hybridization: bool
    use_aromaticity: bool
    use_num_hs: bool
    use_chirality: bool
    use_bond_type: bool
    use_conjugation: bool
    use_ring_status: bool
    use_stereo: bool


@dataclass(frozen=True)
class EnvironmentFeatureConfig:
    compute_murcko_scaffolds: bool
    compute_generic_murcko_scaffolds: bool
    map_kinase_family_vocabularies: bool
    preserve_source_library_metadata: bool


@dataclass(frozen=True)
class ApplicabilityReferenceConfig:
    compare_to_training_compounds: bool
    compute_basic_distance_proxies: bool
    save_training_feature_reference_summary: bool


@dataclass(frozen=True)
class AppConfig:
    input_screening_library_path: Path
    input_step07_config_path: Path
    input_step08_config_path: Path
    input_step09_config_path: Path
    input_training_annotated_long_path: Path
    input_compound_env_path: Path
    input_kinase_env_path: Path
    output_feature_root: Path
    output_classical_feature_path: Path
    output_graph_manifest_path: Path
    output_environment_feature_path: Path
    output_feature_qc_summary_path: Path
    output_manifest_path: Path
    output_report_path: Path
    generate_classical_features: bool
    generate_graph_inputs: bool
    generate_environment_features: bool
    generate_applicability_reference_features: bool
    classical_features: ClassicalFeatureConfig
    graph_features: GraphFeatureConfig
    environment_features: EnvironmentFeatureConfig
    applicability_reference: ApplicabilityReferenceConfig
    save_failed_rows: bool
    save_config_snapshot: bool
    chunk_size: int
    project_root: Path
    logs_dir: Path
    configs_used_dir: Path

    @staticmethod
    def from_dict(raw: dict[str, Any], project_root: Path) -> "AppConfig":
        section = raw.get("script_13b")
        if not isinstance(section, dict):
            raise ValueError("Missing required `script_13b` section in config.yaml.")
        missing = sorted(REQUIRED_SCRIPT_13B_KEYS.difference(section))
        if missing:
            raise ValueError("Missing required script_13b config values: " + ", ".join(missing))

        def resolve(path_like: Any) -> Path:
            path = Path(str(path_like))
            return path if path.is_absolute() else project_root / path

        def parse_bool(value: Any, key: str) -> bool:
            if not isinstance(value, bool):
                raise ValueError(f"script_13b.{key} must be boolean; got {value!r}.")
            return value

        def parse_mapping(section_name: str, required: set[str]) -> dict[str, Any]:
            value = section.get(section_name)
            if not isinstance(value, dict):
                raise ValueError(f"script_13b.{section_name} must be a mapping.")
            missing_keys = sorted(required.difference(value))
            if missing_keys:
                raise ValueError(f"Missing required script_13b.{section_name} values: {', '.join(missing_keys)}")
            return value

        classical_raw = parse_mapping("classical_features", REQUIRED_CLASSICAL_KEYS)
        graph_raw = parse_mapping("graph_features", REQUIRED_GRAPH_KEYS)
        environment_raw = parse_mapping("environment_features", REQUIRED_ENVIRONMENT_KEYS)
        applicability_raw = parse_mapping("applicability_reference", REQUIRED_APPLICABILITY_KEYS)

        return AppConfig(
            input_screening_library_path=resolve(section["input_screening_library_path"]),
            input_step07_config_path=resolve(section["input_step07_config_path"]),
            input_step08_config_path=resolve(section["input_step08_config_path"]),
            input_step09_config_path=resolve(section["input_step09_config_path"]),
            input_training_annotated_long_path=resolve(section["input_training_annotated_long_path"]),
            input_compound_env_path=resolve(section["input_compound_env_path"]),
            input_kinase_env_path=resolve(section["input_kinase_env_path"]),
            output_feature_root=resolve(section["output_feature_root"]),
            output_classical_feature_path=resolve(section["output_classical_feature_path"]),
            output_graph_manifest_path=resolve(section["output_graph_manifest_path"]),
            output_environment_feature_path=resolve(section["output_environment_feature_path"]),
            output_feature_qc_summary_path=resolve(section["output_feature_qc_summary_path"]),
            output_manifest_path=resolve(section["output_manifest_path"]),
            output_report_path=resolve(section["output_report_path"]),
            generate_classical_features=parse_bool(section["generate_classical_features"], "generate_classical_features"),
            generate_graph_inputs=parse_bool(section["generate_graph_inputs"], "generate_graph_inputs"),
            generate_environment_features=parse_bool(section["generate_environment_features"], "generate_environment_features"),
            generate_applicability_reference_features=parse_bool(section["generate_applicability_reference_features"], "generate_applicability_reference_features"),
            classical_features=ClassicalFeatureConfig(
                use_morgan_fingerprints=parse_bool(classical_raw["use_morgan_fingerprints"], "classical_features.use_morgan_fingerprints"),
                morgan_radius=int(classical_raw["morgan_radius"]),
                morgan_nbits=int(classical_raw["morgan_nbits"]),
                include_rdkit_2d_descriptors=parse_bool(classical_raw["include_rdkit_2d_descriptors"], "classical_features.include_rdkit_2d_descriptors"),
            ),
            graph_features=GraphFeatureConfig(**{k: parse_bool(v, f"graph_features.{k}") for k, v in graph_raw.items()}),
            environment_features=EnvironmentFeatureConfig(**{k: parse_bool(v, f"environment_features.{k}") for k, v in environment_raw.items()}),
            applicability_reference=ApplicabilityReferenceConfig(**{k: parse_bool(v, f"applicability_reference.{k}") for k, v in applicability_raw.items()}),
            save_failed_rows=parse_bool(section["save_failed_rows"], "save_failed_rows"),
            save_config_snapshot=parse_bool(section["save_config_snapshot"], "save_config_snapshot"),
            chunk_size=max(1, int(section["chunk_size"])),
            project_root=project_root,
            logs_dir=project_root / "logs",
            configs_used_dir=project_root / "configs_used",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to pipeline config YAML.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must parse to a mapping: {path}")
    return data


def setup_logging(cfg: AppConfig) -> Path:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = cfg.logs_dir / f"{SCRIPT_NAME}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.info("Logging initialized: %s", log_path)
    return log_path


def save_config_snapshot(config_path: Path, cfg: AppConfig) -> Path | None:
    if not cfg.save_config_snapshot:
        return None
    cfg.configs_used_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = cfg.configs_used_dir / f"{SCRIPT_NAME}_config.yaml"
    snapshot_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    logging.info("Saved config snapshot to %s", snapshot_path)
    return snapshot_path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_optional_yaml(path: Path, label: str, warnings_list: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        warnings_list.append(f"Optional {label} config snapshot not found: {path}")
        logging.warning("Optional %s config snapshot not found: %s", label, path)
        return None
    payload = load_yaml(path)
    logging.info("Loaded %s config snapshot from %s", label, path)
    return payload


def load_required_dataframe(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} file not found: {path}")
    frame = pd.read_csv(path)
    logging.info("Loaded %s from %s with shape %s", label, path, frame.shape)
    return frame


def load_optional_dataframe(path: Path, label: str, warnings_list: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        warnings_list.append(f"Optional {label} file not found: {path}")
        logging.warning("Optional %s file not found: %s", label, path)
        return None
    frame = pd.read_csv(path)
    logging.info("Loaded %s from %s with shape %s", label, path, frame.shape)
    return frame


def normalize_screening_library(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_SCREENING_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError("Screening library is missing required columns: " + ", ".join(missing))
    normalized = df.copy()
    for column in OPTIONAL_METADATA_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    ordered_columns = ["screening_compound_id", "standardized_smiles", *OPTIONAL_METADATA_COLUMNS]
    extra_columns = [c for c in normalized.columns if c not in ordered_columns]
    normalized = normalized[ordered_columns + extra_columns]
    normalized["screening_compound_id"] = normalized["screening_compound_id"].astype(str)
    normalized["standardized_smiles"] = normalized["standardized_smiles"].astype(str)
    normalized = normalized.drop_duplicates(subset=["screening_compound_id"], keep="first").sort_values("screening_compound_id", kind="mergesort").reset_index(drop=True)
    return normalized


def rdkit_descriptor_names() -> list[str]:
    return [name for name, _ in Descriptors._descList]


def feature_columns_from_training_reference(path: Path, prefixes: tuple[str, ...], id_candidates: tuple[str, ...]) -> list[str] | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception as exc:
        logging.warning("Could not inspect training reference file %s: %s", path, exc)
        return None
    excluded = set(id_candidates)
    return [c for c in df.columns if c not in excluded and c.startswith(prefixes)]


def resolve_classical_feature_settings(cfg: AppConfig, step07_cfg: dict[str, Any] | None, warnings_list: list[str]) -> dict[str, Any]:
    resolved = {
        "use_morgan_fingerprints": cfg.classical_features.use_morgan_fingerprints,
        "morgan_radius": cfg.classical_features.morgan_radius,
        "morgan_nbits": cfg.classical_features.morgan_nbits,
        "include_rdkit_2d_descriptors": cfg.classical_features.include_rdkit_2d_descriptors,
        "source": "script_13b_config_fallback",
    }
    if isinstance(step07_cfg, dict) and isinstance(step07_cfg.get("script_07"), dict):
        source = step07_cfg["script_07"]
        for key in ["use_morgan_fingerprints", "morgan_radius", "morgan_nbits", "include_rdkit_2d_descriptors"]:
            if key in source:
                resolved[key] = source[key]
        resolved["source"] = str(cfg.input_step07_config_path)
    else:
        warnings_list.append("Falling back to script_13b.classical_features because Step-07 config snapshot could not be loaded.")
    return resolved


def resolve_graph_feature_settings(cfg: AppConfig, step08_cfg: dict[str, Any] | None, warnings_list: list[str]) -> dict[str, bool]:
    resolved = {k: getattr(cfg.graph_features, k) for k in REQUIRED_GRAPH_KEYS}
    if isinstance(step08_cfg, dict) and isinstance(step08_cfg.get("script_08"), dict):
        script_cfg = step08_cfg["script_08"]
        node_features = script_cfg.get("node_features", {}) if isinstance(script_cfg.get("node_features"), dict) else {}
        edge_features = script_cfg.get("edge_features", {}) if isinstance(script_cfg.get("edge_features"), dict) else {}
        for key in ["use_atom_type", "use_degree", "use_formal_charge", "use_hybridization", "use_aromaticity", "use_num_hs", "use_chirality"]:
            if key in node_features:
                resolved[key] = bool(node_features[key])
        for key in ["use_bond_type", "use_conjugation", "use_ring_status", "use_stereo"]:
            if key in edge_features:
                resolved[key] = bool(edge_features[key])
    else:
        warnings_list.append("Falling back to script_13b.graph_features because Step-08 config snapshot could not be loaded.")
    return resolved


def resolve_environment_settings(cfg: AppConfig, step09_cfg: dict[str, Any] | None, warnings_list: list[str]) -> dict[str, Any]:
    resolved = {
        "environment_columns": {},
        "source": "script_13b_config_fallback",
    }
    if isinstance(step09_cfg, dict) and isinstance(step09_cfg.get("script_09"), dict):
        script_cfg = step09_cfg["script_09"]
        env_cols = script_cfg.get("environment_columns")
        if isinstance(env_cols, dict):
            resolved["environment_columns"] = env_cols
        resolved["source"] = str(cfg.input_step09_config_path)
    else:
        warnings_list.append("Falling back to script_13b.environment_features because Step-09 config snapshot could not be loaded.")
    return resolved


def build_mol(smiles: str) -> Chem.Mol | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def build_classical_feature_columns(settings: dict[str, Any]) -> list[str]:
    feature_columns: list[str] = ["rdkit_parse_success"]
    if bool(settings["use_morgan_fingerprints"]):
        feature_columns.extend([f"morgan_{bit_idx}" for bit_idx in range(int(settings["morgan_nbits"]))])
    if bool(settings["include_rdkit_2d_descriptors"]):
        feature_columns.extend([f"rdkit_{name}" for name in rdkit_descriptor_names()])
    return feature_columns


def build_classical_output_columns(metadata_columns: list[str], feature_columns: list[str], training_feature_columns: list[str] | None) -> list[str]:
    base_columns = ["screening_compound_id", "standardized_smiles", *metadata_columns]
    if training_feature_columns:
        return base_columns + training_feature_columns
    return base_columns + feature_columns


def generate_classical_features_chunk(
    chunk_df: pd.DataFrame,
    metadata_columns: list[str],
    expected_columns: list[str],
    descriptor_specs: list[tuple[str, Any]],
    morgan_generator: Any | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    feature_columns = [c for c in expected_columns if c.startswith("morgan_") or c.startswith("rdkit_") or c == "rdkit_parse_success"]
    morgan_columns = [c for c in expected_columns if c.startswith("morgan_")]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in chunk_df.itertuples(index=False):
        record = {"screening_compound_id": row.screening_compound_id, "standardized_smiles": row.standardized_smiles}
        for column in metadata_columns:
            record[column] = getattr(row, column)
        mol = build_mol(str(row.standardized_smiles))
        record["rdkit_parse_success"] = int(mol is not None)
        if mol is None:
            for column in feature_columns:
                if column != "rdkit_parse_success":
                    record[column] = np.nan
            failures.append({"screening_compound_id": row.screening_compound_id, "standardized_smiles": row.standardized_smiles, "failure_reason": "RDKit MolFromSmiles returned None"})
            records.append(record)
            continue
        if morgan_generator is not None and morgan_columns:
            fp = morgan_generator.GetFingerprint(mol)
            arr = np.zeros((len(morgan_columns),), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            for bit_idx, value in enumerate(arr.tolist()):
                record[f"morgan_{bit_idx}"] = int(value)
        for descriptor_name, func in descriptor_specs:
            column_name = f"rdkit_{descriptor_name}"
            if column_name not in expected_columns:
                continue
            try:
                record[column_name] = float(func(mol))
            except Exception as exc:
                record[column_name] = np.nan
                logging.warning("Descriptor %s failed for %s: %s", descriptor_name, row.screening_compound_id, exc)
        records.append(record)

    chunk_features = pd.DataFrame.from_records(records)
    if chunk_features.empty:
        chunk_features = pd.DataFrame(columns=expected_columns)
    chunk_features = chunk_features.sort_values("screening_compound_id", kind="mergesort").reset_index(drop=True).reindex(columns=expected_columns)
    chunk_failures = pd.DataFrame.from_records(failures)
    metadata = {
        "row_count": int(len(chunk_features)),
        "success_count": int((chunk_features["rdkit_parse_success"] == 1).sum()) if "rdkit_parse_success" in chunk_features.columns else 0,
        "failure_count": int((chunk_features["rdkit_parse_success"] == 0).sum()) if "rdkit_parse_success" in chunk_features.columns else 0,
    }
    return chunk_features, chunk_failures, metadata


def generate_classical_features(
    screening_df: pd.DataFrame,
    metadata_columns: list[str],
    settings: dict[str, Any],
    output_path: Path,
    chunk_size: int,
    save_failed_rows: bool,
    failed_rows_path: Path | None,
    training_feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    descriptor_specs = Descriptors._descList if bool(settings["include_rdkit_2d_descriptors"]) else []
    feature_columns = build_classical_feature_columns(settings)
    expected_columns = build_classical_output_columns(metadata_columns, feature_columns, training_feature_columns)
    missing_training_columns = sorted(set(training_feature_columns or []).difference(feature_columns))
    if missing_training_columns:
        raise RuntimeError("Generated classical features do not satisfy the training-time schema: " + ", ".join(missing_training_columns[:20]))

    ensure_parent(output_path)
    if output_path.exists():
        output_path.unlink()
    if failed_rows_path is not None and failed_rows_path.exists():
        failed_rows_path.unlink()

    morgan_generator = None
    if bool(settings["use_morgan_fingerprints"]):
        morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=int(settings["morgan_radius"]),
            fpSize=int(settings["morgan_nbits"]),
        )

    total_rows = int(len(screening_df))
    success_count = 0
    failure_count = 0
    rows_written = 0
    chunk_index = 0
    for start in range(0, total_rows, chunk_size):
        chunk_df = screening_df.iloc[start:start + chunk_size]
        chunk_features, chunk_failures, chunk_meta = generate_classical_features_chunk(
            chunk_df=chunk_df,
            metadata_columns=metadata_columns,
            expected_columns=expected_columns,
            descriptor_specs=descriptor_specs,
            morgan_generator=morgan_generator,
        )
        write_mode = "w" if chunk_index == 0 else "a"
        chunk_features.to_csv(output_path, mode=write_mode, index=False, header=(chunk_index == 0))
        logging.info("Wrote chunk %s to disk: rows=%s path=%s", chunk_index + 1, len(chunk_features), output_path)
        if save_failed_rows and failed_rows_path is not None and not chunk_failures.empty:
            chunk_failures.to_csv(failed_rows_path, mode=("w" if chunk_index == 0 and not failed_rows_path.exists() else "a"), index=False, header=not failed_rows_path.exists())
        rows_written += int(chunk_meta["row_count"])
        success_count += int(chunk_meta["success_count"])
        failure_count += int(chunk_meta["failure_count"])
        chunk_index += 1
        logging.info("Processed %s / %s rows", min(start + len(chunk_df), total_rows), total_rows)

    metadata = {
        "row_count": rows_written,
        "success_count": success_count,
        "failure_count": failure_count,
        "feature_column_count": int(len(feature_columns)),
        "output_path": str(output_path),
        "columns": expected_columns,
    }
    return pd.DataFrame(columns=expected_columns), metadata


def atom_feature_names(graph_cfg: dict[str, bool]) -> list[str]:
    names: list[str] = []
    if graph_cfg.get("use_atom_type", True):
        names.extend([f"atom_type_{symbol}" for symbol in ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "other"]])
    if graph_cfg.get("use_degree", True):
        names.append("atom_degree")
    if graph_cfg.get("use_formal_charge", True):
        names.append("formal_charge")
    if graph_cfg.get("use_hybridization", True):
        names.extend([f"hybridization_{name}" for name in ["SP", "SP2", "SP3", "other"]])
    if graph_cfg.get("use_aromaticity", True):
        names.append("is_aromatic")
    if graph_cfg.get("use_num_hs", True):
        names.append("total_num_hs")
    if graph_cfg.get("use_chirality", True):
        names.extend(["chirality_unspecified", "chirality_cw", "chirality_ccw"])
    return names


def bond_feature_names(graph_cfg: dict[str, bool]) -> list[str]:
    names: list[str] = []
    if graph_cfg.get("use_bond_type", True):
        names.extend([f"bond_type_{name}" for name in ["single", "double", "triple", "aromatic", "other"]])
    if graph_cfg.get("use_conjugation", True):
        names.append("is_conjugated")
    if graph_cfg.get("use_ring_status", True):
        names.append("is_in_ring")
    if graph_cfg.get("use_stereo", True):
        names.extend(["stereo_none", "stereo_any", "stereo_z", "stereo_e", "stereo_other"])
    return names


def generate_graph_manifest(screening_df: pd.DataFrame, graph_cfg: dict[str, bool], metadata_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    node_names = atom_feature_names(graph_cfg)
    edge_names = bond_feature_names(graph_cfg)
    for row in screening_df.itertuples(index=False):
        record = {
            "screening_compound_id": row.screening_compound_id,
            "standardized_smiles": row.standardized_smiles,
            "graph_success_flag": 0,
            "graph_failure_reason": "",
            "number_of_atoms": 0,
            "number_of_bonds": 0,
            "node_feature_dimension": len(node_names),
            "edge_feature_dimension": len(edge_names),
            "serialized_graph_path": "",
        }
        for column in metadata_columns:
            record[column] = getattr(row, column)
        mol = build_mol(str(row.standardized_smiles))
        if mol is None:
            reason = "RDKit MolFromSmiles returned None"
            record["graph_failure_reason"] = reason
            failures.append({"screening_compound_id": row.screening_compound_id, "standardized_smiles": row.standardized_smiles, "failure_reason": reason})
            records.append(record)
            continue
        record["graph_success_flag"] = 1
        record["number_of_atoms"] = int(mol.GetNumAtoms())
        record["number_of_bonds"] = int(mol.GetNumBonds())
        records.append(record)
    manifest = pd.DataFrame.from_records(records).sort_values("screening_compound_id", kind="mergesort").reset_index(drop=True)
    failed_rows = pd.DataFrame.from_records(failures)
    metadata = {
        "row_count": int(len(manifest)),
        "success_count": int(manifest["graph_success_flag"].sum()),
        "failure_count": int((manifest["graph_success_flag"] == 0).sum()),
        "node_feature_dimension": int(len(node_names)),
        "edge_feature_dimension": int(len(edge_names)),
    }
    return manifest, failed_rows, metadata


def safe_scaffold_smiles(mol: Chem.Mol | None, generic: bool = False) -> str | None:
    if mol is None:
        return None
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold_mol is None:
            return None
        if generic:
            scaffold_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
        return Chem.MolToSmiles(scaffold_mol, canonical=True)
    except Exception:
        return None


def aromatic_ring_count(mol: Chem.Mol | None) -> int | float:
    if mol is None:
        return np.nan
    ring_info = mol.GetRingInfo()
    count = 0
    for atom_indices in ring_info.AtomRings():
        if atom_indices and all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in atom_indices):
            count += 1
    return count


def generate_environment_features(screening_df: pd.DataFrame, metadata_columns: list[str], env_cfg: EnvironmentFeatureConfig, env_settings: dict[str, Any], compound_env_df: pd.DataFrame | None, kinase_env_df: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    family_vocab: set[str] = set()
    if env_cfg.map_kinase_family_vocabularies and kinase_env_df is not None and "kinase_family" in kinase_env_df.columns:
        family_vocab = {str(value) for value in kinase_env_df["kinase_family"].dropna().astype(str).tolist()}
    for row in screening_df.itertuples(index=False):
        mol = build_mol(str(row.standardized_smiles))
        record = {"screening_compound_id": row.screening_compound_id, "standardized_smiles": row.standardized_smiles, "environment_parse_success": int(mol is not None)}
        for column in metadata_columns:
            record[column] = getattr(row, column)
        if mol is None:
            reason = "RDKit MolFromSmiles returned None"
            failures.append({"screening_compound_id": row.screening_compound_id, "standardized_smiles": row.standardized_smiles, "failure_reason": reason})
            numeric_columns = [
                "molecular_weight",
                "clogp",
                "tpsa",
                "hbd",
                "hba",
                "rotatable_bonds",
                "heavy_atom_count",
                "aromatic_ring_count",
                "formal_charge",
                "fraction_csp3",
            ]
            for column in numeric_columns:
                record[column] = np.nan
            record["murcko_scaffold"] = None
            record["generic_murcko_scaffold"] = None
            if env_cfg.map_kinase_family_vocabularies:
                record["kinase_family_vocab_match"] = pd.NA
            records.append(record)
            continue
        record["murcko_scaffold"] = safe_scaffold_smiles(mol, generic=False) if env_cfg.compute_murcko_scaffolds else None
        record["generic_murcko_scaffold"] = safe_scaffold_smiles(mol, generic=True) if env_cfg.compute_generic_murcko_scaffolds else None
        record["molecular_weight"] = float(Descriptors.MolWt(mol))
        record["clogp"] = float(Crippen.MolLogP(mol))
        record["tpsa"] = float(rdMolDescriptors.CalcTPSA(mol))
        record["hbd"] = int(Lipinski.NumHDonors(mol))
        record["hba"] = int(Lipinski.NumHAcceptors(mol))
        record["rotatable_bonds"] = int(Lipinski.NumRotatableBonds(mol))
        record["heavy_atom_count"] = int(mol.GetNumHeavyAtoms())
        record["aromatic_ring_count"] = int(aromatic_ring_count(mol))
        record["formal_charge"] = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
        try:
            record["fraction_csp3"] = float(rdMolDescriptors.CalcFractionCSP3(mol))
        except Exception:
            record["fraction_csp3"] = np.nan
        if env_cfg.map_kinase_family_vocabularies:
            mapped_value = row.source_library_name if pd.notna(row.source_library_name) else None
            record["kinase_family_vocab_match"] = bool(mapped_value in family_vocab) if mapped_value is not None and family_vocab else False
        records.append(record)
    env_df = pd.DataFrame.from_records(records).sort_values("screening_compound_id", kind="mergesort").reset_index(drop=True)
    if compound_env_df is not None and "standardized_smiles" in compound_env_df.columns:
        scaffold_reference = compound_env_df[[c for c in ["standardized_smiles", "murcko_scaffold", "generic_murcko_scaffold"] if c in compound_env_df.columns]].drop_duplicates()
        env_df = env_df.merge(scaffold_reference, on="standardized_smiles", how="left", suffixes=("", "_training_reference"), sort=False)
    failed_rows = pd.DataFrame.from_records(failures)
    metadata = {
        "row_count": int(len(env_df)),
        "success_count": int((env_df["environment_parse_success"] == 1).sum()),
        "failure_count": int((env_df["environment_parse_success"] == 0).sum()),
        "environment_columns_source": env_settings.get("source", "script_13b_config_fallback"),
    }
    return env_df, failed_rows, metadata


def build_qc_summary(classical_df: pd.DataFrame | None, graph_df: pd.DataFrame | None, env_df: pd.DataFrame | None, training_feature_columns: list[str] | None, classical_settings: dict[str, Any], graph_settings: dict[str, bool], env_settings: dict[str, Any]) -> pd.DataFrame:
    qc_records: list[dict[str, Any]] = []
    observed_morgan = len([c for c in (classical_df.columns if classical_df is not None else []) if c.startswith("morgan_")])
    qc_records.append({
        "feature_block": "classical_fingerprint_length",
        "expected_setting": int(classical_settings["morgan_nbits"]) if classical_settings["use_morgan_fingerprints"] else 0,
        "observed_setting": int(observed_morgan),
        "match_flag": bool(observed_morgan == (int(classical_settings["morgan_nbits"]) if classical_settings["use_morgan_fingerprints"] else 0)),
        "severity": "critical" if classical_settings["use_morgan_fingerprints"] and observed_morgan != int(classical_settings["morgan_nbits"]) else "info",
        "notes": "Morgan fingerprint dimensionality check.",
    })
    observed_rdkit = len([c for c in (classical_df.columns if classical_df is not None else []) if c.startswith("rdkit_")])
    expected_rdkit = len(rdkit_descriptor_names()) + 1 if classical_settings["include_rdkit_2d_descriptors"] else 1
    qc_records.append({
        "feature_block": "classical_descriptor_count",
        "expected_setting": expected_rdkit,
        "observed_setting": int(observed_rdkit + (1 if classical_df is not None and "rdkit_parse_success" in classical_df.columns else 0)),
        "match_flag": bool((observed_rdkit + (1 if classical_df is not None and "rdkit_parse_success" in classical_df.columns else 0)) == expected_rdkit),
        "severity": "critical" if classical_settings["include_rdkit_2d_descriptors"] and observed_rdkit == 0 else "info",
        "notes": "RDKit descriptor block count including parse-success sentinel.",
    })
    if training_feature_columns is not None and classical_df is not None:
        missing_cols = sorted(set(training_feature_columns).difference(classical_df.columns))
        extra_cols = sorted(set(classical_df.columns).difference(training_feature_columns + [c for c in classical_df.columns if c in OPTIONAL_METADATA_COLUMNS or c in {"screening_compound_id", "standardized_smiles"}]))
        qc_records.append({
            "feature_block": "classical_training_schema_alignment",
            "expected_setting": len(training_feature_columns),
            "observed_setting": len([c for c in classical_df.columns if c in training_feature_columns]),
            "match_flag": not missing_cols,
            "severity": "critical" if missing_cols else "info",
            "notes": f"Missing columns: {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}; extra columns: {extra_cols[:10]}{'...' if len(extra_cols) > 10 else ''}",
        })
    qc_records.append({
        "feature_block": "graph_node_feature_dimension",
        "expected_setting": len(atom_feature_names(graph_settings)),
        "observed_setting": int(graph_df["node_feature_dimension"].dropna().max()) if graph_df is not None and not graph_df.empty else 0,
        "match_flag": bool(graph_df is None or graph_df.empty or int(graph_df["node_feature_dimension"].dropna().max()) == len(atom_feature_names(graph_settings))),
        "severity": "critical" if graph_df is not None and not graph_df.empty and int(graph_df["node_feature_dimension"].dropna().max()) != len(atom_feature_names(graph_settings)) else "info",
        "notes": "Graph node feature configuration consistency check.",
    })
    qc_records.append({
        "feature_block": "graph_edge_feature_dimension",
        "expected_setting": len(bond_feature_names(graph_settings)),
        "observed_setting": int(graph_df["edge_feature_dimension"].dropna().max()) if graph_df is not None and not graph_df.empty else 0,
        "match_flag": bool(graph_df is None or graph_df.empty or int(graph_df["edge_feature_dimension"].dropna().max()) == len(bond_feature_names(graph_settings))),
        "severity": "critical" if graph_df is not None and not graph_df.empty and int(graph_df["edge_feature_dimension"].dropna().max()) != len(bond_feature_names(graph_settings)) else "info",
        "notes": "Graph edge feature configuration consistency check.",
    })
    expected_env_columns = ["murcko_scaffold", "generic_murcko_scaffold", "molecular_weight", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "heavy_atom_count", "aromatic_ring_count", "formal_charge", "fraction_csp3"]
    available_env = [c for c in expected_env_columns if env_df is not None and c in env_df.columns]
    qc_records.append({
        "feature_block": "environment_feature_availability",
        "expected_setting": len(expected_env_columns),
        "observed_setting": len(available_env),
        "match_flag": len(available_env) == len(expected_env_columns),
        "severity": "critical" if len(available_env) != len(expected_env_columns) else "info",
        "notes": f"Environment column source: {env_settings.get('source', 'script_13b_config_fallback')}",
    })
    qc_df = pd.DataFrame.from_records(qc_records)
    return qc_df.sort_values(["severity", "feature_block"], kind="mergesort").reset_index(drop=True)


def enforce_qc(qc_df: pd.DataFrame) -> None:
    critical = qc_df[(qc_df["match_flag"] == False) & (qc_df["severity"].isin(DEFAULT_FAIL_SEVERITIES))]  # noqa: E712
    if not critical.empty:
        details = "; ".join(f"{row.feature_block}: expected={row.expected_setting}, observed={row.observed_setting}" for row in critical.itertuples(index=False))
        raise RuntimeError(f"Feature-space QC detected inference-blocking mismatches: {details}")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)
    logging.info("Wrote %s rows to %s", len(df), path)


def summarize_training_reference(training_df: pd.DataFrame | None, output_root: Path) -> Path | None:
    if training_df is None or training_df.empty:
        return None
    columns = [c for c in ["standardized_smiles", "pKi"] if c in training_df.columns]
    if "standardized_smiles" not in columns:
        return None
    unique_training = training_df[columns].drop_duplicates().reset_index(drop=True)
    summary = {
        "n_rows": int(len(training_df)),
        "n_unique_smiles": int(unique_training["standardized_smiles"].nunique()),
    }
    if "pKi" in unique_training.columns:
        summary["pKi_min"] = float(unique_training["pKi"].min())
        summary["pKi_max"] = float(unique_training["pKi"].max())
        summary["pKi_mean"] = float(unique_training["pKi"].mean())
    path = output_root / "training_feature_reference_summary.json"
    ensure_parent(path)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Saved training feature reference summary to %s", path)
    return path


def compute_distance_proxies(classical_df: pd.DataFrame, training_df: pd.DataFrame | None, output_root: Path) -> Path | None:
    if training_df is None or classical_df.empty or "standardized_smiles" not in training_df.columns:
        return None
    training_smiles = sorted({str(value) for value in training_df["standardized_smiles"].dropna().astype(str).tolist()})
    rows = []
    for row in classical_df[["screening_compound_id", "standardized_smiles"]].itertuples(index=False):
        rows.append({
            "screening_compound_id": row.screening_compound_id,
            "standardized_smiles": row.standardized_smiles,
            "seen_in_training_smiles": int(row.standardized_smiles in training_smiles),
        })
    path = output_root / "screening_applicability_reference_features.csv"
    save_dataframe(pd.DataFrame(rows), path)
    return path


def build_manifest(entries: list[dict[str, Any]]) -> pd.DataFrame:
    manifest = pd.DataFrame.from_records(entries)
    if manifest.empty:
        return pd.DataFrame(columns=["asset_id", "asset_type", "file_path", "row_count", "feature_block", "notes"])
    return manifest.sort_values(["feature_block", "asset_id"], kind="mergesort").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    project_root = config_path.parent.resolve()
    raw_cfg = load_yaml(config_path)
    cfg = AppConfig.from_dict(raw_cfg, project_root)
    log_path = setup_logging(cfg)
    snapshot_path = save_config_snapshot(config_path, cfg)
    warnings_list: list[str] = []

    logging.info("Loading primary screening library from %s", cfg.input_screening_library_path)
    screening_df = normalize_screening_library(load_required_dataframe(cfg.input_screening_library_path, "screening library"))
    metadata_columns = [c for c in OPTIONAL_METADATA_COLUMNS if c in screening_df.columns and screening_df[c].notna().any()]

    step07_cfg = load_optional_yaml(cfg.input_step07_config_path, "Step-07", warnings_list)
    step08_cfg = load_optional_yaml(cfg.input_step08_config_path, "Step-08", warnings_list)
    step09_cfg = load_optional_yaml(cfg.input_step09_config_path, "Step-09", warnings_list)
    training_df = load_optional_dataframe(cfg.input_training_annotated_long_path, "training annotated long", warnings_list)
    compound_env_df = load_optional_dataframe(cfg.input_compound_env_path, "compound environment annotations", warnings_list)
    kinase_env_df = load_optional_dataframe(cfg.input_kinase_env_path, "kinase environment annotations", warnings_list)

    classical_settings = resolve_classical_feature_settings(cfg, step07_cfg, warnings_list)
    graph_settings = resolve_graph_feature_settings(cfg, step08_cfg, warnings_list)
    env_settings = resolve_environment_settings(cfg, step09_cfg, warnings_list)
    logging.info("Resolved classical settings: %s", classical_settings)
    logging.info("Resolved graph settings: %s", graph_settings)
    logging.info("Resolved environment settings: %s", env_settings)

    training_schema_candidates = [
        cfg.project_root / "results/classical_baselines/compound_descriptor_features.csv",
        cfg.project_root / "results/classical_baselines/multitask_regression/model_feature_table.csv",
    ]
    training_feature_columns = None
    for candidate in training_schema_candidates:
        training_feature_columns = feature_columns_from_training_reference(candidate, ("morgan_", "rdkit_"), ("compound_id", "row_uid", "standardized_smiles"))
        if training_feature_columns:
            logging.info("Resolved training classical feature schema from %s", candidate)
            break

    manifest_entries: list[dict[str, Any]] = []
    failed_counts = {"classical": 0, "graph": 0, "environment": 0}
    classical_df = None
    graph_df = None
    env_df = None

    cfg.output_feature_root.mkdir(parents=True, exist_ok=True)
    cfg.output_report_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.generate_classical_features:
        logging.info("Generating classical screening features.")
        failed_path = cfg.output_feature_root / "failed_classical_feature_rows.csv"
        classical_df, classical_meta = generate_classical_features(
            screening_df=screening_df,
            metadata_columns=metadata_columns,
            settings=classical_settings,
            output_path=cfg.output_classical_feature_path,
            chunk_size=cfg.chunk_size,
            save_failed_rows=cfg.save_failed_rows,
            failed_rows_path=failed_path if cfg.save_failed_rows else None,
            training_feature_columns=training_feature_columns,
        )
        manifest_entries.append({"asset_id": "screening_classical_features", "asset_type": "csv", "file_path": str(cfg.output_classical_feature_path.relative_to(cfg.project_root)), "row_count": int(classical_meta["row_count"]), "feature_block": "classical", "notes": f"source={classical_settings['source']}"})
        failed_counts["classical"] = int(classical_meta["failure_count"])
        if cfg.save_failed_rows and failed_path.exists():
            manifest_entries.append({"asset_id": "failed_classical_feature_rows", "asset_type": "csv", "file_path": str(failed_path.relative_to(cfg.project_root)), "row_count": int(classical_meta["failure_count"]), "feature_block": "classical", "notes": "Rows with RDKit parsing failures during classical featurization."})

    if cfg.generate_graph_inputs:
        logging.info("Generating graph/deep-model screening manifest.")
        graph_df, graph_failures, graph_meta = generate_graph_manifest(screening_df, graph_settings, metadata_columns)
        save_dataframe(graph_df, cfg.output_graph_manifest_path)
        manifest_entries.append({"asset_id": "screening_graph_input_manifest", "asset_type": "csv", "file_path": str(cfg.output_graph_manifest_path.relative_to(cfg.project_root)), "row_count": len(graph_df), "feature_block": "graph", "notes": "Graph-ready manifest without binary serialization."})
        failed_counts["graph"] = int(len(graph_failures))
        if cfg.save_failed_rows and not graph_failures.empty:
            failed_path = cfg.output_feature_root / "failed_graph_rows.csv"
            save_dataframe(graph_failures, failed_path)
            manifest_entries.append({"asset_id": "failed_graph_rows", "asset_type": "csv", "file_path": str(failed_path.relative_to(cfg.project_root)), "row_count": len(graph_failures), "feature_block": "graph", "notes": "Rows that failed graph construction."})

    if cfg.generate_environment_features:
        logging.info("Generating environment-like screening features.")
        env_df, env_failures, env_meta = generate_environment_features(screening_df, metadata_columns, cfg.environment_features, env_settings, compound_env_df, kinase_env_df)
        save_dataframe(env_df, cfg.output_environment_feature_path)
        manifest_entries.append({"asset_id": "screening_environment_features", "asset_type": "csv", "file_path": str(cfg.output_environment_feature_path.relative_to(cfg.project_root)), "row_count": len(env_df), "feature_block": "environment", "notes": f"environment_source={env_settings.get('source', 'script_13b_config_fallback')}"})
        failed_counts["environment"] = int(len(env_failures))
        if cfg.save_failed_rows and not env_failures.empty:
            failed_path = cfg.output_feature_root / "failed_environment_feature_rows.csv"
            save_dataframe(env_failures, failed_path)
            manifest_entries.append({"asset_id": "failed_environment_feature_rows", "asset_type": "csv", "file_path": str(failed_path.relative_to(cfg.project_root)), "row_count": len(env_failures), "feature_block": "environment", "notes": "Rows that failed environment feature generation."})

    applicability_paths: dict[str, str] = {}
    if cfg.generate_applicability_reference_features and classical_df is not None:
        logging.info("Preparing applicability-reference basis assets.")
        if cfg.applicability_reference.compare_to_training_compounds:
            distance_proxy_source_df = pd.read_csv(
                cfg.output_classical_feature_path,
                usecols=["screening_compound_id", "standardized_smiles"],
            )
            distance_proxy_path = compute_distance_proxies(distance_proxy_source_df, training_df, cfg.output_feature_root)
            if distance_proxy_path is not None:
                applicability_paths["screening_applicability_reference_features"] = str(distance_proxy_path.relative_to(cfg.project_root))
                manifest_entries.append({"asset_id": "screening_applicability_reference_features", "asset_type": "csv", "file_path": str(distance_proxy_path.relative_to(cfg.project_root)), "row_count": int(classical_meta["row_count"]), "feature_block": "applicability", "notes": "Deterministic novelty proxy relative to training standardized_smiles."})
        if cfg.applicability_reference.save_training_feature_reference_summary:
            summary_path = summarize_training_reference(training_df, cfg.output_feature_root)
            if summary_path is not None:
                applicability_paths["training_feature_reference_summary"] = str(summary_path.relative_to(cfg.project_root))
                manifest_entries.append({"asset_id": "training_feature_reference_summary", "asset_type": "json", "file_path": str(summary_path.relative_to(cfg.project_root)), "row_count": 1, "feature_block": "applicability", "notes": "Basic training reference summary for later AD scoring."})

    qc_df = build_qc_summary(classical_df, graph_df, env_df, training_feature_columns, classical_settings, graph_settings, env_settings)
    save_dataframe(qc_df, cfg.output_feature_qc_summary_path)
    manifest_entries.append({"asset_id": "screening_feature_qc_summary", "asset_type": "csv", "file_path": str(cfg.output_feature_qc_summary_path.relative_to(cfg.project_root)), "row_count": len(qc_df), "feature_block": "qc", "notes": "Feature-space alignment and consistency checks."})
    enforce_qc(qc_df)

    manifest_df = build_manifest(manifest_entries)
    save_dataframe(manifest_df, cfg.output_manifest_path)

    report = {
        "script_name": SCRIPT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths_used": {
            "screening_library": str(cfg.input_screening_library_path),
            "step07_config": str(cfg.input_step07_config_path),
            "step08_config": str(cfg.input_step08_config_path),
            "step09_config": str(cfg.input_step09_config_path),
            "training_annotated_long": str(cfg.input_training_annotated_long_path),
            "compound_environment_annotations": str(cfg.input_compound_env_path),
            "kinase_environment_annotations": str(cfg.input_kinase_env_path),
        },
        "config_snapshots_referenced": {
            "step07_loaded": bool(step07_cfg is not None),
            "step08_loaded": bool(step08_cfg is not None),
            "step09_loaded": bool(step09_cfg is not None),
            "script_13b_snapshot": str(snapshot_path) if snapshot_path is not None else None,
        },
        "total_screening_compounds_processed": int(len(screening_df)),
        "total_compounds_successfully_featurized_for_classical_models": int(classical_meta["success_count"]) if classical_df is not None else 0,
        "total_compounds_successfully_prepared_for_graph_deep_models": int(int(graph_df["graph_success_flag"].sum())) if graph_df is not None else 0,
        "total_compounds_successfully_annotated_with_environment_features": int(int((env_df["environment_parse_success"] == 1).sum())) if env_df is not None else 0,
        "feature_space_consistency_summary": qc_df.to_dict(orient="records"),
        "failed_row_counts": failed_counts,
        "applicability_assets": applicability_paths,
        "warnings": warnings_list,
        "log_path": str(log_path),
        "output_manifest_path": str(cfg.output_manifest_path),
        "config_snapshot_reference": str(snapshot_path) if snapshot_path is not None else None,
    }
    ensure_parent(cfg.output_report_path)
    cfg.output_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Wrote JSON report to %s", cfg.output_report_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s", force=True)
        logging.exception("Script-13B failed: %s", exc)
        raise
