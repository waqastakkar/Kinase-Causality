#!/usr/bin/env python3
"""Package the final reproducibility and release bundle for the kinase causal-QSAR study.

This script is a strict continuation of Steps 01-11. It validates previously
produced assets, mirrors them into a deterministic release directory,
generates a machine-readable manifest and checksums, records environment and
directory-tree snapshots, writes release documentation, and optionally creates
archives. It does not retrain models or recompute benchmark outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
import sys
import tarfile
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

SCRIPT_NAME = "12_package_reproducibility_and_release"
DEFAULT_CHECKSUM = "sha256"
REQUIRED_SCRIPT_12_KEYS = {
    "project_root",
    "output_release_root",
    "output_archive_root",
    "output_manifest_path",
    "output_report_path",
    "output_release_readme_path",
    "output_runbook_path",
    "output_environment_snapshot_path",
    "output_directory_tree_path",
    "include_raw_data",
    "include_interim_data",
    "include_processed_data",
    "include_models",
    "include_results",
    "include_figures",
    "include_manuscript_outputs",
    "include_reports",
    "include_logs",
    "include_configs_used",
    "validate_required_assets",
    "generate_checksums",
    "checksum_algorithm",
    "create_tar_gz",
    "create_zip",
    "required_assets",
    "optional_environment_files",
    "save_config_snapshot",
}

CATEGORY_SPECS = (
    {
        "flag": "include_raw_data",
        "source": Path("data/raw"),
        "release": Path("data/raw"),
        "category": "data_raw",
        "required": False,
    },
    {
        "flag": "include_interim_data",
        "source": Path("data/interim"),
        "release": Path("data/interim"),
        "category": "data_interim",
        "required": False,
    },
    {
        "flag": "include_processed_data",
        "source": Path("data/processed"),
        "release": Path("data/processed"),
        "category": "data_processed",
        "required": False,
    },
    {
        "flag": "include_processed_data",
        "source": Path("data/splits"),
        "release": Path("data/splits"),
        "category": "splits",
        "required": False,
    },
    {
        "flag": "include_models",
        "source": Path("models"),
        "release": Path("models"),
        "category": "model",
        "required": False,
    },
    {
        "flag": "include_results",
        "source": Path("results"),
        "release": Path("results"),
        "category": "result",
        "required": False,
    },
    {
        "flag": "include_figures",
        "source": Path("figures"),
        "release": Path("figures"),
        "category": "figure",
        "required": False,
    },
    {
        "flag": "include_manuscript_outputs",
        "source": Path("manuscript_outputs"),
        "release": Path("manuscript_outputs"),
        "category": "manuscript_asset",
        "required": False,
    },
    {
        "flag": "include_reports",
        "source": Path("reports"),
        "release": Path("reports"),
        "category": "report",
        "required": False,
    },
    {
        "flag": "include_logs",
        "source": Path("logs"),
        "release": Path("logs"),
        "category": "log",
        "required": False,
    },
    {
        "flag": "include_configs_used",
        "source": Path("configs_used"),
        "release": Path("configs_used"),
        "category": "config_snapshot",
        "required": False,
    },
)

PACKAGE_VERSION_LOOKUP = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "scikit-learn",
    "rdkit": "rdkit",
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "matplotlib": "matplotlib",
}


@dataclass
class AppConfig:
    project_root: Path
    output_release_root: Path
    output_archive_root: Path
    output_manifest_path: Path
    output_report_path: Path
    output_release_readme_path: Path
    output_runbook_path: Path
    output_environment_snapshot_path: Path
    output_directory_tree_path: Path
    include_raw_data: bool
    include_interim_data: bool
    include_processed_data: bool
    include_models: bool
    include_results: bool
    include_figures: bool
    include_manuscript_outputs: bool
    include_reports: bool
    include_logs: bool
    include_configs_used: bool
    validate_required_assets: bool
    generate_checksums: bool
    checksum_algorithm: str
    create_tar_gz: bool
    create_zip: bool
    required_assets: list[str]
    optional_environment_files: list[str]
    save_config_snapshot: bool
    logs_dir: Path
    configs_used_dir: Path

    @staticmethod
    def from_dict(raw: dict[str, Any], repo_root: Path) -> "AppConfig":
        section = raw.get("script_12")
        if not isinstance(section, dict):
            raise ValueError("Missing required `script_12` section in config.yaml.")
        missing = sorted(REQUIRED_SCRIPT_12_KEYS.difference(section))
        if missing:
            raise ValueError("Missing required script_12 config values: " + ", ".join(missing))

        def parse_bool(value: Any, name: str) -> bool:
            if isinstance(value, bool):
                return value
            raise ValueError(f"script_12.{name} must be boolean; got {value!r}.")

        def resolve_project(path_like: Any) -> Path:
            path = Path(str(path_like))
            return path if path.is_absolute() else repo_root / path

        project_root = resolve_project(section["project_root"]).resolve()
        if not project_root.exists():
            raise FileNotFoundError(f"Configured project_root does not exist: {project_root}")

        def resolve_with_project(path_like: Any) -> Path:
            path = Path(str(path_like))
            return path if path.is_absolute() else project_root / path

        checksum_algorithm = str(section["checksum_algorithm"]).lower().strip()
        if checksum_algorithm not in hashlib.algorithms_available:
            raise ValueError(
                f"Unsupported checksum algorithm {checksum_algorithm!r}. "
                f"Available examples include: {DEFAULT_CHECKSUM}, md5, sha1."
            )

        required_assets = section["required_assets"]
        optional_environment_files = section["optional_environment_files"]
        if not isinstance(required_assets, list) or not all(isinstance(item, str) for item in required_assets):
            raise ValueError("script_12.required_assets must be a list of strings.")
        if not isinstance(optional_environment_files, list) or not all(isinstance(item, str) for item in optional_environment_files):
            raise ValueError("script_12.optional_environment_files must be a list of strings.")

        return AppConfig(
            project_root=project_root,
            output_release_root=resolve_with_project(section["output_release_root"]),
            output_archive_root=resolve_with_project(section["output_archive_root"]),
            output_manifest_path=resolve_with_project(section["output_manifest_path"]),
            output_report_path=resolve_with_project(section["output_report_path"]),
            output_release_readme_path=resolve_with_project(section["output_release_readme_path"]),
            output_runbook_path=resolve_with_project(section["output_runbook_path"]),
            output_environment_snapshot_path=resolve_with_project(section["output_environment_snapshot_path"]),
            output_directory_tree_path=resolve_with_project(section["output_directory_tree_path"]),
            include_raw_data=parse_bool(section["include_raw_data"], "include_raw_data"),
            include_interim_data=parse_bool(section["include_interim_data"], "include_interim_data"),
            include_processed_data=parse_bool(section["include_processed_data"], "include_processed_data"),
            include_models=parse_bool(section["include_models"], "include_models"),
            include_results=parse_bool(section["include_results"], "include_results"),
            include_figures=parse_bool(section["include_figures"], "include_figures"),
            include_manuscript_outputs=parse_bool(section["include_manuscript_outputs"], "include_manuscript_outputs"),
            include_reports=parse_bool(section["include_reports"], "include_reports"),
            include_logs=parse_bool(section["include_logs"], "include_logs"),
            include_configs_used=parse_bool(section["include_configs_used"], "include_configs_used"),
            validate_required_assets=parse_bool(section["validate_required_assets"], "validate_required_assets"),
            generate_checksums=parse_bool(section["generate_checksums"], "generate_checksums"),
            checksum_algorithm=checksum_algorithm,
            create_tar_gz=parse_bool(section["create_tar_gz"], "create_tar_gz"),
            create_zip=parse_bool(section["create_zip"], "create_zip"),
            required_assets=list(required_assets),
            optional_environment_files=list(optional_environment_files),
            save_config_snapshot=parse_bool(section["save_config_snapshot"], "save_config_snapshot"),
            logs_dir=project_root / "logs",
            configs_used_dir=project_root / "configs_used",
        )


@dataclass
class AssetRecord:
    asset_id: str
    relative_path: str
    absolute_source_path: str
    asset_category: str
    file_type: str
    file_size_bytes: int
    checksum: str
    originating_step: str
    required_or_optional: str
    notes: str


class RequiredAssetError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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


def validate_required_assets(cfg: AppConfig) -> list[str]:
    missing: list[str] = []
    for relative_str in sorted(cfg.required_assets):
        candidate = cfg.project_root / relative_str
        if candidate.exists():
            logging.info("Validated required asset: %s", candidate)
        else:
            missing.append(relative_str)
            logging.error("Missing required asset: %s", candidate)
    if missing and cfg.validate_required_assets:
        raise RequiredAssetError("Required packaging assets are missing: " + ", ".join(missing))
    return missing


def collect_optional_environment_files(cfg: AppConfig, warnings: list[str]) -> list[tuple[Path, Path]]:
    collected: list[tuple[Path, Path]] = []
    for relative_str in sorted(cfg.optional_environment_files):
        source = cfg.project_root / relative_str
        target = cfg.output_release_root / Path(relative_str).name
        if source.exists() and source.is_file():
            collected.append((source, target))
            logging.info("Including optional environment file %s", source)
        else:
            warning = f"Optional environment file missing: {relative_str}"
            warnings.append(warning)
            logging.warning(warning)
    return collected


def category_enabled(cfg: AppConfig, flag_name: str) -> bool:
    return bool(getattr(cfg, flag_name))


def discover_category_files(cfg: AppConfig, warnings: list[str]) -> list[tuple[Path, Path, str, str, str]]:
    discovered: list[tuple[Path, Path, str, str, str]] = []
    for spec in CATEGORY_SPECS:
        if not category_enabled(cfg, spec["flag"]):
            logging.info("Skipping category %s because %s is false.", spec["category"], spec["flag"])
            continue
        source_root = cfg.project_root / spec["source"]
        release_root = cfg.output_release_root / spec["release"]
        if not source_root.exists():
            warning = f"Requested category absent and skipped: {spec['category']} ({source_root})"
            warnings.append(warning)
            logging.warning(warning)
            continue
        if not source_root.is_dir():
            warning = f"Requested category path is not a directory and was skipped: {source_root}"
            warnings.append(warning)
            logging.warning(warning)
            continue
        files = sorted(path for path in source_root.rglob("*") if path.is_file())
        if not files:
            warning = f"Requested category contained no files: {spec['category']} ({source_root})"
            warnings.append(warning)
            logging.warning(warning)
            continue
        logging.info("Discovered %d files for category %s from %s", len(files), spec["category"], source_root)
        for file_path in files:
            rel_under_source = file_path.relative_to(source_root)
            target_path = release_root / rel_under_source
            discovered.append((file_path, target_path, spec["category"], "optional", ""))
    return discovered


def add_required_and_root_assets(cfg: AppConfig) -> list[tuple[Path, Path, str, str, str]]:
    assets: list[tuple[Path, Path, str, str, str]] = []
    readme_path = cfg.project_root / "README.md"
    if readme_path.exists():
        assets.append((readme_path, cfg.output_release_root / "README.md", "readme", "required", "Project README."))
    for relative_str in sorted(cfg.required_assets):
        source = cfg.project_root / relative_str
        if not source.exists() or source == readme_path:
            continue
        target = cfg.output_release_root / Path(relative_str)
        category = infer_category_from_path(source)
        assets.append((source, target, category, "required", "Configured required asset."))
    return assets


def compute_checksum(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def infer_originating_step(path: Path) -> str:
    text = path.as_posix().lower()
    for step in range(1, 13):
        tokens = [f"/{step:02d}_", f"_{step:02d}_", f"script_{step:02d}", f"/{step:02d}"]
        if any(token in text for token in tokens):
            return f"script_{step:02d}"
    if any(part in text for part in ["manuscript_outputs", "model_comparison"]):
        if "11_" in text or "manuscript_asset_manifest" in text:
            return "script_11"
        if "10_" in text:
            return "script_10"
    return "external_or_project_root"


def infer_category_from_path(path: Path) -> str:
    text = path.as_posix().lower()
    if "/data/raw/" in text:
        return "data_raw"
    if "/data/interim/" in text:
        return "data_interim"
    if "/data/processed/" in text:
        return "data_processed"
    if "/data/splits/" in text or text.endswith("split_manifest.csv"):
        return "splits"
    if "/models/" in text:
        return "model"
    if "/results/" in text:
        return "result"
    if "/figures/" in text:
        return "figure"
    if "/manuscript_outputs/" in text:
        return "manuscript_asset"
    if "/reports/" in text:
        return "report"
    if "/logs/" in text:
        return "log"
    if "/configs_used/" in text:
        return "config_snapshot"
    name = path.name.lower()
    if name.startswith("readme"):
        return "readme"
    if name in {"requirements.txt", "environment.yml", "pyproject.toml", "setup.cfg"}:
        return "environment_file"
    return "report"


def file_type_for(path: Path) -> str:
    suffix = "".join(path.suffixes).lower()
    return suffix.lstrip(".") or "no_extension"


def prepare_release_root(cfg: AppConfig) -> None:
    if cfg.output_release_root.exists():
        logging.info("Removing existing release root for deterministic rebuild: %s", cfg.output_release_root)
        shutil.rmtree(cfg.output_release_root)
    ensure_directory(cfg.output_release_root)
    for rel in [
        Path("data/raw"),
        Path("data/interim"),
        Path("data/processed"),
        Path("data/splits"),
        Path("models/classical_baselines"),
        Path("models/deep_baselines"),
        Path("models/causal_models"),
        Path("results/classical_baselines"),
        Path("results/deep_baselines"),
        Path("results/causal_models"),
        Path("results/model_comparison"),
        Path("figures/classical_baselines"),
        Path("figures/deep_baselines"),
        Path("figures/causal_models"),
        Path("figures/model_comparison"),
        Path("manuscript_outputs/main_figures"),
        Path("manuscript_outputs/supplementary_figures"),
        Path("manuscript_outputs/main_tables"),
        Path("manuscript_outputs/supplementary_tables"),
        Path("manuscript_outputs/figure_source_data"),
        Path("manuscript_outputs/table_source_data"),
        Path("reports"),
        Path("logs"),
        Path("configs_used"),
    ]:
        ensure_directory(cfg.output_release_root / rel)


def deduplicate_assets(candidates: Iterable[tuple[Path, Path, str, str, str]]) -> list[tuple[Path, Path, str, str, str]]:
    selected: dict[str, tuple[Path, Path, str, str, str]] = {}
    for source, target, category, req, notes in sorted(candidates, key=lambda item: (item[1].as_posix(), item[0].as_posix())):
        relative = target.as_posix()
        if relative in selected:
            existing = selected[relative]
            if existing[0] != source:
                raise RequiredAssetError(
                    f"Duplicate relative path would be created in release package: {relative} from {existing[0]} and {source}"
                )
            continue
        selected[relative] = (source, target, category, req, notes)
    return list(selected.values())


def copy_assets(cfg: AppConfig, candidates: list[tuple[Path, Path, str, str, str]]) -> list[AssetRecord]:
    records: list[AssetRecord] = []
    for index, (source, target, category, required_or_optional, notes) in enumerate(candidates, start=1):
        ensure_directory(target.parent)
        shutil.copy2(source, target)
        relative_path = target.relative_to(cfg.output_release_root).as_posix()
        checksum = compute_checksum(target, cfg.checksum_algorithm) if cfg.generate_checksums else ""
        record = AssetRecord(
            asset_id=f"asset_{index:05d}",
            relative_path=relative_path,
            absolute_source_path=str(source.resolve()),
            asset_category=category,
            file_type=file_type_for(target),
            file_size_bytes=target.stat().st_size,
            checksum=checksum,
            originating_step=infer_originating_step(source),
            required_or_optional=required_or_optional,
            notes=notes,
        )
        records.append(record)
        logging.info("Packaged %s -> %s", source, target)
    return records


def write_manifest(records: list[AssetRecord], manifest_path: Path) -> None:
    ensure_directory(manifest_path.parent)
    frame = pd.DataFrame([asdict(record) for record in records])
    frame = frame.sort_values(["relative_path", "asset_id"], kind="mergesort")
    frame.to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
    logging.info("Wrote release manifest to %s", manifest_path)


def write_checksums(records: list[AssetRecord], cfg: AppConfig) -> Path | None:
    if not cfg.generate_checksums:
        return None
    checksum_path = cfg.output_release_root / "checksums.txt"
    lines = [f"{record.checksum}  {record.relative_path}" for record in sorted(records, key=lambda item: item.relative_path)]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Wrote checksums to %s", checksum_path)
    return checksum_path


def discover_installed_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"
    except Exception as exc:  # defensive metadata lookup
        return f"unavailable ({exc})"


def build_environment_snapshot(cfg: AppConfig) -> tuple[str, list[str]]:
    lines: list[str] = []
    warnings: list[str] = []
    lines.append("# Kinase causal-QSAR release environment snapshot")
    lines.append(f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"python_executable: {sys.executable}")
    lines.append(f"python_version: {sys.version.replace(chr(10), ' ')}")
    lines.append(f"platform: {platform.platform()}")
    lines.append(f"platform_system: {platform.system()}")
    lines.append(f"platform_release: {platform.release()}")
    lines.append(f"machine: {platform.machine()}")
    lines.append(f"current_working_directory: {Path.cwd()}")
    lines.append(f"project_root: {cfg.project_root}")
    lines.append("")
    lines.append("# Dependency versions")
    for label, dist_name in PACKAGE_VERSION_LOOKUP.items():
        version = discover_installed_version(dist_name)
        if version.startswith("unavailable"):
            warnings.append(f"Could not fully resolve package version for {label}: {version}")
        lines.append(f"{label}: {version}")
    lines.append("")
    lines.append("# Environment file presence")
    for relative_str in sorted(cfg.optional_environment_files):
        exists = (cfg.project_root / relative_str).exists()
        lines.append(f"{relative_str}: {'present' if exists else 'missing'}")
    return "\n".join(lines) + "\n", warnings


def write_environment_snapshot(text: str, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    output_path.write_text(text, encoding="utf-8")
    logging.info("Wrote environment snapshot to %s", output_path)


def build_directory_tree(root: Path) -> str:
    lines = [f"{root.name}/"]

    def walk(current: Path, prefix: str) -> None:
        children = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower(), item.name))
        for index, child in enumerate(children):
            connector = "└── " if index == len(children) - 1 else "├── "
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{prefix}{connector}{child.name}{suffix}")
            if child.is_dir():
                extension = "    " if index == len(children) - 1 else "│   "
                walk(child, prefix + extension)

    walk(root, "")
    return "\n".join(lines) + "\n"


def write_release_readme(cfg: AppConfig) -> None:
    text = f"""# Kinase causal-QSAR release package

This directory is the final reproducibility and release bundle assembled by Script-12 from validated outputs produced by earlier pipeline stages. The package is intended for archival, project handoff, manuscript reproducibility support, and transparent navigation of data, models, figures, tables, and reports.

## What this release contains
- Structured copies of selected project outputs under `data/`, `models/`, `results/`, `figures/`, `manuscript_outputs/`, `reports/`, `logs/`, and `configs_used/`.
- A machine-readable asset manifest at `release_manifest.csv`.
- File checksums at `checksums.txt` when checksum generation is enabled.
- An environment snapshot at `environment_snapshot.txt`.
- A deterministic directory snapshot at `directory_tree.txt`.
- This release README and a pipeline runbook.

## Directory guide
- `data/`: raw, interim, processed, and split artifacts assembled from Steps 01-06.
- `models/`: trained baseline and causal model artifacts from Steps 07-09.
- `results/`: benchmark metrics, predictions, ablations, and integrated model-comparison outputs from Steps 07-10.
- `figures/`: publication-grade benchmarking and comparison figures from Steps 07-10.
- `manuscript_outputs/`: final manuscript figures, tables, manifests, and source-data files from Step-11.
- `reports/`: JSON and tabular pipeline reports, including comparison and manuscript reports.
- `logs/`: run logs captured across the pipeline.
- `configs_used/`: saved config snapshots for reproducibility and auditability.

## Key files
- `release_manifest.csv`: machine-readable inventory of packaged assets and provenance.
- `checksums.txt`: checksum lines for validation of packaged files.
- `reports/10_model_comparison_and_interpretation_report.json`: integrated comparison summary.
- `reports/11_manuscript_figures_and_tables_report.json`: final manuscript assembly report.
- `manuscript_outputs/manuscript_asset_manifest.csv`: manuscript figure/table asset index.

## How to inspect final manuscript assets
- Main figures: `manuscript_outputs/main_figures/`
- Supplementary figures: `manuscript_outputs/supplementary_figures/`
- Main tables: `manuscript_outputs/main_tables/`
- Supplementary tables: `manuscript_outputs/supplementary_tables/`
- Figure source data: `manuscript_outputs/figure_source_data/`
- Table source data: `manuscript_outputs/table_source_data/`

## Notes
- This package was assembled from prior outputs and does not rerun model training, splitting, or benchmark scoring.
- Source data and report files remain unchanged scientific outputs; the release package mirrors them into a handoff-ready structure.
- See `RUNBOOK.md` for the end-to-end pipeline order and validation guidance.
"""
    cfg.output_release_readme_path.write_text(text, encoding="utf-8")
    logging.info("Wrote release README to %s", cfg.output_release_readme_path)


def write_runbook(cfg: AppConfig) -> None:
    text = """# Pipeline runbook for the release package

## Expected pipeline order
1. Script-01: diagnostic extraction of human kinase bioactivity records.
2. Script-02: curation and deterministic aggregation of kinase Ki observations.
3. Script-03: panel selection and sparse matrix construction.
4. Script-04: causal environment annotation.
5. Script-05: selectivity and label task generation.
6. Script-06: benchmark split generation.
7. Script-07: classical baseline model training.
8. Script-08: graph/deep baseline model training.
9. Script-09: causal environment-aware model training.
10. Script-10: unified evaluation, robustness analysis, and interpretation.
11. Script-11: manuscript-ready figures, tables, and source-data assembly.
12. Script-12: final reproducibility, release, and archival packaging.

## Folder mapping by stage
- `data/raw`, `data/interim`, `data/processed`, `data/splits`: data preparation and split artifacts from Steps 01-06.
- `models/`: trained artifacts from Steps 07-09.
- `results/`: metrics, predictions, and comparison outputs from Steps 07-10.
- `figures/`: model-specific and comparison figures from Steps 07-10.
- `manuscript_outputs/`: manuscript-grade figures, tables, manifests, and source data from Step-11.
- `reports/`: script-level JSON and tabular reports across the pipeline.
- `configs_used/`: exact config snapshots used by pipeline stages.

## How to find key release assets
- Final manuscript assets: `manuscript_outputs/`
- Trained models: `models/`
- Benchmarking and comparison outputs: `results/`
- Manuscript and comparison reports: `reports/`
- Config snapshots: `configs_used/`

## How to validate the package
1. Review `release_manifest.csv` for the full asset inventory and provenance fields.
2. Recompute checksums for packaged files and compare against `checksums.txt`.
3. Review `environment_snapshot.txt` for the software/platform snapshot captured during packaging.
4. Use `directory_tree.txt` for a deterministic high-level package overview.

## Reproducibility notes
- Script-12 packages existing outputs only and does not retrain or rescore models.
- The package is generated deterministically by sorted file discovery, sorted manifest rows, and sorted checksum output.
- Missing optional categories are logged explicitly and reported rather than silently ignored.
- The config snapshot for Script-12 is saved under `configs_used/` when enabled.
"""
    cfg.output_runbook_path.write_text(text, encoding="utf-8")
    logging.info("Wrote runbook to %s", cfg.output_runbook_path)


def validate_packaged_outputs(cfg: AppConfig, records: list[AssetRecord], checksum_path: Path | None, warnings: list[str]) -> dict[str, Any]:
    manifest_paths = set()
    for record in records:
        packaged_path = cfg.output_release_root / record.relative_path
        if not packaged_path.exists():
            raise RequiredAssetError(f"Manifest entry points to missing packaged file: {packaged_path}")
        if record.relative_path in manifest_paths:
            raise RequiredAssetError(f"Duplicate relative path found after packaging: {record.relative_path}")
        manifest_paths.add(record.relative_path)
    if cfg.include_manuscript_outputs:
        manuscript_manifest = cfg.output_release_root / "manuscript_outputs/manuscript_asset_manifest.csv"
        if not manuscript_manifest.exists():
            raise RequiredAssetError("Manuscript outputs were included but manuscript_asset_manifest.csv is missing from release package.")
    if cfg.include_results:
        for rel in [
            Path("reports/10_model_comparison_and_interpretation_report.json"),
            Path("reports/11_manuscript_figures_and_tables_report.json"),
        ]:
            candidate = cfg.output_release_root / rel
            if not candidate.exists():
                warning = f"Key report expected for results packaging is missing from release bundle: {rel.as_posix()}"
                warnings.append(warning)
                logging.warning(warning)
    if checksum_path is not None:
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            _, relpath = line.split("  ", 1)
            if not (cfg.output_release_root / relpath).exists():
                raise RequiredAssetError(f"Checksum entry refers to missing file: {relpath}")
    return {
        "manifest_entries_validated": len(records),
        "duplicate_relative_paths": False,
        "checksum_entries_validated": 0 if checksum_path is None else len(checksum_path.read_text(encoding='utf-8').splitlines()),
    }


def create_archives(cfg: AppConfig) -> list[str]:
    archive_paths: list[str] = []
    if not cfg.create_tar_gz and not cfg.create_zip:
        logging.info("Archive creation disabled by config.")
        return archive_paths
    ensure_directory(cfg.output_archive_root)
    base_name = "kinase_causality_qsar_release"
    if cfg.create_tar_gz:
        tar_path = cfg.output_archive_root / f"{base_name}.tar.gz"
        if tar_path.exists():
            tar_path.unlink()
        with tarfile.open(tar_path, "w:gz") as archive:
            archive.add(cfg.output_release_root, arcname=cfg.output_release_root.name)
        archive_paths.append(str(tar_path))
        logging.info("Created tar.gz archive %s", tar_path)
    if cfg.create_zip:
        zip_path = cfg.output_archive_root / f"{base_name}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in sorted(cfg.output_release_root.rglob("*")):
                if file_path.is_file():
                    archive.write(file_path, arcname=file_path.relative_to(cfg.project_root))
        archive_paths.append(str(zip_path))
        logging.info("Created zip archive %s", zip_path)
    return archive_paths


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logging.info("Wrote JSON report to %s", path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    raw_config = load_yaml(repo_root / args.config if not Path(args.config).is_absolute() else Path(args.config))
    cfg = AppConfig.from_dict(raw_config, repo_root)
    log_path = configure_logging(cfg.project_root)
    logging.info("Starting %s", SCRIPT_NAME)
    logging.info("Using project root %s", cfg.project_root)

    config_snapshot_path = save_config_snapshot(raw_config, cfg)
    missing_required_assets = validate_required_assets(cfg)

    warnings: list[str] = []
    prepare_release_root(cfg)

    candidate_assets = []
    candidate_assets.extend(discover_category_files(cfg, warnings))
    candidate_assets.extend(add_required_and_root_assets(cfg))
    candidate_assets.extend(
        (source, target, "environment_file", "optional", "Optional environment/dependency definition file.")
        for source, target in collect_optional_environment_files(cfg, warnings)
    )

    unique_assets = deduplicate_assets(candidate_assets)
    records = copy_assets(cfg, unique_assets)

    manifest_path = cfg.output_manifest_path
    write_manifest(records, manifest_path)

    checksum_path = write_checksums(records, cfg)
    env_text, env_warnings = build_environment_snapshot(cfg)
    warnings.extend(env_warnings)
    write_environment_snapshot(env_text, cfg.output_environment_snapshot_path)
    tree_text = build_directory_tree(cfg.output_release_root)
    cfg.output_directory_tree_path.write_text(tree_text, encoding="utf-8")
    logging.info("Wrote directory tree snapshot to %s", cfg.output_directory_tree_path)
    write_release_readme(cfg)
    write_runbook(cfg)

    validation_summary = validate_packaged_outputs(cfg, records, checksum_path, warnings)
    archive_paths = create_archives(cfg)

    category_counts = Counter(record.asset_category for record in records)
    total_size = sum(record.file_size_bytes for record in records)
    report_payload = {
        "project_root_used": str(cfg.project_root),
        "release_root_path": str(cfg.output_release_root),
        "archive_paths": archive_paths,
        "packaging_categories_included": {
            key: getattr(cfg, key)
            for key in [
                "include_raw_data",
                "include_interim_data",
                "include_processed_data",
                "include_models",
                "include_results",
                "include_figures",
                "include_manuscript_outputs",
                "include_reports",
                "include_logs",
                "include_configs_used",
            ]
        },
        "total_packaged_file_count": len(records),
        "total_packaged_size_bytes": total_size,
        "counts_by_asset_category": dict(sorted(category_counts.items())),
        "required_assets_validated": sorted(set(cfg.required_assets).difference(missing_required_assets)),
        "missing_optional_assets": sorted(set(warnings)),
        "checksum_status": {
            "enabled": cfg.generate_checksums,
            "algorithm": cfg.checksum_algorithm,
            "path": "" if checksum_path is None else str(checksum_path),
        },
        "environment_snapshot_status": {
            "path": str(cfg.output_environment_snapshot_path),
            "written": True,
        },
        "archive_creation_status": {
            "create_tar_gz": cfg.create_tar_gz,
            "create_zip": cfg.create_zip,
            "created_archives": archive_paths,
        },
        "validation_summary": validation_summary,
        "warnings": warnings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_snapshot_reference": "" if config_snapshot_path is None else str(config_snapshot_path),
        "log_path": str(log_path),
        "manifest_path": str(manifest_path),
        "directory_tree_path": str(cfg.output_directory_tree_path),
        "release_readme_path": str(cfg.output_release_readme_path),
        "runbook_path": str(cfg.output_runbook_path),
    }
    write_json(report_payload, cfg.output_report_path)
    logging.info("Completed %s successfully with %d packaged files.", SCRIPT_NAME, len(records))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RequiredAssetError as exc:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s", force=True)
        logging.error("%s", exc)
        raise SystemExit(1)
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s", force=True)
        logging.exception("Unhandled failure in %s: %s", SCRIPT_NAME, exc)
        raise SystemExit(1)
