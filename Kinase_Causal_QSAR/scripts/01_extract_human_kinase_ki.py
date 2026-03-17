#!/usr/bin/env python3
"""Extract high-confidence human kinase Ki records from a local ChEMBL SQLite database.

This script is intentionally defensive because ChEMBL SQLite schemas can vary across
versions (column and table availability, especially around protein classification and
variant annotations).
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


@dataclass
class AppConfig:
    """Runtime configuration for extraction."""

    chembl_sqlite_path: Path
    output_csv_path: Path
    output_sql_path: Path
    logs_dir: Path

    @staticmethod
    def from_dict(raw: dict, project_root: Path) -> "AppConfig":
        try:
            db_path = Path(raw["chembl_sqlite_path"])
        except KeyError as exc:
            raise KeyError("Missing required config key: 'chembl_sqlite_path'") from exc

        output_csv = Path(
            raw.get("output_csv_path", "data/raw/chembl_human_kinase_ki_raw.csv")
        )
        output_sql = Path(raw.get("output_sql_path", "sql/kinase_ki_extraction.sql"))
        logs_dir = Path(raw.get("logs_dir", "logs"))

        if not db_path.is_absolute():
            db_path = project_root / db_path
        if not output_csv.is_absolute():
            output_csv = project_root / output_csv
        if not output_sql.is_absolute():
            output_sql = project_root / output_sql
        if not logs_dir.is_absolute():
            logs_dir = project_root / logs_dir

        return AppConfig(
            chembl_sqlite_path=db_path,
            output_csv_path=output_csv,
            output_sql_path=output_sql,
            logs_dir=logs_dir,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract high-confidence human single-protein kinase Ki activities from a "
            "local ChEMBL SQLite database."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file (default: config.yaml)",
    )
    return parser.parse_args()


def load_config(config_path: Path, project_root: Path) -> AppConfig:
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML mapping/object at top level.")

    return AppConfig.from_dict(raw, project_root=project_root)


def setup_logging(logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"extract_human_kinase_ki_{datetime.now():%Y%m%d_%H%M%S}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    return log_file


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1"
    cur = conn.execute(query, (table_name,))
    return cur.fetchone() is not None


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    if not table_exists(conn, table_name):
        return set()
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cur.fetchall()}


def choose_first_existing(columns: set[str], candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in columns:
            return col
    return None


def build_query(conn: sqlite3.Connection, logger: logging.Logger) -> str:
    """Build SQL dynamically to tolerate minor schema differences across ChEMBL versions."""

    # Validate presence of required core tables.
    required_tables = [
        "activities",
        "assays",
        "target_dictionary",
        "target_type",
        "compound_structures",
        "molecule_dictionary",
    ]
    missing = [table for table in required_tables if not table_exists(conn, table)]
    if missing:
        raise RuntimeError(
            "Missing required ChEMBL tables: " + ", ".join(missing)
        )

    act_cols = get_table_columns(conn, "activities")
    assay_cols = get_table_columns(conn, "assays")

    # Optional activity identifier in some ChEMBL variants.
    activity_id_expr = (
        "a.activity_id"
        if "activity_id" in act_cols
        else "CAST(NULL AS INTEGER)"
    )

    # Optional assay chembl id depending on schema variant.
    assay_chembl_id_expr = (
        "ass.chembl_id"
        if "chembl_id" in assay_cols
        else "CAST(NULL AS TEXT)"
    )

    # Optional source/document identifiers.
    select_optional = []
    joins_optional = []

    if "src_id" in assay_cols and table_exists(conn, "source"):
        source_cols = get_table_columns(conn, "source")
        joins_optional.append("LEFT JOIN source src ON ass.src_id = src.src_id")
        select_optional.extend(
            [
                "ass.src_id AS source_id",
                (
                    "src.src_description AS source_description"
                    if "src_description" in source_cols
                    else "CAST(NULL AS TEXT) AS source_description"
                ),
            ]
        )
    else:
        select_optional.extend(
            [
                "CAST(NULL AS INTEGER) AS source_id",
                "CAST(NULL AS TEXT) AS source_description",
            ]
        )

    if "doc_id" in assay_cols and table_exists(conn, "docs"):
        doc_cols = get_table_columns(conn, "docs")
        joins_optional.append("LEFT JOIN docs d ON ass.doc_id = d.doc_id")
        select_optional.extend(
            [
                "ass.doc_id AS doc_id",
                (
                    "d.doi AS doc_doi" if "doi" in doc_cols else "CAST(NULL AS TEXT) AS doc_doi"
                ),
                (
                    "d.pubmed_id AS doc_pubmed_id"
                    if "pubmed_id" in doc_cols
                    else "CAST(NULL AS TEXT) AS doc_pubmed_id"
                ),
            ]
        )
    else:
        select_optional.extend(
            [
                "CAST(NULL AS INTEGER) AS doc_id",
                "CAST(NULL AS TEXT) AS doc_doi",
                "CAST(NULL AS TEXT) AS doc_pubmed_id",
            ]
        )

    # Variant / mutant exclusion when possible.
    variant_filter_sql = ""
    if "activity_properties" in get_table_columns(conn, "activities"):
        # In some releases, this is a column in activities (rare). Keep defensive support.
        variant_filter_sql = (
            "\n  AND (a.activity_properties IS NULL OR a.activity_properties = '')"
        )
        logger.info("Applying variant filter using activities.activity_properties.")
    elif table_exists(conn, "activity_properties"):
        ap_cols = get_table_columns(conn, "activity_properties")
        # We look for common keys indicating mutation/variant and exclude those activities.
        type_col = choose_first_existing(ap_cols, ["type", "property_type"])
        value_col = choose_first_existing(ap_cols, ["value", "property_value"])
        if type_col and value_col and "activity_id" in ap_cols and "activity_id" in act_cols:
            variant_filter_sql = f"""
  AND a.activity_id NOT IN (
      SELECT ap.activity_id
      FROM activity_properties ap
      WHERE LOWER(ap.{type_col}) LIKE '%variant%'
         OR LOWER(ap.{type_col}) LIKE '%mutant%'
         OR LOWER(ap.{value_col}) LIKE '%variant%'
         OR LOWER(ap.{value_col}) LIKE '%mutant%'
  )"""
            logger.info(
                "Applying variant/mutant exclusion using activity_properties (%s, %s).",
                type_col,
                value_col,
            )
        else:
            logger.warning(
                "activity_properties table exists but expected fields not found; "
                "variant exclusion skipped."
            )
    else:
        logger.warning(
            "No variant annotation table/field detected; continuing without explicit variant exclusion."
        )

    # Kinase restriction via protein classification tables when available.
    # Common ChEMBL setup includes:
    # target_components -> component_class -> protein_classification
    # We enforce kinase via classification text matching to keep compatibility.
    kinase_filter_sql = ""
    if (
        table_exists(conn, "target_components")
        and table_exists(conn, "component_class")
        and table_exists(conn, "protein_classification")
    ):
        pc_cols = get_table_columns(conn, "protein_classification")
        kinase_text_cols: list[str] = []

        # Prefer explicit known text columns first.
        preferred_pc_cols = [
            "pref_name",
            "short_name",
            "protein_class_desc",
            "class_level1",
            "class_level2",
            "class_level3",
            "class_level4",
            "class_level5",
            "class_level6",
            "class_level7",
            "class_level8",
            "l1",
            "l2",
            "l3",
            "l4",
            "l5",
            "l6",
            "l7",
            "l8",
        ]
        for col in preferred_pc_cols:
            if col in pc_cols:
                kinase_text_cols.append(col)

        # Add any additional level-like columns to maximize cross-version compatibility.
        for col in sorted(pc_cols):
            lower = col.lower()
            if col in kinase_text_cols:
                continue
            if lower.startswith("class_level") or lower.startswith("level"):
                kinase_text_cols.append(col)

        if not kinase_text_cols:
            logger.warning(
                "protein_classification exists but no text level columns found; "
                "kinase restriction skipped."
            )
        else:
            kinase_text_expr = " || ' ' ||\n          ".join(
                f"COALESCE(pc.{col}, '')" for col in kinase_text_cols
            )
            kinase_filter_sql = """
  AND td.tid IN (
      SELECT DISTINCT tc.tid
      FROM target_components tc
      JOIN component_class cc ON tc.component_id = cc.component_id
      JOIN protein_classification pc ON cc.protein_class_id = pc.protein_class_id
      WHERE LOWER(
          {kinase_text_expr}
      ) LIKE '%kinase%'
  )""".format(kinase_text_expr=kinase_text_expr)
            logger.info(
                "Using protein classification tables to restrict targets to kinases "
                "(columns: %s).",
                ", ".join(kinase_text_cols),
            )
    else:
        logger.warning(
            "Protein classification tables not fully available; kinase restriction skipped. "
            "Adapt SQL for your ChEMBL schema if alternative classification tables exist."
        )

    select_optional_sql = ",\n    ".join(select_optional)
    joins_optional_sql = "\n".join(joins_optional)

    query = f"""
SELECT
    md.chembl_id AS compound_chembl_id,
    cs.canonical_smiles,
    td.chembl_id AS target_chembl_id,
    td.pref_name AS target_name,
    td.organism,
    {assay_chembl_id_expr} AS assay_chembl_id,
    ass.assay_type,
    ass.confidence_score,
    {activity_id_expr} AS activity_chembl_id,
    a.standard_type,
    a.standard_relation,
    a.standard_value,
    a.standard_units,
    a.pchembl_value,
    {select_optional_sql}
FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
JOIN target_type tt ON td.target_type = tt.target_type
JOIN molecule_dictionary md ON a.molregno = md.molregno
LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
{joins_optional_sql}
WHERE td.organism = 'Homo sapiens'
  AND LOWER(tt.parent_type) = 'single protein'
  AND ass.assay_type = 'B'
  AND ass.confidence_score = 9
  AND a.standard_type = 'Ki'
  AND a.standard_relation = '='
  AND a.standard_units = 'nM'
  AND a.standard_value > 0
{kinase_filter_sql}
{variant_filter_sql}
;""".strip()

    return query


def ensure_output_paths(cfg: AppConfig) -> None:
    cfg.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_sql_path.parent.mkdir(parents=True, exist_ok=True)


def run_extraction(cfg: AppConfig, logger: logging.Logger) -> pd.DataFrame:
    if not cfg.chembl_sqlite_path.exists():
        raise FileNotFoundError(
            f"ChEMBL SQLite file not found: {cfg.chembl_sqlite_path}"
        )

    logger.info("Connecting to SQLite database: %s", cfg.chembl_sqlite_path)
    conn = sqlite3.connect(cfg.chembl_sqlite_path)

    try:
        query = build_query(conn, logger)
        cfg.output_sql_path.write_text(query + "\n", encoding="utf-8")
        logger.info("Saved extraction SQL to: %s", cfg.output_sql_path)

        logger.info("Running extraction query...")
        df = pd.read_sql_query(query, conn)
        logger.info("Query complete: %d rows extracted.", len(df))
    finally:
        conn.close()

    return df


def report_summary(df: pd.DataFrame) -> None:
    total_rows = len(df)
    unique_compounds = df["compound_chembl_id"].nunique(dropna=True)
    unique_targets = df["target_chembl_id"].nunique(dropna=True)

    assay_col = "assay_chembl_id" if "assay_chembl_id" in df.columns else None
    if assay_col:
        unique_assays = df[assay_col].nunique(dropna=True)
    else:
        unique_assays = 0

    print(f"total rows extracted: {total_rows}")
    print(f"unique compounds: {unique_compounds}")
    print(f"unique targets: {unique_targets}")
    print(f"unique assays: {unique_assays}")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    args = parse_args()

    try:
        cfg = load_config(args.config, project_root=project_root)
        log_file = setup_logging(cfg.logs_dir)
        logger = logging.getLogger(__name__)

        logger.info("Log file: %s", log_file)
        ensure_output_paths(cfg)

        df = run_extraction(cfg, logger)
        df.to_csv(cfg.output_csv_path, index=False)
        logger.info("Saved extracted dataset to: %s", cfg.output_csv_path)

        report_summary(df)
        logger.info("Extraction finished successfully.")
        return 0

    except Exception as exc:  # noqa: BLE001 - explicit top-level guard for CLI robustness
        logging.getLogger(__name__).exception("Extraction failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
