#!/usr/bin/env python3
"""Extract high-confidence human kinase Ki records from a local ChEMBL SQLite database.

This script is intentionally defensive because ChEMBL SQLite schemas can vary across
versions (column and table availability, especially around protein classification and
variant annotations).
"""

from __future__ import annotations

import argparse
import json
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
    diagnostics_json_path: Path
    debug_stage_dir: Path
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
        diagnostics_json = Path(
            raw.get("diagnostics_json_path", "reports/01_extraction_diagnostics.json")
        )
        debug_stage_dir = Path(raw.get("debug_stage_dir", "data/raw"))
        logs_dir = Path(raw.get("logs_dir", "logs"))

        if not db_path.is_absolute():
            db_path = project_root / db_path
        if not output_csv.is_absolute():
            output_csv = project_root / output_csv
        if not output_sql.is_absolute():
            output_sql = project_root / output_sql
        if not diagnostics_json.is_absolute():
            diagnostics_json = project_root / diagnostics_json
        if not debug_stage_dir.is_absolute():
            debug_stage_dir = project_root / debug_stage_dir
        if not logs_dir.is_absolute():
            logs_dir = project_root / logs_dir

        return AppConfig(
            chembl_sqlite_path=db_path,
            output_csv_path=output_csv,
            output_sql_path=output_sql,
            diagnostics_json_path=diagnostics_json,
            debug_stage_dir=debug_stage_dir,
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
    parser.add_argument(
        "--disable_variant_filter",
        action="store_true",
        help="Disable variant/mutant exclusion in final extraction query.",
    )
    parser.add_argument(
        "--disable_kinase_classification",
        action="store_true",
        help="Disable kinase-classification restriction in final extraction query.",
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


def get_table_info(conn: sqlite3.Connection, table_name: str) -> list[sqlite3.Row]:
    if not table_exists(conn, table_name):
        return []
    original_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(f"PRAGMA table_info({table_name})")
        return cur.fetchall()
    finally:
        conn.row_factory = original_factory


def choose_first_existing(columns: set[str], candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in columns:
            return col
    return None


def base_filter_sql() -> str:
    return """
ass.assay_type = 'B'
  AND ass.confidence_score = 9
  AND a.standard_type = 'Ki'
  AND a.standard_relation = '='
  AND a.standard_units = 'nM'
  AND a.standard_value > 0
""".strip()


def build_variant_filter(conn: sqlite3.Connection, logger: logging.Logger) -> tuple[str, dict]:
    metadata: dict[str, object] = {
        "applied": False,
        "source": None,
        "property_types_used": [],
        "warning": None,
    }
    act_cols = get_table_columns(conn, "activities")
    if "activity_properties" in act_cols:
        metadata.update({"applied": True, "source": "activities.activity_properties"})
        return (
            """
  AND (
      a.activity_properties IS NULL
      OR LOWER(a.activity_properties) NOT LIKE '%variant%'
         AND LOWER(a.activity_properties) NOT LIKE '%mutant%'
  )""",
            metadata,
        )

    if not table_exists(conn, "activity_properties"):
        metadata["warning"] = "No activity_properties table/column available."
        logger.warning("No activity_properties table/column available; variant exclusion skipped.")
        return "", metadata

    ap_cols = get_table_columns(conn, "activity_properties")
    type_col = choose_first_existing(ap_cols, ["type", "property_type"])
    value_col = choose_first_existing(ap_cols, ["value", "property_value"])
    if not type_col or "activity_id" not in ap_cols or "activity_id" not in act_cols:
        metadata["warning"] = "Missing expected activity_properties fields."
        logger.warning("activity_properties exists but missing expected fields; variant exclusion skipped.")
        return "", metadata

    type_query = f"SELECT DISTINCT {type_col} FROM activity_properties WHERE {type_col} IS NOT NULL"
    property_types = [row[0] for row in conn.execute(type_query).fetchall()]
    matched_types = [
        t for t in property_types if isinstance(t, str) and ("variant" in t.lower() or "mutant" in t.lower())
    ]
    logger.info("Distinct property types in activity_properties.%s: %s", type_col, property_types)

    if not matched_types:
        metadata["warning"] = "No clear variant/mutant property types detected; exclusion skipped."
        logger.warning(
            "No clear variant/mutant property types found in activity_properties; skipping variant exclusion."
        )
        return "", metadata

    escaped = [t.replace("'", "''") for t in matched_types]
    in_clause = ", ".join(f"'{val}'" for val in escaped)
    value_predicate = ""
    if value_col:
        value_predicate = f" OR LOWER(ap.{value_col}) LIKE '%variant%' OR LOWER(ap.{value_col}) LIKE '%mutant%'"

    metadata.update(
        {
            "applied": True,
            "source": "activity_properties",
            "property_types_used": matched_types,
        }
    )
    logger.info(
        "Applying variant exclusion via activity_properties using type values: %s",
        ", ".join(matched_types),
    )
    return (
        f"""
  AND a.activity_id NOT IN (
      SELECT ap.activity_id
      FROM activity_properties ap
      WHERE ap.{type_col} IN ({in_clause})
         {value_predicate}
  )""",
        metadata,
    )


def build_kinase_filter(conn: sqlite3.Connection, logger: logging.Logger) -> tuple[str, dict]:
    metadata: dict[str, object] = {
        "applied": False,
        "columns_used": [],
        "warning": None,
    }
    if not (
        table_exists(conn, "target_components")
        and table_exists(conn, "component_class")
        and table_exists(conn, "protein_classification")
    ):
        metadata["warning"] = "Protein classification path tables are incomplete."
        logger.warning("Protein classification tables not fully available; kinase restriction skipped.")
        return "", metadata

    pc_cols = get_table_columns(conn, "protein_classification")
    pc_info = get_table_info(conn, "protein_classification")
    text_cols = {row["name"] for row in pc_info if "CHAR" in str(row["type"]).upper() or "TEXT" in str(row["type"]).upper()}
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
    columns = [c for c in preferred_pc_cols if c in pc_cols and (c in text_cols or not text_cols)]
    if not columns:
        columns = [
            c for c in sorted(pc_cols) if ("class" in c.lower() or "level" in c.lower() or "name" in c.lower()) and (c in text_cols or not text_cols)
        ]
    if not columns:
        metadata["warning"] = "No suitable text columns found in protein_classification."
        logger.warning("No suitable protein_classification columns found for kinase matching.")
        return "", metadata

    kinase_text_expr = " || ' ' ||\n          ".join(f"COALESCE(pc.{col}, '')" for col in columns)
    metadata.update({"applied": True, "columns_used": columns})
    logger.info("Kinase classification using protein_classification columns: %s", ", ".join(columns))
    return (
        """
  AND td.tid IN (
      SELECT DISTINCT tc.tid
      FROM target_components tc
      JOIN component_class cc ON tc.component_id = cc.component_id
      JOIN protein_classification pc ON cc.protein_class_id = pc.protein_class_id
      WHERE LOWER(
          {kinase_text_expr}
      ) LIKE '%kinase%'
  )""".format(kinase_text_expr=kinase_text_expr),
        metadata,
    )


def build_query(
    conn: sqlite3.Connection,
    logger: logging.Logger,
    disable_variant_filter: bool,
    disable_kinase_classification: bool,
) -> tuple[str, dict, str, str]:
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

    variant_filter_sql, variant_meta = build_variant_filter(conn, logger)
    kinase_filter_sql, kinase_meta = build_kinase_filter(conn, logger)
    if disable_variant_filter:
        logger.warning("Variant filter disabled via --disable_variant_filter.")
        variant_filter_sql = ""
    if disable_kinase_classification:
        logger.warning("Kinase filter disabled via --disable_kinase_classification.")
        kinase_filter_sql = ""

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
  AND {base_filter_sql()}
{kinase_filter_sql}
{variant_filter_sql}
;""".strip()

    return query, {"variant": variant_meta, "kinase": kinase_meta}, kinase_filter_sql, variant_filter_sql


def ensure_output_paths(cfg: AppConfig) -> None:
    cfg.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_sql_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.diagnostics_json_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.debug_stage_dir.mkdir(parents=True, exist_ok=True)


def run_stage_diagnostics(
    conn: sqlite3.Connection,
    cfg: AppConfig,
    logger: logging.Logger,
    kinase_filter_sql: str,
    variant_filter_sql: str,
) -> dict:
    stages: dict[str, dict[str, object]] = {}
    preview_limit = 25

    stage_queries = {
        "A": f"""SELECT a.activity_id, a.assay_id, a.standard_value, a.standard_units
FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id
WHERE {base_filter_sql()}""",
        "B": f"""SELECT a.activity_id, td.tid, td.pref_name, td.organism
FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
JOIN target_type tt ON td.target_type = tt.target_type
WHERE {base_filter_sql()}
  AND LOWER(tt.parent_type) = 'single protein'""",
        "C": f"""SELECT a.activity_id, td.tid, td.pref_name, td.organism
FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
JOIN target_type tt ON td.target_type = tt.target_type
WHERE {base_filter_sql()}
  AND LOWER(tt.parent_type) = 'single protein'
  AND td.organism = 'Homo sapiens'""",
        "D": """SELECT DISTINCT tc.tid
FROM target_components tc
JOIN component_class cc ON tc.component_id = cc.component_id
JOIN protein_classification pc ON cc.protein_class_id = pc.protein_class_id
WHERE 1=1""",
        "E": f"""SELECT a.activity_id, td.tid, td.pref_name
FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
JOIN target_type tt ON td.target_type = tt.target_type
WHERE {base_filter_sql()}
  AND LOWER(tt.parent_type) = 'single protein'
  AND td.organism = 'Homo sapiens'
  {kinase_filter_sql}""",
        "F": f"""SELECT a.activity_id, td.tid, td.pref_name
FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
JOIN target_type tt ON td.target_type = tt.target_type
WHERE {base_filter_sql()}
  AND LOWER(tt.parent_type) = 'single protein'
  AND td.organism = 'Homo sapiens'
  {kinase_filter_sql}
  {variant_filter_sql}""",
    }

    # Stage D needs kinase expression; if unavailable, return 0 with warning.
    if not kinase_filter_sql:
        stage_queries["D"] = "SELECT CAST(NULL AS INTEGER) AS tid WHERE 1 = 0"
        logger.warning("Stage D skipped because kinase classification SQL is unavailable.")
    else:
        stage_queries["D"] += kinase_filter_sql.replace("AND td.tid IN", "AND tc.tid IN")

    for stage, sql in stage_queries.items():
        df = pd.read_sql_query(sql, conn)
        count = len(df)
        logger.info("Stage %s row count: %d", stage, count)
        print(f"Stage {stage} row count: {count}")
        preview_file = None
        if count > 0:
            preview_file = cfg.debug_stage_dir / f"debug_stage_{stage}.csv"
            df.head(preview_limit).to_csv(preview_file, index=False)
            logger.info("Saved Stage %s preview to %s", stage, preview_file)
        stages[stage] = {"count": count, "preview_csv": str(preview_file) if preview_file else None}

    cfg.diagnostics_json_path.write_text(json.dumps(stages, indent=2), encoding="utf-8")
    logger.info("Saved diagnostics JSON to %s", cfg.diagnostics_json_path)
    return stages


def run_extraction(
    cfg: AppConfig,
    logger: logging.Logger,
    disable_variant_filter: bool,
    disable_kinase_classification: bool,
) -> pd.DataFrame:
    if not cfg.chembl_sqlite_path.exists():
        raise FileNotFoundError(
            f"ChEMBL SQLite file not found: {cfg.chembl_sqlite_path}"
        )

    logger.info("Connecting to SQLite database: %s", cfg.chembl_sqlite_path)
    conn = sqlite3.connect(cfg.chembl_sqlite_path)

    try:
        query, filter_meta, kinase_filter_sql, variant_filter_sql = build_query(
            conn,
            logger,
            disable_variant_filter=disable_variant_filter,
            disable_kinase_classification=disable_kinase_classification,
        )
        run_stage_diagnostics(
            conn,
            cfg,
            logger,
            kinase_filter_sql=kinase_filter_sql,
            variant_filter_sql=variant_filter_sql,
        )
        logger.info("Filter metadata: %s", json.dumps(filter_meta))
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

        df = run_extraction(
            cfg,
            logger,
            disable_variant_filter=args.disable_variant_filter,
            disable_kinase_classification=args.disable_kinase_classification,
        )
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
