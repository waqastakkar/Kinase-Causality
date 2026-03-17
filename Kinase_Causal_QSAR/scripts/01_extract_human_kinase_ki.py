#!/usr/bin/env python3
"""Diagnostic extractor for human kinase bioactivity records from ChEMBL SQLite."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


@dataclass
class AppConfig:
    """Runtime configuration for extraction."""

    chembl_sqlite_path: Path
    output_sql_path: Path
    diagnostics_json_path: Path
    debug_stage_dir: Path
    kinase_targets_csv_path: Path
    output_broad_csv_path: Path
    output_strict_csv_path: Path
    output_selected_csv_path: Path
    logs_dir: Path

    @staticmethod
    def from_dict(raw: dict, project_root: Path) -> "AppConfig":
        if "chembl_sqlite_path" not in raw:
            raise KeyError("Missing required config key: 'chembl_sqlite_path'")

        db_path = Path(raw["chembl_sqlite_path"])
        if not db_path.is_absolute():
            db_path = (project_root / db_path).resolve()

        def resolve_output(path_value: str, default: str) -> Path:
            candidate = Path(raw.get(path_value, default))
            if not candidate.is_absolute():
                candidate = (project_root / candidate).resolve()
            else:
                candidate = candidate.resolve()
            if project_root.resolve() not in candidate.parents and candidate != project_root.resolve():
                raise ValueError(
                    f"Configured output path '{path_value}' must be inside project root: {project_root}"
                )
            return candidate

        return AppConfig(
            chembl_sqlite_path=db_path,
            output_sql_path=resolve_output("output_sql_path", "sql/kinase_ki_extraction.sql"),
            diagnostics_json_path=resolve_output(
                "diagnostics_json_path", "reports/01_extraction_diagnostics.json"
            ),
            debug_stage_dir=resolve_output("debug_stage_dir", "data/raw"),
            kinase_targets_csv_path=resolve_output(
                "kinase_targets_csv_path", "data/raw/debug_kinase_targets.csv"
            ),
            output_broad_csv_path=resolve_output(
                "output_broad_csv_path", "data/raw/chembl_human_kinase_broad_raw.csv"
            ),
            output_strict_csv_path=resolve_output(
                "output_strict_csv_path", "data/raw/chembl_human_kinase_strict_raw.csv"
            ),
            output_selected_csv_path=resolve_output(
                "output_csv_path", "data/raw/chembl_human_kinase_ki_raw.csv"
            ),
            logs_dir=resolve_output("logs_dir", "logs"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run staged diagnostics and extract broad/strict human kinase activity datasets "
            "from a local ChEMBL SQLite database."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["broad", "strict"],
        default="broad",
        help=(
            "Selects which extraction query is written to output_sql_path and output_csv_path. "
            "Both broad and strict CSVs are still saved whenever non-empty."
        ),
    )
    return parser.parse_args()


def load_config(config_path: Path, project_root: Path) -> AppConfig:
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML mapping/object at top level.")

    return AppConfig.from_dict(raw, project_root=project_root.resolve())


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


def choose_target_type_predicate(conn: sqlite3.Connection) -> str:
    tt_cols = {row["name"] for row in get_table_info(conn, "target_type")}
    if "target_type" in tt_cols:
        return "LOWER(tt.target_type) = 'single protein'"
    if "parent_type" in tt_cols:
        return "LOWER(tt.parent_type) = 'single protein'"
    return "LOWER(td.target_type) = 'single protein'"


def build_kinase_target_query(conn: sqlite3.Connection, logger: logging.Logger) -> tuple[str, dict]:
    metadata: dict[str, object] = {"classification_sources": [], "columns_used": {}, "warnings": []}
    subqueries: list[str] = []

    if table_exists(conn, "target_components") and table_exists(conn, "component_class") and table_exists(conn, "protein_classification"):
        pc_info = get_table_info(conn, "protein_classification")
        text_cols = [
            row["name"]
            for row in pc_info
            if "CHAR" in str(row["type"]).upper() or "TEXT" in str(row["type"]).upper()
        ]
        preferred = [
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
        cols = [c for c in preferred if c in text_cols]
        if not cols:
            cols = [c for c in text_cols if any(k in c.lower() for k in ["class", "level", "name", "desc"])]
        if cols:
            expr = " || ' ' || ".join(f"COALESCE(pc.{col}, '')" for col in cols)
            subqueries.append(
                f"""
                SELECT DISTINCT tc.tid
                FROM target_components tc
                JOIN component_class cc ON tc.component_id = cc.component_id
                JOIN protein_classification pc ON cc.protein_class_id = pc.protein_class_id
                WHERE LOWER({expr}) LIKE '%kinase%'
                """.strip()
            )
            metadata["classification_sources"].append("protein_classification")
            metadata["columns_used"]["protein_classification"] = cols
        else:
            metadata["warnings"].append("No usable text columns found in protein_classification.")

    td_info = get_table_info(conn, "target_dictionary")
    td_text_cols = [
        row["name"]
        for row in td_info
        if "CHAR" in str(row["type"]).upper() or "TEXT" in str(row["type"]).upper()
    ]
    td_candidates = [c for c in ["pref_name", "target_type", "organism"] if c in td_text_cols]
    if td_candidates:
        td_or = " OR ".join(f"LOWER(COALESCE(td.{col}, '')) LIKE '%kinase%'" for col in td_candidates)
        subqueries.append(f"SELECT DISTINCT td.tid FROM target_dictionary td WHERE {td_or}")
        metadata["classification_sources"].append("target_dictionary_text")
        metadata["columns_used"]["target_dictionary_text"] = td_candidates

    if not subqueries:
        metadata["warnings"].append("No kinase classification sources available; kinase stage joins will be empty.")
        logger.warning("No kinase classification sources available.")
        return "SELECT CAST(NULL AS INTEGER) AS tid WHERE 1 = 0", metadata

    logger.info("Kinase classification columns used: %s", json.dumps(metadata["columns_used"]))
    return "\nUNION\n".join(subqueries), metadata


def stage_filters(stage: str) -> str:
    filters = {
        "A": "ass.assay_type = 'B' AND a.standard_relation = '=' AND a.standard_units = 'nM' AND a.standard_value > 0",
        "B": "ass.assay_type = 'B' AND a.standard_relation = '=' AND a.standard_units = 'nM' AND a.standard_value > 0 AND {single_protein}",
        "C": "ass.assay_type = 'B' AND a.standard_relation = '=' AND a.standard_units = 'nM' AND a.standard_value > 0 AND {single_protein} AND td.organism = 'Homo sapiens'",
        "D": "ass.assay_type = 'B' AND a.standard_relation = '=' AND a.standard_units = 'nM' AND a.standard_value > 0 AND {single_protein} AND td.organism = 'Homo sapiens' AND ass.confidence_score IN (8, 9)",
        "E": "ass.assay_type = 'B' AND a.standard_relation = '=' AND a.standard_units = 'nM' AND a.standard_value > 0 AND {single_protein} AND td.organism = 'Homo sapiens' AND ass.confidence_score = 9",
        "F": "ass.assay_type = 'B' AND a.standard_relation = '=' AND a.standard_units = 'nM' AND a.standard_value > 0 AND {single_protein} AND td.organism = 'Homo sapiens' AND ass.confidence_score IN (8, 9) AND a.standard_type IN ('Ki', 'IC50', 'Kd')",
        "G": "ass.assay_type = 'B' AND a.standard_relation = '=' AND a.standard_units = 'nM' AND a.standard_value > 0 AND {single_protein} AND td.organism = 'Homo sapiens' AND ass.confidence_score = 9 AND a.standard_type = 'Ki'",
    }
    return filters[stage]


def run_stage_diagnostics(conn: sqlite3.Connection, cfg: AppConfig, logger: logging.Logger) -> tuple[dict, str, dict]:
    stages: dict[str, dict[str, object]] = {}
    preview_limit = 100
    single_protein = choose_target_type_predicate(conn)
    kinase_query, kinase_meta = build_kinase_target_query(conn, logger)

    common_from = """
    FROM activities a
    JOIN assays ass ON a.assay_id = ass.assay_id
    JOIN target_dictionary td ON ass.tid = td.tid
    JOIN target_type tt ON td.target_type = tt.target_type
    """

    stage_sql = {
        s: f"""
        SELECT a.activity_id, td.tid, td.chembl_id AS target_chembl_id, td.pref_name AS target_name,
               a.standard_type, a.standard_value, a.standard_units, ass.confidence_score
        {common_from}
        WHERE {stage_filters(s).format(single_protein=single_protein)}
        """
        for s in ["A", "B", "C", "D", "E", "F", "G"]
    }

    stage_sql["H"] = f"""
    SELECT DISTINCT td.tid, td.chembl_id AS target_chembl_id, td.pref_name AS target_name
    FROM target_dictionary td
    JOIN ({kinase_query}) k ON td.tid = k.tid
    ORDER BY td.chembl_id
    """

    stage_sql["I"] = f"""
    SELECT f.*
    FROM ({stage_sql['F']}) f
    JOIN ({kinase_query}) k ON f.tid = k.tid
    """

    stage_sql["J"] = f"""
    SELECT g.*
    FROM ({stage_sql['G']}) g
    JOIN ({kinase_query}) k ON g.tid = k.tid
    """

    for stage in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        df = pd.read_sql_query(stage_sql[stage], conn)
        count = len(df)
        unique_tids = int(df["tid"].nunique()) if "tid" in df.columns else 0
        logger.info("Stage %s row count: %d (unique tids=%d)", stage, count, unique_tids)
        print(f"Stage {stage} row count: {count} (unique tids={unique_tids})")

        preview_path = None
        if count > 0:
            preview_path = cfg.debug_stage_dir / f"debug_stage_{stage}.csv"
            df.head(preview_limit).to_csv(preview_path, index=False)

        if stage == "H":
            df.to_csv(cfg.kinase_targets_csv_path, index=False)
            logger.info("Saved kinase targets to %s", cfg.kinase_targets_csv_path)

        stages[stage] = {
            "count": count,
            "unique_tids": unique_tids,
            "preview_csv": str(preview_path) if preview_path else None,
        }

    stages["kinase_classification"] = kinase_meta
    cfg.diagnostics_json_path.write_text(json.dumps(stages, indent=2), encoding="utf-8")
    logger.info("Saved diagnostics JSON to %s", cfg.diagnostics_json_path)
    return stages, single_protein, kinase_meta


def build_extraction_query(single_protein_predicate: str, kinase_query: str, mode: str) -> str:
    confidence_filter = "ass.confidence_score IN (8, 9)" if mode == "broad" else "ass.confidence_score = 9"
    endpoint_filter = (
        "a.standard_type IN ('Ki', 'IC50', 'Kd')"
        if mode == "broad"
        else "a.standard_type = 'Ki'"
    )

    return f"""
SELECT
    md.chembl_id AS compound_chembl_id,
    cs.canonical_smiles,
    td.chembl_id AS target_chembl_id,
    td.pref_name AS target_name,
    td.organism,
    ass.assay_id,
    ass.chembl_id AS assay_chembl_id,
    ass.confidence_score,
    a.standard_type,
    a.standard_relation,
    a.standard_value,
    a.standard_units
FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
JOIN target_type tt ON td.target_type = tt.target_type
JOIN molecule_dictionary md ON a.molregno = md.molregno
LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
JOIN ({kinase_query}) k ON td.tid = k.tid
WHERE ass.assay_type = 'B'
  AND a.standard_relation = '='
  AND a.standard_units = 'nM'
  AND a.standard_value > 0
  AND {single_protein_predicate}
  AND td.organism = 'Homo sapiens'
  AND {confidence_filter}
  AND {endpoint_filter}
;""".strip()


def ensure_output_paths(cfg: AppConfig) -> None:
    for p in [
        cfg.output_sql_path.parent,
        cfg.diagnostics_json_path.parent,
        cfg.debug_stage_dir,
        cfg.kinase_targets_csv_path.parent,
        cfg.output_broad_csv_path.parent,
        cfg.output_strict_csv_path.parent,
        cfg.output_selected_csv_path.parent,
        cfg.logs_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def run_extraction(cfg: AppConfig, logger: logging.Logger, mode: str) -> dict[str, pd.DataFrame]:
    if not cfg.chembl_sqlite_path.exists():
        raise FileNotFoundError(f"ChEMBL SQLite file not found: {cfg.chembl_sqlite_path}")

    logger.info("Connecting to SQLite database: %s", cfg.chembl_sqlite_path)
    conn = sqlite3.connect(cfg.chembl_sqlite_path)

    try:
        _, single_protein_predicate, _ = run_stage_diagnostics(conn, cfg, logger)
        kinase_query, _ = build_kinase_target_query(conn, logger)

        broad_query = build_extraction_query(single_protein_predicate, kinase_query, mode="broad")
        strict_query = build_extraction_query(single_protein_predicate, kinase_query, mode="strict")

        selected_query = broad_query if mode == "broad" else strict_query
        cfg.output_sql_path.write_text(selected_query + "\n", encoding="utf-8")

        broad_df = pd.read_sql_query(broad_query, conn)
        strict_df = pd.read_sql_query(strict_query, conn)

    finally:
        conn.close()

    if len(broad_df) > 0:
        broad_df.to_csv(cfg.output_broad_csv_path, index=False)
        logger.info("Saved broad extraction to: %s", cfg.output_broad_csv_path)
    else:
        logger.warning("Broad extraction is empty; no broad CSV saved.")

    if len(strict_df) > 0:
        strict_df.to_csv(cfg.output_strict_csv_path, index=False)
        logger.info("Saved strict extraction to: %s", cfg.output_strict_csv_path)
    else:
        logger.warning("Strict extraction is empty; no strict CSV saved.")

    selected_df = broad_df if mode == "broad" else strict_df
    if len(selected_df) > 0:
        selected_df.to_csv(cfg.output_selected_csv_path, index=False)
        logger.info("Saved mode-selected extraction to: %s", cfg.output_selected_csv_path)
    else:
        logger.warning("Mode-selected extraction is empty; legacy output CSV not saved.")

    return {"broad": broad_df, "strict": strict_df, "selected": selected_df}


def report_summary(results: dict[str, pd.DataFrame], mode: str) -> None:
    for name in ["broad", "strict"]:
        df = results[name]
        unique_compounds = df["compound_chembl_id"].nunique(dropna=True) if len(df) else 0
        unique_targets = df["target_chembl_id"].nunique(dropna=True) if len(df) else 0
        print(
            f"{name}: rows={len(df)}, unique_compounds={unique_compounds}, "
            f"unique_targets={unique_targets}"
        )
    print(f"selected mode: {mode}")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    args = parse_args()

    try:
        cfg = load_config(args.config, project_root=project_root)
        log_file = setup_logging(cfg.logs_dir)
        logger = logging.getLogger(__name__)

        logger.info("Log file: %s", log_file)
        ensure_output_paths(cfg)

        results = run_extraction(cfg, logger, mode=args.mode)
        report_summary(results, mode=args.mode)
        logger.info("Extraction finished successfully.")
        return 0

    except Exception as exc:  # noqa: BLE001 - explicit top-level guard for CLI robustness
        logging.getLogger(__name__).exception("Extraction failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
