-- This SQL is a reference template. The extraction script generates a schema-aware SQL
-- at runtime and overwrites this file by default.
--
-- Core filters implemented:
--   * Homo sapiens targets
--   * single protein targets
--   * binding assays (assay_type = 'B')
--   * confidence_score = 9
--   * Ki, '=' relation, nM units, positive values
--
-- Kinase restriction note:
-- If your ChEMBL version includes target_components/component_class/protein_classification,
-- the script adds a kinase filter via those tables. If schema differs, adapt those joins.

SELECT
    md.chembl_id AS compound_chembl_id,
    cs.canonical_smiles,
    td.chembl_id AS target_chembl_id,
    td.pref_name AS target_name,
    td.organism,
    ass.chembl_id AS assay_chembl_id,
    ass.assay_type,
    ass.confidence_score,
    a.activity_id AS activity_chembl_id,
    a.standard_type,
    a.standard_relation,
    a.standard_value,
    a.standard_units,
    a.pchembl_value
FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
JOIN target_type tt ON td.target_type = tt.target_type
JOIN molecule_dictionary md ON a.molregno = md.molregno
LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
WHERE td.organism = 'Homo sapiens'
  AND LOWER(tt.parent_type) = 'single protein'
  AND ass.assay_type = 'B'
  AND ass.confidence_score = 9
  AND a.standard_type = 'Ki'
  AND a.standard_relation = '='
  AND a.standard_units = 'nM'
  AND a.standard_value > 0;
