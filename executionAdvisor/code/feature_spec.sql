

SET memory_limit = '48GB';
SET temp_directory = '/projects/sb2ea/work/duckdb_tmp';
SET preserve_insertion_order = false;


.print '=== STAGE 0: schemas ==='
DESCRIBE SELECT * FROM '/projects/sb2ea/work/events_full.parquet' LIMIT 0;
DESCRIBE SELECT * FROM '/projects/sb2ea/csr/itemid_dict.parquet'  LIMIT 0;
DESCRIBE SELECT * FROM '/projects/sb2ea/cohort/cohort.parquet'    LIMIT 0;


CREATE OR REPLACE TABLE varmap (
    variable  VARCHAR,
    itemid    INTEGER,
    src       VARCHAR,    -- 'chart' | 'lab'
    tier      VARCHAR,
    pref      INTEGER,
    vscale    DOUBLE,
    voffset   DOUBLE,
    raw_lo    DOUBLE,     -- plausibility gate, RAW units, exclusive
    raw_hi    DOUBLE
);

INSERT INTO varmap VALUES
-- ---- core, chart (10 variables, 18 itemids) 
 ('heart_rate'   ,220045,'chart','core',0, 1.0, 0.0,   0.0,   300.0),
 ('sbp'          ,220179,'chart','core',0, 1.0, 0.0,   0.0,   400.0),
 ('sbp'          ,220050,'chart','core',0, 1.0, 0.0,   0.0,   400.0),
 ('sbp'          ,225309,'chart','core',0, 1.0, 0.0,   0.0,   400.0),
 ('dbp'          ,220180,'chart','core',0, 1.0, 0.0,   0.0,   300.0),
 ('dbp'          ,220051,'chart','core',0, 1.0, 0.0,   0.0,   300.0),
 ('dbp'          ,225310,'chart','core',0, 1.0, 0.0,   0.0,   300.0),
 ('mbp'          ,220181,'chart','core',0, 1.0, 0.0,   0.0,   300.0),
 ('mbp'          ,220052,'chart','core',0, 1.0, 0.0,   0.0,   300.0),
 ('mbp'          ,225312,'chart','core',0, 1.0, 0.0,   0.0,   300.0),
 ('resp_rate'    ,220210,'chart','core',0, 1.0, 0.0,   0.0,    70.0),
 ('resp_rate'    ,224690,'chart','core',0, 1.0, 0.0,   0.0,    70.0),
 ('spo2'         ,220277,'chart','core',0, 1.0, 0.0,   0.0,   100.001),
 
 ('temperature'  ,223762,'chart','core',0, 1.0, 0.0,  10.0,    50.0),
 ('temperature'  ,223761,'chart','core',0, 0.55555555555555556, -17.777777777777779, 70.0, 120.0),
 -- GCS subscales: eye 1-4, verbal 1-5, motor 1-6
 ('gcs_eye'      ,220739,'chart','core',0, 1.0, 0.0,   0.5,     4.5),
 ('gcs_motor'    ,223901,'chart','core',0, 1.0, 0.0,   0.5,     6.5),
 ('gcs_verbal'   ,223900,'chart','core',0, 1.0, 0.0,   0.5,     5.5),
-- ---- core, lab (6 variables) 
 ('lactate'      , 50813,'lab'  ,'core',0, 1.0, 0.0,   0.0,    50.0),
 ('bun'          , 51006,'lab'  ,'core',0, 1.0, 0.0,   0.0,   300.0),
 ('creatinine'   , 50912,'lab'  ,'core',0, 1.0, 0.0,   0.0,   150.0),
 ('wbc'          , 51301,'lab'  ,'core',0, 1.0, 0.0,   0.0,  1000.0),
 ('hemoglobin'   , 51222,'lab'  ,'core',0, 1.0, 0.0,   0.0,    30.0),  -- NOT 51221 = hematocrit
 ('platelets'    , 51265,'lab'  ,'core',0, 1.0, 0.0,   0.0, 10000.0),
-- ---- pool, ranked (fills the remaining 4 slots) 
 ('sodium'       , 50983,'lab'  ,'pool', 1, 1.0, 0.0,   0.0,   200.0),
 ('potassium'    , 50971,'lab'  ,'pool', 2, 1.0, 0.0,   0.0,    30.0),
 ('bicarbonate'  , 50882,'lab'  ,'pool', 3, 1.0, 0.0,   0.0, 10000.0),
 ('anion_gap'    , 50868,'lab'  ,'pool', 4, 1.0, 0.0,-100.0, 10000.0),  -- may be <= 0
 ('chloride'     , 50902,'lab'  ,'pool', 5, 1.0, 0.0,   0.0, 10000.0),
 ('glucose_chart',225664,'chart','pool', 6, 1.0, 0.0,   0.0,  2000.0),
 ('glucose_chart',220621,'chart','pool', 6, 1.0, 0.0,   0.0,  2000.0),
 ('glucose_chart',226537,'chart','pool', 6, 1.0, 0.0,   0.0,  2000.0),
 ('calcium'      , 50893,'lab'  ,'pool', 7, 1.0, 0.0,   0.0, 10000.0),
 ('magnesium'    , 50960,'lab'  ,'pool', 8, 1.0, 0.0,   0.0,    20.0),
 ('albumin'      , 50862,'lab'  ,'pool', 9, 1.0, 0.0,   0.0,    10.0),
 ('alt'          , 50878,'lab'  ,'pool',10, 1.0, 0.0,   0.0, 10000.0),
 ('bilirubin_tot', 50885,'lab'  ,'pool',11, 1.0, 0.0,   0.0,   100.0),
 ('inr'          , 51237,'lab'  ,'pool',12, 1.0, 0.0,   0.0,    50.0),
 ('fio2'         ,223835,'chart','pool',13, 1.0, 0.0,   0.0,   100.001);

CREATE OR REPLACE TABLE varinfo AS
SELECT variable, any_value(tier) AS tier, any_value(pref) AS pref
FROM varmap GROUP BY variable;


.print '=== STAGE 2: itemid validation ==='
CREATE OR REPLACE TABLE varmap_checked AS
SELECT
    v.*,
    d.dense_id,
    d.label      AS dict_label,
    d.param_type,
    d.category,
    CASE WHEN d.dense_id IS NULL THEN 0 ELSE 1 END AS dict_hit,
    CASE WHEN d.source = v.src  THEN 1 ELSE 0 END  AS src_agrees
FROM varmap v
LEFT JOIN '/projects/sb2ea/csr/itemid_dict.parquet' d USING (itemid);

SELECT variable, itemid, tier, dict_hit, src_agrees,
       dense_id, dict_label, param_type
FROM varmap_checked
ORDER BY tier, pref, variable, itemid;

.print '--- BLOCKING: itemids missing from the dict ---'
SELECT variable, itemid, tier, src FROM varmap_checked WHERE dict_hit = 0;

.print '--- GCS param_type check (open item in day4_decisions sec 4) ---'
SELECT variable, itemid, param_type, dict_label
FROM varmap_checked WHERE variable LIKE 'gcs%';


.print '=== STAGE 3: train coverage at W24 ==='
CREATE OR REPLACE TABLE train_stays AS
SELECT row_index FROM '/projects/sb2ea/cohort/cohort.parquet' WHERE split = 'train';

CREATE OR REPLACE TABLE n_train AS SELECT count(*) AS n FROM train_stays;

CREATE OR REPLACE TABLE per_stay_counts AS
SELECT e.row_index, m.variable, count(*) AS cnt
FROM '/projects/sb2ea/work/events_full.parquet' e
JOIN train_stays    t USING (row_index)
JOIN varmap_checked m USING (dense_id)          -- CSR stores the dense id
WHERE e.t >= 0.0 AND e.t < 24.0
  AND e.value > m.raw_lo AND e.value < m.raw_hi  -- plausibility gate
GROUP BY e.row_index, m.variable;

CREATE OR REPLACE TABLE coverage AS
SELECT
    vi.variable,
    vi.tier,
    vi.pref,
    coalesce(count(p.row_index), 0)                                   AS n_stays,
    coalesce(count(p.row_index), 0) * 1.0 / (SELECT n FROM n_train)   AS cov,
    coalesce(sum(p.cnt), 0)                                           AS n_records,
    median(p.cnt)                                                     AS median_per_stay
FROM varinfo vi
LEFT JOIN per_stay_counts p USING (variable)
GROUP BY vi.variable, vi.tier, vi.pref
ORDER BY vi.tier, cov DESC;

SELECT variable, tier, n_stays, round(cov,4) AS cov,
       n_records, median_per_stay
FROM coverage ORDER BY tier, cov DESC;

.print '--- WARN: core variables under 50% train coverage ---'
SELECT variable, round(cov,4) AS cov FROM coverage
WHERE tier = 'core' AND cov < 0.50;


CREATE OR REPLACE TABLE fill_floor AS SELECT 0.25 AS floor_cov;

.print '--- pool candidates ranked (floor = 0.25) ---'
SELECT variable, round(cov,4) AS cov, pref,
       CASE WHEN cov >= (SELECT floor_cov FROM fill_floor)
            THEN 'eligible' ELSE 'below floor' END AS status
FROM coverage WHERE tier = 'pool' ORDER BY cov DESC, pref ASC;

.print '=== STAGE 3b: the resolved 20 ==='
CREATE OR REPLACE TABLE chosen AS
SELECT variable, 'core' AS tier, cov, 0 AS ord FROM coverage WHERE tier = 'core'
UNION ALL
SELECT variable, 'fill' AS tier, cov, 1 AS ord FROM (
    SELECT variable, cov,
           row_number() OVER (ORDER BY cov DESC, pref ASC) AS rn
    FROM coverage
    WHERE tier = 'pool' AND cov >= (SELECT floor_cov FROM fill_floor)
) WHERE rn <= 4;

SELECT variable, tier, round(cov,4) AS cov
FROM chosen ORDER BY ord, variable;

.print '--- ASSERT exactly 20 variables (else F != 160, see sec 3) ---'
SELECT count(*) AS n_variables,
       CASE WHEN count(*) = 20 THEN 'OK'
            WHEN count(*) <  20 THEN 'FAIL - too few pool candidates clear the floor'
            ELSE 'FAIL - too many' END AS status,
       count(*) * 8 AS f_features
FROM chosen;

.print '--- ASSERT no chosen variable has zero coverage ---'
SELECT count(*) AS n_zero_coverage,
       CASE WHEN count(*) = 0 THEN 'OK' ELSE 'FAIL' END AS status
FROM chosen WHERE cov <= 0.0;

.print '--- ASSERT every CORE variable has data (else itemids are wrong) ---'
SELECT variable, round(cov,4) AS cov FROM chosen
WHERE tier = 'core' AND cov < 0.05;


.print '=== STAGE 4: range table ==='
CREATE OR REPLACE TABLE range_table AS
SELECT
    m.variable,
    m.itemid,
    m.dense_id,
    m.vscale,
    m.voffset,
    count(*)                                                AS n_train_records,
    quantile_cont(e.value * m.vscale + m.voffset, 0.001)    AS lo,
    quantile_cont(e.value * m.vscale + m.voffset, 0.999)    AS hi,
    min(e.value * m.vscale + m.voffset)                     AS observed_min,
    max(e.value * m.vscale + m.voffset)                     AS observed_max
FROM '/projects/sb2ea/work/events_full.parquet' e
JOIN train_stays    t USING (row_index)
JOIN varmap_checked m USING (dense_id)          -- CSR stores the dense id
WHERE m.variable IN (SELECT variable FROM chosen)
  AND e.value > m.raw_lo AND e.value < m.raw_hi  -- plausibility gate
GROUP BY m.variable, m.itemid, m.dense_id, m.vscale, m.voffset;

.print '--- eyeball these against clinical priors before freezing ---'
SELECT variable, itemid, round(lo,2) AS lo, round(hi,2) AS hi,
       round(observed_min,2) AS obs_min, round(observed_max,2) AS obs_max,
       n_train_records
FROM range_table ORDER BY variable, itemid;


.print '--- gate rejections per itemid (train, Wfull) ---'
CREATE OR REPLACE TABLE gate_rejects AS
SELECT
    m.variable,
    m.itemid,
    m.raw_lo,
    m.raw_hi,
    count(*)                                                    AS n_raw,
    count(*) FILTER (WHERE NOT (e.value > m.raw_lo
                            AND e.value < m.raw_hi))            AS n_rejected,
    round(count(*) FILTER (WHERE NOT (e.value > m.raw_lo
                            AND e.value < m.raw_hi)) * 1.0
          / count(*), 6)                                        AS frac_rejected
FROM '/projects/sb2ea/work/events_full.parquet' e
JOIN train_stays    t USING (row_index)
JOIN varmap_checked m USING (dense_id)
WHERE m.variable IN (SELECT variable FROM chosen)
GROUP BY m.variable, m.itemid, m.raw_lo, m.raw_hi
ORDER BY frac_rejected DESC;

SELECT * FROM gate_rejects;

.print '--- sanity: lo < hi, and bounds are finite ---'
SELECT variable, itemid, lo, hi FROM range_table
WHERE NOT (lo < hi) OR isnan(lo) OR isnan(hi) OR isinf(lo) OR isinf(hi);


CREATE OR REPLACE TABLE range_fallback AS
SELECT
    m.variable,
    count(*)                                             AS n_train_records,
    quantile_cont(e.value * m.vscale + m.voffset, 0.001) AS lo,
    quantile_cont(e.value * m.vscale + m.voffset, 0.999) AS hi
FROM '/projects/sb2ea/work/events_full.parquet' e
JOIN train_stays    t USING (row_index)
JOIN varmap_checked m USING (dense_id)          -- CSR stores the dense id
WHERE m.variable IN (SELECT variable FROM chosen)
  AND e.value > m.raw_lo AND e.value < m.raw_hi  
GROUP BY m.variable;

.print '--- itemids that will use the fallback (report these) ---'
SELECT m.variable, m.itemid, m.dense_id, f.lo AS fallback_lo, f.hi AS fallback_hi
FROM varmap_checked m
JOIN range_fallback f USING (variable)
LEFT JOIN range_table r ON r.itemid = m.itemid
WHERE m.variable IN (SELECT variable FROM chosen) AND r.itemid IS NULL
ORDER BY m.variable, m.itemid;

.print '=== STAGE 5: warning-flag agreement ==='
SELECT
    r.variable,
    r.itemid,
    count(*)                                                       AS n,
    count(*) FILTER (WHERE c.warning = 1)                          AS n_warned,
    count(*) FILTER (WHERE c.valuenum * r.vscale + r.voffset
                             NOT BETWEEN r.lo AND r.hi)            AS n_outside,
    count(*) FILTER (WHERE c.warning = 1
                       AND c.valuenum * r.vscale + r.voffset
                             NOT BETWEEN r.lo AND r.hi)            AS n_both,
    round(count(*) FILTER (WHERE c.warning = 1
                       AND c.valuenum * r.vscale + r.voffset
                             NOT BETWEEN r.lo AND r.hi) * 1.0
          / nullif(count(*) FILTER (WHERE c.warning = 1), 0), 4)   AS frac_warned_outside
FROM '/projects/sb2ea/parquet/chartevents.parquet' c
JOIN range_table r USING (itemid)
WHERE c.valuenum IS NOT NULL
GROUP BY r.variable, r.itemid, r.lo, r.hi, r.vscale, r.voffset
ORDER BY r.variable;


.print '=== STAGE 6: writing artifacts ==='

CREATE OR REPLACE TABLE slots AS
SELECT variable,
       CAST(row_number() OVER (ORDER BY ord, variable) - 1 AS INTEGER) AS slot
FROM chosen;

.print '--- slot assignment (this is the kernel column order) ---'
SELECT * FROM slots ORDER BY slot;

CREATE OR REPLACE TABLE kernel_lookup AS
SELECT
    d.dense_id,
    CAST(coalesce(s.slot, -1) AS SMALLINT) AS slot,
    CAST(coalesce(m.vscale,  1.0) AS FLOAT) AS vscale,
    CAST(coalesce(m.voffset, 0.0) AS FLOAT) AS voffset,
    -- own itemid bounds, else the variable's pooled fallback, else open
    CAST(coalesce(r.lo, f.lo, -1e30) AS FLOAT) AS lo,
    CAST(coalesce(r.hi, f.hi,  1e30) AS FLOAT) AS hi,
    CASE WHEN s.slot IS NULL THEN 'dropped'
         WHEN r.itemid IS NOT NULL THEN 'own'
         WHEN f.variable IS NOT NULL THEN 'fallback'
         ELSE 'UNBOUNDED' END AS bounds_src,
    d.itemid,
    d.label
FROM '/projects/sb2ea/csr/itemid_dict.parquet' d
LEFT JOIN varmap_checked m USING (itemid)
LEFT JOIN slots s          ON s.variable = m.variable
LEFT JOIN range_table r    ON r.itemid   = d.itemid
LEFT JOIN range_fallback f ON f.variable = m.variable
ORDER BY d.dense_id;

.print '--- bounds provenance for kept itemids ---'
SELECT bounds_src, count(*) AS n FROM kernel_lookup
WHERE slot >= 0 GROUP BY bounds_src;

.print '--- ASSERT: dense_id is 0..K-1 dense, no gaps, no dups ---'
SELECT count(*)                          AS k_total,
       min(dense_id)                     AS min_id,
       max(dense_id)                     AS max_id,
       count(DISTINCT dense_id)          AS n_distinct,
       CASE WHEN count(*) = max(dense_id) + 1
             AND count(*) = count(DISTINCT dense_id)
            THEN 'OK' ELSE 'FAIL' END     AS status
FROM kernel_lookup;

.print '--- ASSERT: exactly 20 distinct slots are populated ---'
SELECT count(*) FILTER (WHERE slot >= 0)            AS n_kept_itemids,
       count(DISTINCT slot) FILTER (WHERE slot >= 0) AS n_slots,
       CASE WHEN count(DISTINCT slot) FILTER (WHERE slot >= 0) = 20
            THEN 'OK' ELSE 'FAIL' END                AS status
FROM kernel_lookup;

.print '--- ASSERT: every kept itemid has finite clip bounds ---'
SELECT count(*) AS n_kept_without_bounds
FROM kernel_lookup WHERE slot >= 0 AND (lo <= -1e29 OR hi >= 1e29);

COPY gate_rejects  TO '/projects/sb2ea/manifest/gate_rejects.csv'     (HEADER, DELIMITER ',');
COPY kernel_lookup TO '/projects/sb2ea/manifest/kernel_lookup.csv'    (HEADER, DELIMITER ',');
COPY range_table   TO '/projects/sb2ea/manifest/range_table.csv'      (HEADER, DELIMITER ',');
COPY coverage      TO '/projects/sb2ea/manifest/feature_coverage.csv' (HEADER, DELIMITER ',');
COPY (SELECT c.variable, c.tier, c.cov, s.slot
      FROM chosen c JOIN slots s USING (variable) ORDER BY s.slot)
     TO '/projects/sb2ea/manifest/feature_spec.csv' (HEADER, DELIMITER ',');

.print '=== DONE ==='
.print 'Commit: feature_spec.csv, range_table.csv, feature_coverage.csv,'
.print '        kernel_lookup.csv   (all aggregate / labels only, git-safe)'
.print 'Then freeze. The kernel reads kernel_lookup.csv and nothing else.'
