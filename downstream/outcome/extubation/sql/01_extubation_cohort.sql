-- Extubation Failure Prediction — Cohort 추출 (MIMIC-III BigQuery)
-- 정의:
--   * Extubation: invasive ventilation 종료 (extubation 시점 = ventilator off)
--   * Failure  : extubation 후 48~72h 이내 재삽관 (reintubation)
--
-- Label:
--   1 = extubation 후 48h 또는 72h 이내 재삽관 (failure)
--   0 = extubation 후 동일 horizon 내 재삽관 없음 (success)
--
-- 사용법: BigQuery 콘솔
--   1. Dataset "extubation" 생성 (Location: US)
--   2. 이 SQL 실행
--   3. CSV 다운로드 → cohort 파일

-- ============================================================
-- 1. Mechanical ventilation 시작/종료 시점 추출
--    chartevents: itemid 720(Vent type), 223849(Vent mode)
--    procedureevents_mv: itemid 225792 (Invasive Ventilation)
-- ============================================================
CREATE OR REPLACE TABLE `extubation.vent_episodes` AS
WITH vent_proc AS (
  -- procedureevents_mv: invasive ventilation
  SELECT
    pe.subject_id,
    pe.hadm_id,
    pe.icustay_id,
    pe.starttime AS vent_start,
    pe.endtime   AS vent_end,
    'procedure' AS source
  FROM `physionet-data.mimiciii_clinical.procedureevents_mv` pe
  WHERE pe.itemid = 225792  -- Invasive Ventilation
    AND pe.endtime > pe.starttime
)
SELECT
  subject_id,
  hadm_id,
  icustay_id,
  vent_start,
  vent_end,
  TIMESTAMP_DIFF(vent_end, vent_start, HOUR) AS vent_duration_hours,
  -- Extubation 시점 = vent_end
  vent_end AS extubation_time,
  ROW_NUMBER() OVER (
    PARTITION BY icustay_id ORDER BY vent_start
  ) AS vent_episode_idx
FROM vent_proc
WHERE TIMESTAMP_DIFF(vent_end, vent_start, HOUR) >= 12  -- 최소 12시간 vent
;

-- ============================================================
-- 2. Reintubation 검출
--    같은 ICU stay 안에서 다음 vent episode 가 시작되면 reintubation
-- ============================================================
CREATE OR REPLACE TABLE `extubation.extubation_cohort` AS
WITH next_vent AS (
  SELECT
    e1.subject_id,
    e1.hadm_id,
    e1.icustay_id,
    e1.vent_episode_idx,
    e1.vent_start,
    e1.vent_end       AS extubation_time,
    e1.vent_duration_hours,
    -- 다음 vent episode 의 시작
    e2.vent_start AS next_intubation_time,
    TIMESTAMP_DIFF(e2.vent_start, e1.vent_end, HOUR) AS hours_to_reintub
  FROM `extubation.vent_episodes` e1
  LEFT JOIN `extubation.vent_episodes` e2
    ON e1.icustay_id = e2.icustay_id
   AND e2.vent_episode_idx = e1.vent_episode_idx + 1
)
SELECT
  *,
  -- 라벨 (multi-horizon)
  CASE
    WHEN hours_to_reintub IS NOT NULL AND hours_to_reintub <= 48 THEN 1
    ELSE 0
  END AS extub_fail_48h,
  CASE
    WHEN hours_to_reintub IS NOT NULL AND hours_to_reintub <= 72 THEN 1
    ELSE 0
  END AS extub_fail_72h
FROM next_vent
;

-- ============================================================
-- 3. 통계 확인
-- ============================================================
SELECT
  COUNT(*) AS n_extubations,
  COUNT(DISTINCT subject_id) AS unique_patients,
  SUM(extub_fail_48h) AS n_failures_48h,
  SUM(extub_fail_72h) AS n_failures_72h,
  ROUND(100.0 * SUM(extub_fail_48h) / COUNT(*), 1) AS fail_rate_48h_pct,
  ROUND(100.0 * SUM(extub_fail_72h) / COUNT(*), 1) AS fail_rate_72h_pct
FROM `extubation.extubation_cohort`
;

-- ============================================================
-- 4. CSV 다운로드용
-- ============================================================
SELECT
  subject_id,
  hadm_id,
  icustay_id,
  vent_episode_idx,
  vent_start,
  extubation_time,
  vent_duration_hours,
  hours_to_reintub,
  extub_fail_48h,
  extub_fail_72h
FROM `extubation.extubation_cohort`
ORDER BY subject_id, icustay_id, vent_episode_idx
;
