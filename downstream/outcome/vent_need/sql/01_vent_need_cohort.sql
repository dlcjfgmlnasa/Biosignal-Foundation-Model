-- Mechanical Ventilation Need Prediction — Cohort 추출 (MIMIC-III BigQuery)
-- 정의:
--   * 입원 24h 내 mechanical ventilation 필요 여부 예측
--   * Patient-level binary task
--   * Input window: ICU 입원 후 첫 N시간 waveform (예: 6h or 12h)
--   * Horizon: 24h (input window 끝나는 시점부터)
--
-- Label:
--   1 = ICU 입원 24h 내 invasive ventilation 시작
--   0 = ICU 입원 24h 내 vent 없음 (또는 그 이후)
--
-- 평가 활용:
--   pre-vent waveform 만으로 예측 → vent 필요 환자 조기 식별

-- ============================================================
-- 1. 모든 ICU stay × vent 시작 여부
-- ============================================================
CREATE OR REPLACE TABLE `vent_need.vent_need_cohort` AS
WITH first_vent AS (
  -- 각 ICU stay 의 첫 vent 시작 시점
  SELECT
    pe.icustay_id,
    MIN(pe.starttime) AS first_vent_start
  FROM `physionet-data.mimiciii_clinical.procedureevents_mv` pe
  WHERE pe.itemid = 225792  -- Invasive Ventilation
  GROUP BY pe.icustay_id
)
SELECT
  ic.subject_id,
  ic.hadm_id,
  ic.icustay_id,
  ic.first_careunit,
  ic.intime AS icu_intime,
  ic.outtime AS icu_outtime,
  fv.first_vent_start,
  -- ICU 입원 후 vent 까지 시간
  TIMESTAMP_DIFF(fv.first_vent_start, ic.intime, HOUR) AS hours_to_vent,
  -- 라벨
  CASE
    WHEN fv.first_vent_start IS NOT NULL
     AND TIMESTAMP_DIFF(fv.first_vent_start, ic.intime, HOUR) BETWEEN 0 AND 24
    THEN 1
    ELSE 0
  END AS vent_within_24h,
  -- 12h 라벨도 함께
  CASE
    WHEN fv.first_vent_start IS NOT NULL
     AND TIMESTAMP_DIFF(fv.first_vent_start, ic.intime, HOUR) BETWEEN 0 AND 12
    THEN 1
    ELSE 0
  END AS vent_within_12h,
  -- Demographic
  p.gender,
  DATE_DIFF(DATE(ic.intime), DATE(p.dob), YEAR) AS age
FROM `physionet-data.mimiciii_clinical.icustays` ic
INNER JOIN `physionet-data.mimiciii_clinical.patients` p
  ON ic.subject_id = p.subject_id
LEFT JOIN first_vent fv
  ON ic.icustay_id = fv.icustay_id
WHERE DATE_DIFF(DATE(ic.intime), DATE(p.dob), YEAR) BETWEEN 18 AND 89
  -- ICU stay 가 충분히 길어야 (input window 가용성 확보)
  AND TIMESTAMP_DIFF(ic.outtime, ic.intime, HOUR) >= 6
  -- ICU 입원 시점에 이미 vent 중인 환자 제외 (pre-existing vent)
  AND (
    fv.first_vent_start IS NULL
    OR TIMESTAMP_DIFF(fv.first_vent_start, ic.intime, MINUTE) >= 30
  )
;

-- ============================================================
-- 2. 통계 확인
-- ============================================================
SELECT
  COUNT(*) AS n_icu_stays,
  COUNT(DISTINCT subject_id) AS unique_patients,
  SUM(vent_within_24h) AS n_vent_24h,
  SUM(vent_within_12h) AS n_vent_12h,
  ROUND(100.0 * SUM(vent_within_24h) / COUNT(*), 1) AS vent_24h_pct,
  ROUND(100.0 * SUM(vent_within_12h) / COUNT(*), 1) AS vent_12h_pct
FROM `vent_need.vent_need_cohort`
;

-- ============================================================
-- 3. CSV 다운로드용
-- ============================================================
SELECT
  subject_id,
  hadm_id,
  icustay_id,
  icu_intime,
  icu_outtime,
  hours_to_vent,
  vent_within_12h,
  vent_within_24h,
  age,
  gender
FROM `vent_need.vent_need_cohort`
ORDER BY subject_id, icustay_id
;
