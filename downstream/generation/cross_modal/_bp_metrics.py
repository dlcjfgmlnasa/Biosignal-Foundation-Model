# -*- coding:utf-8 -*-
"""Absolute-mmHg blood-pressure metrics for calibrated PPG->ABP generation.

Patch B (calibrated absolute) 전용 지표 모듈. 파형(mmHg)에서 SBP/DBP/MAP 를 추출하고,
회귀 정확도(MAE/RMSE/ME±SD/Bland-Altman/slope β/decile/AAMI/BHS)와 trending
(4-quadrant concordance / polar ±30° / ΔBland-Altman / within-patient residual r)을
계산한다. numpy 전용 (scipy 의존 없음).

핵심 정직성 지표
----------------
- **slope β**: ``pred_bp = α + β·true_bp`` OLS 기울기. β≈1 = 진짜 예측,
  β≈0 = mean-reversion(환자별 절대압 못 맞추고 population 평균으로 회귀).
- **within-patient residual r**: 환자 평균을 제거한 잔차 상관. 절대 offset 성공을
  상쇄하고 "환자 내부 변동"을 얼마나 따라가는지 본다.
"""

from __future__ import annotations

import numpy as np


# ── Beat detection ───────────────────────────────────────────


def _find_peaks(x: np.ndarray, min_dist: int) -> np.ndarray:
    """Greedy amplitude-ordered local maxima with a minimum-distance constraint.

    scipy.signal.find_peaks 의 경량 대체 (numpy 전용).

    Parameters
    ----------
    x : (T,) 1D waveform.
    min_dist : 인접 peak 간 최소 샘플 간격.

    Returns
    -------
    (K,) 정렬된 peak 인덱스.
    """
    if x.size < 3:
        return np.empty(0, dtype=np.int64)
    # 내부 국소 최대 후보
    cand = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    if cand.size == 0:
        return cand.astype(np.int64)
    order = cand[np.argsort(-x[cand])]  # 진폭 큰 순
    taken = np.zeros(x.size, dtype=bool)
    selected: list[int] = []
    for idx in order:
        lo = max(0, idx - min_dist)
        hi = min(x.size, idx + min_dist + 1)
        if not taken[lo:hi].any():
            selected.append(int(idx))
            taken[idx] = True
    return np.array(sorted(selected), dtype=np.int64)


def waveform_to_bp(win_mmHg: np.ndarray, sr: float = 100.0) -> dict | None:
    """단일 window 파형(mmHg) → {sbp, dbp, map}.

    systolic peak / diastolic trough 를 검출해 중앙값으로 SBP/DBP 를 추정하고,
    검출 실패 시 percentile(95/5) 로 폴백한다. MAP = DBP + (SBP-DBP)/3.

    Returns ``None`` if the window is too short or degenerate (flat).
    """
    win = np.asarray(win_mmHg, dtype=np.float64).ravel()
    if win.size < int(0.5 * sr) or float(np.std(win)) < 1e-6:
        return None
    min_dist = max(1, int(0.33 * sr))  # 최대 ~180 bpm
    pk = _find_peaks(win, min_dist)
    tr = _find_peaks(-win, min_dist)

    sbp = float(np.median(win[pk])) if pk.size >= 2 else float(np.percentile(win, 95))
    dbp = float(np.median(win[tr])) if tr.size >= 2 else float(np.percentile(win, 5))
    if not (sbp > dbp):  # 붕괴 시 percentile 폴백
        sbp = float(np.percentile(win, 95))
        dbp = float(np.percentile(win, 5))
    map_ = dbp + (sbp - dbp) / 3.0
    return {"sbp": sbp, "dbp": dbp, "map": map_}


# ── Absolute regression metrics ──────────────────────────────


def _bhs_grade(p5: float, p10: float, p15: float) -> str:
    """British Hypertension Society grade from cumulative %-within thresholds."""
    if p5 >= 60 and p10 >= 85 and p15 >= 95:
        return "A"
    if p5 >= 50 and p10 >= 75 and p15 >= 90:
        return "B"
    if p5 >= 40 and p10 >= 65 and p15 >= 85:
        return "C"
    return "D"


def _decile_stratified(true: np.ndarray, diff: np.ndarray, n_bins: int = 10) -> list[dict]:
    """True-BP decile 별 MAE / bias. (저·고혈압역 붕괴 노출용.)"""
    if true.size < n_bins:
        return []
    edges = np.quantile(true, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf
    out: list[dict] = []
    for b in range(n_bins):
        m = (true >= edges[b]) & (true < edges[b + 1])
        if not m.any():
            continue
        out.append({
            "decile": b,
            "true_lo": float(edges[b]),
            "n": int(m.sum()),
            "mae": float(np.abs(diff[m]).mean()),
            "bias": float(diff[m].mean()),
        })
    return out


def _component_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """SBP/DBP/MAP 한 성분에 대한 절대 회귀 지표 전체."""
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    d = pred - true
    n = d.size
    me = float(d.mean()) if n else 0.0
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    mae = float(np.abs(d).mean()) if n else 0.0
    rmse = float(np.sqrt((d ** 2).mean())) if n else 0.0

    if n > 1 and float(np.std(true)) > 1e-6:
        beta, alpha = np.polyfit(true, pred, 1)
        beta, alpha = float(beta), float(alpha)
    else:
        beta, alpha = 0.0, (float(pred.mean()) if n else 0.0)

    ae = np.abs(d)
    p5 = float((ae <= 5).mean() * 100) if n else 0.0
    p10 = float((ae <= 10).mean() * 100) if n else 0.0
    p15 = float((ae <= 15).mean() * 100) if n else 0.0

    return {
        "n": int(n),
        "mae": mae,
        "rmse": rmse,
        "me": me,           # bias
        "sd": sd,           # precision
        "loa_lower": me - 1.96 * sd,
        "loa_upper": me + 1.96 * sd,
        "slope_beta": beta,     # ≈1 진짜 / ≈0 mean-reversion
        "intercept": alpha,
        "aami_pass": bool(abs(me) <= 5.0 and sd <= 8.0),
        "bhs_pct_5": p5,
        "bhs_pct_10": p10,
        "bhs_pct_15": p15,
        "bhs_grade": _bhs_grade(p5, p10, p15),
        "deciles": _decile_stratified(true, d),
    }


def absolute_bp_metrics(
    pred_bp: dict[str, np.ndarray],
    true_bp: dict[str, np.ndarray],
    patient_id: np.ndarray | None = None,
) -> dict:
    """SBP/DBP/MAP 각각의 절대 회귀 지표.

    Parameters
    ----------
    pred_bp, true_bp : {"sbp": (N,), "dbp": (N,), "map": (N,)}.
    patient_id : (N,) optional (현재 미사용, 시그니처 호환).

    Returns
    -------
    {"sbp": {...}, "dbp": {...}, "map": {...}}
    """
    out: dict = {}
    for comp in ("sbp", "dbp", "map"):
        if comp in pred_bp and comp in true_bp and len(pred_bp[comp]):
            out[comp] = _component_metrics(pred_bp[comp], true_bp[comp])
    return out


# ── Trending metrics ─────────────────────────────────────────


def _within_patient_deltas(
    pred: np.ndarray, true: np.ndarray, patient_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """환자별 연속 window 차분 (temporal 순서 = 입력 순서 가정)."""
    dp_all: list[np.ndarray] = []
    dt_all: list[np.ndarray] = []
    for pid in np.unique(patient_id):
        m = patient_id == pid
        if m.sum() < 2:
            continue
        dp_all.append(np.diff(pred[m]))
        dt_all.append(np.diff(true[m]))
    if not dp_all:
        return np.empty(0), np.empty(0)
    return np.concatenate(dp_all), np.concatenate(dt_all)


def _within_patient_residual_r(
    pred: np.ndarray, true: np.ndarray, patient_id: np.ndarray,
) -> float:
    """환자 평균 제거 후 잔차 Pearson r (절대 offset 성공 상쇄)."""
    pr = pred.astype(float).copy()
    tr = true.astype(float).copy()
    for pid in np.unique(patient_id):
        m = patient_id == pid
        pr[m] -= pr[m].mean()
        tr[m] -= tr[m].mean()
    if pr.std() < 1e-8 or tr.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(pr, tr)[0, 1])


def _four_quadrant(dp: np.ndarray, dt: np.ndarray, exclusion: float = 0.15) -> dict:
    """4-quadrant concordance (Critchley) + central-zone exclusion."""
    if dp.size == 0:
        return {"concordance": 0.0, "n_used": 0, "exclusion_zone": 0.0}
    zone = exclusion * float(np.max(np.abs(dt))) if dt.size else 0.0
    keep = np.abs(dt) >= zone
    if keep.sum() == 0:
        return {"concordance": 0.0, "n_used": 0, "exclusion_zone": zone}
    same = np.sign(dp[keep]) == np.sign(dt[keep])
    return {
        "concordance": float(same.mean()),
        "n_used": int(keep.sum()),
        "exclusion_zone": zone,
    }


def _polar_concordance(dp: np.ndarray, dt: np.ndarray, radius_excl: float = 0.10,
                       angle_deg: float = 30.0) -> dict:
    """Polar-plot concordance: identity 선(±angle) 내 비율. 소반경 제외."""
    if dp.size == 0:
        return {"polar_concordance": 0.0, "n_used": 0}
    radius = np.sqrt(dp ** 2 + dt ** 2)
    if radius.max() <= 0:
        return {"polar_concordance": 0.0, "n_used": 0}
    keep = radius >= radius_excl * float(radius.max())
    if keep.sum() == 0:
        return {"polar_concordance": 0.0, "n_used": 0}
    # identity 선(Δpred=Δtrue)에서의 각도 편차.
    ang = np.degrees(np.arctan2(dp[keep], dt[keep])) - 45.0
    ang = (ang + 180.0) % 360.0 - 180.0  # wrap [-180,180]
    within = np.abs(ang) <= angle_deg
    return {"polar_concordance": float(within.mean()), "n_used": int(keep.sum())}


def trending_metrics(
    pred_bp: dict[str, np.ndarray],
    true_bp: dict[str, np.ndarray],
    patient_id: np.ndarray,
    anchor_bp: dict[str, np.ndarray] | None = None,
) -> dict:
    """SBP/DBP/MAP trending: 4Q concordance / polar / ΔBA / within-patient r.

    ``anchor_bp`` 는 현재 within-patient 연속 차분 방식에서는 미사용
    (시그니처 호환). 절대 offset 무관한 변동 추종 능력 평가가 목적.
    """
    out: dict = {}
    for comp in ("sbp", "dbp", "map"):
        if comp not in pred_bp or comp not in true_bp or not len(pred_bp[comp]):
            continue
        p = np.asarray(pred_bp[comp], float)
        t = np.asarray(true_bp[comp], float)
        dp, dt = _within_patient_deltas(p, t, patient_id)
        # Δ Bland-Altman
        dd = dp - dt
        ba_bias = float(dd.mean()) if dd.size else 0.0
        ba_sd = float(dd.std(ddof=1)) if dd.size > 1 else 0.0
        out[comp] = {
            "within_patient_residual_r": _within_patient_residual_r(p, t, patient_id),
            "four_quadrant": _four_quadrant(dp, dt),
            "polar": _polar_concordance(dp, dt),
            "delta_ba_bias": ba_bias,
            "delta_ba_loa_lower": ba_bias - 1.96 * ba_sd,
            "delta_ba_loa_upper": ba_bias + 1.96 * ba_sd,
            "n_delta": int(dp.size),
        }
    return out
