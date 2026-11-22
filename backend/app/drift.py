"""
Drift Monitoring
================
Answers one question before every prediction: does the market the model sees
today still look like the market it was trained on?

1. Feature (data) drift: PSI and Kolmogorov–Smirnov per model feature, for
   the current window (last 63 days) vs the reference window (the 504 days
   before it).
2. Volatility regime: realised volatility of the current window against the
   distribution of rolling realised volatility inside the reference period.
   An alert needs both statistical and economic significance: above the
   97.5th percentile *and* at least 1.5× the reference volatility. Volatility
   clusters, so quarterly vol leaves its two-year range far more often than
   5% of the time even when nothing structural has changed; the ratio
   condition brings the false-alarm rate back to ~6% (see the benchmark).
3. Concept drift: has the out-of-sample error rate risen recently, and has
   the up-day rate (the label prior) shifted?

Calibration matters more than the statistic. Market features are strongly
autocorrelated (a 200-day-SMA distance barely changes from one day to the
next), so a 63-day window holds far fewer than 63 independent observations.
Textbook tests (KS, Levene) assume iid samples and fire constantly on such
data. Every decision here is therefore made against an empirical null: the
same statistic computed on each historical window of the same length inside
the reference period (leave-window-out). A feature is flagged only when its
PSI is large in absolute terms (≥ 0.25, the rule of thumb from credit-risk
model monitoring) *and* unusual relative to normal quarter-to-quarter
variation (empirical p < 0.05). The naive KS p-value is still reported, for
comparison only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
TRADING_DAYS = 252


@dataclass(frozen=True)
class DriftConfig:
    reference_window: int = 504     # ~2 years: the regime the model learned from
    current_window: int = 63        # ~1 quarter: what it is seeing now
    bins: int = 10                  # PSI bins (deciles of the reference)
    stride: int = 5                 # spacing of historical windows in the empirical null
    p_threshold: float = 0.05
    dataset_drift_share: float = 0.5  # alert when ≥ half the features drift
    vol_high_pct: float = 97.5      # volatility regime thresholds (percentile) ...
    vol_low_pct: float = 2.5
    vol_alert_ratio: float = 1.5    # ... and minimum ratio to the reference volatility
    vol_warn_pct: float = 90.0      # warning band: ≥ 90th or ≤ 10th percentile
    min_reference: int = 189        # need ≥ 3 quarters of reference data
    concept_min_reference: int = 126


def _finite(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _round_or_none(value: float, digits: int) -> Optional[float]:
    return round(float(value), digits) if np.isfinite(value) else None


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def population_stability_index(reference: Any, current: Any, bins: int = 10, eps: float = 1e-4) -> float:
    """
    PSI = Σ (q − p) · ln(q / p) over bins defined by the reference deciles,
    where p and q are the reference and current shares of each bin.
    0 means identical distributions; ≥ 0.25 is conventionally "significant".
    """
    ref, cur = _finite(reference), _finite(current)
    if len(ref) < 2 or len(cur) < 1:
        raise ValueError("PSI needs at least 2 reference and 1 current observation")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    p = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
    q = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def empirical_p_value(observed: float, null: Sequence[float]) -> float:
    """P(null ≥ observed) with the +1 correction, so it is never exactly 0."""
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null >= observed - 1e-12)) / (1 + len(null)))


def calibrated_psi(reference: Any, current: Any, bins: int = 10, stride: int = 5) -> Tuple[float, float]:
    """
    PSI of ``current`` vs ``reference`` plus its empirical p-value against the
    PSI of every same-length historical window in the reference period (each
    compared with the rest of the reference, leave-window-out).
    """
    ref, cur = _finite(reference), _finite(current)
    observed = population_stability_index(ref, cur, bins)
    w = len(cur)
    null = [
        population_stability_index(np.concatenate([ref[:s], ref[s + w:]]), ref[s:s + w], bins)
        for s in range(0, len(ref) - w + 1, stride)
    ]
    return observed, empirical_p_value(observed, null) if null else float("nan")


def two_proportion_p_value(k1: int, n1: int, k0: int, n0: int, alternative: str = "greater") -> float:
    """z-test for p1 vs p0 (pooled SE). ``greater`` tests H1: p1 > p0."""
    pooled = (k1 + k0) / (n1 + n0)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k0 / n0) / se
    return float(stats.norm.sf(z) if alternative == "greater" else 2 * stats.norm.sf(abs(z)))


# ═══════════════════════════════════════════════════════════════════════════
# CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Sequence[str],
    config: Optional[DriftConfig] = None,
) -> List[Dict[str, Any]]:
    """Per-feature PSI (calibrated) and KS, sorted by PSI, largest first."""
    config = config or DriftConfig()
    rows = []
    for name in features:
        ref, cur = _finite(reference[name]), _finite(current[name])
        psi, p = calibrated_psi(ref, cur, config.bins, config.stride)
        ks = stats.ks_2samp(ref, cur)
        rows.append({
            "feature": name,
            "psi": round(psi, 4),
            "psi_p_value": round(p, 4) if np.isfinite(p) else None,
            "ks_statistic": round(float(ks.statistic), 4),
            "ks_p_value": round(float(ks.pvalue), 6),
            "drifted": bool(psi >= PSI_SIGNIFICANT and p < config.p_threshold),
        })
    return sorted(rows, key=lambda r: r["psi"], reverse=True)


def volatility_regime(
    reference_returns: Any,
    current_returns: Any,
    config: Optional[DriftConfig] = None,
) -> Dict[str, Any]:
    """
    Where does the current window's realised volatility sit within the
    distribution of rolling same-length volatility in the reference period?

    "high": ≥ 97.5th percentile and ≥ 1.5× the reference volatility (alert)
    "low":  ≤ 2.5th percentile and ≤ 1/1.5 of it (warning)
    otherwise "normal".
    """
    config = config or DriftConfig()
    ref, cur = _finite(reference_returns), _finite(current_returns)
    ann = np.sqrt(TRADING_DAYS)
    cur_vol, ref_vol = cur.std(ddof=1) * ann, ref.std(ddof=1) * ann
    rolling = pd.Series(ref).rolling(len(cur)).std(ddof=1).dropna().to_numpy() * ann
    percentile = 100 * (np.mean(rolling < cur_vol) + 0.5 * np.mean(rolling == cur_vol))
    ratio = cur_vol / ref_vol if ref_vol > 0 else 1.0
    if percentile >= config.vol_high_pct and ratio >= config.vol_alert_ratio:
        regime = "high"
    elif percentile <= config.vol_low_pct and ratio <= 1 / config.vol_alert_ratio:
        regime = "low"
    else:
        regime = "normal"
    return {
        "current_vol_pct": round(float(cur_vol) * 100, 2),
        "reference_vol_pct": round(float(ref_vol) * 100, 2),
        "ratio": round(float(cur_vol / ref_vol), 3) if ref_vol > 0 else None,
        "percentile": round(float(percentile), 1),
        "levene_p_value": _round_or_none(stats.levene(cur, ref, center="median").pvalue, 6),
        "regime": regime,
    }


def concept_drift(
    oos: pd.DataFrame,
    window: int = 63,
    min_reference: int = 126,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compare the last ``window`` out-of-sample days with all earlier OOS days.

    * Error rate: one-sided two-proportion z-test (has the model got worse?).
    * Label prior: two-sided test on the up-day rate (has the market changed?).
    Needs a model with real skill to have power; see the drift benchmark.
    """
    labelled = oos.dropna(subset=["target"])
    if len(labelled) < window + min_reference:
        return {"status": "insufficient_data", "window": window, "n_oos": int(len(labelled))}
    recent, ref = labelled.iloc[-window:], labelled.iloc[:-window]
    err1 = int((recent["signal"] != recent["target"]).sum())
    err0 = int((ref["signal"] != ref["target"]).sum())
    up1, up0 = int(recent["target"].sum()), int(ref["target"].sum())
    p_err = two_proportion_p_value(err1, len(recent), err0, len(ref), "greater")
    p_label = two_proportion_p_value(up1, len(recent), up0, len(ref), "two-sided")
    return {
        "status": "degraded" if p_err < alpha else "stable",
        "window": window,
        "n_oos": int(len(labelled)),
        "recent_error_rate": round(err1 / len(recent), 4),
        "reference_error_rate": round(err0 / len(ref), 4),
        "error_p_value": round(p_err, 6),
        "recent_up_rate": round(up1 / len(recent), 4),
        "reference_up_rate": round(up0 / len(ref), 4),
        "label_shift_p_value": round(p_label, 6),
    }


# ═══════════════════════════════════════════════════════════════════════════
# REPORTS
# ═══════════════════════════════════════════════════════════════════════════

def drift_report(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    oos: Optional[pd.DataFrame] = None,
    config: Optional[DriftConfig] = None,
) -> Dict[str, Any]:
    """
    Full drift report for the most recent ``current_window`` rows of ``df``
    (engineered features + ``close_price``; the latest unlabelled row counts).

    status: "drift" (alert) if half the features drift, volatility enters a
    high regime, or the OOS error rate rises significantly; "warning" for
    partial feature drift, a low-volatility regime, or volatility in the outer
    10% of its historical range; else "ok".
    """
    config = config or DriftConfig()
    data = df.dropna(subset=list(feature_cols))
    n, cw = len(data), config.current_window
    if n < cw + config.min_reference:
        return {
            "status": "insufficient_data",
            "alerts": [f"Drift monitoring needs at least {cw + config.min_reference} rows, got {n}."],
            "features": [], "drifted_features": [], "share_drifted": None,
            "volatility": None, "concept": None,
        }
    ref_start = max(0, n - cw - config.reference_window)
    reference, current = data.iloc[ref_start:n - cw], data.iloc[n - cw:]

    features = feature_drift(reference, current, feature_cols, config)
    drifted = [f["feature"] for f in features if f["drifted"]]
    share = len(drifted) / len(feature_cols)

    log_ret = np.log(data["close_price"]).diff()
    vol = volatility_regime(log_ret.iloc[max(1, ref_start):n - cw], log_ret.iloc[n - cw:], config)
    concept = (
        concept_drift(oos, cw, config.concept_min_reference, config.p_threshold)
        if oos is not None else None
    )

    alerts: List[str] = []
    warnings: List[str] = []
    if share >= config.dataset_drift_share:
        shown = ", ".join(drifted[:5]) + ("…" if len(drifted) > 5 else "")
        alerts.append(f"Dataset drift: {len(drifted)}/{len(feature_cols)} features have shifted ({shown}).")
    elif drifted:
        warnings.append(f"{len(drifted)} feature(s) drifted: {', '.join(drifted)}.")

    vol_text = (
        f"realised volatility {vol['current_vol_pct']}% vs {vol['reference_vol_pct']}% in the "
        f"reference period ({vol['percentile']:.0f}th percentile of historical {cw}-day windows)"
    )
    if vol["regime"] == "high":
        alerts.append(f"Volatility regime shift (high): {vol_text}.")
    elif vol["regime"] == "low":
        warnings.append(f"Unusually low volatility: {vol_text}.")
    elif vol["percentile"] >= config.vol_warn_pct or vol["percentile"] <= 100 - config.vol_warn_pct:
        edge = "top" if vol["percentile"] >= 50 else "bottom"
        ratio_text = (
            f"; {vol['ratio']:.2f}× the reference level vs a {config.vol_alert_ratio}× alert threshold"
            if vol["ratio"] else ""
        )
        warnings.append(f"Volatility at the {edge} of its historical range: {vol_text}{ratio_text}.")

    if concept and concept["status"] == "degraded":
        alerts.append(
            f"Concept drift: out-of-sample error rate rose to {concept['recent_error_rate']:.1%} over the "
            f"last {cw} days from {concept['reference_error_rate']:.1%} (p = {concept['error_p_value']:.3f})."
        )

    return {
        "status": "drift" if alerts else "warning" if warnings else "ok",
        "alerts": alerts + warnings,
        "reference_start": reference.index[0].strftime("%Y-%m-%d"),
        "reference_end": reference.index[-1].strftime("%Y-%m-%d"),
        "current_start": current.index[0].strftime("%Y-%m-%d"),
        "current_end": current.index[-1].strftime("%Y-%m-%d"),
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "share_drifted": round(share, 4),
        "drifted_features": drifted,
        "features": features,
        "volatility": vol,
        "concept": concept,
        "thresholds": {
            "psi": PSI_SIGNIFICANT,
            "p_value": config.p_threshold,
            "dataset_drift_share": config.dataset_drift_share,
            "volatility_percentiles": [config.vol_low_pct, config.vol_high_pct],
            "volatility_ratio": config.vol_alert_ratio,
        },
        "config": asdict(config),
    }


def fold_drift(
    df: pd.DataFrame,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    feature_cols: Sequence[str],
    config: Optional[DriftConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Drift of each walk-forward test block against its own training window,
    used to ask whether drifted folds are the ones where the model struggles.
    ``df`` must be the labelled frame the folds index into.
    """
    config = config or DriftConfig()
    log_ret = np.log(df["close_price"]).diff()
    rows = []
    for k, (tr, te) in enumerate(folds):
        ref_idx = tr[-config.reference_window:]
        feats = feature_drift(df.iloc[ref_idx], df.iloc[te], feature_cols, config)
        vol = volatility_regime(log_ret.iloc[ref_idx], log_ret.iloc[te], config)
        rows.append({
            "fold": k + 1,
            "share_drifted": round(float(np.mean([f["drifted"] for f in feats])), 4),
            "mean_psi": round(float(np.mean([f["psi"] for f in feats])), 4),
            "vol_percentile": vol["percentile"],
            "vol_ratio": vol["ratio"],
            "vol_regime": vol["regime"],
        })
    return rows
