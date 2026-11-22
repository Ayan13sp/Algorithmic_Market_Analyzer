"""
Drift-Detector Benchmark (Monte Carlo)
======================================
Measures each detector's false-alarm rate (no drift present) and detection
power (drift present) on synthetic data where the truth is known, comparing
the naive iid tests with the empirically calibrated rules in app.drift.

Every trial uses a 504-day reference window and a 63-day current window,
the same shapes as the live monitor.

    python -m experiments.drift_benchmark --trials 500
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from app.drift import PSI_SIGNIFICANT, DriftConfig, calibrated_psi, concept_drift, volatility_regime
from app.synthetic import ar1_series, garch_returns

REF, CUR = 504, 63
CFG = DriftConfig()


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ── Feature drift ─────────────────────────────────────────────────────────

def _feature_series(kind: str, shift: float, seed: int) -> np.ndarray:
    if kind == "iid":
        x = np.random.default_rng(seed).standard_normal(REF + CUR)
    else:
        x = ar1_series(REF + CUR, phi=0.95, seed=seed)
    x[REF:] += shift
    return x


def _feature_detectors(x: np.ndarray) -> Dict[str, bool]:
    ref, cur = x[:REF], x[REF:]
    psi, p = calibrated_psi(ref, cur, CFG.bins, CFG.stride)
    return {
        "naive KS (p < 0.05)": stats.ks_2samp(ref, cur).pvalue < 0.05,
        "naive PSI (≥ 0.25)": psi >= PSI_SIGNIFICANT,
        "calibrated PSI (≥ 0.25 and empirical p < 0.05)": psi >= PSI_SIGNIFICANT and p < CFG.p_threshold,
    }


# ── Volatility regime ─────────────────────────────────────────────────────

def _vol_detectors(multiplier: float, seed: int) -> Dict[str, bool]:
    r = garch_returns(REF + CUR, seed=seed, vol_multiplier_after=REF, vol_multiplier=multiplier)
    ref, cur = r[:REF], r[REF:]
    vol = volatility_regime(ref, cur, CFG)
    higher = cur.var() > ref.var()
    return {
        "naive Levene (p < 0.05, variance up)": higher and stats.levene(cur, ref, center="median").pvalue < 0.05,
        "calibrated (≥ 97.5th pct and ≥ 1.5× vol)": vol["regime"] == "high",
    }


# ── Concept drift ─────────────────────────────────────────────────────────

def _concept_detectors(err_ref: float, err_cur: float, seed: int) -> Dict[str, bool]:
    rng = np.random.default_rng(seed)
    errors = np.concatenate([rng.random(REF) < err_ref, rng.random(CUR) < err_cur])
    target = rng.integers(0, 2, REF + CUR)
    oos = pd.DataFrame({"target": target, "signal": np.where(errors, 1 - target, target)})
    return {"error-rate z-test (p < 0.05)": concept_drift(oos, CUR, 126, 0.05)["status"] == "degraded"}


SCENARIOS: List[Dict] = [
    # name, truth, detector function
    {"group": "Feature", "scenario": "iid N(0,1), no drift", "drift": False,
     "fn": lambda s: _feature_detectors(_feature_series("iid", 0.0, s))},
    {"group": "Feature", "scenario": "AR(1) φ=0.95 (autocorrelated), no drift", "drift": False,
     "fn": lambda s: _feature_detectors(_feature_series("ar1", 0.0, s))},
    {"group": "Feature", "scenario": "iid, mean shift +1σ", "drift": True,
     "fn": lambda s: _feature_detectors(_feature_series("iid", 1.0, s))},
    {"group": "Feature", "scenario": "AR(1) φ=0.95, level shift +2σ", "drift": True,
     "fn": lambda s: _feature_detectors(_feature_series("ar1", 2.0, s))},
    {"group": "Volatility", "scenario": "GARCH(1,1) clustering, no shift", "drift": False,
     "fn": lambda s: _vol_detectors(1.0, s)},
    {"group": "Volatility", "scenario": "GARCH, volatility × 1.5", "drift": True,
     "fn": lambda s: _vol_detectors(1.5, s)},
    {"group": "Volatility", "scenario": "GARCH, volatility × 2", "drift": True,
     "fn": lambda s: _vol_detectors(2.0, s)},
    {"group": "Concept", "scenario": "error rate 40% → 40%, no drift", "drift": False,
     "fn": lambda s: _concept_detectors(0.40, 0.40, s)},
    {"group": "Concept", "scenario": "error rate 40% → 60%", "drift": True,
     "fn": lambda s: _concept_detectors(0.40, 0.60, s)},
]


def run_benchmark(trials: int = 300, seed: int = 0) -> pd.DataFrame:
    """One row per (scenario, detector): alarm rate with a 95% Wilson interval."""
    rows = []
    for sc in SCENARIOS:
        fn: Callable[[int], Dict[str, bool]] = sc["fn"]
        hits: Dict[str, int] = {}
        for t in range(trials):
            for detector, fired in fn(seed * 100_000 + t).items():
                hits[detector] = hits.get(detector, 0) + int(fired)
        for detector, k in hits.items():
            lo, hi = wilson_interval(k, trials)
            rows.append({
                "group": sc["group"],
                "scenario": sc["scenario"],
                "truth": "drift" if sc["drift"] else "no drift",
                "metric": "detection power" if sc["drift"] else "false-alarm rate",
                "detector": detector,
                "rate": k / trials,
                "ci_low": lo,
                "ci_high": hi,
                "trials": trials,
            })
    return pd.DataFrame(rows)


def to_markdown(results: pd.DataFrame) -> str:
    lines = [
        "| Check | Scenario | Detector | Measures | Rate (95% CI) |",
        "|---|---|---|---|---|",
    ]
    for r in results.itertuples():
        lines.append(
            f"| {r.group} | {r.scenario} | {r.detector} | {r.metric} | "
            f"**{r.rate:.1%}** ({r.ci_low:.1%}–{r.ci_high:.1%}) |"
        )
    lines.append(f"\n{int(results['trials'].iloc[0])} Monte Carlo trials per scenario; "
                 f"reference = {REF} days, current window = {CUR} days.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    start = time.perf_counter()
    results = run_benchmark(args.trials, args.seed)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(results.drop(columns=["trials"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{args.trials} trials per scenario in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
