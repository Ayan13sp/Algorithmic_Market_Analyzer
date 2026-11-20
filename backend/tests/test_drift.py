"""Tests for drift monitoring (app.drift): the statistics, their calibration on
autocorrelated data, the volatility regime rule, concept drift and the report."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.drift import (
    DriftConfig,
    calibrated_psi,
    concept_drift,
    drift_report,
    empirical_p_value,
    feature_drift,
    fold_drift,
    population_stability_index,
    two_proportion_p_value,
    volatility_regime,
)
from app.ml_engine import FEATURE_COLS, engineer_features, labelled_rows
from app.synthetic import ar1_series, garch_returns, random_walk_ohlcv
from app.walk_forward import WalkForwardConfig, walk_forward_splits

RNG = np.random.default_rng(0)


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

class TestPSI:
    def test_identical_distributions_score_zero(self):
        x = RNG.standard_normal(5000)
        assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-9)

    def test_matches_hand_computation(self):
        reference = np.arange(1000.0)            # deciles: 10% in each bin
        current = np.full(50, 999.0)             # everything in the top bin
        eps, p = 1e-4, 0.1
        expected = 9 * (eps - p) * np.log(eps / p) + (1 - p) * np.log(1 / p)
        assert population_stability_index(reference, current) == pytest.approx(expected)

    def test_is_non_negative_and_grows_with_the_shift(self):
        ref = RNG.standard_normal(2000)
        values = [population_stability_index(ref, RNG.standard_normal(500) + s) for s in (0, 0.25, 0.5, 1, 2)]
        assert all(v >= 0 for v in values)
        assert values == sorted(values)

    def test_ignores_non_finite_values(self):
        ref = np.r_[RNG.standard_normal(500), np.nan, np.inf]
        assert np.isfinite(population_stability_index(ref, RNG.standard_normal(100)))

    def test_empirical_p_value_bounds(self):
        null = np.arange(99.0)
        assert empirical_p_value(1_000, null) == pytest.approx(1 / 100)
        assert empirical_p_value(-1, null) == 1.0


def test_two_proportion_test():
    assert two_proportion_p_value(30, 60, 300, 600) == pytest.approx(0.5)
    assert two_proportion_p_value(50, 60, 300, 600) < 1e-6
    assert two_proportion_p_value(50, 60, 300, 600, "two-sided") < 1e-6


class TestCalibration:
    """Naive tests assume iid data; market features are autocorrelated."""

    TRIALS = 80

    def _rates(self, phi):
        naive = calibrated = 0
        for seed in range(self.TRIALS):
            x = ar1_series(567, phi=phi, seed=seed) if phi else np.random.default_rng(seed).standard_normal(567)
            ref, cur = x[:504], x[504:]
            psi, p = calibrated_psi(ref, cur)
            naive += stats.ks_2samp(ref, cur).pvalue < 0.05
            calibrated += psi >= 0.25 and p < 0.05
        return naive / self.TRIALS, calibrated / self.TRIALS

    def test_naive_ks_false_alarms_on_autocorrelated_data_but_calibrated_rule_does_not(self):
        naive, calibrated = self._rates(phi=0.95)
        assert naive > 0.6          # textbook KS fires on most stationary AR(1) windows
        assert calibrated <= 0.15   # calibrated rule stays near its nominal 5%

    def test_calibrated_rule_still_detects_a_real_shift(self):
        hits = 0
        for seed in range(40):
            x = np.random.default_rng(seed).standard_normal(567)
            x[504:] += 1.0
            psi, p = calibrated_psi(x[:504], x[504:])
            hits += psi >= 0.25 and p < 0.05
        assert hits / 40 >= 0.9


# ═══════════════════════════════════════════════════════════════════════════
# CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def test_feature_drift_flags_only_the_shifted_feature():
    ref = pd.DataFrame({"stable": RNG.standard_normal(504), "shifted": RNG.standard_normal(504)})
    cur = pd.DataFrame({"stable": RNG.standard_normal(63), "shifted": RNG.standard_normal(63) + 3})
    rows = {r["feature"]: r for r in feature_drift(ref, cur, ["stable", "shifted"])}
    assert rows["shifted"]["drifted"] and not rows["stable"]["drifted"]
    assert rows["shifted"]["psi"] > rows["stable"]["psi"]


class TestVolatilityRegime:
    def test_doubled_volatility_is_a_high_regime(self):
        r = garch_returns(567, seed=21, vol_multiplier_after=504, vol_multiplier=2.0)
        vol = volatility_regime(r[:504], r[504:])
        assert vol["regime"] == "high" and vol["ratio"] >= 1.5 and vol["percentile"] >= 97.5

    def test_collapsed_volatility_is_a_low_regime(self):
        r = np.random.default_rng(1).normal(0, 0.01, 567)
        r[504:] *= 0.3
        assert volatility_regime(r[:504], r[504:])["regime"] == "low"

    def test_same_volatility_is_normal(self):
        r = np.random.default_rng(2).normal(0, 0.01, 567)
        vol = volatility_regime(r[:504], r[504:])
        assert vol["regime"] == "normal" and 0.7 < vol["ratio"] < 1.4

    def test_record_high_alone_is_not_an_alert(self):
        """At the top of its range but only 1.2× the reference: not economically large."""
        r = np.random.default_rng(3).normal(0, 0.01, 567)
        r[504:] = np.random.default_rng(4).normal(0, 0.012, 63)
        vol = volatility_regime(r[:504], r[504:])
        assert vol["ratio"] < 1.5 and vol["regime"] == "normal"


class TestConceptDrift:
    def _oos(self, err_ref, err_cur, seed=0):
        rng = np.random.default_rng(seed)
        errors = np.r_[rng.random(504) < err_ref, rng.random(63) < err_cur]
        target = rng.integers(0, 2, 567)
        return pd.DataFrame({"target": target, "signal": np.where(errors, 1 - target, target)})

    def test_detects_a_jump_in_error_rate(self):
        out = concept_drift(self._oos(0.35, 0.70))
        assert out["status"] == "degraded" and out["error_p_value"] < 0.001

    def test_stable_error_rate(self):
        assert concept_drift(self._oos(0.45, 0.45))["status"] == "stable"

    def test_needs_enough_history(self):
        assert concept_drift(self._oos(0.4, 0.4).iloc[-100:])["status"] == "insufficient_data"

    def test_unlabelled_live_row_is_ignored(self):
        oos = self._oos(0.4, 0.4)
        oos.loc[len(oos)] = {"target": np.nan, "signal": 1}
        assert concept_drift(oos)["n_oos"] == 567


# ═══════════════════════════════════════════════════════════════════════════
# REPORTS
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def market():
    return engineer_features(random_walk_ohlcv(900, seed=12))


def test_report_structure_and_json(market):
    rep = drift_report(market, FEATURE_COLS)
    assert rep["status"] in {"ok", "warning"}
    assert rep["n_current"] == 63 and rep["n_reference"] == 504
    assert rep["current_end"] == market.index[-1].strftime("%Y-%m-%d")
    assert len(rep["features"]) == len(FEATURE_COLS)
    json.dumps(rep)  # must be serialisable for the API and MLflow


def test_volatility_spike_raises_a_drift_alert(market):
    spiked = market.copy()
    log_close = np.log(spiked["close_price"].to_numpy())
    rets = np.diff(log_close)
    rets[-63:] *= 3.0  # triple the realised volatility of the last quarter
    spiked["close_price"] = np.exp(np.r_[log_close[0], log_close[0] + np.cumsum(rets)])
    rep = drift_report(spiked, FEATURE_COLS)
    assert rep["status"] == "drift"
    assert rep["volatility"]["regime"] == "high"
    assert any("Volatility regime shift" in a for a in rep["alerts"])


def test_dataset_drift_when_most_features_shift(market):
    shifted = market.copy()
    for col in FEATURE_COLS:
        sd = shifted[col].std()
        shifted.iloc[-63:, shifted.columns.get_loc(col)] += 5 * sd
    rep = drift_report(shifted, FEATURE_COLS)
    assert rep["status"] == "drift" and rep["share_drifted"] >= 0.5
    assert rep["alerts"][0].startswith("Dataset drift")


def test_concept_alert_is_included(market):
    rng = np.random.default_rng(5)
    target = rng.integers(0, 2, 567)
    errors = np.r_[rng.random(504) < 0.3, rng.random(63) < 0.75]
    oos = pd.DataFrame({"target": target, "signal": np.where(errors, 1 - target, target)})
    rep = drift_report(market, FEATURE_COLS, oos=oos)
    assert rep["concept"]["status"] == "degraded" and rep["status"] == "drift"


def test_short_history_is_reported_not_raised(market):
    rep = drift_report(market.iloc[-200:], FEATURE_COLS)
    assert rep["status"] == "insufficient_data" and rep["alerts"]


def test_fold_drift_has_one_row_per_fold(market):
    data = labelled_rows(market)
    folds = walk_forward_splits(len(data), WalkForwardConfig(initial_train_size=300))
    rows = fold_drift(data, folds, FEATURE_COLS, DriftConfig())
    assert [r["fold"] for r in rows] == list(range(1, len(folds) + 1))
    assert all(0 <= r["share_drifted"] <= 1 for r in rows)
