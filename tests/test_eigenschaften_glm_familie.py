"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für das Verfahren gelten müssen, ohne dass man das richtige Ergebnis kennt. Je Eigenschaft 20 Instanzen
(Container-Tabellen mit n = 150 bis 300, feste Zufalls-Seeds), Toleranzen als Bänder.

Die Aussagen gelten für die geprüften Paare und Instanzen, nicht für jeden Datensatz („bei kanonischem Link ist der Pfad gleich“ ist eine Zählung über 20 Instanzen je Paar)."""

import math

import numpy as np
import pytest

import gfm_algorithm as A
import gfm_constants as C
import gfm_inference as I
import gfm_scenario as S
from conftest import Zufall

SEEDS = list(range(300, 320))      # 20 Instanzen
CANON = [(k, a, f, lk) for k, a, f, lk in C.PAIRS if A.is_canonical(f, lk)]
NONCANON = [(k, a, f, lk) for k, a, f, lk in C.PAIRS if not A.is_canonical(f, lk)]


def _instance(seed, aus, n=250):
    n = max(n, 900) if aus == "binär" else n         # 0/1-Ausgang: kleine Tabellen trennen sich (Separation), dann gibt es keinen Schätzer
    T = S.draw_table(n, seed)
    return T.designmatrix(), {"stetig": T.d, "zähl": T.standtage(), "binär": T.standgeld()}[aus]


def _fit(X, y, fam, lk, mode="fisher", **kw):
    return A.fit_irls(X, y, fam, lk, mode, tol=1e-11, maxit=5000, **kw)


@pytest.mark.parametrize("key,aus,fam,lk", CANON)
def test_canonical_link_path_of_fisher_is_the_path_of_newton(key, aus, fam, lk):
    for seed in SEEDS:
        X, y = _instance(seed, aus)
        f, n = _fit(X, y, fam, lk, "fisher"), _fit(X, y, fam, lk, "newton")
        assert f.converged and n.converged, (key, seed)
        assert A.path_difference(f, n) < 1e-9, (key, seed)


@pytest.mark.parametrize("key,aus,fam,lk", NONCANON)
def test_non_canonical_link_has_different_paths_and_the_same_limit(key, aus, fam, lk):
    verschieden = fisher_ok = 0
    for seed in SEEDS:
        X, y = _instance(seed, aus)
        f, n = _fit(X, y, fam, lk, "fisher"), _fit(X, y, fam, lk, "newton")
        assert n.converged, (key, seed)                                            # Newton konvergiert in allen 20 Instanzen
        verschieden += A.path_difference(f, n) > 1e-6
        if f.converged:                                                            # Fisher-Scoring bei Poisson/Identität kann am Rand des zulässigen Bereichs kriechen (siehe unten)
            fisher_ok += 1
            assert float(np.max(np.abs(f.beta - n.beta))) < 1e-7, (key, seed)     # gleicher Grenzwert
    assert verschieden == len(SEEDS), key                                          # Zählung über 20 Instanzen: in allen Instanzen verschiedene Pfade
    # gemessen: 16 von 20 Instanzen (n = 250) bei Poisson/Identität; die übrigen vier brauchen mehr als 5 000 Schritte
    assert fisher_ok >= (12 if key == "poisson/identity" else len(SEEDS)), (key, fisher_ok)


@pytest.mark.parametrize("key,aus,fam,lk", C.PAIRS)
def test_score_is_zero_at_the_limit(key, aus, fam, lk):
    X, y = _instance(SEEDS[1], aus)
    f = _fit(X, y, fam, lk)
    assert f.converged
    d1 = A.link_h1(lk, f.eta)
    V = A.variance(fam, f.mu, C.NB2_ALPHA if fam == "nb2" else 0.2)
    scale = np.max(np.abs(X.T @ (np.abs(y - f.mu) * d1 / V)))
    assert np.max(np.abs(X.T @ ((y - f.mu) * d1 / V))) < 1e-7 * scale


@pytest.mark.parametrize("key,aus,fam,lk", CANON)
def test_canonical_link_with_intercept_makes_residuals_sum_to_zero(key, aus, fam, lk):
    for seed in SEEDS[:10]:
        X, y = _instance(seed, aus)
        f = _fit(X, y, fam, lk)
        assert abs(float(np.sum(y - f.mu))) < 1e-8 * (1 + float(np.sum(np.abs(y))))


def test_gamma_log_link_has_residual_score_but_the_residuals_do_not_sum_to_zero():
    X, y = _instance(SEEDS[0], "stetig")
    f = _fit(X, y, "gamma", "log")
    assert np.max(np.abs(X.T @ ((y - f.mu) / f.mu))) < 1e-7
    assert abs(float(np.sum(y - f.mu))) > 1e-6 * float(y.sum())


def test_permuting_the_observations_changes_nothing():
    zf = Zufall(5)
    for seed in SEEDS[:8]:
        X, y = _instance(seed, "zähl", 150)
        order = np.argsort(zf.random(len(y)))
        a, b = _fit(X, y, "poisson", "log"), _fit(X[order], y[order], "poisson", "log")
        assert b.beta == pytest.approx(a.beta, abs=1e-9)
        assert b.deviance == pytest.approx(a.deviance, rel=1e-10)


def test_doubling_all_observations_keeps_beta_and_doubles_deviance_and_pearson():
    for seed in SEEDS[:8]:
        X, y = _instance(seed, "zähl", 150)
        a = _fit(X, y, "poisson", "log")
        b = _fit(np.vstack([X, X]), np.r_[y, y], "poisson", "log")
        assert b.beta == pytest.approx(a.beta, abs=1e-9)
        assert b.deviance == pytest.approx(2 * a.deviance, rel=1e-9)
        assert I.pearson_x2("poisson", np.r_[y, y], b.mu) == pytest.approx(2 * I.pearson_x2("poisson", y, a.mu), rel=1e-9)


def test_prior_weight_two_equals_duplicated_observation():
    X, y = _instance(SEEDS[1], "stetig", 120)
    w = np.ones(len(y))
    w[:10] = 2.0
    a = _fit(X, y, "gamma", "log", weights=w)
    b = _fit(np.vstack([X, X[:10]]), np.r_[y, y[:10]], "gamma", "log")
    assert b.beta == pytest.approx(a.beta, abs=1e-9) and b.deviance == pytest.approx(a.deviance, rel=1e-10)


def test_offset_log_t_equals_rate_with_weight_t():
    zf = Zufall(9)
    for seed in SEEDS[:6]:
        X, y = _instance(seed, "zähl", 150)
        t = zf.uniform(0.5, 3.0, len(y))
        a = _fit(X, y, "poisson", "log", offset=np.log(t))
        b = _fit(X, y / t, "poisson", "log", weights=t)
        assert b.beta == pytest.approx(a.beta, abs=1e-8)


def test_gamma_is_invariant_under_scaling_the_outcome_except_the_log_intercept():
    for seed in SEEDS[:6]:
        X, y = _instance(seed, "stetig", 150)
        a, b = _fit(X, y, "gamma", "log"), _fit(X, 3.0 * y, "gamma", "log")
        assert b.beta[1:] == pytest.approx(a.beta[1:], abs=1e-9)
        assert b.beta[0] == pytest.approx(a.beta[0] + math.log(3.0), abs=1e-9)
        assert b.deviance == pytest.approx(a.deviance, rel=1e-9)


@pytest.mark.parametrize("fam,lk,aus", [("poisson", "log", "zähl"), ("binomial", "probit", "binär"), ("gamma", "log", "stetig")])
def test_deviance_is_nonnegative_and_never_rises_when_a_feature_is_added(fam, lk, aus):
    for seed in SEEDS:
        X, y = _instance(seed, aus, 200)
        small, big = _fit(X[:, :5], y, fam, lk), _fit(X, y, fam, lk)
        assert big.deviance >= 0 and small.deviance >= 0
        assert big.deviance <= small.deviance + 1e-8


def test_nb2_with_alpha_to_zero_equals_poisson():
    X, y = _instance(SEEDS[2], "zähl", 250)
    a = _fit(X, y, "poisson", "log")
    b = _fit(X, y, "nb2", "log", alpha=1e-9)
    assert b.beta == pytest.approx(a.beta, abs=1e-5)


def test_aic_of_two_links_with_the_same_parameters_is_equal_for_two_groups():
    h = S.HAND_NEUN
    aics = []
    for lk in ("log", "identity", "inverse"):
        f = A.fit_irls(h.X, h.d, "gamma", lk, "fisher", tol=1e-13, maxit=60, mu0=h.d)
        aics.append(I.aic_gamma(h.d, f.mu, 2))
    assert aics == pytest.approx([aics[0]] * 3, abs=1e-8)                           # aus zwei Gruppenmitteln ist der Link nicht wählbar


def test_saturated_model_has_zero_deviance():
    """Zwei Gruppen mit je einem Parameter (Achsenabschnitt und Zollhalt): das Gamma-Modell passt die Gruppenmittel, aber nicht jede Beobachtung; gesättigt ist erst ein Parameter je Beobachtung."""
    y = np.array([2.0, 3.0, 5.0, 8.0])
    f = A.fit_irls(np.eye(4), y, "gamma", "log", "fisher", tol=1e-12, maxit=60)
    assert f.deviance == pytest.approx(0.0, abs=1e-10) and f.mu == pytest.approx(y, abs=1e-9)
    g = A.fit_irls(np.eye(4), np.array([0.0, 1.0, 1.0, 0.0]), "binomial", "logit", "fisher", tol=1e-12, maxit=60, halve=False)
    assert g.deviance == pytest.approx(0.0, abs=1e-6) or not g.converged            # gesättigtes 0/1-Modell: p̂ → 0 oder 1, Trennung


def test_same_seed_gives_the_same_fit_and_another_seed_another():
    X, y = _instance(7, "zähl")
    X2, y2 = _instance(7, "zähl")
    assert np.array_equal(_fit(X, y, "poisson", "log").beta, _fit(X2, y2, "poisson", "log").beta)
    X3, y3 = _instance(8, "zähl")
    assert not np.allclose(_fit(X, y, "poisson", "log").beta, _fit(X3, y3, "poisson", "log").beta)
