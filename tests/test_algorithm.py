"""Der Kern gegen Handrechnung: vier Mini-Instanzen mit neun erfundenen Containern, jede Zahl in Brüchen (`fractions`) oder in geschlossener Form nachgerechnet.
Schrittzahlen sind Bänder, nie Konstanten; was in Brüchen steht, ist exakt."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest
from scipy import stats

import gfm_algorithm as A
import gfm_scenario as S


# --- Standgeld, Logit (kanonisch) ---------------------------------------------------------------------------------------------------------
def _logit_fits(hand):
    y, X = hand.standgeld, hand.X
    f = A.fit_irls(X, y, "binomial", "logit", "fisher", tol=1e-13, maxit=60, beta0=[0, 0])
    n = A.fit_irls(X, y, "binomial", "logit", "newton", tol=1e-13, maxit=60, beta0=[0, 0])
    return f, n


def test_logit_first_step_from_zero_is_exactly_minus_one_and_eleven_fifths(hand):
    """Von β = 0 ist p = 1/2, W = 1/4, die Arbeitsantwort z = 4 (y − 1/2); der gewichtete Kleinste-Quadrate-Schritt ist das Gruppenmittel von z je Gruppe: (−1; 11/5)."""
    z0 = [4 * (Fr(int(v)) - Fr(1, 2)) for v in hand.standgeld[:4]]
    z1 = [4 * (Fr(int(v)) - Fr(1, 2)) for v in hand.standgeld[4:]]
    b0 = sum(z0) / 4
    b1 = sum(z1) / 5 - b0
    assert (b0, b1) == (Fr(-1), Fr(11, 5))
    f = A.fit_irls(hand.X, hand.standgeld, "binomial", "logit", "fisher", tol=1e-13, maxit=1, beta0=[0, 0])
    assert f.path[1] == pytest.approx([-1.0, 2.2], abs=1e-14)


def test_logit_limit_in_closed_form_and_path_of_fisher_equals_newton(hand):
    f, n = _logit_fits(hand)
    assert f.converged and n.converged
    assert f.beta == pytest.approx([math.log(1 / 3), math.log(12)], abs=1e-10)       # (ln 1/3; ln 12) = (−1,098612; 2,484907)
    assert A.path_difference(f, n) < 1e-12                                           # kanonischer Link: derselbe Pfad
    assert 5 <= f.iterations <= 10                                                   # Band, keine Konstante


def test_bernoulli_deviance_is_10_ln5_minus_6_ln3_and_depends_only_on_the_fitted_probabilities(hand):
    f, _ = _logit_fits(hand)
    closed = 10 * math.log(5) - 6 * math.log(3)                                      # 9,5027054
    assert f.deviance == pytest.approx(closed, abs=1e-9) and closed == pytest.approx(9.5027054, abs=1e-6)
    p = np.array([1 / 4] * 4 + [4 / 5] * 5)
    other = np.array([1, 0, 0, 0] + [1, 1, 1, 1, 0], float)                          # andere Container mit denselben Zellhäufigkeiten (1 von 4, 4 von 5)
    d_other = float(np.sum(A.unit_deviance("binomial", other, p)))
    assert d_other == pytest.approx(f.deviance, abs=1e-12)
    h = -p * np.log(p) - (1 - p) * np.log(1 - p)
    assert 2 * float(np.sum(h)) == pytest.approx(closed, abs=1e-12)                  # 2 · Summe der Entropien (die Passung geht nicht ein)


# --- Standtage, Poisson, Identität (nicht kanonisch) --------------------------------------------------------------------------------------
def test_poisson_identity_group_means_and_harmonic_first_step(hand):
    y = hand.standtage
    assert (Fr(sum(int(v) for v in y[:4]), 4), Fr(sum(int(v) for v in y[4:]), 5)) == (Fr(6), Fr(52, 5))
    f = A.fit_irls(hand.X, y, "poisson", "identity", "fisher", tol=1e-13, maxit=10, mu0=y)
    h0 = Fr(4) / sum(Fr(1, int(v)) for v in y[:4])
    h1 = Fr(5) / sum(Fr(1, int(v)) for v in y[4:])
    assert (h0, h1) == (Fr(720, 131), Fr(6300, 641))                                 # Schritt 1 von μ0 = y: harmonisches Mittel je Gruppe
    assert f.path[0] == pytest.approx([float(h0), float(h1 - h0)], abs=1e-13)
    assert f.path[1] == pytest.approx([6.0, 4.4], abs=1e-13)                         # Schritt 2: arithmetisches Mittel, 4,4 Tage mehr durch Zoll
    assert f.beta == pytest.approx([6.0, 4.4], abs=1e-12)


def test_poisson_log_limit_is_group_means(hand):
    f = A.fit_irls(hand.X, hand.standtage, "poisson", "log", "fisher", tol=1e-13, maxit=30, mu0=hand.standtage)
    assert f.beta == pytest.approx([math.log(6), math.log(52 / 30)], abs=1e-10)       # Faktor 26/15
    assert Fr(52, 5) / Fr(6) == Fr(26, 15)


def test_one_group_start_five_fisher_hits_in_one_step_newton_squares_the_error():
    y = np.array([4.0, 5.0, 6.0, 9.0])
    X = np.ones((4, 1))
    f = A.fit_irls(X, y, "poisson", "identity", "fisher", tol=1e-13, maxit=3, beta0=[5.0])
    n = A.fit_irls(X, y, "poisson", "identity", "newton", tol=1e-13, maxit=4, beta0=[5.0])
    assert f.path[1][0] == pytest.approx(6.0, abs=1e-13)                             # ein Schritt
    e = [Fr(1)]
    for _ in range(4):
        e.append(e[-1] ** 2 / 6)                                                     # e_(k+1) = e_k² / ȳ mit ȳ = 6
    assert e[:4] == [Fr(1), Fr(1, 6), Fr(1, 216), Fr(1, 279936)]
    for k in range(4):
        assert 6.0 - n.path[k][0] == pytest.approx(float(e[k]), abs=1e-12)
    assert e[4] > 0                                                                  # Newton erreicht das Mittel nie in endlich vielen Schritten


# --- Verweilzeit, Gamma, drei Links, zwei Gruppen ------------------------------------------------------------------------------------------
@pytest.fixture
def gamma_fits(hand):
    return {lk: A.fit_irls(hand.X, hand.d, "gamma", lk, "fisher", tol=1e-13, maxit=60, mu0=hand.d) for lk in ("log", "identity", "inverse")}


def test_gamma_two_groups_all_links_give_the_group_means(gamma_fits, hand):
    assert (sum(S.HAND_D0) / 4, sum(S.HAND_D1) / 5) == (Fr(211, 40), Fr(489, 50))   # 5,275 und 9,78 Tage
    for f in gamma_fits.values():
        assert f.converged
        assert f.mu[:4] == pytest.approx(5.275, abs=1e-9) and f.mu[4:] == pytest.approx(9.78, abs=1e-9)


def test_gamma_two_groups_coefficients_per_link(gamma_fits):
    m0, m1 = 211 / 40, 489 / 50
    assert gamma_fits["log"].beta == pytest.approx([math.log(m0), math.log(m1 / m0)], abs=1e-9)
    assert Fr(489, 50) / Fr(211, 40) == Fr(1956, 1055)
    assert gamma_fits["identity"].beta == pytest.approx([m0, m1 - m0], abs=1e-9)
    assert Fr(489, 50) - Fr(211, 40) == Fr(901, 200)
    assert gamma_fits["inverse"].beta == pytest.approx([1 / m0, 1 / m1 - 1 / m0], abs=1e-9)


def test_gamma_unit_deviance_equals_twice_saturated_minus_model_loglik(gamma_fits, hand):
    f = gamma_fits["log"]
    dev = float(np.sum(A.unit_deviance("gamma", hand.d, f.mu)))
    ll = float(np.sum(stats.gamma.logpdf(hand.d, a=1.0, scale=f.mu)))
    ll_sat = float(np.sum(stats.gamma.logpdf(hand.d, a=1.0, scale=hand.d)))
    assert dev == pytest.approx(2 * (ll_sat - ll), abs=1e-10)
    assert f.deviance == pytest.approx(dev, abs=1e-12)


def test_gamma_log_link_score_but_not_sum_of_residuals_in_general():
    """Beim Gamma mit Log-Link gilt X'((y − μ)/μ) = 0, nicht X'(y − μ) = 0 (der Log-Link ist nicht kanonisch): schon bei einer Gruppe mit zwei Parametern gilt das nicht."""
    T = S.draw_table(400, 7)
    X = T.designmatrix()
    f = A.fit_irls(X, T.d, "gamma", "log", "fisher", tol=1e-12, maxit=100)
    assert np.max(np.abs(X.T @ ((T.d - f.mu) / f.mu))) < 1e-6
    assert abs(float(np.sum(T.d - f.mu))) > 1e-4 * float(T.d.sum())                  # die Summe der Residuen ist nicht null (hier 3,5·10⁻⁴ relativ)


# --- Familien und Links einzeln -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("link", ["identity", "log", "inverse", "logit", "probit", "cloglog"])
def test_link_inverse_roundtrip_and_derivatives(link):
    eta = np.array([0.3, 0.8, 1.4]) if link in ("log", "inverse", "identity") else np.array([-1.2, 0.1, 0.9])
    mu = A.link_h(link, eta)
    assert A.link_g(link, mu) == pytest.approx(eta, abs=1e-12)
    h = 1e-6
    d1 = (A.link_h(link, eta + h) - A.link_h(link, eta - h)) / (2 * h)
    assert A.link_h1(link, eta) == pytest.approx(d1, rel=1e-7, abs=1e-9)
    d2 = (A.link_h1(link, eta + h) - A.link_h1(link, eta - h)) / (2 * h)
    assert A.link_h2(link, eta) == pytest.approx(d2, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("family", list(A.FAMILIES))
def test_variance_derivative_matches_finite_difference(family):
    mu = np.array([0.3, 0.55, 0.9])
    h = 1e-6
    d = (A.variance(family, mu + h, 0.2) - A.variance(family, mu - h, 0.2)) / (2 * h)
    assert A.variance_d1(family, mu, 0.2) == pytest.approx(d, rel=1e-8)


def test_nine_pairs_canonical_flags():
    assert {k for k, _, f, lk in __import__("gfm_constants").PAIRS if A.is_canonical(f, lk)} == {"normal/identity", "gamma/inverse", "poisson/log", "binomial/logit"}
