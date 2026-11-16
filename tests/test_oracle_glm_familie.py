"""Orakel (ORACLE-PLAYBOOK §2, §5): der IRLS-Kern und die Deviancen gegen einen anderen Rechenweg.

Referenzen: (1) die Log-Likelihood aus `scipy.stats` mit eigenen Umkehrfunktionen der Links, minimiert mit BFGS und `jac="3-point"` (nie ohne Gradient: ein Löser ohne Gradient
ist nur auf 1e-4 bis 1e-6 genau) und der Gradient dieser Likelihood am IRLS-Ergebnis; (2) scikit-learn (`LogisticRegression`, `PoissonRegressor`, `GammaRegressor`) und `numpy.linalg.lstsq`;
(3) Deviancen als 2 (ℓ_gesättigt − ℓ) aus `scipy.stats`. Die Referenz streut bei flachen Richtungen (Gamma/Kehrwert nahe dem ungültigen Bereich): dort gilt der Gradient am eigenen
Ergebnis und der Vergleich der Zielfunktionswerte, nicht der Vergleich der Parameter. Tests, die scikit-learn brauchen, überspringen sich ohne.
Plattform: scipy-Lösungen und BLAS-Summen haben Toleranz (1e-5 bis 1e-9)."""

import warnings

import numpy as np
import pytest
from scipy import optimize, special, stats

import gfm_algorithm as A
import gfm_constants as C
import gfm_inference as I
import gfm_scenario as S
from conftest import Zufall

N = 500
ALPHA = 0.2


@pytest.fixture(scope="module")
def table():
    T = S.draw_table(N, 180)
    return T.designmatrix(), {"stetig": T.d, "zähl": T.standtage(), "binär": T.standgeld()}


# --- eigene Referenz (unabhängig von gfm_reference) ---------------------------------------------------------------------------------------
def _h(link, eta):
    return {"identity": lambda e: e, "log": np.exp, "inverse": lambda e: 1.0 / e, "logit": special.expit, "probit": stats.norm.cdf, "cloglog": lambda e: -np.expm1(-np.exp(e))}[link](eta)


def _negll(fam, link, X, y, b):
    with np.errstate(all="ignore"):
        mu = _h(link, X @ b)
    if not np.all(np.isfinite(mu)):
        return 1e12
    if fam == "normal":
        return -float(np.sum(stats.norm.logpdf(y, mu, 1.0)))
    if fam in ("gamma", "poisson", "nb2") and np.any(mu <= 0):
        return 1e12
    if fam == "gamma":
        return -float(np.sum(stats.gamma.logpdf(y, a=1.0, scale=mu)))
    if fam == "poisson":
        return -float(np.sum(stats.poisson.logpmf(y, mu)))
    if fam == "nb2":
        return -float(np.sum(stats.nbinom.logpmf(y, 1 / ALPHA, 1 / (1 + ALPHA * mu))))
    return -float(np.sum(stats.bernoulli.logpmf(y, np.clip(mu, 1e-15, 1 - 1e-15))))


def _grad(f, b, rel=1e-6):
    g = np.zeros_like(b)
    for j in range(len(b)):
        h = rel * (1 + abs(b[j]))
        e = np.zeros_like(b)
        e[j] = h
        g[j] = (f(b + e) - f(b - e)) / (2 * h)
    return g


@pytest.mark.parametrize("key,aus,fam,link", C.PAIRS)
def test_all_nine_pairs_match_an_independent_minimisation(table, key, aus, fam, link):
    X, ys = table
    y = ys[aus]
    for mode in ("fisher", "newton"):
        fit = A.fit_irls(X, y, fam, link, mode, tol=1e-10, maxit=500, alpha=ALPHA)
        assert fit.converged, (key, mode)
        f = lambda b: _negll(fam, link, X, y, b)  # noqa: E731
        assert np.max(np.abs(_grad(f, fit.beta))) < 1e-4, (key, mode)                      # Gradient der unabhängigen Likelihood am IRLS-Ergebnis
        zf = Zufall(3)
        sols = [optimize.minimize(f, fit.beta + zf.normal(len(fit.beta), 0.02) * (1 + np.abs(fit.beta)), method="BFGS", jac="3-point", options={"gtol": 1e-9, "maxiter": 2000}) for _ in range(2)]
        best = min(sols, key=lambda r: r.fun)
        assert f(fit.beta) <= best.fun + 1e-7 * (1 + abs(best.fun)), (key, mode)           # kein besserer Zielfunktionswert der Referenz
        if key != "gamma/inverse":                                                          # dort streut die Referenz über die Startpunkte (flache Richtung)
            assert np.max(np.abs(best.x - fit.beta)) < 1e-5, (key, mode)


def test_sklearn_and_lstsq_agree_with_the_irls_code(table):
    sk = pytest.importorskip("sklearn.linear_model")
    X, ys = table
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b_logit = sk.LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-12, max_iter=1000, fit_intercept=False).fit(X, ys["binär"]).coef_.ravel()
        b_pois = sk.PoissonRegressor(alpha=0, tol=1e-12, max_iter=1000, fit_intercept=False).fit(X, ys["zähl"]).coef_.ravel()
        b_gamma = sk.GammaRegressor(alpha=0, tol=1e-12, max_iter=1000, fit_intercept=False).fit(X, ys["stetig"]).coef_.ravel()
    b_ols = np.linalg.lstsq(X, ys["stetig"], rcond=None)[0]
    assert np.max(np.abs(b_logit - A.fit_irls(X, ys["binär"], "binomial", "logit").beta)) < 1e-7
    assert np.max(np.abs(b_pois - A.fit_irls(X, ys["zähl"], "poisson", "log").beta)) < 1e-5
    assert np.max(np.abs(b_gamma - A.fit_irls(X, ys["stetig"], "gamma", "log").beta)) < 1e-5
    assert np.max(np.abs(b_ols - A.fit_irls(X, ys["stetig"], "normal", "identity").beta)) < 1e-9


# --- Deviancen ----------------------------------------------------------------------------------------------------------------------------
def _sat_ll(fam, y):
    if fam == "normal":
        return stats.norm.logpdf(y, y, 1.0)
    if fam == "poisson":
        return stats.poisson.logpmf(y, y)
    if fam == "binomial":
        return stats.bernoulli.logpmf(y, y)
    if fam == "gamma":
        return stats.gamma.logpdf(y, a=1.0, scale=y)
    return stats.nbinom.logpmf(y, 1 / ALPHA, 1 / (1 + ALPHA * y))


def _ll(fam, y, mu):
    if fam == "normal":
        return stats.norm.logpdf(y, mu, 1.0)
    if fam == "poisson":
        return stats.poisson.logpmf(y, mu)
    if fam == "binomial":
        return stats.bernoulli.logpmf(y, mu)
    if fam == "gamma":
        return stats.gamma.logpdf(y, a=1.0, scale=mu)
    return stats.nbinom.logpmf(y, 1 / ALPHA, 1 / (1 + ALPHA * mu))


@pytest.mark.parametrize("fam", list(A.FAMILIES))
def test_unit_deviance_is_twice_saturated_minus_model_loglik(fam):
    zf = Zufall(11)
    if fam == "binomial":
        y = (zf.random(50) < 0.4).astype(float)
        mu = zf.uniform(0.05, 0.95, 50)
    elif fam in ("poisson", "nb2"):
        y = zf.integers(0, 12, 50).astype(float)
        mu = zf.uniform(0.5, 9, 50)
    else:
        y = zf.uniform(0.3, 12, 50)
        mu = zf.uniform(0.5, 9, 50)
    got = A.unit_deviance(fam, y, mu, ALPHA)
    assert got == pytest.approx(2 * (_sat_ll(fam, y) - _ll(fam, y, mu)), abs=1e-10)
    assert np.all(got >= -1e-12)


def test_logit_deviance_depends_only_on_fitted_probabilities(table):
    X, ys = table
    f = A.fit_irls(X, ys["binär"], "binomial", "logit")
    eta = np.log(f.mu / (1 - f.mu))
    assert f.deviance == pytest.approx(-2 * float(np.sum(f.mu * eta + np.log(1 - f.mu))), abs=1e-8)
    assert f.deviance == pytest.approx(-2 * float(np.sum(stats.bernoulli.logpmf(ys["binär"], f.mu))), abs=1e-8)


def test_ungrouped_equals_grouped_minus_twice_saturated_loglik():
    zf = Zufall(5)
    n = 300
    key = zf.integers(0, 12, n)
    y = (zf.random(n) < 0.35).astype(float)
    p = np.clip(0.2 + 0.05 * (key % 5), 0.05, 0.95)                                         # eine Wahrscheinlichkeit je Zelle (gleiche Merkmale)
    du = -2 * float(np.sum(stats.bernoulli.logpmf(y, p)))
    g = I.grouped_binomial(key, y, p, 12)
    assert du == pytest.approx(g["dev"] - 2 * g["llsat"], abs=1e-9)
    m = np.bincount(key, minlength=12)
    s = np.bincount(key, weights=y, minlength=12)
    ph = np.array([p[key == k][0] for k in range(12)])
    dev_bin = -2 * float(np.sum(stats.binom.logpmf(s, m, ph) - stats.binom.logpmf(s, m, np.where(m > 0, s / np.maximum(m, 1), 0.0))))
    assert g["dev"] == pytest.approx(dev_bin, abs=1e-9)                                     # gruppierte Deviance = 2 (ℓ_gesättigt − ℓ) aus der Binomialverteilung


def test_gamma_profile_likelihood_equals_scipy_gamma_fit_at_the_sample_mean():
    y = Zufall(1820).gamma_int(3, 6.0, 500)
    ll, a = I.gamma_loglik_profile(y, np.full(500, y.mean()))
    a_sp, _, sc = stats.gamma.fit(y, floc=0)
    assert a == pytest.approx(a_sp, rel=1e-3)
    assert ll == pytest.approx(float(np.sum(stats.gamma.logpdf(y, a_sp, scale=sc))), rel=1e-4)


def test_nb2_loglik_and_the_two_ways_to_estimate_alpha(table):
    X, ys = table
    y = ys["zähl"]
    f = A.fit_irls(X, y, "nb2", "log", alpha=ALPHA)
    assert I.loglik_nb2(y, f.mu, ALPHA) == pytest.approx(float(np.sum(stats.nbinom.logpmf(y, 1 / ALPHA, 1 / (1 + ALPHA * f.mu)))), abs=1e-8)
    prof, alt = I.nb2_profile_alpha(X, y), I.nb2_alternating(X, y)
    assert prof["alpha"] == pytest.approx(alt["alpha"], rel=1e-3)
    assert prof["ll"] >= I.loglik_nb2(y, alt["fit"].mu, alt["alpha"]) - 1e-6                 # die Profil-Außenschleife ist nicht schlechter


def test_likelihood_ratio_statistic_equals_difference_of_two_loglikelihoods():
    im = S.draw_import(600, 4)
    y = im.standgeld()
    X1 = np.column_stack([np.ones(600), im.zoll, im.we, S.kunden_dummies(im.kunde)])
    f1 = A.fit_irls(X1, y, "binomial", "probit")
    f0 = A.fit_irls(X1[:, :3], y, "binomial", "probit")
    ll1 = float(np.sum(stats.bernoulli.logpmf(y, f1.mu)))
    ll0 = float(np.sum(stats.bernoulli.logpmf(y, f0.mu)))
    t = I.lr_test(f0.deviance, f1.deviance, 11)
    assert t["stat"] == pytest.approx(2 * (ll1 - ll0), abs=1e-8)
    assert t["p"] == pytest.approx(float(stats.chi2.sf(2 * (ll1 - ll0), 11)), abs=1e-12)
