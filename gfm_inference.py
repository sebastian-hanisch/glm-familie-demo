"""Güte und Wahl: Log-Likelihoods, Deviance (ungruppiert und gruppiert), Pearson-X², AIC mit dem Parameter für Form bzw. α, Likelihood-Quotienten-Test und der
zweite Parameter von NB2 (Außenschleife um das IRLS). Nur numpy und scipy.

Konventionen (README, „Mathematische Formulierung“): die Deviance ist eine **Summe**, kein Mittel; die ungruppierte Bernoulli-Deviance nimmt als gesättigtes Modell
„jeder Container eine eigene Wahrscheinlichkeit“ (Log-Likelihood 0), die gruppierte „jede Zelle gleicher Merkmale eine eigene Wahrscheinlichkeit“;
AIC = −2·Log-Likelihood + 2·(Zahl der Parameter), bei Gamma mit +1 für die auf die Form profilierte Likelihood, bei NB2 mit +1 für α."""

import numpy as np
from scipy import optimize, special, stats

import gfm_algorithm as A


def pearson_x2(family: str, y, mu, alpha: float = 0.2) -> float:
    return float(np.sum((y - mu) ** 2 / A.variance(family, mu, alpha)))


def chi2_crit(df: float, level: float = 0.95) -> float:
    return float(stats.chi2.ppf(level, df))


# --- Log-Likelihoods ---------------------------------------------------------------------------------------------------------------------
def loglik_bernoulli(y, p) -> float:
    p = np.clip(p, A.P_EPS, 1 - A.P_EPS)
    return float(np.sum(special.xlogy(y, p) + special.xlogy(1 - y, 1 - p)))


def loglik_poisson(y, mu) -> float:
    return float(np.sum(special.xlogy(y, mu) - mu - special.gammaln(y + 1)))


def loglik_nb2(y, mu, alpha: float) -> float:
    r = 1.0 / alpha
    return float(np.sum(special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1) + r * np.log(r / (r + mu)) + special.xlogy(y, mu / (r + mu))))


def gamma_loglik_profile(y, mu) -> tuple[float, float]:
    """Gamma-Log-Likelihood bei festem Mittel mu, auf die Form a profiliert (die ML-Form löst Σ[ln a + 1 − ln mu − ψ(a) + ln y − y/mu] = 0). Rückgabe (ll, Form)."""
    ly = np.log(y)

    def ds(a):
        return float(np.sum(np.log(a) + 1.0 - np.log(mu) - special.digamma(a) + ly - y / mu))

    lo, hi = 1e-3, 1e7
    if ds(lo) < 0:
        a = lo
    elif ds(hi) > 0:
        a = hi
    else:
        a = optimize.brentq(ds, lo, hi, xtol=1e-12, rtol=1e-12)
    ll = float(np.sum(a * np.log(a / mu) - special.gammaln(a) + (a - 1) * ly - a * y / mu))
    return ll, float(a)


def aic_gamma(y, mu, n_mean_params: int) -> float:
    """AIC des Gamma-GLM mit der auf die Form profilierten Likelihood: Parameter = Mittelparameter + 1 (Form)."""
    ll, _ = gamma_loglik_profile(y, mu)
    return -2.0 * ll + 2.0 * (n_mean_params + 1)


# --- gruppierte Deviance (Binomial je Zelle) ---------------------------------------------------------------------------------------------
def grouped_binomial(key, y, p, ncell: int) -> dict:
    """Binomial-Deviance und Pearson-X² je Zelle gleicher Merkmale (`key` = Zellnummer 0..ncell−1; alle Container einer Zelle haben dasselbe p̂).
    Rückgabe: dev, x2, cells (nicht leere Zellen), llsat (gesättigte Log-Likelihood der Zellen), small (Anteil Zellen mit erwarteter Zahl < 5)."""
    key = np.asarray(key)
    m = np.bincount(key, minlength=ncell).astype(float)
    ys = np.bincount(key, weights=y, minlength=ncell)
    ph = np.zeros(ncell)
    ph[key] = p
    keep = m > 0
    m, ys, ph = m[keep], ys[keep], ph[keep]
    dev = 2.0 * np.sum(special.xlogy(ys, ys / (m * ph)) + special.xlogy(m - ys, (m - ys) / (m * (1 - ph))))
    x2 = np.sum((ys - m * ph) ** 2 / (m * ph * (1 - ph)))
    llsat = float(np.sum(special.xlogy(ys, ys / m) + special.xlogy(m - ys, 1 - ys / m)))
    small = float(np.mean(np.minimum(m * ph, m * (1 - ph)) < 5))
    return {"dev": float(dev), "x2": float(x2), "cells": int(keep.sum()), "llsat": llsat, "small": small}


def mean_entropy_x2(p) -> float:
    """2 · mittlere Entropie der Wahrscheinlichkeiten p: Mittel von −2 [p ln p + (1 − p) ln(1 − p)]. Die ungruppierte Bernoulli-Deviance geteilt durch n liegt bei richtigem Modell nahe daran."""
    p = np.asarray(p, float)
    return float(np.mean(-2.0 * (special.xlogy(p, p) + special.xlogy(1 - p, 1 - p))))


def lr_test(dev_small: float, dev_big: float, df: int, level: float = 0.95) -> dict:
    """Likelihood-Quotienten-Test: Differenz der ungruppierten Deviances gegen χ² mit df Freiheitsgraden."""
    stat = dev_small - dev_big
    crit = chi2_crit(df, level)
    return {"stat": float(stat), "crit": crit, "reject": bool(stat > crit), "p": float(stats.chi2.sf(stat, df))}


# --- zweiter Parameter: α von NB2 ---------------------------------------------------------------------------------------------------------
def nb2_profile_alpha(X, y, tol: float = 1e-10) -> dict:
    """α durch eine Außenschleife: für jedes α ein IRLS (Log-Link, festes α), die Profil-Log-Likelihood wird über log α maximiert (skalar, begrenzt).
    IRLS allein schätzt α nicht: es löst nur die Score-Gleichung für β bei festem α."""
    def neg_profile(la):
        a = float(np.exp(la))
        f = A.fit_irls(X, y, "nb2", "log", tol=tol, alpha=a)
        return -loglik_nb2(y, f.mu, a)

    sol = optimize.minimize_scalar(neg_profile, bounds=(-6, 3), method="bounded", options={"xatol": 1e-7})
    alpha = float(np.exp(sol.x))
    f = A.fit_irls(X, y, "nb2", "log", tol=tol, alpha=alpha)
    return {"alpha": alpha, "fit": f, "ll": loglik_nb2(y, f.mu, alpha)}


def nb2_alpha_given_mu(y, mu) -> float:
    """ML für r = 1/α bei festem Mittel mu: Nullstelle der Score-Funktion in ln r (Brent)."""
    def s(lr):
        r = np.exp(lr)
        return float(np.sum(special.digamma(y + r) - special.digamma(r) + np.log(r / (r + mu)) + (mu - y) / (r + mu)))

    lo, hi = -6.0, 14.0
    if s(lo) < 0:
        return float(np.exp(-lo))
    if s(hi) > 0:
        return float(np.exp(-hi))
    return float(1.0 / np.exp(optimize.brentq(s, lo, hi, xtol=1e-8)))


def nb2_alternating(X, y, max_outer: int = 30, tol: float = 1e-9) -> dict:
    """NB2 wie `glm.nb` in R: abwechselnd α bei festem Mittel und β bei festem α, bis die Log-Likelihood ruht."""
    f = A.fit_irls(X, y, "poisson", "log")
    alpha = max(nb2_alpha_given_mu(y, f.mu), 1e-8)
    ll_old = -np.inf
    outer = 0
    for outer in range(1, max_outer + 1):
        f = A.fit_irls(X, y, "nb2", "log", alpha=alpha)
        alpha = max(nb2_alpha_given_mu(y, f.mu), 1e-8)
        ll = loglik_nb2(y, f.mu, alpha)
        if abs(ll - ll_old) < tol * (abs(ll) + 1):
            break
        ll_old = ll
    f = A.fit_irls(X, y, "nb2", "log", alpha=alpha)
    return {"alpha": float(alpha), "fit": f, "outer": outer}
