"""Unabhängige Referenz zum IRLS-Code: dieselbe Likelihood, aber ohne IRLS. Die Log-Likelihood kommt aus `scipy.stats`, die Umkehrfunktionen der Links sind
hier eigens geschrieben, und die Minimierung ist BFGS mit zentralen Differenzen (`jac="3-point"`); ein Löser ohne Gradient wäre nur auf etwa 1e-4 bis 1e-6 genau.

Zwei Prüfgrößen: (1) der Abstand der IRLS-Koeffizienten zum besten von mehreren Startpunkten der Minimierung (die Streuung der Startpunkte gehört dazu: bei flachen
Richtungen, etwa beim Kehrwert-Link nahe dem ungültigen Bereich, ist die Referenz selbst ungenau), (2) der Gradient dieser unabhängigen Likelihood am IRLS-Ergebnis."""

import numpy as np
from scipy import optimize, special, stats

import gfm_algorithm as A
import gfm_rng as R


def inv_link(name: str, eta):
    if name == "identity":
        return eta
    if name == "log":
        return np.exp(eta)
    if name == "inverse":
        return 1.0 / eta
    if name == "logit":
        return special.expit(eta)
    if name == "probit":
        return stats.norm.cdf(eta)
    if name == "cloglog":
        return -np.expm1(-np.exp(eta))
    raise ValueError(name)


def negloglik(family: str, link: str, X, y, beta, alpha: float = 0.2) -> float:
    """Negative Log-Likelihood aus scipy.stats (Normal mit Streuung 1, Poisson, Bernoulli, Gamma mit Form 1, NB2); 1e12 außerhalb des gültigen Bereichs."""
    with np.errstate(all="ignore"):
        mu = inv_link(link, X @ beta)
    if not np.all(np.isfinite(mu)):
        return 1e12
    if family == "normal":
        return float(-np.sum(stats.norm.logpdf(y, loc=mu, scale=1.0)))
    if family in ("gamma", "poisson", "nb2") and np.any(mu <= 0):
        return 1e12
    if family == "gamma":
        return float(-np.sum(stats.gamma.logpdf(y, a=1.0, scale=mu)))    # der β-Schätzer hängt bei festem Formparameter nicht von ihm ab
    if family == "poisson":
        return float(-np.sum(stats.poisson.logpmf(y, mu)))
    if family == "nb2":
        return float(-np.sum(stats.nbinom.logpmf(y, 1.0 / alpha, 1.0 / (1.0 + alpha * mu))))
    if family == "binomial":
        return float(-np.sum(stats.bernoulli.logpmf(y, np.clip(mu, A.P_EPS, 1 - A.P_EPS))))
    raise ValueError(family)


def gradient(family: str, link: str, X, y, beta, alpha: float = 0.2, rel: float = 1e-6) -> np.ndarray:
    """Gradient der unabhängigen Negativ-Log-Likelihood (zentrale Differenzen)."""
    beta = np.asarray(beta, float)
    g = np.zeros_like(beta)
    for j in range(len(beta)):
        h = rel * (1.0 + abs(beta[j]))
        e = np.zeros_like(beta)
        e[j] = h
        g[j] = (negloglik(family, link, X, y, beta + e, alpha) - negloglik(family, link, X, y, beta - e, alpha)) / (2 * h)
    return g


def minimize_reference(family: str, link: str, X, y, beta_irls, alpha: float = 0.2, starts: int = 3, seed: int = 5) -> dict:
    """Beste Lösung aus `starts` gestörten Startpunkten um das IRLS-Ergebnis. Rückgabe: beta (beste), spread (größte Streuung der Lösungen), dist (Abstand zum IRLS-Ergebnis),
    grad (größter Betrag des Gradienten der Referenz-Likelihood am IRLS-Ergebnis)."""
    rng = R.SplitMix64(seed)
    beta_irls = np.asarray(beta_irls, float)
    f = lambda b: negloglik(family, link, X, y, b, alpha)  # noqa: E731
    sols = []
    for _ in range(starts):
        x0 = beta_irls + np.array([0.02 * rng.normal() for _ in beta_irls]) * (1 + np.abs(beta_irls))
        res = optimize.minimize(f, x0, method="BFGS", jac="3-point", options={"gtol": 1e-9, "maxiter": 2000})
        sols.append(res.x)
    sols = np.array(sols)
    best = sols[int(np.argmin([f(s) for s in sols]))]
    return {"beta": best, "spread": float(np.max(np.ptp(sols, axis=0))), "dist": float(np.max(np.abs(best - beta_irls))),
            "grad": float(np.max(np.abs(gradient(family, link, X, y, beta_irls, alpha))))}
