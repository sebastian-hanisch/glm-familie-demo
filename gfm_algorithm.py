"""Kern: ein IRLS-Code für die GLM-Familie (Familie und Link sind Parameter), mit Fisher-Scoring und Newton.

Beobachtung y_i mit Mittel mu_i und Varianz phi·V(mu_i); linearer Prädiktor eta_i = x_i'β + Offset_i; Link g(mu_i) = eta_i, also mu_i = h(eta_i).
IRLS-Schritt (Fisher-Scoring, erwartete Information): W_i = w_i·h'(eta_i)²/V(mu_i), Score U = X'(w·(y − mu)·h'/V), β ← β + (X'WX)⁻¹ U;
gleichbedeutend β ← (X'WX)⁻¹ X'W z mit der Arbeitsantwort z_i = eta_i − Offset_i + (y_i − mu_i)/h'(eta_i). Newton nimmt die beobachtete Information:
W_i wird durch W_i − w_i·(y_i − mu_i)·(h''·V − h'²·V')/V² ersetzt. Beim kanonischen Link (Normal/Identität, Binomial/Logit, Poisson/Log, Gamma/Kehrwert)
sind beide gleich. Konvergenz: Betragsmaximum des **vollen** Schritts unter der Toleranz (nie des halbierten, sonst „konvergiert“ der Code mit Schrittweite 0).

Familien: Normal (V = 1), Poisson (V = mu), Binomial (V = mu(1 − mu), Bernoulli), Gamma (V = mu²), NB2 (V = mu + alpha·mu², alpha fest).
Einheitsdeviance (Summe = Deviance D, Vorgewicht 1, Dispersion nicht eingerechnet): siehe `unit_deviance`."""

from dataclasses import dataclass, field

import numpy as np
from scipy import special

FAMILIES = ("normal", "poisson", "binomial", "gamma", "nb2")
LINKS = ("identity", "log", "inverse", "logit", "probit", "cloglog")
CANONICAL = {("normal", "identity"), ("binomial", "logit"), ("poisson", "log"), ("gamma", "inverse")}
P_EPS = 1e-15
SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def is_canonical(family: str, link: str) -> bool:
    return (family, link) in CANONICAL


# --- Links: eta = g(mu), mu = h(eta), h'(eta), h''(eta) ---------------------------------------------------------------------------------
def link_g(link: str, mu):
    if link == "identity":
        return mu
    if link == "log":
        return np.log(mu)
    if link == "inverse":
        return 1.0 / mu
    if link == "logit":
        return special.logit(mu)
    if link == "probit":
        return special.ndtri(mu)
    if link == "cloglog":
        return np.log(-np.log1p(-mu))
    raise ValueError(link)


def link_h(link: str, eta):
    """Umkehrfunktion des Links. Logit mit |eta| ≤ 30 (wie R), Probit und cloglog mit den Wahrscheinlichkeiten auf [1e-15; 1 − 1e-15]."""
    if link == "identity":
        return eta
    if link == "log":
        return np.exp(np.clip(eta, -700, 700))
    if link == "inverse":
        with np.errstate(divide="ignore"):
            return 1.0 / eta
    if link == "logit":
        return special.expit(np.clip(eta, -30, 30))
    if link == "probit":
        return np.clip(special.ndtr(eta), P_EPS, 1 - P_EPS)
    if link == "cloglog":
        return np.clip(-np.expm1(-np.exp(np.clip(eta, -700, 50))), P_EPS, 1 - P_EPS)
    raise ValueError(link)


def link_h1(link: str, eta):
    if link == "identity":
        return np.ones_like(eta)
    if link == "log":
        return np.exp(np.clip(eta, -700, 700))
    if link == "inverse":
        with np.errstate(divide="ignore"):
            return -1.0 / eta ** 2
    if link == "logit":
        p = special.expit(np.clip(eta, -30, 30))
        return p * (1 - p)
    if link == "probit":
        return np.exp(-0.5 * eta ** 2) / SQRT_2PI
    if link == "cloglog":
        return np.exp(eta - np.exp(np.clip(eta, -700, 50)))
    raise ValueError(link)


def link_h2(link: str, eta):
    if link == "identity":
        return np.zeros_like(eta)
    if link == "log":
        return np.exp(np.clip(eta, -700, 700))
    if link == "inverse":
        with np.errstate(divide="ignore"):
            return 2.0 / eta ** 3
    if link == "logit":
        p = special.expit(np.clip(eta, -30, 30))
        return p * (1 - p) * (1 - 2 * p)
    if link == "probit":
        return -eta * np.exp(-0.5 * eta ** 2) / SQRT_2PI
    if link == "cloglog":
        return link_h1("cloglog", eta) * (1 - np.exp(np.clip(eta, -700, 50)))
    raise ValueError(link)


# --- Familien: Varianzfunktion V, V', Einheitsdeviance, gültiger Bereich des Mittels ----------------------------------------------------
def variance(family: str, mu, alpha: float = 0.2):
    if family == "normal":
        return np.ones_like(mu)
    if family == "poisson":
        return mu
    if family == "binomial":
        return mu * (1 - mu)
    if family == "gamma":
        return mu ** 2
    if family == "nb2":
        return mu + alpha * mu ** 2
    raise ValueError(family)


def variance_d1(family: str, mu, alpha: float = 0.2):
    if family == "normal":
        return np.zeros_like(mu)
    if family == "poisson":
        return np.ones_like(mu)
    if family == "binomial":
        return 1 - 2 * mu
    if family == "gamma":
        return 2 * mu
    if family == "nb2":
        return 1 + 2 * alpha * mu
    raise ValueError(family)


def unit_deviance(family: str, y, mu, alpha: float = 0.2):
    """Einheitsdeviance je Beobachtung: Normal (y − mu)²; Poisson 2[y ln(y/mu) − (y − mu)]; Bernoulli 2[y ln(y/mu) + (1 − y) ln((1 − y)/(1 − mu))];
    Gamma 2[−ln(y/mu) + (y − mu)/mu]; NB2 2[y ln(y/mu) − (y + 1/alpha) ln((1 + alpha·y)/(1 + alpha·mu))]."""
    if family == "normal":
        return (y - mu) ** 2
    if family == "poisson":
        return 2 * (special.xlogy(y, y / mu) - (y - mu))
    if family == "binomial":
        return 2 * (special.xlogy(y, y / mu) + special.xlogy(1 - y, (1 - y) / (1 - mu)))
    if family == "gamma":
        return 2 * (-np.log(y / mu) + (y - mu) / mu)
    if family == "nb2":
        return 2 * (special.xlogy(y, y / mu) - (y + 1 / alpha) * np.log((1 + alpha * y) / (1 + alpha * mu)))
    raise ValueError(family)


def mu_valid(family: str, mu) -> bool:
    mu = np.asarray(mu)
    if not np.all(np.isfinite(mu)):
        return False
    if family in ("poisson", "gamma", "nb2"):
        return bool(np.all(mu > 0))
    if family == "binomial":
        return bool(np.all((mu > 0) & (mu < 1)))
    return True


def start_mu(family: str, y):
    """Startmittel aus den Daten: binär (y + 0,5)/2, Zählung y + 0,5, sonst y."""
    y = np.asarray(y, float)
    if family == "binomial":
        return (y + 0.5) / 2.0
    if family in ("poisson", "nb2"):
        return y + 0.5
    return y.copy()


@dataclass
class Fit:
    family: str
    link: str
    mode: str
    beta: np.ndarray
    iterations: int
    converged: bool
    mu: np.ndarray
    eta: np.ndarray
    deviance: float
    path: np.ndarray                 # Koeffizienten nach Schritt 0, 1, …
    scores: np.ndarray               # Betragsmaximum des Scores vor jedem Schritt (Länge = iterations)
    steps: np.ndarray                # Betragsmaximum des vollen Schritts je Schritt
    halvings: int = 0                # Zahl der Halbierungen insgesamt
    message: str = ""
    W: np.ndarray = field(default_factory=lambda: np.zeros(0))
    info: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def fit_irls(X, y, family: str, link: str, mode: str = "fisher", tol: float = 1e-10, maxit: int = 100, beta0=None, mu0=None,
             weights=None, offset=None, alpha: float = 0.2, halve: bool = True) -> Fit:
    """Fisher-Scoring (mode="fisher") oder Newton (mode="newton"). Start: `beta0` (Vektor) oder `mu0` (Startmittel; erste Arbeitsantwort z = eta0),
    sonst Startmittel aus den Daten (`start_mu`). Schritthalbierung (halve): wird die Deviance größer oder das Mittel ungültig, wird der Schritt bis zu 30 Mal halbiert."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    off = np.zeros(n) if offset is None else np.asarray(offset, float)

    def dev_of(b):
        mu_b = link_h(link, X @ b + off)
        if not mu_valid(family, mu_b):
            return np.inf
        return float(np.sum(w * unit_deviance(family, y, mu_b, alpha)))

    message = ""
    try:
        if beta0 is None:
            mu = start_mu(family, y) if mu0 is None else np.asarray(mu0, float)
            with np.errstate(all="ignore"):
                eta = link_g(link, mu)
                W = w * link_h1(link, eta) ** 2 / variance(family, mu, alpha)
                beta = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * (eta - off)))
        else:
            beta = np.array(beta0, float)
    except np.linalg.LinAlgError:
        return Fit(family, link, mode, np.full(p, np.nan), 0, False, np.full(n, np.nan), np.full(n, np.nan), np.nan, np.zeros((1, p)), np.zeros(0), np.zeros(0),
                   message="Normalgleichung singulär (Merkmale linear abhängig)")
    path = [beta.copy()]
    scores, steps = [], []
    d_old = dev_of(beta)
    converged = False
    halvings = 0
    iterations = 0
    for iterations in range(1, maxit + 1):
        eta = X @ beta + off
        mu = link_h(link, eta)
        if not mu_valid(family, mu):
            message = "Das Mittel liegt außerhalb des gültigen Bereichs (Startwert oder Schritt unzulässig)."
            iterations -= 1
            break
        with np.errstate(all="ignore"):
            d1 = link_h1(link, eta)
            V = variance(family, mu, alpha)
            W = w * d1 ** 2 / V
            score = X.T @ (w * (y - mu) * d1 / V)
            if mode == "newton":
                extra = w * (y - mu) * (link_h2(link, eta) * V - d1 ** 2 * variance_d1(family, mu, alpha)) / V ** 2
                W = W - extra
        try:
            step = np.linalg.solve(X.T @ (W[:, None] * X), score)
        except np.linalg.LinAlgError:
            message = "Normalgleichung singulär"
            iterations -= 1
            break
        if not np.all(np.isfinite(step)):
            message = "Schritt nicht endlich"
            iterations -= 1
            break
        scores.append(float(np.max(np.abs(score))))
        steps.append(float(np.max(np.abs(step))))
        full = float(np.max(np.abs(step)))
        new = beta + step
        if halve:
            k = 0
            while dev_of(new) > d_old + 1e-10 * (abs(d_old) + 1) and k < 30:
                step = step / 2
                new = beta + step
                k += 1
            halvings += k
        beta = new
        path.append(beta.copy())
        d_old = dev_of(beta)
        if full < tol:
            converged = True
            break
    eta = X @ beta + off
    mu = link_h(link, eta)
    with np.errstate(all="ignore"):
        W = w * link_h1(link, eta) ** 2 / variance(family, mu, alpha)
        info = X.T @ (W[:, None] * X)
        dev = float(np.sum(w * unit_deviance(family, y, mu, alpha))) if mu_valid(family, mu) else float("nan")
    if not converged and not message:
        message = "nicht konvergiert (Höchstzahl der Schritte erreicht)"
    return Fit(family, link, mode, beta, iterations, converged, mu, eta, dev, np.array(path), np.array(scores), np.array(steps), halvings, message, W, info)


def path_difference(a: Fit, b: Fit) -> float:
    """Größter Betragsunterschied der Koeffizienten zwischen zwei Pfaden, Schritt für Schritt (so weit beide reichen)."""
    m = min(len(a.path), len(b.path))
    return float(np.max(np.abs(a.path[:m] - b.path[:m])))


def standard_errors(fit: Fit) -> np.ndarray:
    """Modell-Standardfehler (Dispersion 1) aus der erwarteten Information am Grenzwert."""
    return np.sqrt(np.diag(np.linalg.inv(fit.info)))
