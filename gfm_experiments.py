"""Die Messreihen der Experimente: je Datensatz eine Anpassung (Einheit „Wiederholung“ = ein Datensatz aus n Containern), dann Anteile über die Datensätze.

Dieselben Funktionen rechnen die vorgerechnete Messung (`tools/gfm_make_frozen.py` → `data/gfm_frozen.json`) und die kurze Live-Wiederholung der App; die
Zufalls-Seeds eines Datensatzes folgen aus Messreihe, n und Nummer (`seed_of`). Nur numpy und scipy; der Zufall kommt aus dem exakten SplitMix64 (`gfm_rng`)."""

import numpy as np

import gfm_algorithm as A
import gfm_inference as I
import gfm_scenario as S

LINKS3 = ("identity", "log", "inverse")
BINARY_LINKS = ("probit", "logit", "cloglog")
EXPORT_PUNKTE = (5.5, 15.0, 20.0)       # Vorhersagepunkte (Vorlauf in Tagen); die Daten liegen in [1; 10]
EXP_LINK_EXPORT, EXP_LINK_IMPORT, EXP_SCHWELLE, EXP_MODELL = 1, 2, 3, 4
LIVE_BASE = 700_000                      # Live-Wiederholungen ziehen Datensätze mit Nummern ab hier, nie die der vorgerechneten Messung
MAX_BETA = 15.0                          # eine Anpassung mit größerem Koeffizienten zählt als nicht gültig (Trennung oder keine Konvergenz)


def seed_of(exp: int, n: int, rep: int) -> int:
    return exp * 1_000_000_007 + n * 100_003 + rep


def _gamma_fit(X, y, link):
    f = A.fit_irls(X, y, "gamma", link, tol=1e-9, maxit=60)
    if not f.converged or not np.all(np.isfinite(f.mu)) or np.any(f.mu <= 0):
        return None
    return f


def _predict(link, beta, v_row):
    eta = float(v_row @ beta)
    if link == "inverse" and eta <= 0:
        return float("nan")
    return float(A.link_h(link, np.array([eta]))[0])


# --- Link-Wahl beim Gamma-GLM ---------------------------------------------------------------------------------------------------------------
def link_choice_export(n: int, rep: int, base: int = 0) -> dict:
    """Ein Datensatz Export-Container (Vorlauf gleichverteilt in [1; 10], wahres Mittel linear, wahrer Link Identität): Gamma-GLM mit Achsenabschnitt und Vorlauf
    bei drei Links; AIC je Link (None: Anpassung nicht gültig) und Vorhersage geteilt durch das wahre Mittel an den Punkten EXPORT_PUNKTE."""
    ex = S.draw_export(n, seed_of(EXP_LINK_EXPORT, n, base + rep))
    X = np.column_stack([np.ones(n), ex.vorlauf])
    out = {"aic": {}, "ratio": {}}
    for link in LINKS3:
        f = _gamma_fit(X, ex.d, link)
        if f is None:
            out["aic"][link] = None
            out["ratio"][link] = None
            continue
        out["aic"][link] = I.aic_gamma(ex.d, f.mu, 2)
        out["ratio"][link] = [_predict(link, f.beta, np.array([1.0, v])) / float(S.export_mittel(v)) for v in EXPORT_PUNKTE]
    return out


def link_choice_import(n: int, rep: int, base: int = 0) -> dict:
    """Ein Datensatz Import-Container (nur 0/1-Merkmale Zoll und Wochenende, wahres Mittel multiplikativ, wahrer Link Log): Gamma-GLM mit drei Links;
    AIC je Link und die größte Abweichung μ̂/μ − 1 über die vier Zellen."""
    im = S.draw_import(n, seed_of(EXP_LINK_IMPORT, n, base + rep))
    X = np.column_stack([np.ones(n), im.zoll, im.we])
    cells = np.array([[1, z, w] for z in (0, 1) for w in (0, 1)], float)
    truth = S.import_mittel(cells[:, 1], cells[:, 2])
    out = {"aic": {}, "celldev": {}}
    for link in LINKS3:
        f = _gamma_fit(X, im.d, link)
        if f is None:
            out["aic"][link] = None
            out["celldev"][link] = None
            continue
        out["aic"][link] = I.aic_gamma(im.d, f.mu, 3)
        eta = cells @ f.beta
        if link == "inverse" and np.any(eta <= 0):
            out["celldev"][link] = None
        else:
            out["celldev"][link] = float(np.max(np.abs(A.link_h(link, eta) / truth - 1)))
    return out


def aggregate_link_export(records: list[dict]) -> dict:
    """Anteile, mit denen der AIC einen Link wählt (Nenner: Datensätze mit mindestens einer gültigen Anpassung), Zahl ungültiger Anpassungen, Median des Verhältnisses
    Vorhersage/Wahrheit je Link und Punkt (nur Datensätze mit definierter Vorhersage) und Anteil der Datensätze mit definierter Vorhersage."""
    wins = {l: 0 for l in LINKS3}
    invalid = {l: 0 for l in LINKS3}
    used = 0
    ratios = {l: [[] for _ in EXPORT_PUNKTE] for l in LINKS3}
    for r in records:
        valid = {l: a for l, a in r["aic"].items() if a is not None}
        for l in LINKS3:
            if r["aic"][l] is None:
                invalid[l] += 1
        if not valid:
            continue
        used += 1
        wins[min(valid, key=valid.get)] += 1
        for l in valid:
            for i, q in enumerate(r["ratio"][l]):
                ratios[l][i].append(q)
    med, defined = {}, {}
    for l in LINKS3:
        med[l], defined[l] = [], []
        for i in range(len(EXPORT_PUNKTE)):
            a = np.array(ratios[l][i], float)
            ok = np.isfinite(a)
            med[l].append(float(np.median(a[ok])) if ok.any() else None)
            defined[l].append(float(ok.mean()) if a.size else None)
    log20 = [q for q in ratios["log"][2] if np.isfinite(q)]
    return {"reps": len(records), "used": used, "wins": wins, "invalid": invalid, "median_ratio": med, "defined": defined,
            "log_over25_at_20": float(np.mean(np.array(log20) > 1.25)) if log20 else None}


def aggregate_link_import(records: list[dict]) -> dict:
    wins = {l: 0 for l in LINKS3}
    invalid = {l: 0 for l in LINKS3}
    used = 0
    dev = {l: [] for l in LINKS3}
    for r in records:
        valid = {l: a for l, a in r["aic"].items() if a is not None}
        for l in LINKS3:
            if r["aic"][l] is None:
                invalid[l] += 1
        if not valid:
            continue
        used += 1
        wins[min(valid, key=valid.get)] += 1
        for l in valid:
            if r["celldev"][l] is not None:
                dev[l].append(r["celldev"][l])
    return {"reps": len(records), "used": used, "wins": wins, "invalid": invalid,
            "median_celldev": {l: (float(np.median(dev[l])) if dev[l] else None) for l in LINKS3}}


# --- Residualdeviance beim Standgeld (Import) ------------------------------------------------------------------------------------------------
def _binary_fit(X, y, link):
    f = A.fit_irls(X, y, "binomial", link, tol=1e-9, maxit=60)
    if not f.converged or np.max(np.abs(f.beta)) > MAX_BETA or not np.all(np.isfinite(f.mu)):
        return None
    return f


def _fit_record(n, y, X, f, key48, key4):
    p = np.clip(f.mu, A.P_EPS, 1 - A.P_EPS)
    rec = {"p": X.shape[1], "du": float(-2.0 * I.loglik_bernoulli(y, p)), "xu": I.pearson_x2("binomial", y, p), "ll": I.loglik_bernoulli(y, p)}
    g48 = I.grouped_binomial(key48, y, p, 48)
    rec.update({"dg48": g48["dev"], "cells48": g48["cells"], "small48": g48["small"], "xg48": g48["x2"], "llsat48": g48["llsat"]})
    if X.shape[1] == 3:
        g4 = I.grouped_binomial(key4, y, p, 4)
        rec.update({"dg4": g4["dev"], "cells4": g4["cells"], "small4": g4["small"]})
    return rec


def standgeld_dataset(n: int, rep: int, welt_mit_kunden: bool, base: int = 0, exp: int = EXP_MODELL, links=BINARY_LINKS, tage: float = S.STANDGELD_TAGE) -> dict:
    """Ein Datensatz Import-Container, Standgeld = Verweilzeit über `tage` Tage (Standard 7). Modell M0 (Achsenabschnitt, Zoll, Wochenende; p = 3) und M1 (zusätzlich elf
    Kundenmerkmale; p = 14) mit drei Links. Zellen: Zoll × Wochenende × Kunde (48) bzw. Zoll × Wochenende (4)."""
    im = S.draw_import(n, seed_of(exp, n, base + rep), kundeneffekt=welt_mit_kunden)
    y = im.standgeld(tage)
    X1 = np.column_stack([np.ones(n), im.zoll, im.we, S.kunden_dummies(im.kunde)])
    X0 = X1[:, :3]
    key48 = im.zoll * 24 + im.we * 12 + im.kunde
    key4 = im.zoll * 2 + im.we
    out = {"_ent": I.mean_entropy_x2(im.wahre_wahrscheinlichkeit(tage)), "_prev": float(y.mean())}   # Schlüssel mit _ sind Beigaben, keine Anpassungen
    for name, X in (("M0", X0), ("M1", X1)):
        for link in links:
            f = _binary_fit(X, y, link)
            out[f"{name}/{link}"] = None if f is None else _fit_record(n, y, X, f, key48, key4)
    return out


def _share(k, m):
    return None if m == 0 else k / m


def aggregate_standgeld(records: list[dict], model: str, link: str) -> dict:
    """Verwerfungsanteile (Niveau 5 %) der Tests für ein Modell und einen Link über die Datensätze mit gültiger Anpassung; Nenner je Test getrennt, wo die
    Freiheitsgrade nicht reichen. Rückgabe auch die mittlere Deviance je Freiheitsgrad und den Anteil der Zellen mit erwarteter Zahl unter 5."""
    key = f"{model}/{link}"
    rows = [r[key] for r in records if r.get(key) is not None]
    m = len(rows)
    if m == 0:
        return {"reps": len(records), "valid": 0}
    out = {"reps": len(records), "valid": m}
    p = rows[0]["p"]
    out["p"] = p   # die Freiheitsgrade (n − p bzw. Zellen − p) bildet `summarize_standgeld`, das n kennt
    out["du"] = [r["du"] for r in rows]
    out["xu"] = [r["xu"] for r in rows]
    out["dg48"] = [r["dg48"] for r in rows]
    out["xg48"] = [r["xg48"] for r in rows]
    out["cells48"] = [r["cells48"] for r in rows]
    out["small48"] = float(np.mean([r["small48"] for r in rows]))
    if p == 3:
        out["dg4"] = [r["dg4"] for r in rows]
        out["small4"] = float(np.mean([r["small4"] for r in rows]))
    return out


def summarize_standgeld(agg: dict, n: int) -> dict:
    """Aus den Listen von `aggregate_standgeld` die Kennzahlen: mittlere Deviance je Freiheitsgrad, Verwerfungsanteile (über 95-%-Wert und unter 5-%-Wert)."""
    if agg.get("valid", 0) == 0:
        return {"reps": agg["reps"], "valid": 0}
    p = agg["p"]
    dfu = n - p
    crit_hi, crit_lo = I.chi2_crit(dfu, 0.95), I.chi2_crit(dfu, 0.05)
    du = np.array(agg["du"])
    xu = np.array(agg["xu"])
    df48 = np.array(agg["cells48"]) - p
    dg48 = np.array(agg["dg48"])
    ok48 = df48 >= 1
    rej48 = [d > I.chi2_crit(int(f), 0.95) for d, f in zip(dg48[ok48], df48[ok48])]
    out = {"reps": agg["reps"], "valid": agg["valid"], "du_per_df": float(np.mean(du) / dfu), "rej_du": float(np.mean(du > crit_hi)), "below_du": float(np.mean(du < crit_lo)),
           "rej_xu": float(np.mean(xu > crit_hi)), "dg48_per_df": float(np.mean(dg48[ok48] / df48[ok48])) if ok48.any() else None,
           "rej_dg48": float(np.mean(rej48)) if rej48 else None, "valid_g48": int(ok48.sum()), "small48": agg["small48"]}
    if p == 3:
        dg4 = np.array(agg["dg4"])
        out["rej_dg4"] = float(np.mean(dg4 > I.chi2_crit(1, 0.95)))
        out["small4"] = agg["small4"]
    return out


def lq_rejection(records: list[dict], link: str) -> dict:
    """Likelihood-Quotienten-Test M0 gegen M1 (11 Freiheitsgrade): Anteil der Verwerfungen über die Datensätze, in denen beide Anpassungen gültig sind."""
    crit = I.chi2_crit(11, 0.95)
    stats_ = [r[f"M0/{link}"]["du"] - r[f"M1/{link}"]["du"] for r in records if r.get(f"M0/{link}") is not None and r.get(f"M1/{link}") is not None]
    return {"valid": len(stats_), "reps": len(records), "rej": float(np.mean(np.array(stats_) > crit)) if stats_ else None}


def aic_choice_binary(records: list[dict]) -> dict:
    """AIC-Wahl zwischen Probit, Logit und cloglog beim Modell M1 (gleiche Parameterzahl): Anteile über die Datensätze mit drei gültigen Anpassungen."""
    wins = {l: 0 for l in BINARY_LINKS}
    used = 0
    for r in records:
        recs = {l: r.get(f"M1/{l}") for l in BINARY_LINKS}
        if any(v is None for v in recs.values()):
            continue
        used += 1
        wins[min(wins, key=lambda l: -2 * recs[l]["ll"] + 2 * recs[l]["p"])] += 1
    return {"used": used, "wins": wins}


def schwelle_dataset(n: int, rep: int, thresholds, base: int = 0) -> dict:
    """Ein Datensatz Import-Container ohne Kundeneffekt (das Modell mit Zoll und Wochenende ist exakt richtig, Probit); für jede Schwelle c (Tage) Standgeld =
    Verweilzeit über c, Modell M0 mit Probit-Link. Alle Schwellen nutzen denselben Datensatz."""
    im = S.draw_import(n, seed_of(EXP_SCHWELLE, n, base + rep), kundeneffekt=False)
    X0 = np.column_stack([np.ones(n), im.zoll, im.we])
    key4 = im.zoll * 2 + im.we
    out = {}
    for c in thresholds:
        y = im.standgeld(c)
        f = _binary_fit(X0, y, "probit")
        if f is None:
            out[c] = None
            continue
        p = np.clip(f.mu, A.P_EPS, 1 - A.P_EPS)
        g4 = I.grouped_binomial(key4, y, p, 4)
        out[c] = {"du": float(-2.0 * I.loglik_bernoulli(y, p)), "dg4": g4["dev"], "cells4": g4["cells"], "prev": float(y.mean()),
                  "ent": I.mean_entropy_x2(im.wahre_wahrscheinlichkeit(c))}
    return out


def summarize_schwelle(records: list[dict], c: float, n: int) -> dict:
    rows = [r[c] for r in records if r.get(c) is not None and r[c]["cells4"] - 3 >= 1]
    m = len(rows)
    du = np.array([r["du"] for r in rows])
    dfu = n - 3
    return {"c": c, "valid": m, "reps": len(records), "prev": float(np.mean([r["prev"] for r in rows])), "ent2": float(np.mean([r["ent"] for r in rows])),
            "du_per_df": float(np.mean(du) / dfu), "rej_hi": float(np.mean(du > I.chi2_crit(dfu, 0.95))), "below_lo": float(np.mean(du < I.chi2_crit(dfu, 0.05))),
            "rej_g4": float(np.mean([r["dg4"] > I.chi2_crit(1, 0.95) for r in rows]))}
