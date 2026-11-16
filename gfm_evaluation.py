"""Auswertung der Experimente und Verdikt-Logik. Die Verdikte sind reine Funktionen (Art, Text) auf Zeilen: die App ruft `getattr(st, Art)(Text)`, `tests/test_verdicts.py`
prüft Schwellen und jede Zahl im Text an handgesetzten Zeilen.

Zähleinheit überall: Wiederholung = ein Datensatz aus n Containern, wo nicht „Container“ steht. Anteile über Datensätze sind Bänder (Monte-Carlo-Fehler), keine exakten Prozente."""

import math

import numpy as np

import gfm_algorithm as A
import gfm_constants as C
import gfm_experiments as E
import gfm_frozen as F
import gfm_inference as I
import gfm_reference as REF
import gfm_scenario as S


# ======================================================================================================================================
# Daten der Experimente 1 und 2
# ======================================================================================================================================
def pair_info(key: str) -> tuple:
    for p in C.PAIRS:
        if p[0] == key:
            return p
    raise KeyError(key)


def dataset_for(daten: str, ausgang: str, seed: int):
    """(X, y, Merkmalsnamen, Tabelle oder None) für „hand4“, „hand9“ oder „tabelle“ und den Ausgang „stetig“, „zähl“ oder „binär“."""
    if daten == "tabelle":
        T = S.draw_table(3000, seed)
        y = {"stetig": T.d, "zähl": T.standtage(), "binär": T.standgeld()}[ausgang]
        return T.designmatrix(), y, S.TABLE_FEATURES, T
    h = S.HAND_VIER if daten == "hand4" else S.HAND_NEUN
    y = {"stetig": h.d, "zähl": h.standtage, "binär": h.standgeld}[ausgang]
    names = ("Achsenabschnitt",) if daten == "hand4" else ("Achsenabschnitt", "Zollhalt")
    return h.X[:, : len(names)], y, names, None


def start_beta(start: str, p: int):
    if start == "daten":
        return None
    return np.r_[5.0, np.zeros(p - 1)] if start == "fuenf" else np.zeros(p)


# ======================================================================================================================================
# Experiment 1: ein Code, neun Paare
# ======================================================================================================================================
def run_exp1(daten: str, paar: str, regel: str, start: str, halb: bool, seed: int) -> dict:
    _, aus, fam, link = pair_info(paar)
    X, y, names, _ = dataset_for(daten, aus, seed)
    b0 = start_beta(start, X.shape[1])
    fits = {}
    for mode in ("fisher", "newton"):
        if regel in (mode, "beide"):
            fits[mode] = A.fit_irls(X, y, fam, link, mode, tol=1e-10, maxit=1000, beta0=b0, alpha=C.NB2_ALPHA, halve=halb)
    out = {"paar": paar, "daten": daten, "ausgang": aus, "family": fam, "link": link, "names": names, "n": X.shape[0], "fits": fits, "canonical": A.is_canonical(fam, link)}
    good = [f for f in fits.values() if f.converged]
    out["limit"] = (good[-1].beta if good else None)
    out["distance"] = {}
    for mode, f in fits.items():
        if out["limit"] is not None:
            d = np.max(np.abs(f.path - out["limit"]), axis=1)
            out["distance"][mode] = d
    if len(fits) == 2:
        out["path_diff"] = A.path_difference(fits["fisher"], fits["newton"])
        if all(f.converged for f in fits.values()):
            out["limit_diff"] = float(np.max(np.abs(fits["fisher"].beta - fits["newton"].beta)))
    return out


def reference_for(daten: str, paar: str, seed: int) -> dict:
    """Abstand des IRLS-Grenzwerts zur unabhängigen Minimierung und Gradientenprobe (Standardstart, Fisher)."""
    _, aus, fam, link = pair_info(paar)
    X, y, _, _ = dataset_for(daten, aus, seed)
    f = A.fit_irls(X, y, fam, link, "newton", tol=1e-10, maxit=300, alpha=C.NB2_ALPHA)
    if not f.converged:
        return {"ok": False}
    ref = REF.minimize_reference(fam, link, X, y, f.beta, C.NB2_ALPHA, starts=3)
    return {"ok": True, **{k: ref[k] for k in ("spread", "dist", "grad")}}


def nine_pairs(seed: int) -> list[dict]:
    """Die neun Paare auf der Container-Tabelle (n = 3 000): Schritte bei zwei Toleranzen, Pfadunterschied, Unterschied der Grenzwerte, Gradientenprobe, Zoll-Koeffizient."""
    T = S.draw_table(3000, seed)
    X = T.designmatrix()
    ys = {"stetig": T.d, "zähl": T.standtage(), "binär": T.standgeld()}
    rows = []
    for key, aus, fam, link in C.PAIRS:
        y = ys[aus]
        kw = {"alpha": C.NB2_ALPHA}
        f8 = A.fit_irls(X, y, fam, link, "fisher", tol=1e-8, maxit=1000, **kw)
        f10 = A.fit_irls(X, y, fam, link, "fisher", tol=1e-10, maxit=1000, **kw)
        n8 = A.fit_irls(X, y, fam, link, "newton", tol=1e-8, maxit=1000, **kw)
        n10 = A.fit_irls(X, y, fam, link, "newton", tol=1e-10, maxit=1000, **kw)
        ok = f10.converged and n10.converged
        rows.append({
            "key": key, "ausgang": aus, "family": fam, "link": link, "canonical": A.is_canonical(fam, link), "ok": ok,
            "fisher8": f8.iterations, "fisher10": f10.iterations, "newton8": n8.iterations, "newton10": n10.iterations,
            "path_diff": A.path_difference(f10, n10), "limit_diff": float(np.max(np.abs(f10.beta - n10.beta))) if ok else None,
            "grad": float(np.max(np.abs(REF.gradient(fam, link, X, y, f10.beta, C.NB2_ALPHA)))) if f10.converged else None,
            "zoll": float(f10.beta[4]) if f10.converged else None, "mu_min": float(np.min(f10.mu)) if f10.converged else None,
        })
    return rows


# ======================================================================================================================================
# Experiment 2: der Link ist die Skala
# ======================================================================================================================================
def _fit_row(X, y, fam, link, names, T):
    f = A.fit_irls(X, y, fam, link, "fisher", tol=1e-10, maxit=300, alpha=C.NB2_ALPHA)
    row = {"family": fam, "link": link, "ok": f.converged and A.mu_valid(fam, f.mu), "iterations": f.iterations}
    if not row["ok"]:
        row["message"] = f.message or "Anpassung nicht gültig (das angepasste Mittel verlässt den zulässigen Bereich)"
        return row
    row["beta"] = f.beta
    row["zoll"] = float(f.beta[names.index("Zoll")] if "Zoll" in names else f.beta[names.index("Zollhalt")])
    row["we"] = float(f.beta[names.index("Wochenende")]) if "Wochenende" in names else None
    row["mu_min"] = float(np.min(f.mu))
    row["mu"] = f.mu
    if T is not None:
        row["bias_art"] = [100.0 * (float(f.mu[T.art == a].sum() / y[T.art == a].sum()) - 1.0) for a in range(4)]
        row["sum_resid"] = float(np.sum(y - f.mu))
    else:
        row["group_means"] = (float(f.mu[X[:, 1] == 0][0]), float(f.mu[X[:, 1] == 1][0])) if X.shape[1] > 1 else (float(f.mu[0]), None)
    return row


def run_exp2(daten: str, ausgang: str, familie: str, link: str, seed: int) -> dict:
    X, y, names, T = dataset_for(daten, ausgang, seed)
    return {"daten": daten, "ausgang": ausgang, "names": names, "n": X.shape[0],
            "links": [_fit_row(X, y, familie, lk, names, T) for lk in C.E2_LINKS],
            "families": [_fit_row(X, y, fm, link, names, T) for fm in C.E2_FAMILIEN]}


def scale_text(link: str, beta_zoll: float) -> str:
    if link == "log":
        return f"Faktor {num(math.exp(beta_zoll))} auf das Mittel"
    if link == "identity":
        return f"{beta_zoll:+.3f} Tage".replace(".", ",")
    return "auf der Kehrwert-Skala, kein einfacher Faktor"


# ======================================================================================================================================
# Experiment 3: Link-Wahl (live: ein Datensatz; vorgerechnet: Anteile über Datensätze)
# ======================================================================================================================================
def run_exp3_live(welt: str, n: int, seed: int) -> dict:
    if welt == "export":
        ex = S.draw_export(n, seed)
        X = np.column_stack([np.ones(n), ex.vorlauf])
        grid = np.linspace(0.0, 25.0, 126)
        out = {"welt": welt, "n": n, "data": (ex.vorlauf, ex.d), "grid": grid, "truth": S.export_mittel(grid), "fits": {}}
        for link in E.LINKS3:
            f = E._gamma_fit(X, ex.d, link)
            if f is None:
                out["fits"][link] = None
                continue
            eta = f.beta[0] + f.beta[1] * grid
            with np.errstate(all="ignore"):
                curve = A.link_h(link, eta)
            if link == "inverse":
                curve = np.where(eta > 0, curve, np.nan)
            pts = [E._predict(link, f.beta, np.array([1.0, v])) / float(S.export_mittel(v)) for v in C.E3_PUNKTE]
            out["fits"][link] = {"aic": I.aic_gamma(ex.d, f.mu, 2), "beta": f.beta, "curve": curve, "ratio": pts}
        return out
    im = S.draw_import(n, seed)
    X = np.column_stack([np.ones(n), im.zoll, im.we])
    cells = np.array([[1, z, w] for z in (0, 1) for w in (0, 1)], float)
    truth = S.import_mittel(cells[:, 1], cells[:, 2])
    out = {"welt": welt, "n": n, "cells": cells, "truth": truth, "fits": {}}
    for link in E.LINKS3:
        f = E._gamma_fit(X, im.d, link)
        if f is None:
            out["fits"][link] = None
            continue
        eta = cells @ f.beta
        mu = A.link_h(link, eta) if not (link == "inverse" and np.any(eta <= 0)) else np.full(4, np.nan)
        out["fits"][link] = {"aic": I.aic_gamma(im.d, f.mu, 3), "beta": f.beta, "cell_mu": mu, "ratio": list(mu / truth)}
    return out


def aic_winner(live: dict):
    valid = {l: f["aic"] for l, f in live["fits"].items() if f is not None}
    return min(valid, key=valid.get) if valid else None


def frozen_link_row(welt: str, n: int) -> dict | None:
    fz = F.load()
    for row in fz["link_export" if welt == "export" else "link_import"]:
        if row["n"] == n:
            return row
    return None


def live_repeat_exp3(welt: str, n: int, reps: int, seed: int) -> dict:
    base = E.LIVE_BASE + 1000 * seed
    if welt == "export":
        return E.aggregate_link_export([E.link_choice_export(n, r, base) for r in range(reps)])
    return E.aggregate_link_import([E.link_choice_import(n, r, base) for r in range(reps)])


def live_reps_limit(n: int) -> int:
    """Höchstzahl der Live-Wiederholungen: Datensätze × Container ≤ LIVE_BUDGET (mindestens LIVE_REPS_MIN, höchstens 200)."""
    return int(max(C.LIVE_REPS_MIN, min(200, C.LIVE_BUDGET // max(n, 1))))


# ======================================================================================================================================
# Experiment 4 und 5: Residualdeviance, gruppiert, Likelihood-Quotient
# ======================================================================================================================================
def run_exp4_live(c: int, n: int, welt: str, modell: str, seed: int) -> dict:
    """Ein Datensatz Import-Container, Standgeld ab `c` Tagen, Probit-Anpassung (der wahre Link) mit Modell M0 (p = 3) oder M1 (p = 14)."""
    im = S.draw_import(n, seed, kundeneffekt=(welt == "mit"))
    y = im.standgeld(c)
    X1 = np.column_stack([np.ones(n), im.zoll, im.we, S.kunden_dummies(im.kunde)])
    X = X1 if modell == "M1" else X1[:, :3]
    out = {"c": c, "n": n, "welt": welt, "modell": modell, "p": X.shape[1], "prevalence": float(y.mean()), "ent2": I.mean_entropy_x2(im.wahre_wahrscheinlichkeit(c))}
    f = E._binary_fit(X, y, "probit")
    if f is None:
        out["ok"] = False
        return out
    p = np.clip(f.mu, A.P_EPS, 1 - A.P_EPS)
    du = float(-2.0 * I.loglik_bernoulli(y, p))
    dfu = n - X.shape[1]
    out.update({"ok": True, "du": du, "dfu": dfu, "crit_hi": I.chi2_crit(dfu, 0.95), "crit_lo": I.chi2_crit(dfu, 0.05), "xu": I.pearson_x2("binomial", y, p)})
    g48 = I.grouped_binomial(im.zoll * 24 + im.we * 12 + im.kunde, y, p, 48)
    out["g48"] = {**g48, "df": g48["cells"] - X.shape[1]}
    if X.shape[1] == 3:
        g4 = I.grouped_binomial(im.zoll * 2 + im.we, y, p, 4)
        out["g4"] = {**g4, "df": g4["cells"] - 3}
    # Beziehung ungruppiert = gruppiert − 2·(gesättigte Log-Likelihood der Zellen) (zur Anzeige)
    out["relation_gap"] = abs(du - (g48["dev"] - 2.0 * g48["llsat"]))
    return out


def live_repeat_exp4(c: int, n: int, welt: str, modell: str, reps: int, seed: int) -> dict:
    base = E.LIVE_BASE + 1000 * seed
    recs = [E.standgeld_dataset(n, r, welt == "mit", base=base, exp=E.EXP_MODELL + 10, links=("probit",), tage=float(c)) for r in range(reps)]
    s = E.summarize_standgeld(E.aggregate_standgeld(recs, modell, "probit"), n)
    s["ent2"] = float(np.mean([r["_ent"] for r in recs]))
    s["prev"] = float(np.mean([r["_prev"] for r in recs]))
    return s


def frozen_deviance_row(c: int, n: int, welt: str, modell: str) -> dict | None:
    """Die vorgerechnete Zeile zu den Einstellungen, falls es sie gibt: die Schwellenreihe (Welt ohne Kundeneffekt, Modell M0, n = 1 050) oder die Modelltabelle (Schwelle 7)."""
    fz = F.load()
    if welt == "ohne" and modell == "M0" and n == fz["schwelle"]["n"]:
        for row in fz["schwelle"]["rows"]:
            if row["c"] == c:
                return {"source": "schwelle", **row}
    if c == int(C.STANDGELD_TAGE) and str(n) in fz["modell"][welt]:
        return {"source": "modell", **fz["modell"][welt][str(n)][f"{modell}/probit"]}
    return None


def exp5_rows(n: int, link: str) -> list[dict]:
    """Verwerfungsanteile der Tests (Niveau 5 %) für vier Szenarien bei Standgeld ab 7 Tagen, Link `link`, n Containern (vorgerechnet)."""
    fz = F.load()["modell"]
    mit, ohne = fz["mit"][str(n)], fz["ohne"][str(n)]
    return [
        {"name": "Kunden fehlen im Modell (Welt mit Kundeneffekt, Modell M0 falsch)", "welt": "mit", "modell": "M0", "falsch": True, "s": mit[f"M0/{link}"], "lq": mit["lq"][link]},
        {"name": "Welt mit Kundeneffekt, Modell M1 (richtig)", "welt": "mit", "modell": "M1", "falsch": False, "s": mit[f"M1/{link}"], "lq": None},
        {"name": "Welt ohne Kundeneffekt, Modell M0 (richtig)", "welt": "ohne", "modell": "M0", "falsch": False, "s": ohne[f"M0/{link}"], "lq": ohne["lq"][link]},
        {"name": "Welt ohne Kundeneffekt, Modell M1 (richtig, mit überzähligen Merkmalen)", "welt": "ohne", "modell": "M1", "falsch": False, "s": ohne[f"M1/{link}"], "lq": None},
    ]


# ======================================================================================================================================
# Experiment 6: wo der Rahmen reißt
# ======================================================================================================================================
def separation_path(steps=(1, 2, 3, 5, 10, 20, 40)) -> list[dict]:
    """Logit mit vollständig getrennten Daten (x = −2, −1, 1, 2; y = 0, 0, 1, 1) und Achsenabschnitt: Koeffizient und Deviance nach k IRLS-Schritten.
    Es gibt keinen Grenzwert: die Koeffizienten wachsen mit jedem Schritt, bis die Rechengrenze |η| ≤ 30 greift."""
    X = np.array([[1, -2], [1, -1], [1, 1], [1, 2]], float)
    y = np.array([0, 0, 1, 1], float)
    rows = []
    for k in steps:
        f = A.fit_irls(X, y, "binomial", "logit", "fisher", tol=0.0, maxit=k, beta0=[0.0, 0.0], halve=False)
        rows.append({"k": k, "beta1": float(f.beta[1]), "deviance": float(f.deviance), "converged": f.converged})
    return rows


def nb2_alpha_live(seed: int) -> dict:
    """NB2-α auf den Standtagen der Container-Tabelle: Profil-Außenschleife gegen abwechselnde Schätzung."""
    T = S.draw_table(3000, seed)
    X, y = T.designmatrix(), T.standtage()
    prof = I.nb2_profile_alpha(X, y)
    alt = I.nb2_alternating(X, y)
    return {"alpha_profile": prof["alpha"], "alpha_alternating": alt["alpha"], "outer": alt["outer"], "mean": float(y.mean()), "var": float(y.var(ddof=1))}


# ======================================================================================================================================
# Handbeispiel: die Zahlen, die die App zeigt (der Test rechnet sie in Brüchen nach)
# ======================================================================================================================================
def hand_logit_limit() -> tuple:
    """Grenzwert des Logit-Modells auf dem Standgeld der neun Container in geschlossener Form: (ln 1/3; ln 12) und die Deviance 10 ln 5 − 6 ln 3."""
    return (math.log(1 / 3), math.log(12)), 10 * math.log(5) - 6 * math.log(3)


# ======================================================================================================================================
# Verdikte
# ======================================================================================================================================
def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f} %".replace(".", ",")


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def sci(x: float, digits: int = 1) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma: 6,0·10⁻¹¹."""
    if x == 0:
        return "0"
    mant, exp = f"{x:.{digits}e}".split("e")
    sup = str(int(exp)).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mant.replace('.', ',')}·10{sup}"


def ntxt(n: int) -> str:
    """Tausendergruppe mit Leerzeichen: 3 000."""
    return f"{n:,}".replace(",", " ")


LINK_ADJ = C.LINK_ADJ


def verdict_exp1(res: dict, ref: dict | None) -> tuple[str, str]:
    """Verdikt zu Experiment 1 aus dem Ergebnis von `run_exp1` und der Referenz (oder None)."""
    fits = res["fits"]
    label = f"{C.FAMILY_LABEL[res['family']]} mit {LINK_ADJ[res['link']]}"
    if not fits or any(not f.converged for f in fits.values()):
        bad = [m for m, f in fits.items() if not f.converged]
        return "warning", (f"{label}: {', '.join('Fisher' if m == 'fisher' else 'Newton' for m in bad)} konvergiert nicht "
                           f"({'; '.join(sorted({fits[m].message for m in bad if fits[m].message}))}). Eine Nicht-Konvergenz wird gemeldet, nicht verschwiegen: ohne Schritthalbierung kann ein Schritt das Mittel aus dem zulässigen Bereich tragen.")
    parts = []
    if "fisher" in fits:
        parts.append(f"Fisher-Scoring {fits['fisher'].iterations} Schritte")
    if "newton" in fits:
        parts.append(f"Newton {fits['newton'].iterations} Schritte")
    text = f"{label}, n = {ntxt(res['n'])} Container: " + ", ".join(parts) + " (Toleranz 10⁻¹⁰ am vollen Schritt; der letzte Schritt zeigt nur, dass der Schritt unter der Toleranz liegt)."
    if "path_diff" in res:
        if res["canonical"]:
            text += f" Der Link ist kanonisch: der Pfad von Fisher-Scoring und Newton ist derselbe (größter Unterschied {sci(res['path_diff'])})."
        else:
            text += f" Der Link ist nicht kanonisch: die Pfade sind verschieden (größter Unterschied {num(res['path_diff'], 2)}), der Grenzwert ist derselbe (Unterschied {sci(res.get('limit_diff', float('nan')))})."
    if ref and ref.get("ok"):
        text += f" Eine unabhängige Minimierung der Likelihood aus scipy.stats landet {sci(ref['dist'])} daneben (Streuung ihrer Startpunkte {sci(ref['spread'])}); Gradient der unabhängigen Likelihood am IRLS-Ergebnis {sci(ref['grad'])}."
    return "success", text


def verdict_exp2(res: dict) -> tuple[str, str]:
    """Verdikt zu Experiment 2: Mittel und Koeffizienten bei drei Links (Spalte `links`)."""
    rows = [r for r in res["links"] if r["ok"]]
    if res["daten"] == "hand9" and len(rows) == len(res["links"]):
        means = [r["group_means"] for r in rows]
        same = all(abs(m[0] - means[0][0]) < 1e-6 and abs(m[1] - means[0][1]) < 1e-6 for m in means)
        parts = "; ".join(f"{C.LINK_LABEL[r['link']]}: Zoll {r['zoll']:+.4f}".replace(".", ",") for r in rows)
        if same:
            return "success", (f"Zwei Gruppen sind ein gesättigtes Modell: alle drei Links geben dieselben angepassten Mittel ({num(means[0][0])} und {num(means[0][1])} Tage, die Gruppenmittel), "
                               f"aber verschiedene Koeffizienten ({parts}). Der Link ändert die Skala, auf der man den Effekt liest, nicht die Anpassung.")
        return "info", f"Die angepassten Mittel unterscheiden sich zwischen den Links ({parts}); das gesättigte Modell der zwei Gruppen gilt nur beim Gamma-Modell mit Achsenabschnitt und Zollhalt."
    bad = [r for r in res["links"] if not r["ok"]]
    parts = "; ".join(f"{C.LINK_LABEL[r['link']]}: Zoll {r['zoll']:+.4f}".replace(".", ",") + f" ({scale_text(r['link'], r['zoll'])})" for r in rows)
    text = f"Container-Tabelle (n = {ntxt(res['n'])}, ein Datensatz): {parts}."
    if bad:
        text += " Nicht gültig: " + ", ".join(C.LINK_LABEL[r["link"]] for r in bad) + " (das angepasste Mittel verlässt den zulässigen Bereich)."
    mins = [(r["link"], r["mu_min"]) for r in rows if r["mu_min"] <= 0]
    if mins:
        text += " Das kleinste angepasste Mittel ist negativ bei: " + ", ".join(f"{C.LINK_LABEL[l]} ({num(m, 2)})" for l, m in mins) + "."
    return "info", text


def verdict_exp3(welt: str, n: int, live: dict, row: dict | None) -> tuple[str, str]:
    win = aic_winner(live)
    wahr = "identity" if welt == "export" else "log"
    kind = "info"
    text = f"Dieser Datensatz (n = {ntxt(n)}): "
    text += f"der AIC wählt den Link {C.LINK_LABEL[win]}" + (" (der wahre)." if win == wahr else f" (der wahre ist {C.LINK_LABEL[wahr]}).") if win else "keine gültige Anpassung."
    if row:
        w = row["wins"]
        used = row["used"]
        text += (f" Über {ntxt(row['reps'])} vorgerechnete Datensätze gleicher Größe wählt der AIC Identität / Log / Kehrwert in "
                 f"{pct(w['identity'] / used)} / {pct(w['log'] / used)} / {pct(w['inverse'] / used)} (wahrer Link: {C.LINK_LABEL[wahr]}; "
                 f"{'Export, Vorlauf in [1; 10]' if welt == 'export' else 'Import, nur 0/1-Merkmale'}).")
    return kind, text


def verdict_exp4(live: dict, frozen: dict | None) -> tuple[str, str]:
    if not live.get("ok"):
        return "warning", "Dieser Datensatz hat keine gültige Probit-Anpassung (Trennung oder keine Konvergenz); die Zahl solcher Datensätze zählt die vorgerechnete Messung mit."
    text = (f"Dieser Datensatz (n = {ntxt(live['n'])}, Standgeld ab {live['c']} Tagen, Anteil {pct(live['prevalence'])}, p = {live['p']}): Deviance {num(live['du'], 1)}, "
            f"D/(n − p) = {num(live['du'] / live['dfu'])}; 2·mittlere Entropie der wahren Wahrscheinlichkeiten {num(live['ent2'])}. Der Test gegen χ² mit {live['dfu']} Freiheitsgraden "
            f"{'lehnt nicht ab' if live['du'] <= live['crit_hi'] else 'lehnt ab'} (95-%-Wert {num(live['crit_hi'], 1)}).")
    modell_falsch = live.get("welt") == "mit" and live.get("modell") == "M0"          # nur in der Welt mit Kundeneffekt fehlt dem Modell M0 etwas
    if live["du"] > live["crit_hi"] and not modell_falsch:
        text += " Das Modell ist in dieser Welt richtig: die Verwerfung ist ein Fehlalarm."
    elif live["du"] <= live["crit_hi"] and modell_falsch:
        text += " Das Modell ist hier falsch (die Kunden fehlen); die ungruppierte Deviance merkt es in diesem Datensatz nicht."
    if frozen and frozen.get("source") == "schwelle":
        text += (f" Vorgerechnet über {frozen['valid']} Datensätze (Welt ohne Kundeneffekt, Modell exakt richtig): über dem 95-%-Wert in {pct(frozen['rej_hi'])}, unter dem 5-%-Wert in "
                 f"{pct(frozen['below_lo'])}; gruppiert (4 Zellen) über dem 95-%-Wert in {pct(frozen['rej_g4'])}.")
    elif frozen:
        text += (f" Vorgerechnet über {frozen['valid']} Datensätze: ungruppiert über dem 95-%-Wert in {pct(frozen['rej_du'])}, unter dem 5-%-Wert in {pct(frozen['below_du'])}"
                 + (f"; gruppiert (48 Zellen) über dem 95-%-Wert in {pct(frozen['rej_dg48'])}." if frozen.get("rej_dg48") is not None else "."))
    return "info", text


def verdict_exp5(rows: list[dict], n: int, link: str) -> tuple[str, str]:
    falsch = next(r for r in rows if r["falsch"])
    s, lq = falsch["s"], falsch["lq"]
    richtig = next(r for r in rows if r["welt"] == "ohne" and r["modell"] == "M0")
    text = (f"Kunden fehlen im Modell (n = {ntxt(n)}, {LINK_ADJ[link]}, {s['valid']} gültige von {s['reps']} Datensätzen): die ungruppierte Deviance verwirft in {pct(s['rej_du'])}, "
            f"die gruppierte (48 Zellen) in {pct(s['rej_dg48'])}, der Likelihood-Quotienten-Test in {pct(lq['rej'])}. "
            f"Richtiges Modell (Welt ohne Kundeneffekt, Modell M0): Likelihood-Quotienten-Test {pct(richtig['lq']['rej'])}, gruppiert (4 Zellen) {pct(richtig['s']['rej_dg4'])}, "
            f"gruppiert (48 Zellen) {pct(richtig['s']['rej_dg48'])} bei {pct(richtig['s']['small48'], 0)} Zellen mit erwarteter Zahl unter 5.")
    return "info", text


def verdict_exp6(rows: list[dict], alpha: dict) -> tuple[str, str]:
    last = rows[-1]
    return "info", (f"Separation (Logit, vier Container): der Koeffizient wächst von Schritt zu Schritt (nach {rows[0]['k']} Schritt {num(rows[0]['beta1'], 2)}, nach {last['k']} Schritten {num(last['beta1'], 2)}) "
                    f"und hat keinen Grenzwert. NB2-α: Profil-Außenschleife {num(alpha['alpha_profile'], 4)}, abwechselnde Schätzung {num(alpha['alpha_alternating'], 4)}, "
                    f"auf den Standtagen der Container-Tabelle (Mittel {num(alpha['mean'], 2)}, Varianz {num(alpha['var'], 2)}).")


def band_pp(m: int) -> float:
    """Drei Standardfehler eines Anteils bei m Datensätzen im ungünstigsten Fall (Anteil 0,5), in Anteilen."""
    return 3 * 0.5 / math.sqrt(max(m, 1))


def kern_answers(nine: list[dict], fz: dict) -> list[str]:
    """Die drei Kurzantworten der Kernfragen aus den Zahlen der Demo (neun Paare, vorgerechnete Messungen)."""
    pl = next(r for r in nine if r["key"] == "poisson/log")
    pi = next(r for r in nine if r["key"] == "poisson/identity")
    if pl["zoll"] is None or pi["zoll"] is None:
        # Auf manchen gezogenen Datensätzen konvergiert eine der beiden Anpassungen nicht (der Identitäts-Link kann das Mittel unter null drücken): dann keine Zahl behaupten.
        fehlt = "beim Log-Link" if pl["zoll"] is None else "bei der Identität"
        a1 = (f"Auf dieser Container-Tabelle (n = 3 000, ein Datensatz) konvergiert die Anpassung {fehlt} nicht; "
              "der Vergleich der beiden Lesarten des Zollhalt-Koeffizienten entfällt für diesen Datensatz (anderen Zufalls-Seed ziehen).")
    else:
        a1 = (f"Auf der Container-Tabelle (n = 3 000, ein Datensatz) wirkt der Zollhalt auf die Standtage mit {num(pl['zoll'])} beim Log-Link (Faktor {num(math.exp(pl['zoll']))}) und mit "
              f"{num(pi['zoll'])} Tagen bei der Identität: dieselben Daten, zwei Lesarten.")
    ex = next(r for r in fz["link_export"] if r["n"] == 50)
    im = next(r for r in fz["link_import"] if r["n"] == 50)
    a2 = (f"Im Vehikel nur bei einem stetigen Merkmal: beim Vorlauf im Export wählt der AIC bei n = 50 den wahren Identitäts-Link in {pct(ex['wins']['identity'] / ex['used'])} der Datensätze, "
          f"bei den 0/1-Merkmalen des Imports den wahren Log-Link nur in {pct(im['wins']['log'] / im['used'])} (Wiederholung = ein Datensatz).")
    rows = {r["c"]: r for r in fz["schwelle"]["rows"]}
    lo, hi = min(r["rej_hi"] for r in rows.values()), max(r["rej_hi"] for r in rows.values())
    a3 = (f"Im Vehikel nicht bei einem Ja/Nein-Ausgang ohne Gruppierung: bei richtigem Modell (n = 1 050) verwirft die ungruppierte Deviance je nach Standgeld-Schwelle in {pct(lo)} bis {pct(hi)} der Datensätze "
          f"(Schwelle 4 Tage: {pct(rows[4]['rej_hi'])}, 7 Tage: {pct(rows[7]['rej_hi'])}).")
    return [a1, a2, a3]


def exp6_table() -> list[dict]:
    """Die Zeilen von Experiment 6: gilt die IRLS-Rechnung, womit belegt, in welcher Stufe."""
    return [
        {"Zeile": "Separation", "Gilt die IRLS-Rechnung?": "nein: kein Grenzwert, der Koeffizient wächst", "Beleg": "diese Demo (live, vier Container)", "Stufe": "eigene Rechnung"},
        {"Zeile": "nicht kanonischer Link", "Gilt die IRLS-Rechnung?": "ja, als Fisher-Scoring; Pfad verschieden von Newton, Grenzwert gleich", "Beleg": "diese Demo (Experiment 1)", "Stufe": "eigene Rechnung"},
        {"Zeile": "zweiter Parameter (NB2-α)", "Gilt die IRLS-Rechnung?": "nur mit Außenschleife um das IRLS", "Beleg": "diese Demo (live)", "Stufe": "eigene Rechnung"},
        {"Zeile": "Lasso (IRLS als äußere Schleife)", "Gilt die IRLS-Rechnung?": "als äußere Schleife; innen ein Koordinatenabstieg, kein gewichtetes Kleinste-Quadrate-Problem",
         "Beleg": "frühere Messung, nicht Teil dieser Demo", "Stufe": "eigene Rechnung, früher"},
        {"Zeile": "Quantilregression", "Gilt die IRLS-Rechnung?": "nein: der Pinball-Verlust ist keine Log-Likelihood einer Exponentialfamilie", "Beleg": "nicht gemessen", "Stufe": "Begründung aus der Definition"},
        {"Zeile": "Kalibrierung", "Gilt die IRLS-Rechnung?": "nein: sie verändert vorhandene Wahrscheinlichkeiten nachträglich", "Beleg": "nicht gemessen", "Stufe": "Begründung aus der Definition"},
        {"Zeile": "gemischtes Modell", "Gilt die IRLS-Rechnung?": "nein: Zufallseffekte bringen Varianzparameter und ein Integral hinzu", "Beleg": "nicht gemessen", "Stufe": "Begründung aus der Definition"},
    ]
