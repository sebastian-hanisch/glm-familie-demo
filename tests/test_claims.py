"""Claims-Tests: jede Zahl der Befunde B1 bis B12 der README mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a).

Die Handbeispiel-Zahlen H1 bis H8 stehen in `test_algorithm.py` (Brüche), die Vehikel-Prüfsteine in `test_oracle_vehikel.py`, die Deviancen in `test_oracle_glm_familie.py`.
Plattform (Lehre 2026-11-15): Gleitkomma-Werte als Band, Schrittzahlen als Band, Zählungen aus Neu-Rechnung ± 2, Anteile als Band aus drei Standardfehlern; exakt nur Brüche.
Die vorgerechneten Messungen (`data/gfm_frozen.json`) werden hier gegen die README-Zeilen gelesen (der Text steht als Zeichenkette hier) und in `test_frozen.py` gegen die Neu-Rechnung gehalten;
mit GFM_FULL=1 rechnet `test_frozen` alle Zeilen neu (rund 10 Minuten)."""

import math

import numpy as np
import pytest
from scipy import special

import gfm_algorithm as A
import gfm_constants as C
import gfm_evaluation as ev
import gfm_frozen as F
import gfm_inference as I
import gfm_rng as R
import gfm_scenario as S

FZ = F.load()
pct, num = ev.pct, ev.num


@pytest.fixture(scope="module")
def nine():
    return {r["key"]: r for r in ev.nine_pairs(180)}


# --- B1: neun Paare, Container-Tabelle, Zufalls-Seed 180 --------------------------------------------------------------------------------------
# Schlüssel: (kanonisch, Fisher 1e-10, Fisher 1e-8, Newton 1e-10, Newton 1e-8, Pfadunterschied, Zoll-Koeffizient)
B1 = {
    "normal/identity": (True, 1, 1, 1, 1, 0.0, 5.242),
    "gamma/log": (False, 10, 8, 6, 5, 0.205, 0.764),
    "gamma/inverse": (True, 7, 7, 7, 7, 0.0, -0.115),
    "poisson/log": (True, 5, 5, 5, 5, 0.0, 0.705),
    "poisson/identity": (False, 103, 82, 7, 6, 2.78, 5.214),
    "nb2/log": (False, 7, 6, 4, 4, 0.0182, 0.705),
    "binomial/logit": (True, 9, 8, 9, 8, 0.0, 1.755),
    "binomial/probit": (False, 12, 11, 8, 8, 0.608, 1.064),
    "binomial/cloglog": (False, 12, 11, 8, 8, 0.498, 1.418),
}


@pytest.mark.parametrize("key", list(B1))
def test_b1_nine_pairs_table(nine, key):
    canon, f10, f8, n10, n8, pd, zoll = B1[key]
    r = nine[key]
    assert r["canonical"] == canon and r["ok"]
    band = 15 if key == "poisson/identity" else 1                                    # Schrittzahlen als Band (Fisher bei Poisson/Identität kriecht)
    assert abs(r["fisher10"] - f10) <= band and abs(r["fisher8"] - f8) <= band
    assert abs(r["newton10"] - n10) <= 1 and abs(r["newton8"] - n8) <= 1
    if canon:
        assert r["path_diff"] < 1e-12                                                # kanonisch: derselbe Pfad (größter Unterschied 1,7·10⁻¹³)
    else:
        assert r["path_diff"] == pytest.approx(pd, rel=0.05)
    assert r["zoll"] == pytest.approx(zoll, abs=6e-4)
    assert r["limit_diff"] < 1e-9 and r["grad"] < 1e-5                              # gleiche Grenzwerte; Gradient der unabhängigen Likelihood am IRLS-Ergebnis (gemessen 2,2·10⁻⁶ unter Linux, 1,8·10⁻⁶ unter Windows: Rauschen der Differenzen)


def test_b1_four_canonical_pairs_and_five_others(nine):
    assert sum(r["canonical"] for r in nine.values()) == 4 and sum(not r["canonical"] for r in nine.values()) == 5
    assert max(r["limit_diff"] for r in nine.values()) < 1e-9
    assert min(r["path_diff"] for r in nine.values() if not r["canonical"]) > 1e-2 and max(r["path_diff"] for r in nine.values() if r["canonical"]) < 1e-12


def test_b2_fisher_steps_over_twenty_tables():
    fs, ns = [], []
    for seed in range(300, 320):
        T = S.draw_table(3000, seed)
        X, y = T.designmatrix(), T.standtage()
        f = A.fit_irls(X, y, "poisson", "identity", "fisher", tol=1e-10, maxit=3000)
        n = A.fit_irls(X, y, "poisson", "identity", "newton", tol=1e-10, maxit=3000)
        assert f.converged and n.converged, seed
        fs.append(f.iterations)
        ns.append(n.iterations)
    assert 30 <= min(fs) <= 40 and 140 <= max(fs) <= 175 and 45 <= float(np.median(fs)) <= 60      # gemessen 35 bis 157, Median 53
    assert set(ns) <= {6, 7}


# --- B3: Skala und Familie, Container-Tabelle (Zufalls-Seed 180, ein Datensatz) --------------------------------------------------------------
def test_b3_scale_and_family():
    by = {(r["family"], r["link"]): r for r in ev.run_exp2("tabelle", "zähl", "poisson", "log", 180)["links"]}
    assert by[("poisson", "log")]["zoll"] == pytest.approx(0.7048, abs=6e-5) and num(math.exp(by[("poisson", "log")]["zoll"])) == "2,024"
    assert by[("poisson", "identity")]["zoll"] == pytest.approx(5.2142, abs=6e-5) and by[("poisson", "inverse")]["zoll"] == pytest.approx(-0.0976, abs=6e-5)
    assert by[("poisson", "log")]["mu_min"] == pytest.approx(1.212, abs=6e-4) and by[("poisson", "identity")]["mu_min"] == pytest.approx(0.288, abs=6e-4)
    fam = {r["family"]: r for r in ev.run_exp2("tabelle", "stetig", "gamma", "log", 180)["families"]}      # Verweilzeit d, Log-Link, drei Familien
    assert [fam[f]["zoll"] for f in ("normal", "poisson", "gamma")] == pytest.approx([0.7613, 0.7623, 0.7638], abs=6e-5)
    assert [fam[f]["we"] for f in ("normal", "poisson", "gamma")] == pytest.approx([0.2016, 0.1780, 0.1536], abs=6e-5)
    assert fam["normal"]["bias_art"] == pytest.approx([0.38, -1.27, 0.64, 0.50], abs=0.006)
    assert fam["gamma"]["bias_art"] == pytest.approx([-0.46, 2.18, -1.39, -0.50], abs=0.006)
    assert fam["poisson"]["bias_art"] == pytest.approx([0.0] * 4, abs=1e-9)           # nur beim Poisson-Modell stimmt das Mittel je Art genau
    ols = {r["link"]: r for r in ev.run_exp2("tabelle", "stetig", "normal", "identity", 180)["links"]}
    assert ols["identity"]["mu_min"] == pytest.approx(-1.366, abs=6e-4)               # OLS auf d: negative Verweilzeit
    hand = {r["link"]: r for r in ev.run_exp2("hand9", "stetig", "gamma", "log", 180)["links"]}
    assert all(hand[lk]["group_means"] == pytest.approx((5.275, 9.78), abs=1e-6) for lk in hand)


# --- B4 und B5: Link-Wahl (vorgerechnet; Anteile aus der Datei, Datei in test_frozen gegen die Neu-Rechnung) ------------------------------------
B4_WINS = ["81,3 % / 16,6 % / 2,1 %", "91,5 % / 8,5 % / 0,0 %", "99,9 % / 0,1 % / 0,0 %", "100,0 % / 0,0 % / 0,0 %", "100,0 % / 0,0 % / 0,0 %"]
B4_LOG = [("0,906", "1,992", "3,738"), ("0,901", "2,010", "3,789"), ("0,903", "2,016", "3,810"), ("0,904", "2,025", "3,824"), ("0,903", "2,020", "3,819")]
B4_INVERSE_DEFINED = [("3,2 %", "0,1 %"), ("0,4 %", "0,0 %"), ("0,0 %", "0,0 %"), ("0,0 %", "0,0 %"), ("0,0 %", "0,0 %")]
B5_WINS = ["14,2 % / 50,9 % / 34,9 %", "14,4 % / 44,2 % / 41,4 %", "27,9 % / 40,2 % / 31,9 %", "62,5 % / 23,3 % / 14,2 %", "93,3 % / 5,7 % / 1,0 %"]


def test_b4_link_choice_export():
    rows = FZ["link_export"]
    assert [(r["n"], r["reps"]) for r in rows] == [(20, 1000), (50, 1000), (200, 1000), (1000, 500), (5000, 100)]
    for r, wins, lg, dfn in zip(rows, B4_WINS, B4_LOG, B4_INVERSE_DEFINED):
        u, w = r["used"], r["wins"]
        assert f"{pct(w['identity'] / u)} / {pct(w['log'] / u)} / {pct(w['inverse'] / u)}" == wins
        assert tuple(num(x) for x in r["median_ratio"]["log"]) == lg
        assert (pct(r["defined"]["inverse"][1]), pct(r["defined"]["inverse"][2])) == dfn
        assert r["median_ratio"]["identity"] == pytest.approx([1.0, 1.0, 1.0], abs=0.006)       # der wahre Link trifft das Mittel an allen drei Punkten
        assert r["median_ratio"]["inverse"][0] == pytest.approx(0.845, abs=0.002)
        assert r["log_over25_at_20"] >= 0.99
    # Zählung über die Zeilen: der AIC trennt im Export ab n = 50 (Identität in mindestens 90 % der Datensätze)
    share = [r["wins"]["identity"] / r["used"] for r in rows]
    assert [s >= 0.9 for s in share] == [False, True, True, True, True]
    assert all(r["invalid"] == {"identity": 0, "log": 0, "inverse": 0} for r in rows)
    # Band: Anteile gegen den Monte-Carlo-Fehler (3 Standardfehler) um die Werte der ersten Rechnung (80,1 / 90,8 / 99,8 %)
    for s, first, r in zip(share[:3], (0.801, 0.908, 0.998), rows):
        assert abs(s - first) <= 3 * math.sqrt(max(first * (1 - first), 0.002) / r["used"]) + 0.01


def test_b5_link_choice_import():
    rows = FZ["link_import"]
    assert [(r["n"], r["reps"]) for r in rows] == [(20, 1000), (50, 1000), (200, 1000), (1000, 600), (5000, 300)]
    for r, wins in zip(rows, B5_WINS):
        u, w = r["used"], r["wins"]
        assert f"{pct(w['log'] / u)} / {pct(w['identity'] / u)} / {pct(w['inverse'] / u)}" == wins
    assert [rows[i]["invalid"] for i in (0, 1)] == [{"identity": 16, "log": 7, "inverse": 17}, {"identity": 1, "log": 0, "inverse": 10}]
    assert all(rows[i]["invalid"] == {"identity": 0, "log": 0, "inverse": 0} for i in (2, 3, 4))
    # bei n = 50 wählt der AIC häufiger einen falschen als den wahren Link (14,4 % gegen 85,6 %)
    assert rows[1]["wins"]["log"] / rows[1]["used"] < 0.5
    # die wahre Wahl gewinnt erst ab n = 1 000 die Mehrheit
    assert [r["wins"]["log"] / r["used"] > 0.5 for r in rows] == [False, False, False, True, True]
    med = {r["n"]: r["median_celldev"] for r in rows}
    assert (num(med[50]["identity"]), num(med[50]["log"]), num(med[50]["inverse"])) == ("0,268", "0,262", "0,300")
    assert (num(med[5000]["identity"]), num(med[5000]["log"]), num(med[5000]["inverse"])) == ("0,091", "0,029", "0,091")


# --- B6: Schwellenreihe -----------------------------------------------------------------------------------------------------------------------
B6 = [  # c, Anteil, 2·H̄, D/(n − p), über 95 %, unter 5 %, gruppiert 4 Zellen, gültig
    (2, "0,902", "0,582", "0,583", "0,0 %", "100,0 %", "4,3 %", 539), (3, "0,735", "1,028", "1,029", "7,7 %", "0,0 %", "4,2 %", 600),
    (4, "0,568", "1,191", "1,191", "100,0 %", "0,0 %", "4,5 %", 600), (5, "0,431", "1,167", "1,168", "99,8 %", "0,0 %", "5,0 %", 600),
    (7, "0,247", "0,925", "0,926", "0,0 %", "53,0 %", "7,0 %", 600), (10, "0,114", "0,568", "0,566", "0,0 %", "100,0 %", "6,5 %", 600),
    (14, "0,045", "0,287", "0,285", "0,0 %", "100,0 %", "7,2 %", 600), (21, "0,011", "0,092", "0,097", "0,0 %", "100,0 %", "5,2 %", 427),
]


def test_b6_threshold_series():
    rows = FZ["schwelle"]["rows"]
    assert (FZ["schwelle"]["n"], FZ["schwelle"]["reps"]) == (1050, 600)
    for r, e in zip(rows, B6):
        assert (r["c"], num(r["prev"]), num(r["ent2"]), num(r["du_per_df"]), pct(r["rej_hi"]), pct(r["below_lo"]), pct(r["rej_g4"]), r["valid"]) == e
    # Zählung über alle acht Zeilen (richtiges Modell): über dem 95-%-Wert in 2, unter dem 5-%-Wert in mindestens der Hälfte in 5, nahe dem Niveau in 1
    assert sum(r["rej_hi"] >= 0.99 for r in rows) == 2
    assert sum(r["below_lo"] >= 0.5 for r in rows) == 5 and sum(r["below_lo"] >= 0.99 for r in rows) == 4
    assert [r["c"] for r in rows if 0.05 < r["rej_hi"] < 0.95] == [3]
    # D/(n − p) gleich 2·H̄ auf höchstens 0,005 (Deutung: die Entropie, nicht die Passung, bestimmt den Wert)
    assert max(abs(r["du_per_df"] - r["ent2"]) for r in rows) <= 0.005
    # nur die Zeile mit der seltensten Ereignisklasse hat ungültige Datensätze
    assert [r["c"] for r in rows if r["valid"] < r["reps"]] == [2, 21]


# --- B7 und B8: Gruppierung, Likelihood-Quotient, falscher Link --------------------------------------------------------------------------------
def _cell(welt, n, key):
    return FZ["modell"][welt][str(n)][key]


def test_b7_grouping_and_lr():
    wrong = [("350", "2,2 %", "100,0 %", "100 %", "5,0 %", 303), ("1050", "5,0 %", "100,0 %", "43 %", "5,5 %", 397), ("3500", "13,2 %", "100,0 %", "2 %", "5,6 %", 250)]
    for n, du, g48, small, g4, lq_valid in wrong:
        s, lq = _cell("mit", n, "M0/probit"), _cell("mit", n, "lq")["probit"]
        assert (pct(s["rej_du"]), pct(s["rej_dg48"]), pct(s["small48"], 0), pct(s["rej_dg4"])) == (du, g48, small, g4)
        assert pct(lq["rej"]) == "100,0 %" and lq["valid"] == lq_valid
    right_mit = [("350", "0,0 %", "11,6 %", "97 %", 303), ("1050", "0,0 %", "9,6 %", "69 %", 397), ("3500", "0,0 %", "8,8 %", "23 %", 250)]
    for n, du, g48, small, valid in right_mit:
        s = _cell("mit", n, "M1/probit")
        assert (pct(s["rej_du"]), pct(s["rej_dg48"]), pct(s["small48"], 0), s["valid"]) == (du, g48, small, valid)
    right_ohne = [("350", "0,0 %", "22,5 %", "100 %", "4,0 %", "6,9 %", 394), ("1050", "0,0 %", "11,0 %", "59 %", "6,8 %", "4,0 %", 400), ("3500", "0,0 %", "6,0 %", "6 %", "6,8 %", "5,6 %", 250)]
    for n, du, g48, small, g4, lq, valid in right_ohne:
        s, l = _cell("ohne", n, "M0/probit"), _cell("ohne", n, "lq")["probit"]
        assert (pct(s["rej_du"]), pct(s["rej_dg48"]), pct(s["small48"], 0), pct(s["rej_dg4"]), pct(l["rej"]), l["valid"]) == (du, g48, small, g4, lq, valid)
    # Band um das Niveau 5 % für den Likelihood-Quotienten-Test bei richtigem Modell (drei Standardfehler: ± 3,3 / 3,3 / 4,1 Prozentpunkte)
    for n, band in (("350", 0.033), ("1050", 0.033), ("3500", 0.041)):
        assert abs(_cell("ohne", n, "lq")["probit"]["rej"] - 0.05) <= band + 0.02       # gemessen 6,9 / 4,0 / 5,6 %: die erste Zeile liegt 1,9 Punkte über dem Niveau
    # Aussagen mit Zählung: falsches Modell: ungruppierte Deviance unter 15 % in allen drei Zeilen, gruppiert und LQ in 100 %
    assert all(_cell("mit", n, "M0/probit")["rej_du"] < 0.15 for n in ("350", "1050", "3500"))
    # die Zellen mit erwarteter Zahl unter 5 werden mit n seltener (richtiges Modell, 48 Zellen)
    sm = [_cell("mit", n, "M1/probit")["small48"] for n in ("350", "1050", "3500")]
    assert sm[0] > sm[1] > sm[2]


def test_b8_wrong_link():
    logit = [pct(_cell("mit", n, "M1/logit")["rej_dg48"]) for n in ("350", "1050", "3500")]
    cll = [pct(_cell("mit", n, "M1/cloglog")["rej_dg48"]) for n in ("350", "1050", "3500")]
    prb = [pct(_cell("mit", n, "M1/probit")["rej_dg48"]) for n in ("350", "1050", "3500")]
    assert (logit, cll, prb) == (["11,9 %", "9,8 %", "12,4 %"], ["19,3 %", "35,0 %", "82,4 %"], ["11,6 %", "9,6 %", "8,8 %"])
    diff_logit = [100 * (_cell("mit", n, "M1/logit")["rej_dg48"] - _cell("mit", n, "M1/probit")["rej_dg48"]) for n in ("350", "1050", "3500")]
    assert max(diff_logit) == pytest.approx(3.6, abs=0.05) and diff_logit[2] == max(diff_logit)              # Logit statt Probit: höchstens 3,6 Prozentpunkte (n = 3 500)
    diff_cll = [100 * (_cell("mit", n, "M1/cloglog")["rej_dg48"] - _cell("mit", n, "M1/probit")["rej_dg48"]) for n in ("350", "1050", "3500")]
    assert diff_cll[1] == pytest.approx(25.4, abs=0.05) and diff_cll[2] > diff_cll[1] > 20                    # cloglog ab n = 1 050: 25,4 Prozentpunkte und mehr
    aic = {n: _cell("mit", n, "aic") for n in ("350", "1050", "3500")}
    assert [round(100 * aic[n]["wins"]["probit"] / aic[n]["used"], 1) for n in aic] == [52.8, 66.0, 82.0]
    assert [400 - _cell("mit", n, "M1/probit")["valid"] for n in ("350", "1050")] == [97, 3]


# --- B9: Entropie-Formel in einer anderen Konfiguration ---------------------------------------------------------------------------------------
def test_b9_entropy_formula():
    x, w = np.polynomial.hermite_e.hermegauss(200)                          # Quadratur statt Zufall: x ~ N(0; 1)
    w = w / w.sum()
    p = special.expit(-0.5 + x)
    hbar = float(np.sum(w * (-(p * np.log(p) + (1 - p) * np.log(1 - p)))))
    assert 2 * hbar == pytest.approx(1.1635, abs=6e-5)
    assert (2 * 200 * hbar - 2) / 198 == pytest.approx(1.1651, abs=6e-5)
    rng = R.SplitMix64(2024)
    rej, dev, k = 0, [], 0
    for _ in range(300):
        xs = np.array([rng.normal() for _ in range(200)])
        y = np.array([float(rng.random() < q) for q in special.expit(-0.5 + xs)])
        f = A.fit_irls(np.column_stack([np.ones(200), xs]), y, "binomial", "logit", tol=1e-10, maxit=60)
        if f.converged:
            k += 1
            dev.append(f.deviance)
            rej += f.deviance > I.chi2_crit(198, 0.95)
    assert k == 300
    assert float(np.mean(dev)) / 198 == pytest.approx(1.170, abs=0.012)
    assert 0.40 <= rej / k <= 0.65                                           # gemessen 52,3 %; Band aus drei Standardfehlern (± 9 Punkte)


# --- B10: α und Separation -------------------------------------------------------------------------------------------------------------------
def test_b10_alpha_and_separation():
    a = ev.nb2_alpha_live(180)
    assert a["alpha_profile"] == pytest.approx(0.1766, abs=6e-5) and a["alpha_alternating"] == pytest.approx(a["alpha_profile"], rel=1e-3)
    assert a["mean"] == pytest.approx(6.221, abs=5e-4) and a["var"] == pytest.approx(43.50, abs=5e-3)
    rows = {r["k"]: r for r in ev.separation_path()}
    assert rows[1]["beta1"] == pytest.approx(1.2, abs=1e-12)                 # exakt 6/5 nach dem ersten Schritt
    assert [round(rows[k]["beta1"], 2) for k in (1, 2, 3, 5, 10, 20)] == [1.20, 2.02, 2.91, 4.84, 9.82, 19.82]
    assert all(rows[k]["beta1"] < rows[k2]["beta1"] for k, k2 in ((1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 40)))     # wächst weiter
    assert rows[40]["beta1"] == pytest.approx(35.39, abs=0.02)               # Rechengrenze |η| ≤ 30 greift (keine Eigenschaft der Daten)
    assert not any(r["converged"] for r in rows.values())


# --- B11: Fisher kriecht in kleinen Tabellen -----------------------------------------------------------------------------------------------
def test_b11_fisher_creeps_in_small_tables():
    ok, newton_ok = 0, 0
    first = None
    for seed in range(300, 320):
        T = S.draw_table(250, seed)
        X, y = T.designmatrix(), T.standtage()
        f = A.fit_irls(X, y, "poisson", "identity", "fisher", tol=1e-11, maxit=5000)
        n = A.fit_irls(X, y, "poisson", "identity", "newton", tol=1e-11, maxit=5000)
        ok += f.converged
        newton_ok += n.converged
        if seed == 300:
            first = (f, n)
    assert newton_ok == 20 and 14 <= ok <= 18                                 # gemessen 16 von 20
    assert not first[0].converged and first[1].converged                      # Zufalls-Seed 300: Fisher braucht mehr als 5 000 Schritte
    assert float(first[1].mu.min()) == pytest.approx(0.4745, abs=2e-3)


# --- B12: unabhängige Referenz in den vier in der App geprüften Paaren --------------------------------------------------------------------------
@pytest.mark.parametrize("pair", ["poisson/identity", "gamma/inverse", "binomial/logit", "nb2/log"])
def test_b12_reference_bands(pair):
    ref = ev.reference_for("tabelle", pair, 180)
    assert ref["ok"] and ref["dist"] < 1e-6 and ref["grad"] < 1e-5
    assert ref["spread"] < 1e-5


def test_every_number_group_of_the_readme_has_a_test_name():
    """Die README nennt diese Testfunktionen; sie müssen existieren."""
    import inspect
    import sys
    me = sys.modules[__name__]
    names = {n for n, _ in inspect.getmembers(me, inspect.isfunction)}
    readme = (__import__("pathlib").Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    import re
    cited = set(re.findall(r"`(test_b\d+_[a-z0-9_]+)`", readme))
    assert cited and cited <= names, sorted(cited - names)


def test_the_c_export_factor_and_hand_numbers_of_the_readme():
    assert S.C_EXPORT == pytest.approx(1.0294, abs=5e-5)
    assert C.NB2_ALPHA == 0.2 and C.STANDGELD_TAGE == 7.0
