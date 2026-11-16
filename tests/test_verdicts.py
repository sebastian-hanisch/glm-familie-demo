"""Verdikte an von Hand gesetzten Zeilen (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier. Ein Verdikt ist eine reine Funktion (Art, Text)."""

from types import SimpleNamespace

import numpy as np
import pytest

import gfm_evaluation as ev


def _fit(iterations, converged=True, message=""):
    return SimpleNamespace(iterations=iterations, converged=converged, message=message)


def _res(canonical, fits, **kw):
    base = {"family": "poisson", "link": "log" if canonical else "identity", "n": 3000, "canonical": canonical, "fits": fits}
    base.update(kw)
    return base


# --- Hilfen -----------------------------------------------------------------------------------------------------------------------------
def test_number_formats_use_the_decimal_comma_and_thin_groups():
    assert ev.num(0.12345) == "0,123" and ev.num(7.0, 1) == "7,0"
    assert ev.pct(0.1234) == "12,3 %" and ev.pct(0.5, 0) == "50 %"
    assert ev.sci(6.0e-11) == "6,0·10⁻¹¹" and ev.sci(0) == "0" and ev.sci(2.5e3, 2) == "2,50·10³"
    assert ev.ntxt(3000) == "3 000" and ev.ntxt(350) == "350"


def test_live_repetition_budget_is_inversely_proportional_to_n_and_bounded():
    assert ev.live_reps_limit(20) == 200                              # höchstens 200
    assert ev.live_reps_limit(1000) == 40 and ev.live_reps_limit(200) == 200
    assert ev.live_reps_limit(5000) == 8 and ev.live_reps_limit(100000) == 5   # mindestens 5
    assert ev.band_pp(100) == pytest.approx(0.15)                     # drei Standardfehler im ungünstigsten Fall


# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------
def test_verdict_exp1_canonical_names_the_equal_path():
    kind, text = ev.verdict_exp1(_res(True, {"fisher": _fit(5), "newton": _fit(5)}, path_diff=1.4e-13, limit_diff=0.0), None)
    assert kind == "success"
    assert "Poisson mit Log-Link" in text and "n = 3 000" in text and "Fisher-Scoring 5 Schritte, Newton 5 Schritte" in text
    assert "kanonisch" in text and "derselbe (größter Unterschied 1,4·10⁻¹³)" in text and "nicht kanonisch" not in text


def test_verdict_exp1_non_canonical_names_different_paths_and_the_same_limit():
    kind, text = ev.verdict_exp1(_res(False, {"fisher": _fit(103), "newton": _fit(7)}, path_diff=2.78, limit_diff=8.0e-11), {"ok": True, "dist": 2.7e-9, "spread": 4.6e-9, "grad": 2.1e-7})
    assert kind == "success"
    assert "Poisson mit Identitäts-Link" in text and "Fisher-Scoring 103 Schritte, Newton 7 Schritte" in text
    assert "nicht kanonisch: die Pfade sind verschieden (größter Unterschied 2,78)" in text and "Unterschied 8,0·10⁻¹¹" in text
    assert "landet 2,7·10⁻⁹ daneben (Streuung ihrer Startpunkte 4,6·10⁻⁹)" in text and "2,1·10⁻⁷" in text


def test_verdict_exp1_reports_non_convergence_with_the_message():
    kind, text = ev.verdict_exp1(_res(False, {"fisher": _fit(1000, False, "nicht konvergiert (Höchstzahl der Schritte erreicht)"), "newton": _fit(7)}), None)
    assert kind == "warning" and "Fisher konvergiert nicht" in text and "Höchstzahl der Schritte" in text
    assert ev.verdict_exp1(_res(True, {}), None)[0] == "warning"       # keine Anpassung gewählt: nichts zu bewerten


def test_verdict_exp1_single_rule_has_no_path_comparison():
    kind, text = ev.verdict_exp1(_res(True, {"newton": _fit(4)}), None)
    assert kind == "success" and "Newton 4 Schritte" in text and "Fisher" not in text and "Pfad" not in text


# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------
def _row2(link, zoll, means=(5.275, 9.78), ok=True):
    return {"link": link, "ok": ok, "zoll": zoll, "group_means": means, "mu_min": 1.0, "we": None, "family": "gamma"}


def test_verdict_exp2_two_groups_give_equal_means_and_different_coefficients():
    res = {"daten": "hand9", "n": 9, "links": [_row2("log", 0.6174), _row2("identity", 4.505), _row2("inverse", -0.08732)], "families": []}
    kind, text = ev.verdict_exp2(res)
    assert kind == "success" and "gesättigtes Modell" in text
    assert "5,275 und 9,780 Tage" in text and "Log: Zoll +0,6174" in text and "Identität: Zoll +4,5050" in text and "Kehrwert: Zoll -0,0873" in text


def test_verdict_exp2_unequal_means_are_not_called_saturated():
    res = {"daten": "hand9", "n": 9, "links": [_row2("log", 0.6, (5.3, 9.8)), _row2("identity", 4.5, (5.0, 9.8)), _row2("inverse", -0.08, (5.3, 9.8))], "families": []}
    kind, text = ev.verdict_exp2(res)
    assert kind == "info" and "unterscheiden sich" in text


def test_verdict_exp2_table_names_scale_and_negative_means():
    rows = [{**_row2("log", 0.617), "mu_min": 0.89}, {**_row2("identity", 4.409), "mu_min": -1.22}, {**_row2("inverse", 0.0, ok=False)}]
    kind, text = ev.verdict_exp2({"daten": "tabelle", "n": 3000, "links": rows, "families": []})
    assert kind == "info" and "n = 3 000, ein Datensatz" in text
    assert "Faktor 1,853 auf das Mittel" in text and "+4,409 Tage" in text
    assert "Nicht gültig: Kehrwert" in text and "negativ bei: Identität (-1,22)" in text


# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------
def _live3(aic_identity, aic_log, aic_inverse):
    return {"fits": {"identity": {"aic": aic_identity}, "log": {"aic": aic_log}, "inverse": {"aic": aic_inverse}}}


def test_verdict_exp3_names_the_winner_and_whether_it_is_the_true_link():
    row = {"reps": 1000, "used": 1000, "wins": {"identity": 915, "log": 85, "inverse": 0}}
    kind, text = ev.verdict_exp3("export", 50, _live3(10.0, 12.0, 14.0), row)
    assert "wählt den Link Identität (der wahre)" in text
    assert "1 000 vorgerechnete Datensätze gleicher Größe wählt der AIC Identität / Log / Kehrwert in 91,5 % / 8,5 % / 0,0 %" in text and "Export, Vorlauf in [1; 10]" in text
    _, text2 = ev.verdict_exp3("import", 200, _live3(10.0, 12.0, 9.0), None)
    assert "wählt den Link Kehrwert (der wahre ist Log)" in text2 and "vorgerechnete" not in text2


def test_aic_winner_ignores_invalid_fits():
    assert ev.aic_winner({"fits": {"identity": None, "log": {"aic": 5.0}, "inverse": {"aic": 4.0}}}) == "inverse"
    assert ev.aic_winner({"fits": {"identity": None, "log": None, "inverse": None}}) is None


# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------
def _live4(du, crit_hi=391.4, crit_lo=305.0, welt="mit", modell="M0"):
    return {"ok": True, "n": 350, "c": 7, "prevalence": 0.237, "p": 3, "du": du, "dfu": 347, "crit_hi": crit_hi, "crit_lo": crit_lo, "ent2": 0.831, "welt": welt, "modell": modell}


def test_verdict_exp4_live_numbers_and_rejection_wording():
    _, inside = ev.verdict_exp4(_live4(341.5), None)
    assert "n = 350, Standgeld ab 7 Tagen, Anteil 23,7 %, p = 3" in inside and "Deviance 341,5" in inside and "D/(n − p) = 0,984" in inside
    assert "2·mittlere Entropie der wahren Wahrscheinlichkeiten 0,831" in inside and "347 Freiheitsgraden lehnt nicht ab (95-%-Wert 391,4)" in inside
    _, outside = ev.verdict_exp4(_live4(400.0), None)
    assert "lehnt ab (95-%-Wert 391,4)" in outside and "lehnt nicht ab" not in outside
    _, edge = ev.verdict_exp4(_live4(391.4), None)
    assert "lehnt nicht ab" in edge                                   # genau auf dem Wert: nicht verworfen (Verwerfung nur darüber)


def test_verdict_exp4_says_whether_a_rejection_is_a_false_alarm_or_a_miss():
    _, alarm = ev.verdict_exp4(_live4(400.0, welt="ohne"), None)                        # Welt ohne Kundeneffekt: das Modell ist richtig
    assert "lehnt ab" in alarm and "Das Modell ist in dieser Welt richtig: die Verwerfung ist ein Fehlalarm." in alarm
    _, alarm_m1 = ev.verdict_exp4(_live4(400.0, welt="mit", modell="M1"), None)           # Welt mit Kundeneffekt, Modell M1: auch richtig
    assert "Fehlalarm" in alarm_m1
    _, miss = ev.verdict_exp4(_live4(341.5, welt="mit", modell="M0"), None)               # Kunden fehlen, Test verwirft nicht
    assert "lehnt nicht ab" in miss and "Das Modell ist hier falsch (die Kunden fehlen); die ungruppierte Deviance merkt es in diesem Datensatz nicht." in miss
    _, caught = ev.verdict_exp4(_live4(400.0, welt="mit", modell="M0"), None)             # Kunden fehlen, Test verwirft: weder Fehlalarm noch Fehlen
    assert "Fehlalarm" not in caught and "falsch" not in caught
    _, quiet = ev.verdict_exp4(_live4(341.5, welt="ohne"), None)
    assert "Fehlalarm" not in quiet and "falsch" not in quiet


def test_verdict_exp4_names_the_frozen_row_by_source():
    fz_schwelle = {"source": "schwelle", "valid": 600, "rej_hi": 1.0, "below_lo": 0.0, "rej_g4": 0.045}
    _, t1 = ev.verdict_exp4(_live4(341.5), fz_schwelle)
    assert "über 600 Datensätze (Welt ohne Kundeneffekt, Modell exakt richtig): über dem 95-%-Wert in 100,0 %, unter dem 5-%-Wert in 0,0 %; gruppiert (4 Zellen) über dem 95-%-Wert in 4,5 %" in t1
    fz_modell = {"source": "modell", "valid": 400, "rej_du": 0.0226, "below_du": 0.0175, "rej_dg48": 1.0}
    _, t2 = ev.verdict_exp4(_live4(341.5), fz_modell)
    assert "ungruppiert über dem 95-%-Wert in 2,3 %, unter dem 5-%-Wert in 1,8 %; gruppiert (48 Zellen) über dem 95-%-Wert in 100,0 %." in t2


def test_verdict_exp4_without_a_valid_fit_is_a_warning():
    kind, text = ev.verdict_exp4({"ok": False}, None)
    assert kind == "warning" and "keine gültige Probit-Anpassung" in text


# --- Experiment 5 und 6 --------------------------------------------------------------------------------------------------------------
def _s5(rej_du, rej_dg48, small48=0.5, rej_dg4=0.05):
    return {"valid": 390, "reps": 400, "rej_du": rej_du, "rej_dg48": rej_dg48, "small48": small48, "rej_dg4": rej_dg4}


def test_verdict_exp5_names_all_tests_for_the_wrong_and_the_right_model():
    rows = [{"falsch": True, "welt": "mit", "modell": "M0", "s": _s5(0.022, 1.0), "lq": {"rej": 1.0}},
            {"falsch": False, "welt": "ohne", "modell": "M0", "s": _s5(0.0, 0.225, 0.996, 0.04), "lq": {"rej": 0.0685}}]
    kind, text = ev.verdict_exp5(rows, 350, "probit")
    assert kind == "info"
    assert "n = 350, Probit-Link, 390 gültige von 400 Datensätzen" in text and "ungruppierte Deviance verwirft in 2,2 %" in text
    assert "gruppierte (48 Zellen) in 100,0 %, der Likelihood-Quotienten-Test in 100,0 %" in text
    assert "Likelihood-Quotienten-Test 6,9 %, gruppiert (4 Zellen) 4,0 %, gruppiert (48 Zellen) 22,5 % bei 100 % Zellen" in text


def test_verdict_exp6_names_separation_and_both_alpha_estimates():
    rows = [{"k": 1, "beta1": 1.2}, {"k": 40, "beta1": 35.387}]
    kind, text = ev.verdict_exp6(rows, {"alpha_profile": 0.17665, "alpha_alternating": 0.17665, "mean": 6.221, "var": 43.5})
    assert kind == "info" and "nach 1 Schritt 1,20, nach 40 Schritten 35,39" in text and "Profil-Außenschleife 0,1767, abwechselnde Schätzung 0,1767" in text
    assert "Mittel 6,22, Varianz 43,50" in text


def test_kern_answers_cite_frozen_rows():
    nine = [{"key": "poisson/log", "zoll": 0.7048}, {"key": "poisson/identity", "zoll": 5.2142}]
    fz = {"link_export": [{"n": 50, "used": 1000, "wins": {"identity": 915}}], "link_import": [{"n": 50, "used": 1000, "wins": {"log": 144}}],
          "schwelle": {"rows": [{"c": 4, "rej_hi": 1.0}, {"c": 7, "rej_hi": 0.0}, {"c": 10, "rej_hi": 0.0}]}}
    a1, a2, a3 = ev.kern_answers(nine, fz)
    assert "0,705 beim Log-Link (Faktor 2,023)" in a1 and "5,214 Tagen bei der Identität" in a1
    assert "91,5 %" in a2 and "14,4 %" in a2
    assert "0,0 % bis 100,0 %" in a3 and "Schwelle 4 Tage: 100,0 %, 7 Tage: 0,0 %" in a3


def test_kern_answers_survive_a_fit_that_did_not_converge():
    """Auf manchen gezogenen Datensätzen konvergiert die Identitäts-Anpassung nicht (zoll = None): keine Ausnahme, keine erfundene Zahl."""
    fz = {"link_export": [{"n": 50, "used": 1000, "wins": {"identity": 915}}], "link_import": [{"n": 50, "used": 1000, "wins": {"log": 144}}],
          "schwelle": {"rows": [{"c": 4, "rej_hi": 1.0}, {"c": 7, "rej_hi": 0.0}, {"c": 10, "rej_hi": 0.0}]}}
    a1, a2, a3 = ev.kern_answers([{"key": "poisson/log", "zoll": 0.7048}, {"key": "poisson/identity", "zoll": None}], fz)
    assert "konvergiert die Anpassung bei der Identität nicht" in a1 and "5," not in a1
    a1, _, _ = ev.kern_answers([{"key": "poisson/log", "zoll": None}, {"key": "poisson/identity", "zoll": 5.2142}], fz)
    assert "konvergiert die Anpassung beim Log-Link nicht" in a1
    assert "91,5 %" in a2 and "Schwelle 4 Tage" in a3


def test_exp6_table_has_seven_rows_with_a_stage_each():
    rows = ev.exp6_table()
    assert len(rows) == 7 and all(r["Stufe"] and r["Beleg"] for r in rows)
    assert [r["Zeile"] for r in rows] == ["Separation", "nicht kanonischer Link", "zweiter Parameter (NB2-α)", "Lasso (IRLS als äußere Schleife)", "Quantilregression", "Kalibrierung", "gemischtes Modell"]
    assert np.all([r["Beleg"] == "nicht gemessen" for r in rows[4:]])
