"""Live-Rechnungen der App: jede Kombination der Regler liefert ein Ergebnis oder eine Meldung, nie eine Ausnahme; die Live-Wiederholungen sind deterministisch und stimmig."""

import itertools
import math

import numpy as np
import pytest

import gfm_constants as C
import gfm_evaluation as ev


@pytest.mark.parametrize("daten", list(C.HAND_DATEN))
def test_experiment_1_never_raises_for_any_combination_of_controls(daten):
    regeln = ("beide",) if daten == "tabelle" else ("fisher", "newton", "beide")           # „beide“ rechnet auf der großen Tabelle beide Regeln; spart Zeit
    for paar, regel, start, halb in itertools.product(C.PAIR_KEYS, regeln, ("daten", "fuenf", "null"), (False, True)):
        res = ev.run_exp1(daten, paar, regel, start, halb, 180)
        kind, text = ev.verdict_exp1(res, None)
        assert kind in ("success", "warning") and text
        for f in res["fits"].values():
            assert np.all(np.isfinite(f.beta)) or not f.converged        # nie ∞ oder NaN als Ergebnis einer „konvergierten“ Anpassung
            assert f.converged or f.message                              # Nicht-Konvergenz kommt mit einer Meldung


def test_experiment_2_never_raises_for_any_combination_of_controls():
    for daten, aus, fam, link in itertools.product(C.E2_DATEN, C.E2_AUSGANG, C.E2_FAMILIEN, C.E2_LINKS):
        res = ev.run_exp2(daten, aus, fam, link, 180)
        kind, text = ev.verdict_exp2(res)
        assert kind in ("success", "info") and text
        assert all(r["ok"] or r.get("message") for r in res["links"] + res["families"])


@pytest.mark.parametrize("welt,n", [("export", 20), ("export", 5000), ("import", 20), ("import", 5000)])
def test_experiment_3_live_at_the_edges(welt, n):
    live = ev.run_exp3_live(welt, n, 180)
    assert ev.aic_winner(live) in (None, "identity", "log", "inverse")
    row = ev.frozen_link_row(welt, n)
    assert row is not None and row["n"] == n                                # für jedes n und jede Welt gibt es eine vorgerechnete Zeile
    kind, text = ev.verdict_exp3(welt, n, live, row)
    assert kind == "info" and f"n = {ev.ntxt(n)}" in text


@pytest.mark.parametrize("c,n,welt,modell", [(2, 350, "mit", "M1"), (21, 350, "ohne", "M0"), (4, 3500, "ohne", "M1"), (7, 1050, "mit", "M0")])
def test_experiment_4_live_at_the_edges(c, n, welt, modell):
    live = ev.run_exp4_live(c, n, welt, modell, 180)
    kind, text = ev.verdict_exp4(live, ev.frozen_deviance_row(c, n, welt, modell))
    assert kind in ("info", "warning") and text
    if live["ok"]:
        assert live["dfu"] == n - live["p"] and live["crit_lo"] < live["dfu"] < live["crit_hi"]
        assert live["relation_gap"] < 1e-8                                  # ungruppiert = gruppiert − 2·ℓ_gesättigt


def test_frozen_row_lookup_names_its_source():
    assert ev.frozen_deviance_row(4, 1050, "ohne", "M0")["source"] == "schwelle"
    assert ev.frozen_deviance_row(7, 350, "mit", "M0")["source"] == "modell"
    assert ev.frozen_deviance_row(4, 350, "mit", "M0") is None             # weder Schwellenreihe noch Schwelle 7
    assert ev.frozen_deviance_row(7, 1050, "ohne", "M1")["source"] == "modell"


def test_every_experiment_5_cell_exists():
    for n, link in itertools.product(C.E5_N, C.E5_LINK):
        rows = ev.exp5_rows(n, link)
        assert len(rows) == 4 and sum(r["falsch"] for r in rows) == 1
        assert ev.verdict_exp5(rows, n, link)[0] == "info"


def test_live_repetitions_are_deterministic_and_consistent():
    a = ev.live_repeat_exp3("export", 20, 12, 180)
    assert a == ev.live_repeat_exp3("export", 20, 12, 180)
    assert a["used"] == a["reps"] == 12 and sum(a["wins"].values()) == 12
    assert ev.live_repeat_exp3("export", 20, 12, 181) != a                  # anderer Zufalls-Seed, andere Datensätze
    b = ev.live_repeat_exp3("import", 50, 10, 180)
    assert sum(b["wins"].values()) == b["used"] <= 10
    s = ev.live_repeat_exp4(7, 350, "mit", "M0", 8, 180)
    assert s["reps"] == 8 and 0 <= s["rej_du"] <= 1 and 0 <= s["below_du"] <= 1 and s["valid"] <= 8
    assert math.isfinite(s["ent2"]) and 0 < s["prev"] < 1
    t = ev.live_repeat_exp4(7, 350, "mit", "M1", 6, 180)
    assert t["valid"] <= 6


def test_live_repetitions_use_other_datasets_than_the_frozen_ones():
    import gfm_experiments as E
    assert E.seed_of(E.EXP_LINK_EXPORT, 20, E.LIVE_BASE + 180 * 1000) != E.seed_of(E.EXP_LINK_EXPORT, 20, 999)
    assert E.LIVE_BASE > max(g[1] for g in C.LINK_EXPORT_GRID + C.LINK_IMPORT_GRID + C.MODELL_GRID) and E.LIVE_BASE > C.SCHWELLE_REPS


def test_nine_pairs_table_and_references_are_deterministic():
    assert ev.nine_pairs(7) == ev.nine_pairs(7)
    assert ev.reference_for("hand9", "binomial/logit", 7) == ev.reference_for("hand9", "binomial/logit", 7)
    assert ev.reference_for("hand4", "poisson/identity", 7)["ok"]
