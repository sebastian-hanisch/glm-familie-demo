"""Orakel des Vehikels: die Teile des Container-Generators, auf die sich die Experimente stützen, gegen ihre bekannte Form.

(1) SplitMix64 gegen die veröffentlichte Folge (Zufalls-Seed 1) und gegen sich selbst (gleicher Zufalls-Seed, gleiche Folge);
(2) der ln-Median und die Streuung je Container gegen die ausgeschriebene Formel;
(3) das Mittel der Verweilzeit im Export ist exakt linear im Vorlauf (Faktor C_EXPORT), im Import multiplikativ in Zoll und Wochenende (Monte Carlo, Bänder in Standardfehlern);
(4) das Standgeld im Import ist ein Probit, exakt linear in Zoll, Wochenende und Kunde: eine Probit-Anpassung trifft alle 14 Koeffizienten innerhalb von 4,5 Standardfehlern."""

import math

import numpy as np
import pytest
from scipy import stats

import gfm_algorithm as A
import gfm_rng as R
import gfm_scenario as S


def test_splitmix64_published_sequence_for_seed_1_and_determinism():
    r = R.SplitMix64(1)
    assert [r.next_u64() for _ in range(3)] == [10451216379200822465, 13757245211066428519, 17911839290282890590]
    a, b = R.SplitMix64(180), R.SplitMix64(180)
    assert [a.normal() for _ in range(5)] == [b.normal() for _ in range(5)]
    assert R.SplitMix64(180).random() == pytest.approx(0.6813822276742082, abs=0)


def test_generators_are_deterministic_and_pinned():
    t1, t2 = S.draw_table(4, 180), S.draw_table(4, 180)
    assert np.array_equal(t1.d, t2.d)
    assert t1.d == pytest.approx([1.1939956004883356, 5.556775918450356, 7.838553535248569, 2.651962657121611], rel=1e-12)
    assert S.draw_import(3, 5).d == pytest.approx([9.899426470997335, 1.8282228525714763, 6.293782097555623], rel=1e-12)
    assert S.draw_export(3, 5).d == pytest.approx([6.19180090346578, 2.7489416522058643, 6.319239981995036], rel=1e-12)
    assert not np.array_equal(S.draw_table(4, 181).d, t1.d)                   # anderer Zufalls-Seed, andere Container


def test_log_median_and_s_equal_the_written_out_formulas():
    UK = S.UK
    cases = [  # art, kunde, zoll, we, vorlauf, erwarteter ln-Median
        (0, 3, 1, 0, 0.0, math.log(3.5) + 0.7 + UK[3]),
        (0, 5, 0, 1, 0.0, math.log(3.5) + 0.3 + UK[5]),
        (0, 7, 1, 1, 0.0, math.log(3.5) + 1.0 + UK[7]),
        (1, 2, 0, 0, 4.0, math.log(0.6 + 0.85 * 4.0) + 0.3 * UK[2]),
        (2, 9, 0, 0, 2.5, math.log(0.3 + 0.8 * 2.5) + 0.3 * UK[9]),
        (3, 1, 0, 1, 0.0, math.log(7.0) + 0.3 + UK[1]),
    ]
    for art, kunde, zoll, we, v, expected in cases:
        mu, s = S.log_median_and_s(np.array([art]), np.array([kunde]), np.array([zoll]), np.array([we]), np.array([v]))
        assert mu[0] == pytest.approx(expected, abs=1e-12)
        assert s[0] == S.S_BASE[art]
    assert S.S_BASE == (0.55, 0.25, 0.35, 0.90)


def test_customer_effects_have_the_stated_spread():
    assert len(S.UK) == 12
    assert np.std(S.UK, ddof=1) == pytest.approx(0.379, abs=5e-4)
    assert np.std(S.UK, ddof=0) == pytest.approx(0.363, abs=5e-4)
    assert S.C_EXPORT == pytest.approx(1.0294, abs=5e-5)


def test_export_mean_is_exactly_linear_in_the_lead_time():
    ex = S.draw_export(40000, 902)
    ratio = float(ex.d.sum() / np.sum(S.export_mittel(ex.vorlauf)))
    assert ratio == pytest.approx(1.0, abs=0.006)                              # Σ d / Σ Formelmittel, Band von etwa 4 Standardfehlern
    for lo, hi in [(1, 3), (3, 5), (5, 7), (7, 9), (9, 10.01)]:                # je Vorlauf-Fünftel
        m = (ex.vorlauf >= lo) & (ex.vorlauf < hi)
        assert float(ex.d[m].mean() / S.export_mittel(ex.vorlauf[m]).mean()) == pytest.approx(1.0, abs=0.02)


def test_import_mean_is_multiplicative_in_customs_and_weekend():
    im = S.draw_import(40000, 903)
    cell = {(z, w): float(im.d[(im.zoll == z) & (im.we == w)].mean()) for z in (0, 1) for w in (0, 1)}
    assert cell[(1, 0)] / cell[(0, 0)] == pytest.approx(math.exp(0.7), rel=0.04)
    assert cell[(0, 1)] / cell[(0, 0)] == pytest.approx(math.exp(0.3), rel=0.04)
    assert cell[(1, 1)] / cell[(0, 0)] == pytest.approx(math.exp(1.0), rel=0.06)


def test_standgeld_in_import_is_an_exact_probit_in_customs_weekend_and_customer():
    n = 30000
    im = S.draw_import(n, 904)
    y = im.standgeld()
    X = np.column_stack([np.ones(n), im.zoll, im.we, S.kunden_dummies(im.kunde)])
    f = A.fit_irls(X, y, "binomial", "probit", tol=1e-10, maxit=60)
    assert f.converged
    s = S.S_BASE[0]
    k = np.arange(1, S.K_KUNDEN)
    truth = np.r_[(S.LN35 - S.LN7 + S.UK[0]) / s, 0.7 / s, 0.3 / s, (S.UK[k] - S.UK[0]) / s]
    se = np.sqrt(np.diag(np.linalg.inv(f.info)))
    assert np.max(np.abs((f.beta - truth) / se)) < 4.5                         # 14 Koeffizienten innerhalb von 4,5 Standardfehlern
    p_true = stats.norm.cdf((im.mu - S.LN7) / im.s)
    assert float(y.mean()) == pytest.approx(float(p_true.mean()), abs=3 * math.sqrt(p_true.mean() * (1 - p_true.mean()) / n))
    assert im.wahre_wahrscheinlichkeit() == pytest.approx(p_true, abs=1e-12)
