"""Randfälle (DEMO-PLAYBOOK §12): n = p, n < p, konstante und kollineare Spalten, y = 0, Gamma mit y ≤ 0, ungültiges Mittel (Identität, Kehrwert), Separation, Nicht-Konvergenz,
Schritthalbierung, leere und kleine Zellen, Konvergenzkriterium am vollen Schritt."""

import numpy as np
import pytest

import gfm_algorithm as A
import gfm_inference as I
import gfm_scenario as S


def test_n_equal_p_saturates_the_model():
    y = np.array([2.0, 3.0, 5.0])
    f = A.fit_irls(np.eye(3), y, "poisson", "log", "fisher", tol=1e-12, maxit=60)
    assert f.converged and f.mu == pytest.approx(y, abs=1e-8) and f.deviance == pytest.approx(0.0, abs=1e-8)


def test_n_smaller_than_p_and_collinear_columns_are_reported_not_hidden():
    X = np.array([[1.0, 2.0, 4.0], [1.0, 3.0, 6.0]])                 # zwei Zeilen, drei Parameter: X'WX singulär
    f = A.fit_irls(X, np.array([1.0, 2.0]), "poisson", "log")
    assert not f.converged and f.message
    Xc = np.column_stack([np.ones(6), np.arange(6.0), 2 * np.arange(6.0)])   # kollineare Spalten
    g = A.fit_irls(Xc, np.array([1.0, 2.0, 2.0, 4.0, 5.0, 7.0]), "poisson", "log")
    assert not g.converged and g.message
    assert not np.any(np.isinf(g.beta))                              # nie ∞ als Zahl


def test_constant_column_next_to_the_intercept_is_singular():
    X = np.column_stack([np.ones(5), np.full(5, 3.0)])
    f = A.fit_irls(X, np.array([1.0, 2.0, 4.0, 3.0, 5.0]), "gamma", "log")
    assert not f.converged and f.message


def test_poisson_with_zero_counts_starts_at_y_plus_half():
    y = np.array([0.0, 0.0, 1.0, 3.0])
    assert A.start_mu("poisson", y) == pytest.approx([0.5, 0.5, 1.5, 3.5])
    assert A.start_mu("binomial", np.array([0.0, 1.0])) == pytest.approx([0.25, 0.75])
    assert A.start_mu("gamma", np.array([2.0, 4.0])) == pytest.approx([2.0, 4.0])
    f = A.fit_irls(np.ones((4, 1)), y, "poisson", "log")
    assert f.converged and f.mu == pytest.approx(1.0, abs=1e-8)       # das Mittel 1 (4 / 4)


def test_gamma_needs_positive_observations():
    f = A.fit_irls(np.ones((3, 1)), np.array([1.0, 0.0, 2.0]), "gamma", "log")
    assert not f.converged                                           # y = 0: Gamma-Deviance und Gewichte nicht definiert, Meldung statt Zahl
    assert f.message


def test_identity_link_with_negative_fitted_mean_is_reported_or_halved():
    """OLS (Normal/Identität) auf der Container-Tabelle sagt negative Verweilzeiten voraus; bei Poisson mit Identität verhindert die Schritthalbierung ein negatives Mittel."""
    T = S.draw_table(3000, 180)
    X = T.designmatrix()
    ols = A.fit_irls(X, T.d, "normal", "identity")
    assert ols.converged and float(ols.mu.min()) < 0                 # Normal/Identität: das kleinste angepasste Mittel ist negativ
    p = A.fit_irls(X, T.standtage(), "poisson", "identity", "fisher", maxit=1000)
    assert p.converged and float(p.mu.min()) > 0


def test_inverse_link_with_nonpositive_eta_gives_a_message_never_inf():
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 4.0, 8.0])
    f = A.fit_irls(X, y, "gamma", "inverse", "fisher", beta0=[-1.0, 0.1])   # η < 0 beim Start: Mittel unzulässig
    assert not f.converged and f.message
    assert np.all(np.isfinite(f.beta))


def test_separation_has_no_limit_and_is_reported_after_the_step_budget():
    X = np.array([[1, -2], [1, -1], [1, 1], [1, 2]], float)
    y = np.array([0, 0, 1, 1], float)
    f = A.fit_irls(X, y, "binomial", "logit", "fisher", tol=1e-10, maxit=25, beta0=[0, 0], halve=False)
    assert not f.converged and "nicht konvergiert" in f.message
    assert f.path[-1][1] > f.path[10][1] > f.path[5][1] > 0           # der Koeffizient wächst mit jedem Schritt
    assert f.deviance < 1e-6                                          # die Deviance läuft gegen 0, nicht gegen einen Grenzwert des Koeffizienten


def test_budget_of_steps_is_reported_as_non_convergence():
    X, y = np.ones((4, 1)), np.array([4.0, 5.0, 6.0, 9.0])
    f = A.fit_irls(X, y, "poisson", "identity", "newton", tol=1e-15, maxit=2, beta0=[5.0])
    assert not f.converged and f.iterations == 2 and "nicht konvergiert" in f.message


def test_convergence_is_judged_on_the_full_step_not_the_halved_one():
    """Mit Schritthalbierung steht in `steps` der volle Schritt; `converged` heißt: der volle Schritt war unter der Toleranz."""
    T = S.draw_table(500, 11)
    f = A.fit_irls(T.designmatrix(), T.standtage(), "poisson", "identity", "fisher", tol=1e-10, maxit=1000, halve=True)
    assert f.converged and f.steps[-1] < 1e-10 and all(s >= 1e-10 for s in f.steps[:-1])


def test_probit_and_cloglog_probabilities_are_clipped_away_from_zero_and_one():
    assert A.link_h("probit", np.array([-40.0, 40.0])) == pytest.approx([1e-15, 1 - 1e-15], abs=1e-16)
    assert A.link_h("cloglog", np.array([-800.0, 60.0])) == pytest.approx([1e-15, 1 - 1e-15], abs=1e-16)
    assert A.link_h("logit", np.array([-100.0, 100.0])) == pytest.approx(A.link_h("logit", np.array([-30.0, 30.0])))   # Kappung |η| ≤ 30


def test_grouped_deviance_ignores_empty_cells_and_counts_small_cells():
    y = np.array([0, 1, 1, 0, 1, 0], float)
    key = np.array([0, 0, 3, 3, 3, 5])                               # Zellen 1, 2, 4 bleiben leer
    p = np.array([0.4, 0.4, 0.6, 0.6, 0.6, 0.3])
    g = I.grouped_binomial(key, y, p, 8)
    assert g["cells"] == 3
    relabelled = I.grouped_binomial(np.array([0, 0, 1, 1, 1, 2]), y, p, 3)
    assert g["dev"] == pytest.approx(relabelled["dev"], abs=1e-12) and g["small"] == 1.0   # alle erwarteten Zahlen unter 5


def test_lr_test_rejects_exactly_above_the_chi2_quantile():
    crit = I.chi2_crit(11, 0.95)
    assert I.lr_test(crit + 1e-9, 0.0, 11)["reject"] and not I.lr_test(crit - 1e-9, 0.0, 11)["reject"]
    assert crit == pytest.approx(19.6751, abs=1e-3)


def test_hand_data_have_the_stated_shape():
    h = S.HAND_NEUN
    assert h.n == 9 and h.X.shape == (9, 2)
    assert list(h.standtage) == [4, 5, 6, 9, 7, 9, 10, 12, 14] and list(h.standgeld) == [0, 0, 0, 1, 0, 1, 1, 1, 1]
    assert S.HAND_VIER.n == 4
