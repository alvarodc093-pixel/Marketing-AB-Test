"""Las funciones propias dan lo mismo que las librerías de referencia."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ab.datos import efecto_por_segmento
from ab.estadistica import (
    Grupo,
    bayes_beta_binomial,
    bootstrap_diferencia,
    efecto_minimo_detectable,
    holm,
    potencia,
    tamano_muestral,
    prueba_srm,
    prueba_z,
)

sm = pytest.importorskip("statsmodels.api")
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest  # noqa: E402

A = Grupo(23_524, 420)       # los números reales del experimento
B = Grupo(564_577, 14_423)


def test_z_coincide_con_statsmodels():
    r = prueba_z(A, B)
    z, p = proportions_ztest([B.conversiones, A.conversiones], [B.n, A.n])
    assert r.z == pytest.approx(z)
    assert r.p_valor == pytest.approx(p)


def test_ic_diferencia_coincide_con_statsmodels_wald():
    r = prueba_z(A, B)
    bajo, alto = confint_proportions_2indep(B.conversiones, B.n, A.conversiones, A.n,
                                            method="wald", compare="diff")
    assert r.ic_diferencia == pytest.approx((bajo, alto))


def test_ic_lift_coincide_con_statsmodels_log():
    r = prueba_z(A, B)
    bajo, alto = confint_proportions_2indep(B.conversiones, B.n, A.conversiones, A.n,
                                            method="log", compare="ratio")
    assert r.ic_lift == pytest.approx((bajo - 1, alto - 1))


def test_sin_efecto_no_es_significativo():
    r = prueba_z(Grupo(10_000, 200), Grupo(10_000, 200))
    assert r.diferencia == 0 and r.p_valor == pytest.approx(1)


def test_grupo_sin_conversiones_no_rompe():
    r = prueba_z(Grupo(500, 0), Grupo(5_000, 30))
    assert np.isnan(r.lift) and r.diferencia > 0


def test_srm():
    assert prueba_srm([5_000, 5_000], [0.5, 0.5]) == pytest.approx(1)
    assert prueba_srm([5_300, 4_700], [0.5, 0.5]) < 0.001
    assert prueba_srm([400, 9_600], [4, 96]) == pytest.approx(1)


def test_bootstrap_parecido_al_ic_analitico():
    ca = np.r_[np.ones(A.conversiones), np.zeros(A.n - A.conversiones)]
    cb = np.r_[np.ones(B.conversiones), np.zeros(B.n - B.conversiones)]
    bajo, alto = bootstrap_diferencia(ca, cb)
    r = prueba_z(A, B)
    assert bajo == pytest.approx(r.ic_diferencia[0], abs=3e-4)
    assert alto == pytest.approx(r.ic_diferencia[1], abs=3e-4)


def test_bayes():
    r = bayes_beta_binomial(A, B)
    assert r.prob_b_mejor > 0.999
    assert r.lift_esperado == pytest.approx(prueba_z(A, B).lift, abs=0.01)
    iguales = bayes_beta_binomial(Grupo(10_000, 200), Grupo(10_000, 200))
    assert iguales.prob_b_mejor == pytest.approx(0.5, abs=0.01)


def test_potencia_coincide_con_statsmodels():
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    # statsmodels usa el arcoseno (h de Cohen): con n grandes las dos coinciden.
    p0, ef, n = 0.02, 0.004, 20_000
    mia = potencia(p0, ef, n, n)
    ref = NormalIndPower().power(proportion_effectsize(p0 + ef, p0), nobs1=n, alpha=0.05, ratio=1)
    assert mia == pytest.approx(ref, abs=0.01)


def test_mde_y_tamano_son_coherentes():
    mde = efecto_minimo_detectable(0.0179, A.n, B.n)
    assert potencia(0.0179, mde, A.n, B.n) == pytest.approx(0.8, abs=1e-6)
    n_a, n_b = tamano_muestral(0.02, 0.005)
    assert potencia(0.02, 0.005, n_a, n_b) == pytest.approx(0.8, abs=0.01)
    assert n_a == n_b


def test_holm_coincide_con_statsmodels():
    p = [0.01, 0.04, 0.03, 0.005, 0.2]
    assert holm(p) == pytest.approx(multipletests(p, method="holm")[1].tolist())


def test_efecto_por_segmento():
    df = pd.DataFrame({
        "grupo": ["control"] * 200 + ["anuncios"] * 200,
        "convertido": [1] * 10 + [0] * 90 + [1] * 10 + [0] * 90 + [1] * 30 + [0] * 70 + [1] * 10 + [0] * 90,
        "segmento": (["x"] * 100 + ["y"] * 100) * 2,
    })
    t = efecto_por_segmento(df, "segmento").set_index("segmento")
    assert t.loc["x", "diferencia_pp"] == pytest.approx(20)
    assert t.loc["y", "diferencia_pp"] == pytest.approx(0)
    assert t.loc["x", "significativo"] and not t.loc["y", "significativo"]
    assert t["pct_usuarios"].sum() == pytest.approx(1)


def test_chi2_igual_que_scipy():
    obs = [23_524, 564_577]
    assert prueba_srm(obs, [4, 96]) == pytest.approx(
        stats.chisquare(obs, np.array([0.04, 0.96]) * sum(obs)).pvalue)
