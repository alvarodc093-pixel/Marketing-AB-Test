"""Pruebas estadísticas para un A/B test de conversión (dos proporciones).

Todo está implementado con numpy y scipy, sin cajas negras, para poder
explicar cada número. Los tests comparan los resultados con statsmodels.

Convención: A = control, B = tratamiento.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Grupo:
    n: int              # usuarios
    conversiones: int

    @property
    def tasa(self) -> float:
        return self.conversiones / self.n


# ------------------------------------------------------------ frecuentista --

@dataclass(frozen=True)
class ResultadoZ:
    tasa_a: float
    tasa_b: float
    diferencia: float           # B - A, en proporción (0.0077 = 0,77 puntos)
    ic_diferencia: tuple[float, float]
    lift: float                 # (B - A) / A
    ic_lift: tuple[float, float]
    z: float
    p_valor: float


def prueba_z(a: Grupo, b: Grupo, alfa: float = 0.05) -> ResultadoZ:
    """Test z de dos proporciones (bilateral).

    - El estadístico usa la proporción conjunta (hipótesis nula: tasas iguales).
    - El intervalo de la diferencia usa el error estándar sin agrupar (Wald).
    - El intervalo del lift relativo se calcula sobre log(B/A) con el método
      delta y se vuelve a la escala original: es asimétrico, como debe ser.
    """
    pa, pb = a.tasa, b.tasa
    conjunta = (a.conversiones + b.conversiones) / (a.n + b.n)
    ee_nulo = np.sqrt(conjunta * (1 - conjunta) * (1 / a.n + 1 / b.n))
    z = (pb - pa) / ee_nulo
    p = 2 * stats.norm.sf(abs(z))

    zc = stats.norm.ppf(1 - alfa / 2)
    ee = np.sqrt(pa * (1 - pa) / a.n + pb * (1 - pb) / b.n)
    dif = pb - pa
    ic_dif = (dif - zc * ee, dif + zc * ee)

    if pa > 0 and pb > 0:
        log_ratio = np.log(pb / pa)
        ee_log = np.sqrt((1 - pa) / (a.n * pa) + (1 - pb) / (b.n * pb))
        ic_lift = (float(np.exp(log_ratio - zc * ee_log) - 1), float(np.exp(log_ratio + zc * ee_log) - 1))
        lift = pb / pa - 1
    else:  # sin conversiones en un grupo el lift relativo no está definido
        ic_lift, lift = (float("nan"), float("nan")), float("nan")

    return ResultadoZ(pa, pb, dif, (float(ic_dif[0]), float(ic_dif[1])), lift, ic_lift, float(z), float(p))


def prueba_srm(observados: list[int], proporciones_esperadas: list[float]) -> float:
    """Sample Ratio Mismatch: p-valor de que el reparto observado venga del diseñado."""
    obs = np.asarray(observados, dtype=float)
    esperados = obs.sum() * np.asarray(proporciones_esperadas) / np.sum(proporciones_esperadas)
    return float(stats.chisquare(obs, esperados).pvalue)


def bootstrap_diferencia(conv_a: np.ndarray, conv_b: np.ndarray, n_remuestras: int = 5000,
                         alfa: float = 0.05, semilla: int = 42) -> tuple[float, float]:
    """IC por percentiles de B - A remuestreando usuarios.

    Como la conversión es 0/1, remuestrear un grupo equivale a sacar una
    binomial con su tasa observada: mucho más rápido e idéntico en distribución.
    """
    rng = np.random.default_rng(semilla)
    na, nb = len(conv_a), len(conv_b)
    pa, pb = conv_a.mean(), conv_b.mean()
    difs = rng.binomial(nb, pb, n_remuestras) / nb - rng.binomial(na, pa, n_remuestras) / na
    return tuple(np.quantile(difs, [alfa / 2, 1 - alfa / 2]))


# --------------------------------------------------------------- bayesiano --

@dataclass(frozen=True)
class ResultadoBayes:
    prob_b_mejor: float
    lift_esperado: float
    ic_lift: tuple[float, float]       # intervalo creíble del 95 %
    perdida_esperada_b: float          # cuánta conversión se pierde si elijo B y era peor


def bayes_beta_binomial(a: Grupo, b: Grupo, muestras: int = 200_000, semilla: int = 42,
                        previa: tuple[float, float] = (1, 1)) -> ResultadoBayes:
    """Posteriores Beta con previa uniforme y comparación por Monte Carlo."""
    rng = np.random.default_rng(semilla)
    al, be = previa
    ta = rng.beta(al + a.conversiones, be + a.n - a.conversiones, muestras)
    tb = rng.beta(al + b.conversiones, be + b.n - b.conversiones, muestras)
    lift = tb / ta - 1
    return ResultadoBayes(
        prob_b_mejor=float((tb > ta).mean()),
        lift_esperado=float(lift.mean()),
        ic_lift=tuple(np.quantile(lift, [0.025, 0.975])),
        perdida_esperada_b=float(np.maximum(ta - tb, 0).mean()),
    )


# ---------------------------------------------------------------- potencia --

def potencia(p_base: float, efecto_abs: float, n_a: int, n_b: int, alfa: float = 0.05) -> float:
    """Potencia de un test z bilateral para detectar una diferencia absoluta dada."""
    p_b = p_base + efecto_abs
    ee = np.sqrt(p_base * (1 - p_base) / n_a + p_b * (1 - p_b) / n_b)
    zc = stats.norm.ppf(1 - alfa / 2)
    return float(stats.norm.sf(zc - abs(efecto_abs) / ee) + stats.norm.cdf(-zc - abs(efecto_abs) / ee))


def efecto_minimo_detectable(p_base: float, n_a: int, n_b: int, alfa: float = 0.05,
                             potencia_objetivo: float = 0.8) -> float:
    """Menor diferencia absoluta que se detectaría con la potencia pedida (búsqueda binaria)."""
    bajo, alto = 1e-7, 1 - p_base - 1e-9
    for _ in range(100):
        medio = (bajo + alto) / 2
        if potencia(p_base, medio, n_a, n_b, alfa) < potencia_objetivo:
            bajo = medio
        else:
            alto = medio
    return alto


def tamano_muestral(p_base: float, efecto_abs: float, ratio_b_a: float = 1.0,
                    alfa: float = 0.05, potencia_objetivo: float = 0.8) -> tuple[int, int]:
    """Usuarios necesarios en A y B (B = ratio x A) para detectar efecto_abs."""
    p_b = p_base + efecto_abs
    za = stats.norm.ppf(1 - alfa / 2)
    zb = stats.norm.ppf(potencia_objetivo)
    var = p_base * (1 - p_base) + p_b * (1 - p_b) / ratio_b_a
    n_a = int(np.ceil((za + zb) ** 2 * var / efecto_abs ** 2))
    return n_a, int(np.ceil(n_a * ratio_b_a))


# ------------------------------------------------------- comparaciones múltiples --

def holm(p_valores: list[float]) -> list[float]:
    """Corrección de Holm-Bonferroni: p-valores ajustados en el orden original."""
    p = np.asarray(p_valores, dtype=float)
    m = len(p)
    orden = np.argsort(p)
    ajustados = np.empty(m)
    maximo = 0.0
    for rango, i in enumerate(orden):
        maximo = max(maximo, min(1.0, (m - rango) * p[i]))
        ajustados[i] = maximo
    return ajustados.tolist()
