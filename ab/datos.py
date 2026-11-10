"""Carga y limpieza del dataset y tablas por segmento."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ab.estadistica import Grupo, holm, prueba_z

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "data" / "marketing_AB.csv"
URL = "https://www.kaggle.com/api/v1/datasets/download/faviovaz/marketing-ab-testing"

DIAS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DIAS_ES = dict(zip(DIAS, ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]))
TRAMOS_ANUNCIOS = [0, 9, 24, 49, 99, 199, np.inf]
ETIQUETAS_TRAMOS = ["1-9", "10-24", "25-49", "50-99", "100-199", "200 o más"]
FRANJAS = ["madrugada (0-5 h)", "mañana (6-11 h)", "tarde (12-17 h)", "noche (18-23 h)"]


def descargar(destino: Path = CSV) -> None:
    """Baja el CSV público de Kaggle (licencia CC0) si no está ya."""
    if destino.exists():
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as r:
        with zipfile.ZipFile(io.BytesIO(r.read())) as z:
            destino.write_bytes(z.read("marketing_AB.csv"))


def cargar(ruta: Path = CSV) -> pd.DataFrame:
    df = pd.read_csv(ruta, index_col=0)
    df = df.rename(columns={
        "user id": "usuario", "test group": "grupo", "converted": "convertido",
        "total ads": "impresiones", "most ads day": "dia_max", "most ads hour": "hora_max",
    })
    df["grupo"] = df["grupo"].map({"psa": "control", "ad": "anuncios"})
    df["convertido"] = df["convertido"].astype(int)
    df["dia_max"] = pd.Categorical(df["dia_max"].map(DIAS_ES), categories=list(DIAS_ES.values()), ordered=True)
    df["tramo_impresiones"] = pd.cut(df["impresiones"], TRAMOS_ANUNCIOS, labels=ETIQUETAS_TRAMOS)
    df["franja_horaria"] = pd.Categorical(
        pd.cut(df["hora_max"], [-1, 5, 11, 17, 23], labels=FRANJAS).astype(str),
        categories=FRANJAS, ordered=True)
    return df


def grupos(df: pd.DataFrame) -> tuple[Grupo, Grupo]:
    """(control, anuncios)."""
    res = df.groupby("grupo")["convertido"].agg(["size", "sum"])
    return (Grupo(int(res.loc["control", "size"]), int(res.loc["control", "sum"])),
            Grupo(int(res.loc["anuncios", "size"]), int(res.loc["anuncios", "sum"])))


def efecto_por_segmento(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Tasa de cada grupo, lift, IC y p-valor (bruto y ajustado por Holm) en cada segmento.

    Un segmento sin usuarios de alguno de los dos grupos no se puede comparar: se omite
    (y no cuenta en la corrección de Holm).
    """
    filas = []
    for valor, sub in df.groupby(columna, observed=True):
        if sub["grupo"].nunique() < 2:
            continue
        a, b = grupos(sub)
        r = prueba_z(a, b)
        filas.append({
            columna: valor, "usuarios": a.n + b.n, "pct_usuarios": (a.n + b.n) / len(df),
            "conv_control": r.tasa_a, "conv_anuncios": r.tasa_b,
            "diferencia_pp": 100 * r.diferencia,
            "ic_bajo_pp": 100 * r.ic_diferencia[0], "ic_alto_pp": 100 * r.ic_diferencia[1],
            "lift": r.lift, "p_valor": r.p_valor,
        })
    tabla = pd.DataFrame(filas)
    tabla["p_holm"] = holm(tabla["p_valor"].tolist())
    tabla["significativo"] = tabla["p_holm"] < 0.05
    return tabla
