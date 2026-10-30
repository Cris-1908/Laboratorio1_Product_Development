"""Reglas de validación por fuente.

errores      -> el archivo es inválido: va a cuarentena y la tarea falla.
advertencias -> situación esperada del negocio: se registra, NO falla
                (petróleo faltante, día completo en cero, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REGLAS = {
    "ventas": dict(
        cols=["id", "date", "store_nbr", "family", "sales", "onpromotion"],
        llave=["date", "store_nbr", "family"],
        numericas=["store_nbr", "sales", "onpromotion"],
        no_negativas=["sales", "onpromotion"],
        sin_nulos=["date", "store_nbr", "family", "sales", "onpromotion"],
    ),
    "transacciones": dict(
        cols=["date", "store_nbr", "transactions"],
        llave=["date", "store_nbr"],
        numericas=["store_nbr", "transactions"],
        no_negativas=["transactions"],
        sin_nulos=["date", "store_nbr", "transactions"],
    ),
    "petroleo": dict(
        cols=["date", "dcoilwtico"],
        llave=["date"],
        numericas=["dcoilwtico"],
        no_negativas=[],
        sin_nulos=["date"],
    ),
    "feriados": dict(
        cols=["date", "type", "locale", "locale_name", "description", "transferred"],
        llave=None,
        numericas=[],
        no_negativas=[],
        sin_nulos=["date", "type"],
    ),
}


@dataclass
class Resultado:
    errores: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errores


def validar_df(df: pd.DataFrame, fuente: str, fecha_negocio) -> Resultado:
    r = REGLAS[fuente]
    res = Resultado()
    fecha_negocio = pd.Timestamp(fecha_negocio).normalize()

    if df.empty:
        res.errores.append("archivo vacío")
        return res

    faltan = [c for c in r["cols"] if c not in df.columns]
    if faltan:
        res.errores.append(f"columnas faltantes: {faltan}")
        return res  # sin esquema no tiene sentido seguir

    for c in r["sin_nulos"]:
        n = int(df[c].isna().sum())
        if n:
            res.errores.append(f"{n} nulos en '{c}'")

    for c in r["numericas"]:
        num = pd.to_numeric(df[c], errors="coerce")
        no_num = int((num.isna() & df[c].notna()).sum())
        if no_num:
            res.errores.append(f"{no_num} valores no numéricos en '{c}'")
        elif c in r["no_negativas"]:
            n = int((num < 0).sum())
            if n:
                res.errores.append(f"{n} valores negativos en '{c}'")

    fechas = pd.to_datetime(df["date"], errors="coerce")
    if fechas.isna().any():
        res.errores.append(f"{int(fechas.isna().sum())} fechas no interpretables")
    elif (fechas.dt.normalize() != fecha_negocio).any():
        distintas = sorted({f"{x:%Y-%m-%d}" for x in fechas.dt.normalize().unique()})
        res.errores.append(f"fecha del contenido {distintas} no coincide con la del archivo {fecha_negocio:%Y-%m-%d}")

    if r["llave"]:
        n = int(df.duplicated(r["llave"]).sum())
        if n:
            res.errores.append(f"{n} filas duplicadas por {r['llave']}")

    # --- esperado del negocio: se registra, no falla ---
    if fuente == "ventas" and "sales" in df and pd.to_numeric(df["sales"], errors="coerce").fillna(0).sum() == 0:
        res.advertencias.append("día completo con ventas en cero")
    if fuente == "petroleo" and df["dcoilwtico"].isna().any():
        res.advertencias.append("precio de petróleo faltante (esperado en feriados/fines de semana)")
    return res
