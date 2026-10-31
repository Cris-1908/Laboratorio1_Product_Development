"""Lectura "as-of corte": solo lo que ya había llegado y era válido al corte.

Convenciones:
  - `corte` sin hora (fecha pura) significa fin de ese día (23:59:59): información
    disponible al cierre del día. Con hora explícita se usa tal cual.
  - Por (fuente, fecha_negocio) se toma la última versión VÁLIDA con llegada_ts <= corte.
  - Solo se leen archivos que pasaron la validación (data/procesado). Lo que está en
    cuarentena nunca se lee.
  - Cada fila lleva `available_at` y `version`. La carga histórica tiene version=0.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ARCHIVO_HISTORICO = {
    "ventas": "train.csv",
    "transacciones": "transactions.csv",
    "petroleo": "oil.csv",
    "feriados": "holidays_events.csv",
}
EPOCH = pd.Timestamp("1970-01-01")


def corte_ts(corte) -> pd.Timestamp:
    t = pd.Timestamp(corte)
    if t == t.normalize():
        t = t + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return t


def cargar_manifest(bandeja_dir) -> pd.DataFrame:
    m = pd.read_csv(Path(bandeja_dir) / "manifest.csv", parse_dates=["llegada_ts", "fecha_negocio"])
    return m


def leer_asof(fuente: str, corte, data_dir) -> pd.DataFrame:
    data_dir = Path(data_dir)
    corte = corte_ts(corte)

    hist = pd.read_csv(data_dir / "historico" / ARCHIVO_HISTORICO[fuente], parse_dates=["date"])
    hist["available_at"] = EPOCH
    hist["version"] = 0

    m = cargar_manifest(data_dir / "bandeja")
    m = m[(m.fuente == fuente) & (m.llegada_ts <= corte)]
    m = m[m.archivo.map(lambda a: (data_dir / "procesado" / a).exists()).astype(bool)]
    m = m.sort_values(["llegada_ts", "version"]).groupby("fecha_negocio").tail(1)

    partes = [hist]
    for r in m.itertuples():
        df = pd.read_csv(data_dir / "procesado" / r.archivo, parse_dates=["date"])
        df["available_at"] = r.llegada_ts
        df["version"] = r.version
        partes.append(df)
    return pd.concat(partes, ignore_index=True).sort_values("date", kind="stable").reset_index(drop=True)


def ultimo_corte_procesado(data_dir) -> str | None:
    """Fecha (YYYY-MM-DD) de la última llegada ya validada y procesada; None si no hay nada."""
    data_dir = Path(data_dir)
    m = cargar_manifest(data_dir / "bandeja")
    m = m[m.archivo.map(lambda a: (data_dir / "procesado" / a).exists()).astype(bool)]
    return None if m.empty else f"{m.llegada_ts.max():%Y-%m-%d}"
