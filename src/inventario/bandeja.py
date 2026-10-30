"""Simulador del sistema de origen: carga histórica + bandeja de archivos.

Salida (bajo `salida`):
  historico/            carga inicial (hasta `corte_historico`)
  bandeja/<fuente>/<fecha>_v<k>.csv   un archivo por fecha, fuente y versión
  bandeja/manifest.csv  fuente, fecha_negocio, version, archivo, llegada_ts
  anomalias_verdad.csv  (fuera de la bandeja) qué se rompió a propósito, para probar

Reglas de la simulación:
  - ventas y transacciones llegan D+rezago a las 06:00.
  - petróleo llega con rezago; feriados llegan con anticipación.
  - correcciones: la v1 llega distorsionada y días después llega la v2 con el dato real.
  - con `con_anomalias=True` se inyectan archivos defectuosos (para probar validación).
Todo es determinista dado `seed`.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HORA_LLEGADA = 6
TIPOS_ANOMALIA = (
    "llegada_tardia", "faltante", "duplicados",
    "negativos", "fecha_incorrecta", "columna_faltante",
)


def _llegada(fecha, dias: int) -> pd.Timestamp:
    return pd.Timestamp(fecha) + pd.Timedelta(days=dias, hours=HORA_LLEGADA)


def _distorsionar(df: pd.DataFrame, col: str, rng: np.random.Generator, entero: bool) -> pd.DataFrame:
    """v1 defectuosa: ~10% de las filas con valor alterado."""
    d = df.reset_index(drop=True).copy()
    n = max(1, int(len(d) * 0.1))
    idx = rng.choice(len(d), size=n, replace=False)
    vals = d[col].to_numpy(dtype=float).copy()
    vals[idx] = vals[idx] * rng.uniform(0.6, 1.6, n)
    d[col] = np.round(vals).astype(int) if entero else np.round(vals, 3)
    return d


def _anomalia(df: pd.DataFrame, tipo: str, rng: np.random.Generator) -> pd.DataFrame:
    d = df.reset_index(drop=True).copy()
    if tipo == "duplicados":
        extra = d.iloc[rng.choice(len(d), size=min(20, len(d)), replace=False)]
        return pd.concat([d, extra], ignore_index=True)
    if tipo == "negativos":
        idx = rng.choice(len(d), size=min(5, len(d)), replace=False)
        d.loc[idx, "sales"] = -(d.loc[idx, "sales"].abs() + 1)
        return d
    if tipo == "fecha_incorrecta":
        d["date"] = d["date"] + pd.Timedelta(days=1)
        return d
    if tipo == "columna_faltante":
        return d.drop(columns=["onpromotion"])
    return d  # llegada_tardia / faltante no cambian el contenido


def construir_bandeja(raw_dir, salida, cfg: dict, con_anomalias: bool = False) -> pd.DataFrame:
    raw_dir, salida = Path(raw_dir), Path(salida)
    b = cfg["bandeja"]
    fu = cfg["fuentes"]
    rng = np.random.default_rng(b["seed"])
    corte = pd.Timestamp(b["corte_historico"])
    cola = int(b["dias_ultimos_sin_tocar"])

    train = pd.read_csv(raw_dir / "train.csv", parse_dates=["date"])
    trans = pd.read_csv(raw_dir / "transactions.csv", parse_dates=["date"])
    oil = pd.read_csv(raw_dir / "oil.csv", parse_dates=["date"])
    hol = pd.read_csv(raw_dir / "holidays_events.csv", parse_dates=["date"])
    stores = pd.read_csv(raw_dir / "stores.csv")

    hist_dir, bandeja = salida / "historico", salida / "bandeja"
    for d in (hist_dir, bandeja):
        if d.exists():
            shutil.rmtree(d)  # el generador es dueño de estos directorios; el flujo nunca escribe aquí
        d.mkdir(parents=True)

    # --- Carga histórica ---
    train[train.date <= corte].to_csv(hist_dir / "train.csv", index=False)
    trans[trans.date <= corte].to_csv(hist_dir / "transactions.csv", index=False)
    oil[oil.date <= corte].to_csv(hist_dir / "oil.csv", index=False)
    hol[hol.date <= corte].to_csv(hist_dir / "holidays_events.csv", index=False)
    stores.to_csv(hist_dir / "stores.csv", index=False)

    manifest: list[dict] = []
    verdad: list[dict] = []

    def emitir(df, fuente, fecha, version, llegada):
        nombre = f"{pd.Timestamp(fecha):%Y-%m-%d}_v{version}.csv"
        (bandeja / fuente).mkdir(parents=True, exist_ok=True)
        df.to_csv(bandeja / fuente / nombre, index=False)
        manifest.append(dict(
            fuente=fuente, fecha_negocio=f"{pd.Timestamp(fecha):%Y-%m-%d}",
            version=version, archivo=f"{fuente}/{nombre}", llegada_ts=llegada.isoformat(),
        ))

    # --- Ventas y transacciones: por fecha ---
    ventas = {f: g for f, g in train[train.date > corte].groupby("date")}
    transac = {f: g for f, g in trans[trans.date > corte].groupby("date")}
    f_ventas = sorted(ventas)
    f_trans = sorted(transac)

    def elegir(fechas: list, n: int) -> list:
        pool = fechas[: max(0, len(fechas) - cola)]
        if n > len(pool):
            raise ValueError(f"Se piden {n} fechas pero solo hay {len(pool)} candidatas")
        return [pool[i] for i in rng.permutation(len(pool))[:n]]

    corr_v = set(elegir(f_ventas, b["n_correcciones_ventas"]))
    corr_t = set(elegir(f_trans, b["n_correcciones_transacciones"]))

    # anomalías solo sobre ventas, en fechas distintas a las corregidas
    anom: dict = {}
    if con_anomalias:
        libres = [f for f in f_ventas if f not in corr_v]
        total = sum(b["anomalias"].get(t, 0) for t in TIPOS_ANOMALIA)
        pool = elegir(libres, total)
        k = 0
        for tipo in TIPOS_ANOMALIA:
            for _ in range(b["anomalias"].get(tipo, 0)):
                anom[pool[k]] = tipo
                k += 1

    rez_v = fu["ventas"]["rezago_dias"]
    for f in f_ventas:
        g = ventas[f]
        base = _llegada(f, rez_v)
        if f in corr_v:
            emitir(_distorsionar(g, "sales", rng, entero=False), "ventas", f, 1, base)
            emitir(g, "ventas", f, 2, base + pd.Timedelta(days=int(rng.integers(3, 11))))
            verdad.append(dict(fuente="ventas", fecha_negocio=f"{f:%Y-%m-%d}", tipo="correccion"))
            continue
        tipo = anom.get(f)
        if tipo == "faltante":
            verdad.append(dict(fuente="ventas", fecha_negocio=f"{f:%Y-%m-%d}", tipo=tipo))
            continue
        if tipo == "llegada_tardia":
            base = base + pd.Timedelta(days=4)
        df = _anomalia(g, tipo, rng) if tipo else g
        emitir(df, "ventas", f, 1, base)
        if tipo:
            verdad.append(dict(fuente="ventas", fecha_negocio=f"{f:%Y-%m-%d}", tipo=tipo))

    rez_t = fu["transacciones"]["rezago_dias"]
    for f in f_trans:
        g = transac[f]
        base = _llegada(f, rez_t)
        if f in corr_t:
            emitir(_distorsionar(g, "transactions", rng, entero=True), "transacciones", f, 1, base)
            emitir(g, "transacciones", f, 2, base + pd.Timedelta(days=int(rng.integers(3, 11))))
            verdad.append(dict(fuente="transacciones", fecha_negocio=f"{f:%Y-%m-%d}", tipo="correccion"))
        else:
            emitir(g, "transacciones", f, 1, base)

    # --- Petróleo: un archivo por fecha, con rezago (los NaN son esperados) ---
    for f, g in oil[oil.date > corte].groupby("date"):
        emitir(g, "petroleo", f, 1, _llegada(f, fu["petroleo"]["rezago_dias"]))

    # --- Feriados: un archivo por fecha, llegan con anticipación ---
    ant = fu["feriados"]["anticipacion_dias"]
    for f, g in hol[hol.date > corte].groupby("date"):
        emitir(g, "feriados", f, 1, _llegada(f, -ant))

    m = pd.DataFrame(manifest).sort_values(["llegada_ts", "fuente", "fecha_negocio", "version"]).reset_index(drop=True)
    m.to_csv(bandeja / "manifest.csv", index=False)
    pd.DataFrame(verdad, columns=["fuente", "fecha_negocio", "tipo"]).to_csv(salida / "anomalias_verdad.csv", index=False)
    return m
