"""Seguimiento del desempeño: pronósticos ya publicados vs. ventas reales a medida que llegan,
y verificación de que un pronóstico pasado se reproduce exactamente.

Reglas:
  - Solo se evalúan pronósticos con corte <= corte del seguimiento.
  - Las ventas reales son las disponibles AL corte del seguimiento (incluye correcciones ya llegadas).
  - Se usa el pedido PUBLICADO (con los costos vigentes en ese momento), no se recalcula con costos nuevos.
  - Un pronóstico sin ventas reales aún queda en la tabla con n_evaluadas = 0 (no es un error).
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .lectura import corte_ts
from .modelos import (
    Panel, _escribir_atomico, construir_panel, costo_negocio, fecha_origen, generar_pronostico, pinball,
    sesgo, valor_real, wape,
)

COLUMNAS = ["corte_seguimiento", "corte_pronostico", "modelo", "fecha_min", "fecha_max", "n_total", "n_evaluadas",
            "cobertura_datos", "pinball", "wape", "sesgo", "costo", "costo_por_unidad", "cobertura_intervalo",
            "nominal_intervalo"]


def evaluar_publicado(df: pd.DataFrame, real: Panel | None) -> dict:
    """Métricas de un pronóstico publicado contra las ventas reales conocidas (NaN = aún no llegó)."""
    qcols = sorted([c for c in df.columns if re.fullmatch(r"q\d\d", c)], key=lambda c: int(c[1:]))
    qs = [int(c[1:]) / 100 for c in qcols]
    n_total = int(len(df))
    vacio = dict(n_total=n_total, n_evaluadas=0, cobertura_datos=0.0, pinball=np.nan, wape=np.nan, sesgo=np.nan,
                 costo=np.nan, costo_por_unidad=np.nan, cobertura_intervalo=np.nan, nominal_intervalo=qs[-1] - qs[0])
    if real is None:
        return vacio
    y = valor_real(real, df)
    m = ~np.isnan(y)
    if not m.any():
        return vacio
    Q = df[qcols].to_numpy(float)[m]
    p50 = Q[:, qs.index(0.5)]
    costo = costo_negocio(y[m], df.pedido.to_numpy(float)[m], df.cu.to_numpy(float)[m], df.co.to_numpy(float)[m])
    return dict(n_total=n_total, n_evaluadas=int(m.sum()), cobertura_datos=float(m.mean()),
                pinball=pinball(y[m], Q, qs), wape=wape(y[m], p50), sesgo=sesgo(y[m], p50), costo=costo,
                costo_por_unidad=costo / max(float(y[m].sum()), 1e-9),
                cobertura_intervalo=float(np.mean((y[m] >= Q[:, 0]) & (y[m] <= Q[:, -1]))),
                nominal_intervalo=qs[-1] - qs[0])


def seguimiento_desempeno(corte, data_dir) -> pd.DataFrame:
    data_dir = Path(data_dir)
    corte = corte_ts(corte)
    pubs = []
    for c in sorted((data_dir / "forecasts").glob("corte=*")):
        f_corte = pd.Timestamp(c.name.split("=", 1)[1])
        if f_corte.normalize() <= corte.normalize() and (c / "pronostico.csv").exists():
            pubs.append((f_corte, pd.read_csv(c / "pronostico.csv", parse_dates=["fecha"])))
    if not pubs:
        return pd.DataFrame(columns=COLUMNAS)

    origen = fecha_origen(corte)
    primera = min(df.fecha.min() for _, df in pubs)
    real = None
    if primera <= origen:  # hay al menos un día de ese pronóstico que ya debió llegar
        real = construir_panel(corte, data_dir, dias=int((origen - primera).days) + 2)

    filas = []
    for f_corte, df in pubs:
        res = evaluar_publicado(df, real)
        filas.append(dict(corte_seguimiento=f"{corte:%Y-%m-%d}", corte_pronostico=f"{f_corte:%Y-%m-%d}",
                          modelo=str(df.modelo.iloc[0]), fecha_min=f"{df.fecha.min():%Y-%m-%d}",
                          fecha_max=f"{df.fecha.max():%Y-%m-%d}", **res))
    return pd.DataFrame(filas, columns=COLUMNAS)


def publicar_seguimiento(df: pd.DataFrame, corte, data_dir) -> Path:
    ruta = Path(data_dir) / "seguimiento" / f"corte={corte_ts(corte):%Y-%m-%d}" / "seguimiento.csv"
    _escribir_atomico(ruta, df.to_csv(index=False))
    return ruta


def _sha(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def verificar_reproduccion(corte, data_dir, cfg_modelo: dict, cfg_costos: dict) -> dict:
    """Regenera un pronóstico publicado desde cero (features y modelo nuevos, en un directorio
    temporal que solo enlaza los datos de entrada) y compara el contenido con lo publicado."""
    data_dir = Path(data_dir).resolve()
    c = f"{corte_ts(corte):%Y-%m-%d}"
    ruta = data_dir / "forecasts" / f"corte={c}" / "pronostico.csv"
    publicado = ruta.read_text(encoding="utf-8")
    modelo = str(pd.read_csv(ruta, nrows=1).modelo.iloc[0])
    with tempfile.TemporaryDirectory() as tmp:
        for carpeta in ("historico", "bandeja", "procesado"):
            os.symlink(data_dir / carpeta, Path(tmp) / carpeta)
        regenerado = generar_pronostico(c, tmp, cfg_modelo, cfg_costos,
                                        modelo="baseline" if modelo == "baseline" else "lgbm").to_csv(index=False)
    return dict(corte=c, modelo=modelo, igual=publicado == regenerado, sha_publicado=_sha(publicado),
                sha_regenerado=_sha(regenerado))
