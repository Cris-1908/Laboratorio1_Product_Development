"""Lógica de orquestación SIN dependencia de Airflow (para poder probarla con pytest).

Los DAGs solo traducen el contexto de Airflow a estos argumentos y llaman a las funciones del paquete.
"""
from __future__ import annotations

import pandas as pd

from .lectura import ultimo_corte_procesado


def corte_desde_contexto(eventos_asset=None, logical_date=None, run_after=None, data_dir=None) -> str:
    """Decide la fecha de corte (YYYY-MM-DD, fin de día) de una corrida:
    1) el corte que anunció la ingesta en el evento de asset que disparó la corrida (el MÁS RECIENTE
       si se acumularon varios);
    2) el logical_date de corridas programadas o de backfill;
    3) run_after (corrida manual sin logical_date);
    4) el último día ya ingerido y validado."""
    cortes = []
    for eventos in (eventos_asset or {}).values():
        for e in eventos:
            extra = getattr(e, "extra", None) or {}
            if extra.get("corte"):
                cortes.append(str(extra["corte"]))
    if cortes:
        return max(cortes)
    for fecha in (logical_date, run_after):
        if fecha is not None:
            return pd.Timestamp(fecha).strftime("%Y-%m-%d")
    ultimo = ultimo_corte_procesado(data_dir) if data_dir is not None else None
    if ultimo is None:
        raise ValueError("No hay fecha de corte: sin evento, sin logical_date y sin datos procesados")
    return ultimo
