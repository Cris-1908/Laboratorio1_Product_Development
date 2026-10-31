"""Piezas compartidas por los DAGs. Aquí NO hay lógica de negocio: solo rutas, el asset y la traducción
del contexto de Airflow a argumentos simples (la lógica vive en src/inventario/)."""
from __future__ import annotations

import os
from pathlib import Path

from airflow.sdk import Asset

from inventario.orquestacion import corte_desde_contexto

PROYECTO = Path(os.environ.get("PROYECTO_DIR", "/opt/airflow/proyecto"))
DATA = PROYECTO / "data"

# La ingesta lo emite al terminar bien; preparación de datos y seguimiento se disparan con él.
BANDEJA_PROCESADA = Asset(name="bandeja_procesada", uri=f"file://{DATA}/procesado")

ARGS_DEFAULT = {"retries": 1}


def corte_de(contexto: dict) -> str:
    """Fecha de corte de esta corrida (ver inventario.orquestacion.corte_desde_contexto)."""
    run = contexto.get("dag_run")
    return corte_desde_contexto(
        eventos_asset=contexto.get("triggering_asset_events"),
        logical_date=contexto.get("logical_date"),
        run_after=getattr(run, "run_after", None),
        data_dir=DATA,
    )
