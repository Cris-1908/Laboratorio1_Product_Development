"""DAG 4: seguimiento del desempeño. Se dispara cuando llegan datos nuevos (asset de la ingesta).

Compara los pronósticos ya publicados con las ventas reales disponibles al corte (incluyendo
correcciones que ya llegaron). Un pronóstico cuyos días aún no llegan aparece con n_evaluadas = 0.
Escribe data/seguimiento/corte=<fecha>/seguimiento.csv (idempotente).
"""
from __future__ import annotations

import pendulum
from airflow.sdk import dag, task

from comun import ARGS_DEFAULT, BANDEJA_PROCESADA, DATA, corte_de


@dag(
    dag_id="seguimiento_desempeno",
    schedule=[BANDEJA_PROCESADA],
    start_date=pendulum.datetime(2016, 12, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=ARGS_DEFAULT,
    tags=["lab1", "seguimiento"],
)
def seguimiento_desempeno():
    @task
    def medir(**contexto):
        from inventario.seguimiento import publicar_seguimiento, seguimiento_desempeno as medir_desempeno

        corte = corte_de(contexto)
        df = medir_desempeno(corte, DATA)
        ruta = publicar_seguimiento(df, corte, DATA)
        print(df.round(4).to_string(index=False) if len(df) else "Sin pronósticos publicados todavía")
        return str(ruta)

    medir()


seguimiento_desempeno()
