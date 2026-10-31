"""DAG 1: ingesta y validación de las fuentes (diario).

- Lee la bandeja (solo lectura), valida cada archivo y deja lo válido en data/procesado y lo inválido en
  data/cuarentena.
- Si llegó algo inválido en las últimas 24 h, la tarea FALLA: no se emite el asset y los DAGs que dependen
  de él no corren sobre datos dudosos. Lo inválido nunca se lee (queda fuera del lector as-of).
- Situaciones esperadas del negocio (petróleo faltante, 25-dic sin ventas, ceros) solo generan advertencias.
- Idempotente. Backfill: airflow backfill create --dag-id ingesta_validacion --from-date ... --to-date ...
"""
from __future__ import annotations

import pendulum
from airflow.sdk import dag, task

from comun import ARGS_DEFAULT, BANDEJA_PROCESADA, DATA, corte_de


@dag(
    dag_id="ingesta_validacion",
    schedule="@daily",
    # las primeras llegadas (feriados) ocurren ~30 días antes del primer día de la bandeja
    start_date=pendulum.datetime(2016, 12, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=ARGS_DEFAULT,
    tags=["lab1", "ingesta"],
)
def ingesta_validacion():
    @task(outlets=[BANDEJA_PROCESADA])
    def ingestar_y_validar(*, outlet_events, **contexto):
        import pandas as pd

        from inventario.config import cargar_yaml
        from inventario.ingesta import exigir_ok, ingestar

        corte = corte_de(contexto)
        rep = ingestar(corte, DATA, cargar_yaml("fuentes.yaml"))
        print(rep.resumen())
        for archivo, avisos in rep.advertencias.items():
            print(f"AVISO (esperado) {archivo}: {'; '.join(avisos)}")
        # Falla solo por archivos inválidos que llegaron en este último día
        exigir_ok(rep, desde=pd.Timestamp(corte) - pd.Timedelta(days=1))
        # Se anuncia el corte para que los DAGs siguientes trabajen con la misma fecha
        outlet_events[BANDEJA_PROCESADA].extra = {"corte": corte}
        return {"corte": corte, "procesados": len(rep.procesados), "en_cuarentena": len(rep.cuarentena),
                "fechas_faltantes": {k: len(v) for k, v in rep.faltantes.items()}}

    ingestar_y_validar()


ingesta_validacion()
