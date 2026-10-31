"""DAG 2: preparación de datos para el modelo. Se dispara solo cuando la ingesta termina bien (asset).

Materializa en data/features/corte=<fecha>/ las features de entrenamiento y de pronóstico, calculadas
solo con información disponible al corte. Idempotente: si los datos válidos y la configuración no
cambiaron, no recalcula; si llegó una corrección, la firma cambia y se recalcula.
"""
from __future__ import annotations

import pendulum
from airflow.sdk import dag, task

from comun import ARGS_DEFAULT, BANDEJA_PROCESADA, DATA, corte_de


@dag(
    dag_id="preparacion_datos",
    schedule=[BANDEJA_PROCESADA],
    start_date=pendulum.datetime(2016, 12, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=ARGS_DEFAULT,
    tags=["lab1", "preparacion"],
)
def preparacion_datos():
    @task
    def preparar_features(**contexto):
        from inventario.config import cargar_yaml
        from inventario.modelos import preparar_features as preparar

        corte = corte_de(contexto)
        carpeta = preparar(corte, DATA, cargar_yaml("modelo.yaml"))
        print(f"Features del corte {corte} en {carpeta}")
        return {"corte": corte, "carpeta": str(carpeta)}

    preparar_features()


preparacion_datos()
