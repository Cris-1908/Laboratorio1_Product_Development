"""DAG 3: pronóstico semanal y pedido sugerido.

El logical_date de cada corrida es el corte: solo se usa información disponible a esa fecha.
- asegurar_datos: ingesta idempotente hasta el corte. Si hay archivos inválidos NO se usan (quedan en
  cuarentena y fuera del lector as-of); la calidad queda registrada en la metadata del pronóstico.
- preparar_y_entrenar: features del corte y, si el modelo vigente es LightGBM, su entrenamiento.
- pronosticar_y_publicar: pronóstico + pedido (según config/costos.yaml) publicados de forma idempotente.
Backfill: airflow backfill create --dag-id pronostico_pedido --from-date ... --to-date ...
"""
from __future__ import annotations

import pendulum
from airflow.sdk import dag, task

from comun import ARGS_DEFAULT, DATA, corte_de


@dag(
    dag_id="pronostico_pedido",
    schedule="@weekly",
    start_date=pendulum.datetime(2017, 1, 8, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=ARGS_DEFAULT,
    tags=["lab1", "pronostico"],
)
def pronostico_pedido():
    @task
    def asegurar_datos(**contexto):
        from inventario.config import cargar_yaml
        from inventario.ingesta import ingestar

        corte = corte_de(contexto)
        rep = ingestar(corte, DATA, cargar_yaml("fuentes.yaml"))
        calidad = {"archivos_en_cuarentena": len(rep.cuarentena),
                   "fechas_sin_archivo": {k: v for k, v in rep.faltantes.items() if v}}
        print(rep.resumen())
        return {"corte": corte, "calidad": calidad}

    @task
    def preparar_y_entrenar(datos: dict):
        from inventario.config import cargar_yaml
        from inventario.modelos import entrenar_y_guardar, leer_vigente, preparar_features

        cfg = cargar_yaml("modelo.yaml")
        preparar_features(datos["corte"], DATA, cfg)
        if leer_vigente() != "baseline":
            entrenar_y_guardar(datos["corte"], DATA, cfg)
        return datos

    @task
    def pronosticar_y_publicar(datos: dict):
        from inventario.config import cargar_yaml
        from inventario.modelos import generar_pronostico, publicar_pronostico

        cfg_m, cfg_c = cargar_yaml("modelo.yaml"), cargar_yaml("costos.yaml")
        df = generar_pronostico(datos["corte"], DATA, cfg_m, cfg_c, modelo="vigente")
        carpeta = publicar_pronostico(df, datos["corte"], DATA, cfg_m, extra=datos["calidad"])
        print(f"{len(df)} filas ({df.modelo.iloc[0]}) publicadas en {carpeta}")
        return str(carpeta)

    pronosticar_y_publicar(preparar_y_entrenar(asegurar_datos()))


pronostico_pedido()
