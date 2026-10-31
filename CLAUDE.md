# Reglas del proyecto: Motor de Reposición de Inventario (Lab 1)

Producto de ML reproducible (Git + DVC + Airflow 3) que genera pronóstico de ventas con cuantiles, pedido sugerido y seguimiento del desempeño. Datos: Corporación Favorita (Kaggle).

## Reglas innegociables

1. **La bandeja es solo lectura.** Nada en `data/bandeja/` se modifica ni se borra desde el flujo. Solo el script `scripts/generar_bandeja.py` la crea (simula el sistema de origen).
2. **Disponibilidad temporal.** Toda predicción usa únicamente información con `llegada_ts <= corte`. Para cada (fuente, fecha_negocio) se toma la última versión disponible a esa fecha. Nunca leer "lo último que hay".
3. **Lags >= horizonte.** Ninguna feature puede usar información posterior al corte. Lo único legítimamente futuro-conocido: calendario y feriados ya publicados.
4. **La lógica vive en `src/inventario/`**, con pruebas en `tests/`. Los DAGs en `dags/` solo orquestan (importan y llaman funciones).
5. **Idempotencia.** Re-ejecutar una tarea produce el mismo resultado: escribir por partición con reemplazo atómico (archivo temporal + rename) o upsert por llave `(fecha, tienda, familia)`.
6. **Datos inválidos detienen el flujo** (cuarentena + falla). Situaciones esperadas del negocio (25 dic sin ventas, tienda cerrada, ceros, petróleo faltante en fin de semana) se registran pero NO fallan.
7. **Costos y parámetros en `config/*.yaml`**, versionados. Cambiar un costo cambia el pedido sin tocar código.
8. **Ningún dato ni modelo en Git.** Todo en `data/` va a DVC. `data/.gitignore` lo mantiene DVC (nunca una regla `data/*` en la raíz).
9. **Semillas fijas** en todo lo aleatorio.
10. Trabajar por ramas y Pull Requests; `main` protegida; pytest debe pasar antes de mergear.

## Estructura

- `config/`: costos.yaml, modelo.yaml, fuentes.yaml
- `src/inventario/`: bandeja, lectura as-of, validación, features, modelos, pedido, seguimiento
- `dags/`: ingesta_validacion, preparacion_datos, pronostico_pedido, seguimiento_desempeno
- `tests/`: pytest
- `scripts/`: utilidades de línea de comandos

## Comandos

- Tests: `pytest -q`
- Generar bandeja: `python scripts/generar_bandeja.py --raw data/raw --salida data`
- Ingestar y validar hasta un corte: `python scripts/ingestar.py --corte 2017-02-01` (sale con código 1 si hay cuarentena)
- Capas: `data/historico`, `data/bandeja` (solo lectura), `data/procesado` (válido, todas las versiones), `data/cuarentena`
- Con anomalías simuladas (para pruebas): agregar `--anomalias`
- Backtest y decisión de reemplazo: `python scripts/backtest.py --corte 2017-08-01 [--promover]` (escribe `reports/`)
- Pronóstico + pedido: `python scripts/pronosticar.py --corte 2017-06-30 --modelo vigente|baseline|lgbm` (publica en `data/forecasts/corte=<fecha>/`)
- Features/entrenamiento: `python scripts/preparar.py --corte <f>` y `python scripts/entrenar.py --corte <f>`
- Seguimiento: `python scripts/seguimiento.py --corte <f>`; reproducción: `python scripts/verificar_pronostico.py --corte <f>`
- DVC: `dvc repro` / `dvc push` (cortes en `params.yaml`); backfill Airflow: `bash scripts/backfill.sh`
- Modelo vigente: `config/vigente.json` (en Git, para que el cambio sea revisable en PR)
- Regla de features: usar solo información con fecha <= origen del pronóstico; lags >= horizonte; nunca `onpromotion` futuro.

## Estilo

Python 3.11+, tipado básico, funciones puras donde se pueda, sin lógica de negocio dentro de DAGs. Documentar en `DECISIONES.md` cualquier decisión sobre disponibilidad temporal, validación y correcciones.
