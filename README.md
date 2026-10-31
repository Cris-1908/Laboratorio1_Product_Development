# Motor de Reposición de Inventario (Lab 1, Maestría en Data Science)

Producto de ML reproducible con Git, DVC y Airflow 3. Estado: pipeline completo (ingesta, modelos, DVC, DAGs de Airflow).

## Requisitos
Python 3.11+, Docker y Docker Compose, Git, DVC.

## Puesta en marcha (por ahora)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Descargar de Kaggle (store-sales-time-series-forecasting) y colocar en data/raw/:
#    train.csv, stores.csv, holidays_events.csv, oil.csv, transactions.csv

# 2. Generar carga histórica y bandeja
python scripts/generar_bandeja.py --raw data/raw --salida data

# 3. Pruebas
pytest -q
```

```bash
# 4. Validar e ingerir lo que haya llegado hasta un corte (código de salida 1 si hay cuarentena)
python scripts/ingestar.py --corte 2017-06-30

# 5. Backtest baseline vs LightGBM y decisión de reemplazo (escribe reports/)
python scripts/backtest.py --corte 2017-06-30 --promover

# 6. Pronóstico y pedido sugerido para un corte
python scripts/pronosticar.py --corte 2017-06-30 --modelo vigente
```

```bash
# 7. Seguimiento: compara pronósticos publicados contra lo real que ya llegó al corte
python scripts/seguimiento.py --corte 2017-08-14

# 8. Reproducir un pronóstico pasado con los datos tal como estaban (compara SHA del CSV publicado)
python scripts/verificar_pronostico.py --corte 2017-06-30
```

## DVC (datos, pipeline y reproducibilidad)

```bash
dvc init
dvc remote add -d local ~/dvc_remoto      # remoto local; puede ser S3/GCS/etc.
dvc add data/raw                          # datos crudos de Kaggle
dvc repro                                 # bandeja -> ingesta -> features -> entrenamiento -> pronostico -> seguimiento
dvc push
git add dvc.yaml dvc.lock params.yaml data/raw.dvc data/.gitignore config/vigente.json
```

- Los cortes se cambian en `params.yaml` (`corte`, `corte_seguimiento`). Las salidas llevan `persist: true`: un corte nuevo **no borra** los anteriores.
- `data/.gitignore` lo mantiene DVC. No pongas una regla `data/*` en el `.gitignore` raíz (rompe `dvc checkout`).
- Para recuperar todo en otra máquina: `git clone` + `dvc pull`.

## Airflow 3 (Docker)

```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/3.3.2/docker-compose.yaml'   # y se versiona
mkdir -p airflow/{logs,plugins,config,dags}
cp .env.example .env                       # ajustar AIRFLOW_UID con `id -u`
docker build -t lab1-airflow:latest .
docker compose up airflow-init
docker compose up -d                       # UI: http://localhost:8080  (airflow / airflow)
bash scripts/backfill.sh                   # despausa los DAGs y lanza backfill
```

DAGs: `ingesta_validacion` (diario) -> asset `bandeja_procesada` -> `preparacion_datos` y `seguimiento_desempeno`; `pronostico_pedido` (semanal). Toda la lógica está en `src/inventario/`.

