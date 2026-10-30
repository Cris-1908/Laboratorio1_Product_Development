# Motor de Reposición de Inventario (Lab 1, Maestría en Data Science)

Producto de ML reproducible con Git, DVC y Airflow 3. Estado: en construcción.

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

_(pendiente: DVC, Airflow, reproducción de un pronóstico pasado, evidencia)_
