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

```bash
# 4. Validar e ingerir lo que haya llegado hasta un corte (código de salida 1 si hay cuarentena)
python scripts/ingestar.py --corte 2017-06-30

# 5. Backtest baseline vs LightGBM y decisión de reemplazo (escribe reports/)
python scripts/backtest.py --corte 2017-06-30 --promover

# 6. Pronóstico y pedido sugerido para un corte
python scripts/pronosticar.py --corte 2017-06-30 --modelo vigente
```

_(pendiente: DVC, Airflow, seguimiento del desempeño, reproducción de un pronóstico pasado, evidencia)_
