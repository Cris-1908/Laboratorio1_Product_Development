import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_sintetico(tmp_path):
    """Mini versión de los CSV de Kaggle (3 tiendas, 4 familias, ene-feb 2017)."""
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    fechas = pd.date_range("2016-12-20", "2017-03-10")
    filas, i = [], 0
    for f in fechas:
        for s in (1, 2, 3):
            for fam in ("GROCERY I", "DAIRY", "BEVERAGES", "PRODUCE"):
                filas.append((i, f, s, fam, float(rng.poisson(50)), int(rng.integers(0, 3))))
                i += 1
    pd.DataFrame(filas, columns=["id", "date", "store_nbr", "family", "sales", "onpromotion"]).to_csv(raw / "train.csv", index=False)

    pd.DataFrame(
        [(f, s, int(rng.integers(500, 900))) for f in fechas for s in (1, 2, 3)],
        columns=["date", "store_nbr", "transactions"],
    ).to_csv(raw / "transactions.csv", index=False)

    habiles = fechas[fechas.dayofweek < 5]
    oil = pd.DataFrame({"date": habiles, "dcoilwtico": 50 + rng.normal(0, 1, len(habiles))})
    oil.loc[[3, 20], "dcoilwtico"] = np.nan  # 3: histórico; 20: dentro de la bandeja
    oil.to_csv(raw / "oil.csv", index=False)

    pd.DataFrame(
        {"date": pd.to_datetime(["2016-12-24", "2017-01-01", "2017-02-27"]), "type": "Holiday",
         "locale": "National", "locale_name": "Ecuador", "description": ["Navidad", "Año Nuevo", "Carnaval"],
         "transferred": False}
    ).to_csv(raw / "holidays_events.csv", index=False)

    pd.DataFrame(
        {"store_nbr": [1, 2, 3], "city": "Quito", "state": "Pichincha", "type": "D", "cluster": 13}
    ).to_csv(raw / "stores.csv", index=False)
    return raw


@pytest.fixture
def cfg_bandeja():
    return {
        "bandeja": {
            "corte_historico": "2016-12-31", "seed": 7,
            "n_correcciones_ventas": 5, "n_correcciones_transacciones": 3,
            "dias_ultimos_sin_tocar": 10,
            "anomalias": {"llegada_tardia": 2, "faltante": 2, "duplicados": 2,
                          "negativos": 2, "fecha_incorrecta": 2, "columna_faltante": 2},
        },
        "fuentes": {
            "ventas": {"rezago_dias": 1}, "transacciones": {"rezago_dias": 1},
            "petroleo": {"rezago_dias": 2}, "feriados": {"anticipacion_dias": 30},
        },
    }


@pytest.fixture
def raw_largo(tmp_path):
    """Serie sintética más larga (jun-2016 a mar-2017, 3 tiendas x 4 familias) con patrón semanal."""
    raw = tmp_path / "raw_largo"
    raw.mkdir()
    rng = np.random.default_rng(1)
    fechas = pd.date_range("2016-06-01", "2017-03-10")
    base = {(s, f): rng.uniform(20, 120) for s in (1, 2, 3) for f in ("GROCERY I", "DAIRY", "BEVERAGES", "PRODUCE")}
    semana = np.array([0.8, 0.85, 0.9, 0.95, 1.1, 1.4, 1.2])
    filas, i = [], 0
    for f in fechas:
        for (s, fam), b in base.items():
            lam = b * semana[f.dayofweek] * (1 + 0.3 * (f.day == 15))
            filas.append((i, f, s, fam, float(rng.poisson(lam)), int(rng.integers(0, 3))))
            i += 1
    pd.DataFrame(filas, columns=["id", "date", "store_nbr", "family", "sales", "onpromotion"]).to_csv(raw / "train.csv", index=False)
    pd.DataFrame([(f, s, 500) for f in fechas for s in (1, 2, 3)],
                 columns=["date", "store_nbr", "transactions"]).to_csv(raw / "transactions.csv", index=False)
    habiles = fechas[fechas.dayofweek < 5]
    pd.DataFrame({"date": habiles, "dcoilwtico": 50.0}).to_csv(raw / "oil.csv", index=False)
    pd.DataFrame({"date": pd.to_datetime(["2016-12-24", "2017-01-01", "2017-02-27"]), "type": "Holiday",
                  "locale": "National", "locale_name": "Ecuador", "description": ["Navidad", "Año Nuevo", "Carnaval"],
                  "transferred": False}).to_csv(raw / "holidays_events.csv", index=False)
    pd.DataFrame({"store_nbr": [1, 2, 3], "city": "Quito", "state": "Pichincha", "type": "D", "cluster": 13}
                 ).to_csv(raw / "stores.csv", index=False)
    return raw


@pytest.fixture
def cfg_modelo_test():
    return {
        "horizonte_dias": 7, "cuantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
        "baseline": {"n_semanas": 6},
        "entrenamiento": {"dias_historia": 140, "paso_origen_dias": 14, "dias_extra_features": 100},
        "lightgbm": {"n_rounds": 40, "learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 5,
                     "feature_fraction": 0.9, "bagging_fraction": 0.9, "seed": 42, "num_threads": 1},
        "validacion": {"esquema": "rolling_origin", "n_folds": 2, "paso_dias": 14},
        "reemplazo": {"mejora_min_costo_pct": 2.0, "folds_min_ganados": 2, "max_empeora_pinball_pct": 0.0},
    }
