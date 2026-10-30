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
