import numpy as np
import pandas as pd

from inventario.validacion import validar_df

F = "2017-01-05"


def _ventas(n=4):
    return pd.DataFrame({
        "id": range(n), "date": [F] * n, "store_nbr": range(n),
        "family": ["GROCERY I"] * n, "sales": [10.0] * n, "onpromotion": [0] * n,
    })


def test_ventas_validas():
    r = validar_df(_ventas(), "ventas", F)
    assert r.ok and not r.advertencias


def test_duplicados_negativos_fecha_columna_vacio_texto():
    d = _ventas(); d.loc[1, ["store_nbr"]] = 0
    assert any("duplicadas" in e for e in validar_df(d, "ventas", F).errores)

    d = _ventas(); d.loc[0, "sales"] = -3
    assert any("negativos" in e for e in validar_df(d, "ventas", F).errores)

    assert any("no coincide" in e for e in validar_df(_ventas(), "ventas", "2017-01-06").errores)
    assert any("columnas faltantes" in e for e in validar_df(_ventas().drop(columns="onpromotion"), "ventas", F).errores)
    assert any("vacío" in e for e in validar_df(_ventas().iloc[0:0], "ventas", F).errores)

    d = _ventas().astype({"sales": object}); d.loc[0, "sales"] = "abc"
    assert any("no numéricos" in e for e in validar_df(d, "ventas", F).errores)


def test_ceros_y_petroleo_faltante_son_esperados_no_errores():
    d = _ventas(); d["sales"] = 0.0
    r = validar_df(d, "ventas", F)
    assert r.ok and r.advertencias

    o = pd.DataFrame({"date": [F], "dcoilwtico": [np.nan]})
    r = validar_df(o, "petroleo", F)
    assert r.ok and r.advertencias


def test_transacciones_y_feriados():
    t = pd.DataFrame({"date": [F, F], "store_nbr": [1, 2], "transactions": [100, 200]})
    assert validar_df(t, "transacciones", F).ok
    t.loc[0, "transactions"] = -1
    assert not validar_df(t, "transacciones", F).ok

    h = pd.DataFrame({"date": [F], "type": ["Holiday"], "locale": ["National"],
                      "locale_name": ["Ecuador"], "description": ["X"], "transferred": [False]})
    assert validar_df(h, "feriados", F).ok
