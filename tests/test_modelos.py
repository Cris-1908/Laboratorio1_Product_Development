import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from inventario.bandeja import construir_bandeja
from inventario.config import cargar_yaml
from inventario.ingesta import exigir_ok, ingestar
from inventario.modelos import (
    FEATURES, backtest, col, construir_panel, corte_de_origen, costo_negocio, dataset_entrenamiento,
    decidir_reemplazo, entrenar_lgbm, features_origen, fecha_origen, generar_pronostico, pedido_sugerido,
    pinball, pronostico_baseline, pronostico_lgbm, publicar_pronostico, sesgo, wape,
)

CORTE = "2017-03-11"
QS = [0.1, 0.25, 0.5, 0.75, 0.9]


@pytest.fixture
def mundo(raw_largo, cfg_bandeja, tmp_path):
    data = tmp_path / "data"
    construir_bandeja(raw_largo, data, cfg_bandeja, con_anomalias=False)
    exigir_ok(ingestar(CORTE, data, cfg_bandeja))
    return data


# ------------------------------------------------------------------ fechas y panel
def test_origen_y_corte_son_inversos():
    o = pd.Timestamp("2017-02-10")
    assert fecha_origen(corte_de_origen(o)) == o
    assert fecha_origen("2017-03-11") == pd.Timestamp("2017-03-10")


def test_panel_solo_incluye_hasta_el_origen(mundo):
    p = construir_panel("2017-02-01", mundo)
    assert p.fechas[-1] == pd.Timestamp("2017-01-31")
    assert pd.Timestamp("2017-02-01") not in p.fechas
    assert pd.Timestamp("2016-12-24") in p.feriados  # feriado nacional conocido


# ------------------------------------------------------------------ sin fuga
def test_features_no_cambian_si_cambia_el_futuro(mundo):
    p = construir_panel(CORTE, mundo)
    i = p.i_origen - 20
    fams = sorted(p.series.family.unique())
    base = features_origen(p, i, 7, fams)
    p.Y[i + 1:] = 99999.0  # envenenar todo lo posterior al origen
    p.P[i + 1:] = 99999.0
    despues = features_origen(p, i, 7, fams)
    pd.testing.assert_frame_equal(base[FEATURES], despues[FEATURES])


def test_lags_siempre_en_o_antes_del_origen(mundo):
    p = construir_panel(CORTE, mundo)
    i = p.i_origen - 20
    fams = sorted(p.series.family.unique())
    p.Y[:] = np.arange(len(p.fechas))[:, None]  # el valor de cada día es su índice
    f = features_origen(p, i, 14, fams)
    # dow_last = log1p(valor del último mismo-día-de-semana <= origen) debe ser <= log1p(i)
    assert (f.dow_last <= np.log1p(i) + 1e-9).all()
    assert (f.dow4 <= np.log1p(i) + 1e-9).all()


def test_asof_distinto_por_corte(mundo):
    a = construir_panel("2017-02-01", mundo)
    b = construir_panel("2017-03-01", mundo)
    assert len(b.fechas) > len(a.fechas)


# ------------------------------------------------------------------ modelos
def test_baseline_cuantiles_ordenados(mundo, cfg_modelo_test):
    p = construir_panel(CORTE, mundo)
    fc = pronostico_baseline(p, 7, QS, 6)
    assert len(fc) == 7 * 12
    v = fc[[col(q) for q in QS]].to_numpy()
    assert (np.diff(v, axis=1) >= 0).all()
    assert fc.fecha.min() == pd.Timestamp("2017-03-11")


def test_lgbm_entrena_y_pronostica_sin_cruces_ni_negativos(mundo, cfg_modelo_test):
    p = construir_panel(CORTE, mundo)
    fams = sorted(p.series.family.unique())
    X = dataset_entrenamiento(p, 7, cfg_modelo_test["entrenamiento"], fams)
    assert len(X) > 500 and X.y_real.notna().all()
    mods = entrenar_lgbm(X, QS, cfg_modelo_test["lightgbm"])
    fc = pronostico_lgbm(p, mods, 7, QS, fams)
    v = fc[[col(q) for q in QS]].to_numpy()
    assert len(fc) == 7 * 12 and (v >= 0).all() and (np.diff(v, axis=1) >= 0).all()


def test_entrenamiento_nunca_usa_objetivos_posteriores_al_origen_del_panel(mundo, cfg_modelo_test):
    p = construir_panel("2017-02-15", mundo)
    fams = sorted(p.series.family.unique())
    X = dataset_entrenamiento(p, 7, cfg_modelo_test["entrenamiento"], fams)
    assert X.fecha.max() <= p.fechas[-1]


# ------------------------------------------------------------------ métricas y pedido
def test_metricas_basicas():
    y = np.array([10.0, 20.0])
    Q = np.array([[10.0, 10.0], [20.0, 20.0]])
    assert pinball(y, Q, [0.5, 0.5]) == 0.0
    assert wape(y, np.array([12.0, 18.0])) == pytest.approx(4 / 30)
    assert sesgo(y, np.array([12.0, 22.0])) == pytest.approx(4 / 30)
    # pinball q=0.9, y=10, pred=0 -> 0.9*10 = 9 ; pred=20 -> 0.1*10 = 1
    assert pinball(np.array([10.0]), np.array([[0.0]]), [0.9]) == pytest.approx(9.0)
    assert pinball(np.array([10.0]), np.array([[20.0]]), [0.9]) == pytest.approx(1.0)


def test_costo_negocio():
    assert costo_negocio(np.array([10.0]), np.array([8.0]), np.array([4.0]), np.array([1.0])) == 8.0
    assert costo_negocio(np.array([10.0]), np.array([13.0]), np.array([4.0]), np.array([1.0])) == 3.0


def _fc(familia):
    return pd.DataFrame({"fecha": [pd.Timestamp("2017-03-11")], "h": [1], "store_nbr": [1], "family": [familia],
                         "q10": [10.0], "q25": [15.0], "q50": [20.0], "q75": [25.0], "q90": [30.0]})


def test_newsvendor_interpola_el_cuantil_critico():
    cfg = {"default": {"costo_quiebre": 1.0, "costo_merma": 1.0},
           "por_familia": {"X": {"costo_quiebre": 7.0, "costo_merma": 3.0}}}  # 0.7 entre q50 y q75
    assert pedido_sugerido(_fc("X"), cfg, QS).pedido.iloc[0] == pytest.approx(24.0)  # 20 + (0.2/0.25)*5
    assert pedido_sugerido(_fc("Y"), cfg, QS).pedido.iloc[0] == pytest.approx(20.0)  # 0.5 -> q50


def test_cambiar_costos_cambia_pedido_sin_tocar_codigo():
    cfg = {"default": {"costo_quiebre": 1.0, "costo_merma": 1.0}}
    p1 = pedido_sugerido(_fc("X"), cfg, QS).pedido.iloc[0]
    cfg["default"]["costo_quiebre"] = 9.0
    assert pedido_sugerido(_fc("X"), cfg, QS).pedido.iloc[0] > p1


def test_perecedero_pide_menos_que_alta_rotacion():
    cfg = cargar_yaml("costos.yaml")
    assert pedido_sugerido(_fc("DAIRY"), cfg, QS).pedido.iloc[0] < pedido_sugerido(_fc("GROCERY I"), cfg, QS).pedido.iloc[0]


# ------------------------------------------------------------------ reemplazo
def _res(vig, ret, pin_v=1.0, pin_r=1.0):
    filas = []
    for f, (cv, cr) in enumerate(zip(vig, ret), start=1):
        filas += [dict(fold=f, modelo="baseline", costo=cv, pinball=pin_v),
                  dict(fold=f, modelo="lgbm", costo=cr, pinball=pin_r)]
    return pd.DataFrame(filas)


CFG_R = {"mejora_min_costo_pct": 2.0, "folds_min_ganados": 3, "max_empeora_pinball_pct": 0.0}


def test_reemplazo_gana():
    d = decidir_reemplazo(_res([100] * 4, [90, 90, 90, 99]), "baseline", "lgbm", CFG_R)
    assert d["reemplaza"] and d["folds_ganados"] == 3


def test_no_reemplaza_si_gana_pocos_folds():
    assert not decidir_reemplazo(_res([100] * 4, [90, 90, 101, 101]), "baseline", "lgbm", CFG_R)["reemplaza"]


def test_no_reemplaza_si_mejora_es_menor_al_umbral():
    assert not decidir_reemplazo(_res([100] * 4, [99] * 4), "baseline", "lgbm", CFG_R)["reemplaza"]


def test_no_reemplaza_si_empeora_pinball():
    assert not decidir_reemplazo(_res([100] * 4, [80] * 4, 1.0, 1.1), "baseline", "lgbm", CFG_R)["reemplaza"]


# ------------------------------------------------------------------ backtest y publicación
def test_backtest_mismos_folds_para_todos_los_modelos(mundo, cfg_modelo_test):
    res = backtest(CORTE, mundo, cfg_modelo_test, cargar_yaml("costos.yaml"))
    assert set(res.modelo) == {"baseline", "lgbm"}
    a = res[res.modelo == "baseline"][["fold", "origen"]].reset_index(drop=True)
    b = res[res.modelo == "lgbm"][["fold", "origen"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == cfg_modelo_test["validacion"]["n_folds"]
    assert (res.n > 0).all() and res[["pinball", "wape", "sesgo", "costo"]].notna().all().all()
    # el último origen debe dejar H días reales evaluables
    assert res.origen.max() == pd.Timestamp("2017-03-10") - pd.Timedelta(days=7)


def test_publicacion_idempotente_y_con_metadata(mundo, cfg_modelo_test):
    cfg_c = cargar_yaml("costos.yaml")
    df = generar_pronostico("2017-03-01", mundo, cfg_modelo_test, cfg_c, modelo="baseline")
    c = publicar_pronostico(df, "2017-03-01", mundo, cfg_modelo_test)
    h1 = hashlib.sha256((c / "pronostico.csv").read_bytes()).hexdigest()
    publicar_pronostico(generar_pronostico("2017-03-01", mundo, cfg_modelo_test, cfg_c, modelo="baseline"),
                        "2017-03-01", mundo, cfg_modelo_test)
    assert h1 == hashlib.sha256((c / "pronostico.csv").read_bytes()).hexdigest()
    meta = json.loads((c / "metadata.json").read_text())
    assert {"git_sha", "huella_datos", "modelo", "corte"} <= set(meta)
    assert df.fecha.min() == pd.Timestamp("2017-03-01")  # corte 03-01 => origen 02-28, primer día pronosticado 03-01


def test_lgbm_reproducible_mismo_corte_mismo_resultado(mundo, cfg_modelo_test):
    cfg_c = cargar_yaml("costos.yaml")
    a = generar_pronostico("2017-03-05", mundo, cfg_modelo_test, cfg_c, modelo="lgbm")
    b = generar_pronostico("2017-03-05", mundo, cfg_modelo_test, cfg_c, modelo="lgbm")
    pd.testing.assert_frame_equal(a, b)
    assert (mundo / "modelos" / "lgbm_2017-03-04" / "meta.json").exists()
