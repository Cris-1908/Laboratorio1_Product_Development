import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from inventario.bandeja import construir_bandeja
from inventario.config import cargar_yaml
from inventario.ingesta import IngestaInvalida, exigir_ok, ingestar
from inventario.lectura import cargar_manifest, leer_asof, ultimo_corte_procesado
from inventario.modelos import (
    cargar_features, cargar_modelo, carpeta_features, construir_panel, entrenar_y_guardar, escribir_vigente,
    generar_pronostico, leer_vigente, preparar_features, pronostico_lgbm, publicar_pronostico,
)
from inventario.orquestacion import corte_desde_contexto
from inventario.seguimiento import evaluar_publicado, publicar_seguimiento, seguimiento_desempeno, verificar_reproduccion

CORTE = "2017-03-11"


@pytest.fixture
def mundo(raw_largo, cfg_bandeja, tmp_path):
    data = tmp_path / "data"
    construir_bandeja(raw_largo, data, cfg_bandeja, con_anomalias=False)
    exigir_ok(ingestar(CORTE, data, cfg_bandeja))
    return data


def _hash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------- features materializadas
def test_features_idempotentes_y_se_regeneran_si_cambia_la_config(mundo, cfg_modelo_test):
    c = preparar_features("2017-03-05", mundo, cfg_modelo_test)
    h1 = (_hash(c / "entrenamiento.csv.gz"), _hash(c / "pronostico.csv.gz"), _hash(c / "meta.json"))
    preparar_features("2017-03-05", mundo, cfg_modelo_test)
    assert h1 == (_hash(c / "entrenamiento.csv.gz"), _hash(c / "pronostico.csv.gz"), _hash(c / "meta.json"))
    cfg2 = json.loads(json.dumps(cfg_modelo_test))
    cfg2["entrenamiento"]["dias_historia"] = 70
    preparar_features("2017-03-05", mundo, cfg2)
    assert _hash(c / "meta.json") != h1[2]  # firma distinta => recalculó


def test_lgbm_desde_features_igual_al_calculo_directo(mundo, cfg_modelo_test):
    qs = cfg_modelo_test["cuantiles"]
    ruta = entrenar_y_guardar("2017-03-05", mundo, cfg_modelo_test)
    mods, meta = cargar_modelo(ruta)
    _, Xp, fmeta = cargar_features(carpeta_features("2017-03-05", mundo))
    from inventario.modelos import predecir_lgbm
    desde_csv = predecir_lgbm(mods, Xp, qs).sort_values(["store_nbr", "family", "fecha"]).reset_index(drop=True)
    panel = construir_panel("2017-03-05", mundo, dias=300)
    directo = pronostico_lgbm(panel, mods, 7, qs, meta["fam_cats"]).sort_values(["store_nbr", "family", "fecha"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(desde_csv, directo, check_exact=False, rtol=1e-9)


def test_modelo_se_reentrena_si_cambian_los_parametros(mundo, cfg_modelo_test):
    r = entrenar_y_guardar("2017-03-05", mundo, cfg_modelo_test)
    m1 = _hash(r / "meta.json")
    assert entrenar_y_guardar("2017-03-05", mundo, cfg_modelo_test) == r and _hash(r / "meta.json") == m1
    cfg2 = json.loads(json.dumps(cfg_modelo_test))
    cfg2["lightgbm"]["n_rounds"] = 20
    entrenar_y_guardar("2017-03-05", mundo, cfg2)
    assert _hash(r / "meta.json") != m1


# --------------------------------------------------------------- vigente y publicación
def test_vigente_por_defecto_baseline_y_se_puede_cambiar(tmp_path):
    ruta = tmp_path / "vigente.json"
    assert leer_vigente(ruta) == "baseline"
    escribir_vigente("lgbm", {"reemplaza": True}, ruta)
    assert leer_vigente(ruta) == "lgbm"


def test_vigente_decide_el_modelo_del_pronostico(mundo, cfg_modelo_test, tmp_path):
    ruta, cfg_c = tmp_path / "v.json", cargar_yaml("costos.yaml")
    assert generar_pronostico("2017-03-01", mundo, cfg_modelo_test, cfg_c, "vigente", ruta).modelo.iloc[0] == "baseline"
    escribir_vigente("lgbm", None, ruta)
    assert generar_pronostico("2017-03-01", mundo, cfg_modelo_test, cfg_c, "vigente", ruta).modelo.iloc[0].startswith("lgbm_")


def test_metadata_incluye_calidad_extra(mundo, cfg_modelo_test):
    cfg_c = cargar_yaml("costos.yaml")
    df = generar_pronostico("2017-03-01", mundo, cfg_modelo_test, cfg_c, modelo="baseline")
    c = publicar_pronostico(df, "2017-03-01", mundo, cfg_modelo_test, extra={"archivos_en_cuarentena": 2})
    assert json.loads((c / "metadata.json").read_text())["archivos_en_cuarentena"] == 2


# --------------------------------------------------------------- seguimiento
def _publicar(mundo, cfg_m, corte, modelo="baseline"):
    df = generar_pronostico(corte, mundo, cfg_m, cargar_yaml("costos.yaml"), modelo=modelo)
    publicar_pronostico(df, corte, mundo, cfg_m)
    return df


def test_seguimiento_completo_parcial_y_sin_datos(mundo, cfg_modelo_test):
    _publicar(mundo, cfg_modelo_test, "2017-03-01")  # pronostica 03-01 .. 03-07 (origen 02-28)
    completo = seguimiento_desempeno("2017-03-11", mundo).iloc[0]
    assert completo.n_evaluadas == completo.n_total and completo.cobertura_datos == 1.0
    assert np.isfinite([completo.pinball, completo.wape, completo.costo]).all()
    assert 0 <= completo.cobertura_intervalo <= 1

    parcial = seguimiento_desempeno("2017-03-04", mundo).iloc[0]  # origen 03-03: llegaron 03-01..03-03
    assert 0 < parcial.n_evaluadas < parcial.n_total

    aun_no = seguimiento_desempeno("2017-03-01", mundo).iloc[0]   # origen 02-28: nada del pronóstico llegó
    assert aun_no.n_evaluadas == 0 and np.isnan(aun_no.pinball)


def test_seguimiento_ignora_pronosticos_posteriores_al_corte(mundo, cfg_modelo_test):
    _publicar(mundo, cfg_modelo_test, "2017-03-01")
    _publicar(mundo, cfg_modelo_test, "2017-03-08")
    assert list(seguimiento_desempeno("2017-03-05", mundo).corte_pronostico) == ["2017-03-01"]
    assert len(seguimiento_desempeno("2017-03-11", mundo)) == 2


def test_seguimiento_sin_pronosticos_devuelve_tabla_vacia(mundo):
    assert seguimiento_desempeno("2017-03-11", mundo).empty


def test_seguimiento_usa_el_pedido_publicado_no_el_recalculado(mundo, cfg_modelo_test):
    df = _publicar(mundo, cfg_modelo_test, "2017-03-01")
    real = construir_panel("2017-03-11", mundo, dias=40)
    base = evaluar_publicado(df, real)["costo"]
    df2 = df.copy()
    df2["pedido"] = df2["pedido"] * 2          # si alguien "cambia" lo publicado, el costo cambia
    assert evaluar_publicado(df2, real)["costo"] != base


def test_seguimiento_se_publica_idempotente(mundo, cfg_modelo_test):
    _publicar(mundo, cfg_modelo_test, "2017-03-01")
    r1 = publicar_seguimiento(seguimiento_desempeno("2017-03-11", mundo), "2017-03-11", mundo)
    h = _hash(r1)
    publicar_seguimiento(seguimiento_desempeno("2017-03-11", mundo), "2017-03-11", mundo)
    assert h == _hash(r1)


# --------------------------------------------------------------- reproducibilidad
@pytest.mark.parametrize("modelo", ["baseline", "lgbm"])
def test_un_pronostico_pasado_se_reproduce_exacto(mundo, cfg_modelo_test, modelo):
    cfg_c = cargar_yaml("costos.yaml")
    _publicar(mundo, cfg_modelo_test, "2017-03-01", modelo)
    r = verificar_reproduccion("2017-03-01", mundo, cfg_modelo_test, cfg_c)
    assert r["igual"] and r["sha_publicado"] == r["sha_regenerado"]


def test_la_reproduccion_no_cambia_aunque_despues_llegue_una_correccion(raw_largo, cfg_bandeja, cfg_modelo_test, tmp_path):
    data = tmp_path / "d"
    construir_bandeja(raw_largo, data, cfg_bandeja)
    m = cargar_manifest(data / "bandeja")
    v2 = m[(m.fuente == "ventas") & (m.version == 2)].sort_values("llegada_ts").iloc[0]
    corte_pub = (v2.llegada_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # antes de que llegue la v2
    ingestar(corte_pub, data, cfg_bandeja)
    cfg_c = cargar_yaml("costos.yaml")
    df = generar_pronostico(corte_pub, data, cfg_modelo_test, cfg_c, modelo="baseline")
    publicar_pronostico(df, corte_pub, data, cfg_modelo_test)
    ingestar(CORTE, data, cfg_bandeja)  # ahora ya llegaron las correcciones
    assert verificar_reproduccion(corte_pub, data, cfg_modelo_test, cfg_c)["igual"]


# --------------------------------------------------------------- ingesta: ventana y bordes
def test_exigir_ok_solo_falla_por_archivos_inválidos_nuevos(raw_largo, cfg_bandeja, tmp_path):
    data = tmp_path / "d"
    construir_bandeja(raw_largo, data, cfg_bandeja, con_anomalias=True)
    verdad = pd.read_csv(data / "anomalias_verdad.csv", parse_dates=["fecha_negocio"])
    mala = verdad[verdad.tipo == "duplicados"].fecha_negocio.min()
    llegada = mala + pd.Timedelta(days=1, hours=6)
    rep = ingestar(llegada + pd.Timedelta(days=5), data, cfg_bandeja)
    with pytest.raises(IngestaInvalida):
        exigir_ok(rep, desde=llegada.normalize() - pd.Timedelta(days=1))   # llegó en la ventana: falla
    # una corrida posterior no debe fallar por ese archivo ya reportado (solo mira lo que llegó el último día)
    rep2 = ingestar(llegada + pd.Timedelta(days=5), data, cfg_bandeja)
    exigir_ok(rep2, desde=llegada.normalize() + pd.Timedelta(days=4))


def test_asof_antes_de_la_primera_llegada_devuelve_solo_historico(mundo):
    df = leer_asof("ventas", "2016-12-15", mundo)
    assert (df.version == 0).all() and df.date.max() <= pd.Timestamp("2016-12-31")


def test_ultimo_corte_procesado(mundo):
    assert ultimo_corte_procesado(mundo) == "2017-03-11"


# --------------------------------------------------------------- orquestación (sin Airflow)
def _ev(corte):
    return SimpleNamespace(extra={"corte": corte})


def test_corte_prefiere_el_evento_mas_reciente_del_asset():
    assert corte_desde_contexto({"a": [_ev("2017-02-01"), _ev("2017-02-03")]}, logical_date="2020-01-01") == "2017-02-03"


def test_corte_usa_logical_date_luego_run_after_luego_ultimo_procesado(mundo):
    assert corte_desde_contexto(None, logical_date=pd.Timestamp("2017-02-05T00:00:00Z")) == "2017-02-05"
    assert corte_desde_contexto({}, run_after=pd.Timestamp("2017-02-06")) == "2017-02-06"
    assert corte_desde_contexto({"a": [SimpleNamespace(extra=None)]}, data_dir=mundo) == "2017-03-11"


def test_corte_sin_ninguna_fuente_falla(tmp_path):
    with pytest.raises(ValueError):
        corte_desde_contexto(None)
