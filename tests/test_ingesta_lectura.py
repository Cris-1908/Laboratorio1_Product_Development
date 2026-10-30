import hashlib

import pandas as pd
import pytest

from inventario.bandeja import construir_bandeja
from inventario.ingesta import IngestaInvalida, exigir_ok, ingestar
from inventario.lectura import cargar_manifest, corte_ts, leer_asof

CORTE_FINAL = "2017-03-11"  # esperadas: 2017-01-01 .. 2017-03-10 (ventas/transacciones D+1)
MALAS = {"duplicados", "negativos", "fecha_incorrecta", "columna_faltante"}


def _hash_dir(d):
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(d).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@pytest.fixture
def mundo(raw_sintetico, cfg_bandeja, tmp_path):
    data = tmp_path / "data"
    construir_bandeja(raw_sintetico, data, cfg_bandeja, con_anomalias=True)
    verdad = pd.read_csv(data / "anomalias_verdad.csv")
    return data, cfg_bandeja, verdad


def _archivo(fecha):
    return f"ventas/{fecha}_v1.csv"


def test_cuarentena_exacta_contra_verdad(mundo):
    data, cfg, verdad = mundo
    rep = ingestar(CORTE_FINAL, data, cfg)
    esperadas = {_archivo(f) for f in verdad[verdad.tipo.isin(MALAS)].fecha_negocio}
    assert set(rep.cuarentena) == esperadas
    assert not rep.ok
    for a in esperadas:  # copia + motivo en cuarentena, y NO en procesado
        assert (data / "cuarentena" / a).exists()
        assert (data / "cuarentena" / f"{a}.motivo.json").exists()
        assert not (data / "procesado" / a).exists()


def test_faltantes_se_reportan_sin_fallar_y_validos_se_procesan(mundo):
    data, cfg, verdad = mundo
    rep = ingestar(CORTE_FINAL, data, cfg)
    assert set(rep.faltantes["ventas"]) == set(verdad[verdad.tipo == "faltante"].fecha_negocio)
    assert rep.faltantes["transacciones"] == []
    tardias = verdad[verdad.tipo == "llegada_tardia"].fecha_negocio
    assert all((data / "procesado" / _archivo(f)).exists() for f in tardias)


def test_exigir_ok_falla_con_cuarentena(mundo):
    data, cfg, _ = mundo
    with pytest.raises(IngestaInvalida):
        exigir_ok(ingestar(CORTE_FINAL, data, cfg))


def test_sin_anomalias_todo_ok(raw_sintetico, cfg_bandeja, tmp_path):
    construir_bandeja(raw_sintetico, tmp_path, cfg_bandeja, con_anomalias=False)
    rep = ingestar(CORTE_FINAL, tmp_path, cfg_bandeja)
    exigir_ok(rep)
    assert rep.faltantes == {"ventas": [], "transacciones": []}
    assert any("petróleo" in w for ws in rep.advertencias.values() for w in ws)  # esperado, no falla


def test_idempotente_y_la_bandeja_no_cambia(mundo):
    data, cfg, _ = mundo
    antes = _hash_dir(data / "bandeja")
    ingestar(CORTE_FINAL, data, cfg)
    h1 = _hash_dir(data / "procesado"), _hash_dir(data / "cuarentena")
    ingestar(CORTE_FINAL, data, cfg)
    assert h1 == (_hash_dir(data / "procesado"), _hash_dir(data / "cuarentena"))
    assert antes == _hash_dir(data / "bandeja")


def test_incremental_igual_a_completa(raw_sintetico, cfg_bandeja, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        construir_bandeja(raw_sintetico, d, cfg_bandeja, con_anomalias=True)
    ingestar(CORTE_FINAL, a, cfg_bandeja)
    ingestar("2017-02-01", b, cfg_bandeja)
    ingestar(CORTE_FINAL, b, cfg_bandeja, desde="2017-02-01")
    assert _hash_dir(a / "procesado") == _hash_dir(b / "procesado")
    assert _hash_dir(a / "cuarentena") == _hash_dir(b / "cuarentena")


def test_asof_correccion_v1_antes_v2_despues(mundo):
    data, cfg, _ = mundo
    ingestar(CORTE_FINAL, data, cfg)
    m = cargar_manifest(data / "bandeja")
    v2 = m[(m.fuente == "ventas") & (m.version == 2)].iloc[0]
    f = v2.fecha_negocio
    antes = leer_asof("ventas", v2.llegada_ts - pd.Timedelta(hours=1), data)
    despues = leer_asof("ventas", v2.llegada_ts, data)
    va, vd = antes[antes.date == f], despues[despues.date == f]
    assert len(va) and (va.version == 1).all()
    assert len(vd) and (vd.version == 2).all()
    assert not va.sales.reset_index(drop=True).equals(vd.sales.reset_index(drop=True))


def test_asof_sin_fuga_de_futuro(mundo):
    data, cfg, _ = mundo
    ingestar(CORTE_FINAL, data, cfg)
    for c in ("2017-01-15", "2017-02-10", "2017-03-05"):
        df = leer_asof("ventas", c, data)
        assert df.available_at.max() <= corte_ts(c)
        nuevos = df[df.version > 0]
        # D llega a D+1 06:00: con corte fin del día C, la última fecha posible es C-1
        assert nuevos.date.max() <= pd.Timestamp(c) - pd.Timedelta(days=1)


def test_asof_llegada_tardia_ausente_antes_presente_despues(mundo):
    data, cfg, verdad = mundo
    ingestar(CORTE_FINAL, data, cfg)
    d = pd.Timestamp(verdad[verdad.tipo == "llegada_tardia"].fecha_negocio.iloc[0])
    pronto = leer_asof("ventas", d + pd.Timedelta(days=1), data)   # llegaría D+5
    tarde = leer_asof("ventas", d + pd.Timedelta(days=5), data)
    assert (pronto.date == d).sum() == 0
    assert (tarde.date == d).sum() > 0


def test_asof_nunca_lee_cuarentena(mundo):
    data, cfg, verdad = mundo
    ingestar(CORTE_FINAL, data, cfg)
    df = leer_asof("ventas", CORTE_FINAL, data)
    for f in verdad[verdad.tipo.isin(MALAS)].fecha_negocio:
        assert (df.date == pd.Timestamp(f)).sum() == 0
    assert not df.duplicated(["date", "store_nbr", "family"]).any()
    assert (df.sales >= 0).all()


def test_corte_fecha_pura_es_fin_de_dia():
    assert corte_ts("2017-01-31") == pd.Timestamp("2017-01-31 23:59:59")
    assert corte_ts("2017-01-31 06:00") == pd.Timestamp("2017-01-31 06:00")
