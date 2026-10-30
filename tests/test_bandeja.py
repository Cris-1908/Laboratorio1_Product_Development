import hashlib

import pandas as pd

from inventario.bandeja import construir_bandeja


def _hash_dir(d):
    h = hashlib.sha256()
    for p in sorted(d.rglob("*.csv")):
        h.update(p.relative_to(d).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def test_historico_y_bandeja_no_se_traslapan(raw_sintetico, cfg_bandeja, tmp_path):
    out = tmp_path / "out"
    m = construir_bandeja(raw_sintetico, out, cfg_bandeja)
    hist = pd.read_csv(out / "historico" / "train.csv", parse_dates=["date"])
    assert hist.date.max() <= pd.Timestamp("2016-12-31")
    ventas = m[m.fuente == "ventas"]
    assert pd.to_datetime(ventas.fecha_negocio).min() > pd.Timestamp("2016-12-31")


def test_reproducible_con_misma_semilla(raw_sintetico, cfg_bandeja, tmp_path):
    construir_bandeja(raw_sintetico, tmp_path / "a", cfg_bandeja, con_anomalias=True)
    construir_bandeja(raw_sintetico, tmp_path / "b", cfg_bandeja, con_anomalias=True)
    assert _hash_dir(tmp_path / "a" / "bandeja") == _hash_dir(tmp_path / "b" / "bandeja")


def test_idempotente_regenerar_sobre_mismo_directorio(raw_sintetico, cfg_bandeja, tmp_path):
    out = tmp_path / "out"
    construir_bandeja(raw_sintetico, out, cfg_bandeja)
    h1 = _hash_dir(out / "bandeja")
    construir_bandeja(raw_sintetico, out, cfg_bandeja)
    assert h1 == _hash_dir(out / "bandeja")


def test_correcciones_llegan_despues_y_v1_difiere(raw_sintetico, cfg_bandeja, tmp_path):
    out = tmp_path / "out"
    m = construir_bandeja(raw_sintetico, out, cfg_bandeja)
    v2 = m[(m.fuente == "ventas") & (m.version == 2)]
    assert len(v2) == cfg_bandeja["bandeja"]["n_correcciones_ventas"]
    for _, r in v2.iterrows():
        v1 = m[(m.fuente == "ventas") & (m.fecha_negocio == r.fecha_negocio) & (m.version == 1)].iloc[0]
        assert pd.Timestamp(r.llegada_ts) > pd.Timestamp(v1.llegada_ts)
        a = pd.read_csv(out / "bandeja" / v1.archivo)
        b = pd.read_csv(out / "bandeja" / r.archivo)
        assert not a.sales.equals(b.sales)


def test_ritmos_de_llegada(raw_sintetico, cfg_bandeja, tmp_path):
    m = construir_bandeja(raw_sintetico, tmp_path / "o", cfg_bandeja)
    for fuente, dias in (("ventas", 1), ("petroleo", 2), ("feriados", -30)):
        r = m[(m.fuente == fuente) & (m.version == 1)].iloc[0]
        esperado = pd.Timestamp(r.fecha_negocio) + pd.Timedelta(days=dias, hours=6)
        if fuente == "ventas" and r.archivo in set(m.archivo):  # puede ser tardía por anomalía; aquí no hay
            assert pd.Timestamp(r.llegada_ts) == esperado
        elif fuente != "ventas":
            assert pd.Timestamp(r.llegada_ts) == esperado


def test_anomalias_se_registran_y_faltante_no_tiene_archivo(raw_sintetico, cfg_bandeja, tmp_path):
    out = tmp_path / "out"
    m = construir_bandeja(raw_sintetico, out, cfg_bandeja, con_anomalias=True)
    verdad = pd.read_csv(out / "anomalias_verdad.csv")
    assert set(verdad[verdad.tipo != "correccion"].tipo) == {
        "llegada_tardia", "faltante", "duplicados", "negativos", "fecha_incorrecta", "columna_faltante"}
    for _, r in verdad[verdad.tipo == "faltante"].iterrows():
        assert m[(m.fuente == "ventas") & (m.fecha_negocio == r.fecha_negocio)].empty
    for _, r in verdad[verdad.tipo == "duplicados"].iterrows():
        a = m[(m.fuente == "ventas") & (m.fecha_negocio == r.fecha_negocio)].iloc[0]
        df = pd.read_csv(out / "bandeja" / a.archivo)
        assert df.duplicated(["date", "store_nbr", "family"]).any()
    for _, r in verdad[verdad.tipo == "negativos"].iterrows():
        a = m[(m.fuente == "ventas") & (m.fecha_negocio == r.fecha_negocio)].iloc[0]
        assert (pd.read_csv(out / "bandeja" / a.archivo).sales < 0).any()


def test_ultimos_dias_sin_anomalias(raw_sintetico, cfg_bandeja, tmp_path):
    out = tmp_path / "out"
    construir_bandeja(raw_sintetico, out, cfg_bandeja, con_anomalias=True)
    verdad = pd.read_csv(out / "anomalias_verdad.csv", parse_dates=["fecha_negocio"])
    ultimo = pd.Timestamp("2017-03-10")
    assert (verdad.fecha_negocio <= ultimo - pd.Timedelta(days=10)).all()
