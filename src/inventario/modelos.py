"""Modelado: panel as-of, features sin fuga, baseline, LightGBM cuantil, métricas,
pedido newsvendor, backtest rolling-origin y criterio de reemplazo.

Principio: las features de un pronóstico emitido al origen `i` usan SOLO información
con índice <= i (el último día cuyo archivo ya llegó al corte). Se entrena con el
enfoque "directo": una fila por (origen, horizonte h, serie), con `h` como feature.
El objetivo se modela en log1p; los cuantiles se conservan al des-transformar.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import costos_para
from .lectura import cargar_manifest, corte_ts, leer_asof

REZAGO_VENTAS = pd.Timedelta(days=1, hours=6)
TIPOS_FERIADO = {"Holiday", "Transfer", "Additional", "Bridge", "Event"}
FEATURES = ["h", "dow", "dom", "mes", "pago", "feriado", "store_nbr", "fam_cod",
            "m7", "m28", "m91", "dow4", "dow_last", "p7", "p28", "tend"]
CATEGORICAS = ["store_nbr", "fam_cod"]


def col(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def fecha_origen(corte) -> pd.Timestamp:
    """Último día de ventas cuyo archivo ya llegó al corte."""
    return (corte_ts(corte) - REZAGO_VENTAS).normalize()


def corte_de_origen(origen) -> pd.Timestamp:
    """Corte (fecha pura = fin de día) en el que `origen` es el último día disponible."""
    return pd.Timestamp(origen).normalize() + pd.Timedelta(days=1)


# --------------------------------------------------------------------------- panel
@dataclass
class Panel:
    fechas: pd.DatetimeIndex
    series: pd.DataFrame            # store_nbr, family (una fila por columna de Y)
    Y: np.ndarray                   # (T, S) ventas; NaN = sin dato
    P: np.ndarray                   # (T, S) onpromotion
    feriados: set = field(default_factory=set)

    @property
    def i_origen(self) -> int:
        return len(self.fechas) - 1


def construir_panel(corte, data_dir, dias: int | None = None) -> Panel:
    origen = fecha_origen(corte)
    v = leer_asof("ventas", corte, data_dir)
    v = v[v.date <= origen]
    if dias is not None:
        v = v[v.date >= origen - pd.Timedelta(days=dias)]
    fechas = pd.date_range(v.date.min(), origen)
    idx = v.set_index(["date", "store_nbr", "family"])
    Y = idx["sales"].unstack(["store_nbr", "family"]).reindex(fechas).sort_index(axis=1)
    P = idx["onpromotion"].unstack(["store_nbr", "family"]).reindex(fechas).reindex(columns=Y.columns)
    series = Y.columns.to_frame(index=False)

    h = leer_asof("feriados", corte, data_dir)
    ok = (h.locale == "National") & (h.type.isin(TIPOS_FERIADO)) & (h.transferred.astype(str).str.lower() == "false")
    feriados = set(pd.to_datetime(h.loc[ok, "date"]).dt.normalize())
    return Panel(fechas, series, Y.to_numpy(float, copy=True), P.to_numpy(float, copy=True), feriados)


# ------------------------------------------------------------------------ features
def _nanmean(a: np.ndarray) -> np.ndarray:
    cnt = np.sum(~np.isnan(a), axis=0)
    s = np.nansum(a, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def features_origen(panel: Panel, i: int, H: int, fam_cats: list[str]) -> pd.DataFrame:
    """Features de las H fechas siguientes al origen `i`. Solo usa Y[<= i], P[<= i]."""
    Y, P = panel.Y, panel.P
    T, S = Y.shape

    def ventana(M, n):
        return M[max(0, i - n + 1): i + 1]

    lm7, lm28, lm91 = (np.log1p(_nanmean(ventana(Y, n))) for n in (7, 28, 91))
    p7, p28 = (_nanmean(ventana(P, n)) for n in (7, 28))
    store = panel.series.store_nbr.to_numpy()
    fam = panel.series.family.to_numpy()
    fam_cod = pd.Categorical(fam, categories=fam_cats).codes

    partes = []
    for h in range(1, H + 1):
        j = i + h
        fecha = panel.fechas[i] + pd.Timedelta(days=h)
        k0 = math.ceil(h / 7)  # el lag mínimo en semanas que cae en o antes del origen
        lags = [Y[j - 7 * k] for k in range(k0, k0 + 4) if j - 7 * k >= 0]
        if lags:
            dow4, ultimo = _nanmean(np.vstack(lags)), lags[0]
        else:
            dow4 = ultimo = np.full(S, np.nan)
        partes.append(pd.DataFrame(dict(
            fecha=fecha, h=h, store_nbr=store, family=fam,
            dow=fecha.dayofweek, dom=fecha.day, mes=fecha.month,
            pago=int(fecha.day == 15 or fecha.is_month_end),
            feriado=int(fecha.normalize() in panel.feriados),
            fam_cod=fam_cod, m7=lm7, m28=lm28, m91=lm91,
            dow4=np.log1p(dow4), dow_last=np.log1p(ultimo),
            p7=p7, p28=p28, tend=lm7 - lm28,
            y_real=Y[j] if j < T else np.full(S, np.nan),
        )))
    return pd.concat(partes, ignore_index=True)


def dataset_entrenamiento(panel: Panel, H: int, ent: dict, fam_cats: list[str]) -> pd.DataFrame:
    paso = int(ent["paso_origen_dias"])
    n = max(1, int(ent["dias_historia"]) // paso)
    i_last = panel.i_origen
    origenes = [i for i in (i_last - H - k * paso for k in range(n)) if i >= 14]
    if not origenes:
        raise ValueError("historia insuficiente para entrenar")
    X = pd.concat([features_origen(panel, i, H, fam_cats) for i in origenes], ignore_index=True)
    return X[X.y_real.notna() & X.m28.notna()].reset_index(drop=True)


# ------------------------------------------------------------------------ modelos
def pronostico_baseline(panel: Panel, H: int, cuantiles: list[float], n_semanas: int = 8) -> pd.DataFrame:
    """Seasonal naive: cuantiles empíricos del mismo día de la semana en las últimas N semanas."""
    i, Y = panel.i_origen, panel.Y
    S = Y.shape[1]
    partes = []
    for h in range(1, H + 1):
        j = i + h
        k0 = math.ceil(h / 7)
        vals = [Y[j - 7 * k] for k in range(k0, k0 + n_semanas) if j - 7 * k >= 0]
        d = dict(fecha=panel.fechas[i] + pd.Timedelta(days=h), h=h,
                 store_nbr=panel.series.store_nbr.to_numpy(), family=panel.series.family.to_numpy())
        if vals:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                Q = np.nanquantile(np.vstack(vals), cuantiles, axis=0)
        else:
            Q = np.full((len(cuantiles), S), np.nan)
        for k, q in enumerate(cuantiles):
            d[col(q)] = Q[k]
        partes.append(pd.DataFrame(d))
    fc = pd.concat(partes, ignore_index=True)
    return fc[fc[col(cuantiles[0])].notna()].reset_index(drop=True)


def _params_lgbm(cfg: dict, q: float) -> dict:
    return dict(objective="quantile", alpha=q, learning_rate=cfg["learning_rate"],
                num_leaves=cfg["num_leaves"], min_data_in_leaf=cfg["min_data_in_leaf"],
                feature_fraction=cfg["feature_fraction"], bagging_fraction=cfg["bagging_fraction"],
                bagging_freq=1, seed=cfg["seed"], num_threads=cfg["num_threads"],
                deterministic=True, force_row_wise=True, verbose=-1)


def entrenar_lgbm(X: pd.DataFrame, cuantiles: list[float], cfg: dict) -> dict:
    import lightgbm as lgb
    ds = lgb.Dataset(X[FEATURES], label=np.log1p(X.y_real.to_numpy()),
                     categorical_feature=CATEGORICAS, free_raw_data=False)
    return {q: lgb.train(_params_lgbm(cfg, q), ds, num_boost_round=int(cfg["n_rounds"])) for q in cuantiles}


def predecir_lgbm(modelos: dict, X: pd.DataFrame, cuantiles: list[float]) -> pd.DataFrame:
    P = np.column_stack([np.expm1(modelos[q].predict(X[FEATURES])) for q in cuantiles])
    P = np.sort(np.clip(P, 0, None), axis=1)  # sin negativos y sin cruce de cuantiles
    fc = X[["fecha", "h", "store_nbr", "family"]].reset_index(drop=True).copy()
    for k, q in enumerate(cuantiles):
        fc[col(q)] = P[:, k]
    return fc


def pronostico_lgbm(panel: Panel, modelos: dict, H: int, cuantiles: list[float], fam_cats: list[str]) -> pd.DataFrame:
    X = features_origen(panel, panel.i_origen, H, fam_cats)
    X = X[X.m28.notna()].reset_index(drop=True)  # solo series activas (con ventas en los últimos 28 días)
    return predecir_lgbm(modelos, X, cuantiles)


# ------------------------------------------------------------------------ métricas
def pinball(y: np.ndarray, Q: np.ndarray, cuantiles: list[float]) -> float:
    e = y[:, None] - Q
    qs = np.asarray(cuantiles)[None, :]
    return float(np.mean(np.maximum(qs * e, (qs - 1) * e)))


def wape(y, p) -> float:
    return float(np.abs(y - p).sum() / max(y.sum(), 1e-9))


def sesgo(y, p) -> float:
    return float((p.sum() - y.sum()) / max(y.sum(), 1e-9))


def costo_negocio(y, orden, cu, co) -> float:
    return float(np.sum(cu * np.maximum(y - orden, 0) + co * np.maximum(orden - y, 0)))


def _interp(qs: list[float], V: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= qs[0]:
        return V[:, 0]
    if alpha >= qs[-1]:
        return V[:, -1]
    k = int(np.searchsorted(qs, alpha)) - 1
    w = (alpha - qs[k]) / (qs[k + 1] - qs[k])
    return (1 - w) * V[:, k] + w * V[:, k + 1]


def pedido_sugerido(fc: pd.DataFrame, cfg_costos: dict, cuantiles: list[float]) -> pd.DataFrame:
    """Newsvendor: se pide el cuantil cu/(cu+co) del pronóstico (interpolado entre cuantiles)."""
    V = fc[[col(q) for q in cuantiles]].to_numpy(float)
    cc = {f: costos_para(f, cfg_costos) for f in fc.family.unique()}
    cu = fc.family.map(lambda f: cc[f][0]).to_numpy(float)
    co = fc.family.map(lambda f: cc[f][1]).to_numpy(float)
    alpha = cu / (cu + co)
    pedido = np.empty(len(fc))
    for a in np.unique(alpha):
        m = alpha == a
        pedido[m] = _interp(cuantiles, V[m], a)
    out = fc.copy()
    out["alpha"], out["cu"], out["co"], out["pedido"] = alpha, cu, co, np.round(pedido, 2)
    return out


def valor_real(real: Panel, fc: pd.DataFrame) -> np.ndarray:
    mi = pd.MultiIndex.from_frame(real.series[["store_nbr", "family"]])
    c = mi.get_indexer(pd.MultiIndex.from_frame(fc[["store_nbr", "family"]]))
    r = real.fechas.get_indexer(pd.DatetimeIndex(fc.fecha))
    ok = (r >= 0) & (c >= 0)
    return np.where(ok, real.Y[np.clip(r, 0, None), np.clip(c, 0, None)], np.nan)


def evaluar(fc: pd.DataFrame, real: Panel, cuantiles: list[float], cfg_costos: dict) -> dict:
    ped = pedido_sugerido(fc, cfg_costos, cuantiles)
    y = valor_real(real, ped)
    m = ~np.isnan(y)
    Q = ped[[col(q) for q in cuantiles]].to_numpy(float)[m]
    p50 = Q[:, cuantiles.index(0.5)]
    costo = costo_negocio(y[m], ped.pedido.to_numpy()[m], ped.cu.to_numpy()[m], ped.co.to_numpy()[m])
    return dict(pinball=pinball(y[m], Q, cuantiles), wape=wape(y[m], p50), sesgo=sesgo(y[m], p50),
                costo=costo, costo_por_unidad=costo / max(y[m].sum(), 1e-9), n=int(m.sum()))


# ------------------------------------------------------------------------ backtest
def backtest(corte_final, data_dir, cfg_modelo: dict, cfg_costos: dict, modelos=("baseline", "lgbm")):
    """Rolling-origin: mismos folds para todos los modelos. Cada fold lee la información
    tal como estaba disponible a su corte (as-of) y se evalúa contra la realidad final."""
    H, qs = int(cfg_modelo["horizonte_dias"]), cfg_modelo["cuantiles"]
    val, ent = cfg_modelo["validacion"], cfg_modelo["entrenamiento"]
    n, paso = int(val["n_folds"]), int(val["paso_dias"])
    real = construir_panel(corte_final, data_dir, dias=n * paso + 2 * H + 10)
    D = real.fechas[-1]
    origenes = [D - pd.Timedelta(days=H + k * paso) for k in reversed(range(n))]
    dias_panel = int(ent["dias_historia"]) + int(ent["dias_extra_features"]) + H

    filas = []
    for f, o in enumerate(origenes, start=1):
        panel = construir_panel(corte_de_origen(o), data_dir, dias=dias_panel)
        fam_cats = sorted(panel.series.family.unique())
        for m in modelos:
            if m == "baseline":
                fc = pronostico_baseline(panel, H, qs, cfg_modelo["baseline"]["n_semanas"])
            else:
                X = dataset_entrenamiento(panel, H, ent, fam_cats)
                mods = entrenar_lgbm(X, qs, cfg_modelo["lightgbm"])
                fc = pronostico_lgbm(panel, mods, H, qs, fam_cats)
            filas.append(dict(fold=f, origen=o, corte=corte_de_origen(o), modelo=m, **evaluar(fc, real, qs, cfg_costos)))
    return pd.DataFrame(filas)


def decidir_reemplazo(res: pd.DataFrame, vigente: str, retador: str, cfg: dict) -> dict:
    """El retador reemplaza al vigente si baja el costo de negocio >= X% en >= K folds
    y su pinball promedio no empeora más de lo permitido."""
    a = res[res.modelo == vigente].set_index("fold")
    b = res[res.modelo == retador].set_index("fold")
    mejora = ((a.costo - b.costo) / a.costo * 100).to_numpy()
    ganados = int((mejora >= cfg["mejora_min_costo_pct"]).sum())
    requeridos = min(int(cfg["folds_min_ganados"]), len(mejora))
    cambio_pin = float((b.pinball.mean() - a.pinball.mean()) / a.pinball.mean() * 100)
    ok_costo = ganados >= requeridos
    ok_pin = cambio_pin <= cfg["max_empeora_pinball_pct"]
    motivo = (f"costo: {ganados}/{len(mejora)} folds con mejora >= {cfg['mejora_min_costo_pct']}% "
              f"(requeridos {requeridos}); pinball {cambio_pin:+.2f}% "
              f"(máx {cfg['max_empeora_pinball_pct']}%)")
    return dict(reemplaza=bool(ok_costo and ok_pin), folds_ganados=ganados, folds_requeridos=requeridos,
                mejora_costo_pct=[round(float(x), 2) for x in mejora], cambio_pinball_pct=round(cambio_pin, 2),
                motivo=motivo)


# --------------------------------------------------------- producción y trazabilidad
def _escribir_atomico(ruta: Path, contenido: str | bytes) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=ruta.parent, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(contenido if isinstance(contenido, bytes) else contenido.encode("utf-8"))
    os.replace(tmp, ruta)


def huella_datos(corte, data_dir) -> str:
    """Hash de lo que se sabía al corte (archivos, versiones y llegada), para trazabilidad."""
    m = cargar_manifest(Path(data_dir) / "bandeja")
    m = m[m.llegada_ts <= corte_ts(corte)].sort_values(["fuente", "fecha_negocio", "version"])
    return hashlib.sha256(m[["archivo", "llegada_ts"]].to_csv(index=False).encode()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or "desconocido"
    except Exception:
        return "desconocido"


def entrenar_y_guardar(corte, data_dir, cfg_modelo: dict) -> Path:
    """Entrena LightGBM con lo disponible al corte y guarda el modelo en data/modelos/lgbm_<origen>/."""
    H, qs, ent = int(cfg_modelo["horizonte_dias"]), cfg_modelo["cuantiles"], cfg_modelo["entrenamiento"]
    panel = construir_panel(corte, data_dir, dias=int(ent["dias_historia"]) + int(ent["dias_extra_features"]) + H)
    fam_cats = sorted(panel.series.family.unique())
    X = dataset_entrenamiento(panel, H, ent, fam_cats)
    mods = entrenar_lgbm(X, qs, cfg_modelo["lightgbm"])
    destino = Path(data_dir) / "modelos" / f"lgbm_{panel.fechas[-1]:%Y-%m-%d}"
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=destino.parent))
    for q, b in mods.items():
        b.save_model(str(tmp / f"{col(q)}.txt"))
    meta = dict(corte=str(corte_ts(corte)), origen=f"{panel.fechas[-1]:%Y-%m-%d}", cuantiles=qs, horizonte=H,
                fam_cats=fam_cats, filas_entrenamiento=int(len(X)), params=cfg_modelo["lightgbm"],
                git_sha=_git_sha(), huella_datos=huella_datos(corte, data_dir))
    (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if destino.exists():
        shutil.rmtree(destino)
    os.replace(tmp, destino)
    return destino


def cargar_modelo(ruta) -> tuple[dict, dict]:
    import lightgbm as lgb
    ruta = Path(ruta)
    meta = json.loads((ruta / "meta.json").read_text(encoding="utf-8"))
    mods = {q: lgb.Booster(model_file=str(ruta / f"{col(q)}.txt")) for q in meta["cuantiles"]}
    return mods, meta


def leer_vigente(data_dir) -> str:
    ruta = Path(data_dir) / "modelos" / "vigente.json"
    return json.loads(ruta.read_text())["modelo"] if ruta.exists() else "baseline"


def escribir_vigente(data_dir, modelo: str, decision: dict | None = None) -> None:
    _escribir_atomico(Path(data_dir) / "modelos" / "vigente.json",
                      json.dumps(dict(modelo=modelo, decision=decision), ensure_ascii=False, indent=2))


def generar_pronostico(corte, data_dir, cfg_modelo: dict, cfg_costos: dict, modelo: str = "vigente") -> pd.DataFrame:
    """Pronóstico + pedido sugerido para el corte. `modelo`: baseline | lgbm (entrena al corte) | vigente."""
    H, qs, ent = int(cfg_modelo["horizonte_dias"]), cfg_modelo["cuantiles"], cfg_modelo["entrenamiento"]
    if modelo == "vigente":
        modelo = "baseline" if leer_vigente(data_dir) == "baseline" else "lgbm"
    panel = construir_panel(corte, data_dir, dias=int(ent["dias_historia"]) + int(ent["dias_extra_features"]) + H)
    if modelo == "baseline":
        fc = pronostico_baseline(panel, H, qs, cfg_modelo["baseline"]["n_semanas"])
        id_modelo = "baseline"
    else:
        ruta = entrenar_y_guardar(corte, data_dir, cfg_modelo)
        mods, meta = cargar_modelo(ruta)
        fc = pronostico_lgbm(panel, mods, H, qs, meta["fam_cats"])
        id_modelo = ruta.name
    out = pedido_sugerido(fc, cfg_costos, qs)
    out.insert(0, "modelo", id_modelo)
    out.insert(0, "corte", corte_ts(corte).strftime("%Y-%m-%d"))
    return out.sort_values(["store_nbr", "family", "fecha"]).reset_index(drop=True)


def publicar_pronostico(df: pd.DataFrame, corte, data_dir, cfg_modelo: dict) -> Path:
    """Escritura idempotente (reemplazo atómico) con metadata de trazabilidad."""
    c = corte_ts(corte).strftime("%Y-%m-%d")
    carpeta = Path(data_dir) / "forecasts" / f"corte={c}"
    _escribir_atomico(carpeta / "pronostico.csv", df.to_csv(index=False))
    meta = dict(corte=c, modelo=str(df.modelo.iloc[0]), filas=int(len(df)), git_sha=_git_sha(),
                huella_datos=huella_datos(corte, data_dir), horizonte=cfg_modelo["horizonte_dias"],
                cuantiles=cfg_modelo["cuantiles"])
    _escribir_atomico(carpeta / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
    return carpeta
