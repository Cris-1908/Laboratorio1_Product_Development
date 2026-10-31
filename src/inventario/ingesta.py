"""Ingesta: bandeja (solo lectura) -> procesado (válido) o cuarentena (inválido).

- Valida cada archivo disponible al corte (todas las versiones, no solo la última).
- Válido: copia byte a byte a data/procesado/<fuente>/<archivo> con reemplazo atómico.
- Inválido: copia a data/cuarentena/<fuente>/<archivo> + <archivo>.motivo.json. La
  bandeja nunca se modifica. Después el llamador usa `exigir_ok` para fallar la tarea.
- Idempotente: re-ejecutar produce exactamente los mismos archivos.
- `desde` permite ingesta incremental (solo llegadas en (desde, corte]).
- Reporta fechas esperadas sin archivo (`faltantes`) sin fallar: pueden ser retrasos.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .lectura import cargar_manifest, corte_ts
from .validacion import validar_df


class IngestaInvalida(RuntimeError):
    pass


@dataclass
class ReporteIngesta:
    corte: pd.Timestamp
    procesados: list[str] = field(default_factory=list)
    cuarentena: dict[str, list[str]] = field(default_factory=dict)
    llegada_cuarentena: dict[str, pd.Timestamp] = field(default_factory=dict)
    advertencias: dict[str, list[str]] = field(default_factory=dict)
    faltantes: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.cuarentena

    def resumen(self) -> str:
        l = [f"Corte {self.corte}: {len(self.procesados)} procesados, "
             f"{len(self.cuarentena)} en cuarentena, "
             f"{sum(len(v) for v in self.faltantes.values())} fechas esperadas sin archivo"]
        for a, mot in self.cuarentena.items():
            l.append(f"  CUARENTENA {a}: {'; '.join(mot)}")
        for f, fechas in self.faltantes.items():
            if fechas:
                l.append(f"  FALTANTES {f}: {', '.join(fechas)}")
        return "\n".join(l)


def exigir_ok(rep: ReporteIngesta, desde=None) -> None:
    """Falla si hay archivos en cuarentena. Con `desde`, solo cuenta los que llegaron después
    (así un archivo inválido viejo, ya reportado, no hace fallar todas las corridas siguientes)."""
    if desde is None:
        nuevos = dict(rep.cuarentena)
    else:
        limite = corte_ts(desde)
        nuevos = {a: m for a, m in rep.cuarentena.items() if rep.llegada_cuarentena[a] > limite}
    if nuevos:
        raise IngestaInvalida("; ".join(f"{a}: {', '.join(m)}" for a, m in nuevos.items()))


def _copia_atomica(origen: Path, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=destino.parent, suffix=".tmp")
    os.close(fd)
    try:
        shutil.copyfile(origen, tmp)
        os.replace(tmp, destino)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def _fechas_esperadas(fuente: str, corte: pd.Timestamp, cfg: dict) -> set[pd.Timestamp]:
    """Fechas cuyo archivo ya debería haber llegado al corte (ventas y transacciones)."""
    if fuente not in ("ventas", "transacciones"):
        return set()
    rez = int(cfg["fuentes"][fuente]["rezago_dias"])
    inicio = pd.Timestamp(cfg["bandeja"]["corte_historico"]) + pd.Timedelta(days=1)
    fin = (corte - pd.Timedelta(days=rez, hours=6)).normalize()
    if fin < inicio:
        return set()
    sin_op = set(cfg.get("validacion", {}).get("dias_sin_operacion", []))
    return {d for d in pd.date_range(inicio, fin) if f"{d:%m-%d}" not in sin_op}


def ingestar(corte, data_dir, cfg: dict, desde=None) -> ReporteIngesta:
    data_dir = Path(data_dir)
    corte = corte_ts(corte)
    rep = ReporteIngesta(corte=corte)
    bandeja, proc, cuar = data_dir / "bandeja", data_dir / "procesado", data_dir / "cuarentena"

    proc.mkdir(parents=True, exist_ok=True)
    cuar.mkdir(parents=True, exist_ok=True)
    m = cargar_manifest(bandeja)
    nuevos = m[m.llegada_ts <= corte]
    if desde is not None:
        nuevos = nuevos[nuevos.llegada_ts > corte_ts(desde)]
    nuevos = nuevos.sort_values(["llegada_ts", "fuente", "fecha_negocio", "version"])

    for r in nuevos.itertuples():
        origen = bandeja / r.archivo
        try:
            df = pd.read_csv(origen)
            res = validar_df(df, r.fuente, r.fecha_negocio)
            errores, avisos = res.errores, res.advertencias
        except Exception as e:  # archivo ilegible / corrupto
            errores, avisos = [f"no se pudo leer: {type(e).__name__}"], []
        if avisos:
            rep.advertencias[r.archivo] = avisos
        if errores:
            rep.cuarentena[r.archivo] = errores
            rep.llegada_cuarentena[r.archivo] = r.llegada_ts
            _copia_atomica(origen, cuar / r.archivo)
            (cuar / f"{r.archivo}.motivo.json").write_text(
                json.dumps({"archivo": r.archivo, "llegada_ts": r.llegada_ts.isoformat(),
                            "errores": errores}, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            _copia_atomica(origen, proc / r.archivo)
            rep.procesados.append(r.archivo)

    # fechas esperadas sin ningún archivo llegado al corte
    llegados = m[m.llegada_ts <= corte]
    for fuente in ("ventas", "transacciones"):
        tienen = set(llegados[llegados.fuente == fuente].fecha_negocio)
        falt = sorted(_fechas_esperadas(fuente, corte, cfg) - tienen)
        rep.faltantes[fuente] = [f"{d:%Y-%m-%d}" for d in falt]
    return rep
