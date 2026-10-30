"""Carga de configuración versionada y utilidades de costos (newsvendor)."""
from __future__ import annotations

from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]
CONFIG_DIR = RAIZ / "config"


def cargar_yaml(nombre_o_ruta: str | Path) -> dict:
    ruta = Path(nombre_o_ruta)
    if not ruta.is_absolute() and not ruta.exists():
        ruta = CONFIG_DIR / ruta
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def costos_para(familia: str, cfg_costos: dict) -> tuple[float, float]:
    """Devuelve (costo_quiebre, costo_merma) de una familia."""
    c = cfg_costos.get("por_familia", {}).get(familia, cfg_costos["default"])
    cu, co = float(c["costo_quiebre"]), float(c["costo_merma"])
    if cu <= 0 or co <= 0:
        raise ValueError(f"Costos deben ser positivos: {familia} cu={cu} co={co}")
    return cu, co


def cuantil_objetivo(familia: str, cfg_costos: dict) -> float:
    """Razón crítica del newsvendor: cu / (cu + co)."""
    cu, co = costos_para(familia, cfg_costos)
    return cu / (cu + co)
