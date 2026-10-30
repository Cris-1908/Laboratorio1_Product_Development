#!/usr/bin/env python
"""Genera la carga histórica y la bandeja a partir de los CSV de Kaggle."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inventario.bandeja import construir_bandeja  # noqa: E402
from inventario.config import cargar_yaml  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default="data/raw", help="carpeta con los CSV de Kaggle")
    p.add_argument("--salida", default="data", help="raíz donde se crean historico/ y bandeja/")
    p.add_argument("--config", default="fuentes.yaml")
    p.add_argument("--anomalias", action="store_true", help="inyecta archivos defectuosos para pruebas")
    a = p.parse_args()

    m = construir_bandeja(a.raw, a.salida, cargar_yaml(a.config), con_anomalias=a.anomalias)
    print(f"Bandeja lista: {len(m)} archivos en {a.salida}/bandeja")
    print(m.groupby("fuente").size().to_string())


if __name__ == "__main__":
    main()
