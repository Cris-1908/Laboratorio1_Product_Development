#!/usr/bin/env python
"""Entrena LightGBM con lo disponible al corte y guarda el modelo en data/modelos/."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inventario.config import cargar_yaml  # noqa: E402
from inventario.modelos import entrenar_y_guardar  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corte", required=True, help="YYYY-MM-DD (fin de día)")
    p.add_argument("--data", default="data")
    a = p.parse_args()
    ruta = entrenar_y_guardar(a.corte, a.data, cargar_yaml("modelo.yaml"))
    print("Modelo en", ruta)


if __name__ == "__main__":
    main()
