#!/usr/bin/env python
"""Materializa las features de entrenamiento y pronóstico as-of corte en data/features/."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inventario.config import cargar_yaml  # noqa: E402
from inventario.modelos import preparar_features  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corte", required=True, help="YYYY-MM-DD (fin de día)")
    p.add_argument("--data", default="data")
    a = p.parse_args()
    carpeta = preparar_features(a.corte, a.data, cargar_yaml("modelo.yaml"))
    print("Features en", carpeta)


if __name__ == "__main__":
    main()
