#!/usr/bin/env python
"""Compara los pronósticos ya publicados con las ventas reales disponibles al corte."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inventario.config import cargar_yaml  # noqa: E402
import pandas as pd  # noqa: E402
from inventario.seguimiento import publicar_seguimiento, seguimiento_desempeno  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corte", required=True, help="YYYY-MM-DD (fin de día)")
    p.add_argument("--data", default="data")
    a = p.parse_args()
    df = seguimiento_desempeno(a.corte, a.data)
    ruta = publicar_seguimiento(df, a.corte, a.data)
    pd.set_option("display.width", 220)
    print(df.round(4).to_string(index=False))
    print("Guardado en", ruta)


if __name__ == "__main__":
    main()
