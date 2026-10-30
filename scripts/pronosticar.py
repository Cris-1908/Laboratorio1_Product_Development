#!/usr/bin/env python
"""Genera y publica el pronóstico + pedido sugerido para un corte, usando solo lo disponible a ese corte."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inventario.config import cargar_yaml  # noqa: E402
from inventario.modelos import generar_pronostico, publicar_pronostico  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corte", required=True, help="YYYY-MM-DD (fin de día)")
    p.add_argument("--modelo", default="vigente", choices=["vigente", "baseline", "lgbm"])
    p.add_argument("--data", default="data")
    a = p.parse_args()

    cfg_m, cfg_c = cargar_yaml("modelo.yaml"), cargar_yaml("costos.yaml")
    df = generar_pronostico(a.corte, a.data, cfg_m, cfg_c, modelo=a.modelo)
    carpeta = publicar_pronostico(df, a.corte, a.data, cfg_m)
    print(f"{len(df)} filas ({df.modelo.iloc[0]}) publicadas en {carpeta}")


if __name__ == "__main__":
    main()
