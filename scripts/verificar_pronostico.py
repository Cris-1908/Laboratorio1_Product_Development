#!/usr/bin/env python
"""Regenera un pronóstico publicado desde cero y verifica que el resultado sea idéntico."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inventario.config import cargar_yaml  # noqa: E402
from inventario.seguimiento import verificar_reproduccion  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corte", required=True, help="YYYY-MM-DD (fin de día)")
    p.add_argument("--data", default="data")
    a = p.parse_args()
    r = verificar_reproduccion(a.corte, a.data, cargar_yaml("modelo.yaml"), cargar_yaml("costos.yaml"))
    print(r)
    sys.exit(0 if r["igual"] else 1)


if __name__ == "__main__":
    main()
