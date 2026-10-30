#!/usr/bin/env python
"""Valida e ingesta lo que haya llegado a la bandeja hasta el corte. Sale con código 1 si hay cuarentena."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inventario.config import cargar_yaml  # noqa: E402
from inventario.ingesta import ingestar  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corte", required=True, help="YYYY-MM-DD (fin de día) o YYYY-MM-DDTHH:MM")
    p.add_argument("--desde", default=None, help="ingesta incremental: solo llegadas posteriores a esta fecha")
    p.add_argument("--data", default="data")
    p.add_argument("--config", default="fuentes.yaml")
    a = p.parse_args()

    rep = ingestar(a.corte, a.data, cargar_yaml(a.config), desde=a.desde)
    print(rep.resumen())
    sys.exit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
