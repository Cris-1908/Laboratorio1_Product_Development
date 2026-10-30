#!/usr/bin/env python
"""Backtest rolling-origin baseline vs LightGBM; decide si el retador reemplaza al vigente.

Escribe reports/backtest_<corte>.csv y, con --promover, actualiza data/modelos/vigente.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from inventario.config import cargar_yaml  # noqa: E402
from inventario.modelos import backtest, decidir_reemplazo, escribir_vigente, leer_vigente  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corte", required=True, help="corte final (fecha pura = fin de día)")
    p.add_argument("--data", default="data")
    p.add_argument("--promover", action="store_true", help="si el retador gana, lo deja como vigente")
    a = p.parse_args()

    cfg_m, cfg_c = cargar_yaml("modelo.yaml"), cargar_yaml("costos.yaml")
    res = backtest(a.corte, a.data, cfg_m, cfg_c)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))

    # Vigente de partida = baseline; el retador es LightGBM
    dec = decidir_reemplazo(res, "baseline", "lgbm", cfg_m["reemplazo"])
    print("\nDecisión:", json.dumps(dec, ensure_ascii=False, indent=2))

    out = Path("reports")
    out.mkdir(exist_ok=True)
    res.to_csv(out / f"backtest_{a.corte}.csv", index=False)
    (out / f"decision_{a.corte}.json").write_text(json.dumps(dec, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.promover:
        escribir_vigente(a.data, "lgbm" if dec["reemplaza"] else "baseline", dec)
        print("Modelo vigente:", leer_vigente(a.data))


if __name__ == "__main__":
    main()
