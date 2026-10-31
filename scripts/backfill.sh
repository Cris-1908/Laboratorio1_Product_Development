#!/usr/bin/env bash
# Corre el flujo completo sobre un periodo usando Airflow (hay que haberlo levantado antes; ver README).
# Uso: scripts/backfill.sh 2017-08-01 2017-08-15
set -euo pipefail
[ "$#" -eq 2 ] || { echo "Uso: $0 DESDE HASTA (YYYY-MM-DD)"; exit 1; }
DESDE="$1"; HASTA="$2"
af() { docker compose exec -T airflow-scheduler airflow "$@"; }

for d in ingesta_validacion preparacion_datos pronostico_pedido seguimiento_desempeno; do
  af dags unpause "$d"
done
# La ingesta es diaria; el pronóstico, semanal (su primera tarea vuelve a asegurar la ingesta, así no
# depende del orden en que corran). La preparación y el seguimiento se disparan solos por el asset.
af backfill create --dag-id ingesta_validacion --from-date "$DESDE" --to-date "$HASTA"
af backfill create --dag-id pronostico_pedido  --from-date "$DESDE" --to-date "$HASTA"
echo "Backfill lanzado. Sigue el avance en http://localhost:8080"
