#!/usr/bin/env bash
# Envoltorio del pipeline para cron: activa el entorno, ejecuta `run`, añade la
# salida a logs/pipeline.log y propaga el código de salida (0, 1 o 2).
#
# Ejemplo de crontab (todos los días a las 06:00):
#   0 6 * * * /ruta/a/dsia-airbnb-madrid/scripts/run_pipeline.sh --top-n 20
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
    # shellcheck disable=SC1091
    source .venv/Scripts/activate
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) inicio run $*" >> logs/pipeline.log
python -m airbnb_price run "$@" >> logs/pipeline.log 2>&1
code=$?
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) fin exit=${code}" >> logs/pipeline.log
exit "${code}"
