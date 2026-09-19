#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

mkdir -p experiments/g1b_baselines/results reports
exec 9>experiments/g1b_baselines/results/g1b_pipeline.lock
if ! flock -n 9; then
    echo "Another G1B pipeline watcher/evaluation already owns the lock."
    exit 1
fi

exec >>reports/g1b_pipeline.log 2>&1
echo "[$(date --iso-8601=seconds)] Waiting for frozen classical tuning artifact."

tuning_path="experiments/g1b_baselines/results/classical_val_tuning.json"
tuning_pid="${1:-}"
while [[ ! -s "$tuning_path" ]]; do
    if [[ -n "$tuning_pid" ]] && ! kill -0 "$tuning_pid" 2>/dev/null; then
        echo "[$(date --iso-8601=seconds)] Grid-search PID $tuning_pid exited without producing $tuning_path."
        exit 2
    fi
    sleep 30
done

"$python_bin" - "$tuning_path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
required = {"watershed", "mean_shift", "slic_merge"}
if set(data) != required or any(not data[name].get("optimal_params") for name in required):
    raise SystemExit(f"Frozen tuning artifact is incomplete: {path}")
print(f"Validated frozen tuning artifact: {path}")
PY

echo "[$(date --iso-8601=seconds)] Starting resumable official G1B evaluation."
PYTHONPATH=. "$python_bin" -u experiments/g1b_baselines/eval/evaluate_g1b_baselines.py
echo "[$(date --iso-8601=seconds)] G1B evaluation finished."
