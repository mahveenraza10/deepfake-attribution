#!/bin/bash
# ---------------------------------------------------------------------------
# setup.sh — build the python environment. Run ONCE, from the repo root, on a
#            machine with internet access:
#
#              bash setup.sh
#
# Paths come from env.sh; override FAVC / EVATTR_STORE there or in the
# environment. Model weights are fetched separately by slurm/download.sbatch so
# that a slow ~23 GB pull runs as a batch job rather than on a login node, which
# may kill it on memory or on logout.
#
# Uses `uv`; swap in `python -m venv` + `pip install` if you prefer.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

if [[ ! -x "${EVATTR_VENV}/bin/python" ]]; then
    echo "== creating venv at ${EVATTR_VENV}"
    uv venv --python 3.12 "${EVATTR_VENV}"
fi
export VIRTUAL_ENV="${EVATTR_VENV}"

echo "== installing torch (cu124 -- change the index-url to match your CUDA)"
uv pip install --python "${EVATTR_VENV}/bin/python" \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "== installing the rest (see requirements.txt for why librosa is pinned)"
uv pip install --python "${EVATTR_VENV}/bin/python" -r requirements.txt

mkdir -p "${EVATTR_OUT}"
echo "== done. next: sbatch slurm/preflight.sbatch, then sbatch slurm/run.sbatch"
