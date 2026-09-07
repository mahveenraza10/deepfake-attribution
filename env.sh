#!/bin/bash
# ---------------------------------------------------------------------------
# env.sh — every path this experiment uses, in one place. Sourced, never run:
#
#     source ./env.sh
#
# Set the three variables below for your site, either by editing them here or by
# exporting them beforehand:
#
#     export FAVC=/data/FakeAVCeleb_v1.2
#     export EVATTR_STORE=/scratch/$USER/evattr
#     bash setup.sh
# ---------------------------------------------------------------------------

# --- site configuration ----------------------------------------------------
# FakeAVCeleb root: the directory that directly contains the four quadrant
# folders (RealVideo-RealAudio, FakeVideo-RealAudio, ...) and meta_data.csv.
export FAVC="${FAVC:?set FAVC to the FakeAVCeleb_v1.2 root}"

# Where the venv and outputs go. Keep this off a quota'd home directory: the
# venv alone is ~5.5 GB.
export EVATTR_STORE="${EVATTR_STORE:?set EVATTR_STORE to a writable directory with ~10 GB free}"

# Hugging Face cache. Point several users at one shared directory and the
# ~23 GB Omni download is paid once rather than per user.
export HF_HOME="${HF_HOME:-${EVATTR_STORE}/hf-cache}"

# Uncomment on a site whose compute nodes have no route to huggingface.co, and
# pre-stage the weights with slurm/download.sbatch from a node that does.
# export HF_HUB_OFFLINE=1

# ffmpeg is needed to decode the clips' audio tracks. Add its directory here if
# it is not already on PATH (some clusters serve it from a module or CVMFS).
# export PATH="${PATH}:/path/to/ffmpeg/bin"
# ---------------------------------------------------------------------------

export EVATTR_VENV="${EVATTR_VENV:-${EVATTR_STORE}/venv}"
export EVATTR_OUT="${EVATTR_OUT:-${EVATTR_STORE}/runs}"

# Without this a batch job's .out stays empty until the job ends, which makes
# run_vlm.py's per-clip progress line useless for watching a long run.
export PYTHONUNBUFFERED=1
