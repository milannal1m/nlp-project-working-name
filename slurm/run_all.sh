#!/bin/bash
# One-command deploy + launch on bwUniCluster.
#
# Bundles: update code -> (optional) env setup -> submit the whole pipeline.
# Run from anywhere inside the repo on a LOGIN node (after you are on the VPN
# and SSH'd in):
#
#     bash slurm/run_all.sh                 # pull + setup + submit
#     bash slurm/run_all.sh --skip-setup    # skip env setup (already built)
#     bash slurm/run_all.sh --cpu-part cpu_il   # override the CPU partition
#     bash slurm/run_all.sh --skip-pull --skip-setup   # just submit
#
# Flags:
#   --skip-setup     skip slurm/setup_env.sh (use when nlp-env is ready)
#   --skip-pull      do not fetch/checkout/pull the Spreadsheet branch
#   --cpu-part NAME  CPU partition for baseline+aggregate jobs (default: cpu)
#   --gpu-part NAME  GPU partition for gen+eval jobs (default: gpu_a100_il)

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

SKIP_SETUP=0
SKIP_PULL=0
CPU_PART=""
GPU_PART=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-setup) SKIP_SETUP=1; shift ;;
        --skip-pull)  SKIP_PULL=1; shift ;;
        --cpu-part)   CPU_PART="${2:?--cpu-part needs a value}"; shift 2 ;;
        --gpu-part)   GPU_PART="${2:?--gpu-part needs a value}"; shift 2 ;;
        -h|--help)    grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1 (try --help)"; exit 1 ;;
    esac
done

echo "================================================================"
echo " Repo: $REPO_ROOT"
echo "================================================================"

# (a) Update code -------------------------------------------------------
if [[ "$SKIP_PULL" -eq 0 ]]; then
    echo "[1/4] Updating to latest origin/Spreadsheet ..."
    git fetch origin
    git checkout Spreadsheet
    git pull --ff-only origin Spreadsheet
else
    echo "[1/4] Skipping git pull (--skip-pull)."
fi

# (b) Show partitions so the user can sanity-check names ----------------
echo "[2/4] Available partitions (sinfo -s):"
sinfo -s 2>/dev/null || echo "      (sinfo unavailable on this node)"
[[ -n "$CPU_PART" ]] && echo "      -> CPU partition override: $CPU_PART"
[[ -n "$GPU_PART" ]] && echo "      -> GPU partition override: $GPU_PART"

# (c) Environment setup -------------------------------------------------
if [[ "$SKIP_SETUP" -eq 0 ]]; then
    echo "[3/4] Setting up conda env (can take 10-30 min on first run) ..."
    bash slurm/setup_env.sh
else
    echo "[3/4] Skipping env setup (--skip-setup)."
fi

# (d) Submit ------------------------------------------------------------
echo "[4/4] Submitting pipeline ..."
export CPU_PARTITION="${CPU_PART:-${CPU_PARTITION:-}}"
export GPU_PARTITION="${GPU_PART:-${GPU_PARTITION:-}}"
bash slurm/submit_all.sh

echo
echo "Launched. Track with:  watch -n 10 squeue --me"
echo "Results land in:       results/results.md , results/results.csv , results/charts/"
