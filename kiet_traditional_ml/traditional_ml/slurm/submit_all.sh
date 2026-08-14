#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || echo .)}"
mkdir -p traditional_ml/logs
SL=traditional_ml/slurm

PART=()
[[ -n "${CPU_PARTITION:-}" ]] && PART+=(--partition="${CPU_PARTITION}")

PY="${TML_PYTHON:-$HOME/.conda/envs/nlp-env/bin/python}"
if [[ "${SKIP_BOOTSTRAP:-0}" != "1" ]]; then
  echo "[bootstrap] preparing nlp-env via $PY ..."
  "$PY" -c "import sklearn, scipy, joblib" 2>/dev/null \
    || "$PY" -m pip install --quiet "scikit-learn>=1.3" "scipy>=1.10" "joblib>=1.3"
  "$PY" -c "import xgboost" 2>/dev/null \
    || "$PY" -m pip install --quiet "xgboost>=2.0"
  "$PY" -c "from importlib.metadata import version; v=version('peft').split('.'); assert (int(v[0]),int(v[1]))<(0,19)" 2>/dev/null \
    || "$PY" -m pip install --quiet "peft<0.19"
  "$PY" -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt_tab','wordnet','omw-1.4')]"
  "$PY" -c "import sklearn, scipy, joblib, datasets, evaluate, xgboost; print('[bootstrap] ok, xgboost', xgboost.__version__)"
fi

submit() {
  local label="$1"; shift
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN sbatch --parsable ${PART[*]:-} $*" >&2
    echo "000000"
  else
    sbatch --parsable ${PART[@]+"${PART[@]}"} "$@"
  fi
}

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for s in train generate evaluate aggregate; do
    if out=$(sbatch --test-only ${PART[@]+"${PART[@]}"} "${SL}/${s}.sbatch" 2>&1); then
      echo "[preflight ok]   ${s}"
    else
      echo "[preflight FAIL] ${s}"; echo "${out}" | sed 's/^/    /'; exit 1
    fi
  done
fi

JID_TRAIN=$(submit train "${SL}/train.sbatch")
echo "train:     ${JID_TRAIN}"

JID_TRAIN_B=$(submit train-B --job-name=tml-train-B \
  --dependency="afterany:${JID_TRAIN}" "${SL}/train.sbatch")
echo "train-B:   ${JID_TRAIN_B}  (afterany train — resumes or no-ops)"

JID_GEN=$(submit generate --dependency="afterany:${JID_TRAIN_B}" "${SL}/generate.sbatch")
echo "generate:  ${JID_GEN}  (afterany train-B)"

JID_GEN_B=$(submit generate-B --job-name=tml-gen-B \
  --dependency="afterany:${JID_GEN}" "${SL}/generate.sbatch")
echo "generate-B: ${JID_GEN_B}  (afterany generate)"

JID_EVAL=$(submit evaluate --dependency="afterany:${JID_GEN_B}" "${SL}/evaluate.sbatch")
echo "evaluate:  ${JID_EVAL}  (afterany generate-B)"

JID_EVAL_B=$(submit evaluate-B --job-name=tml-eval-B \
  --dependency="afterany:${JID_EVAL}" "${SL}/evaluate.sbatch")
echo "evaluate-B: ${JID_EVAL_B}  (afterany evaluate)"

JID_AGG=$(submit aggregate --dependency="afterany:${JID_EVAL_B}" "${SL}/aggregate.sbatch")
echo "aggregate: ${JID_AGG}  (afterany evaluate-B)"

echo "Submitted. Watch: squeue -u \$USER ; tail -f traditional_ml/logs/*.out"
