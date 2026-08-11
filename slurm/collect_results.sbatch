#!/bin/bash
#SBATCH --job-name=lora-collect
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
COMMON="${SLURM_SUBMIT_DIR:-.}/slurm/_common.sh"
[[ -f "$COMMON" ]] || { echo "FATAL: cannot find $COMMON" >&2; exit 1; }
source "$COMMON"

BUNDLE="RESULTS_BUNDLE"
banner "collecting -> ${BUNDLE}"

"$PY" aggregate.py                      || echo "[warn] main aggregate failed"
"$PY" -m traditional_ml.aggregate       || echo "[warn] traditional_ml aggregate failed"

rm -rf "${BUNDLE}"
mkdir -p "${BUNDLE}"/{metrics,summaries,charts,logs,models,adapter}

cp -f results/metrics/*.json                   "${BUNDLE}/metrics/"            2>/dev/null || true
cp -f traditional_ml/results/metrics/*.json    "${BUNDLE}/metrics/"            2>/dev/null || true
cp -f results/results.csv results/results.md   "${BUNDLE}/"                    2>/dev/null || true
cp -f traditional_ml/results/results.csv       "${BUNDLE}/results_traditional_ml.csv" 2>/dev/null || true
cp -f traditional_ml/results/results.md        "${BUNDLE}/results_traditional_ml.md"  2>/dev/null || true
cp -f results/charts/*.png                     "${BUNDLE}/charts/"             2>/dev/null || true
cp -f traditional_ml/results/charts/*.png      "${BUNDLE}/charts/"             2>/dev/null || true

cp -f summaries/Phi-3-LoRA-Full_4bit_*_summaries.jsonl "${BUNDLE}/summaries/"  2>/dev/null || true
cp -f traditional_ml/summaries/ML-*_summaries.jsonl    "${BUNDLE}/summaries/"  2>/dev/null || true

cp -f "${ADAPTER_DIR}"/adapter_config.json "${BUNDLE}/adapter/"                2>/dev/null || true
cp -f "${ADAPTER_DIR}"/trainer_logs/trainer_state.json "${BUNDLE}/adapter/"    2>/dev/null || true
for ds in "${DATASETS[@]}"; do
  for m in logreg nb xgb; do
    if [[ -f "traditional_ml/models/${ds}/${m}/meta.json" ]]; then
      cp -f "traditional_ml/models/${ds}/${m}/meta.json" "${BUNDLE}/models/${ds}__${m}__meta.json"
    fi
  done
done

cp -f logs/*.out logs/*.err                     "${BUNDLE}/logs/"             2>/dev/null || true
cp -f traditional_ml/logs/*.out traditional_ml/logs/*.err "${BUNDLE}/logs/"   2>/dev/null || true

{
  echo "# Run manifest"
  echo
  echo "Collected: $(date '+%F %T %Z') on $(hostname)"
  echo
  echo '## Configuration'
  echo '```'
  echo "adapter          : ${ADAPTER_DIR}"
  echo "tokenized data   : ${TOKENIZED_PATH}"
  echo "model index      : ${MODEL_INDEX} (Phi-3-LoRA-Full_4bit)"
  echo "train data       : FULL splits, unbalanced (cnn_dailymail 287113 + xsum 204045)"
  echo "max_seq_length   : 2048   (inference max_input_length 2048 — matched)"
  echo "effective batch  : 32     (per_device 4 x grad_accum 2 x 4 GPUs)"
  echo "decoding         : greedy, per-dataset max_new_tokens {cnn:128, xsum:64}, no_repeat_ngram=3"
  echo "traditional ML   : LogReg / NB / XGBoost, full-data hparam search, full-val calibration"
  echo '```'
  echo
  echo '## Package versions'
  echo '```'
  "$PY" -c "
import importlib.metadata as md
for p in ['torch','transformers','peft','bitsandbytes','accelerate','datasets',
          'scikit-learn','xgboost','evaluate','rouge_score','bert_score','summac','nltk']:
    try: print(f'{p:16s} {md.version(p)}')
    except Exception: print(f'{p:16s} MISSING')
" 2>/dev/null
  echo '```'
  echo
  echo '## Record counts'
  echo '```'
  for f in "${BUNDLE}"/summaries/*.jsonl; do
    [[ -e "$f" ]] && printf '%-52s %s\n' "$(basename "$f")" "$(wc -l < "$f")"
  done
  echo '```'
  echo
  echo '## Metrics present'
  echo '```'
  ls -1 "${BUNDLE}/metrics/" 2>/dev/null
  echo '```'
  echo
  echo '## SLURM accounting'
  echo '```'
  sacct -u "$USER" -S now-7days -X \
    -o JobID,JobName%22,Partition%14,State,Elapsed,Timelimit,MaxRSS 2>/dev/null | tail -40
  echo '```'
} > "${BUNDLE}/RUN_MANIFEST.md"

du -sh "${BUNDLE}"
tar -czf "${BUNDLE}.tar.gz" "${BUNDLE}"
ls -lh "${BUNDLE}.tar.gz"

echo
echo "===== results.csv ====="
cat results/results.csv 2>/dev/null || true
echo
echo "===== traditional_ml results.csv ====="
cat traditional_ml/results/results.csv 2>/dev/null || true

banner "collect done."
