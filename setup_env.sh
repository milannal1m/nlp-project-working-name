#!/bin/bash
set -e

ENV_NAME="nlp-project"

module load devel/miniforge/25.3.1-python-3.12
source /opt/bwhpc/common/devel/miniforge/25.3.1-py3.12/etc/profile.d/conda.sh

conda create -n "$ENV_NAME" python=3.11 -y
conda activate "$ENV_NAME"

pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

pip install \
  "transformers>=4.40.0,<5.0.0" \
  "datasets>=2.18.0" \
  "bitsandbytes>=0.43.0" \
  "accelerate>=0.27.0" \
  "sumy==0.12.0" \
  "evaluate>=0.4.0" \
  "rouge-score>=0.1.2" \
  "sentencepiece>=0.1.99" \
  "protobuf>=3.20.0" \
  "absl-py>=1.0.0" \
  "bert-score>=0.3.13" \
  "nltk>=3.8.1"

pip install --no-deps git+https://github.com/tingofurro/summac.git

# Fix summac incompatibility with transformers>=4.40: truncation_strategy is passed twice
SUMMAC_DIR=$(python -c "import summac, os; print(os.path.dirname(summac.__file__))")
sed -i 's/truncation=True, max_length=self.max_input_length, return_tensors="pt", truncation_strategy="only_first"/truncation=True, max_length=self.max_input_length, return_tensors="pt"/' \
  "$SUMMAC_DIR/model_summac.py"

echo "Done. Activate with: conda activate $ENV_NAME"
