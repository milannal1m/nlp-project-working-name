#!/bin/bash
echo "========================================="
echo "Building Isolated GPU Environment (Cluster & Local)"
echo "========================================="

# 1. Smart Cluster Detection
# If this runs on the Uni Ulm cluster, it loads the modules. If local, it skips silently.
if command -v module &> /dev/null; then
    echo "Cluster environment detected. Loading university conda modules..."
    module load devel/miniforge/25.3.1-python-3.12
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Local environment detected. Assuming Conda is already installed..."
fi

# Hook Conda into the bash script so 'conda activate' works
eval "$(conda shell.bash hook)"

# 2. Terms of Service Bypass (Prevents Anaconda from freezing the script)
echo "Silencing Anaconda ToS..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main > /dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r > /dev/null 2>&1 || true

# 3. Create the Clean Python 3.7 Sandbox
echo "Building pristine Python 3.7 sandbox (qa-eval)..."
conda remove -n qa-eval --all -y 2>/dev/null || true
conda create -n qa-eval python=3.7 pip -c conda-forge -y
conda activate qa-eval

# 4. Lock Legacy Build Tools
echo "Locking pip and legacy build parameters..."
pip install "pip==21.3.1" "setuptools==59.5.0" wheel Cython

# 5. BYPASS COMPILERS: Natively pull heavy legacy binaries via Conda
echo "Installing massive core frameworks via pre-compiled Conda binaries..."
conda install -c conda-forge spacy=2.2.4 transformers=3.0.2 python-lmdb cffi numpy scipy h5py jsonnet -y

# 6. Install the Language Models and Support Libraries
echo "Fetching Spacy english components..."
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz

echo "Securing AllenNLP layer..."
pip install --no-build-isolation allennlp==1.1.0 allennlp-models==1.1.0 overrides==3.1.0
pip install edlib

# 7. Install CUDA-Enabled PyTorch for Nvidia GPUs
echo "Binding PyTorch binaries to Nvidia CUDA drivers..."
pip install --no-deps torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

echo "========================================="
echo "SUCCESS! GPU Environment built cleanly."
echo "To run the pipeline:"
echo "1. conda activate qa-eval"
echo "2. python run_qa.py (or submit via SLURM)"
echo "========================================="