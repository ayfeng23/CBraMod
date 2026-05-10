#!/bin/bash
#SBATCH --job-name=cbramod
#SBATCH --output=logs/cbramod_%j.out
#SBATCH --error=logs/cbramod_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h200
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=h200:1
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"

mkdir -p logs

PYTHON="/home/ayf4/.conda/envs/cbramod/bin/python"

echo "Using python: $PYTHON"
"$PYTHON" --version
"$PYTHON" -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

"$PYTHON" -u finetune_main.py
