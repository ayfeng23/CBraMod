#!/bin/bash
#SBATCH --job-name=cbramod_validate
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ayf4/scratch_pi_zf59/ayf4/aic_results/cbramod_pretrain/validate_%j.out
#SBATCH --error=/home/ayf4/scratch_pi_zf59/ayf4/aic_results/cbramod_pretrain/validate_%j.err

set -euo pipefail

mkdir -p /home/ayf4/scratch_pi_zf59/ayf4/aic_results/cbramod_pretrain

source /home/ayf4/miniconda3/etc/profile.d/conda.sh
conda activate cbramod

echo "=== node $(hostname)  $(date) ==="
nvidia-smi || true

cd /home/ayf4/cbramod
python testing/pretrain_validate_gpu.py \
    --data        /home/ayf4/scratch_pi_zf59/ayf4/data/tueg_v2.0.1_lmdb \
    --device      cuda:0 \
    --batch_size  128 \
    --num_workers 8 \
    --iters       100
