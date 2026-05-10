#!/bin/bash
#SBATCH --job-name=cbramod_resume
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/ayf4/scratch_pi_zf59/ayf4/aic_results/cbramod_pretrain/resume_%j.out
#SBATCH --error=/home/ayf4/scratch_pi_zf59/ayf4/aic_results/cbramod_pretrain/resume_%j.err

set -euo pipefail

# Required env: RESUME_CKPT=/path/to/epochN_lossX.pth
# Optional:
#   EPOCHS (total target epochs, default 40)
#   RESUME_EPOCH (start epoch, used when no .full.pth companion; scheduler is fast-forwarded)
#   RESUME_LR (override LR after restore, unset = scheduler decides)
#   BATCH_SIZE (128), CLIP_VALUE (1)
: "${RESUME_CKPT:?set RESUME_CKPT to the checkpoint path}"
EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-128}
CLIP_VALUE=${CLIP_VALUE:-1}

OUTDIR=/home/ayf4/scratch_pi_zf59/ayf4/aic_results/cbramod_pretrain/resume_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

source /home/ayf4/miniconda3/etc/profile.d/conda.sh
conda activate cbramod

echo "=== node $(hostname)  $(date)  jobid=$SLURM_JOB_ID ==="
echo "resume=$RESUME_CKPT  total_epochs=$EPOCHS  resume_lr=$RESUME_LR  outdir=$OUTDIR"
nvidia-smi

EXTRA_ARGS=()
if [[ -n "${RESUME_EPOCH:-}" ]]; then
    EXTRA_ARGS+=(--resume_epoch "$RESUME_EPOCH")
fi
if [[ -n "${RESUME_LR:-}" ]]; then
    EXTRA_ARGS+=(--resume_lr "$RESUME_LR")
fi

cd /home/ayf4/cbramod
python pretrain_main.py \
    --cuda 0 \
    --epochs        "$EPOCHS" \
    --batch_size    "$BATCH_SIZE" \
    --lr            5e-4 \
    --clip_value    "$CLIP_VALUE" \
    --resume        "$RESUME_CKPT" \
    --dataset_dir   /home/ayf4/scratch_pi_zf59/ayf4/data/tueg_v2.0.1_lmdb \
    --model_dir     "$OUTDIR" \
    --wandb \
    --wandb_project cbramod-pretrain-tueg \
    --wandb_name    "resume-${SLURM_JOB_ID}-from-$(basename "$RESUME_CKPT" .pth)" \
    "${EXTRA_ARGS[@]}"
