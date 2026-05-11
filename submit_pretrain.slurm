#!/bin/bash
#SBATCH --job-name=cbramod_pretrain
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/home/ayf4/scratch_pi_zf59/ayf4/aic_results/cbramod_pretrain/pretrain_%j.out
#SBATCH --error=/home/ayf4/scratch_pi_zf59/ayf4/aic_results/cbramod_pretrain/pretrain_%j.err

set -euo pipefail

EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-5e-4}
CLIP_VALUE=${CLIP_VALUE:-1}    # paper default 1.0
OBJECTIVE=${OBJECTIVE:-recon}  # recon (default) or ntp

OUTDIR=/home/ayf4/scratch_pi_zf59/ayf4/aic_results/cbramod_pretrain/run_${SLURM_JOB_ID}_${OBJECTIVE}
mkdir -p "$OUTDIR"

source /home/ayf4/miniconda3/etc/profile.d/conda.sh
conda activate cbramod

echo "=== node $(hostname)  $(date)  jobid=$SLURM_JOB_ID ==="
echo "objective=$OBJECTIVE  epochs=$EPOCHS  batch=$BATCH_SIZE  lr=$LR  outdir=$OUTDIR"
nvidia-smi

cd /home/ayf4/cbramod
python pretrain_main.py \
    --cuda 0 \
    --objective     "$OBJECTIVE" \
    --epochs        "$EPOCHS" \
    --batch_size    "$BATCH_SIZE" \
    --lr            "$LR" \
    --clip_value    "$CLIP_VALUE" \
    --dataset_dir   /home/ayf4/scratch_pi_zf59/ayf4/data/tueg_v2.0.1_lmdb \
    --model_dir     "$OUTDIR" \
    --wandb \
    --wandb_project cbramod-pretrain-tueg \
    --wandb_name    "pretrain-${SLURM_JOB_ID}-${OBJECTIVE}-e${EPOCHS}-b${BATCH_SIZE}-clip${CLIP_VALUE}"
