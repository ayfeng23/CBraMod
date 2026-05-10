#!/bin/bash
#SBATCH --job-name=tueg_preprocess
#SBATCH --array=0-150
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ayf4/scratch_pi_zf59/ayf4/data/tueg_v2.0.1_shards/logs/shard_%a.out
#SBATCH --error=/home/ayf4/scratch_pi_zf59/ayf4/data/tueg_v2.0.1_shards/logs/shard_%a.err

set -euo pipefail

# Bench showed n_jobs=1 is fastest for 19-channel filter; pin BLAS to 1 thread.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

source /home/ayf4/miniconda3/etc/profile.d/conda.sh
conda activate cbramod

INPUT_ROOT=/home/ayf4/scratch_pi_zf59/ayf4/data/v2.0.1/edf
OUTPUT_ROOT=/home/ayf4/scratch_pi_zf59/ayf4/data/tueg_v2.0.1_shards

mkdir -p "$OUTPUT_ROOT/logs"

SHARD=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")

python /home/ayf4/cbramod/preprocessing/preprocessing_tueg_shard.py \
    --shard  "$SHARD" \
    --input_root  "$INPUT_ROOT" \
    --output_root "$OUTPUT_ROOT"
