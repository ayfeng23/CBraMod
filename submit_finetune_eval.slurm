#!/bin/bash
#SBATCH --job-name=cbramod_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# --partition, --time, --output, --error are set by the sbatch caller in pretrain_trainer.py.

set -euo pipefail

# Strip inherited wandb service-token from the parent pretrain process; otherwise
# wandb.init() tries to connect to the parent's unix socket which only exists
# on the pretrain node, not this eval compute node.
unset WANDB_SERVICE

if [ "$#" -lt 6 ]; then
  echo "usage: $0 <ckpt_path> <pretrain_epoch> <parent_run_dir> <save_models:true|false> <objective> <need_mask:true|false> [<wandb_run_id> <wandb_project>]" >&2
  exit 2
fi

CKPT_PATH="$1"
PRETRAIN_EPOCH="$2"        # 1-indexed; matches the file label saved by pretrain_trainer
PARENT_RUN_DIR="$3"
SAVE_MODELS_BOOL="$4"      # true | false
OBJECTIVE="$5"             # recon | ntp
NEED_MASK_BOOL="$6"        # true | false
WANDB_RUN_ID="${7:-}"
WANDB_PROJECT="${8:-cbramod-pretrain-tueg}"

OUTPUT_DIR="${PARENT_RUN_DIR}/eval_epoch${PRETRAIN_EPOCH}"
mkdir -p "$OUTPUT_DIR"

if [ "$NEED_MASK_BOOL" = "true" ]; then NEED_MASK_FLAG="--need-mask"; else NEED_MASK_FLAG="--no-need-mask"; fi
if [ "$SAVE_MODELS_BOOL" = "true" ]; then SAVE_MODELS_FLAG="--save-models"; else SAVE_MODELS_FLAG="--no-save-models"; fi

WANDB_ARGS=()
if [ -n "$WANDB_RUN_ID" ]; then
  WANDB_ARGS+=(--parent-wandb-run-id "$WANDB_RUN_ID" --wandb-project "$WANDB_PROJECT" --pretrain-epoch "$PRETRAIN_EPOCH")
fi

cd /home/ayf4/cbramod
echo "=== eval start: $(hostname)  $(date)  jobid=${SLURM_JOB_ID:-?} ==="
echo "ckpt=$CKPT_PATH"
echo "pretrain_epoch=$PRETRAIN_EPOCH  save_models=$SAVE_MODELS_BOOL  objective=$OBJECTIVE  need_mask=$NEED_MASK_BOOL"
echo "wandb_run_id=${WANDB_RUN_ID:-<unset>}  wandb_project=$WANDB_PROJECT"
echo "outdir=$OUTPUT_DIR"
nvidia-smi -L

/home/ayf4/.conda/envs/cbramod/bin/python -u finetune_eval_runner.py \
  --ckpt-path "$CKPT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --benchmarks all \
  --objective "$OBJECTIVE" \
  $NEED_MASK_FLAG \
  $SAVE_MODELS_FLAG \
  "${WANDB_ARGS[@]}"
