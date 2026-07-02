#!/bin/bash
# Graph-level PNC age regression (harness experiment 0009). ONE script, selected by env vars:
#   TBC_AGE_MODEL   = gcn | mpsn      (gcn -> graph/gcn batch=32 ; mpsn -> simplicial/mpsn batch=1)
#   TBC_AGE_MODALITY= fc  | sc
#   TBC_AGE_FEATURE = identity | profile
#   TBC_AGE_SHUFFLE = 1               (optional: use the shuffle-age gate cache; fc/identity only)
# npz caches are SHARED by both models per (modality x feature); data_name includes the model so
# the processed-lift cache + split dir never collide (they are keyed by data_name, not npz content).
# Submit e.g.:
#   cluster-submit --node gpunode01 --gpu rtx2080 slurm/age.sh -J age-gcn-fc-id  -- --export=ALL,TBC_AGE_MODEL=gcn,TBC_AGE_MODALITY=fc,TBC_AGE_FEATURE=identity
#SBATCH --job-name=age-pnc
#SBATCH --partition=rad2
#SBATCH --qos=16cpu
#SBATCH --account=rad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

set -euo pipefail

MODEL="${TBC_AGE_MODEL:-gcn}"
MODALITY="${TBC_AGE_MODALITY:-fc}"
FEATURE="${TBC_AGE_FEATURE:-identity}"
SHUFFLE="${TBC_AGE_SHUFFLE:-0}"

REPO="$(pwd)"
SIF="$REPO/topobench.sif"
NPZ="/workspace/npz_cache"

case "$MODEL" in
  gcn)  MODEL_CFG="graph/gcn";        BATCH=32 ;;
  mpsn) MODEL_CFG="simplicial/mpsn";  BATCH=1  ;;
  *) echo "[age] ERROR: unknown TBC_AGE_MODEL='$MODEL'" >&2; exit 2 ;;
esac

if [ "$SHUFFLE" = "1" ]; then
  CACHE="$NPZ/brain_pnc_age_fc_identity_shuffle"
  DATA_NAME="brain_pnc_age_${MODEL}_fc_identity_shuf"
else
  CACHE="$NPZ/brain_pnc_age_${MODALITY}_${FEATURE}"
  DATA_NAME="brain_pnc_age_${MODEL}_${MODALITY}_${FEATURE}"
fi

echo "======================================================================"
echo "[age] model=$MODEL modality=$MODALITY feature=$FEATURE shuffle=$SHUFFLE"
echo "[age] git SHA   : $(git rev-parse HEAD 2>/dev/null || echo '?')"
echo "[age] node      : $(hostname)"
echo "[age] data_name : $DATA_NAME"
echo "[age] cache_dir : $CACHE"
echo "======================================================================"
nvidia-smi || true

singularity exec --nv \
  --bind "$REPO:/workspace" \
  --pwd /workspace \
  --env WANDB_MODE=offline \
  --env PYTHONPATH=/workspace \
  "$SIF" \
  python -m topobench \
    dataset=graph/brain_pnc_age \
    model="$MODEL_CFG" \
    model.readout.pooling_type=mean \
    dataset.dataloader_params.batch_size="$BATCH" \
    dataset.loader.parameters.data_name="$DATA_NAME" \
    dataset.loader.parameters.cache_dir="$CACHE" \
    trainer.max_epochs=100

echo "[age] DONE model=$MODEL modality=$MODALITY feature=$FEATURE shuffle=$SHUFFLE"
