#!/bin/bash
# GCN baseline for the PNC Yeo-7 community task (harness experiment 0008), matched to MPSN 0007.
# ONE script, three variants selected by env var TBC_GCN_VARIANT (default: honest):
#   honest   — primary matched baseline, N=973, k=10, local_stats (6-dim), identity WITHHELD.
#   leak     — identity-leak gate: node features = one-hot region (400-dim). Expect ~perfect.
#   shuffle  — label-shuffle gate: labels permuted within subject. Expect ~majority chance.
# Submit e.g.:
#   cluster-submit --node gpunode0N --gpu <type> slurm/gcn.sh -- -J gcn-pnc-honest
#   cluster-submit --node gpunode0N --gpu <type> slurm/gcn.sh -- -J gcn-pnc-leak    --export=ALL,TBC_GCN_VARIANT=leak
#   cluster-submit --node gpunode0N --gpu <type> slurm/gcn.sh -- -J gcn-pnc-shuffle --export=ALL,TBC_GCN_VARIANT=shuffle
# --node resolves partition/qos/account; --gpu pins the GPU type. WANDB is forced offline.
#SBATCH --job-name=gcn-pnc
#SBATCH --partition=rad2
#SBATCH --qos=16cpu
#SBATCH --account=rad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

set -euo pipefail

VARIANT="${TBC_GCN_VARIANT:-honest}"

REPO="$(pwd)"                 # = cluster project_root (the deployed fork), bind-mounted to /workspace
SIF="$REPO/topobench.sif"
NPZ="/workspace/npz_cache"    # npz caches uploaded via cluster-upload-dataset, seen inside the container

# Per-variant overrides. Distinct data_name per variant so the processed cache + split dir never
# collide (they are keyed by data_name, NOT by npz content). cache_dir points at the matching npz.
case "$VARIANT" in
  honest)
    DATA_NAME="brain_pnc_community_gcn_k10full"
    CACHE="$NPZ/brain_pnc_community_k10_full"
    EXTRA=(trainer.max_epochs=100)
    ;;
  leak)
    DATA_NAME="brain_pnc_community_gcn_k10leak"
    CACHE="$NPZ/brain_pnc_community_k10_leak"
    EXTRA=(trainer.max_epochs=40 dataset.parameters.num_features=400)
    ;;
  shuffle)
    DATA_NAME="brain_pnc_community_gcn_k10shuf"
    CACHE="$NPZ/brain_pnc_community_k10_shuf"
    EXTRA=(trainer.max_epochs=40)
    ;;
  *)
    echo "[gcn] ERROR: unknown TBC_GCN_VARIANT='$VARIANT' (want honest|leak|shuffle)" >&2
    exit 2
    ;;
esac

echo "======================================================================"
echo "[gcn] variant        : $VARIANT"
echo "[gcn] git SHA         : $(git rev-parse HEAD 2>/dev/null || echo '?')"
echo "[gcn] node            : $(hostname)"
echo "[gcn] container        : $SIF"
echo "[gcn] data_name       : $DATA_NAME"
echo "[gcn] npz cache_dir    : $CACHE"
echo "[gcn] extra overrides : ${EXTRA[*]}"
echo "======================================================================"
nvidia-smi || true

singularity exec --nv \
  --bind "$REPO:/workspace" \
  --pwd /workspace \
  --env WANDB_MODE=offline \
  --env PYTHONPATH=/workspace \
  "$SIF" \
  python -m topobench \
    dataset=graph/brain_pnc_community \
    model=graph/gcn \
    dataset.dataloader_params.batch_size=32 \
    dataset.loader.parameters.data_name="$DATA_NAME" \
    dataset.loader.parameters.cache_dir="$CACHE" \
    "${EXTRA[@]}"

echo "[gcn] DONE variant=$VARIANT"
