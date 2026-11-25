#!/bin/bash
# Generic Muninn-X job for the SLURM cluster (ssh gpu-gw). Everything lives under
# /home/infres/ahamdaoui-23/Muninn-X; never use sudo.
#
# Usage (from /home/infres/ahamdaoui-23/Muninn-X):
#   sbatch --job-name=e8_unet cluster/job.sbatch scripts/certified_compare.py --ckpt runs/traj_toy/unet_b16k/model.pt
#   sbatch --time=24:00:00 --job-name=train_maze cluster/job.sbatch scripts/train_xxx.py --arg value
# Logs: logs/<job-name>_<job-id>.out / .err
#
#SBATCH --job-name=muninnx
#SBATCH --partition=3090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /home/infres/ahamdaoui-23/Muninn-X
mkdir -p logs

echo "job $SLURM_JOB_ID on $(hostname) | $(date)"
echo "git: $(git log --oneline -1)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "cmd: python -u $*"
echo "------------------------------------------------------------"

# The memory cap in muninnx.utils.misc.get_device (90% of VRAM) applies here too.
.venv/bin/python -u "$@"

echo "------------------------------------------------------------"
echo "done | $(date)"
