#!/bin/bash
# One-off environment setup + Maze2D download on a CPU node (compute nodes have internet; a job is not
# killed when the SSH session that submitted it closes, unlike background processes on the login node).
#   sbatch cluster/setup.sbatch
#SBATCH --job-name=muninnx_setup
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /home/infres/ahamdaoui-23/Muninn-X
export PATH=$HOME/.local/bin:$PATH
export VIRTUAL_ENV=$PWD/.venv
export MINARI_DATASETS_PATH=$PWD/data/minari   # keep datasets inside the project folder
mkdir -p logs data/minari
echo "job $SLURM_JOB_ID on $(hostname) | $(date)"

[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch
uv pip install -e ".[dev]"
uv pip install "minari[hdf5,hf]" gymnasium-robotics
.venv/bin/python -c "import torch, muninnx, minari, gymnasium_robotics; print('env OK | torch', torch.__version__, 'cuda', torch.version.cuda, '| minari', minari.__version__)"

for ds in D4RL/pointmaze/umaze-v2 D4RL/pointmaze/medium-v2 D4RL/pointmaze/large-v2; do
  echo "downloading $ds"
  .venv/bin/python -c "import minari; minari.download_dataset('$ds')"
done
.venv/bin/python - <<'EOF'
import minari
for ds in ["D4RL/pointmaze/umaze-v2", "D4RL/pointmaze/medium-v2", "D4RL/pointmaze/large-v2"]:
    d = minari.load_dataset(ds)
    print(f"{ds}: {d.total_episodes} episodes, {d.total_steps} steps, obs {d.observation_space}")
EOF
du -sh data/minari .venv
echo "done | $(date)"
