#!/bin/bash
#SBATCH --job-name=install_flash_attn
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
mkdir -p logs

source /scratch/yh6210/miniconda3/etc/profile.d/conda.sh
conda activate open_r1

# Locate CUDA on this GPU node and export CUDA_HOME for nvcc.
NVCC_PATH=$(which nvcc 2>/dev/null || true)
if [[ -z "$NVCC_PATH" ]]; then
    # Common GPU-node install locations on NYU HPC.
    for cand in /usr/local/cuda /usr/local/cuda-12.* /opt/cuda /share/apps/cuda/12.* ; do
        if [[ -x "$cand/bin/nvcc" ]]; then
            export CUDA_HOME="$cand"
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
            break
        fi
    done
else
    # Strip /bin/nvcc → CUDA_HOME
    export CUDA_HOME="$(dirname $(dirname $NVCC_PATH))"
fi

if [[ -z "$CUDA_HOME" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "ERROR: CUDA_HOME not found / nvcc missing"
    exit 1
fi

echo "=== Build environment ==="
echo "CUDA_HOME = $CUDA_HOME"
nvcc --version | tail -2
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=name,driver_version --format=csv | head -2

echo ""
echo "=== Building flash-attn (~5-15 min) ==="
# MAX_JOBS limits parallel ninja workers (high RAM cost otherwise).
MAX_JOBS=4 pip install flash-attn --no-build-isolation -v 2>&1 | tail -50

echo ""
echo "=== Verification ==="
python -c "import flash_attn; print('✅ flash-attn', flash_attn.__version__)"
