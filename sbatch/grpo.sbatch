#!/bin/bash
#SBATCH --job-name=grpo_hint
#SBATCH --account=torch_pr_520_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
mkdir -p logs

source /scratch/yh6210/miniconda3/etc/profile.d/conda.sh
conda activate open_r1
cd /home/yh6210/Research_Projects/CIA/open-r1
# Make the `open_r1` package importable without needing `pip install -e .`
export PYTHONPATH="$PWD/src:$PYTHONPATH"
# DeepSpeed strict version check vs system CUDA 13 / torch cu128 — bypass.
# CUDA 13 driver is backward-compatible with cu12 toolkit, just DS enforces.
export DS_SKIP_CUDA_CHECK=1
# Reduce allocator fragmentation on H200 (per CUDA OOM advice).
export PYTORCH_ALLOC_CONF=expandable_segments:True

: "${CONFIG:=recipes/CIA/grpo/hint.yaml}"
: "${ACC_CFG:=recipes/accelerate_configs/zero3.yaml}"

echo "=== GRPO training ==="
echo "CONFIG  = $CONFIG"
echo "ACC_CFG = $ACC_CFG"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv | head -3

# Sanity-check dataset path from YAML.
DS=$(grep -E '^dataset_name:' "$CONFIG" | head -1 | sed 's/^dataset_name:\s*//')
if [[ ! -d "$DS" ]]; then
    echo "ERROR: dataset_name path '$DS' (from $CONFIG) does not exist."
    echo "       Run prep_hint_grpo.py first."
    exit 1
fi
echo "Dataset path OK: $DS"

: "${NUM_GPUS:=2}"
echo "Launching on $NUM_GPUS GPU(s)"

accelerate launch \
    --config_file "$ACC_CFG" \
    --num_processes "$NUM_GPUS" \
    src/open_r1/grpo.py \
    --config "$CONFIG"
