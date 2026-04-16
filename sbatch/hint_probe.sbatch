#!/bin/bash
#SBATCH --job-name=hint_probe
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
mkdir -p logs

source /scratch/yh6210/miniconda3/etc/profile.d/conda.sh
conda activate open_r1
cd /home/yh6210/Research_Projects/CIA

: "${MODEL:=Meta-Llama-3-8B-Instruct}"
: "${SEED:=8888}"
: "${TAU:=0.1}"

GEN="/scratch/yh6210/results/open-r1/hint_mmlu_results/hint_mmlu_false_${MODEL}_results_with_ai_label.jsonl"
if [[ ! -f "$GEN" ]]; then
    # fallback to seeded variant
    GEN="/scratch/yh6210/results/open-r1/hint_mmlu_results/hint_mmlu_false_${MODEL}_${SEED}_results.jsonl"
fi
echo "Input : $GEN"

OUT="/scratch/yh6210/results/open-r1/hint_mmlu_results/hint_mmlu_false_${MODEL}_${SEED}_probe_labeled.jsonl"

python -m CPF_utils.hint_probe train_and_label \
    --gen_jsonl "$GEN" \
    --model_name "$MODEL" \
    --seed "$SEED" --tau "$TAU" \
    --output "$OUT"

echo "--- CIA on probe-labeled ---"
python compute_cia_batch.py --glob "$OUT" --task hint
