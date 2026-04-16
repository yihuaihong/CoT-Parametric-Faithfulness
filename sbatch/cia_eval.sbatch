#!/bin/bash
#SBATCH --job-name=cia_eval
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
mkdir -p logs

source /scratch/yh6210/miniconda3/etc/profile.d/conda.sh
conda activate open_r1

cd /home/yh6210/Research_Projects/CIA

# --- arguments (override via: sbatch --export=ALL,MODEL=...,DATASET=... ...) ---
: "${MODEL:=gemma-2-9b-it}"
: "${DATASET:=TwoHopFact}"
: "${SEED:=8888}"
: "${BATCH:=32}"

echo "MODEL=$MODEL DATASET=$DATASET SEED=$SEED BATCH=$BATCH"

# 1) Inference (only if the generation jsonl is missing)
GEN_FILE="/scratch/yh6210/results/open-r1/twohop_results/${DATASET}_${MODEL}_${SEED}_results.jsonl"
if [[ "$DATASET" == "Hint_MMLU" ]]; then
    GEN_FILE="/scratch/yh6210/results/open-r1/hint_mmlu_results/hint_mmlu_false_${MODEL}_${SEED}_results.jsonl"
elif [[ "$DATASET" == *Multiplication* ]]; then
    GEN_FILE="/scratch/yh6210/results/open-r1/math_results/${DATASET}_${MODEL}_use-cot-is-True_${SEED}_results.jsonl"
fi

if [[ ! -f "$GEN_FILE" ]]; then
    echo "Running CoT inference -> $GEN_FILE"
    python cpf_evaluation.py --eval_acc --use_cot_prompt \
        --model_name "$MODEL" --dataset_name "$DATASET" \
        --batch_size "$BATCH" --seed "$SEED"
else
    echo "Generation exists: $GEN_FILE"
fi

# 2) CPF (logit lens + probe) — produces labeled jsonl
python cpf_evaluation.py --eval_cpf --use_cot_prompt \
    --model_name "$MODEL" --dataset_name "$DATASET" \
    --batch_size "$BATCH" --seed "$SEED"

# 3) (Multiplication only) partial-product corruption labels
if [[ "$DATASET" == *Multiplication* ]]; then
    OUT="/scratch/yh6210/results/open-r1/math_results/${DATASET}_${MODEL}_${SEED}_corruption_labeled.jsonl"
    python -m CPF_utils.multiplication_corruption \
        --input "$GEN_FILE" --output "$OUT" \
        --model_name "$MODEL"
fi

# 4) CIA aggregation
LABELED_GLOB="/scratch/yh6210/results/open-r1/twohop_results/${DATASET}_${MODEL}_*_labeled.jsonl"
TASK=two_hop
if [[ "$DATASET" == "Hint_MMLU" ]]; then
    LABELED_GLOB="/scratch/yh6210/results/open-r1/hint_mmlu_results/*${MODEL}*with_ai_label.jsonl"
    TASK=hint
elif [[ "$DATASET" == *Multiplication* ]]; then
    LABELED_GLOB="/scratch/yh6210/results/open-r1/math_results/${DATASET}_${MODEL}_*corruption_labeled.jsonl"
    TASK=multiplication
fi

echo "Computing CIA on: $LABELED_GLOB"
python compute_cia_batch.py --glob "$LABELED_GLOB" --task "$TASK"
