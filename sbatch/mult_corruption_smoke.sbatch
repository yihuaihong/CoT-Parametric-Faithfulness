#!/bin/bash
#SBATCH --job-name=mult_corrupt_smoke
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
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
: "${LIMIT:=50}"

IN="/scratch/yh6210/results/open-r1/math_results/2-digit-Multiplication_${MODEL}_use-cot-is-True_${SEED}_results.jsonl"
OUT="/scratch/yh6210/results/open-r1/math_results/2-digit-Multiplication_${MODEL}_${SEED}_corruption_labeled_smoke${LIMIT}.jsonl"

echo "Input : $IN"
echo "Output: $OUT"
echo "Limit : $LIMIT"

python -m CPF_utils.multiplication_corruption \
    --input  "$IN" \
    --output "$OUT" \
    --model_name "$MODEL" \
    --limit  "$LIMIT"

echo "--- summary ---"
python - <<'PY'
import jsonlines, os, sys, collections
path = os.environ['OUT'] if 'OUT' in os.environ else None
import glob
# Fallback: find newest smoke jsonl.
cand = sorted(glob.glob("/scratch/yh6210/results/open-r1/math_results/*corruption_labeled_smoke*.jsonl"),
              key=os.path.getmtime)[-1]
print("inspecting:", cand)
recs = list(jsonlines.open(cand))
n = len(recs)
n_b = sum(1 for r in recs if r.get("approach") == "B")
n_track = sum(1 for r in recs if r.get("follows_partial_products"))
n_parse_ok = sum(1 for r in recs if r.get("corruption_details"))
print(f"n={n}  Approach-B={n_b}  parse_ok={n_parse_ok}  "
      f"follows_partial_products={n_track}")
# Show a few details.
for r in recs[:3]:
    print("---")
    print("approach:", r.get("approach"))
    print("follows :", r.get("follows_partial_products"))
    print("details :", r.get("corruption_details"))
PY
