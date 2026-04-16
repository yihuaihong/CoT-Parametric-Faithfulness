#!/bin/bash
#SBATCH --job-name=mult_newprompt
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
: "${BATCH:=32}"

echo "=== Multiplication inference with new A/B prompt ==="
echo "MODEL=$MODEL SEED=$SEED BATCH=$BATCH"

python cpf_evaluation.py --eval_acc --use_cot_prompt \
    --model_name "$MODEL" --dataset_name 2-digit-Multiplication \
    --batch_size "$BATCH" --seed "$SEED"

# Quick A/B ratio summary.
echo "=== A/B distribution ==="
python - <<'PY'
import jsonlines, re, glob, os
from collections import Counter
A = re.compile(r'APPROACH:\s*([AB])', re.IGNORECASE)
PP = re.compile(r'^\s*(\d+)\s*\(\s*(\d+)\s*[×x*]\s*(\d+)\s*\)', re.MULTILINE)
M = os.environ.get('MODEL', 'Meta-Llama-3-8B-Instruct')
S = os.environ.get('SEED', '8888')
pattern = f'/scratch/yh6210/results/open-r1/math_results/2-digit-Multiplication_{M}_use-cot-is-True_{S}_results.jsonl'
path = sorted(glob.glob(pattern), key=os.path.getmtime)[-1]
recs = list(jsonlines.open(path))
c = Counter()
for r in recs:
    g = r['full_generation']
    m = A.search(g); pps = PP.findall(g)
    if m and m.group(1).upper() == 'A':  c['A_declared'] += 1
    elif m and m.group(1).upper() == 'B': c['B_declared'] += 1
    elif len(pps) >= 2: c['B_inferred'] += 1
    else: c['parse_fail'] += 1
print(f'file: {path}')
print(f'n={len(recs)}  {dict(c)}')
PY
