#!/bin/bash
#SBATCH --job-name=inspect_comp
#SBATCH --account=torch_pr_520_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
mkdir -p logs

source /scratch/yh6210/miniconda3/etc/profile.d/conda.sh
conda activate open_r1

python - <<'PY'
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

MODEL = "/scratch/yh6210/transformers/Meta-Llama-3-8B-Instruct"
DS = "/scratch/yh6210/open-r1/datasets/Hint_MMLU_cia"

print("Loading model + tokenizer …")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="sdpa").to("cuda").eval()

ds = load_from_disk(DS)["train"]
print(f"Dataset loaded: {len(ds)} train samples")

N = 3
for i in range(N):
    row = ds[i]
    prompt_text = row["problem"]
    hint = row["hints"]
    gold = row["solution"]
    b_int = row["labels"]

    # Apply chat template same way TRL does (user-only turn).
    messages = [{"role": "user", "content": prompt_text}]
    chat = tok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt",
                 truncation=True, max_length=1024).to("cuda")

    # GRPO smoke sampling params: T=1.0, top_p=1.0, max=1024.
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tok.pad_token_id,
        )
    gen = tok.decode(out[0][inputs['input_ids'].shape[1]:],
                     skip_special_tokens=False)

    print("=" * 80)
    print(f"SAMPLE #{i+1}  |  hint_letter={hint}  gold={gold}  B_INT={b_int}")
    print("-" * 80)
    print("PROMPT (first 300 chars):")
    print(prompt_text[:300] + ("..." if len(prompt_text) > 300 else ""))
    print("-" * 80)
    print(f"COMPLETION (len={len(gen)} chars, {len(out[0])-inputs['input_ids'].shape[1]} tokens):")
    print(gen[:2000] + ("\n...[truncated]..." if len(gen) > 2000 else ""))
    print()
PY
