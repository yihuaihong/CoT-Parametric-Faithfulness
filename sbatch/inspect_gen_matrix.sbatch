#!/bin/bash
#SBATCH --job-name=gen_matrix
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
mkdir -p logs
source /scratch/yh6210/miniconda3/etc/profile.d/conda.sh
conda activate open_r1

python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

MODEL = "/scratch/yh6210/transformers/Meta-Llama-3-8B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"   # match TRL behavior

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="sdpa", use_cache=False).to("cuda")

ds = load_from_disk("/scratch/yh6210/open-r1/datasets/Hint_MMLU_cia")["train"]
prompt_text = ds[0]["problem"]
messages = [{"role": "user", "content": prompt_text}]
chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(chat, return_tensors="pt").to("cuda")

# EOS ids match our grpo.py patch
eos_ids = [128001, 128009]

def try_gen(name, train_mode, grad_ckpt, use_cache):
    model.config.use_cache = use_cache
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    if train_mode:
        model.train()
    else:
        model.eval()
    torch.manual_seed(42)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos_ids,
        )
    text = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    n_tok = len(out[0]) - inputs['input_ids'].shape[1]
    print("="*80)
    print(f"{name}  train={train_mode} grad_ckpt={grad_ckpt} use_cache={use_cache}")
    print(f"  tokens generated: {n_tok}")
    print(f"  tail 200 chars: {text[-200:]!r}")
    print()

# 4 combinations
try_gen("[ctrl]   eval + no-ckpt + cache ",   False, False, True)
try_gen("[A] train + no-ckpt + cache     ",   True,  False, True)
try_gen("[B] train + ckpt    + no-cache  ",   True,  True,  False)
try_gen("[C] eval  + ckpt    + no-cache  ",   False, True,  False)
try_gen("[D] train + no-ckpt + no-cache  ",   True,  False, False)
PY
