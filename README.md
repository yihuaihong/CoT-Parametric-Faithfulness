# CoT-Parametric Faithfulness (CIA)

**Making LLMs Say What They Think: Measuring and Improving CoT-Interpretability Alignment**

This repository implements the CoT-Interpretability Alignment (CIA) framework for measuring and improving the alignment between a model's verbalized Chain-of-Thought and its internal reasoning strategies.

## Repository Structure

```
CIA/
├── CPF_utils/                          # Core library
│   ├── metrics.py                      # CIA macro-F1 metric computation
│   ├── layer_config.py                 # Per-model/task probe layer config (Table 7)
│   ├── logitlens_utils.py              # Logit Lens evaluation (§3, TwoHop)
│   ├── probing_utils.py                # Linear probe training & evaluation
│   ├── hint_probe.py                   # Hint probe (§C.2): P(hint) shift + probe
│   ├── mult_probe.py                   # Multiplication probe (§C.3): corruption labels
│   ├── multiplication_corruption.py    # Partial-product corruption labeling
│   ├── transition_analysis.py          # §6.1: Decompose CIA gains into Reasoning/Reporting
│   ├── evaluation_utils.py             # Task-specific eval (TwoHop, Hint, Mult)
│   ├── data_utils.py                   # Dataset loading
│   ├── tokenization_utils.py           # Token position finding
│   └── model_utils.py                  # Model loading helpers
│
├── open-r1/                            # GRPO training backbone (fork of HuggingFace open-r1)
│   ├── src/open_r1/
│   │   ├── grpo.py                     # GRPO entry point (with eval() patch + EOS fix)
│   │   ├── rewards.py                  # Reward registry (6 CIA rewards registered)
│   │   └── utils/
│   │       ├── completion_logger.py    # wandb completion table callback
│   │       ├── model_utils.py          # Tokenizer/model loading (pad_token fix)
│   │       └── data.py                 # Dataset loading (load_from_disk support)
│   ├── recipes/CIA/grpo/               # GRPO training configs
│   │   ├── hint.yaml                   # Full Hint training
│   │   ├── two_hop.yaml                # Full TwoHop training
│   │   ├── multiplication.yaml         # Full Multiplication training
│   │   └── hint_smoke*.yaml            # Smoke test variants
│   ├── recipes/accelerate_configs/     # DeepSpeed ZeRO configs
│   └── scripts/cia/                    # Dataset prep scripts
│       ├── prep_hint_grpo.py
│       ├── prep_two_hop_grpo.py
│       └── prep_multiplication_grpo.py
│
├── sbatch/                             # SLURM job scripts
│   ├── grpo.sbatch                     # Generic GRPO training launcher
│   ├── cia_eval.sbatch                 # CIA evaluation pipeline
│   ├── hint_probe.sbatch              # Hint probe training
│   ├── mult_corruption_smoke.sbatch   # Corruption labeling
│   ├── mult_inference_newprompt.sbatch # Mult inference with A/B prompt
│   └── inspect_completions.sbatch     # Sample model outputs
│
├── experiments/                        # Experiment reports
│   └── adamw_bnb8_vs_fp32_report.md
│
├── compute_cia_batch.py                # Batch CIA computation over labeled jsonls
├── cpf_evaluation.py                   # Main evaluation entry point
└── Paper/                              # Paper PDF
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n cia python=3.11 -y
conda activate cia
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate datasets trl==0.14.0 deepspeed==0.15.4
pip install bitsandbytes wandb jsonlines
```

### 2. CIA Evaluation (Measuring Faithfulness)

#### Compute CIA on existing labeled results:
```bash
python compute_cia_batch.py \
    --glob '/path/to/labeled_results/*.jsonl' \
    --task two_hop|hint|multiplication
```

#### Train a Hint probe (§C.2):
```bash
python -m CPF_utils.hint_probe train_and_label \
    --gen_jsonl /path/to/hint_mmlu_false_<model>_results_with_ai_label.jsonl \
    --model_name Meta-Llama-3-8B-Instruct \
    --seed 8888
```

#### Run multiplication corruption labeling (§C.3):
```bash
python -m CPF_utils.multiplication_corruption \
    --input /path/to/mult_results.jsonl \
    --output /path/to/mult_corruption_labeled.jsonl \
    --model_name Meta-Llama-3-8B-Instruct
```

### 3. Post-Training with GRPO (§5)

#### Prepare dataset:
```bash
python open-r1/scripts/cia/prep_hint_grpo.py \
    --labeled_jsonl /path/to/hint_probe_labeled.jsonl \
    --out_dir /path/to/datasets/Hint_MMLU_cia
```

#### Launch GRPO training:
```bash
sbatch --job-name=grpo_hint \
    --export=ALL,CONFIG=recipes/CIA/grpo/hint.yaml \
    sbatch/grpo.sbatch
```

#### Transition analysis (§6.1):
```bash
python -m CPF_utils.transition_analysis \
    --pre /path/to/pre_training_labeled.jsonl \
    --post /path/to/post_training_labeled.jsonl \
    --task hint
```

## Models

Evaluated on three 8B-class instruction-tuned LLMs:
- **Llama-3-8B-Instruct** (Meta)
- **Gemma-2-9B-IT** (Google)
- **Qwen3-8B** (Alibaba)

## Tasks

| Task | Strategy S | B_INT source | B_CoT source |
|------|-----------|-------------|-------------|
| **Two-Hop Factual Reasoning** | Using annotated bridge entity | Linear Probe / Logit Lens | CoT bridge entity extraction |
| **Hint Interventions (MMLU)** | Relying on injected hint | Linear Probe (prob-shift τ=0.1) | Regex acknowledgment detection |
| **2-Digit Multiplication** | Following step-by-step long mult | Parser proxy (pp1+pp2==final) or Probe | APPROACH: A/B declaration |

## Reward Function (§5.1)

```
r(y_i) = r_base(y_i) + λ · 𝟙(B_CoT(y_i) = B_INT(y_i))
```

- `r_base`: task-specific accuracy (letter match / text match / integer match)
- `λ = 1.0`: faithfulness reward weight
- `B_CoT`: strategy verbalized in CoT (online, per rollout)
- `B_INT`: internal strategy from probe (offline, per prompt)

## Key Findings

1. Current LLMs exhibit low CIA scores (0.316–0.554) across all tasks
2. GRPO post-training can substantially improve CIA while maintaining accuracy
3. Improvements arise through two distinct modes:
   - **Reasoning ↑**: model changes *how it reasons* (e.g., multiplication)
   - **Reporting ↑**: model changes *how it reports* (e.g., hint, two-hop)

## Critical Implementation Notes

- **TRL gradient_checkpointing bug**: `model.train()` + `gradient_checkpointing` + `use_cache=False` produces garbled generation. Our `grpo.py` monkey-patches `unwrap_model_for_generation` to toggle `model.eval()` during rollout.
- **Llama-3 tokenizer**: `unk_token_id=None` — use `_safe_fallback_tid()` helper.
- **TRL EOS**: `GenerationConfig` missing `eos_token_id` — patched to include `[128001, 128009]`.
- **Probing**: use `padding_side="right"` for hidden-state extraction to avoid bf16+left-pad NaN.

## Citation

```bibtex
@article{hong2025cia,
  title={Making LLMs Say What They Think: Measuring and Improving CoT-Interpretability Alignment},
  author={Hong, Yihuai and others},
  year={2025}
}
```
