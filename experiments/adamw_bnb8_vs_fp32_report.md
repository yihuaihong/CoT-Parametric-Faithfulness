# adamw_bnb_8bit vs adamw_torch_fused — GRPO optimizer comparison

**Goal:** empirically measure the training-quality impact of `adamw_bnb_8bit` (bitsandbytes 8-bit AdamW) vs `adamw_torch_fused` (fp32 AdamW) in a GRPO fine-tuning setting, since published validation (Dettmers et al. 2022) is on SFT/language-modeling, not GRPO.

## Motivation

`adamw_bnb_8bit` compresses Adam's momentum + variance from fp32 → block-wise int8, saving ~24-28 GB for an 8B model. Needed for 2× H100 ZeRO-3 full FT.

**Known unknowns for GRPO specifically:**
- GRPO has noisy advantage → optimizer quantization noise could compound
- No published head-to-head (as of 2025-10).
- H100/H200 sm_90 had a bnb kernel config bug earlier in this env; fresh bnb 0.49.2 may or may not have fixed it.

## Setup

Same everything except `optim:` field.

| field | value |
|---|---|
| model | Meta-Llama-3-8B-Instruct |
| task | Hint MMLU (Llama probe labels, A1 prompt) |
| hardware | 2× H200 + ZeRO-2 |
| steps | 100 |
| `num_generations` | 8 |
| `per_device_train_batch_size` | 1, grad_accum=2 |
| `max_prompt_length` | 1024 |
| `max_completion_length` | 1024 |
| `temperature` | 1.0 |
| `learning_rate` | 1e-6 (cosine_with_min_lr) |
| `beta` | 0.01 (KL) |
| `epsilon` | 0.2 (clip) |
| `seed` | 42 |

## Runs

| Config | YAML | sbatch job | wandb run |
|---|---|---|---|
| A. fp32 AdamW | `recipes/CIA/grpo/hint_compare_fp32.yaml` | **6340072** | compare_fp32_100steps |
| B. bnb 8bit | `recipes/CIA/grpo/hint_compare_bnb8.yaml` | **6340073** | compare_bnb8_100steps |

## Metrics to compare

- `train/loss` curve (should match if bnb has no impact)
- `train/reward` final value + trajectory
- `train/reward_std`
- `train/grad_norm`
- `train/kl`
- `train/completion_length`
- per-step wall time (speed hit from bnb)
- peak GPU memory (memory savings from bnb)

## Results

_[to be filled after jobs complete]_

### Speed (wall-clock)
| Config | avg step time | 100-step total | delta vs fp32 |
|---|---|---|---|
| fp32 | | | — |
| bnb 8bit | | | |

### Memory (peak per GPU)
| Config | peak MA | peak reserved |
|---|---|---|
| fp32 | | |
| bnb 8bit | | |

### Loss / Reward at step 100
| Config | loss | reward | reward_std | grad_norm | kl |
|---|---|---|---|---|---|
| fp32 | | | | | |
| bnb 8bit | | | | | |

### Divergence over time
- Max absolute diff in loss between A and B across 100 steps:
- Pearson correlation of loss curves:
- Pearson correlation of reward curves:

### Failure modes observed
_[any OOMs, NaNs, kernel errors, instability]_

## Conclusion

_[to be filled — is 8bit safe for full GRPO training? recommended or not?]_

## Notes / fallback optimizers

If `adamw_bnb_8bit` crashes (e.g., sm_90 kernel config bug earlier on
H200), try these in order — quality is **identical** (same 8-bit
block-wise quantization), only memory-management differs:

1. `paged_adamw_8bit` — same bnb quant, but CUDA Unified Memory pages
   state to CPU under pressure. Avoids OOM crashes. Slightly slower
   when memory is comfortable (UM bookkeeping + page-fault cost during
   warm-up). Recommended when near OOM.
2. `paged_adamw_32bit` — fp32 state but paged. No memory savings from
   quantization; fallback only when 8-bit quant itself crashes.
3. DeepSpeed ZeRO-3 + `offload_optimizer_device: cpu` — strongest
   memory relief but needs DS to compile CUDA op (we hit a mismatch
   earlier; requires `DS_SKIP_CUDA_CHECK=1`).

## References

- Dettmers et al. 2022 "8-bit Optimizers via Block-wise Quantization" (ICLR 2022)
  https://arxiv.org/abs/2110.02861
- bitsandbytes GitHub: https://github.com/TimDettmers/bitsandbytes
- HF TRL docs: `optim` field in `GRPOConfig`
