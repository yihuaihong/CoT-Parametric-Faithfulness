"""Prepare a GRPO-ready HuggingFace dataset for the MMLU Hint task.

Reads a jsonl produced by `CPF_utils/hint_probe.py train_and_label`:
each row has `probe_internal` (= B_INT from the trained hint probe),
`hint` (letter), original `unbiased_prompt` / `biased_prompt`,
`correct_answer`, `pred_biased`, `pred_unbiased`, etc.

We use the BIASED prompt as `problem` (GRPO rollout conditions on it),
`correct_answer` as `solution`, `probe_internal` as `labels`, and
`hint` letter as `hints` column for
`hint_parametric_faithfulness_reward`.

Usage:
    python scripts/cia/prep_hint_grpo.py \
        --labeled_jsonl /scratch/.../hint_mmlu_false_<model>_<seed>_probe_labeled.jsonl \
        --out_dir       /scratch/yh6210/open-r1/datasets/Hint_MMLU_cia
"""
from __future__ import annotations
import argparse
from pathlib import Path

import re
import jsonlines
import numpy as np
from datasets import Dataset, DatasetDict


_MC_RE = re.compile(r"<mc>\s*\(?([A-D])\)?", re.IGNORECASE)
_FB_RE = re.compile(r"(?:answer\s+is|answer:)\s*\(?([A-D])\)?", re.IGNORECASE)


def _has_extractable_letter(text: str) -> bool:
    """True iff a final option letter can be parsed from the generation,
    matching the regex used by `hint_accuracy_reward`."""
    if not text:
        return False
    return bool(_MC_RE.search(text) or _FB_RE.search(text))


HINT_PROMPT_TEMPLATE = (
    "You will be shown a multiple choice question with lettered options "
    "listed under `Choices:`.\n"
    "Please think step-by-step to explain your reasoning in "
    "<thinking></thinking> tags, and write the final option letter in "
    "the <mc></mc> tags.\n\n"
    "{question}"
)


def _prompt_to_text(prompt_field) -> str:
    """Extract the question body from the stored biased_prompt and
    re-wrap with the canonical HINT_PROMPT_TEMPLATE (paper §A). Keeps
    the GRPO rollout prompt format identical across runs (A1 strategy —
    full instructions embedded in the user message)."""
    if isinstance(prompt_field, str):
        raw = prompt_field
    else:
        raw = ""
        for m in prompt_field:
            if m.get("role") in ("human", "user"):
                raw = str(m.get("content", ""))
                break
        if not raw:
            return ""

    # Strip the existing generic instructions up to "Question:" if
    # they are present so we don't double them.
    idx = raw.find("Question:")
    body = raw[idx:] if idx >= 0 else raw
    return HINT_PROMPT_TEMPLATE.format(question=body.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_jsonl", required=True)
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--val_frac",   type=float, default=0.2)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--filter_parse_fail", action="store_true", default=True,
                    help="Drop rows whose biased_generation has no extractable "
                         "answer letter (cleans noisy B_INT/B_CoT labels). "
                         "Especially helps Qwen (~15%% extract fails).")
    ap.add_argument("--no_filter_parse_fail", dest="filter_parse_fail",
                    action="store_false")
    args = ap.parse_args()

    labeled = list(jsonlines.open(args.labeled_jsonl))
    print(f"Loaded {len(labeled)} labeled records")

    # Keep only single-turn samples (hint sits inside the user message).
    # For posthoc_False, the hint is the model's prior assistant answer;
    # we can't cleanly pack that into GRPO's single-prompt chat template
    # without either distorting the dialogue or losing the hint. Simplest
    # and faithful: drop those rows.
    single_turn = [r for r in labeled
                   if isinstance(r.get("biased_prompt"), list)
                   and len(r["biased_prompt"]) == 1]
    skipped = len(labeled) - len(single_turn)
    print(f"Kept {len(single_turn)} single-turn records "
          f"(skipped {skipped} multi-turn / posthoc)")

    # Optionally drop rows whose biased_generation didn't reach a parsable
    # answer letter — those rows have unreliable B_INT (probe trained on
    # truncated state) and B_CoT (Gemini ack uses incomplete CoT).
    if args.filter_parse_fail:
        before = len(single_turn)
        single_turn = [r for r in single_turn
                       if _has_extractable_letter(r.get("biased_generation", ""))]
        print(f"Filtered out {before - len(single_turn)} rows with parse-fail "
              f"biased_generation; {len(single_turn)} remaining")

    records = []
    for i, r in enumerate(single_turn):
        problem = _prompt_to_text(r["biased_prompt"])
        records.append({
            "index":    i,
            "problem":  problem,
            "solution": str(r.get("correct_answer", "")).upper().strip(),
            "labels":   int(bool(r.get("probe_internal", 0))),
            # Gemini-labeled ground truth for B_CoT — prefer over regex
            # heuristics at training time (reward fn can read this).
            "cot_ack_label": int(bool(r.get("acknowledge_hint_ai", False))),
            "hints":    str(r.get("hint", "")).upper().strip(),
        })

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(records)); rng.shuffle(idx)
    n = len(idx)
    n_tr = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    splits = {
        "train":      idx[:n_tr],
        "validation": idx[n_tr:n_tr + n_val],
        "test":       idx[n_tr + n_val:],
    }

    ds = DatasetDict({k: Dataset.from_list([records[i] for i in v])
                      for k, v in splits.items()})
    pos = np.mean([r["labels"] for r in records])
    print(f"Split sizes: {[len(ds[k]) for k in ds]}  "
          f"B_INT=1 rate: {pos:.3f}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(args.out_dir)
    print(f"Saved DatasetDict to {args.out_dir}")


if __name__ == "__main__":
    main()
