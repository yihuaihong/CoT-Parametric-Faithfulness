"""Prepare a GRPO-ready HuggingFace dataset for 2-digit multiplication.

B_INT is computed ONLINE inside
`multiplication_parametric_faithfulness_reward` from the completion
itself (pp1 + pp2 == final), so we do NOT need to write a `labels`
column. We just wrap the raw multiplication problems into
`{problem, solution}` HF format.

Source: the jsonl that was produced by the original
`run_multiplication_acc_evaluation`, which has fields
`prompt` (e.g. "37 × 84 = ?") and `correct_answer` (integer string).

Usage:
    python scripts/cia/prep_multiplication_grpo.py \
        --source_jsonl /scratch/.../2-digit-Multiplication_<model>_use-cot-is-True_<seed>_results.jsonl \
        --out_dir      /scratch/yh6210/open-r1/datasets/Mult2d_cia
"""
from __future__ import annotations
import argparse
from pathlib import Path

import jsonlines
import numpy as np
from datasets import Dataset, DatasetDict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_jsonl", required=True,
                    help="existing multiplication generation jsonl")
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--val_frac",   type=float, default=0.2)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    src = list(jsonlines.open(args.source_jsonl))
    print(f"Loaded {len(src)} multiplication records")

    records = []
    for i, r in enumerate(src):
        problem = str(r.get("prompt", "")).strip()
        answer = str(r.get("correct_answer", "")).strip()
        if not problem or not answer:
            continue
        records.append({
            "index":    i,
            "problem":  problem,
            "solution": answer,
            # Placeholder column (kept for schema consistency with other
            # tasks; multiplication reward ignores `labels`).
            "labels":   0,
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
    print(f"Split sizes: {[len(ds[k]) for k in ds]}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(args.out_dir)
    print(f"Saved DatasetDict to {args.out_dir}")


if __name__ == "__main__":
    main()
