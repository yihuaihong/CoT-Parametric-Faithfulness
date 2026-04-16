"""Prepare a GRPO-ready HuggingFace dataset for the TwoHopFact task.

Reads:
  * The raw TwoHopFact dataset (via CPF_utils.data_utils.load_dataset).
  * A labeled jsonl produced by CPF_utils/logitlens_utils.py (dual-position
    probe). Each row has `correct_bridge`, `correct_in_topk`, etc.

Writes:
  A HuggingFace DatasetDict saved to `--out_dir` with columns:
    problem          - the two-hop question (prompt for GRPO)
    solution         - final answer (e3.value) for `accuracy_reward`
    labels           - B_INT ∈ {0,1} from dual-position logit-lens union
                        (`correct_in_topk`)
    bridge_entities  - annotated bridge entity (for the existing
                        `two_hop_parametric_faithfulness` reward to read)
    index            - original row index

Usage:
    python scripts/cia/prep_two_hop_grpo.py \
        --labeled_jsonl /scratch/.../TwoHopFact_<model>_logit_lens_<seed>_results_labeled.jsonl \
        --out_dir      /scratch/yh6210/open-r1/datasets/TwoHopFact_cia \
        --train_frac   0.6 --val_frac 0.2
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Let us import CPF utilities.
CPF_PATH = "/home/yh6210/Research_Projects/CIA"
if CPF_PATH not in sys.path:
    sys.path.insert(0, CPF_PATH)

import jsonlines
import numpy as np
from datasets import Dataset, DatasetDict

from CPF_utils.data_utils import load_dataset as load_twohop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_jsonl", required=True,
                    help="logit-lens labeled jsonl (has correct_in_topk, "
                         "correct_bridge, pred_bridge, etc.)")
    ap.add_argument("--dataset_name", default="TwoHopFact")
    ap.add_argument("--dataset_dir", default="/scratch/yh6210/datasets")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load raw TwoHopFact as a pandas DataFrame; need 'r2(r1(e1)).prompt' +
    # 'e2.value' (bridge) + 'e3.value' (final answer).
    df = load_twohop(dataset_name=args.dataset_name,
                     dataset_dir=args.dataset_dir,
                     sample_num=0, seed=args.seed)
    print(f"Raw dataset rows: {len(df)}")

    labeled = list(jsonlines.open(args.labeled_jsonl))
    print(f"Labeled rows:     {len(labeled)}")
    if len(labeled) != len(df):
        # Trim dataset to match labeled length (labeled was computed on a
        # subset). Per TwoHopFact split convention in this repo, labeled
        # order matches dataset order.
        df = df.iloc[:len(labeled)].reset_index(drop=True)

    # Build records.
    records = []
    for i, (row, lbl) in enumerate(zip(df.itertuples(index=False), labeled)):
        question = getattr(row, "r2_r1_e1__prompt", None) or \
                   df["r2(r1(e1)).prompt"].iloc[i]
        bridge = df["e2.value"].iloc[i]
        answer = df["e3.value"].iloc[i]
        b_int = 1 if lbl.get("correct_in_topk") else 0
        records.append({
            "index": i,
            "problem": str(question),
            "solution": str(answer),
            "labels": int(b_int),
            "bridge_entities": str(bridge),
        })

    # 6:2:2 split.
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n = len(idx)
    n_tr = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    tr_idx = idx[:n_tr]
    val_idx = idx[n_tr:n_tr + n_val]
    te_idx = idx[n_tr + n_val:]

    def subset(sel):
        return Dataset.from_list([records[i] for i in sel])

    ds = DatasetDict({
        "train": subset(tr_idx),
        "validation": subset(val_idx),
        "test": subset(te_idx),
    })
    pos_rate = np.mean([r["labels"] for r in records])
    print(f"Split sizes: {[len(ds[k]) for k in ds]}  "
          f"B_INT=1 rate: {pos_rate:.3f}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(args.out_dir)
    print(f"Saved DatasetDict to {args.out_dir}")


if __name__ == "__main__":
    main()
