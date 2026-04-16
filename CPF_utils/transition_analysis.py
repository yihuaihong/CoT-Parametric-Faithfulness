"""Transition analysis for CIA improvements (paper §6.1).

Given two labeled jsonl files — one from a vanilla (pre-training) model
and one from a GRPO-trained model — classify each sample's migration in
the (B_INT, B_CoT) 4-cell grid and decompose CIA gains into three modes:

    Reasoning ↑   = B_INT changed toward matching B_CoT (internal shift)
    Reporting ↑   = B_CoT changed toward matching B_INT (CoT shift)
    Faithfulness ↓ = transition that *increases* disagreement

Produces a (16 transitions × delta_pct) table and per-mode aggregates,
mirroring paper Table 4.

Usage:
    python -m CPF_utils.transition_analysis \
        --pre  /path/to/<task>_<model>_<seed>_results_labeled.jsonl \
        --post /path/to/<task>_<model>_<seed>_results_labeled_POST.jsonl \
        --task two_hop|hint|multiplication \
        [--output   summary.json]
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Literal

import jsonlines

from CPF_utils.metrics import (
    labels_from_two_hop,
    labels_from_hint,
    labels_from_multiplication,
)


TASK_LABELERS = {
    "two_hop":        labels_from_two_hop,
    "hint":           labels_from_hint,
    "multiplication": labels_from_multiplication,
}


def _cell(bi: int, bc: int) -> str:
    return f"({bi},{bc})"


def classify_transition(pre: tuple[int, int], post: tuple[int, int]) -> str:
    """Categorize a (pre → post) transition using paper §6.1 semantics.

    Only two faithful cells exist: (1,1) and (0,0).
    A move *into* one of them = improvement; *out of* it = regression.

    Return one of:
        "Reasoning_up"     — B_INT changed to match B_CoT (faithful now)
        "Reporting_up"     — B_CoT changed to match B_INT (faithful now)
        "Faithfulness_down" — move from faithful to unfaithful
        "Faithfulness_preserved" — pre and post are the same faithful cell
        "Unfaithful_drift" — between the two unfaithful cells
    """
    pre_int, pre_cot = pre
    post_int, post_cot = post
    pre_faithful = (pre_int == pre_cot)
    post_faithful = (post_int == post_cot)

    if pre == post:
        return "Faithfulness_preserved" if post_faithful else "Unfaithful_preserved"

    if post_faithful and not pre_faithful:
        # Which side moved to cause agreement?
        int_changed = pre_int != post_int
        cot_changed = pre_cot != post_cot
        if int_changed and not cot_changed:
            return "Reasoning_up"
        if cot_changed and not int_changed:
            return "Reporting_up"
        # Both changed simultaneously — classify by larger shift; we
        # categorize as Reasoning when B_INT matches target (1,1).
        return "Reasoning_up" if post == (1, 1) else "Reporting_up"

    if pre_faithful and not post_faithful:
        return "Faithfulness_down"

    # Both unfaithful but different → just drift.
    return "Unfaithful_drift"


def analyze(pre_records: list[dict], post_records: list[dict],
            task: str, **label_kwargs) -> dict:
    if task not in TASK_LABELERS:
        raise ValueError(f"Unknown task: {task}")
    label_fn = TASK_LABELERS[task]

    if len(pre_records) != len(post_records):
        raise ValueError(
            f"pre ({len(pre_records)}) and post ({len(post_records)}) "
            f"row counts differ — records must correspond 1:1 by index"
        )

    pre_bi, pre_bc = label_fn(pre_records, **label_kwargs)
    post_bi, post_bc = label_fn(post_records, **label_kwargs)

    # 4×4 transition matrix (pre_cell → post_cell)
    transition_counts: Counter[tuple[str, str]] = Counter()
    mode_counts: Counter[str] = Counter()
    n = len(pre_records)

    for i in range(n):
        pre_cell = (int(pre_bi[i]), int(pre_bc[i]))
        post_cell = (int(post_bi[i]), int(post_bc[i]))
        key = (_cell(*pre_cell), _cell(*post_cell))
        transition_counts[key] += 1
        mode_counts[classify_transition(pre_cell, post_cell)] += 1

    # Top transitions by |delta| (same format as paper Table 4).
    # Express each transition as % of n.
    deltas = sorted(
        [(k, v / n * 100) for k, v in transition_counts.items()],
        key=lambda kv: -abs(kv[1]),
    )

    # Aggregate CIA-relevant movers (only non-self transitions).
    moved_deltas = [(k, d) for k, d in deltas if k[0] != k[1]]
    top4_movers = moved_deltas[:4]

    summary = {
        "n": n,
        "pre_breakdown_pct": _breakdown(pre_bi, pre_bc),
        "post_breakdown_pct": _breakdown(post_bi, post_bc),
        "mode_counts_pct": {k: v / n * 100 for k, v in mode_counts.items()},
        "top_transitions": [
            {"pre": k[0], "post": k[1], "delta_pct": d, "type": classify_transition(
                _parse(k[0]), _parse(k[1]))}
            for k, d in moved_deltas[:10]
        ],
        "table4_top4": [
            {"pre": k[0], "post": k[1], "delta_pct": d, "type": classify_transition(
                _parse(k[0]), _parse(k[1]))}
            for k, d in top4_movers
        ],
    }
    return summary


def _parse(cell_str: str) -> tuple[int, int]:
    # "(1,0)" -> (1, 0)
    s = cell_str.strip("()")
    a, b = s.split(",")
    return int(a), int(b)


def _breakdown(bi: list[int], bc: list[int]) -> dict:
    n = len(bi)
    counts = Counter()
    for a, b in zip(bi, bc):
        counts[_cell(a, b)] += 1
    return {k: counts[k] / n * 100 for k in ("(1,1)", "(1,0)", "(0,1)", "(0,0)")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="vanilla labeled jsonl")
    ap.add_argument("--post", required=True, help="GRPO-trained labeled jsonl")
    ap.add_argument("--task", required=True,
                    choices=["two_hop", "hint", "multiplication"])
    ap.add_argument("--output", help="write summary JSON here")
    args = ap.parse_args()

    pre = list(jsonlines.open(args.pre))
    post = list(jsonlines.open(args.post))
    summary = analyze(pre, post, args.task)

    out = json.dumps(summary, indent=2)
    print(out)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(out)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
