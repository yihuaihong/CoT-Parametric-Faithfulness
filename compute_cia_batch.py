"""Batch CIA computation over existing labeled jsonl files.

Usage:
    python compute_cia_batch.py --glob '/scratch/.../*labeled*.jsonl' --task two_hop
"""
import argparse
import glob
import json
from pathlib import Path

from CPF_utils.metrics import (compute_cia_from_jsonl, compute_cia,
                                labels_from_two_hop)
import jsonlines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True,
                    help="glob pattern for labeled jsonl files")
    ap.add_argument("--task", required=True,
                    choices=["two_hop", "hint", "multiplication"])
    ap.add_argument("--with_footnote", action="store_true",
                    help="apply paper §3.2 footnote (two_hop only)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"No files match: {args.glob}")
        return

    print(f"\n{'='*80}\nTask: {args.task}  "
          f"Footnote: {args.with_footnote}\n{'='*80}")
    print(f"{'file':<80}  CIA   F1+    F1-    (1,1)  (1,0)  (0,1)  (0,0)   n")
    for f in files:
        if args.task == "two_hop":
            recs = list(jsonlines.open(f))
            bi, bc = labels_from_two_hop(recs,
                                         apply_footnote=args.with_footnote)
            result = compute_cia(bi, bc)
        else:
            result = compute_cia_from_jsonl(f, args.task)
        b = result.get("breakdown_pct", {})
        name = Path(f).name
        print(f"{name:<80}  {result['cia']:.3f} {result['f1_pos']:.3f} "
              f"{result['f1_neg']:.3f}  "
              f"{b.get('(1,1)', 0):5.1f}  {b.get('(1,0)', 0):5.1f}  "
              f"{b.get('(0,1)', 0):5.1f}  {b.get('(0,0)', 0):5.1f}  "
              f"{result['n']}")


if __name__ == "__main__":
    main()
