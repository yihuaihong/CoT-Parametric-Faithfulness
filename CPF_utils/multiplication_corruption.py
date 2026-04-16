"""Partial-product corruption labeling for 2-digit multiplication (paper §C.3).

Behavioral test to obtain B_INT for the multiplication task:

  For each sample where the model selected Approach B (long multiplication),
  we corrupt one of the two partial products inside the CoT and let the
  model continue generating. If the new summation tracks the corrupted
  partial products (= genuine_follow), B_INT = 1 (genuine step-by-step).
  If the summation stays at the original value (= parametric recall),
  B_INT = 0.

Usage (offline, once per model/seed):
    python -m CPF_utils.multiplication_corruption \
        --input  /path/to/2-digit-Multiplication_<model>_..._results.jsonl \
        --output /path/to/..._results_labeled.jsonl \
        --model_dir /scratch/yh6210/transformers --model_name <name>

The script reads an existing generation jsonl (from
`run_multiplication_acc_evaluation`), runs two corruption rollouts per
Approach-B sample, and writes a new jsonl with these extra fields:
    approach                 - 'A' or 'B' (parsed from CoT)
    original_pp1, pp2, sum_  - parsed partial products / summation
    corruption_sum_pp1       - summation produced after corrupting PP1
    corruption_sum_pp2       - summation produced after corrupting PP2
    follows_partial_products - bool: True iff either corruption produced
                               a summation that matches (corrupted_pp + other)
                               rather than the original (correct) sum
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path

import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- CoT parsers ----------

APPROACH_RE = re.compile(r"APPROACH:\s*([AB])", re.IGNORECASE)
PP_RE = re.compile(
    r"^\s*(\d+)\s*\(\s*(\d+)\s*[×x*]\s*(\d+)\s*\)", re.MULTILINE)


def parse_cot(full_generation: str) -> dict | None:
    """Extract approach and partial products from a CoT response.

    Approach is inferred from:
      1) Explicit 'APPROACH: B' string (paper prompt), else
      2) presence of two partial-product lines (this repo's prompt
         which always asks for long mult without an A/B choice).
    """
    m = APPROACH_RE.search(full_generation)
    pps = PP_RE.findall(full_generation)

    if m and m.group(1).upper() == "A":
        return {"approach": "A"}
    if len(pps) < 2:
        # Not enough partial products to test corruption.
        approach = ("B" if (m and m.group(1).upper() == "B")
                    else None)
        return {"approach": approach}

    pp1_val, pp1_a, pp1_b = int(pps[0][0]), int(pps[0][1]), int(pps[0][2])
    pp2_val, pp2_a, pp2_b = int(pps[1][0]), int(pps[1][1]), int(pps[1][2])

    # Final answer.
    ans_m = re.search(r"FINAL ANSWER:?\s*(\d+)", full_generation,
                      re.IGNORECASE)
    final_ans = int(ans_m.group(1)) if ans_m else None

    return {
        "approach": "B",
        "pp1": pp1_val, "pp1_factors": (pp1_a, pp1_b),
        "pp2": pp2_val, "pp2_factors": (pp2_a, pp2_b),
        "final": final_ans,
    }


def locate_pp_char_spans(full_generation: str) -> list[tuple[int, int, int]]:
    """Return list of (char_start, char_end, value) for each partial product
    line that matches PP_RE. Used to build a corrupted prefix."""
    spans = []
    for m in PP_RE.finditer(full_generation):
        val = int(m.group(1))
        # char span of the numeric value only (group 1)
        val_start = m.start(1)
        val_end = m.end(1)
        spans.append((val_start, val_end, val))
    return spans


# ---------- corruption rollout ----------

def build_corrupted_prefix(full_generation: str, pp_index: int,
                           corrupted_value: int) -> str | None:
    """Replace the value of the pp_index-th partial product with
    corrupted_value, keep the rest of generation up to (but not including)
    the summation line (step 3 or 4). Returns None if we can't locate it."""
    spans = locate_pp_char_spans(full_generation)
    if pp_index >= len(spans):
        return None
    start, end, _ = spans[pp_index]
    # Corrupt PP value in-place.
    corrupted = (full_generation[:start] + str(corrupted_value)
                 + full_generation[end:])

    # Truncate before the summation appears. We cut at the FIRST line that
    # contains a plain number of the right magnitude after the two PPs.
    # Simplest: cut at the second "------" line AFTER the two PPs, keep
    # everything up to and including it; the model continues from there.
    after_pps_spans = locate_pp_char_spans(corrupted)
    if len(after_pps_spans) < 2:
        return None
    # Keep text up to end of second PP line.
    second_pp_end = after_pps_spans[1][1]
    # Extend to end of that line.
    nl = corrupted.find("\n", second_pp_end)
    if nl == -1:
        return corrupted
    # Also include the "3." header and the two re-listed PPs + separator
    # so the model is primed to produce a summation. Simplest: keep
    # through to the next line that is just "------" after we've seen
    # step-3 marker.
    rest = corrupted[nl:]
    # Cut at "FINAL ANSWER:" or the line before the summation number.
    # We actually want to STOP before the final summation digit. A robust
    # heuristic: cut just after the last "------" before "FINAL ANSWER:".
    fa_idx = rest.find("FINAL ANSWER")
    if fa_idx == -1:
        fa_idx = len(rest)
    # Find last "------" before fa_idx.
    head = rest[:fa_idx]
    last_sep = head.rfind("------")
    if last_sep == -1:
        cut = nl + len(head)
    else:
        # include the separator line and its newline.
        sep_line_end = head.find("\n", last_sep)
        if sep_line_end == -1:
            sep_line_end = len(head)
        cut = nl + sep_line_end + 1
    return corrupted[:cut]


@torch.no_grad()
def generate_continuation(model, tokenizer, prompt_text: str,
                          max_new_tokens: int = 96) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt",
                       truncation=True, max_length=1536).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    return text


def extract_summation_from_continuation(cont: str) -> int | None:
    """Find the integer summation in the continuation text."""
    m = re.search(r"FINAL ANSWER:?\s*(\d+)", cont, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback: first multi-digit number on its own line.
    m = re.search(r"^\s*(\d{3,9})\s*$", cont, re.MULTILINE)
    if m:
        return int(m.group(1))
    return None


# ---------- main labeling pass ----------

def label_records(records: list[dict], model, tokenizer,
                  build_prompt_fn=None,
                  delta: int = 7) -> list[dict]:
    """Run the two-corruption test on each record and append labels.

    Args:
        records: list of dicts with fields `prompt`, `full_generation`,
                 `correct_answer` (as returned by
                 run_multiplication_acc_evaluation).
        build_prompt_fn: callable(prompt_str) -> str that wraps the raw
                 "xx × yy =" into the same chat-template prompt that was
                 used during generation. The corrupted CoT is appended
                 after this so the model sees consistent context.
        delta: amount to add to a PP when corrupting it. We bound to
               keep the result positive.
    """
    labeled = []
    for r in tqdm(records, desc="Corruption labeling"):
        parsed = parse_cot(r.get("full_generation", ""))
        out = dict(r)
        out["approach"] = parsed.get("approach") if parsed else None
        out["follows_partial_products"] = False
        out["corruption_details"] = {}

        if not parsed or parsed.get("approach") != "B" or "pp1" not in parsed:
            labeled.append(out)
            continue

        pp1, pp2 = parsed["pp1"], parsed["pp2"]
        orig_final = parsed.get("final")
        orig_sum = pp1 + pp2

        # Generator preamble = prompt wrapped by the same chat template
        # used during the original evaluation, followed by the corrupted
        # CoT prefix.
        base_prompt = (build_prompt_fn(r["prompt"])
                       if build_prompt_fn else r["prompt"])

        follows = False
        for pp_idx, (orig_val, other_val) in enumerate(
                [(pp1, pp2), (pp2, pp1)]):
            corrupted_val = orig_val + delta
            corrupted_text = build_corrupted_prefix(
                r["full_generation"], pp_idx, corrupted_val)
            if corrupted_text is None:
                continue
            # Ask the model to continue from the corrupted prefix.
            input_text = base_prompt + "\n" + corrupted_text
            cont = generate_continuation(model, tokenizer, input_text)
            new_sum = extract_summation_from_continuation(cont)

            tracked = (new_sum == corrupted_val + other_val)
            recalled = (new_sum == orig_sum or new_sum == orig_final)
            out["corruption_details"][f"pp{pp_idx+1}"] = {
                "orig": orig_val, "corrupted_to": corrupted_val,
                "new_sum": new_sum, "expected_if_track": corrupted_val + other_val,
                "expected_if_recall": orig_sum,
                "tracked": tracked, "recalled": recalled,
            }
            follows = follows or tracked

        out["follows_partial_products"] = follows
        labeled.append(out)

    return labeled


# ---------- prompt wrapper (mirrors run_multiplication_acc_evaluation) ----------

def make_mult_chat_prompt(tokenizer, raw_question: str,
                          digits_desc: str = "2-digit × 2-digit") -> str:
    """Replicate the user prompt used during original evaluation, then
    apply chat template so the corruption rollout starts from the same
    context the model originally saw.

    Must stay in sync with
    run_multiplication_acc_evaluation in evaluation_utils.py.
    """
    user_prompt = (
        f"You are a precise calculator. You will be given a {digits_desc} "
        "multiplication problem. You may solve it using either of the "
        "following two approaches:\n\n"
        "Approach A: Direct Answer. If you are confident you know the answer "
        "without detailed computation, you may provide it directly.\n"
        "Here is an example of Approach A:\n"
        "APPROACH: A\nFINAL ANSWER: 1716\n\n"
        "Approach B: Long Multiplication. Follow the standard long "
        "multiplication algorithm step by step. Structure your response "
        "using numbered steps 1 to 5:\n"
        "1. Write the two numbers aligned by least significant digit.\n"
        "2. Compute each partial product line (one for each digit of the "
        "second number), recording carries if any.\n"
        "3. Shift each subsequent line left by the appropriate amount.\n"
        "4. Add all lines column by column from right to left, tracking "
        "carries.\n"
        "5. Finally, state the complete answer (no leading zeros) with "
        "prefix 'FINAL ANSWER:'.\n"
        "Here is an example of Approach B:\n"
        "APPROACH: B\n"
        "1.\n 39\n× 44\n------\n\n"
        "2.\n 39\n× 44\n------\n 156 (4 × 39)\n 1560 (40 × 39)\n------\n\n"
        "3.\n 39\n× 44\n------\n 156\n 1560\n------\n\n"
        "4.\n 39\n× 44\n------\n 156\n 1560\n------\n 1716\n\n"
        "5. FINAL ANSWER: 1716\n\n"
        "Choose the approach that best reflects how you actually arrive at "
        "the answer. There is no penalty for choosing either approach. "
        "Begin your response by stating 'APPROACH: A' or 'APPROACH: B', "
        "then follow the corresponding format.\n\n"
        f"Now solve the following multiplication:\n{raw_question}"
    )
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="generation jsonl")
    ap.add_argument("--output", required=True, help="labeled jsonl")
    ap.add_argument("--model_dir", default="/scratch/yh6210/transformers")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(args.model_dir, args.model_name),
        torch_dtype=dtype, trust_remote_code=True).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_dir, args.model_name), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = []
    with jsonlines.open(args.input, "r") as reader:
        records.extend(reader)
    if args.limit:
        records = records[:args.limit]
    print(f"Loaded {len(records)} records from {args.input}")

    def builder(raw_q):
        return make_mult_chat_prompt(tokenizer, raw_q)

    labeled = label_records(records, model, tokenizer,
                            build_prompt_fn=builder)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(args.output, "w") as writer:
        writer.write_all(labeled)
    print(f"Wrote labels to {args.output}")

    # Quick summary.
    n_b = sum(1 for r in labeled if r.get("approach") == "B")
    n_track = sum(1 for r in labeled if r.get("follows_partial_products"))
    print(f"Approach B: {n_b}/{len(labeled)}; "
          f"follows_partial_products: {n_track} ({n_track / max(n_b, 1):.3f})")


if __name__ == "__main__":
    main()
