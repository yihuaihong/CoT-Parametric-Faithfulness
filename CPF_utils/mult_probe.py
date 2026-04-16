"""Multiplication-task linear probe (paper §C.3).

Input:
    A corruption-labeled jsonl produced by
    `CPF_utils/multiplication_corruption.py`, where each row has:
        - approach              : 'A' or 'B' (parsed from original CoT)
        - follows_partial_products  : bool  ← B_INT ground-truth
        - full_generation       : CoT text (needed to locate summation)
        - corruption_details    : per-PP intervention results

Feature:
    Hidden state at the token position **immediately preceding the
    generation of the summation result**. We re-run model forward over
    `prompt + CoT_up_to_summation` and grab the last token's hidden
    state at the Table-7 probe layer.

Probe:
    nn.Linear(hidden, 2) trained with CE on B_INT labels. Standardize
    features, grad-clip, early-stop on val acc — mirror hint_probe.py.

CLI:
    python -m CPF_utils.mult_probe train_and_label \
        --labeled_jsonl <corruption_labeled.jsonl> \
        --model_name Meta-Llama-3-8B-Instruct --seed 8888

Writes a `*_probe_labeled.jsonl` alongside input with a new
`probe_internal` field per row — ready to be consumed by
`multiplication_parametric_faithfulness_reward` in probe mode.
"""
from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from typing import Sequence

import jsonlines
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from CPF_utils.layer_config import get_best_layer


# Matches the summation number (result of adding partial products).
# In paper/repo CoT format it usually appears after the 2nd PP block and
# before "FINAL ANSWER:". We split at that point to recover the "prompt
# up to the summation" prefix.
_PP_RE = re.compile(r"^\s*(\d+)\s*\(\s*(\d+)\s*[×x*]\s*(\d+)\s*\)",
                    re.MULTILINE)
_FINAL_RE = re.compile(r"FINAL ANSWER:?\s*(\d+)", re.IGNORECASE)


def locate_pre_summation_prefix(full_generation: str) -> str | None:
    """Return everything up to (but not including) the first line that
    contains the summation result. If we can't find ≥2 PPs we fall back
    to the entire generation minus the final-answer line.
    """
    pps = list(_PP_RE.finditer(full_generation))
    if len(pps) < 2:
        # Fallback: cut before FINAL ANSWER.
        fa = _FINAL_RE.search(full_generation)
        return full_generation[:fa.start()].rstrip() if fa else full_generation.rstrip()

    # Cut just after the second PP line (end of line).
    second_pp_end = pps[1].end()
    nl_idx = full_generation.find("\n", second_pp_end)
    cut = nl_idx + 1 if nl_idx >= 0 else second_pp_end
    # Advance past any "------" separator block that appears before the
    # summation number.
    rest = full_generation[cut:]
    sep_iter = re.finditer(r"^-{3,}\s*$", rest, re.MULTILINE)
    last_sep_end = None
    for m in sep_iter:
        # Stop once we're past the summation line (find first standalone digits after PPs).
        if re.search(r"^\s*\d{3,9}\s*$", rest[:m.start()], re.MULTILINE):
            break
        last_sep_end = m.end()
    if last_sep_end is not None:
        cut = cut + last_sep_end
    return full_generation[:cut].rstrip()


def make_mult_chat_prefix(tokenizer, raw_question: str, cot_prefix: str) -> str:
    """Apply the same chat template GRPO/eval use, then append the
    CoT-up-to-summation prefix as the start of the assistant's reply."""
    messages = [{"role": "user", "content": raw_question}]
    base = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return base + cot_prefix


@torch.no_grad()
def extract_hidden_at_last_token(
        model, tokenizer, texts: Sequence[str],
        layer: int, batch_size: int = 4,
        max_length: int = 2048) -> torch.Tensor:
    """Last-real-token hidden state at `layer` (0-indexed transformer
    block). Uses right padding to avoid the Llama-3 bf16 + left-pad NaN
    issue encountered in hint_probe.py."""
    feats = []
    device = model.device
    tokenizer.padding_side = "right"
    for i in tqdm(range(0, len(texts), batch_size),
                  desc=f"hs layer {layer}"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states[layer + 1]
        # Right padding: last non-pad index = sum(attention_mask)-1.
        last_idx = inputs.attention_mask.sum(dim=1) - 1
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
        feats.append(hs.gather(1, idx).squeeze(1).float().cpu())
    return torch.cat(feats, dim=0)


def train_probe(X_train, y_train, X_val, y_val,
                epochs: int = 12, lr: float = 5e-4,
                weight_decay: float = 0.01, batch_size: int = 64,
                device: str = "cuda") -> tuple[nn.Linear, dict]:
    hidden = X_train.shape[1]
    probe = nn.Linear(hidden, 2).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                            weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    Xtr, ytr = X_train.to(device), y_train.to(device)
    Xv, yv = X_val.to(device), y_val.to(device)

    best_state, best_acc = None, -1.0
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for s in range(0, len(Xtr), batch_size):
            bs = perm[s:s + batch_size]
            logits = probe(Xtr[bs])
            loss = ce(logits, ytr[bs])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(bs)
        probe.eval()
        with torch.no_grad():
            acc = (probe(Xv).argmax(-1) == yv).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in probe.state_dict().items()}
        print(f"  ep {ep+1}/{epochs}  loss={total/len(Xtr):.4f}  val_acc={acc:.4f}")
    probe.load_state_dict(best_state)
    return probe, {"val_acc": best_acc, "epochs": epochs}


def train_and_label(
        labeled_jsonl: str, model_name: str, model_dir: str,
        seed: int = 8888, train_frac: float = 0.6, val_frac: float = 0.2,
        device: int = 0, dtype: str = "bfloat16",
        output: str | None = None) -> str:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(device)

    records = list(jsonlines.open(labeled_jsonl))
    print(f"Loaded {len(records)} corruption-labeled records")

    # Keep only Approach-B rows with valid B_INT label.
    usable = [r for r in records
              if r.get("approach") == "B"
              and r.get("corruption_details")
              and "follows_partial_products" in r]
    print(f"Usable (Approach-B, labeled): {len(usable)}/{len(records)}")
    if len(usable) < 50:
        raise RuntimeError("Not enough Approach-B samples for probe training; "
                           "re-run multiplication_corruption.py on a larger set.")

    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_dir, model_name),
        torch_dtype=getattr(torch, dtype),
        trust_remote_code=True).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_dir, model_name), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    layer = get_best_layer(model_name, "multiplication", n_layers=n_layers)
    print(f"Using probe layer {layer}/{n_layers-1} (Table 7)")

    # 1) Build prompt+CoT prefix for each sample, extract hidden states.
    texts = []
    for r in usable:
        prefix = locate_pre_summation_prefix(r.get("full_generation", ""))
        raw_q = r.get("prompt", "")
        texts.append(make_mult_chat_prefix(tokenizer, raw_q, prefix or ""))
    print(f"Built {len(texts)} prompt+CoT strings")

    X = extract_hidden_at_last_token(model, tokenizer, texts, layer,
                                     batch_size=4)
    # Sanitize + standardize.
    bad = torch.isnan(X).any(dim=1) | torch.isinf(X).any(dim=1)
    n_bad = int(bad.sum())
    if n_bad:
        print(f"WARN: {n_bad} bad feature rows; zeroing them out")
    X_clean = X[~bad]
    if len(X_clean) == 0:
        raise RuntimeError("All feature rows are NaN/Inf")
    mu = X_clean.mean(0, keepdim=True)
    sigma = X_clean.std(0, keepdim=True).clamp_min(1e-6)
    print(f"  clean stats: |X|max={X_clean.abs().max():.3f} "
          f"mean={X_clean.mean():.4f} std={X_clean.std():.4f}")
    X = torch.where(bad.unsqueeze(1), torch.zeros_like(X), (X - mu) / sigma)

    # 2) Labels.
    y = torch.tensor(
        [int(bool(r["follows_partial_products"])) for r in usable],
        dtype=torch.long)
    n_pos = int(y.sum())
    print(f"Labels: pos (follows_partial)={n_pos}  neg={len(y)-n_pos}")

    # 3) Train/val/test split on CLEAN rows.
    n = len(usable)
    clean_idx = np.array([i for i in range(n) if not bad[i].item()])
    rng = np.random.default_rng(seed)
    rng.shuffle(clean_idx)
    n_c = len(clean_idx)
    n_tr = int(n_c * train_frac)
    n_val = int(n_c * val_frac)
    tr = clean_idx[:n_tr]
    va = clean_idx[n_tr:n_tr + n_val]
    te = clean_idx[n_tr + n_val:]
    print(f"Train={len(tr)} Val={len(va)} Test={len(te)} (clean={n_c}/{n})")

    # 4) Train probe.
    probe, info = train_probe(X[tr], y[tr], X[va], y[va],
                              epochs=12, lr=5e-4, weight_decay=0.01)
    print(f"Probe val acc: {info['val_acc']:.4f}")

    # 5) Label ALL records (Approach-B usable rows get probe_internal;
    # others get 0 since we can't probe — reward will fall back to
    # parser mode on those).
    probe.eval()
    with torch.no_grad():
        preds = probe(X.to("cuda")).argmax(-1).cpu().tolist()

    # Build usable-idx → pred lookup.
    pred_map = {id(r): p for r, p in zip(usable, preds)}

    out_path = output or Path(labeled_jsonl).with_name(
        Path(labeled_jsonl).stem + "_probe_labeled.jsonl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(out_path), "w") as w:
        for r in records:
            r = dict(r)
            if id(r) in pred_map:
                r["probe_internal"] = bool(pred_map[id(r)])
            else:
                # Not Approach-B or unusable: fall back to original
                # behavioral label from corruption test.
                r["probe_internal"] = bool(r.get("follows_partial_products", False))
            w.write(r)
    print(f"Wrote {out_path}")

    # Save probe weights alongside for later "probe-mode" reward runs.
    ckpt_path = Path(out_path).with_suffix(".probe.pt")
    torch.save({
        "state_dict": probe.state_dict(),
        "layer": layer,
        "mean": mu.squeeze(0).cpu(),
        "std": sigma.squeeze(0).cpu(),
        "model_name": model_name,
        "val_acc": info["val_acc"],
    }, ckpt_path)
    print(f"Saved probe checkpoint to {ckpt_path}")

    return str(out_path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train_and_label")
    p.add_argument("--labeled_jsonl", required=True,
                   help="output of multiplication_corruption.py")
    p.add_argument("--model_name", required=True)
    p.add_argument("--model_dir", default="/scratch/yh6210/transformers")
    p.add_argument("--seed", type=int, default=8888)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.cmd == "train_and_label":
        train_and_label(
            labeled_jsonl=args.labeled_jsonl, model_name=args.model_name,
            model_dir=args.model_dir, seed=args.seed,
            train_frac=args.train_frac, val_frac=args.val_frac,
            device=args.device, dtype=args.dtype, output=args.output)


if __name__ == "__main__":
    main()
