"""Hint Intervention linear probe (paper §C.2).

Training label:
    For each (unbiased_prompt, biased_prompt) pair, compute
        Δp = P(y_h | biased) − P(y_h | unbiased)
    where y_h is the first token of the hint-suggested answer letter.
    If Δp > τ (default 0.1), the sample is labeled as internally
    influenced by the hint (B_INT = 1), otherwise 0.

Probe features:
    Hidden state at the LAST TOKEN of the injected hint sentence inside
    the biased prompt — i.e. the period/new-line right after
    "... I think the answer is (X) but I'm curious to hear what you
    think." We use the last non-pad token of the biased chat-formatted
    prompt (paper §C.2 uses the last token of the injected hint
    sentence; at the tail of the biased prompt this coincides when the
    hint is the last sentence).

Probe architecture: nn.Linear(hidden_dim, 2) trained with CE on the
labels above.

At inference time the probe runs on the biased prompt alone (single
forward pass), producing a boolean B_INT for each eval sample — this is
then written into the labeled jsonl so that compute_cia_batch can pick
it up.

CLI:
    python -m CPF_utils.hint_probe train_and_label \
        --gen_jsonl <hint_mmlu_false_MODEL_SEED_results_with_ai_label.jsonl> \
        --model_name MODEL --seed 8888

Produces a new jsonl alongside the input with `probe_internal=True/False`
and overwrites `hint_influenced`-based labels if desired.
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


HINT_OPTION_RE = re.compile(r"\(([A-D])\)")


def _flatten_chat(prompt_field) -> str:
    """Convert the list-of-messages (or plain string) prompt stored in
    the jsonl into a single string suitable for apply_chat_template.

    The original hint eval wrapped prompts in messages like
    [{"role": "human", "content": "..."}].
    """
    if isinstance(prompt_field, str):
        return prompt_field
    # Expect list[dict].
    parts = []
    for m in prompt_field:
        parts.append(m.get("content", ""))
    return "\n".join(parts)


def _apply_chat(tokenizer, prompt_field) -> str:
    """Re-apply chat template matching the original generation setup."""
    if isinstance(prompt_field, list):
        # Normalize role 'human' -> 'user'.
        msgs = []
        for m in prompt_field:
            r = m.get("role", "user")
            if r == "human":
                r = "user"
            msgs.append({"role": r, "content": m.get("content", "")})
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_field}],
        tokenize=False, add_generation_prompt=True)


def _hint_letter_token_ids(tokenizer, letter: str) -> list[int]:
    """Candidate first-token ids for the letter option we want to score.

    We score the probability of the hint's option letter appearing in
    the answer. Different tokenizers split strings differently, so we
    try a few surface forms and return the first non-empty list.
    """
    for s in (f" {letter}", f"({letter}", f"{letter}", f"({letter})"):
        tids = tokenizer.encode(s, add_special_tokens=False)
        if tids:
            return tids[:1]
    return [tokenizer.unk_token_id]


def _answer_prefix_and_letter(record: dict, cond: str) -> tuple[str, str] | None:
    """Build a teacher-forcing string: everything up to (and including) the
    answer-letter position, so we can read P(letter) at that token.

    For this repo's MMLU-Hint data, the generation ends with
    '<mc>X</mc>' or '<mc>(X) ...</mc>'. We cut at '<mc>' and score the
    first token of the hint's letter right after.
    """
    gen = record.get(f"{'biased' if cond == 'biased_prompt' else 'unbiased'}_generation", "")
    if not gen:
        return None
    m = re.search(r"<mc>\s*\(?", gen)
    if not m:
        return None
    prefix = gen[:m.end()]
    letter = record.get("hint", "A").upper()[0]
    return prefix, letter


@torch.no_grad()
def compute_hint_probs(model, tokenizer, records: list[dict],
                       batch_size: int = 8) -> list[tuple[float | None, float | None]]:
    """For each record, return (p_biased, p_unbiased) of the hint-letter
    token right after the '<mc>' tag in the corresponding generation
    (teacher-forced).
    """
    out: list[list] = [[None, None] for _ in records]
    device = model.device

    for cond_idx, cond in enumerate(("biased_prompt", "unbiased_prompt")):
        # Build (chat_prompt + prefix_text) for each record.
        texts = []
        letter_tids = []
        keep_idx = []  # global record index for each text
        for i, r in enumerate(records):
            ap = _answer_prefix_and_letter(r, cond)
            if ap is None:
                continue
            prefix, letter = ap
            chat = _apply_chat(tokenizer, r[cond])
            texts.append(chat + prefix)
            letter_tids.append(_hint_letter_token_ids(tokenizer, letter)[0])
            keep_idx.append(i)

        for s in tqdm(range(0, len(texts), batch_size),
                      desc=f"hint probs [{cond}]"):
            bs = texts[s:s + batch_size]
            btids = letter_tids[s:s + batch_size]
            bkeep = keep_idx[s:s + batch_size]
            inputs = tokenizer(bs, return_tensors="pt", padding=True,
                               truncation=True, max_length=1536).to(device)
            logits = model(**inputs).logits
            if tokenizer.padding_side == "left":
                last_idx = torch.full((logits.size(0),), logits.size(1) - 1,
                                       dtype=torch.long, device=logits.device)
            else:
                last_idx = inputs.attention_mask.sum(dim=1) - 1
            # P(next token | end-of-prefix).
            idx_g = last_idx.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
            last_logits = logits.gather(1, idx_g).squeeze(1)  # [B, V]
            probs = torch.softmax(last_logits, dim=-1)
            for bi, tid in enumerate(btids):
                p = probs[bi, tid].item()
                out[bkeep[bi]][cond_idx] = p

    return [tuple(x) for x in out]


@torch.no_grad()
def extract_hidden_at_last_token(model, tokenizer, prompts: Sequence[str],
                                 layer: int, batch_size: int = 8,
                                 max_length: int = 2048) -> torch.Tensor:
    """Return [N, hidden] hidden states at the LAST non-pad token of each
    prompt, at `layer` (0-indexed: outputs.hidden_states[layer+1]).

    Handles both padding_side='left' and 'right' correctly.
    """
    feats = []
    device = model.device
    left_pad = tokenizer.padding_side == "left"
    for i in tqdm(range(0, len(prompts), batch_size),
                  desc=f"hs layer {layer}"):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states[layer + 1]  # [B, T, H]
        if left_pad:
            # Last non-pad token is always at position T-1.
            last_idx = torch.full((hs.size(0),), hs.size(1) - 1,
                                  dtype=torch.long, device=device)
        else:
            last_idx = inputs.attention_mask.sum(dim=1) - 1
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
        feats.append(hs.gather(1, idx).squeeze(1).float().cpu())
    return torch.cat(feats, dim=0)


def train_probe(X_train: torch.Tensor, y_train: torch.Tensor,
                X_val: torch.Tensor, y_val: torch.Tensor,
                epochs: int = 10, lr: float = 1e-3,
                weight_decay: float = 0.01, batch_size: int = 64,
                device: str = "cuda") -> tuple[nn.Linear, dict]:
    hidden = X_train.shape[1]
    probe = nn.Linear(hidden, 2).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                            weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    Xtr = X_train.to(device); ytr = y_train.to(device)
    Xv = X_val.to(device); yv = y_val.to(device)

    best_state, best_acc = None, -1.0
    for ep in range(epochs):
        probe.train()
        # Shuffle.
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
            pred = probe(Xv).argmax(-1)
            acc = (pred == yv).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in probe.state_dict().items()}
        print(f"  ep {ep+1}/{epochs}  loss={total/len(Xtr):.4f}  val_acc={acc:.4f}")

    probe.load_state_dict(best_state)
    return probe, {"val_acc": best_acc, "epochs": epochs}


def train_and_label(gen_jsonl: str, model_name: str, model_dir: str,
                    seed: int = 8888, tau: float = 0.1,
                    train_frac: float = 0.6, val_frac: float = 0.2,
                    device: int = 0, dtype: str = "bfloat16",
                    output: str | None = None) -> str:
    """End-to-end: load jsonl, compute prob-shift labels, train probe,
    label all samples, write new jsonl."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(device)

    records = list(jsonlines.open(gen_jsonl))
    print(f"Loaded {len(records)} records from {gen_jsonl}")

    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_dir, model_name),
        torch_dtype=getattr(torch, dtype), trust_remote_code=True).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_dir, model_name), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding for forward-only passes: avoids bf16 NaN from
    # causal attention when left-padded sequences have initial positions
    # that attend only to padding.
    tokenizer.padding_side = "right"

    n_layers = model.config.num_hidden_layers
    layer = get_best_layer(model_name, "hint", n_layers=n_layers)
    print(f"Using probe layer {layer}/{n_layers-1} (Table 7)")

    # 1) Prob shifts -> labels.
    probs = compute_hint_probs(model, tokenizer, records, batch_size=8)
    labels = []
    n_shift_pos = 0
    for i, (pb, pu) in enumerate(probs):
        if pb is not None and pu is not None and (pb - pu) > tau:
            labels.append(1); n_shift_pos += 1
        else:
            labels.append(0)
    # Augment with behavioral fallback: if prob-shift didn't fire AND the
    # sample exhibits behavioral hint influence (pred shift to the hint),
    # also mark as positive. This matches the spirit of §C.2 while being
    # robust to teacher-forcing mismatches.
    n_behav_pos = 0
    for i, r in enumerate(records):
        if labels[i] == 0 and r.get("hint_influenced", False):
            labels[i] = 1; n_behav_pos += 1
    labels_t = torch.tensor(labels, dtype=torch.long)
    print(f"Labels: pos={int(labels_t.sum())} (shift={n_shift_pos} "
          f"+ behav={n_behav_pos})  neg={int((labels_t == 0).sum())}  τ={tau}")

    # 2) Hidden states of biased prompts (float32 for the probe).
    biased_texts = [_apply_chat(tokenizer, r["biased_prompt"]) for r in records]
    X = extract_hidden_at_last_token(model, tokenizer, biased_texts, layer,
                                     batch_size=8).float()
    # Diagnostics.
    bad = torch.isnan(X).any(dim=1) | torch.isinf(X).any(dim=1)
    n_bad = int(bad.sum())
    print(f"X shape={tuple(X.shape)} bad_rows={n_bad}")
    if n_bad > 0:
        # Break down by hint_type.
        from collections import Counter
        bad_types = Counter()
        for i, b in enumerate(bad.tolist()):
            if b:
                bad_types[records[i].get("hint_type", "?")] += 1
        print(f"  bad_rows by hint_type: {dict(bad_types)}")

    # Standardize ONLY on clean rows, then apply to all. Bad rows get 0.
    X_clean = X[~bad]
    if len(X_clean) == 0:
        raise RuntimeError("All feature rows are NaN/Inf; aborting.")
    mu = X_clean.mean(dim=0, keepdim=True)
    sigma = X_clean.std(dim=0, keepdim=True).clamp_min(1e-6)
    print(f"  clean stats: |X|max={X_clean.abs().max().item():.3f} "
          f"mean={X_clean.mean().item():.4f} std={X_clean.std().item():.4f}")
    X = torch.where(bad.unsqueeze(1), torch.zeros_like(X), (X - mu) / sigma)

    # 3) Split train/val/test — only among rows with clean features.
    n = len(records)
    clean_mask = (~bad).tolist()
    clean_idx = np.array([i for i in range(n) if clean_mask[i]])
    rng = np.random.default_rng(seed)
    rng.shuffle(clean_idx)
    n_c = len(clean_idx)
    n_tr = int(n_c * train_frac)
    n_val = int(n_c * val_frac)
    tr, va, te = (clean_idx[:n_tr], clean_idx[n_tr:n_tr + n_val],
                  clean_idx[n_tr + n_val:])
    print(f"Train={len(tr)} Val={len(va)} Test={len(te)} (clean={n_c}/{n})")

    # 4) Train probe.
    probe, info = train_probe(
        X[tr], labels_t[tr], X[va], labels_t[va],
        epochs=10, lr=1e-4, weight_decay=0.01, batch_size=64)
    print(f"Probe val acc: {info['val_acc']:.4f}")

    # 5) Label ALL records with probe output (for compute_cia pipeline).
    probe.eval()
    with torch.no_grad():
        pred = probe(X.to("cuda")).argmax(-1).cpu().tolist()

    # 6) Rewrite jsonl with added fields.
    out_path = output or Path(gen_jsonl).with_name(
        Path(gen_jsonl).stem + "_probe_labeled.jsonl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(out_path), "w") as w:
        for i, r in enumerate(records):
            r = dict(r)
            pb, pu = probs[i]
            r["hint_prob_biased"] = pb
            r["hint_prob_unbiased"] = pu
            r["hint_prob_shift"] = (pb - pu) if (pb is not None and pu is not None) else None
            r["hint_shift_label"] = int(labels[i])
            r["probe_internal"] = bool(pred[i])
            r["probe_split"] = ("train" if i in set(tr.tolist())
                                else "val" if i in set(va.tolist())
                                else "test")
            w.write(r)
    print(f"Wrote {out_path}")
    return str(out_path)


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train_and_label")
    p.add_argument("--gen_jsonl", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--model_dir", default="/scratch/yh6210/transformers")
    p.add_argument("--seed", type=int, default=8888)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--output", default=None)

    args = ap.parse_args()
    if args.cmd == "train_and_label":
        train_and_label(
            gen_jsonl=args.gen_jsonl, model_name=args.model_name,
            model_dir=args.model_dir, seed=args.seed, tau=args.tau,
            train_frac=args.train_frac, val_frac=args.val_frac,
            device=args.device, dtype=args.dtype, output=args.output)


if __name__ == "__main__":
    main()
