import pandas as pd
from collections import Counter
import numpy as np
from CPF_utils.data_utils import load_dataset
import torch
import random
import numpy as np
from tqdm import tqdm
from os.path import join
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from CPF_utils import data_utils, evaluation_utils


def compute_bridge_token_stats(df: pd.DataFrame, tokenizer, dataset_name: str, bridge_col: str = 'e2.value'):
    """
    计算数据集的bridge first token分布统计
    - unique比例
    - top 10 most common first_token_ids (with count and token text)
    - 返回first_token_ids list和unique set，用于后续overlap计算
    """
    bridges = df[bridge_col].astype(str).tolist()
    n_samples = len(bridges)

    first_token_ids = []
    for bridge in bridges:
        if not bridge or bridge in ('nan', 'null'):
            first_token_ids.append(tokenizer.unk_token_id)
            continue
        tokens = tokenizer.encode(bridge.strip(), add_special_tokens=False)
        first_token_ids.append(tokens[0] if tokens else tokenizer.unk_token_id)

    # Unique比例
    unique_ids = set(first_token_ids)
    unique_ratio = len(unique_ids) / n_samples if n_samples > 0 else 0

    # Top 10 frequent
    counter = Counter(first_token_ids)
    top10 = counter.most_common(10)

    # 转为可读（id + decoded token）
    top10_readable = []
    for token_id, count in top10:
        token_text = tokenizer.decode([token_id]).strip()
        top10_readable.append((token_id, token_text, count, f"{count / n_samples:.2%}"))

    print(f"\n=== {dataset_name} Bridge First Token Stats ===")
    print(f"Total samples: {n_samples}")
    print(f"Unique first_token_ids: {len(unique_ids)} (比例: {unique_ratio:.2%})")
    print(f"Top 10 most common first_token_ids:")
    print(f"{'Rank':<4} {'Token ID':<10} {'Token Text':<20} {'Count':<8} {'Ratio'}")
    for rank, (tid, text, count, ratio) in enumerate(top10_readable, 1):
        print(f"{rank:<4} {tid:<10} {text:<20} {count:<8} {ratio}")

    return first_token_ids, unique_ids


def compare_two_datasets(train_df, eval_df, tokenizer, train_name="SOCRATES", eval_name="TwoHopFact"):
    """
    比较两个数据集的bridge token分布
    """
    train_ids, train_unique = compute_bridge_token_stats(train_df, tokenizer, train_name)
    eval_ids, eval_unique = compute_bridge_token_stats(eval_df, tokenizer, eval_name)

    # Overlap
    intersection = train_unique & eval_unique
    union = train_unique | eval_unique
    overlap_ratio = len(intersection) / len(union) if union else 0

    print(f"\n=== Cross-Dataset Comparison ===")
    print(f"Train unique tokens: {len(train_unique)}")
    print(f"Eval unique tokens: {len(eval_unique)}")
    print(f"Intersection: {len(intersection)}")
    print(f"Union: {len(union)}")
    print(f"Overlap Ratio (Jaccard similarity): {overlap_ratio:.2%}")

    if overlap_ratio < 0.5:
        print("Overlap < 50% → 这很可能是一个主要的distribution shift原因！")
    else:
        print("Overlap ≥ 50% → token分布相对相似，shift可能来自其他方面（如prompt style）")

# ================== 使用示例 ==================
# 假设你已经加载了DataFrame和tokenizer
train_df = load_dataset('SOCRATES')  # 你的7232样本
eval_df = load_dataset('TwoHopFact')  # 你的45595样本


# ----- device -----
torch.cuda.set_device(0)

# ----- model & tokenizer -----
model = AutoModelForCausalLM.from_pretrained(
    join('/scratch/yh6210/transformers', "gemma-2-9b-it"),
    dtype=torch.float32,
    trust_remote_code=True,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    join('/scratch/yh6210/transformers', "gemma-2-9b-it"),
    trust_remote_code=True
)

if "qwen" in model.config.model_type.lower():
    tokenizer.pad_token = "<|endoftext|>"
elif tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

# 直接运行
compare_two_datasets(train_df, eval_df, tokenizer, "SOCRATES", "TwoHopFact")