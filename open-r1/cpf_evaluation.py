import torch
import random
import numpy as np
from tqdm import tqdm
from os.path import join
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from CPF_utils import data_utils, evaluation_utils


# ======================
# Argument parsing
# ======================

def parse_args():
    parser = argparse.ArgumentParser()

    # reproducibility
    parser.add_argument("--seed", type=int, default=8888) #5555 6666 7777 8888 9999
    parser.add_argument("--device", type=int, default=0)

    # model
    parser.add_argument("--model_dir", type=str, default="/scratch/yh6210/transformers")
    parser.add_argument("--model_name", type=str, default="gemma-2-9b-it")

    # dataset
    parser.add_argument("--dataset_name", type=str, default="TwoHopFact")
    parser.add_argument("--dataset_dir", type=str, default="/scratch/yh6210/datasets")

    # evaluation switches
    parser.add_argument("--eval_acc", action="store_true")
    parser.add_argument("--eval_cpf", action="store_true")

    # eval hyperparams
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument(
        "--use_cot_prompt",
        action="store_true",  # 加这个参数就 True，不加就 False
        help="是否使用竖式 CoT 提示（默认不使用，即 direct 模式）"
    )

    # datasets hyperparams 调试用，后面删掉，用于从数据集中抽样，用于快速跑通数据集
    parser.add_argument("--sample_num", type=int, default=0)

    return parser.parse_args()


# ======================
# Main
# ======================

def main():
    args = parse_args()

    # ----- seed -----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    tqdm.pandas()

    # ----- device -----
    torch.cuda.set_device(args.device)

    # ----- model & tokenizer -----
    model = AutoModelForCausalLM.from_pretrained(
        join(args.model_dir, args.model_name),
        dtype=torch.float32,
        trust_remote_code=True,
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        join(args.model_dir, args.model_name),
        trust_remote_code=True
    )

    if "qwen" in model.config.model_type.lower():
        tokenizer.pad_token = "<|endoftext|>"
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    num_layers = model.config.num_hidden_layers
    print(f"Loaded model with {num_layers} layers")

    # ----- dataset -----
    dataset = data_utils.load_dataset(
        dataset_name = args.dataset_name,
        dataset_dir = args.dataset_dir,
        sample_num = args.sample_num,
        seed = args.seed
    )

    # ----- evaluation -----
    if args.eval_acc:
        acc = evaluation_utils.accuracy_evaluation(
            model,
            args.model_name,
            dataset,
            args.dataset_name,
            tokenizer,
            batch_size=args.batch_size,
            use_cot_prompt=args.use_cot_prompt,
        )
        # print(f"accuracy_results: {acc} on Dataset: {args.dataset_name} with sample_num: {args.sample_num}")

    if args.eval_cpf:
        cpf = evaluation_utils.CPF_evaluation(
            model,
            dataset,
            args.dataset_name,
            tokenizer,
            batch_size=args.batch_size,
        )
        print(f"CPF_results: {cpf} on Dataset: {args.dataset_name} with sample_num: {args.sample_num}")


if __name__ == "__main__":
    main()

# python cpf_evaluation.py --eval_acc --use_cot_prompt --batch_size 32 --sample_num 500 --dataset_name TwoHopFact
# python cpf_evaluation.py --eval_acc --use_cot_prompt --batch_size 64 --sample_num 500 --dataset_name TwoHopFact
# python cpf_evaluation.py --eval_cpf --use_cot_prompt --batch_size 32 --sample_num 500 --dataset_name TwoHopFact

# Example Command: python cpf_evaluation.py --eval_cpf --use_cot_prompt --batch_size 64 --sample_num 100 --dataset_name SOCRATES
# python cpf_evaluation.py --eval_acc --use_cot_prompt --batch_size 64 --sample_num 0 --dataset_name Hint_MMLU
# python cpf_evaluation.py --eval_acc --use_cot_prompt --batch_size 32 --sample_num 500 --dataset_name 2-digit-Multiplication
# python cpf_evaluation.py --eval_acc --batch_size 32 --sample_num 500 --dataset_name 2-digit-Multiplication
# python cpf_evaluation.py --eval_acc --use_cot_prompt --batch_size 32 --sample_num 500 --dataset_name 2-digit-Multiplication --model_name Meta-Llama-3-8B-Instruct #Qwen3-8B

# python cpf_evaluation.py --eval_acc --use_cot_prompt --batch_size 32 --sample_num 500 --dataset_name Hint_MMLU