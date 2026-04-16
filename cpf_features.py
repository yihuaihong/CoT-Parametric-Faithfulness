import torch
import random
import numpy as np
from tqdm import tqdm
from os.path import join
import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from CPF_utils import data_utils, evaluation_utils


# ======================
# Argument parsing
# ======================

def parse_args():
    parser = argparse.ArgumentParser()

    # reproducibility
    parser.add_argument("--seed", type=int, default=8888)
    parser.add_argument("--device", type=int, default=0)

    # model
    parser.add_argument("--model_dir", type=str, default="/scratch/yh6210/transformers")
    parser.add_argument("--model_name", type=str, default="gemma-2-9b-it")

    # dataset
    parser.add_argument("--dataset_name", type=str, required=True, help="必须指定 dataset_name（如 TwoHopFact, Hint_MMLU, 2-digit-Multiplication 等）")
    parser.add_argument("--dataset_dir", type=str, default="/scratch/yh6210/datasets")

    # extraction hyperparams
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument(
        "--use_cot_prompt",
        action="store_true",
        help="是否使用 CoT 提示（仅对支持的任务有效，如 Multiplication 和 TwoHopFact；Hint 任务忽略此参数）"
    )

    parser.add_argument(
        "--skip_generation",
        action="store_true",  # 注意改成 store_true
        help="跳过生成模型回答（默认不跳过，即会生成）"
    )

    # 提取 hidden states（本脚本专用于提取 hs）
    parser.add_argument("--extract_hs", action="store_true", help="提取并保存 prompt 最后一个 token 的 hidden states（所有层）")
    parser.add_argument("--hs_save_dir", type=str, default="/scratch/yh6210/CPF/hs_cache", help="hidden states 保存目录")
    parser.add_argument("--hs_layers", type=int, nargs="+", default=None, help="指定要提取的层（例如 --hs_layers 10 20 30；默认提取所有层）")

    # datasets hyperparams（调试用）
    parser.add_argument("--sample_num", type=int, default=0, help="采样样本数（0 表示全量）")

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
        torch_dtype=torch.float32,
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
    print(f"Loaded model {args.model_name} with {num_layers} layers")

    # ----- dataset -----
    dataset = data_utils.load_dataset(
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
        sample_num=args.sample_num,
        seed=args.seed
    )
    print(f"Loaded dataset {args.dataset_name} with {len(dataset)} samples")

    # ----- hidden states extraction -----
    if not args.extract_hs:
        print("未启用 --extract_hs，脚本将退出（本脚本专用于 hidden states 提取）")
        return

    os.makedirs(args.hs_save_dir, exist_ok=True)

    # 统一生成保存文件名（包含 cot/nocot 标识，仅对支持的任务有效）
    cot_str = "cot" if args.use_cot_prompt else "nocot"
    sample_str = f"{args.sample_num}" if args.sample_num > 0 else "full"
    save_file = os.path.join(
        args.hs_save_dir,
        f"{args.model_name.replace('/', '_')}_{args.dataset_name}_{cot_str}_{sample_str}_hs.pt"
    )
    print(f"开始提取 hidden states（{len(dataset)} samples），将保存至 {save_file}")

    # 根据 dataset_name 调用对应的提取函数（所有函数已支持纯 extract_hs 模式）
    if args.dataset_name in ['TwoHopFact', 'HoppingtooLate']:
        evaluation_utils.run_two_hop_acc_evaluation(
            model=model,
            dataset=dataset,
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            extract_hs=True,
            hs_layers=args.hs_layers,  # 支持指定层
            hs_save_path=save_file
        )

    elif args.dataset_name in ['Hint_MMLU', 'Hint_GPQA']:
        # Hint 任务忽略 use_cot_prompt（始终使用 biased_prompt）
        evaluation_utils.run_hint_acc_evaluation(
            model=model,
            dataset=dataset,
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            extract_hs=args.extract_hs,
            skip_generation=args.skip_generation,
            gemini_labeler=False,
            output_path=f'/scratch/yh6210/results/open-r1/hint_mmlu_results/hint_mmlu_false_{args.model_name}_results.jsonl',
            hs_layers=args.hs_layers,
            hs_save_path=save_file
        )

    elif 'Multiplication' in args.dataset_name:  # 兼容 2/3/4-digit-Multiplication
        evaluation_utils.run_multiplication_acc_evaluation(
            model=model,
            dataset=dataset,
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            use_cot_prompt=args.use_cot_prompt,  # 传递 CoT 控制
            extract_hs=True,
            hs_layers=args.hs_layers,
            hs_save_path=save_file
        )

    else:
        raise ValueError(f"不支持的 dataset_name: {args.dataset_name}（当前支持 TwoHopFact/HoppingtooLate, Hint_MMLU/Hint_GPQA, *-digit-Multiplication）")

    print(f"Hidden states 提取完成！保存路径: {save_file}")


if __name__ == "__main__":
    main()



# # TwoHopFact 全量提取（所有层）
# python cpf_features.py --extract_hs --dataset_name TwoHopFact --use_cot_prompt
#
# # Hint_MMLU 采样 500（指定后几层节省内存）(有提取hs以及收集模型回复的作用，后续通过python gemini_label_for_hint_mmlu来实现bias features的作用，打标faithful and unfaithful)

# python cpf_features.py --extract_hs --dataset_name Hint_MMLU --sample_num 500 --hs_layers 10 15 20 25 30
# python cpf_features.py --extract_hs --dataset_name Hint_MMLU --sample_num 500 --batch_size 32


# python cpf_features.py --extract_hs --dataset_name Hint_MMLU --sample_num 500 --batch_size 24 --model_name Meta-Llama-3-8B-Instruct
# python cpf_features.py --extract_hs --skip_generation --dataset_name Hint_MMLU --batch_size 24 --model_name Meta-Llama-3-8B-Instruct
# python cpf_features.py --extract_hs --dataset_name Hint_MMLU --sample_num 500 --batch_size 16 --model_name Qwen3-8B
# python cpf_features.py --extract_hs --dataset_name Hint_MMLU --batch_size 16 --model_name Qwen3-8B
# python cpf_features.py --extract_hs --dataset_name Hint_MMLU --batch_size 16 --model_name gemma-2-9b-it
#
# # 4-digit Multiplication CoT 模式
# python cpf_features.py --extract_hs --dataset_name 4-digit-Multiplication --use_cot_prompt --batch_size 32
#
# # Direct 模式 Multiplication
# python cpf_features.py --extract_hs --dataset_name 4-digit-Multiplication  # 不加 --use_cot_prompt





