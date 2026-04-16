import jsonlines
import re
from datasets import load_dataset, get_dataset_config_names
from pathlib import Path
from tqdm import tqdm
import glob


def extract_question_and_choices(prompt_content: str):
    """
    从 unbiased_prompt 的 content 中提取纯 question 文本。
    """
    question_match = re.search(r'Question:\s*(.+?)(?:\n\nChoices:|$)', prompt_content, re.DOTALL)
    if not question_match:
        return None
    question = question_match.group(1).strip()
    return question


def build_mmlu_correct_map():
    """
    只构建一次 MMLU 正确答案映射（全局加载一次）。
    """
    print("正在获取 MMLU 数据集所有 config 并构建映射（只需运行一次）...")
    configs = get_dataset_config_names("cais/mmlu")
    configs = [c for c in configs if c != 'auxiliary_train']  # 跳过超大训练集
    print(f"将加载 {len(configs)} 个主体 config 的 test/validation/dev split")

    mmlu_map = {}
    total_loaded = 0
    splits_to_load = ['test', 'validation', 'dev']

    for config in configs:
        for split in splits_to_load:
            try:
                dataset = load_dataset("cais/mmlu", config, split=split)
                for item in dataset:
                    question = item['question'].strip()
                    answer_idx = item['answer']
                    correct_letter = chr(65 + answer_idx)
                    if question not in mmlu_map:
                        mmlu_map[question] = correct_letter
                total_loaded += len(dataset)
                print(f"  成功加载 {config} - {split}: {len(dataset)} 个样本")
            except Exception:
                pass

    print(f"\nMMLU 正确答案映射构建完成，共 {len(mmlu_map)} 个唯一问题（总加载 {total_loaded} 个样本）")
    return mmlu_map


def add_correct_answer_to_anthropic_dataset(input_jsonl_path: str, mmlu_map: dict, output_jsonl_path: str = None):
    """
    为单个 Anthropic jsonl 文件添加 correct_answer。
    """
    if output_jsonl_path is None:
        output_jsonl_path = str(Path(input_jsonl_path).with_name(Path(input_jsonl_path).stem + "_with_correct.jsonl"))

    print(f"\n正在加载 Anthropic 数据集: {Path(input_jsonl_path).name}")
    samples = []
    with jsonlines.open(input_jsonl_path) as reader:
        for obj in reader:
            samples.append(obj)
    print(f"加载完成，共 {len(samples)} 个样本")

    missing_count = 0
    parse_fail_count = 0
    for sample in tqdm(samples, desc=f"添加 correct_answer ({Path(input_jsonl_path).name})"):
        if not sample["unbiased_prompt"]:
            parse_fail_count += 1
            sample["correct_answer"] = None
            continue

        unbiased_content = sample["unbiased_prompt"][0]["content"]
        question = extract_question_and_choices(unbiased_content)

        if question is None:
            parse_fail_count += 1
            sample["correct_answer"] = None
            continue

        correct = mmlu_map.get(question, None)
        sample["correct_answer"] = correct
        if correct is None:
            missing_count += 1

    total_missing = missing_count + parse_fail_count
    print(
        f"添加完成！未匹配/解析失败的样本数: {total_missing}/{len(samples)} ({total_missing / len(samples) * 100:.2f}%)")
    if total_missing > 0:
        print(f"  - 解析失败: {parse_fail_count}")
        print(f"  - 未匹配 question: {missing_count}")

    print(f"正在保存到: {Path(output_jsonl_path).name}")
    with jsonlines.open(output_jsonl_path, mode='w') as writer:
        writer.write_all(samples)


# ==================== 批量处理所有文件 ====================
if __name__ == "__main__":
    folder_path = "/scratch/yh6210/datasets/antropic_faithfulness/"

    # 只构建一次 MMLU 映射（全局共享）
    mmlu_map = build_mmlu_correct_map()

    # 获取文件夹下所有 .jsonl 文件（排除已带 _with_correct 的）
    jsonl_files = glob.glob(folder_path + "*.jsonl")
    jsonl_files = [f for f in jsonl_files if "_with_correct.jsonl" not in f]

    if not jsonl_files:
        print("警告: 文件夹中未找到 .jsonl 文件！请检查路径")
    else:
        print(f"\n找到 {len(jsonl_files)} 个待处理文件，开始批量添加 correct_answer...")
        for file_path in jsonl_files:
            add_correct_answer_to_anthropic_dataset(file_path, mmlu_map)

        print("\n所有文件处理完成！每个原始 jsonl 旁都会生成对应的 _with_correct.jsonl 文件")