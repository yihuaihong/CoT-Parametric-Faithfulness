import jsonlines
import json
from pathlib import Path
import glob

folder = '/scratch/yh6210/datasets/antropic_faithfulness/'
files = glob.glob(folder + '*.jsonl')

for file_path in files:
    print(f"\n处理文件: {file_path}")
    # 放入上面的读取逻辑

    # 检查文件是否存在
    if not Path(file_path).exists():
        print(f"文件不存在: {file_path}")
    else:
        print(f"正在读取文件: {file_path}\n")

        # 使用 jsonlines 逐行读取（适合大文件）
        samples = []
        with jsonlines.open(file_path) as reader:
            for i, obj in enumerate(reader):
                samples.append(obj)
                if i >= 1:  # 只读取前 2 个样本（索引 0 和 1）
                    break

        if not samples:
            print("文件为空或无法读取")
        else:
            # 1. 显示文件结构（所有列名 / keys）
            first_sample = samples[0]
            print("文件结构（所有列名 / keys）:")
            print(list(first_sample.keys()))
            print("\n" + "=" * 60 + "\n")

            # 2. 显示前 2 个完整 example（包括所有列的值）
            for idx, sample in enumerate(samples, 1):
                print(f"前 {idx} 个 example（完整内容）:")
                # 使用 json.dumps 漂亮打印（indent=4）
                print(json.dumps(sample, indent=4, ensure_ascii=False))
                print("\n" + "=" * 60 + "\n")

            print(f"成功读取了 {len(samples)} 个样本（文件总样本数可能更多）")
            print("提示：fewshot_order 文件通常很大，每个样本包含很多 few-shot 示例，prompt 很长。")



