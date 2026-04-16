import pandas as pd
import os

# 文件路径（根据你的路径）
csv_path = '/scratch/yh6210/datasets/HoppingTooLate/HoppingTooLate.csv'

# 检查文件是否存在
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"文件不存在: {csv_path}")

# 读取CSV
df = pd.read_csv(csv_path)

# 检查列是否存在
col_name = 'source_prompt'
if col_name not in df.columns:
    raise ValueError(f"列 '{col_name}' 不存在！当前列名: {df.columns.tolist()}")

# 提取列并转为字符串（防止NaN）
source_prompts = df[col_name].astype(str)

# 检查是否以 " is" 或 "is" 结尾（考虑常见情况：句尾 " is" 或直接 "is"）
# 常见prompt结尾如 "The capital of France is" → 以 " is" 结尾（注意空格）
ends_with_is_space = source_prompts.str.endswith(' is')
ends_with_is = source_prompts.str.endswith('is')

# 总样本数
total = len(source_prompts)

# 统计
count_is_space = ends_with_is_space.sum()
count_is = ends_with_is.sum()
count_both = (ends_with_is_space | ends_with_is).sum()

# 不匹配的样本
not_matching = source_prompts[~(ends_with_is_space | ends_with_is)]
not_matching_count = len(not_matching)

print(f"=== 检查结果 ===")
print(f"总样本数: {total}")
print(f"以 ' is' 结尾: {count_is_space} ({count_is_space / total:.2%})")
print(f"以 'is' 结尾（无空格）: {count_is} ({count_is / total:.2%})")
print(f"至少一种匹配: {count_both} ({count_both / total:.2%})")
print(f"不匹配样本数: {not_matching_count} ({not_matching_count / total:.2%})")

if not_matching_count > 0:
    print(f"\n不匹配示例（前20个）:")
    print(not_matching.head(20).tolist())

    # 可选：保存不匹配到新CSV
    not_matching_df = df.loc[not_matching.index]
    output_path = '/scratch/yh6210/datasets/HoppingTooLate/HoppingTooLate_not_ending_with_is.csv'
    not_matching_df.to_csv(output_path, index=False)
    print(f"\n不匹配样本已保存到: {output_path}")
else:
    print("\n所有source_prompt都以 'is' 或 ' is' 结尾！完美！")