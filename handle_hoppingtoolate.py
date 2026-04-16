import pandas as pd
import os

# 文件路径
csv_path = '/scratch/yh6210/datasets/HoppingTooLate/HoppingTooLate.csv'

# 检查文件是否存在
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"文件不存在: {csv_path}")

# 读取CSV
df = pd.read_csv(csv_path)

# 打印原始列名和前几行示例，便于检查
print("原始列名:")
print(df.columns.tolist())
print("\n前5行示例:")
print(df.head())

# ================== 列名重命名 ==================
rename_map = {
    'source_prompt': 'r2(r1(e1)).prompt',
    'e2_label': 'e2.value',
}

# 执行重命名（只改存在的列）
df = df.rename(columns=rename_map)

# ================== 生成新列 r2(r1(e1)).subject_cut.prompt ==================
# 通过去掉 'r2(r1(e1)).prompt' 结尾的 " is"（带空格）得到
# 使用 rstrip 安全去除尾部空格+"is"，防止多余空格或异常
df['r2(r1(e1)).subject_cut.prompt'] = df['r2(r1(e1)).prompt'].astype(str).str.rstrip(' is')

# 可选验证：检查新列是否正确（随机抽样或全量）
print("\n=== 新列生成验证 ===")
print("随机5个样本对比:")
sample = df[['r2(r1(e1)).prompt', 'r2(r1(e1)).subject_cut.prompt']].sample(5, random_state=42)
print(sample)

# 检查是否有残留 " is"（应为0）
remaining_is = (df['r2(r1(e1)).subject_cut.prompt'].str.endswith(' is')).sum()
print(f"\n新列仍有结尾 ' is' 的样本数: {remaining_is} (应为0)")

# 打印新列名，确认
print("\n最终列名:")
print(df.columns.tolist())

# 保存覆盖原文件（或新文件）
output_path = '/scratch/yh6210/datasets/HoppingTooLate/HoppingTooLate.csv'
df.to_csv(output_path, index=False)
print(f"\n已保存修改后的CSV到: {output_path}")