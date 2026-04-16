import os
import pandas as pd
import numpy as np
from CPF_utils.data_utils import read_dataframe

import pandas as pd
import os

# 假设你的dataset_dir和read_dataframe函数已定义
# 如果没有，直接用pd.read_csv也行

dataset_dir = "/scratch/yh6210/datasets"  # 替换成你的实际路径
# csv_path = os.path.join(dataset_dir, 'TwoHopFact/TwoHopFact.csv')
#
# # 加载DataFrame（用你的read_dataframe，或直接pd.read_csv）
# df = read_dataframe(csv_path)  # 或 df = pd.read_csv(csv_path)

# ================== 清洗并保存新CSV ==================
def clean_and_save_twohopfact(dataset_dir: str, new_filename: str = 'TwoHopFact.csv'):
    csv_path = os.path.join(dataset_dir, 'TwoHopFact/TwoHopFact.csv')
    print(f"Loading {csv_path}...")
    df = read_dataframe(csv_path)

    # 关键：清洗所有subject相关列（以subject_cut.prompt结尾的）
    subject_cols = [col for col in df.columns if 'r1(e1).subject_cut.prompt' in col]
    print(f"Found subject columns: {subject_cols}")

    cleaned_count = 0
    for col in subject_cols:
        original = df[col].copy()
        # 清洗逻辑：去掉尾部常见噪声（'、"、.、,、等），strip空格
        df[col] = df[col].astype(str).str.rstrip("'\".,")  # 去尾部' " . ,
        df[col] = df[col].str.strip()  # 去前后空格

        # 统计变化
        changed = (original != df[col])
        cleaned_count += changed.sum()
        print(f"Cleaned {changed.sum()} entries in {col}")

    print(f"Total cleaned entries: {cleaned_count}")

    # 保存新CSV
    new_path = os.path.join(dataset_dir, 'TwoHopFact', new_filename)
    df.to_csv(new_path, index=False)
    print(f"Cleaned dataset saved to {new_path}")


clean_and_save_twohopfact(dataset_dir, 'TwoHopFact.csv')