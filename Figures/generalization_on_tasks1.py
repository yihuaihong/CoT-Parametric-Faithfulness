import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 示例数据（请替换成您的真实 CPF^{LP} 值）
# 行顺序: Base, Trained on TwoHopFact, Trained on MMLU-Hint, Trained on 2-Digit Mult
tasks = ['TwoHopFact', 'MMLU-Hint', '2-Digit Mult']

data_llama = np.array([
    [0.257, 0.203, 0.354],  # Base
    [0.45, 0.28, 0.38],  # Trained on TwoHopFact (假设提升)
    [0.30, 0.40, 0.36],  # Trained on MMLU-Hint
    [0.29, 0.25, 0.55],  # Trained on 2-Digit Mult
])

data_qwen = np.array([
    [0.241, 0.236, 0.554],
    [0.38, 0.28, 0.58],
    [0.29, 0.42, 0.60],
    [0.30, 0.30, 0.85],
])

data_gemma = np.array([
    [0.122, 0.247, 0.431],
    [0.28, 0.28, 0.45],
    [0.18, 0.38, 0.48],
    [0.20, 0.30, 0.70],
])

models = ['Llama-3-8B-Instruct', 'Qwen3-8B', 'Gemma2-9B-it']
datasets = [data_llama, data_qwen, data_gemma]
settings = ['Base', 'Trained on TwoHopFact', 'Trained on MMLU-Hint', 'Trained on 2-Digit Mult']

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
bar_width = 0.2
x = np.arange(len(tasks))

for i, (ax, data, model) in enumerate(zip(axes, datasets, models)):
    for j, setting in enumerate(settings):
        ax.bar(x + j * bar_width, data[j], bar_width, label=setting, alpha=0.8)

    ax.set_title(model, fontsize=12)
    ax.set_xlabel('Test Task')
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(tasks, rotation=15)
    ax.legend(fontsize=9)

axes[0].set_ylabel('CPF$^{\\text{LP}}$ $\\uparrow$')
plt.suptitle('Cross-task Generalization of CPF Improvements', fontsize=14)
plt.tight_layout()
plt.savefig('cpf_generalization_bar.pdf')  # 保存为PDF，用于LaTeX
plt.show()