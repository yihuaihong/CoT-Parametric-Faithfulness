# import numpy as np
# import matplotlib.pyplot as plt
# import colorsys
#
#
# def darken_color(hex_color, factor=0.6):
#     """将十六进制颜色变暗（factor越小越暗）"""
#     hex_color = hex_color.lstrip('#')
#     rgb = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
#     h, l, s = colorsys.rgb_to_hls(*rgb)
#     l = max(0, l * factor)  # 降低亮度
#     rgb_dark = colorsys.hls_to_rgb(h, l, s)
#     return '#%02x%02x%02x' % tuple(int(x * 255) for x in rgb_dark)
#
#
# # 示例数据（请替换成您的真实 CPF^{LP} 值）
# data_llama = np.array([
#     [0.257, 0.203, 0.354],  # Base
#     [0.45, 0.28, 0.38],  # Trained on TwoHopFact
#     [0.30, 0.40, 0.36],  # Trained on MMLU-Hint
#     [0.29, 0.25, 0.55],  # Trained on 2-Digit Mult
# ])
#
# data_qwen = np.array([
#     [0.241, 0.236, 0.554],
#     [0.38, 0.28, 0.58],
#     [0.29, 0.42, 0.60],
#     [0.30, 0.30, 0.85],
# ])
#
# data_gemma = np.array([
#     [0.122, 0.247, 0.431],
#     [0.28, 0.28, 0.45],
#     [0.18, 0.38, 0.48],
#     [0.20, 0.30, 0.70],
# ])
#
# # 合并数据
# all_data = np.stack([data_llama, data_qwen, data_gemma])  # shape: (3 models, 4 settings, 3 tasks)
#
# tasks = ['TwoHopFact', 'MMLU-Hint', '2-Digit Multiplication']
# models = ['Llama-3-8B-Instruct', 'Qwen3-8B', 'Gemma2-9B-it']
# trained_settings = ['Trained on TwoHopFact', 'Trained on MMLU-Hint', 'Trained on 2-Digit Multiplication']
#
# # 每个 Trained on 的浅色（用于 gain/top）
# gain_colors = ['#81a1c1', '#a3be8c', '#d08770']  # 对应三个 Trained on
#
# # 自动生成对应的深色（用于 Base/bottom）
# base_colors = [darken_color(c, factor=0.55) for c in gain_colors]  # 更深版本
#
# fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
# bar_width = 0.25
# x = np.arange(len(models))
#
# for task_idx, ax in enumerate(axes):
#     task_name = tasks[task_idx]
#     ax.set_title(f'Test Task: {task_name}', fontsize=15)
#
#     # x轴标签
#     ax.set_xticks(x)
#     ax.set_xticklabels(models, rotation=30, ha='center', fontsize=12)
#
#     # 该 test task 下的 Base 值（每个模型不同）
#     base_values = all_data[:, 0, task_idx]
#
#     for setting_idx, setting in enumerate(trained_settings):
#         offset = x + (setting_idx - 1) * bar_width  # 三根柱子居中
#         trained_values = all_data[:, setting_idx + 1, task_idx]
#         delta_values = trained_values - base_values  # 允许负值
#
#         # 底部：Base（对应 Trained on 的深色版本）
#         ax.bar(offset, base_values, bar_width,
#                color=base_colors[setting_idx], edgecolor='black', linewidth=0.8,
#                label='Base' if (task_idx == 0 and setting_idx == 0) else None)
#
#         # 顶部：Gain（对应 Trained on 的浅色，半透明 alpha=0.7 以减轻遮挡感）
#         ax.bar(offset, delta_values, bar_width, bottom=base_values,
#                color=gain_colors[setting_idx], alpha=0.7, edgecolor='black', linewidth=0.8,
#                label=setting if task_idx == 0 else None)
#
#     ax.grid(axis='y', linestyle='--', alpha=0.5)
#
# # 纵坐标
# axes[0].set_ylabel('CPF$^{\\text{LP}}$ $\\uparrow$', fontsize=14)
# for ax in axes:
#     ax.tick_params(axis='y', labelsize=12)
#
# # 图例
# handles, labels = axes[0].get_legend_handles_labels()
# base_handle = plt.Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black', label='Base (deeper shade)')
# handles = [base_handle] + handles
# labels = ['Base (deeper shade per training task)'] + trained_settings
# axes[0].legend(handles=handles, fontsize=11, title='Bar Components', title_fontsize=12, loc='upper left')
#
# plt.suptitle(
#     'Cross-task Generalization of CPF Improvements\n(Bottom: Base in deeper shade of training task color; Top: semi-transparent gain/loss)',
#     fontsize=18)
#
# plt.subplots_adjust(bottom=0.25, top=0.88)
# plt.tight_layout()
#
# plt.savefig('cpf_generalization_stacked_bar.pdf', dpi=300, bbox_inches='tight')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 示例数据（请替换成您的真实 CPF^{LP} 值）
data_llama = np.array([
    [0.257, 0.203, 0.354],  # Base
    [0.407, 0.263, 0.360],  # Trained on TwoHopFact
    [0.345, 0.291, 0.343],  # Trained on MMLU-Hint
    [0.272, 0.214, 0.551],  # Trained on 2-Digit Mult
])

data_qwen = np.array([
    [0.241, 0.236, 0.554],
    [0.324, 0.422, 0.58],
    [0.312, 0.437, 0.533],
    [0.242, 0.257, 0.853],
])

data_gemma = np.array([
    [0.122, 0.247, 0.431],
    [0.224, 0.312, 0.45],
    [0.181, 0.348, 0.435],
    [0.130, 0.261, 0.711],
])

# 合并数据
all_data = np.stack([data_llama, data_qwen, data_gemma])  # shape: (3 models, 4 settings, 3 tasks)

tasks = ['TwoHopFact', 'MMLU-Hint', '2-Digit Multiplication']
models = ['Llama-3-8B-Instruct', 'Qwen3-8B', 'Gemma2-9B-it']
settings = ['Base', 'Trained on TwoHopFact', 'Trained on MMLU-Hint', 'Trained on 2-Digit Multiplication']

# 低饱和度柔和颜色
colors = ['#88c0d0', '#81a1c1', '#a3be8c', '#d08770']

fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
bar_width = 0.2
x = np.arange(len(models))  # 模型位置

for task_idx, ax in enumerate(axes):
    task_name = tasks[task_idx]
    ax.set_title(f'Test Task: {task_name}', fontsize=15)  # 略微加大子图标题
    # ax.set_xlabel('Model', fontsize=12)

    # x轴标签居中、旋转30度
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(models, rotation=30, ha='center', fontsize=12)

    for setting_idx, setting in enumerate(settings):
        values = all_data[:, setting_idx, task_idx]
        ax.bar(x + setting_idx * bar_width, values, bar_width,
               label=setting, color=colors[setting_idx], alpha=0.9, edgecolor='black', linewidth=0.8)

    if task_idx == 0:
        ax.legend(fontsize=11, title='Training Setting', title_fontsize=12, loc='upper left')

# 纵坐标 label 和刻度数字加大
axes[0].set_ylabel('CPF$^{\\text{LP}}$ $\\uparrow$', fontsize=14)  # ylabel 加大
for ax in axes:
    ax.tick_params(axis='y', labelsize=12)  # y轴刻度数字加大（共享y轴也生效）

plt.suptitle('Cross-task Generalization of CPF Improvements', fontsize=18)  # 总标题略微加大

# 留出底部空间防止标签溢出
plt.subplots_adjust(bottom=0.25)
plt.tight_layout()

plt.savefig('cpf_generalization_bar_by_task.pdf')
plt.show()