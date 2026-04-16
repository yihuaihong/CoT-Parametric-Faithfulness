# import random
# import os
#
# def generate_multiplication_dataset(
#     num_digits_a: int,
#     num_digits_b: int,
#     num_samples: int = 5000,
#     output_file: str = None,
#     min_val_a: int = None,
#     max_val_a: int = None,
#     min_val_b: int = None,
#     max_val_b: int = None,
#     seed: int = 42
# ):
#     """
#     生成 n-digit × m-digit 乘法数据集，格式类似 processed_valid_large.txt：
#     每行： "d1 d2 ... dn * e1 e2 ... em"（数字间带空格，* 两边带空格）
#
#     参数：
#         num_digits_a: 第一个数的位数（2 或 3）
#         num_digits_b: 第二个数的位数（2 或 3）
#         num_samples: 生成样本数（默认 5000）
#         output_file: 输出 txt 文件路径（如果 None，则只打印前几行示例）
#         min_val_a / max_val_a: 第一个数的范围（默认避免前导零）
#         min_val_b / max_val_b: 第二个数的范围
#         seed: 随机种子，便于复现
#
#     示例：
#         generate_multiplication_dataset(3, 3, num_samples=5000, output_file='3digit_multiplication.txt')
#         generate_multiplication_dataset(2, 2, num_samples=5000, output_file='2digit_multiplication.txt')
#     """
#     random.seed(seed)
#
#     # 默认范围：避免前导零
#     if min_val_a is None:
#         min_val_a = 10 ** (num_digits_a - 1)   # e.g., 3-digit: 100
#     if max_val_a is None:
#         max_val_a = 10 ** num_digits_a - 1     # e.g., 3-digit: 999
#     if min_val_b is None:
#         min_val_b = 10 ** (num_digits_b - 1)
#     if max_val_b is None:
#         max_val_b = 10 ** num_digits_b - 1
#
#     print(f"正在生成 {num_digits_a}-digit × {num_digits_b}-digit 乘法数据集（{num_samples} 个样本）...")
#     print(f"范围: 第一个数 [{min_val_a}, {max_val_a}], 第二个数 [{min_val_b}, {max_val_b}]")
#
#     lines = []
#     for _ in range(num_samples):
#         a = random.randint(min_val_a, max_val_a)
#         b = random.randint(min_val_b, max_val_b)
#
#         # 转成字符串，带空格分隔每位数字
#         a_str_spaced = ' '.join(str(a))
#         b_str_spaced = ' '.join(str(b))
#
#         # 每行格式： "1 3 3 8 * 5 1 0 5"
#         line = f"{a_str_spaced} * {b_str_spaced}"
#
#         lines.append(line)
#
#         # 可选：计算 product 验证（不写入文件）
#         # product = a * b
#         # print(f"{a} × {b} = {product}")
#
#     if output_file:
#         output_path = os.path.abspath(output_file)
#         with open(output_file, 'w', encoding='utf-8') as f:
#             for line in lines:
#                 f.write(line + '\n')
#         print(f"数据集生成完成！保存到: {output_path}")
#         print(f"总样本数: {len(lines)}")
#     else:
#         print("前 10 个样本示例:")
#         for line in lines[:10]:
#             print(line)
#
#     # 返回 lines（便于后续处理）
#     return lines
#
#
# # ====================== 使用示例 ======================
# if __name__ == "__main__":
#     # 生成 3-digit × 3-digit 数据集（结果通常 5-6 位）
#     generate_multiplication_dataset(
#         num_digits_a=3,
#         num_digits_b=3,
#         num_samples=5000,
#         output_file='processed_valid_3digit.txt',
#         seed=8888  # 和你之前代码一致
#     )
#
#     # 生成 2-digit × 2-digit 数据集（结果通常 3-4 位）
#     generate_multiplication_dataset(
#         num_digits_a=2,
#         num_digits_b=2,
#         num_samples=5000,
#         output_file='processed_valid_2digit.txt',
#         seed=8888
#     )



import random
import os

def generate_multiplication_dataset(
    num_digits_a: int,
    num_digits_b: int,
    num_samples: int = 5000,
    output_file: str = None,
    min_val_a: int = None,
    max_val_a: int = None,
    min_val_b: int = None,
    max_val_b: int = None,
    seed: int = 42
):
    """
    生成 n-digit × m-digit 乘法数据集，格式类似 processed_valid_large.txt：
    每行： "d1 d2 ... dn * e1 e2 ... em"（数字间带空格，* 两边带空格）

    参数：
        num_digits_a: 第一个数的位数（2 或 3）
        num_digits_b: 第二个数的位数（2 或 3）
        num_samples: 生成样本数（默认 5000）
        output_file: 输出 txt 文件路径（如果 None，则只打印前几行示例）
        min_val_a / max_val_a: 第一个数的范围（默认避免前导零）
        min_val_b / max_val_b: 第二个数的范围
        seed: 随机种子，便于复现

    示例：
        generate_multiplication_dataset(3, 3, num_samples=5000, output_file='3digit_multiplication.txt')
        generate_multiplication_dataset(2, 2, num_samples=5000, output_file='2digit_multiplication.txt')
    """
    random.seed(seed)

    # 默认范围：避免前导零
    if min_val_a is None:
        min_val_a = 10 ** (num_digits_a - 1)   # e.g., 3-digit: 100
    if max_val_a is None:
        max_val_a = 10 ** num_digits_a - 1     # e.g., 3-digit: 999
    if min_val_b is None:
        min_val_b = 10 ** (num_digits_b - 1)
    if max_val_b is None:
        max_val_b = 10 ** num_digits_b - 1

    print(f"正在生成 {num_digits_a}-digit × {num_digits_b}-digit 乘法数据集（{num_samples} 个样本）...")
    print(f"范围: 第一个数 [{min_val_a}, {max_val_a}], 第二个数 [{min_val_b}, {max_val_b}]")

    lines = []
    for _ in range(num_samples):
        a = random.randint(min_val_a, max_val_a)
        b = random.randint(min_val_b, max_val_b)

        # 转成字符串，带空格分隔每位数字
        a_str_spaced = ' '.join(str(a))
        b_str_spaced = ' '.join(str(b))

        # 每行格式： "1 3 3 8 * 5 1 0 5"
        line = f"{a_str_spaced} * {b_str_spaced}"

        lines.append(line)

    if output_file:
        output_path = os.path.abspath(output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        print(f"数据集生成完成！保存到: {output_path}")
        print(f"总样本数: {len(lines)}")
    else:
        print("前 10 个样本示例:")
        for line in lines[:10]:
            print(line)

    # 返回 lines（便于后续处理）
    return lines


def write_dataset(lines, output_file):
    """辅助函数：将 lines 写入文件"""
    output_path = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"已保存 {len(lines)} 个样本到: {output_path}")


# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 生成 2-digit × 2-digit 数据集，总数 15000
    total_samples = 15000
    seed = 8888  # 保持和你原来一致，便于复现

    lines = generate_multiplication_dataset(
        num_digits_a=2,
        num_digits_b=2,
        num_samples=total_samples,
        output_file=None,  # 先不直接写文件，留到划分后分别写
        seed=seed
    )

    # 打乱数据集（使用不同种子，避免与生成顺序完全相关）
    random.seed(seed + 1)  # 或者固定一个新种子，比如 9999
    random.shuffle(lines)

    # 按 6:2:2 比例划分：9000 train, 3000 val, 3000 test
    train_size = 9000
    val_size = 3000
    test_size = 3000

    train_lines = lines[:train_size]
    val_lines = lines[train_size:train_size + val_size]
    test_lines = lines[train_size + val_size:]

    # 保存到三个文件
    write_dataset(train_lines, '2digit_train.txt')
    write_dataset(val_lines, '2digit_val.txt')
    write_dataset(test_lines, '2digit_test.txt')

    print("\n划分完成！")
    print(f"Train: {len(train_lines)} samples")
    print(f"Val:   {len(val_lines)} samples")
    print(f"Test:  {len(test_lines)} samples")
    print(f"总计:   {len(train_lines) + len(val_lines) + len(test_lines)} samples")