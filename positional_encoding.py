"""
Positional Encoding — 正弦位置编码
给 Self-Attention 注入词序信息
"""
import numpy as np


def sinusoidal_positional_encoding(seq_len, d_model):
    """
    生成正弦位置编码
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    参数:
        seq_len: 句子长度（几个词）
        d_model: 向量维度
    
    返回:
        pe: (seq_len, d_model) 位置编码矩阵
    """
    pe = np.zeros((seq_len, d_model))
    
    # pos: 每个词的位置 [0, 1, 2, ..., seq_len-1]
    # shape (seq_len, 1)，方便广播
    pos = np.arange(seq_len).reshape(-1, 1)
    
    # i: 维度索引 [0, 1, 2, ..., d_model-1]
    # 对偶数维 (2i) 用 sin，奇数维 (2i+1) 用 cos
    i = np.arange(d_model)
    
    # 公式里的分母: 10000^(2i/d_model)
    # 对每个维度算一个"周期"
    div_term = np.exp(i * -np.log(10000.0) / d_model)
    # 等价于: 1 / (10000^(2i/d_model))
    
    # 偶数维度 (0, 2, 4, ...) 用 sin
    pe[:, 0::2] = np.sin(pos * div_term[0::2])
    # 奇数维度 (1, 3, 5, ...) 用 cos
    pe[:, 1::2] = np.cos(pos * div_term[1::2])
    
    return pe


# ============================================================
# 1. 生成位置编码
# ============================================================
# 假设: 句子 6 个词，每个词向量 8 维
seq_len = 6
d_model = 8

pe = sinusoidal_positional_encoding(seq_len, d_model)

print("位置编码矩阵 (6个词, 每个8维):")
print(np.round(pe, 3))
print()

# ============================================================
# 2. 观察不同维度的周期
# ============================================================
print("=" * 50)
print("观察: 不同维度的周期差异")
print("=" * 50)
print(f"\n第0维 (i=0, 周期最短):")
print(f"  pos 0~5: {np.round(pe[:, 0], 3)}")
print(f"  变化最快——相邻位置区分度大")

print(f"\n第2维 (i=1):")
print(f"  pos 0~5: {np.round(pe[:, 2], 3)}")

print(f"\n第6维 (i=3):")
print(f"  pos 0~5: {np.round(pe[:, 6], 3)}")

print(f"\n第7维 (i=3, cos):")
print(f"  pos 0~5: {np.round(pe[:, 7], 3)}")
print(f"  变化最慢——长距离位置的区分")
print()

# ============================================================
# 3. 可视化方案（用数字模拟颜色）
# ============================================================
print("=" * 50)
print("每个维度的值随位置变化（数值 = 编码值）")
print("=" * 50)
print(f"\n{'pos':>4}", end="")
for d in range(d_model):
    print(f"  dim{d:>2}", end="")
print()
for pos_i in range(seq_len):
    print(f"{pos_i:>4}", end="")
    for d in range(d_model):
        print(f"  {pe[pos_i, d]:>5.2f}", end="")
    print()
print()

# ============================================================
# 4. 把位置编码加到词向量上
# ============================================================
print("=" * 50)
print("位置编码 + 词向量")
print("=" * 50)

# 模拟 6 个词的输入
X = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0],
    [0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
])

# 最终输入 = 词向量 + 位置编码
X_final = X + pe

print(f"\n词向量 X shape: {X.shape}")
print(f"位置编码 PE shape: {pe.shape}")
print(f"最终输入 X+PE shape: {X_final.shape}")
print()
print("最终输入的每个位置的值 = 词义 + 位置信息")
print("模型既知道这个词本身的意思，也知道它在哪里")
