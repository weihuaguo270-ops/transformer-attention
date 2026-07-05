"""
PyTorch 版多头注意力 — 与 NumPy 版 multi_head_attention.py 对应

用 nn.Parameter 管理权重，更接近真实框架的使用方式。
"""
import torch
import torch.nn as nn
from utils import softmax, split_heads, combine_heads


class MultiHeadAttention(nn.Module):
    """
    PyTorch 版多头自注意力

    与 NumPy 版 MultiHeadAttention 逻辑完全一致，但：
      1. 用 nn.Module 封装 — 方便管理参数和 forward
      2. 权重用 nn.Parameter — 可自动微分
      3. forward 接受 (seq_len, d_model) 输入
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 用 nn.Parameter 替代 NumPy 的 random.randn
        # 效果相同，但 PyTorch 会跟踪梯度
        self.Wq = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.Wk = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.Wv = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.Wo = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

    def forward(self, x, use_mask=True):
        """
        前向传播 — 与 NumPy 版 MultiHeadAttention.forward 完全对应

        参数:
            x: (seq_len, d_model)
            use_mask: 是否使用因果掩码

        返回:
            (seq_len, d_model)
        """
        seq_len = x.shape[0]

        # Step 1: QKV 投影 + 拆头
        Q = split_heads(x @ self.Wq, self.num_heads)
        K = split_heads(x @ self.Wk, self.num_heads)
        V = split_heads(x @ self.Wv, self.num_heads)

        # Step 2: 所有头并行 Attention
        scores = (Q @ K.transpose(1, 2)) / (self.d_k ** 0.5)

        # Step 3: 因果掩码
        if use_mask:
            mask = torch.triu(torch.full((seq_len, seq_len), -1e9), diagonal=1)
            scores = scores + mask

        # Step 4: Softmax
        attn_weights = softmax(scores)

        # Step 5: 加权求和
        head_outputs = attn_weights @ V

        # Step 6: 合并 + 输出投影
        combined = combine_heads(head_outputs, self.num_heads)
        return combined @ self.Wo


# ============================================================
# 演示
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    d_model, num_heads = 4, 2
    X = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
    ])

    mha = MultiHeadAttention(d_model, num_heads)
    output = mha(X, use_mask=False)

    print("多头 Attention 输出:")
    print(output)
    print(f"\n配置: {num_heads} 个头, d_model={d_model}, 每头维度={d_model // num_heads}")
