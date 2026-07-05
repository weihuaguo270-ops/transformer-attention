"""
PyTorch 版 Self-Attention — 与 NumPy 版 attention.py 对应

用 torch 的矩阵乘法替代 np，语义完全一致。
"""
import torch
import torch.nn.functional as F
from utils import softmax


def attention_np_style(Q, K, V, mask=None):
    """
    与 NumPy 版完全一致的 Attention 计算

    公式: Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

    参数:
        Q, K, V: 形状 (seq_len, d_k)
        mask: 可选，形状 (seq_len, seq_len)，掩码矩阵

    返回:
        output: (seq_len, d_k)
        attn_weights: (seq_len, seq_len) 注意力权重
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / (d_k ** 0.5)  # 与 NumPy 版 np.sqrt 等价

    if mask is not None:
        scores = scores + mask

    attn_weights = softmax(scores)
    output = attn_weights @ V

    return output, attn_weights


# ============================================================
# 演示 — 与 NumPy 版 attention.py 输出对齐
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    d_model, d_k = 4, 3
    W_q = torch.randn(d_model, d_k)
    W_k = torch.randn(d_model, d_k)
    W_v = torch.randn(d_model, d_k)
    X = torch.randn(3, d_model)

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    # Part A: 无掩码
    output, weights = attention_np_style(Q, K, V)
    print("=== Part A: 无掩码 Self-Attention ===")
    print(f"输出形状: {tuple(output.shape)}")
    print(f"注意力权重行和: {weights.sum(dim=-1).tolist()}")

    # Part B: 因果掩码
    causal_mask = torch.triu(torch.full((3, 3), -1e9), diagonal=1)
    output_causal, weights_causal = attention_np_style(Q, K, V, causal_mask)
    print(f"\n=== Part B: 因果掩码 Self-Attention ===")
    print(f"词0只看自己: {weights_causal[0].tolist()}")
    print(f"词1只看前2: {weights_causal[1].tolist()}")
    print(f"词2看全部: {weights_causal[2].tolist()}")
