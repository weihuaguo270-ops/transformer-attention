"""
PyTorch 版工具函数 — 与 NumPy 版 utils.py 对应

使用 torch 的内置函数替代手写实现，对比差异。
"""
import torch
import torch.nn.functional as F


def softmax(x, dim=-1):
    """
    PyTorch 版 Softmax — 直接用 F.softmax

    对比 NumPy 版:
      NumPy: 手写 exp(x-max) / sum(exp(x-max))
      PyTorch: F.softmax(x, dim=-1) — 内部处理了数值稳定性

    参数:
        x: 输入张量
        dim: 做 softmax 的维度（默认最后一维）
    """
    return F.softmax(x, dim=dim)


def split_heads(x, num_heads):
    """
    将 Q/K/V 拆成多头

    PyTorch 版与 NumPy 版逻辑完全一致:
      (seq, d_model) → reshape → (seq, num_heads, d_k) → transpose → (num_heads, seq, d_k)
    """
    seq_len, d_model = x.shape
    d_k = d_model // num_heads
    x = x.reshape(seq_len, num_heads, d_k)
    return x.transpose(0, 1)  # (num_heads, seq_len, d_k)


def combine_heads(x, num_heads):
    """
    合并多头 — split_heads 的逆操作
    """
    _, seq_len, d_k = x.shape
    x = x.transpose(0, 1)  # (seq_len, num_heads, d_k)
    return x.reshape(seq_len, -1)


def layer_norm(x, eps=1e-6):
    """
    PyTorch 版 LayerNorm — 与 NumPy 版数学等价

    对比 NumPy 版:
      NumPy: (x - mean) / (std + eps)
      PyTorch: 同上，但可以用 nn.LayerNorm 替代

    这里保持手写以对比精度，实际项目中用 nn.LayerNorm。
    """
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)  # unbiased=False 与 NumPy 默认一致
    return (x - mean) / (std + eps)
