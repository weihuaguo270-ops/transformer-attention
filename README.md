# Transformer Attention 机制 — 手写学习笔记

纯 NumPy 实现，理解 Attention 最核心的计算过程。

## 文件说明

| 文件 | 内容 |
|------|------|
| `attention.py` | 单头 Self-Attention + 因果掩码（Causal Mask） |
| `multi_head_attention.py` | 多头 Self-Attention |
| `kv_cache.py` | KV Cache — 自回归生成推理加速原理 |
| `positional_encoding.py` | 正弦位置编码（Sinusoidal PE） |

## 文件依赖关系

```
attention.py (单头 + 掩码)
     ↓
multi_head_attention.py (多头)
     ↓
kv_cache.py (推理加速)
     ↓
positional_encoding.py (位置编码)
```

每个文件独立可运行，按顺序阅读效果最佳。

## 运行

```bash
pip install numpy
python attention.py
python multi_head_attention.py
python kv_cache.py
python positional_encoding.py
```

## 学习路线

1. Self-Attention 原理 → `attention.py`
2. 因果掩码（GPT 自回归）→ `attention.py Part B`
3. 多头注意力（Multi-Head）→ `multi_head_attention.py`
4. KV Cache 推理加速 → `kv_cache.py`
5. 位置编码 → `positional_encoding.py`
